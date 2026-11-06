#!/usr/bin/env python3
"""Shared DeepSpeed, generation, reference-model and export functions for v3 GRPO."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "stage2a_tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "stage2a_tools"))

from stage2a_tools.libero_sft_e2e import (  # noqa: E402
    DEFAULT_DATASET,
    ShardedArrowReader,
    choose_refs,
    resolve_arrow_files,
    scan_shards,
)
from stage2a_tools.sft_pipeline import (  # noqa: E402
    canonical_record,
    init_deepspeed_context,
    read_table,
    row_dict,
    to_device,
)
from stage3_tools.rlvr_core import (  # noqa: E402
    PAPER_CONFIRMED_RLVR,
    progress_reward,
    reward_breakdown_to_dict,
    standardize_group_rewards,
)

ART = Path(os.environ.get("V3_RUNS_ROOT", str(ROOT / "runs"))) / "rlvr" / "rlvr_train"
OUT_ROOT = (
    Path(os.environ.get("V3_RUNS_ROOT", str(ROOT / "runs"))) / "rlvr" / "rlvr_outputs"
)
SFT_CKPT = (
    ROOT
    / "stage2a_artifacts"
    / "libero_sft_outputs"
    / "hf_exports"
    / "libero_sft_v3_from_existing_20260814"
)
FINAL_FORBIDDEN = Path(str(ROOT / "models/SOLE-R1-8B")).resolve()
BASE_REPO_ID = "Qwen/Qwen3-VL-8B-Instruct"
BASE_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def deepspeed_config(args, world_size: int) -> dict[str, Any]:
    cfg = json.loads(Path(args.deepspeed_config).read_text(encoding="utf-8"))
    cfg["train_micro_batch_size_per_gpu"] = 1
    cfg["gradient_accumulation_steps"] = 1
    cfg["train_batch_size"] = world_size
    cfg.setdefault("bf16", {})["enabled"] = True
    cfg.setdefault("zero_optimization", {})[
        "stage3_gather_16bit_weights_on_model_save"
    ] = True
    cfg.setdefault("optimizer", {})["type"] = "AdamW"
    params = cfg.setdefault("optimizer", {}).setdefault("params", {})
    params["lr"] = args.lr
    params["betas"] = [0.9, 0.95]
    params["weight_decay"] = 0.1
    params.setdefault("eps", 1e-8)
    return cfg


def build_engine(args, world_size: int):
    import deepspeed

    processor = AutoProcessor.from_pretrained(
        str(args.policy_checkpoint), local_files_only=True, trust_remote_code=True
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.policy_checkpoint),
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    cfg = deepspeed_config(args, world_size)
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config=cfg
    )
    return processor, engine, optimizer, cfg


def build_reference(args, device):
    ref = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.policy_checkpoint),
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    ref.config.use_cache = False
    ref.eval().to(device)
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def decode_completion(tokenizer, generated_ids: torch.Tensor) -> str:
    text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]
    return text.strip()


def prompt_inputs_for_generation(processor, rec: dict[str, Any], device):
    prompt_text = processor.apply_chat_template(
        rec["prompt_messages"], tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[prompt_text],
        images=[rec["image"]],
        return_tensors="pt",
        padding=False,
        add_special_tokens=False,
    )
    return prompt_text, to_device(inputs, device)


def gather_samples(
    local_samples: list[dict[str, Any]], rank: int, world_size: int
) -> list[dict[str, Any]]:
    import torch.distributed as dist

    gathered: list[list[dict[str, Any]]] = [None for _ in range(world_size)]  # type: ignore[list-item]
    dist.all_gather_object(gathered, local_samples)
    flat = [item for part in gathered for item in part]
    flat.sort(key=lambda x: int(x["sample_index"]))
    return flat


def max_memory(local_rank: int, device) -> dict[str, Any]:
    import torch.distributed as dist

    local = torch.tensor(
        [
            float(torch.cuda.max_memory_allocated(local_rank)),
            float(torch.cuda.max_memory_reserved(local_rank)),
        ],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(local, op=dist.ReduceOp.MAX)
    return {
        "peak_allocated_gib_max_rank": float(local[0].item() / 2**30),
        "peak_reserved_gib_max_rank": float(local[1].item() / 2**30),
    }


def export_hf(
    engine, processor, export_dir: Path, rank: int, args, step: int
) -> dict[str, Any]:
    import torch.distributed as dist

    started = time.time()
    export_dir.mkdir(parents=True, exist_ok=True)
    prev_use_cache = getattr(engine.module.config, "use_cache", None)
    state_dict = engine._zero3_consolidated_16bit_state_dict()
    try:
        if rank == 0:
            engine.module.config.use_cache = True
            engine.module.save_pretrained(
                str(export_dir),
                state_dict=state_dict,
                safe_serialization=True,
                max_shard_size="5GB",
            )
            processor.save_pretrained(str(export_dir))
            for name in [
                "generation_config.json",
                "chat_template.jinja",
                "video_preprocessor_config.json",
            ]:
                src = Path(args.policy_checkpoint) / name
                dst = export_dir / name
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
            write_json(
                export_dir / "REPRODUCTION_EXPORT_METADATA.json",
                {
                    "stage": "stage3_rlvr",
                    "run_name": args.run_name,
                    "optimizer_step": step,
                    "base_repo_id": BASE_REPO_ID,
                    "base_revision": BASE_REVISION,
                    "policy_initialized_from": str(args.policy_checkpoint),
                    "reference_initialized_from": str(args.policy_checkpoint),
                    "used_official_final_weights": False,
                    "lora": False,
                    "qlora": False,
                    "quantization": "none",
                    "frozen_modules": [],
                    "full_parameter": True,
                    "exported_at": time.time(),
                },
            )
    finally:
        if prev_use_cache is not None:
            engine.module.config.use_cache = prev_use_cache
        del state_dict
        gc.collect()
    if dist.is_initialized():
        dist.barrier()
    required = [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "preprocessor_config.json",
    ]
    return {
        "optimizer_step": step,
        "export_dir": str(export_dir),
        "elapsed_sec": time.time() - started,
        "required_files_present": {
            name: (export_dir / name).exists() for name in required
        },
        "pass": all((export_dir / name).exists() for name in required),
    }
