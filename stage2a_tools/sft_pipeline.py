#!/usr/bin/env python3
import argparse
import gc
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import time
import traceback
from datetime import timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.ipc as ipc
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("V3_RUNS_ROOT", str(ROOT / "runs"))) / "sft"
ART.mkdir(parents=True, exist_ok=True)
BASE = Path(os.environ.get("V3_BASE_MODEL", str(ROOT / "models/Qwen3-VL-8B-Instruct")))
ARROW = ROOT / "data/sole_cot_dataset_hf/data-00000-of-00001.arrow"
FINAL = Path(str(ROOT / "models/SOLE-R1-8B")).resolve()
OUT_ROOT = ROOT / "runs/sft"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_table(path: Path = ARROW):
    try:
        with ipc.open_file(path) as reader:
            return reader.read_all()
    except Exception:
        with ipc.open_stream(path) as reader:
            return reader.read_all()


def row_dict(table, idx: int) -> dict[str, Any]:
    return {name: table[name][idx].as_py() for name in table.column_names}


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            x.get("text", "")
            for x in content
            if isinstance(x, dict) and x.get("type") == "text"
        )
    return ""


def answer_int(text: str):
    m = re.search(r"<answer>\s*([+-]?\d+)%\s*</answer>", text)
    return int(m.group(1)) if m else None


def canonical_record(
    row: dict[str, Any], idx: int, source_shard: str = str(ARROW)
) -> dict[str, Any]:
    image_bytes = row["image"]["bytes"]
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    messages = json.loads(row["prompt"])
    if messages[-1].get("role") != "assistant":
        raise ValueError(f"row {idx}: final role is not assistant")
    completion = text_from_content(messages[-1].get("content"))
    if completion != row["completion"] or completion != row["solution"]:
        raise ValueError(f"row {idx}: completion/solution mismatch")
    ans = answer_int(completion)
    if ans is not None and str(ans) != str(row.get("solution_orig")):
        raise ValueError(f"row {idx}: solution_orig mismatch")
    return {
        "row_index": idx,
        "image": img,
        "image_bytes": image_bytes,
        "prompt_messages": messages[:-1],
        "completion": completion,
        "metadata": {
            "row_index": idx,
            "source_shard": source_shard,
            "data_source": row.get("data_source"),
            "image_name": row.get("image_name"),
            "solution_orig": row.get("solution_orig"),
            "solution": row.get("solution"),
            "from_annotation": row.get("from_annotation"),
            "from_zero": "_from_zero" in (row.get("image_name") or "")
            or img.size == (784, 784),
            "answer": ans,
            "raw_image_sha256": sha_bytes(image_bytes),
            "raw_prompt_sha256": sha_bytes(row["prompt"].encode()),
            "raw_completion_sha256": sha_bytes(row["completion"].encode()),
            "image_size": img.size,
        },
    }


class SoleSFTCollator:
    def __init__(self, processor, padding=False):
        self.processor = processor
        self.padding = padding
        self.image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    def build_one(self, rec: dict[str, Any]) -> dict[str, Any]:
        prompt_text = self.processor.apply_chat_template(
            rec["prompt_messages"], tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + rec["completion"] + "<|im_end|>\n"
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=[rec["image"]],
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,
        )
        full_inputs = self.processor(
            text=[full_text],
            images=[rec["image"]],
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,
        )
        labels = full_inputs["input_ids"].clone()
        prefix_len = prompt_inputs["input_ids"].shape[1]
        labels[:, :prefix_len] = -100
        image_positions = (
            (full_inputs["input_ids"][0] == self.image_token_id)
            .nonzero(as_tuple=False)
            .flatten()
            .tolist()
        )
        for pos in image_positions:
            labels[0, pos] = -100
        supervised = self.processor.tokenizer.decode(
            [
                i
                for i, y in zip(
                    full_inputs["input_ids"][0].tolist(), labels[0].tolist()
                )
                if y != -100
            ]
        )
        assertions = {
            "completion_occurs_once": full_text.count(rec["completion"]) == 1,
            "completion_not_in_prompt": rec["completion"] not in prompt_text,
            "prompt_labels_masked": bool((labels[:, :prefix_len] == -100).all().item()),
            "image_labels_masked": all(
                labels[0, p].item() == -100 for p in image_positions
            ),
            "think_supervised": "<think>" in supervised,
            "answer_supervised": "<answer>" in supervised,
            "answer_not_truncated": "</answer>" in supervised,
            "supervised_matches": supervised == rec["completion"] + "<|im_end|>\n",
        }
        batch = {k: v for k, v in full_inputs.items()}
        batch["labels"] = labels
        stats = {
            "row_index": rec["row_index"],
            "prompt_sha256": sha_bytes(prompt_text.encode()),
            "full_sha256": sha_bytes(full_text.encode()),
            "input_len": int(full_inputs["input_ids"].shape[1]),
            "prefix_len": int(prefix_len),
            "completion_token_len": int((labels != -100).sum().item()),
            "image_token_count": len(image_positions),
            "pixel_values_rows": int(full_inputs["pixel_values"].shape[0]),
            "pixel_values_shape": list(full_inputs["pixel_values"].shape),
            "image_grid_thw": (
                full_inputs.get("image_grid_thw").cpu().tolist()
                if "image_grid_thw" in full_inputs
                else None
            ),
            "assertions": assertions,
            "labels_pass": all(assertions.values()),
            "metadata": rec["metadata"],
        }
        return batch, stats


def to_device(batch: dict[str, Any], device):
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


def init_deepspeed_context(args):
    import deepspeed
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", args.master_port)
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    timeout_seconds = int(os.environ.get("DEEPSPEED_TIMEOUT", "7200"))
    os.environ.setdefault("DEEPSPEED_TIMEOUT", str(timeout_seconds))
    if not dist.is_initialized():
        deepspeed.init_distributed(
            dist_backend="nccl", timeout=timedelta(seconds=timeout_seconds)
        )
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    return deepspeed, dist, rank, local_rank, world_size


def trajectory_group(image_name: str) -> str:
    return (
        image_name.split("_timestep_")[0] if "_timestep_" in image_name else image_name
    )
