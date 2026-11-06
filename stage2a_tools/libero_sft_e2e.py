#!/usr/bin/env python3
"""V3 full-parameter DeepSpeed SFT, dataset scan and HF export."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "stage2a_tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "stage2a_tools"))

from sft_pipeline import (  # noqa: E402
    BASE,
    ROOT as SFT_ROOT,
    SoleSFTCollator,
    canonical_record,
    init_deepspeed_context,
    read_table,
    row_dict,
    to_device,
    trajectory_group,
)

ART = (
    Path(os.environ.get("V3_RUNS_ROOT", str(ROOT / "runs"))) / "sft" / "libero_sft_e2e"
)
OUT_ROOT = (
    Path(os.environ.get("V3_RUNS_ROOT", str(ROOT / "runs")))
    / "sft"
    / "libero_sft_outputs"
)
DEFAULT_DATASET = Path(
    os.environ.get("V3_DATASET", str(ROOT / "data/sole_cot_dataset_hf"))
)
BASE_REPO_ID = "Qwen/Qwen3-VL-8B-Instruct"
BASE_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
FINAL_FORBIDDEN = Path(str(ROOT / "models/SOLE-R1-8B")).resolve()


ANSWER_RE = re.compile(r"<answer>\s*([+-]?\d+)%\s*</answer>")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_arrow_files(dataset_path: str | Path) -> list[Path]:
    path = Path(dataset_path)
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"dataset path does not exist: {path}")
    files = sorted(path.glob("data-*.arrow"))
    if not files:
        files = sorted(path.glob("*.arrow"))
    if not files:
        raise FileNotFoundError(f"no Arrow files found under {path}")
    return files


def parse_answer(text: str) -> int | None:
    match = ANSWER_RE.search(text or "")
    return int(match.group(1)) if match else None


def scan_shards(
    dataset_path: str | Path, *, limit_rows: int | None = None
) -> dict[str, Any]:
    files = resolve_arrow_files(dataset_path)
    refs: list[dict[str, Any]] = []
    bad = Counter()
    answer_values: list[int] = []
    image_sizes = Counter()
    source_counts = Counter()
    from_annotation_counts = Counter()
    started = time.time()
    for shard_id, path in enumerate(files):
        table = read_table(path)
        for row_idx in range(table.num_rows):
            if limit_rows is not None and len(refs) >= limit_rows:
                break
            row = {
                name: table[name][row_idx].as_py()
                for name in table.column_names
                if name != "image"
            }
            prompt = row.get("prompt")
            completion = row.get("completion")
            solution = row.get("solution")
            solution_orig = row.get("solution_orig")
            image_name = row.get("image_name") or f"{path.name}:{row_idx}"
            answer = parse_answer(completion)
            try:
                messages = json.loads(prompt)
            except Exception:
                messages = []
                bad["prompt_json_error"] += 1
            if not messages or messages[-1].get("role") != "assistant":
                bad["last_role_not_assistant"] += 1
            elif messages[-1].get("content") != completion:
                bad["prompt_completion_mismatch"] += 1
            if completion != solution:
                bad["completion_solution_mismatch"] += 1
            if answer is None:
                bad["answer_parse_failure"] += 1
            else:
                answer_values.append(answer)
                if str(answer) != str(solution_orig):
                    bad["answer_solution_orig_mismatch"] += 1
                if answer < -100 or answer > 100:
                    bad["answer_out_of_range"] += 1
            source_counts[row.get("data_source")] += 1
            from_annotation_counts[row.get("from_annotation")] += 1
            refs.append(
                {
                    "global_index": len(refs),
                    "shard_id": shard_id,
                    "shard_path": str(path),
                    "row_index": row_idx,
                    "image_name": image_name,
                    "trajectory_group": trajectory_group(image_name),
                    "answer": answer,
                    "data_source": row.get("data_source"),
                    "from_annotation": row.get("from_annotation"),
                }
            )
        if limit_rows is not None and len(refs) >= limit_rows:
            break
    # Cheap size check on one row per shard. Full image decoding is deferred to preflight.
    for path in files:
        table = read_table(path)
        if table.num_rows:
            img = table["image"][0].as_py()
            try:
                if isinstance(img, dict) and img.get("bytes"):
                    import io

                    size = Image.open(io.BytesIO(img["bytes"])).size
                    image_sizes[str(size)] += 1
            except Exception:
                bad["image_size_probe_failure"] += 1
    summary = {
        "dataset_path": str(dataset_path),
        "arrow_files": [str(p) for p in files],
        "num_shards": len(files),
        "num_rows_scanned": len(refs),
        "num_trajectory_groups": len({r["trajectory_group"] for r in refs}),
        "bad_counts": dict(bad),
        "answer_min": min(answer_values) if answer_values else None,
        "answer_max": max(answer_values) if answer_values else None,
        "unique_answers": len(set(answer_values)),
        "source_counts": dict(source_counts),
        "from_annotation_counts": {
            str(k): int(v) for k, v in from_annotation_counts.items()
        },
        "sampled_image_sizes_by_shard_probe": dict(image_sizes),
        "elapsed_sec": time.time() - started,
    }
    return {"summary": summary, "refs": refs}


class ShardedArrowReader:
    def __init__(self):
        self.current_path: str | None = None
        self.current_table = None

    def row(self, ref: dict[str, Any]) -> dict[str, Any]:
        path = ref["shard_path"]
        if path != self.current_path:
            self.current_table = read_table(Path(path))
            self.current_path = path
        return row_dict(self.current_table, int(ref["row_index"]))


def choose_refs(
    refs: list[dict[str, Any]],
    *,
    train_samples: int | None,
    val_samples: int,
    seed: int,
    group_disjoint_val: bool,
) -> dict[str, Any]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = {}
    for ref in refs:
        groups.setdefault(ref["trajectory_group"], []).append(ref)
    group_names = list(groups)
    rng.shuffle(group_names)

    val_groups: set[str] = set()
    val_refs: list[dict[str, Any]] = []
    if val_samples > 0:
        for group in group_names:
            if len(val_refs) >= val_samples:
                break
            val_groups.add(group)
            shuffled = list(groups[group])
            rng.shuffle(shuffled)
            val_refs.extend(shuffled)
        val_refs = val_refs[:val_samples]

    train_pool = [
        r
        for r in refs
        if not (group_disjoint_val and r["trajectory_group"] in val_groups)
    ]
    rng.shuffle(train_pool)
    if train_samples is None or train_samples <= 0:
        train_refs = train_pool
    else:
        train_refs = train_pool[:train_samples]
    train_groups = {r["trajectory_group"] for r in train_refs}
    split = {
        "seed": seed,
        "split_type": "deterministic_shuffle_with_optional_group_disjoint_validation",
        "train_samples_requested": train_samples,
        "val_samples_requested": val_samples,
        "group_disjoint_val": group_disjoint_val,
        "train_refs": train_refs,
        "val_refs": val_refs,
        "train_num_samples": len(train_refs),
        "val_num_samples": len(val_refs),
        "train_num_groups": len(train_groups),
        "val_num_groups": len({r["trajectory_group"] for r in val_refs}),
        "train_val_groups_disjoint": not bool(
            train_groups & {r["trajectory_group"] for r in val_refs}
        ),
        "train_answer_min": min(
            (r["answer"] for r in train_refs if r["answer"] is not None), default=None
        ),
        "train_answer_max": max(
            (r["answer"] for r in train_refs if r["answer"] is not None), default=None
        ),
        "val_answer_min": min(
            (r["answer"] for r in val_refs if r["answer"] is not None), default=None
        ),
        "val_answer_max": max(
            (r["answer"] for r in val_refs if r["answer"] is not None), default=None
        ),
    }
    return split


def deepspeed_config(args, world_size: int) -> dict[str, Any]:
    cfg = json.loads(Path(args.deepspeed_config).read_text(encoding="utf-8"))
    cfg["train_micro_batch_size_per_gpu"] = args.micro_batch_size
    cfg["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    cfg["train_batch_size"] = (
        world_size * args.micro_batch_size * args.gradient_accumulation_steps
    )
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


class ConstantLRScheduler:
    def __init__(self, optimizer, lr: float):
        self.optimizer = optimizer
        self.lr = lr
        self.last_step = 0
        self.set_step(0)

    def set_step(self, step: int) -> float:
        self.last_step = int(step)
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr
        return self.lr

    def step(self) -> float:
        return self.set_step(self.last_step + 1)

    def state_dict(self) -> dict[str, Any]:
        return {"type": "constant", "lr": self.lr, "last_step": self.last_step}


def build_engine(args, world_size: int):
    import deepspeed

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    processor = AutoProcessor.from_pretrained(
        str(BASE), local_files_only=True, trust_remote_code=True
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(BASE),
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
    return processor, engine, optimizer, ConstantLRScheduler(optimizer, args.lr), cfg


def distributed_eval(
    engine,
    reader: ShardedArrowReader,
    collator: SoleSFTCollator,
    refs: list[dict[str, Any]],
    rank: int,
    world_size: int,
):
    import torch.distributed as dist

    engine.eval()
    local_refs = refs[rank::world_size]
    nll = 0.0
    tokens = 0.0
    count = 0.0
    finite = 1.0
    with torch.no_grad():
        for ref in local_refs:
            row = reader.row(ref)
            rec = canonical_record(row, int(ref["row_index"]), ref["shard_path"])
            batch, stats = collator.build_one(rec)
            batch = to_device(batch, engine.device)
            out = engine(**batch, use_cache=False)
            loss = out.loss.detach().float()
            tok = float(stats["completion_token_len"])
            nll += float(loss.cpu().item()) * tok
            tokens += tok
            count += 1.0
            finite *= 1.0 if bool(torch.isfinite(loss).item()) else 0.0
    buf = torch.tensor(
        [nll, tokens, count, finite], dtype=torch.float64, device=engine.device
    )
    if dist.is_initialized():
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    engine.train()
    return {
        "loss": float(buf[0].item() / max(1.0, buf[1].item())),
        "token_count": int(buf[1].item()),
        "row_count": int(buf[2].item()),
        "all_finite": bool(buf[3].item() >= world_size),
    }


def train_one_epoch_or_subset(
    engine,
    reader: ShardedArrowReader,
    collator: SoleSFTCollator,
    train_refs: list[dict[str, Any]],
    args,
    scheduler: ConstantLRScheduler,
    rank: int,
    world_size: int,
    processor=None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch.distributed as dist

    global_batch = world_size * args.micro_batch_size * args.gradient_accumulation_steps
    requested_samples = len(train_refs)
    if args.max_optimizer_steps and args.max_optimizer_steps > 0:
        optimizer_steps = min(
            args.max_optimizer_steps, math.ceil(requested_samples / global_batch)
        )
    else:
        optimizer_steps = math.ceil(requested_samples / global_batch)
    total_slots = optimizer_steps * global_batch
    padded_samples = max(0, total_slots - requested_samples)
    padded_refs = list(train_refs)
    if padded_samples:
        padded_refs.extend(train_refs[:padded_samples])

    records = []
    periodic_artifacts = []
    train_started = time.time()
    step_nll = 0.0
    step_tokens = 0.0
    step_finite = True
    step_grad_finite = True
    step_wall = 0.0
    total_microsteps = optimizer_steps * args.gradient_accumulation_steps
    for micro in range(total_microsteps):
        optimizer_step_index = micro // args.gradient_accumulation_steps
        sample_index = micro * world_size + rank
        ref = padded_refs[sample_index]
        scheduler.set_step(optimizer_step_index)
        row = reader.row(ref)
        rec = canonical_record(row, int(ref["row_index"]), ref["shard_path"])
        batch, stats = collator.build_one(rec)
        batch = to_device(batch, engine.device)
        t0 = time.time()
        out = engine(**batch, use_cache=False)
        loss = out.loss
        finite_loss = bool(torch.isfinite(loss).item())
        engine.backward(loss)
        boundary = bool(engine.is_gradient_accumulation_boundary())
        grad_finite = True
        if boundary:
            for _, p in engine.module.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all().item():
                    grad_finite = False
                    break
        engine.step()
        torch.cuda.synchronize()
        wall = time.time() - t0
        tok = float(stats["completion_token_len"])
        local = torch.tensor(
            [
                float(loss.detach().float().cpu().item()) * tok,
                tok,
                1.0 if finite_loss else 0.0,
                1.0 if grad_finite else 0.0,
                wall,
            ],
            dtype=torch.float64,
            device=engine.device,
        )
        if dist.is_initialized():
            dist.all_reduce(local, op=dist.ReduceOp.SUM)
        step_nll += float(local[0].item())
        step_tokens += float(local[1].item())
        step_finite = step_finite and (local[2].item() >= world_size)
        step_grad_finite = step_grad_finite and (local[3].item() >= world_size)
        step_wall += float(local[4].item()) / max(1, world_size)
        if boundary:
            opt_step = optimizer_step_index + 1
            elapsed = time.time() - train_started
            avg_step_sec = elapsed / max(1, opt_step)
            remaining_steps = max(0, optimizer_steps - opt_step)
            eta_sec = avg_step_sec * remaining_steps
            samples_seen = min(total_slots, opt_step * global_batch)
            records.append(
                {
                    "optimizer_step": opt_step,
                    "loss": step_nll / max(1.0, step_tokens),
                    "global_samples": global_batch,
                    "global_samples_seen": samples_seen,
                    "global_sample_slots_total": total_slots,
                    "global_supervised_tokens": int(step_tokens),
                    "loss_finite": step_finite,
                    "gradients_finite": step_grad_finite,
                    "lr": args.lr,
                    "mean_rank_wall_time_sec": step_wall,
                    "elapsed_train_sec": elapsed,
                    "eta_sec": eta_sec,
                }
            )
            if rank == 0:
                progress = {
                    "run_name": args.run_name,
                    "optimizer_step": opt_step,
                    "optimizer_steps_total": optimizer_steps,
                    "global_samples_seen": samples_seen,
                    "global_sample_slots_total": total_slots,
                    "requested_train_samples": requested_samples,
                    "padded_repeat_samples_to_complete_accumulation": padded_samples,
                    "loss": records[-1]["loss"],
                    "lr": args.lr,
                    "elapsed_train_sec": elapsed,
                    "avg_step_sec": avg_step_sec,
                    "eta_sec": eta_sec,
                    "eta_min": eta_sec / 60.0,
                    "eta_hours": eta_sec / 3600.0,
                }
                write_json(ART / f"{args.run_name}_progress.json", progress)
                print(
                    "[SFT progress] "
                    f"step {opt_step}/{optimizer_steps} | "
                    f"samples {samples_seen}/{total_slots} "
                    f"(real_rows={requested_samples}, pad={padded_samples}) | "
                    f"loss={records[-1]['loss']:.4f} | "
                    f"avg_step={avg_step_sec:.1f}s | "
                    f"ETA={eta_sec/3600.0:.2f}h",
                    flush=True,
                )
            if (
                args.checkpoint_every_steps
                and args.checkpoint_every_steps > 0
                and opt_step % args.checkpoint_every_steps == 0
                and opt_step < optimizer_steps
            ):
                periodic_artifacts.append(
                    save_periodic_checkpoint_and_export(
                        engine=engine,
                        processor=processor,
                        args=args,
                        opt_step=opt_step,
                        rank=rank,
                        records=records,
                        train_meta={
                            "requested_train_samples": requested_samples,
                            "effective_global_batch": global_batch,
                            "optimizer_steps": optimizer_steps,
                            "total_global_sample_slots": total_slots,
                            "padded_repeat_samples_to_complete_accumulation": padded_samples,
                        },
                    )
                )
            scheduler.step()
            step_nll = 0.0
            step_tokens = 0.0
            step_finite = True
            step_grad_finite = True
            step_wall = 0.0
    meta = {
        "requested_train_samples": requested_samples,
        "effective_global_batch": global_batch,
        "optimizer_steps": optimizer_steps,
        "total_global_sample_slots": total_slots,
        "padded_repeat_samples_to_complete_accumulation": padded_samples,
        "periodic_artifacts": periodic_artifacts,
    }
    return records, meta


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


def ensure_export_metadata(export_dir: Path, processor) -> None:
    processor.save_pretrained(str(export_dir))
    for name in [
        "generation_config.json",
        "chat_template.jinja",
        "video_preprocessor_config.json",
    ]:
        src = BASE / name
        dst = export_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    marker = {
        "base_repo_id": BASE_REPO_ID,
        "base_revision": BASE_REVISION,
        "used_official_final_weights": False,
        "exported_at": time.time(),
        "export_note": "stage2a_libero_sft_export_hf_compatible_with_reward_server",
    }
    write_json(export_dir / "REPRODUCTION_EXPORT_METADATA.json", marker)


def export_hf_checkpoint(
    engine, processor, export_dir: Path, rank: int
) -> dict[str, Any]:
    import torch.distributed as dist

    started = time.time()
    export_dir.mkdir(parents=True, exist_ok=True)
    previous_use_cache = getattr(engine.module.config, "use_cache", None)
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
            ensure_export_metadata(export_dir, processor)
    finally:
        if previous_use_cache is not None:
            engine.module.config.use_cache = previous_use_cache
        del state_dict
        gc.collect()
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        files = sorted(p.name for p in export_dir.iterdir() if p.is_file())
        required = [
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "preprocessor_config.json",
        ]
        return {
            "export_dir": str(export_dir),
            "elapsed_sec": time.time() - started,
            "files": files,
            "required_files_present": {
                name: (export_dir / name).exists() for name in required
            },
            "pass": all((export_dir / name).exists() for name in required),
        }
    return {
        "export_dir": str(export_dir),
        "elapsed_sec": time.time() - started,
        "pass": True,
    }


def step_export_dir(final_export_dir: str | Path, opt_step: int) -> Path:
    final_path = Path(final_export_dir)
    return final_path.parent / f"{final_path.name}_step{opt_step:06d}"


def save_periodic_checkpoint_and_export(
    engine,
    processor,
    args,
    opt_step: int,
    rank: int,
    records: list[dict[str, Any]],
    train_meta: dict[str, Any],
) -> dict[str, Any]:
    started = time.time()
    tag = f"{args.run_name}_step{opt_step:06d}"
    summary: dict[str, Any] = {
        "optimizer_step": opt_step,
        "tag": tag,
        "started_at": started,
        "save_deepspeed_checkpoint": bool(args.save_deepspeed_checkpoint),
        "export_every_checkpoint": bool(args.export_every_checkpoint),
    }
    client_state = {
        "global_optimizer_step": opt_step,
        "train_meta": train_meta,
        "last_train_record": records[-1] if records else None,
        "base_repo_id": BASE_REPO_ID,
        "base_revision": BASE_REVISION,
        "used_official_final_weights": False,
        "checkpoint_label": "libero_sft_periodic",
    }
    if args.save_deepspeed_checkpoint:
        ckpt_root = Path(args.checkpoint_dir)
        save_ok = engine.save_checkpoint(
            str(ckpt_root), tag=tag, client_state=client_state
        )
        summary["deepspeed_checkpoint"] = {
            "dir": str(ckpt_root),
            "tag": tag,
            "saved": bool(save_ok),
        }
    if args.export_every_checkpoint:
        if processor is None:
            raise RuntimeError("processor is required for --export-every-checkpoint")
        export_dir = step_export_dir(args.export_dir, opt_step)
        summary["hf_export"] = export_hf_checkpoint(engine, processor, export_dir, rank)
    summary["finished_at"] = time.time()
    summary["elapsed_sec"] = summary["finished_at"] - started
    if rank == 0:
        write_json(ART / f"{args.run_name}_periodic_step{opt_step:06d}.json", summary)
        hf_path = summary.get("hf_export", {}).get("export_dir")
        print(
            "[SFT checkpoint] "
            f"step {opt_step} saved tag={tag}"
            + (f" hf_export={hf_path}" if hf_path else ""),
            flush=True,
        )
    return summary


def train_export(args) -> None:
    rec: dict[str, Any] = {
        "started_at": time.time(),
        "mode": "preflight" if args.max_optimizer_steps else "one_epoch",
        "dataset_path": args.dataset,
        "deepspeed_config": args.deepspeed_config,
        "base_repo_id": BASE_REPO_ID,
        "base_revision": BASE_REVISION,
        "base_snapshot": str(BASE),
        "used_official_final_weights": False,
        "full_parameter": True,
        "lora": False,
        "qlora": False,
        "quantization": "none",
        "frozen_modules": [],
        "precision": "bf16",
        "optimizer": {"type": "AdamW", "betas": [0.9, 0.95], "weight_decay": 0.1},
        "lr": args.lr,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "export_dir": args.export_dir,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "export_every_checkpoint": args.export_every_checkpoint,
    }
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    try:
        _, dist, rank, local_rank, world_size = init_deepspeed_context(args)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(local_rank)
        rec.update({"rank": rank, "local_rank": local_rank, "world_size": world_size})
        if Path(args.export_dir).resolve() == FINAL_FORBIDDEN:
            raise RuntimeError(
                "refusing to write into official SOLE-R1 final weights directory"
            )
        scan = scan_shards(args.dataset)
        if scan["summary"]["bad_counts"]:
            raise RuntimeError(
                f"dataset contract failures: {scan['summary']['bad_counts']}"
            )
        split = choose_refs(
            scan["refs"],
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            seed=args.seed,
            group_disjoint_val=args.group_disjoint_val,
        )
        if rank == 0:
            ART.mkdir(parents=True, exist_ok=True)
            write_json(ART / f"{args.run_name}_dataset_scan.json", scan["summary"])
            split_for_disk = {
                k: v for k, v in split.items() if k not in {"train_refs", "val_refs"}
            }
            split_for_disk["train_refs_head"] = split["train_refs"][:20]
            split_for_disk["val_refs"] = split["val_refs"]
            write_json(ART / f"{args.run_name}_split_manifest.json", split_for_disk)
        processor, engine, _, scheduler, ds_cfg = build_engine(args, world_size)
        collator = SoleSFTCollator(processor)
        reader = ShardedArrowReader()
        rec["dataset_summary"] = scan["summary"]
        rec["split_summary"] = {
            k: v for k, v in split.items() if k not in {"train_refs", "val_refs"}
        }
        rec["deepspeed_runtime_config"] = ds_cfg
        rec["engine_zero_stage"] = getattr(
            engine, "zero_optimization_stage", lambda: None
        )()
        total_params = sum(
            int(getattr(p, "ds_numel", p.numel())) for p in engine.module.parameters()
        )
        trainable_params = sum(
            int(getattr(p, "ds_numel", p.numel()))
            for p in engine.module.parameters()
            if p.requires_grad
        )
        rec["total_parameters_rank_view"] = int(total_params)
        rec["trainable_parameters_rank_view"] = int(trainable_params)
        rec["all_rank_view_parameters_trainable"] = bool(
            total_params == trainable_params
        )
        if split["val_refs"]:
            rec["pre_validation"] = distributed_eval(
                engine, reader, collator, split["val_refs"], rank, world_size
            )
        train_records, train_meta = train_one_epoch_or_subset(
            engine,
            reader,
            collator,
            split["train_refs"],
            args,
            scheduler,
            rank,
            world_size,
            processor=processor,
        )
        rec["train_step_records"] = train_records
        rec["train_meta"] = train_meta
        if split["val_refs"]:
            rec["post_validation"] = distributed_eval(
                engine, reader, collator, split["val_refs"], rank, world_size
            )
        ckpt_state = {
            "global_optimizer_step": len(train_records),
            "scheduler_state": scheduler.state_dict(),
            "dataset_summary": scan["summary"],
            "split_summary": rec["split_summary"],
            "base_repo_id": BASE_REPO_ID,
            "base_revision": BASE_REVISION,
            "used_official_final_weights": False,
            "checkpoint_label": (
                "libero_sft_preflight"
                if args.max_optimizer_steps
                else "libero_sft_one_epoch"
            ),
        }
        if args.save_deepspeed_checkpoint:
            ckpt_root = Path(args.checkpoint_dir)
            save_ok = engine.save_checkpoint(
                str(ckpt_root), tag=args.run_name, client_state=ckpt_state
            )
            rec["deepspeed_checkpoint"] = {
                "dir": str(ckpt_root),
                "tag": args.run_name,
                "saved": bool(save_ok),
            }
        rec["hf_export"] = export_hf_checkpoint(
            engine, processor, Path(args.export_dir), rank
        )
        rec["gpu_memory"] = max_memory(local_rank, engine.device)
        rec["pass"] = (
            len(train_records) == train_meta["optimizer_steps"]
            and all(r["loss_finite"] and r["gradients_finite"] for r in train_records)
            and rec["hf_export"].get("pass", False)
        )
    except Exception as exc:
        rec["pass"] = False
        rec["error_type"] = type(exc).__name__
        rec["error"] = str(exc)
        rec["traceback"] = traceback.format_exc().splitlines()[-120:]
    finally:
        rec["finished_at"] = time.time()
        rec["elapsed_sec"] = rec["finished_at"] - rec["started_at"]
        out = (
            ART
            / f"{args.run_name}_train_export_rank{int(os.environ.get('RANK', '0'))}.json"
        )
        write_json(out, rec)
        if int(os.environ.get("RANK", "0")) == 0:
            write_json(ART / f"{args.run_name}_train_export.json", rec)
            print(
                json.dumps(
                    {
                        k: rec.get(k)
                        for k in [
                            "pass",
                            "error_type",
                            "error",
                            "train_meta",
                            "hf_export",
                            "gpu_memory",
                            "elapsed_sec",
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                )[:12000]
            )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    if not rec.get("pass", False):
        raise RuntimeError(rec.get("error", "training did not pass"))


def scan_cmd(args) -> None:
    out = scan_shards(args.dataset, limit_rows=args.limit_rows)
    ART.mkdir(parents=True, exist_ok=True)
    write_json(ART / f"{args.run_name}_dataset_scan.json", out["summary"])
    print(json.dumps(out["summary"], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scan")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--limit-rows", type=int, default=None)
    p.add_argument("--run-name", default="libero_sft")
    p.set_defaults(func=scan_cmd)

    p = sub.add_parser("train-export")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--run-name", default="libero_sft_preflight")
    p.add_argument(
        "--deepspeed-config",
        default=str(ROOT / "DEEPSPEED_OR_FSDP_CONFIG/zero3_no_offload.json"),
    )
    p.add_argument("--master-port", default="29641")
    p.add_argument("--seed", type=int, default=20260807)
    p.add_argument(
        "--train-samples",
        type=int,
        default=32,
        help="0 or negative means all available training refs",
    )
    p.add_argument("--val-samples", type=int, default=8)
    p.add_argument(
        "--group-disjoint-val", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--max-optimizer-steps", type=int, default=2)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--checkpoint-dir", default=str(OUT_ROOT / "deepspeed_checkpoints"))
    p.add_argument(
        "--save-deepspeed-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=0,
        help="Save periodic checkpoints every N optimizer steps; 0 disables periodic saves",
    )
    p.add_argument(
        "--export-every-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also export every periodic checkpoint as HF safetensors",
    )
    p.add_argument(
        "--export-dir", default=str(OUT_ROOT / "hf_exports/libero_sft_preflight")
    )
    p.set_defaults(func=train_export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
