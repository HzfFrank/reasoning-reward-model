#!/usr/bin/env python3
"""SOLE-R1 LIBERO RLVR v3 trainer.

This version is intentionally independent from the v2 entrypoint.  It keeps the
validated ZeRO-3/full-parameter/export plumbing, but makes the RLVR-specific
choices more explicit and diagnosable:

* completion logprobs include the same ``<|im_end|>\n`` suffix used by SFT;
* near-zero reward-std groups produce zero advantages instead of noise updates;
* default prompt pool is the full progress dataset supplied by the caller;
* out-of-range progress answers are treated as format failures by default;
* a fixed trajectory-disjoint eval split is evaluated at step 0 and periodically;
* all non-paper assumptions are written to the run metadata.

The paper-confirmed RLVR constants are G=8, temperature=0.7, top_p=0.9,
epsilon=0.2, KL beta=0.01, 5000 steps, and learning rate 1e-6.  Parameters not
specified in the paper are deliberately surfaced as CLI options and metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage2a_tools.libero_sft_e2e import (  # noqa: E402
    DEFAULT_DATASET,
    ShardedArrowReader,
    choose_refs,
    scan_shards,
)
from stage2a_tools.sft_pipeline import (
    canonical_record,
    init_deepspeed_context,
    to_device,
)  # noqa: E402
from stage3_tools import libero_rlvr_train as v1  # noqa: E402
from stage3_tools.rlvr_core import (  # noqa: E402
    ASSUMED_RLVR_DETAILS,
    PAPER_CONFIRMED_RLVR,
    completion_token_logps,
    grpo_token_loss,
    progress_reward,
    reward_breakdown_to_dict,
)
from stage3_tools.tensorboard_event_writer import TensorBoardMetricWriter  # noqa: E402

ART_V3 = (
    Path(os.environ.get("V3_RUNS_ROOT", str(ROOT / "runs")))
    / "rlvr"
    / "rlvr_train_grpo_v3"
)
OUT_ROOT_V3 = (
    Path(os.environ.get("V3_RUNS_ROOT", str(ROOT / "runs")))
    / "rlvr"
    / "rlvr_outputs_grpo_v3"
)
FINAL_FORBIDDEN = v1.FINAL_FORBIDDEN
BASE_REPO_ID = v1.BASE_REPO_ID
BASE_REVISION = v1.BASE_REVISION


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _pad_float_rows(
    rows: list[torch.Tensor], *, pad_value: float = 0.0
) -> torch.Tensor:
    if not rows:
        raise ValueError("rows must be non-empty")
    return pad_sequence(
        [r.reshape(-1) for r in rows], batch_first=True, padding_value=pad_value
    )


def _pad_bool_rows(rows: list[torch.Tensor]) -> torch.Tensor:
    if not rows:
        raise ValueError("rows must be non-empty")
    return pad_sequence(
        [r.reshape(-1).bool() for r in rows], batch_first=True, padding_value=False
    )


def _per_token_from_model(
    model, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(**batch, use_cache=False)
    return completion_token_logps(outputs.logits, batch["input_ids"], batch["labels"])


def v3_assumptions(args) -> dict[str, Any]:
    return {
        **ASSUMED_RLVR_DETAILS,
        "tau": {
            "value": args.tau,
            "reason": (
                "The paper gives alpha * exp(-abs(error) / tau) but not tau. "
                "The default 10 makes a 1-point integer progress error mildly worse "
                "while making a 10-point error a large reward drop, which is useful "
                "for progress calibration on a [-100, 100] integer scale."
            ),
        },
        "rlvr_optimizer_adamw": {
            "value": {"betas": [0.9, 0.95], "weight_decay": 0.1, "eps": 1e-8},
            "reason": (
                "The RLVR appendix states lr=1e-6 but does not restate AdamW betas, "
                "weight decay, or eps. We reuse the SFT AdamW settings to avoid adding "
                "an optimizer confound between SFT and RLVR."
            ),
        },
        "max_new_tokens": {
            "value": args.max_new_tokens,
            "reason": (
                "The paper does not publish a generation length cap. The default 192 "
                "covers the observed SFT-style think/answer completions while keeping "
                "4xA100 RLVR rollouts practical."
            ),
        },
        "completion_eos_suffix": {
            "value": args.completion_eos_suffix,
            "reason": "SFT supervised prompt + completion + <|im_end|>\\n, so v3 includes the same suffix in RLVR logprob training.",
        },
        "advantage_std_threshold": {
            "value": args.advantage_std_threshold,
            "reason": (
                "The paper does not define the std=0 case. When sampled rewards are "
                "equal or numerically indistinguishable, the group contains no reliable "
                "ranking signal, so v3 sets all advantages to zero."
            ),
        },
        "advantage_std_definition": {
            "value": "population" if not args.advantage_std_unbiased else "unbiased",
            "reason": "The paper writes std over the G candidates and does not specify Bessel correction; v3 defaults to population std.",
        },
        "advantage_eps": {
            "value": args.advantage_eps,
            "reason": "After the explicit near-zero std guard, the default uses no denominator eps to stay closer to the paper formula.",
        },
        "importance_sampling_level": {
            "value": args.importance_sampling_level,
            "reason": (
                "Token-level is the default because it follows the stable TRL/RLinf-style "
                "masked implementation. v3 also exposes sequence_sum for a more literal "
                "completion-level rho_i ablation."
            ),
        },
        "kl_estimator": {
            "value": "Schulman k3 per completion token",
            "reason": "The paper gives beta * D_KL(p_phi || p_ref) but not the finite-sample estimator; k3 is nonnegative and common in TRL-style GRPO.",
        },
        "out_of_range_answer_policy": {
            "value": args.out_of_range_policy,
            "reason": "The paper defines progress in [-100, 100] but does not specify parseable out-of-range rewards; v3 defaults to fail_format for strictness.",
        },
        "prompt_pool_size": {
            "value": None if args.prompt_pool_size <= 0 else args.prompt_pool_size,
            "reason": "The paper says RLVR runs over the progress dataset. v3 defaults to the full supplied progress pool to avoid v2-style small-pool repetition.",
        },
        "fixed_eval": {
            "value": {
                "eval_samples": args.eval_samples,
                "eval_group_size": args.eval_group_size,
                "eval_every_steps": args.eval_every_steps,
            },
            "reason": (
                "The paper does not define this diagnostic. It is added to separate real "
                "checkpoint improvement from noisy train-time rewards on changing prompts; "
                "the default is small to avoid dominating RLVR runtime."
            ),
        },
        "distributed_topology": {
            "value": "DeepSpeed ZeRO-3 no-offload on the available 4xA100 unless overridden",
            "reason": "The paper reports 8xH100; this project must fit the available four-card machine without LoRA, quantization, or frozen modules.",
        },
    }


def filter_progress_refs(
    refs: list[dict[str, Any]], args
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rejected = Counter()
    valid: list[dict[str, Any]] = []
    for ref in refs:
        answer = ref.get("answer")
        if answer is None:
            rejected["missing_or_unparseable_answer"] += 1
            continue
        if args.strict_target_range and not (-100 <= int(answer) <= 100):
            rejected["target_out_of_range"] += 1
            continue
        valid.append(ref)
    return valid, {
        "input_refs": len(refs),
        "valid_progress_refs": len(valid),
        "rejected_counts": dict(rejected),
        "strict_target_range": bool(args.strict_target_range),
    }


def prepare_refs_v3(args) -> dict[str, Any]:
    scan = scan_shards(args.dataset, limit_rows=args.scan_limit_rows)
    valid_refs, filter_summary = filter_progress_refs(scan["refs"], args)
    if not valid_refs:
        raise RuntimeError("no valid progress refs after v3 filtering")
    train_samples = None if args.prompt_pool_size <= 0 else args.prompt_pool_size
    split = choose_refs(
        valid_refs,
        train_samples=train_samples,
        val_samples=args.eval_samples,
        seed=args.seed,
        group_disjoint_val=True,
    )
    if not split["train_refs"]:
        raise RuntimeError("no RLVR train refs selected")
    split_summary = {
        k: v for k, v in split.items() if k not in {"train_refs", "val_refs"}
    }
    split_summary["train_refs_preview"] = split["train_refs"][:20]
    split_summary["eval_refs"] = split["val_refs"]
    return {
        "scan_summary": scan["summary"],
        "filter_summary": filter_summary,
        "split_summary": split_summary,
        "train_refs": split["train_refs"],
        "eval_refs": split["val_refs"],
    }


def build_completion_batch_v3(
    processor, rec: dict[str, Any], completion: str, args
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt_text = processor.apply_chat_template(
        rec["prompt_messages"], tokenize=False, add_generation_prompt=True
    )
    full_text = prompt_text + completion + args.completion_eos_suffix
    prompt_inputs = processor(
        text=[prompt_text],
        images=[rec["image"]],
        return_tensors="pt",
        padding=False,
        add_special_tokens=False,
    )
    full_inputs = processor(
        text=[full_text],
        images=[rec["image"]],
        return_tensors="pt",
        padding=False,
        add_special_tokens=False,
    )
    labels = full_inputs["input_ids"].clone()
    prefix_len = int(prompt_inputs["input_ids"].shape[1])
    labels[:, :prefix_len] = -100
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_token_id is not None:
        labels[full_inputs["input_ids"] == image_token_id] = -100
    batch = {k: v for k, v in full_inputs.items()}
    batch["labels"] = labels
    stats = {
        "input_len": int(full_inputs["input_ids"].shape[1]),
        "prefix_len": prefix_len,
        "supervised_tokens": int((labels != -100).sum().item()),
        "completion_eos_suffix": args.completion_eos_suffix,
    }
    return batch, stats


def generate_local_samples_v3(
    args,
    processor,
    engine,
    rec: dict[str, Any],
    step: int,
    rank: int,
    world_size: int,
    *,
    group_size: int,
    seed_offset: int = 0,
) -> list[dict[str, Any]]:
    prompt_text, inputs = v1.prompt_inputs_for_generation(processor, rec, engine.device)
    prompt_len = int(inputs["input_ids"].shape[1])
    local: list[dict[str, Any]] = []
    engine.module.eval()
    with torch.no_grad():
        for sample_index in range(rank, group_size, world_size):
            seed = args.seed + seed_offset + step * 1000 + sample_index
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            out = engine.module.generate(
                **inputs,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=processor.tokenizer.eos_token_id,
                synced_gpus=True,
            )
            completion = v1.decode_completion(
                processor.tokenizer, out[0, prompt_len:].detach().cpu()
            )
            reward = progress_reward(
                completion,
                int(rec["metadata"]["answer"]),
                tau=args.tau,
                out_of_range_policy=args.out_of_range_policy,
            )
            local.append(
                {
                    "sample_index": sample_index,
                    "completion": completion,
                    "completion_char_len": len(completion),
                    "reward": reward_breakdown_to_dict(reward),
                }
            )
    engine.module.train()
    return local


def attach_advantages_v3(
    samples: list[dict[str, Any]], rank: int, args
) -> list[dict[str, Any]]:
    import torch.distributed as dist

    payload = samples if rank == 0 else None
    obj = [payload]
    if dist.is_initialized():
        dist.broadcast_object_list(obj, src=0)
    out = obj[0]
    if rank == 0:
        rewards = torch.tensor(
            [float(s["reward"]["total"]) for s in out], dtype=torch.float32
        )
        mean = rewards.mean()
        std = rewards.std(unbiased=bool(args.advantage_std_unbiased))
        std_val = float(std.item())
        zeroed = bool(
            args.scale_rewards == "group"
            and std_val <= float(args.advantage_std_threshold)
        )
        if args.scale_rewards == "group":
            if zeroed:
                advantages = torch.zeros_like(rewards)
            else:
                advantages = (rewards - mean) / (std + float(args.advantage_eps))
        elif args.scale_rewards == "none":
            advantages = rewards - mean
        else:
            raise ValueError(f"unsupported scale_rewards: {args.scale_rewards}")
        diag = {
            "group_reward_mean": float(mean.item()),
            "group_reward_std": std_val,
            "advantage_zeroed_by_std_guard": zeroed,
            "advantage_std_threshold": float(args.advantage_std_threshold),
        }
        for s, adv in zip(out, advantages.tolist()):
            s["advantage"] = float(adv)
            s["advantage_diagnostics"] = diag
        obj = [out]
        if dist.is_initialized():
            dist.broadcast_object_list(obj, src=0)
    else:
        if dist.is_initialized():
            dist.broadcast_object_list(obj, src=0)
    return obj[0]


def summarize_rewards(samples: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(s["reward"]["total"]) for s in samples]
    reward_mean = sum(rewards) / max(1, len(rewards))
    reward_std = math.sqrt(
        sum((r - reward_mean) ** 2 for r in rewards) / max(1, len(rewards))
    )
    parse_ok = sum(1 for s in samples if s["reward"]["parse_ok"])
    abs_errors = [
        float(s["reward"]["abs_error"])
        for s in samples
        if s["reward"]["abs_error"] is not None
    ]
    return {
        "sample_count": len(samples),
        "parse_ok_count": parse_ok,
        "format_pass_rate": parse_ok / max(1, len(samples)),
        "reward_mean": reward_mean,
        "reward_std_population": reward_std,
        "mean_abs_error": (
            sum(abs_errors) / max(1, len(abs_errors)) if abs_errors else None
        ),
        "exact_match_rate": sum(1 for x in abs_errors if x == 0.0)
        / max(1, len(samples)),
    }


def run_fixed_eval(
    args,
    processor,
    engine,
    reader,
    eval_refs: list[dict[str, Any]],
    step: int,
    rank: int,
    world_size: int,
) -> dict[str, Any] | None:
    if args.eval_samples <= 0 or not eval_refs:
        return None
    import torch.distributed as dist

    started = time.time()
    all_samples: list[dict[str, Any]] = []
    for eval_index, ref in enumerate(eval_refs):
        rec = canonical_record(
            reader.row(ref), int(ref["row_index"]), ref["shard_path"]
        )
        local = generate_local_samples_v3(
            args,
            processor,
            engine,
            rec,
            eval_index,
            rank,
            world_size,
            group_size=args.eval_group_size,
            seed_offset=9_000_000,
        )
        samples = v1.gather_samples(local, rank, world_size)
        if rank == 0:
            for sample in samples:
                sample["eval_ref"] = {
                    "row_index": int(ref["row_index"]),
                    "shard_path": ref["shard_path"],
                    "trajectory_group": ref["trajectory_group"],
                    "target_progress": int(rec["metadata"]["answer"]),
                }
            all_samples.extend(samples)
    if dist.is_initialized():
        dist.barrier()
    if rank != 0:
        return None
    summary = summarize_rewards(all_samples)
    summary.update(
        {
            "eval_step": step,
            "eval_samples": len(eval_refs),
            "eval_group_size": args.eval_group_size,
            "elapsed_sec": time.time() - started,
        }
    )
    eval_path = ART_V3 / args.run_name / "fixed_eval_rollouts.jsonl"
    with eval_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"step": step, "summary": summary, "samples": all_samples},
                ensure_ascii=False,
            )
            + "\n"
        )
    return summary


def one_rlvr_step_v3(
    args,
    processor,
    engine,
    ref_model,
    rec: dict[str, Any],
    step: int,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    import torch.distributed as dist

    step_started = time.time()
    local_rollouts = generate_local_samples_v3(
        args, processor, engine, rec, step, rank, world_size, group_size=args.group_size
    )
    samples = v1.gather_samples(local_rollouts, rank, world_size)
    samples = attach_advantages_v3(samples, rank, args)
    local_samples = [s for s in samples if int(s["sample_index"]) % world_size == rank]
    if not local_samples:
        raise RuntimeError(f"rank {rank} received no rollout samples")

    old_rows: list[torch.Tensor] = []
    ref_rows: list[torch.Tensor] = []
    old_masks: list[torch.Tensor] = []
    token_counts: list[float] = []
    old_mean_vals: list[float] = []
    ref_mean_vals: list[float] = []

    engine.eval()
    with torch.no_grad():
        for sample in local_samples:
            batch, _ = build_completion_batch_v3(
                processor, rec, sample["completion"], args
            )
            batch = to_device(batch, engine.device)
            old_logps, mask = _per_token_from_model(engine, batch)
            ref_logps, ref_mask = _per_token_from_model(ref_model, batch)
            if not torch.equal(mask, ref_mask):
                raise RuntimeError("policy/reference completion masks differ")
            old_rows.append(old_logps.squeeze(0).detach())
            ref_rows.append(ref_logps.squeeze(0).detach())
            old_masks.append(mask.squeeze(0).detach())
            token_count = float(mask.sum().detach().cpu().item())
            token_counts.append(token_count)
            old_mean_vals.append(
                float(
                    (old_logps * mask).sum().detach().cpu().item()
                    / max(1.0, token_count)
                )
            )
            ref_mean_vals.append(
                float(
                    (ref_logps * mask).sum().detach().cpu().item()
                    / max(1.0, token_count)
                )
            )
    engine.train()

    new_rows: list[torch.Tensor] = []
    new_masks: list[torch.Tensor] = []
    new_mean_vals: list[float] = []
    for sample in local_samples:
        batch, _ = build_completion_batch_v3(processor, rec, sample["completion"], args)
        batch = to_device(batch, engine.device)
        new_logps, mask = _per_token_from_model(engine, batch)
        new_rows.append(new_logps.squeeze(0))
        new_masks.append(mask.squeeze(0))
        token_count = float(mask.sum().detach().cpu().item())
        new_mean_vals.append(
            float(
                (new_logps * mask).sum().detach().cpu().item() / max(1.0, token_count)
            )
        )

    for old_mask, new_mask in zip(old_masks, new_masks):
        if old_mask.shape != new_mask.shape or not torch.equal(
            old_mask.to(new_mask.device), new_mask
        ):
            raise RuntimeError("old/new completion masks differ")

    per_token_logps = _pad_float_rows(new_rows).to(engine.device)
    old_per_token_logps = _pad_float_rows(old_rows).to(engine.device)
    ref_per_token_logps = _pad_float_rows(ref_rows).to(engine.device)
    completion_mask = _pad_bool_rows(new_masks).to(engine.device)
    advantages = torch.tensor(
        [float(s["advantage"]) for s in local_samples],
        dtype=torch.float32,
        device=engine.device,
    )

    loss, loss_stats = grpo_token_loss(
        per_token_logps,
        old_per_token_logps,
        ref_per_token_logps,
        completion_mask,
        advantages,
        epsilon=args.clip_epsilon,
        beta=args.kl_beta,
        loss_type=args.loss_type,
        importance_sampling_level=args.importance_sampling_level,
        max_completion_length=args.max_new_tokens,
    )
    finite_loss = bool(torch.isfinite(loss).item())
    engine.backward(loss)
    grad_finite = True
    for _, p in engine.module.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all().item():
            grad_finite = False
            break
    engine.step()

    reward_summary = summarize_rewards(samples)
    advantage_diags = [s.get("advantage_diagnostics", {}) for s in samples]
    advantage_zeroed = bool(
        advantage_diags
        and all(bool(d.get("advantage_zeroed_by_std_guard")) for d in advantage_diags)
    )
    local_stats = torch.tensor(
        [
            1.0 if finite_loss else 0.0,
            1.0 if grad_finite else 0.0,
            float(loss.detach().cpu().item()),
            float(loss_stats["policy_loss"].detach().cpu().item()),
            float(loss_stats["kl"].detach().cpu().item()),
            float(loss_stats["ratio"].detach().cpu().item()),
            float(loss_stats["ratio_abs"].detach().cpu().item()),
            float(loss_stats["clipped_ratio"].detach().cpu().item()),
            float(loss_stats["clip_fraction"].detach().cpu().item()),
            float(len(local_samples)),
            float(loss_stats["token_count"].detach().cpu().item()),
        ],
        dtype=torch.float64,
        device=engine.device,
    )
    if dist.is_initialized():
        dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)

    record = {
        "optimizer_step": step,
        "grpo_implementation": "v3_token_level_default_with_sequence_sum_option",
        "loss_type": args.loss_type,
        "importance_sampling_level": args.importance_sampling_level,
        "scale_rewards": args.scale_rewards,
        "advantage_eps": args.advantage_eps,
        "advantage_std_unbiased": args.advantage_std_unbiased,
        "advantage_std_threshold": args.advantage_std_threshold,
        "advantage_zeroed_by_std_guard": advantage_zeroed,
        "out_of_range_policy": args.out_of_range_policy,
        "completion_eos_suffix": args.completion_eos_suffix,
        "prompt_row_index": int(rec["row_index"]),
        "target_progress": int(rec["metadata"]["answer"]),
        **reward_summary,
        "effective_learning_signal": bool(
            (not advantage_zeroed)
            and reward_summary["reward_std_population"] > args.advantage_std_threshold
        ),
        "loss_finite_all_ranks": bool(local_stats[0].item() >= world_size),
        "gradients_finite_all_ranks": bool(local_stats[1].item() >= world_size),
        "mean_rank_total_loss": float(local_stats[2].item() / max(1, world_size)),
        "mean_rank_policy_loss": float(local_stats[3].item() / max(1, world_size)),
        "mean_rank_kl": float(local_stats[4].item() / max(1, world_size)),
        "mean_rank_ratio": float(local_stats[5].item() / max(1, world_size)),
        "mean_rank_ratio_abs": float(local_stats[6].item() / max(1, world_size)),
        "mean_rank_clipped_ratio": float(local_stats[7].item() / max(1, world_size)),
        "mean_rank_clip_fraction": float(local_stats[8].item() / max(1, world_size)),
        "global_rollout_samples": int(local_stats[9].item()),
        "global_supervised_tokens": int(local_stats[10].item()),
        "elapsed_sec": time.time() - step_started,
        "rank_new_mean_logps": new_mean_vals,
        "rank_old_mean_logps": old_mean_vals,
        "rank_ref_mean_logps": ref_mean_vals,
        "rank_advantages": [float(s["advantage"]) for s in local_samples],
        "rank_completion_tokens": token_counts,
    }
    if rank == 0:
        rollout_record = {
            "step": step,
            "prompt_metadata": rec["metadata"],
            "samples": samples,
        }
        with (ART_V3 / args.run_name / "rollouts.jsonl").open(
            "a", encoding="utf-8"
        ) as f:
            f.write(json.dumps(rollout_record, ensure_ascii=False) + "\n")
    return record


def export_hf_v3(
    engine, processor, export_dir: Path, rank: int, args, step: int
) -> dict[str, Any]:
    result = v1.export_hf(engine, processor, export_dir, rank, args, step)
    if rank == 0:
        write_json(
            export_dir / "RLVR_V3_EXPORT_METADATA.json",
            {
                "stage": "stage3_rlvr_grpo_v3",
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
                "paper_confirmed": PAPER_CONFIRMED_RLVR,
                "assumptions": v3_assumptions(args),
                "exported_at": time.time(),
            },
        )
    return result


def export_dir_for(args, step: int, final: bool = False) -> Path:
    base = Path(args.export_root) / args.run_name
    return base / ("final" if final else f"step{step:06d}")


def train(args) -> None:
    run_dir = ART_V3 / args.run_name
    rec: dict[str, Any] = {
        "started_at": time.time(),
        "run_name": args.run_name,
        "mode": args.mode,
        "policy_checkpoint": str(args.policy_checkpoint),
        "reference_checkpoint": str(args.policy_checkpoint),
        "used_official_final_weights": Path(args.policy_checkpoint).resolve()
        == FINAL_FORBIDDEN,
        "paper_confirmed": PAPER_CONFIRMED_RLVR,
        "assumptions": v3_assumptions(args),
        "config": vars(args),
        "full_parameter": True,
        "lora": False,
        "qlora": False,
        "quantization": "none",
        "frozen_modules": [],
    }
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    writer = None
    try:
        if rec["used_official_final_weights"]:
            raise RuntimeError(
                "refusing to initialize RLVR from official SOLE-R1 final weights"
            )
        _, _, rank, local_rank, world_size = init_deepspeed_context(args)
        if args.group_size < world_size:
            raise RuntimeError(
                "group-size must be >= world size so every rank participates in generation"
            )
        if args.eval_samples > 0 and args.eval_group_size < world_size:
            raise RuntimeError(
                "eval-group-size must be >= world size so every rank participates in eval generation"
            )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(local_rank)
        rec.update({"rank": rank, "local_rank": local_rank, "world_size": world_size})
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        refs_pack = prepare_refs_v3(args)
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json(run_dir / "dataset_scan.json", refs_pack["scan_summary"])
            write_json(run_dir / "progress_filter.json", refs_pack["filter_summary"])
            write_json(run_dir / "split_manifest.json", refs_pack["split_summary"])
            write_json(
                run_dir / "paper_and_assumptions.json",
                {
                    "paper_confirmed": PAPER_CONFIRMED_RLVR,
                    "assumptions": v3_assumptions(args),
                },
            )

        processor, engine, _, ds_cfg = v1.build_engine(args, world_size)
        ref_model = v1.build_reference(args, engine.device)
        reader = ShardedArrowReader()
        writer = TensorBoardMetricWriter(run_dir, enabled=args.tensorboard)
        rec["tensorboard_requested"] = bool(args.tensorboard)
        rec["tensorboard_error"] = writer.tb_error
        rec["tensorboard_dir"] = (
            str(run_dir / "tensorboard") if writer.tb is not None else None
        )
        rec["metrics_jsonl"] = str(run_dir / "metrics.jsonl")
        rec["dataset_summary"] = refs_pack["scan_summary"]
        rec["filter_summary"] = refs_pack["filter_summary"]
        rec["split_summary"] = refs_pack["split_summary"]
        rec["deepspeed_runtime_config"] = ds_cfg
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
            total_params == trainable_params and total_params > 0
        )

        train_refs = refs_pack["train_refs"]
        eval_refs = refs_pack["eval_refs"]
        records: list[dict[str, Any]] = []
        eval_records: list[dict[str, Any]] = []
        exports: list[dict[str, Any]] = []
        train_started = time.time()

        eval0 = run_fixed_eval(
            args, processor, engine, reader, eval_refs, 0, rank, world_size
        )
        if rank == 0 and eval0:
            eval_records.append(eval0)
            writer.write(
                0,
                {
                    "eval/reward_mean": eval0["reward_mean"],
                    "eval/reward_std": eval0["reward_std_population"],
                    "eval/format_pass_rate": eval0["format_pass_rate"],
                    "eval/mean_abs_error": (
                        eval0["mean_abs_error"]
                        if eval0["mean_abs_error"] is not None
                        else float("nan")
                    ),
                    "eval/exact_match_rate": eval0["exact_match_rate"],
                    "eval/elapsed_sec": eval0["elapsed_sec"],
                },
            )

        for step in range(1, args.steps + 1):
            ref = train_refs[(step - 1) % len(train_refs)]
            base_rec = canonical_record(
                reader.row(ref), int(ref["row_index"]), ref["shard_path"]
            )
            step_record = one_rlvr_step_v3(
                args, processor, engine, ref_model, base_rec, step, rank, world_size
            )
            records.append(step_record)
            if rank == 0:
                advs = [float(x) for x in step_record.get("rank_advantages", [])]
                metrics = {
                    "train/reward_mean": step_record["reward_mean"],
                    "train/reward_std": step_record["reward_std_population"],
                    "train/format_pass_rate": step_record["format_pass_rate"],
                    "train/mean_abs_error": (
                        step_record["mean_abs_error"]
                        if step_record["mean_abs_error"] is not None
                        else float("nan")
                    ),
                    "train/exact_match_rate": step_record["exact_match_rate"],
                    "train/total_loss": step_record["mean_rank_total_loss"],
                    "train/policy_loss": step_record["mean_rank_policy_loss"],
                    "train/kl": step_record["mean_rank_kl"],
                    "train/ratio": step_record["mean_rank_ratio"],
                    "train/ratio_abs": step_record["mean_rank_ratio_abs"],
                    "train/clipped_ratio": step_record["mean_rank_clipped_ratio"],
                    "train/clip_fraction": step_record["mean_rank_clip_fraction"],
                    "train/global_rollout_samples": step_record[
                        "global_rollout_samples"
                    ],
                    "train/global_supervised_tokens": step_record[
                        "global_supervised_tokens"
                    ],
                    "train/step_time_sec": step_record["elapsed_sec"],
                    "train/lr": args.lr,
                    "train/advantage_zeroed_by_std_guard": (
                        1.0 if step_record["advantage_zeroed_by_std_guard"] else 0.0
                    ),
                }
                if advs:
                    metrics["train/local_advantage_mean"] = sum(advs) / len(advs)
                    metrics["train/local_advantage_abs_mean"] = sum(
                        abs(x) for x in advs
                    ) / len(advs)
                writer.write(step, metrics)
                elapsed = time.time() - train_started
                avg = elapsed / step
                progress = {
                    "run_name": args.run_name,
                    "step": step,
                    "steps_total": args.steps,
                    "reward_mean": step_record["reward_mean"],
                    "format_pass_rate": step_record["format_pass_rate"],
                    "mean_abs_error": step_record["mean_abs_error"],
                    "loss": step_record["mean_rank_total_loss"],
                    "kl": step_record["mean_rank_kl"],
                    "advantage_zeroed_by_std_guard": step_record[
                        "advantage_zeroed_by_std_guard"
                    ],
                    "elapsed_sec": elapsed,
                    "avg_step_sec": avg,
                    "eta_sec": max(0, args.steps - step) * avg,
                }
                write_json(run_dir / "progress.json", progress)
                print(
                    "[RLVR v3 progress] "
                    f"step {step}/{args.steps} | reward={step_record['reward_mean']:.4f} | "
                    f"format={step_record['format_pass_rate']:.2f} | err={step_record['mean_abs_error']} | "
                    f"loss={step_record['mean_rank_total_loss']:.6g} | kl={step_record['mean_rank_kl']:.6g} | "
                    f"zero_adv={int(step_record['advantage_zeroed_by_std_guard'])} | avg_step={avg:.1f}s | "
                    f"ETA={progress['eta_sec']/3600.0:.2f}h",
                    flush=True,
                )
            if (
                args.eval_every_steps
                and args.eval_samples > 0
                and step % args.eval_every_steps == 0
            ):
                ev = run_fixed_eval(
                    args, processor, engine, reader, eval_refs, step, rank, world_size
                )
                if rank == 0 and ev:
                    eval_records.append(ev)
                    writer.write(
                        step,
                        {
                            "eval/reward_mean": ev["reward_mean"],
                            "eval/reward_std": ev["reward_std_population"],
                            "eval/format_pass_rate": ev["format_pass_rate"],
                            "eval/mean_abs_error": (
                                ev["mean_abs_error"]
                                if ev["mean_abs_error"] is not None
                                else float("nan")
                            ),
                            "eval/exact_match_rate": ev["exact_match_rate"],
                            "eval/elapsed_sec": ev["elapsed_sec"],
                        },
                    )
            if args.export_every_steps and step % args.export_every_steps == 0:
                exports.append(
                    export_hf_v3(
                        engine, processor, export_dir_for(args, step), rank, args, step
                    )
                )
        if args.export_final:
            exports.append(
                export_hf_v3(
                    engine,
                    processor,
                    export_dir_for(args, args.steps, final=True),
                    rank,
                    args,
                    args.steps,
                )
            )

        rec["step_records"] = records
        rec["fixed_eval_records"] = eval_records
        rec["exports"] = exports
        rec["gpu_memory"] = v1.max_memory(local_rank, engine.device)
        rec["effective_learning_signal_steps"] = sum(
            1 for r in records if r.get("effective_learning_signal")
        )
        rec["advantage_zeroed_steps"] = sum(
            1 for r in records if r.get("advantage_zeroed_by_std_guard")
        )
        rec["pass"] = (
            len(records) == args.steps
            and all(
                r["loss_finite_all_ranks"] and r["gradients_finite_all_ranks"]
                for r in records
            )
            and rec["all_rank_view_parameters_trainable"]
            and all(e.get("pass", False) for e in exports)
            and not rec["used_official_final_weights"]
        )
    except Exception as exc:
        rec["pass"] = False
        rec["error_type"] = type(exc).__name__
        rec["error"] = str(exc)
        rec["traceback"] = traceback.format_exc().splitlines()[-200:]
    finally:
        if writer is not None:
            writer.close()
        rec["finished_at"] = time.time()
        rec["elapsed_sec"] = rec["finished_at"] - rec["started_at"]
        rank_now = int(os.environ.get("RANK", str(rank)))
        run_dir.mkdir(parents=True, exist_ok=True)
        out = run_dir / f"{args.run_name}_rank{rank_now}.json"
        write_json(out, rec)
        if rank_now == 0:
            main = run_dir / f"{args.run_name}.json"
            write_json(main, rec)
            print(
                json.dumps(
                    {
                        "pass": rec.get("pass"),
                        "error_type": rec.get("error_type"),
                        "error": rec.get("error"),
                        "steps": len(rec.get("step_records", [])),
                        "effective_learning_signal_steps": rec.get(
                            "effective_learning_signal_steps"
                        ),
                        "advantage_zeroed_steps": rec.get("advantage_zeroed_steps"),
                        "last_step": (
                            rec.get("step_records", [{}])[-1]
                            if rec.get("step_records")
                            else None
                        ),
                        "last_eval": (
                            rec.get("fixed_eval_records", [None])[-1]
                            if rec.get("fixed_eval_records")
                            else None
                        ),
                        "exports": rec.get("exports"),
                        "tensorboard_dir": rec.get("tensorboard_dir"),
                        "metrics_jsonl": rec.get("metrics_jsonl"),
                        "report": str(main),
                        "elapsed_sec": rec.get("elapsed_sec"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )[:24000],
                flush=True,
            )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    if not rec.get("pass", False):
        raise RuntimeError(rec.get("error", "training did not pass"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preflight", "train"], default="preflight")
    parser.add_argument("--master-port", default="29681")
    parser.add_argument("--run-name", default="libero_rlvr_grpo_v3_preflight_20260810")
    parser.add_argument("--policy-checkpoint", type=Path, default=v1.SFT_CKPT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--deepspeed-config",
        default=str(ROOT / "DEEPSPEED_OR_FSDP_CONFIG/zero3_no_offload.json"),
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--scan-limit-rows", type=int, default=None)
    parser.add_argument(
        "--prompt-pool-size",
        type=int,
        default=0,
        help="0 or negative means all valid progress refs",
    )
    parser.add_argument(
        "--strict-target-range", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument(
        "--loss-type", choices=["grpo", "bnpo", "dapo", "dr_grpo"], default="grpo"
    )
    parser.add_argument(
        "--importance-sampling-level",
        choices=["token", "sequence", "sequence_sum"],
        default="token",
    )
    parser.add_argument("--scale-rewards", choices=["group", "none"], default="group")
    parser.add_argument("--advantage-eps", type=float, default=0.0)
    parser.add_argument("--advantage-std-threshold", type=float, default=1e-5)
    parser.add_argument(
        "--advantage-std-unbiased", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--out-of-range-policy",
        choices=["fail_format", "clip", "shape"],
        default="fail_format",
    )
    parser.add_argument("--completion-eos-suffix", default="<|im_end|>\n")
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--eval-group-size", type=int, default=4)
    parser.add_argument("--eval-every-steps", type=int, default=100)
    parser.add_argument("--export-root", default=str(OUT_ROOT_V3 / "hf_exports"))
    parser.add_argument("--export-every-steps", type=int, default=100)
    parser.add_argument(
        "--export-final", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--tensorboard", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
