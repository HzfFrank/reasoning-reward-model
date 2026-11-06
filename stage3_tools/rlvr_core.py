#!/usr/bin/env python3
"""Core RLVR math and reward utilities for the SOLE-R1 reproduction.

This module is intentionally independent from the Stage 2A SFT pipeline.  It
contains only lightweight, testable pieces: strict progress-answer parsing,
rule-based reward computation, group advantage normalization, and the clipped
GRPO objective terms described in the SOLE-R1 paper.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Iterable, Literal, Sequence

import torch

PAPER_CONFIRMED_RLVR = {
    "group_size_G": 8,
    "temperature": 0.7,
    "top_p": 0.9,
    "clip_epsilon": 0.2,
    "kl_beta": 0.01,
    "rlvr_steps": 5000,
    "learning_rate": 1e-6,
    "format_reward_max": 0.5,
    "accuracy_reward_max_alpha": 1.5,
    "reward_total_range": [0.0, 2.0],
}

ASSUMED_RLVR_DETAILS = {
    "tau": {
        "value": 10.0,
        "reason": "The paper gives r_acc = alpha * exp(-abs(error) / tau) but does not publish tau.",
    },
    "zero_reward_std_advantage": {
        "value": "all_zero_advantages",
        "reason": "The paper says divide by group std but does not specify the degenerate all-equal reward case.",
    },
    "std_definition": {
        "value": "population_std_over_G",
        "reason": "The paper writes std over the G sampled candidates without a Bessel correction.",
    },
    "kl_estimator": {
        "value": "token_or_sequence estimator must be fixed before full RLVR",
        "reason": "The paper states beta * D_KL(p_phi || p_ref) but does not expose implementation code.",
    },
    "out_of_range_answer_policy": {
        "value": "fail_format for v3 strict runs",
        "reason": "The paper defines progress as a signed integer in [-100, 100] but does not define rewards for parseable out-of-range integers.",
    },
}


@dataclass(frozen=True)
class ParsedProgressAnswer:
    ok: bool
    answer: int | None
    reason: str
    matched_text: str | None = None


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    format_reward: float
    accuracy_reward: float
    answer: int | None
    target: int
    abs_error: int | None
    parse_ok: bool
    parse_reason: str


ANSWER_RE = re.compile(
    r"^\s*<think>.*?</think>\s*<answer>\s*([+-]?\d+)\s*%\s*</answer>\s*$",
    re.DOTALL,
)


def parse_progress_answer_strict(text: str) -> ParsedProgressAnswer:
    """Parse the strict SOLE progress format used by RLVR reward tests.

    The training prompt asks the model to answer with
    ``<think>...</think><answer>p%</answer>`` where p is an integer.  This
    strict parser rejects missing tags, multiple answers, and non-integers.
    Out-of-range integers are still parsed, because prior Stage 1 golden tests
    compute a shaped reward for them instead of making them parse failures.
    """

    if not isinstance(text, str):
        return ParsedProgressAnswer(False, None, "not_a_string")
    if text.count("<answer>") != 1 or text.count("</answer>") != 1:
        return ParsedProgressAnswer(False, None, "answer_tag_count_not_one")
    if text.count("<think>") != 1 or text.count("</think>") != 1:
        return ParsedProgressAnswer(False, None, "think_tag_count_not_one")
    match = ANSWER_RE.match(text)
    if not match:
        return ParsedProgressAnswer(False, None, "format_mismatch")
    answer = int(match.group(1))
    reason = "ok" if -100 <= answer <= 100 else "out_of_range"
    return ParsedProgressAnswer(True, answer, reason, match.group(0))


def progress_reward(
    text: str,
    target: int,
    *,
    tau: float,
    alpha: float = 1.5,
    format_reward: float = 0.5,
    out_of_range_policy: Literal["fail_format", "clip", "shape"] = "shape",
) -> RewardBreakdown:
    """Compute SOLE-R1's rule reward for one generated completion."""

    if tau is None or tau <= 0:
        raise ValueError("tau must be provided and positive")
    parsed = parse_progress_answer_strict(text)
    if not parsed.ok or parsed.answer is None:
        return RewardBreakdown(
            total=0.0,
            format_reward=0.0,
            accuracy_reward=0.0,
            answer=None,
            target=int(target),
            abs_error=None,
            parse_ok=False,
            parse_reason=parsed.reason,
        )
    answer = int(parsed.answer)
    parse_reason = parsed.reason
    if parse_reason == "out_of_range":
        if out_of_range_policy == "fail_format":
            return RewardBreakdown(
                total=0.0,
                format_reward=0.0,
                accuracy_reward=0.0,
                answer=answer,
                target=int(target),
                abs_error=None,
                parse_ok=False,
                parse_reason=parse_reason,
            )
        if out_of_range_policy == "clip":
            answer = max(-100, min(100, answer))
            parse_reason = "out_of_range_clipped"
        elif out_of_range_policy != "shape":
            raise ValueError(f"unsupported out_of_range_policy: {out_of_range_policy}")
    error = abs(answer - int(target))
    acc = float(alpha * math.exp(-error / tau))
    return RewardBreakdown(
        total=float(format_reward + acc),
        format_reward=float(format_reward),
        accuracy_reward=acc,
        answer=answer,
        target=int(target),
        abs_error=int(error),
        parse_ok=True,
        parse_reason=parse_reason,
    )


def standardize_group_rewards(
    rewards: Sequence[float],
    *,
    eps: float = 1e-8,
) -> list[float]:
    """Return GRPO within-group standardized advantages."""

    if not rewards:
        raise ValueError("rewards must be non-empty")
    vals = [float(r) for r in rewards]
    mean = sum(vals) / len(vals)
    variance = sum((r - mean) ** 2 for r in vals) / len(vals)
    std = math.sqrt(variance)
    if std <= eps:
        return [0.0 for _ in vals]
    return [(r - mean) / std for r in vals]


def grpo_clipped_policy_loss(
    new_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    *,
    epsilon: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the negative clipped GRPO policy objective for sampled outputs."""

    if new_logps.shape != old_logps.shape or new_logps.shape != advantages.shape:
        raise ValueError(
            "new_logps, old_logps, and advantages must have identical shapes"
        )
    ratios = torch.exp(new_logps - old_logps)
    clipped_ratios = torch.clamp(ratios, 1.0 - epsilon, 1.0 + epsilon)
    objective_terms = torch.minimum(ratios * advantages, clipped_ratios * advantages)
    loss = -objective_terms.mean()
    stats = {
        "ratio_mean": float(ratios.detach().float().mean().cpu().item()),
        "ratio_min": float(ratios.detach().float().min().cpu().item()),
        "ratio_max": float(ratios.detach().float().max().cpu().item()),
        "objective_mean": float(objective_terms.detach().float().mean().cpu().item()),
        "policy_loss": float(loss.detach().float().cpu().item()),
    }
    return loss, stats


def completion_token_logps(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-token logprobs and the completion mask for causal LM labels.

    ``labels`` must use ``-100`` for prompt/image/padding positions.  The causal
    shift is the same one used by Hugging Face causal language models: logits at
    position ``t`` predict the token at position ``t + 1``.
    """

    if input_ids.ndim != 2 or labels.ndim != 2:
        raise ValueError("input_ids and labels must be rank-2 tensors")
    if logits.ndim != 3:
        raise ValueError("logits must be rank-3 [batch, seq, vocab]")
    if input_ids.shape != labels.shape:
        raise ValueError("input_ids and labels must have identical shapes")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits sequence shape must match input_ids")

    shifted_logits = logits[:, :-1, :].float()
    shifted_input_ids = input_ids[:, 1:]
    shifted_labels = labels[:, 1:]
    completion_mask = shifted_labels.ne(-100)
    token_logps = (
        torch.log_softmax(shifted_logits, dim=-1)
        .gather(-1, shifted_input_ids.unsqueeze(-1))
        .squeeze(-1)
    )
    return token_logps, completion_mask


def grpo_token_loss(
    per_token_logps: torch.Tensor,
    old_per_token_logps: torch.Tensor | None,
    ref_per_token_logps: torch.Tensor | None,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    *,
    epsilon: float = 0.2,
    beta: float = 0.01,
    loss_type: Literal["grpo", "bnpo", "dapo", "dr_grpo"] = "grpo",
    importance_sampling_level: Literal["token", "sequence", "sequence_sum"] = "token",
    max_completion_length: int | None = None,
    dapo_normalizer: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute TRL/RLinf-style masked token-level GRPO loss.

    This mirrors the shape of Hugging Face TRL 0.23's GRPOTrainer and RLinf's
    PPO/GRPO policy loss: advantages are sequence-level rewards, but the ratio,
    clipping, KL penalty, and completion mask are applied on completion tokens.
    """

    if old_per_token_logps is None:
        old_per_token_logps = per_token_logps.detach()
    if per_token_logps.shape != old_per_token_logps.shape:
        raise ValueError("new and old per-token logprobs must have identical shapes")
    if completion_mask.shape != per_token_logps.shape:
        raise ValueError("completion_mask must match per-token logprob shape")
    if advantages.ndim != 1 or advantages.shape[0] != per_token_logps.shape[0]:
        raise ValueError("advantages must be rank-1 with one value per sequence")
    if beta and ref_per_token_logps is None:
        raise ValueError("ref_per_token_logps is required when beta > 0")
    if (
        ref_per_token_logps is not None
        and ref_per_token_logps.shape != per_token_logps.shape
    ):
        raise ValueError("reference per-token logprobs must match policy logprobs")

    mask = completion_mask.float()
    token_count = mask.sum().clamp_min(1.0)
    log_ratio = per_token_logps - old_per_token_logps
    if importance_sampling_level == "token":
        log_importance_weights = log_ratio
    elif importance_sampling_level == "sequence":
        seq_log_ratio = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)
        log_importance_weights = seq_log_ratio.unsqueeze(-1)
    elif importance_sampling_level == "sequence_sum":
        seq_log_ratio = (log_ratio * mask).sum(-1)
        log_importance_weights = seq_log_ratio.unsqueeze(-1)
    else:
        raise ValueError(
            f"unsupported importance_sampling_level: {importance_sampling_level}"
        )

    ratios = torch.exp(log_importance_weights)
    clipped = torch.clamp(ratios, 1.0 - epsilon, 1.0 + epsilon)
    adv = advantages.float().unsqueeze(1)
    unclipped_policy_loss = -torch.minimum(ratios * adv, clipped * adv)
    per_token_loss = unclipped_policy_loss

    if beta:
        ref_minus_new = ref_per_token_logps - per_token_logps  # type: ignore[operator]
        per_token_kl = torch.exp(ref_minus_new) - ref_minus_new - 1.0
        per_token_loss = per_token_loss + beta * per_token_kl
    else:
        per_token_kl = torch.zeros_like(per_token_logps)

    if loss_type == "grpo":
        loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)).mean()
    elif loss_type == "bnpo":
        loss = (per_token_loss * mask).sum() / token_count
    elif loss_type == "dapo":
        normalizer = dapo_normalizer
        if normalizer is None:
            normalizer = token_count
        if not torch.is_tensor(normalizer):
            normalizer = torch.tensor(
                float(normalizer),
                dtype=per_token_loss.dtype,
                device=per_token_loss.device,
            )
        loss = (per_token_loss * mask).sum() / normalizer.clamp_min(1.0)
    elif loss_type == "dr_grpo":
        if max_completion_length is None or max_completion_length <= 0:
            raise ValueError("max_completion_length must be positive for dr_grpo")
        loss = (per_token_loss * mask).sum() / (
            per_token_loss.shape[0] * max_completion_length
        )
    else:
        raise ValueError(f"unsupported loss_type: {loss_type}")

    is_low_clipped = (ratios < 1.0 - epsilon) & (adv < 0)
    is_high_clipped = (ratios > 1.0 + epsilon) & (adv > 0)
    clip_mask = (is_low_clipped | is_high_clipped).float()
    if ratios.shape[1] == 1:
        metric_mask = torch.ones_like(ratios)
        metric_count = torch.tensor(float(ratios.numel()), device=ratios.device)
    else:
        metric_mask = mask
        metric_count = token_count
    stats = {
        "policy_loss": (unclipped_policy_loss * mask).sum() / token_count,
        "kl": (per_token_kl * mask).sum() / token_count,
        "ratio": (ratios * metric_mask).sum() / metric_count.clamp_min(1.0),
        "ratio_abs": ((ratios - 1.0).abs() * metric_mask).sum()
        / metric_count.clamp_min(1.0),
        "clipped_ratio": (clipped * metric_mask).sum() / metric_count.clamp_min(1.0),
        "clip_fraction": (clip_mask * metric_mask).sum() / metric_count.clamp_min(1.0),
        "token_count": token_count,
    }
    return loss, stats


def reward_breakdown_to_dict(item: RewardBreakdown) -> dict:
    return asdict(item)


def assert_finite_values(values: Iterable[float], name: str) -> None:
    bad = [v for v in values if not math.isfinite(float(v))]
    if bad:
        raise ValueError(f"{name} contains non-finite values: {bad[:3]}")
