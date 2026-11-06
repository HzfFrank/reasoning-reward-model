from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from toolkits.libero_sole.common import find_episode_dirs, write_json


SUPPORTED_GOAL_PREDICATES = {"On"}


def _as_vec(state: dict[str, Any], key: str) -> np.ndarray:
    value = state.get(key)
    if value is None:
        raise ValueError(f"semantic state is missing {key!r}")
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] < 3:
        raise ValueError(f"semantic state field {key!r} must have at least 3 values")
    return arr[:3]


def _normalize_goal_predicate(predicate: Any) -> str | None:
    if isinstance(predicate, (list, tuple)) and predicate:
        return str(predicate[0])
    if isinstance(predicate, str):
        return predicate
    return None


def _load_semantic_states(path: Path) -> list[dict[str, Any]]:
    with open(path, "rb") as f:
        states = pickle.load(f)
    normalized = []
    for state in states:
        if isinstance(state, list):
            if len(state) != 1:
                raise ValueError(
                    "semantic_states.pkl contains batched entries; expected per-env states"
                )
            state = state[0]
        if not isinstance(state, dict):
            raise ValueError("semantic_states.pkl entries must be dictionaries")
        normalized.append(state)
    return normalized


def compute_pick_place_progress(
    semantic_states: list[dict[str, Any]], beta: float, monotonic: bool
) -> tuple[np.ndarray, dict[str, Any]]:
    if not semantic_states:
        raise ValueError("empty semantic state sequence")

    predicate = _normalize_goal_predicate(semantic_states[0].get("goal_predicate"))
    if predicate is not None and predicate not in SUPPORTED_GOAL_PREDICATES:
        raise ValueError(f"unsupported goal predicate {predicate!r}")

    eef = np.stack([_as_vec(s, "eef_pos") for s in semantic_states])
    obj = np.stack([_as_vec(s, "object_pos") for s in semantic_states])
    goal = np.stack([_as_vec(s, "goal_pos") for s in semantic_states])

    d_eef_obj = np.linalg.norm(eef - obj, axis=1)
    d_obj_goal = np.linalg.norm(obj - goal, axis=1)
    y = (1.0 - beta) * d_eef_obj + beta * d_obj_goal

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    denom = y_max - y_min
    if denom <= 1e-9:
        progress = np.zeros_like(y, dtype=np.float32)
    else:
        progress = ((-y + y_max) / denom * 100.0).astype(np.float32)

    if monotonic:
        progress = np.maximum.accumulate(progress)
    if semantic_states[-1].get("success"):
        progress[-1] = max(float(progress[-1]), 100.0)

    diagnostics = {
        "formula": (
            "progress = (-y + max(y)) / (max(y) - min(y)) * 100; "
            "y=(1-beta)*d_eef_obj+beta*d_obj_goal"
        ),
        "beta": beta,
        "monotonic": monotonic,
        "object_name": semantic_states[0].get("object_name"),
        "goal_name": semantic_states[0].get("goal_name"),
        "goal_predicate": semantic_states[0].get("goal_predicate"),
        "y_min": y_min,
        "y_max": y_max,
        "d_eef_obj_first": float(d_eef_obj[0]),
        "d_eef_obj_last": float(d_eef_obj[-1]),
        "d_obj_goal_first": float(d_obj_goal[0]),
        "d_obj_goal_last": float(d_obj_goal[-1]),
        "progress_first": float(progress[0]) if len(progress) else None,
        "progress_last": float(progress[-1]) if len(progress) else None,
    }
    return progress.astype(np.float32), diagnostics


def process_episode(episode_dir: Path, beta: float, monotonic: bool, overwrite: bool) -> bool:
    semantic_path = episode_dir / "semantic_states.pkl"
    progress_path = episode_dir / "progress.npy"
    metadata_path = episode_dir / "progress_metadata.json"
    if progress_path.exists() and not overwrite:
        print(f"skip existing progress: {episode_dir}")
        return False
    if not semantic_path.is_file():
        print(f"skip missing semantic_states.pkl: {episode_dir}")
        return False

    try:
        semantic_states = _load_semantic_states(semantic_path)
        progress, diagnostics = compute_pick_place_progress(
            semantic_states=semantic_states, beta=beta, monotonic=monotonic
        )
    except ValueError as exc:
        reason = {"supported": False, "reason": str(exc)}
        write_json(metadata_path, reason)
        print(f"skip unsupported progress: {episode_dir}: {exc}")
        return False

    np.save(progress_path, progress)
    diagnostics["supported"] = True
    write_json(metadata_path, diagnostics)
    print(
        f"wrote {progress_path} shape={progress.shape} "
        f"first={progress[0]:.2f} last={progress[-1]:.2f}"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute LIBERO Spatial SOLE-R1 style progress labels."
    )
    parser.add_argument("path", help="Episode directory or trajectory root.")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--no-monotonic", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.beta <= 1.0:
        raise ValueError("--beta must be in [0, 1]")

    episode_dirs = find_episode_dirs(args.path)
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories found under {args.path}")

    written = 0
    for episode_dir in episode_dirs:
        written += int(
            process_episode(
                episode_dir=episode_dir,
                beta=args.beta,
                monotonic=not args.no_monotonic,
                overwrite=args.overwrite,
            )
        )
    print(f"processed={len(episode_dirs)} wrote={written}")


if __name__ == "__main__":
    main()
