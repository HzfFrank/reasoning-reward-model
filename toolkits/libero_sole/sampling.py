from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from toolkits.libero_sole.common import episode_id_from_dir, find_episode_dirs, write_json
from toolkits.libero_sole.compute_progress import _load_semantic_states


DEFAULT_NUM_SAMPLES = 25
DEFAULT_MAX_EVENT_SAMPLES = 5
DEFAULT_LIFT_THRESHOLD = 0.03
EVENT_PRIORITIES = {"success": 0, "contact": 1, "lift": 2}


def sample_timesteps(
    episode_length: int,
    semantic_states: list[dict[str, Any]] | None = None,
    contact_records: list[dict[str, Any]] | None = None,
    strategy: str = "hybrid",
    num_samples: int = DEFAULT_NUM_SAMPLES,
    interval: int | None = None,
    max_event_samples: int = DEFAULT_MAX_EVENT_SAMPLES,
) -> list[int]:
    """Sample sparse timesteps from a trajectory.

    The returned list is sorted, unique, and clipped to ``[0, episode_length - 1]``.
    Downstream sample builders should use the previous element in this list as the
    previous timestep, not ``t - 1``.
    """
    if episode_length <= 0:
        return []

    normalized_strategy = strategy.replace("-", "_").lower()
    if normalized_strategy == "uniform":
        timesteps = _uniform_timesteps(episode_length, num_samples)
    elif normalized_strategy == "interval":
        if interval is None:
            raise ValueError("interval sampling requires interval")
        timesteps = _interval_timesteps(episode_length, interval)
    elif normalized_strategy in {"event_aware", "event"}:
        timesteps = _uniform_timesteps(episode_length, num_samples)
        timesteps.extend(
            event_timesteps(
                episode_length=episode_length,
                semantic_states=semantic_states,
                contact_records=contact_records,
                max_event_samples=None,
            )
        )
    elif normalized_strategy == "hybrid":
        timesteps = _uniform_timesteps(episode_length, num_samples)
        timesteps.extend(
            event_timesteps(
                episode_length=episode_length,
                semantic_states=semantic_states,
                contact_records=contact_records,
                max_event_samples=max_event_samples,
            )
        )
    else:
        raise ValueError(
            "unsupported sampling strategy "
            f"{strategy!r}; expected uniform, interval, event_aware, or hybrid"
        )

    return _sanitize_timesteps(timesteps, episode_length)


def previous_timestep_pairs(timesteps: list[int]) -> list[dict[str, int]]:
    """Return current/previous pairs using the sampled sequence order."""
    pairs = []
    for idx, timestep in enumerate(timesteps):
        previous = timesteps[idx - 1] if idx > 0 else timesteps[0]
        pairs.append({"previous": previous, "current": timestep})
    return pairs


def event_timesteps(
    episode_length: int,
    semantic_states: list[dict[str, Any]] | None = None,
    contact_records: list[dict[str, Any]] | None = None,
    max_event_samples: int | None = None,
) -> list[int]:
    """Find task-event timesteps for event-aware or hybrid sampling."""
    candidates = event_candidates(
        episode_length=episode_length,
        semantic_states=semantic_states,
        contact_records=contact_records,
    )
    if max_event_samples is not None:
        candidates = _limit_event_candidates(candidates, max_event_samples)
    return _sanitize_timesteps([c["timestep"] for c in candidates], episode_length)


def event_candidates(
    episode_length: int,
    semantic_states: list[dict[str, Any]] | None = None,
    contact_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return typed event timestep candidates before hybrid limiting."""
    if episode_length <= 0:
        return []

    candidates: list[dict[str, Any]] = []
    success_timestep = _first_success_timestep(semantic_states, episode_length)
    event_cutoff = success_timestep if success_timestep is not None else episode_length - 1

    contact_flags = _contact_flags(contact_records, episode_length)
    for timestep in _false_to_true_timesteps(contact_flags):
        if timestep <= event_cutoff:
            candidates.append({"type": "contact", "timestep": timestep})

    lift_flags = _lift_flags(semantic_states, episode_length)
    for timestep in _false_to_true_timesteps(lift_flags):
        if timestep <= event_cutoff:
            candidates.append({"type": "lift", "timestep": timestep})

    if success_timestep is not None:
        candidates.append({"type": "success", "timestep": success_timestep})

    return sorted(
        candidates,
        key=lambda item: (int(item["timestep"]), EVENT_PRIORITIES.get(str(item["type"]), 99)),
    )


def _limit_event_candidates(
    candidates: list[dict[str, Any]], max_event_samples: int
) -> list[dict[str, Any]]:
    if max_event_samples <= 0 or not candidates:
        return []

    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(str(candidate["type"]), []).append(candidate)

    nonempty_types = sorted(grouped, key=lambda t: EVENT_PRIORITIES.get(t, 99))
    per_type_limit = max(1, math.ceil(max_event_samples / max(1, len(nonempty_types))))
    prelimited: list[dict[str, Any]] = []
    for event_type in nonempty_types:
        prelimited.extend(sorted(grouped[event_type], key=lambda item: int(item["timestep"]))[:per_type_limit])

    selected_keys = {(str(item["type"]), int(item["timestep"])) for item in prelimited}
    if len(prelimited) < max_event_samples:
        for candidate in sorted(candidates, key=lambda item: int(item["timestep"])):
            key = (str(candidate["type"]), int(candidate["timestep"]))
            if key in selected_keys:
                continue
            prelimited.append(candidate)
            selected_keys.add(key)
            if len(prelimited) >= max_event_samples:
                break

    if len(prelimited) > max_event_samples:
        prelimited = sorted(
            prelimited,
            key=lambda item: (
                EVENT_PRIORITIES.get(str(item["type"]), 99),
                int(item["timestep"]),
            ),
        )[:max_event_samples]
    return sorted(
        prelimited,
        key=lambda item: (int(item["timestep"]), EVENT_PRIORITIES.get(str(item["type"]), 99)),
    )


def _uniform_timesteps(episode_length: int, num_samples: int) -> list[int]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if num_samples >= episode_length:
        return list(range(episode_length))
    values = np.linspace(0, episode_length - 1, num=num_samples)
    return [int(round(value)) for value in values]


def _interval_timesteps(episode_length: int, interval: int) -> list[int]:
    if interval <= 0:
        raise ValueError("interval must be positive")
    timesteps = list(range(0, episode_length, interval))
    if timesteps[-1] != episode_length - 1:
        timesteps.append(episode_length - 1)
    return timesteps


def _sanitize_timesteps(timesteps: list[int], episode_length: int) -> list[int]:
    return sorted({int(t) for t in timesteps if 0 <= int(t) < episode_length})


def _load_contact_records(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    with open(path, "rb") as f:
        records = pickle.load(f)
    normalized = []
    for record in records:
        if isinstance(record, list):
            record = record[0] if len(record) == 1 else {}
        normalized.append(record if isinstance(record, dict) else {})
    return normalized


def _contact_flags(
    contact_records: list[dict[str, Any]] | None, episode_length: int
) -> list[bool | None]:
    flags: list[bool | None] = [None] * episode_length
    if contact_records is None:
        return flags
    for idx, record in enumerate(contact_records[:episode_length]):
        contacts = record.get("contacts") or []
        flags[idx] = any(bool(contact.get("is_robot_object")) for contact in contacts)
    return flags


def _lift_flags(
    semantic_states: list[dict[str, Any]] | None, episode_length: int
) -> list[bool | None]:
    flags: list[bool | None] = [None] * episode_length
    if not semantic_states:
        return flags
    initial_obj = _as_vec(semantic_states[0], "object_pos")
    if initial_obj is None:
        return flags
    for idx, state in enumerate(semantic_states[:episode_length]):
        obj = _as_vec(state, "object_pos")
        flags[idx] = None if obj is None else bool((obj[2] - initial_obj[2]) > DEFAULT_LIFT_THRESHOLD)
    return flags


def _false_to_true_timesteps(flags: list[bool | None]) -> list[int]:
    timesteps: list[int] = []
    previous: bool | None = None
    for idx, current in enumerate(flags):
        if current is None:
            continue
        if previous is False and current is True:
            timesteps.append(idx)
        previous = current
    return timesteps


def _first_success_timestep(
    semantic_states: list[dict[str, Any]] | None, episode_length: int
) -> int | None:
    if not semantic_states:
        return None
    for idx, state in enumerate(semantic_states[:episode_length]):
        if bool(state.get("success", False)):
            return idx
    return None


def _as_vec(state: dict[str, Any], key: str) -> np.ndarray | None:
    value = state.get(key)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] < 3:
        return None
    return arr[:3]


def _load_progress(episode_dir: Path, episode_length: int) -> np.ndarray | None:
    for name in ("progress_contact.npy", "progress.npy"):
        path = episode_dir / name
        if path.is_file():
            return np.load(path)[:episode_length]
    return None


def _gap_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "values": []}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.shape[0]),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "values": [float(v) for v in values],
    }


def _progress_values(progress: np.ndarray | None, timesteps: list[int]) -> list[float] | None:
    if progress is None:
        return None
    return [float(progress[t]) for t in timesteps]


def _progress_transitions(
    progress: np.ndarray | None, timesteps: list[int]
) -> list[dict[str, float | int]]:
    if progress is None:
        return []
    transitions = []
    for timestep, next_timestep in zip(timesteps, timesteps[1:]):
        current_progress = float(progress[timestep])
        next_progress = float(progress[next_timestep])
        transitions.append(
            {
                "timestep": int(timestep),
                "progress": current_progress,
                "next_timestep": int(next_timestep),
                "next_progress": next_progress,
                "delta_progress": next_progress - current_progress,
            }
        )
    return transitions


def summarize_episode_sampling(
    episode_dir: Path,
    strategy: str,
    num_samples: int,
    interval: int | None,
    max_event_samples: int,
) -> dict[str, Any]:
    semantic_path = episode_dir / "semantic_states.pkl"
    semantic_states = _load_semantic_states(semantic_path) if semantic_path.is_file() else None
    contact_records = _load_contact_records(episode_dir / "contact_records.pkl")

    image_count = _count_images(episode_dir / "images" / "main")
    lengths = [value for value in [image_count, len(semantic_states or [])] if value > 0]
    if not lengths:
        raise ValueError(f"cannot infer episode length for {episode_dir}")
    episode_length = min(lengths)

    timesteps = sample_timesteps(
        episode_length=episode_length,
        semantic_states=semantic_states,
        contact_records=contact_records,
        strategy=strategy,
        num_samples=num_samples,
        interval=interval,
        max_event_samples=max_event_samples,
    )
    timestep_gaps = [b - a for a, b in zip(timesteps, timesteps[1:])]

    progress = _load_progress(episode_dir, episode_length)
    progress_values = _progress_values(progress, timesteps)
    progress_transitions = _progress_transitions(progress, timesteps)
    progress_gaps = [float(item["delta_progress"]) for item in progress_transitions]

    raw_events = event_candidates(
        episode_length=episode_length,
        semantic_states=semantic_states,
        contact_records=contact_records,
    )
    limited_events = _limit_event_candidates(raw_events, max_event_samples)

    return {
        "episode": episode_id_from_dir(episode_dir),
        "episode_dir": str(episode_dir),
        "episode_length": episode_length,
        "strategy": strategy,
        "num_samples": num_samples,
        "interval": interval,
        "max_event_samples": max_event_samples,
        "sampled_timesteps": timesteps,
        "previous_pairs": previous_timestep_pairs(timesteps),
        "event_candidates": raw_events,
        "limited_event_candidates": limited_events,
        "sample_count": len(timesteps),
        "timestep_gaps": timestep_gaps,
        "timestep_gap_summary": _gap_summary([float(v) for v in timestep_gaps]),
        "progress_values": progress_values,
        "progress_transitions": progress_transitions,
        "progress_gaps": progress_gaps,
        "progress_gap_summary": _gap_summary(progress_gaps),
    }


def _count_images(path: Path) -> int:
    if not path.is_dir():
        return 0
    return len(sorted(path.glob("*.png")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug SOLE-R1 style timestep sampling.")
    parser.add_argument("path", help="Episode directory or trajectory root")
    parser.add_argument(
        "--strategy",
        choices=["uniform", "interval", "event_aware", "event-aware", "event", "hybrid"],
        default="hybrid",
    )
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--interval", type=int, default=None)
    parser.add_argument("--max-event-samples", type=int, default=DEFAULT_MAX_EVENT_SAMPLES)
    parser.add_argument("--max-episodes", type=int, default=3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    episode_dirs = find_episode_dirs(args.path)
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories found under {args.path}")
    episode_dirs = episode_dirs[: args.max_episodes]

    episodes = [
        summarize_episode_sampling(
            episode_dir=episode_dir,
            strategy=args.strategy,
            num_samples=args.num_samples,
            interval=args.interval,
            max_event_samples=args.max_event_samples,
        )
        for episode_dir in episode_dirs
    ]
    payload = {
        "sampler": {
            "interface": "sample_timesteps(episode_length, semantic_states=None, contact_records=None, strategy='hybrid', num_samples=25, interval=None, max_event_samples=5)",
            "previous_timestep_rule": "previous is the preceding element in the sampled timestep sequence, not raw t-1",
            "hybrid_rule": "uniform samples plus at most max_event_samples typed event timesteps, deduplicated and sorted",
        },
        "episodes": episodes,
    }

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        write_json(output_path, payload)
        print(f"wrote sampling debug: {output_path}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
