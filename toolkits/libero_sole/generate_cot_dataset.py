from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image as PILImage

from toolkits.libero_sole.common import (
    episode_id_from_dir,
    find_episode_dirs,
    load_json,
    success_from_episode,
    task_description_from_episode,
)
from toolkits.libero_sole.compute_progress import _load_semantic_states
from toolkits.libero_sole.sampling import (
    DEFAULT_MAX_EVENT_SAMPLES,
    DEFAULT_NUM_SAMPLES,
)
from toolkits.libero_sole.sampling import (
    sample_timesteps as sample_sparse_timesteps,
)

SYSTEM_PROMPT = (
    "You are an expert roboticist with the goal of predicting task progress "
    "percentages given frames from a video of a robot attempting to complete "
    "a task. You first think, in the form of an internal monologue, before "
    "providing your final answer. Your reasoning process MUST BE enclosed "
    "within <think> </think> tags and should include detailed reasoning. "
    "Your final answer MUST BE enclosed within <answer> </answer> tags and "
    "should be a integer (positive or negative) representing current task "
    "progress percentage. Example output format: "
    "<think>[detailed reasoning process]</think>"
    "<answer>[current task progress]%</answer>"
)

DEFAULT_QWEN_MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
OFFICIAL_COLUMNS = [
    "image",
    "problem",
    "solution_orig",
    "solution",
    "prompt",
    "image_name",
    "from_annotation",
    "completion",
    "data_source",
]


@dataclass(frozen=True)
class SamplingConfig:
    strategy: str = "hybrid"
    interval: int | None = None
    num_samples: int = DEFAULT_NUM_SAMPLES
    max_event_samples: int = DEFAULT_MAX_EVENT_SAMPLES
    include_last: bool = True
    max_sample_gap: int | None = None


@dataclass(frozen=True)
class CotSample:
    episode_dir: Path
    episode_id: str
    timestep: int
    previous_timestep: int
    total_steps: int
    image_path: Path
    task_description: str
    progress: float
    previous_progress: float
    success: bool | None
    transition: dict[str, Any]
    template_reasoning: str


@dataclass(frozen=True)
class DebugItem:
    sample: CotSample
    qwen_reasoning: str
    row: dict[str, Any]


def sample_timesteps(
    total_steps: int,
    config: SamplingConfig,
    semantic_states: list[dict[str, Any]] | None = None,
    contact_records: list[dict[str, Any]] | None = None,
) -> list[int]:
    """Sample timesteps without binding downstream CoT logic to one strategy."""
    if total_steps <= 0:
        return []
    strategy = config.strategy.replace("-", "_").lower()
    if strategy == "num_samples":
        strategy = "uniform"
    if strategy == "interval" and config.interval is None:
        raise ValueError("interval sampling requires interval")
    timesteps = sample_sparse_timesteps(
        episode_length=total_steps,
        semantic_states=semantic_states,
        contact_records=contact_records,
        strategy=strategy,
        num_samples=config.num_samples,
        interval=config.interval,
        max_event_samples=config.max_event_samples,
    )

    if config.include_last and (total_steps - 1) not in timesteps:
        timesteps.append(total_steps - 1)
    return _densify_timesteps(
        sorted(set(timesteps)), total_steps, config.max_sample_gap
    )


def _densify_timesteps(
    timesteps: list[int], total_steps: int, max_gap: int | None
) -> list[int]:
    if not timesteps or max_gap is None or max_gap <= 0:
        return timesteps
    dense = [timesteps[0]]
    for start, end in zip(timesteps, timesteps[1:]):
        cursor = start + max_gap
        while cursor < end:
            dense.append(cursor)
            cursor += max_gap
        dense.append(end)
    return sorted({t for t in dense if 0 <= t < total_steps})


def _sampling_name(config: SamplingConfig) -> str:
    strategy = config.strategy.replace("-", "_").lower()
    if strategy == "interval":
        base = f"interval_{config.interval}"
    elif strategy in {"num_samples", "uniform"}:
        base = f"uniform_{config.num_samples}"
    elif strategy in {"event_aware", "event"}:
        base = f"event_aware_{config.num_samples}"
    elif strategy == "hybrid":
        base = f"hybrid_{config.num_samples}_maxevent_{config.max_event_samples}"
    else:
        base = strategy
    if config.max_sample_gap is not None and config.max_sample_gap > 0:
        base = f"{base}_maxgap_{config.max_sample_gap}"
    return f"{base}_with_last" if config.include_last else base


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _format_progress_bar(done: int, total: int, width: int = 32) -> str:
    if total <= 0:
        return "[................................] 0/0   0.0%"
    fraction = min(max(done / total, 0.0), 1.0)
    filled = int(round(fraction * width))
    return (
        f"[{'#' * filled}{'.' * (width - filled)}] "
        f"{done}/{total} {fraction * 100:5.1f}%"
    )


def _write_progress_json(
    path: Path,
    status: str,
    processed: int,
    total: int,
    start_time: float,
    batch_idx: int | None = None,
    message: str | None = None,
) -> None:
    elapsed = max(time.monotonic() - start_time, 0.0)
    rate = processed / elapsed if elapsed > 0 else 0.0
    remaining = max(total - processed, 0)
    eta_seconds = remaining / rate if rate > 0 else None
    _write_json(
        path,
        {
            "status": status,
            "processed": processed,
            "total": total,
            "percent": round((processed / total * 100.0) if total else 0.0, 3),
            "batch_idx": batch_idx,
            "elapsed_seconds": round(elapsed, 3),
            "samples_per_second": round(rate, 6),
            "eta_seconds": round(eta_seconds, 3) if eta_seconds is not None else None,
            "message": message,
        },
    )


def _load_progress(episode_dir: Path, preferred_name: str) -> np.ndarray:
    candidates = [preferred_name, "progress_contact.npy", "progress.npy"]
    for name in candidates:
        path = episode_dir / name
        if path.is_file():
            return np.load(path)
    raise FileNotFoundError(
        f"missing progress file under {episode_dir}; tried {', '.join(candidates)}"
    )


def _success_flags_from_states(semantic_states: list[dict[str, Any]]) -> list[bool]:
    return [bool(state.get("success", False)) for state in semantic_states]


def _first_success_timestep(success_flags: list[bool]) -> int | None:
    return next((idx for idx, success in enumerate(success_flags) if success), None)


def _truncate_timesteps_at_success(
    timesteps: list[int], success_flags: list[bool]
) -> list[int]:
    first_success = _first_success_timestep(success_flags)
    if first_success is None:
        return timesteps
    kept = [t for t in timesteps if t <= first_success]
    if first_success not in kept:
        kept.append(first_success)
    return sorted(set(kept))


def _as_vec(state: dict[str, Any], key: str) -> np.ndarray | None:
    value = state.get(key)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] < 3:
        return None
    return arr[:3]


def _distance(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(a - b))


def _norm_delta(after: np.ndarray | None, before: np.ndarray | None) -> float | None:
    if after is None or before is None:
        return None
    return float(np.linalg.norm(after - before))


def _load_contact_records(episode_dir: Path) -> list[dict[str, Any]] | None:
    path = episode_dir / "contact_records.pkl"
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


def _has_robot_object_contact(record: dict[str, Any] | None) -> bool | None:
    if record is None:
        return None
    contacts = record.get("contacts") or []
    return any(bool(contact.get("is_robot_object")) for contact in contacts)


def _nearest_contact_proxy(record: dict[str, Any] | None) -> float | None:
    if not record:
        return None
    finger_geoms = record.get("finger_geoms") or {}
    object_geoms = record.get("object_geoms") or {}
    robot_points = []
    for name, value in finger_geoms.items():
        if "finger" not in name or "pad" not in name:
            continue
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.shape[0] >= 3:
            robot_points.append(arr[:3])
    object_points = []
    for value in object_geoms.values():
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.shape[0] >= 3:
            object_points.append(arr[:3])
    if not robot_points or not object_points:
        return None
    robots = np.stack(robot_points)
    objects = np.stack(object_points)
    distances = np.linalg.norm(robots[:, None, :] - objects[None, :, :], axis=2)
    return float(np.min(distances, axis=1).sum())


def _trend(delta: float | None, eps: float) -> str:
    if delta is None:
        return "unknown"
    if delta < -eps:
        return "decreased"
    if delta > eps:
        return "increased"
    return "unchanged"


def _contact_phrase(before: bool | None, after: bool | None) -> str:
    if before is False and after is True:
        return "the gripper establishes contact with the object"
    if before is True and after is True:
        return "the gripper maintains contact with the object"
    if before is True and after is False:
        return "the gripper lost contact with the object"
    if before is False and after is False:
        return "the gripper still has not made contact with the object"
    return "contact with the object is not directly observed"


def _relation_to_object(distance: float | None, contact: bool | None) -> str:
    if contact is True:
        return "in contact with the object"
    if distance is None:
        return "at an unknown distance from the object"
    if distance < 0.07:
        return "very close to the object"
    if distance < 0.14:
        return "near the object"
    return "away from the object"


def _relation_to_goal(distance: float | None) -> str:
    if distance is None:
        return "in an unclear relation to the target"
    if distance < 0.04:
        return "at the target location"
    if distance < 0.09:
        return "near the target location"
    return "away from the target location"


def _lifted(obj: np.ndarray | None, initial_obj: np.ndarray | None) -> bool | None:
    if obj is None or initial_obj is None:
        return None
    return bool((obj[2] - initial_obj[2]) > 0.03)


def _lift_phrase(before: bool | None, after: bool | None) -> str:
    if before is None:
        if after is True:
            return "the object is lifted"
        if after is False:
            return "the object has not been lifted yet"
        return "the object lift state is unclear"
    if before is False and after is True:
        return "the object is lifted from its starting surface"
    if before is True and after is True:
        return "the object remains lifted"
    if before is True and after is False:
        return "the object was lowered"
    if before is False and after is False:
        return "the object has not been lifted yet"
    return "the object lift state is unclear"


def _movement_phrase(motion_norm: float | None, goal_trend: str) -> str:
    if motion_norm is None:
        return "the object motion is unclear"
    if motion_norm < 0.005:
        return "the object is nearly stationary"
    if goal_trend == "decreased":
        return "the object moves closer to the target"
    if goal_trend == "increased":
        return "the object drifts away from the target"
    return "the object moved, but not clearly toward the target"


def _progress_stage(progress: float) -> str:
    if progress < 20:
        return "early approach"
    if progress < 50:
        return "approach or grasping"
    if progress < 80:
        return "transport or placement"
    if progress < 99:
        return "near completion"
    return "complete"


def _infer_pick_place_phase(
    contact_after: bool | None,
    lifted_before: bool | None,
    lifted_after: bool | None,
    movement_event: str,
    goal_relation: str,
    goal_trend: str,
    approach_trend: str,
    progress: float,
    success_after: bool,
) -> dict[str, str]:
    if success_after or progress >= 99:
        return {
            "name": "place",
            "description": "the object is at the target and the task is complete",
        }
    if lifted_after is True and (
        (
            "moved toward the target" in movement_event
            or "closer to the target" in movement_event
        )
        or goal_relation in {"near the target location", "at the target location"}
        or progress >= 60
    ):
        return {
            "name": "transport",
            "description": (
                "the object has been grasped or lifted and is being moved toward "
                "the target"
            ),
        }
    if lifted_before is False and lifted_after is True:
        return {
            "name": "lift",
            "description": "the robot has lifted the object after grasping it",
        }
    if contact_after is True:
        return {
            "name": "contact/grasp",
            "description": "the gripper is controlling or trying to secure the object",
        }
    if approach_trend == "decreased" or progress > 0:
        return {
            "name": "approach",
            "description": "the robot is approaching the object before grasping it",
        }
    return {
        "name": "approach",
        "description": "the robot has not yet reached the object",
    }


def _task_effect(task_phase: dict[str, str], progress_delta: float) -> str:
    """Explain task progress without exposing the internal phase label."""
    phase = task_phase["name"]
    if phase == "place":
        return "placing the object at the target satisfies the main goal of the task"
    if phase == "transport":
        return (
            "carrying the secured object toward the target directly advances the "
            "placement objective"
        )
    if phase == "lift":
        return (
            "lifting the object off its starting surface shows that the robot has "
            "moved beyond simple contact and is preparing to carry it"
        )
    if phase == "contact/grasp":
        return (
            "making and maintaining contact is useful because the robot needs a "
            "stable grasp before it can lift or carry the object"
        )
    if phase == "approach" and progress_delta > 0.5:
        return (
            "moving the gripper closer is useful initial progress because the "
            "robot must reach the object before it can manipulate it"
        )
    return "the motion does not yet create a clear change toward completing the task"


def _remaining_subgoal(
    current_contact: bool | None,
    current_lifted: bool | None,
    current_goal_relation: str,
    success_after: bool,
) -> str:
    if success_after:
        return "no remaining subgoal; the task is complete"
    if current_contact is not True:
        return "reach the object and establish a stable grasp"
    if current_lifted is not True:
        return "lift the object while maintaining control"
    if current_goal_relation != "at the target location":
        return "carry the object to the target location and place it there"
    return "settle the object at the target location so the task is satisfied"


def _goal_reference_text(goal_relation: str) -> str:
    if goal_relation == "at the target location":
        return "at the intended target"
    if goal_relation == "near the target location":
        return "near the intended target"
    if goal_relation == "away from the target location":
        return "not yet at the intended target"
    return "with the target relation unclear"


def _previous_scene_text(
    object_relation: str,
    goal_relation: str,
    lifted: bool | None,
) -> str:
    lift_text = (
        "already raised from the surface"
        if lifted is True
        else "still on or near its starting surface"
        if lifted is False
        else "at an unclear height"
    )
    return (
        "At the previous timestep, the gripper was "
        f"{object_relation}. The object was {lift_text} and "
        f"{_goal_reference_text(goal_relation)}."
    )


def _manipulation_event_text(
    task_phase: dict[str, str],
    contact_event: str,
    movement_event: str,
    lift_event: str,
    success_after: bool,
) -> str:
    """Summarize the transition as a robot manipulation event."""
    phase = task_phase["name"]
    if success_after:
        return "the robot places the object at the target and completes the task"
    if phase == "transport":
        if "closer to the target" in movement_event:
            return "the robot carries the object closer to the target"
        return "the robot keeps hold of the object while repositioning it"
    if phase == "lift":
        return "the robot raises the object after securing it"
    if phase == "contact/grasp":
        if "establishes contact" in contact_event:
            return "the robot makes first useful contact with the object"
        return "the robot keeps working to secure the object before lifting"
    if phase == "approach":
        if "has not made contact" in contact_event:
            return "the robot approaches the object but has not started manipulating it"
        return "the robot is getting into position to manipulate the object"
    if "lifted" in lift_event:
        return "the robot changes the object's height"
    return movement_event


def _join_event_clauses(events: list[str]) -> str:
    if not events:
        return ""
    if len(events) == 1:
        return events[0]
    if len(events) == 2:
        return f"{events[0]} and {events[1]}"
    return ", ".join(events[:-1]) + f", and {events[-1]}"


def _current_change_text(
    manipulation_event: str,
    contact_event: str,
    movement_event: str,
    lift_event: str,
) -> str:
    details = []
    if "contact is not directly observed" not in contact_event:
        details.append(contact_event)
    if "unclear" not in lift_event and "not been lifted" not in lift_event:
        details.append(lift_event)
    if "unclear" not in movement_event and "stationary" not in movement_event:
        details.append(movement_event)
    if details:
        return (
            "At the current timestep, "
            f"{manipulation_event}. The transition also shows "
            f"{_join_event_clauses(details)}."
        )
    return f"At the current timestep, {manipulation_event}."


def _progress_interpretation_text(
    current_progress: int,
    previous_progress: int,
    task_effect: str,
) -> str:
    if current_progress <= previous_progress:
        return (
            "This suggests little visible advancement beyond the previous step, "
            f"so I would keep the progress estimate near {current_progress}%."
        )
    return (
        f"This change helps the task because {task_effect}. Given the previous "
        f"task progress of {previous_progress}%, I would predict the current "
        f"task progress to be about {current_progress}%."
    )


def _semantic_event_payload(
    contact_event: str,
    approach_trend: str,
    movement_event: str,
    lift_event: str,
    goal_trend: str,
    manipulation_event: str,
    success_before: bool,
    success_after: bool,
    task_effect: str,
    remaining: str,
) -> dict[str, str]:
    if approach_trend == "decreased":
        gripper_motion = (
            "the gripper moved into a better position for manipulating the object"
        )
    elif approach_trend == "increased":
        gripper_motion = "the gripper moved farther from the object"
    else:
        gripper_motion = (
            "the gripper position did not clearly improve relative to the object"
        )

    if goal_trend == "decreased":
        goal_motion = "the object moved closer to the target location"
    elif goal_trend == "increased":
        goal_motion = "the object moved farther from the target location"
    else:
        goal_motion = "the object did not make a clear move toward the target location"

    if not success_before and success_after:
        success_event = "the task became successful at the current sampled timestep"
    elif success_after:
        success_event = "the task is already successful"
    else:
        success_event = "the task is not complete yet"

    return {
        "contact_event": contact_event,
        "gripper_motion": gripper_motion,
        "manipulation_event": manipulation_event,
        "object_motion": movement_event,
        "object_lift": lift_event,
        "goal_motion": goal_motion,
        "task_effect": task_effect,
        "remaining_subgoal": remaining,
        "success": success_event,
    }


def extract_state_transition(
    semantic_states: list[dict[str, Any]],
    progress: np.ndarray,
    timestep: int,
    previous_timestep: int,
    contact_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convert simulator state changes into semantic event facts."""
    prev_t = max(0, min(previous_timestep, len(semantic_states) - 1))
    cur_t = max(0, min(timestep, len(semantic_states) - 1))
    prev_state = semantic_states[prev_t]
    cur_state = semantic_states[cur_t]
    first_state = semantic_states[0]

    prev_contact = (
        contact_records[prev_t]
        if contact_records and prev_t < len(contact_records)
        else None
    )
    cur_contact = (
        contact_records[cur_t]
        if contact_records and cur_t < len(contact_records)
        else None
    )

    prev_eef = _as_vec(prev_state, "eef_pos")
    cur_eef = _as_vec(cur_state, "eef_pos")
    prev_obj = _as_vec(prev_state, "object_pos")
    cur_obj = _as_vec(cur_state, "object_pos")
    prev_goal = _as_vec(prev_state, "goal_pos")
    cur_goal = _as_vec(cur_state, "goal_pos")
    initial_obj = _as_vec(first_state, "object_pos")

    prev_contact_bool = _has_robot_object_contact(prev_contact)
    cur_contact_bool = _has_robot_object_contact(cur_contact)
    prev_contact_proxy = _nearest_contact_proxy(prev_contact)
    cur_contact_proxy = _nearest_contact_proxy(cur_contact)
    prev_dist = prev_contact_proxy or _distance(prev_eef, prev_obj)
    cur_dist = cur_contact_proxy or _distance(cur_eef, cur_obj)
    prev_goal_dist = _distance(prev_obj, prev_goal)
    cur_goal_dist = _distance(cur_obj, cur_goal)
    goal_trend = _trend(
        (
            None
            if prev_goal_dist is None or cur_goal_dist is None
            else cur_goal_dist - prev_goal_dist
        ),
        eps=0.005,
    )
    approach_trend = _trend(
        None if prev_dist is None or cur_dist is None else cur_dist - prev_dist,
        eps=0.005,
    )
    prev_lifted = _lifted(prev_obj, initial_obj)
    cur_lifted = _lifted(cur_obj, initial_obj)
    object_motion = _norm_delta(cur_obj, prev_obj)
    prev_progress = float(progress[prev_t])
    cur_progress = float(progress[cur_t])
    success_before = bool(prev_state.get("success", False))
    success_after = bool(cur_state.get("success", False))

    contact_event = _contact_phrase(prev_contact_bool, cur_contact_bool)
    lift_event = _lift_phrase(prev_lifted, cur_lifted)
    movement_event = _movement_phrase(object_motion, goal_trend)
    current_goal_relation = _relation_to_goal(cur_goal_dist)
    task_phase = _infer_pick_place_phase(
        contact_after=cur_contact_bool,
        lifted_before=prev_lifted,
        lifted_after=cur_lifted,
        movement_event=movement_event,
        goal_relation=current_goal_relation,
        goal_trend=goal_trend,
        approach_trend=approach_trend,
        progress=cur_progress,
        success_after=success_after,
    )
    task_effect = _task_effect(
        task_phase=task_phase,
        progress_delta=cur_progress - prev_progress,
    )
    remaining = _remaining_subgoal(
        current_contact=cur_contact_bool,
        current_lifted=cur_lifted,
        current_goal_relation=current_goal_relation,
        success_after=success_after,
    )

    previous_state_text = _previous_scene_text(
        object_relation=_relation_to_object(prev_dist, prev_contact_bool),
        goal_relation=_relation_to_goal(prev_goal_dist),
        lifted=prev_lifted,
    )
    manipulation_event = _manipulation_event_text(
        task_phase=task_phase,
        contact_event=contact_event,
        movement_event=movement_event,
        lift_event=lift_event,
        success_after=success_after,
    )
    current_state_text = _current_change_text(
        manipulation_event=manipulation_event,
        contact_event=contact_event,
        movement_event=movement_event,
        lift_event=lift_event,
    )
    semantic_events = _semantic_event_payload(
        contact_event=contact_event,
        approach_trend=approach_trend,
        movement_event=movement_event,
        lift_event=lift_event,
        goal_trend=goal_trend,
        manipulation_event=manipulation_event,
        success_before=success_before,
        success_after=success_after,
        task_effect=task_effect,
        remaining=remaining,
    )

    return {
        "timesteps": {
            "first": 0,
            "previous": prev_t,
            "current": cur_t,
            "previous_source": "sampled_sequence",
        },
        "previous_state": previous_state_text,
        "current_state": current_state_text,
        "task_phase": task_phase,
        "semantic_events": semantic_events,
        "task_effect": task_effect,
        "remaining_subgoal": remaining,
        "progress": {
            "first": int(round(float(progress[0]))),
            "previous": int(round(prev_progress)),
            "current": int(round(cur_progress)),
            "stage": _progress_stage(cur_progress),
        },
        "task_state": {
            "object": "the target object",
            "goal": "the target location",
            "success_before": success_before,
            "success_after": success_after,
        },
    }


def generate_template_reasoning(
    task_description: str, transition: dict[str, Any], progress: float
) -> str:
    """Generate SOLE-style reasoning from semantic events, not raw state labels."""
    previous_progress = transition["progress"]["previous"]
    current_progress = int(round(float(progress)))
    remaining = transition["remaining_subgoal"]
    if remaining.startswith("no remaining subgoal"):
        remaining_sentence = "No further subgoal remains because the task is complete."
    else:
        remaining_sentence = (
            f"To complete the task, the robot still needs to {remaining}."
        )
    interpretation = _progress_interpretation_text(
        previous_progress=previous_progress,
        current_progress=current_progress,
        task_effect=transition["task_effect"],
    )
    return " ".join(
        [
            transition["previous_state"],
            transition["current_state"],
            interpretation,
            remaining_sentence,
        ]
    )


def build_problem_prompt(
    task_description: str, first_progress: float, previous_progress: float
) -> str:
    return (
        "Here is an image containing multiple camera views of a robot attempting "
        "to complete a task. The views on the top are from an external camera. "
        "The views on the bottom are from the robot's wrist camera. The views "
        "from the very first timestep are shown to the left. The views from the "
        "previous timestep are shown in the middle. The views from the current "
        "timestep are shown to the right. The task description is: "
        f"{task_description}. The task progress for the very first timestep is "
        f"{int(round(first_progress))}%. The task progress for the previous "
        f"timestep is {int(round(previous_progress))}%. Predict the task "
        "progress for the current timestep."
    )


def build_prompt_json(problem: str, completion: str | None = None) -> str:
    prompt: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": problem},
            ],
        },
    ]
    if completion is not None:
        prompt.append({"role": "assistant", "content": completion})
    return json.dumps(prompt, ensure_ascii=False)


def build_qwen_instruction(sample: CotSample) -> str:
    qwen_facts = {
        "timesteps": sample.transition["timesteps"],
        "previous_state": sample.transition["previous_state"],
        "current_state": sample.transition["current_state"],
        "semantic_events": sample.transition["semantic_events"],
        "progress": sample.transition["progress"],
        "task_state": sample.transition["task_state"],
    }
    transition_text = json.dumps(qwen_facts, ensure_ascii=False, indent=2)
    return (
        "Rewrite the template into fluent SOLE-R1 style robot reasoning. Use the "
        "image only as visual context and the semantic events as ground truth. "
        "Keep the facts from the template, but make the wording sound like a "
        "robotics expert explaining what changed between the previous sampled "
        "timestep and the current sampled timestep. Describe actions and task "
        "progress naturally rather than as state labels. Do not invent new objects, "
        "do not mention simulator variables, geometry names, coordinates, numeric "
        "distances, or internal phase labels such as approach, grasp, lift, transport, "
        "or place, and do not change the progress. Do not infer success unless it is "
        "stated in the semantic events. Return only "
        "<think>...</think>. Do not include <answer>.\n\n"
        f"Task description:\n{sample.task_description}\n\n"
        f"Ground-truth current progress: {int(round(sample.progress))}%\n"
        f"Previous sampled progress: {int(round(sample.previous_progress))}%\n\n"
        f"Semantic events:\n{transition_text}\n\n"
        f"Template reasoning:\n{sample.template_reasoning}"
    )


def _strip_think_tags(text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<answer>.*?</answer>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


class BaseParaphraser:
    def generate(self, samples: list[CotSample]) -> list[str]:
        raise NotImplementedError


class TemplateParaphraser(BaseParaphraser):
    def generate(self, samples: list[CotSample]) -> list[str]:
        return [sample.template_reasoning for sample in samples]


class QwenVlParaphraser(BaseParaphraser):
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 256,
    ) -> None:
        import torch
        from PIL import Image
        from transformers import AutoProcessor

        self.Image = Image
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
        model_cls = self._resolve_model_class()
        dtype = self._resolve_dtype(torch_dtype)
        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": True,
            "device_map": device,
        }
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        self.model = model_cls.from_pretrained(model_path, **kwargs)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if (
            tokenizer is not None
            and getattr(tokenizer, "pad_token_id", None) is not None
        ):
            self.model.generation_config.pad_token_id = tokenizer.pad_token_id
        self.model.eval()

    @staticmethod
    def _resolve_model_class() -> Any:
        try:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq

            return AutoModelForVision2Seq

    def _resolve_dtype(self, torch_dtype: str) -> Any:
        if torch_dtype == "auto":
            return None
        if torch_dtype == "bfloat16":
            return self.torch.bfloat16
        if torch_dtype == "float16":
            return self.torch.float16
        if torch_dtype == "float32":
            return self.torch.float32
        raise ValueError(f"unsupported torch dtype: {torch_dtype}")

    def _chat_prompt(self, sample: CotSample) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(sample.image_path)},
                    {"type": "text", "text": build_qwen_instruction(sample)},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, samples: list[CotSample]) -> list[str]:
        images = [
            self.Image.open(sample.image_path).convert("RGB") for sample in samples
        ]
        prompts = [self._chat_prompt(sample) for sample in samples]
        inputs = self.processor(
            text=prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        input_len = inputs["input_ids"].shape[1]
        generated = generated[:, input_len:]
        texts = self.processor.batch_decode(generated, skip_special_tokens=True)
        return [_strip_think_tags(text) for text in texts]


def _image_files(view_dir: Path) -> list[Path]:
    files = sorted(view_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"missing image frames: {view_dir}")
    return files


def _load_rgb(path: Path) -> PILImage.Image:
    return PILImage.open(path).convert("RGB")


def _resize_like(image: PILImage.Image, ref: PILImage.Image) -> PILImage.Image:
    if image.size == ref.size:
        return image
    return image.resize(ref.size, PILImage.BILINEAR)


def _make_grid(paths: list[Path]) -> PILImage.Image:
    images = [_load_rgb(path) for path in paths]
    ref = images[0]
    images = [_resize_like(img, ref) for img in images]
    grid = PILImage.new("RGB", (ref.width * 3, ref.height * 2))
    for idx, img in enumerate(images[:3]):
        grid.paste(img, (idx * ref.width, 0))
    for idx, img in enumerate(images[3:]):
        grid.paste(img, (idx * ref.width, ref.height))
    return grid


def build_sampled_sole_image(
    episode_dir: Path,
    timestep: int,
    previous_timestep: int,
    sampling_name: str,
    overwrite: bool,
) -> Path:
    main_files = _image_files(episode_dir / "images" / "main")
    wrist_files = _image_files(episode_dir / "images" / "wrist")
    total = min(len(main_files), len(wrist_files))
    if timestep >= total or previous_timestep >= total:
        raise IndexError("sampled timestep exceeds image sequence length")
    out_dir = episode_dir / "sole_images_sampled" / sampling_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"t{timestep:06d}_prev{previous_timestep:06d}.png"
    if out_path.exists() and not overwrite:
        return out_path.resolve()
    grid = _make_grid(
        [
            main_files[0],
            main_files[previous_timestep],
            main_files[timestep],
            wrist_files[0],
            wrist_files[previous_timestep],
            wrist_files[timestep],
        ]
    )
    grid.save(out_path)
    return out_path.resolve()


def build_samples_for_episode(
    episode_dir: Path,
    sampling: SamplingConfig,
    progress_name: str,
    overwrite_images: bool,
) -> list[CotSample]:
    semantic_states = _load_semantic_states(episode_dir / "semantic_states.pkl")
    contact_records = _load_contact_records(episode_dir)
    progress = _load_progress(episode_dir, progress_name)
    image_total = min(
        len(_image_files(episode_dir / "images" / "main")),
        len(_image_files(episode_dir / "images" / "wrist")),
    )
    total_steps = min(len(semantic_states), len(progress), image_total)
    success_flags = _success_flags_from_states(semantic_states[:total_steps])
    first_success = _first_success_timestep(success_flags)
    timesteps = sample_timesteps(
        total_steps,
        sampling,
        semantic_states=semantic_states[:total_steps],
        contact_records=contact_records[:total_steps] if contact_records else None,
    )
    timesteps = _truncate_timesteps_at_success(timesteps, success_flags)
    sampling_name = _sampling_name(sampling)
    episode_id = episode_id_from_dir(episode_dir)
    task_description = task_description_from_episode(episode_dir)
    success = success_from_episode(episode_dir)

    samples = []
    sampled_metadata = []
    for sample_idx, timestep in enumerate(timesteps):
        previous_timestep = timesteps[sample_idx - 1] if sample_idx > 0 else 0
        if sample_idx > 0 and success_flags[previous_timestep]:
            continue
        if first_success is not None and timestep > first_success:
            continue
        image_path = build_sampled_sole_image(
            episode_dir=episode_dir,
            timestep=timestep,
            previous_timestep=previous_timestep,
            sampling_name=sampling_name,
            overwrite=overwrite_images,
        )
        transition = extract_state_transition(
            semantic_states=semantic_states,
            progress=progress,
            timestep=timestep,
            previous_timestep=previous_timestep,
            contact_records=contact_records,
        )
        template = generate_template_reasoning(
            task_description=task_description,
            transition=transition,
            progress=float(progress[timestep]),
        )
        sample = CotSample(
            episode_dir=episode_dir,
            episode_id=episode_id,
            timestep=timestep,
            previous_timestep=previous_timestep,
            total_steps=total_steps,
            image_path=image_path,
            task_description=task_description,
            progress=float(progress[timestep]),
            previous_progress=float(progress[previous_timestep]),
            success=success,
            transition=transition,
            template_reasoning=template,
        )
        samples.append(sample)
        sampled_metadata.append(
            {
                "episode_id": episode_id,
                "sample_index": sample_idx,
                "timestep": timestep,
                "first_frame": 0,
                "previous_frame": previous_timestep,
                "current_frame": timestep,
                "previous_source": "sampled_sequence",
                "progress": float(progress[timestep]),
                "image_path": str(image_path),
            }
        )
    _write_json(
        episode_dir / "sole_image_sampled_metadata.json",
        {
            "sampling": sampling.__dict__,
            "first_success_timestep": first_success,
            "samples": sampled_metadata,
        },
    )
    return samples


def _progress_answer(progress: float) -> str:
    rounded = int(round(float(progress)))
    return f"{max(0, min(100, rounded))}%"


def build_dataset_row(
    sample: CotSample, reasoning: str, include_debug: bool
) -> dict[str, Any]:
    answer = _progress_answer(sample.progress)
    problem = build_problem_prompt(
        task_description=sample.task_description,
        first_progress=sample.transition["progress"]["first"],
        previous_progress=sample.previous_progress,
    )
    completion = f"<think>{reasoning.strip()}</think><answer>{answer}</answer>"
    row = {
        "image": str(sample.image_path),
        "problem": problem,
        "solution_orig": answer.rstrip("%"),
        "solution": completion,
        "prompt": build_prompt_json(problem, completion=completion),
        "image_name": f"{sample.episode_id}_timestep_{sample.timestep}.png",
        "from_annotation": 0,
        "completion": completion,
        "data_source": "libero_spatial_rlinf_sole_cot_v4",
        "progress": float(sample.progress),
        "metadata": {
            "episode_id": sample.episode_id,
            "episode_dir": str(sample.episode_dir),
            "image_path": str(sample.image_path),
            "timestep": sample.timestep,
            "previous_timestep": sample.previous_timestep,
            "previous_source": "sampled_sequence",
            "total_steps": sample.total_steps,
            "success": sample.success,
            "task_phase": sample.transition["task_phase"],
        },
    }
    if include_debug:
        row["metadata"]["transition"] = sample.transition
        row["metadata"]["template_reasoning"] = sample.template_reasoning
    return row


def _chunks(items: list[CotSample], size: int) -> Iterable[list[CotSample]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _sample_key(sample: CotSample) -> tuple[str, int]:
    return sample.episode_id, sample.timestep


def _row_key(row: dict[str, Any]) -> tuple[str, int] | None:
    metadata = row.get("metadata") or {}
    episode_id = metadata.get("episode_id")
    timestep = metadata.get("timestep")
    if episode_id is None or timestep is None:
        return None
    return str(episode_id), int(timestep)


def write_debug_package(path: Path, debug_items: list[DebugItem], limit: int) -> None:
    if limit > 0:
        debug_items = debug_items[:limit]
    tmp_dir = path.with_suffix("")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    for idx, item in enumerate(debug_items):
        sample_dir = tmp_dir / f"sample_{idx:03d}"
        sample_dir.mkdir(parents=True)
        shutil.copy2(item.sample.image_path, sample_dir / "sole_image.png")
        task = load_json(item.sample.episode_dir / "task.json", default={}) or {}
        _write_json(sample_dir / "task.json", task)
        _write_json(
            sample_dir / "progress.json",
            {
                "first": item.sample.transition["progress"]["first"],
                "previous": int(round(item.sample.previous_progress)),
                "current": int(round(item.sample.progress)),
                "raw_current": item.sample.progress,
            },
        )
        _write_json(
            sample_dir / "semantic_events.json",
            {
                "timesteps": item.sample.transition["timesteps"],
                "previous_state": item.sample.transition["previous_state"],
                "current_state": item.sample.transition["current_state"],
                "semantic_events": item.sample.transition["semantic_events"],
                "progress": item.sample.transition["progress"],
                "task_state": item.sample.transition["task_state"],
            },
        )
        (sample_dir / "template_reasoning.txt").write_text(
            item.sample.template_reasoning, encoding="utf-8"
        )
        (sample_dir / "qwen_reasoning.txt").write_text(
            item.qwen_reasoning, encoding="utf-8"
        )
        (sample_dir / "final_completion.txt").write_text(
            item.row["completion"], encoding="utf-8"
        )
        _write_json(sample_dir / "dataset_row.json", item.row)
    readme = (
        "Debug package for RLinf LIBERO Spatial SOLE-R1 CoT v4.\n"
        "Each sample uses [first frame | previous sampled frame | current frame] "
        "for main and wrist cameras. semantic_events.json contains natural task "
        "events only; numeric simulator distances, geom names, coordinates, and "
        "internal phase labels are intentionally excluded from CoT inputs.\n"
    )
    (tmp_dir / "README.txt").write_text(readme, encoding="utf-8")
    if path.exists():
        path.unlink()
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zf:
        for file_path in sorted(tmp_dir.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(tmp_dir))
    shutil.rmtree(tmp_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SOLE-R1 style CoT SFT data from LIBERO rollout samples."
    )
    parser.add_argument("path", help="Episode directory or trajectory root.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Defaults to <path>/sole_cot_dataset.jsonl.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing JSONL and skip already generated episode/timestep rows.",
    )
    parser.add_argument(
        "--progress-json",
        default=None,
        help="Optional path for a small JSON progress file updated after each batch.",
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=[
            "hybrid",
            "uniform",
            "interval",
            "event_aware",
            "event-aware",
            "event",
            "num_samples",
        ],
        default="hybrid",
    )
    parser.add_argument("--interval", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument(
        "--max-event-samples", type=int, default=DEFAULT_MAX_EVENT_SAMPLES
    )
    parser.add_argument(
        "--max-sample-gap",
        type=int,
        default=None,
        help="Densify sampled timesteps so adjacent samples are at most this far apart; omit or set <=0 to disable.",
    )
    parser.add_argument("--no-include-last", action="store_true")
    parser.add_argument(
        "--progress-name",
        default="progress_contact.npy",
        help="Preferred progress file name; falls back to progress.npy.",
    )
    parser.add_argument(
        "--paraphraser",
        choices=["template", "qwen"],
        default="template",
        help="Use template-only output or Qwen3-VL paraphrasing.",
    )
    parser.add_argument("--model-path", default=DEFAULT_QWEN_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--include-debug",
        action="store_true",
        help="Store transition facts and template reasoning in metadata.",
    )
    parser.add_argument(
        "--write-debug-files",
        action="store_true",
        help="Also write transition/template debug JSONL next to the output.",
    )
    parser.add_argument(
        "--hf-dataset-dir",
        default=None,
        help="Optional HuggingFace Dataset save_to_disk output directory.",
    )
    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Overwrite sampled SOLE images if they already exist.",
    )
    parser.add_argument(
        "--debug-package",
        default=None,
        help="Optional ZIP path containing per-sample debug artifacts.",
    )
    parser.add_argument("--debug-limit", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.path).expanduser().resolve()
    episode_dirs = find_episode_dirs(root)
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories found under {args.path}")

    sampling = SamplingConfig(
        strategy=args.sampling_strategy,
        interval=args.interval,
        num_samples=args.num_samples,
        max_event_samples=args.max_event_samples,
        include_last=not args.no_include_last,
        max_sample_gap=args.max_sample_gap,
    )
    samples: list[CotSample] = []
    for episode_dir in episode_dirs:
        episode_samples = build_samples_for_episode(
            episode_dir=episode_dir,
            sampling=sampling,
            progress_name=args.progress_name,
            overwrite_images=args.overwrite_images,
        )
        samples.extend(episode_samples)
        print(f"sampled {len(episode_samples)} timesteps: {episode_dir}")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else root / "sole_cot_dataset.jsonl"
    )
    progress_path = (
        Path(args.progress_json).expanduser().resolve()
        if args.progress_json
        else output.with_suffix(".progress.json")
    )
    existing_rows = _read_jsonl(output) if args.resume else []
    existing_keys = {key for row in existing_rows if (key := _row_key(row)) is not None}
    if output.exists() and not args.resume:
        output.unlink()
    if progress_path.exists() and not args.resume:
        progress_path.unlink()
    if args.write_debug_files and output.with_suffix(".debug.jsonl").exists() and not args.resume:
        output.with_suffix(".debug.jsonl").unlink()
    if existing_keys:
        before = len(samples)
        samples = [sample for sample in samples if _sample_key(sample) not in existing_keys]
        print(
            f"resume: skipped {before - len(samples)} existing samples; "
            f"remaining {len(samples)}",
            flush=True,
        )

    debug_rows = []
    debug_items: list[DebugItem] = []
    processed = len(existing_rows)
    total_samples = processed + len(samples)
    start_time = time.monotonic()
    _write_progress_json(progress_path, "sampled", processed, total_samples, start_time)
    print(
        "progress " + _format_progress_bar(processed, total_samples) + " status=sampled",
        flush=True,
    )
    if samples:
        paraphraser: BaseParaphraser
        print(
            f"initializing paraphraser={args.paraphraser} samples={len(samples)} "
            f"batch_size={args.batch_size} output={output}",
            flush=True,
        )
        if args.paraphraser == "qwen":
            paraphraser = QwenVlParaphraser(
                model_path=args.model_path,
                device=args.device,
                torch_dtype=args.torch_dtype,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            paraphraser = TemplateParaphraser()
        print("paraphraser initialized", flush=True)
    else:
        paraphraser = TemplateParaphraser()
        print("no remaining samples; skip paraphraser initialization", flush=True)

    for batch_idx, batch in enumerate(_chunks(samples, args.batch_size), start=1):
        first = batch[0]
        message = (
            f"batch={batch_idx} size={len(batch)} first_episode={first.episode_id} "
            f"first_timestep={first.timestep}"
        )
        print("generating " + message, flush=True)
        _write_progress_json(
            progress_path,
            "generating",
            processed,
            total_samples,
            start_time,
            batch_idx=batch_idx,
            message=message,
        )
        try:
            reasonings = paraphraser.generate(batch)
        except Exception as exc:
            _write_progress_json(
                progress_path,
                "failed",
                processed,
                total_samples,
                start_time,
                batch_idx=batch_idx,
                message=f"{type(exc).__name__}: {exc}",
            )
            raise
        batch_rows = []
        for sample, reasoning in zip(batch, reasonings):
            reasoning = _strip_think_tags(reasoning) or sample.template_reasoning
            row = build_dataset_row(
                sample=sample,
                reasoning=reasoning,
                include_debug=args.include_debug,
            )
            batch_rows.append(row)
            if args.debug_package:
                debug_items.append(
                    DebugItem(sample=sample, qwen_reasoning=reasoning, row=row)
                )
            if args.write_debug_files:
                debug_rows.append(
                    {
                        "episode_id": sample.episode_id,
                        "timestep": sample.timestep,
                        "previous_timestep": sample.previous_timestep,
                        "image": str(sample.image_path),
                        "progress": sample.progress,
                        "transition": sample.transition,
                        "template_reasoning": sample.template_reasoning,
                        "reasoning": reasoning,
                    }
                )
        _append_jsonl(output, batch_rows)
        if args.write_debug_files:
            _append_jsonl(output.with_suffix(".debug.jsonl"), debug_rows)
            debug_rows.clear()
        processed += len(batch_rows)
        _write_progress_json(
            progress_path,
            "running",
            processed,
            total_samples,
            start_time,
            batch_idx=batch_idx,
            message=f"wrote batch_rows={len(batch_rows)}",
        )
        print(
            "progress "
            + _format_progress_bar(processed, total_samples)
            + f" status=running wrote_batch_rows={len(batch_rows)}",
            flush=True,
        )

    rows = _read_jsonl(output)
    _write_progress_json(progress_path, "complete", len(rows), total_samples, start_time)
    print(
        "progress " + _format_progress_bar(len(rows), total_samples) + " status=complete",
        flush=True,
    )
    if args.hf_dataset_dir:
        from datasets import Dataset, Image

        hf_rows = []
        for row in rows:
            hf_rows.append({key: row[key] for key in OFFICIAL_COLUMNS})
        dataset = Dataset.from_list(hf_rows).cast_column("image", Image())
        dataset.save_to_disk(str(Path(args.hf_dataset_dir).expanduser().resolve()))
    if args.debug_package:
        write_debug_package(
            Path(args.debug_package).expanduser().resolve(),
            debug_items,
            args.debug_limit,
        )
    print(
        f"wrote CoT dataset samples={len(rows)} episodes={len(episode_dirs)} "
        f"paraphraser={args.paraphraser}: {output}",
        flush=True,
    )
    if args.hf_dataset_dir:
        print(f"wrote HF dataset: {Path(args.hf_dataset_dir).expanduser().resolve()}")
    if args.debug_package:
        print(f"wrote debug package: {Path(args.debug_package).expanduser().resolve()}")


if __name__ == "__main__":
    main()
