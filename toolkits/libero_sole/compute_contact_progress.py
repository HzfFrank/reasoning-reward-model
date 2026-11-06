from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from toolkits.libero_sole.common import find_episode_dirs, write_json
from toolkits.libero_sole.compute_progress import (
    _as_vec,
    _load_semantic_states,
    compute_pick_place_progress,
)


def _load_contact_records(path: Path) -> list[dict[str, Any]]:
    with open(path, "rb") as f:
        records = pickle.load(f)
    normalized = []
    for record in records:
        if isinstance(record, list):
            if len(record) != 1:
                raise ValueError(
                    "contact_records.pkl contains batched entries; expected per-env records"
                )
            record = record[0]
        if not isinstance(record, dict):
            raise ValueError("contact_records.pkl entries must be dictionaries")
        normalized.append(record)
    return normalized


def _point_dict_to_array(points: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    names = []
    values = []
    for name, value in points.items():
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.shape[0] < 3:
            continue
        names.append(str(name))
        values.append(arr[:3])
    if not values:
        return [], np.empty((0, 3), dtype=np.float64)
    return names, np.stack(values)


def _select_robot_points(record: dict[str, Any]) -> tuple[list[str], np.ndarray, str]:
    finger_geoms = record.get("finger_geoms") or {}
    pad_points = {
        name: value
        for name, value in finger_geoms.items()
        if "finger" in name and "pad" in name
    }
    names, points = _point_dict_to_array(pad_points)
    if len(names):
        return names, points, "gripper finger pad collision geoms"

    finger_points = {
        name: value for name, value in finger_geoms.items() if "finger" in name
    }
    names, points = _point_dict_to_array(finger_points)
    if len(names):
        return names, points, "gripper finger collision geoms"

    names, points = _point_dict_to_array(finger_geoms)
    if len(names):
        return names, points, "all saved gripper contact geoms"

    names, points = _point_dict_to_array(record.get("finger_sites") or {})
    if len(names):
        return names, points, "saved gripper sites"

    raise ValueError("contact record has no gripper/finger geom or site positions")


def _select_object_points(
    record: dict[str, Any], semantic_state: dict[str, Any]
) -> tuple[list[str], np.ndarray, str]:
    names, points = _point_dict_to_array(record.get("object_geoms") or {})
    if len(names):
        return names, points, "target object collision geom centers"

    names, points = _point_dict_to_array(record.get("object_sites") or {})
    if len(names):
        return names, points, "target object sites"

    object_pos = _as_vec(semantic_state, "object_pos")
    return (
        [str(semantic_state.get("object_name") or "object_center")],
        object_pos[None],
        "object center fallback",
    )


def _nearest_sum(
    robot_points: np.ndarray, object_points: np.ndarray
) -> tuple[float, list[int], list[float]]:
    distances = np.linalg.norm(robot_points[:, None, :] - object_points[None, :, :], axis=2)
    nearest_indices = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(distances.shape[0]), nearest_indices]
    return float(np.sum(nearest_distances)), nearest_indices.tolist(), nearest_distances.tolist()


def _normalize_progress(y: np.ndarray, monotonic: bool) -> np.ndarray:
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    denom = y_max - y_min
    if denom <= 1e-9:
        progress = np.zeros_like(y, dtype=np.float32)
    else:
        progress = ((-y + y_max) / denom * 100.0).astype(np.float32)
    if monotonic:
        progress = np.maximum.accumulate(progress)
    return progress.astype(np.float32)


def compute_contact_progress(
    semantic_states: list[dict[str, Any]],
    contact_records: list[dict[str, Any]],
    beta: float,
    monotonic: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    total = min(len(semantic_states), len(contact_records))
    if total == 0:
        raise ValueError("empty semantic/contact sequence")
    semantic_states = semantic_states[:total]
    contact_records = contact_records[:total]

    y_contact = []
    nearest_examples = []
    robot_rule = None
    object_rule = None
    robot_point_count = None
    object_point_count = None
    for t, (semantic_state, record) in enumerate(zip(semantic_states, contact_records)):
        robot_names, robot_points, robot_rule = _select_robot_points(record)
        object_names, object_points, object_rule = _select_object_points(record, semantic_state)
        value, nearest_indices, nearest_distances = _nearest_sum(robot_points, object_points)
        y_contact.append(value)
        robot_point_count = len(robot_names)
        object_point_count = len(object_names)
        if t in (0, total - 1):
            nearest_examples.append(
                {
                    "timestep": t,
                    "robot_points": robot_names,
                    "matched_object_points": [object_names[i] for i in nearest_indices],
                    "nearest_distances": nearest_distances,
                }
            )

    obj = np.stack([_as_vec(s, "object_pos") for s in semantic_states])
    goal = np.stack([_as_vec(s, "goal_pos") for s in semantic_states])
    y_goal = np.linalg.norm(obj - goal, axis=1)
    y_contact = np.asarray(y_contact, dtype=np.float64)
    y = (1.0 - beta) * y_contact + beta * y_goal
    progress = _normalize_progress(y, monotonic=monotonic)
    if semantic_states[-1].get("success"):
        progress[-1] = max(float(progress[-1]), 100.0)

    robot_object_contact_counts = []
    for record in contact_records:
        contacts = record.get("contacts") or []
        robot_object_contact_counts.append(
            sum(1 for contact in contacts if contact.get("is_robot_object"))
        )

    diagnostics = {
        "formula": (
            "progress_contact = (-y + max(y)) / (max(y) - min(y)) * 100; "
            "y=(1-beta)*y_contact + beta*y_goal; "
            "y_contact=sum over gripper points of nearest target-object point distance"
        ),
        "beta": beta,
        "monotonic": monotonic,
        "pair_selection_rule": (
            "Use gripper finger pad collision geom centers when available; "
            "for each robot point choose the nearest target object collision geom center; "
            "sum these nearest distances. Fall back to finger geoms, gripper sites, "
            "object sites, then object center only if needed."
        ),
        "robot_point_source": robot_rule,
        "object_point_source": object_rule,
        "robot_point_count": robot_point_count,
        "object_point_count": object_point_count,
        "object_name": semantic_states[0].get("object_name"),
        "goal_name": semantic_states[0].get("goal_name"),
        "y_contact_first": float(y_contact[0]),
        "y_contact_last": float(y_contact[-1]),
        "y_goal_first": float(y_goal[0]),
        "y_goal_last": float(y_goal[-1]),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "progress_first": float(progress[0]),
        "progress_last": float(progress[-1]),
        "robot_object_contact_count_max": (
            int(max(robot_object_contact_counts)) if robot_object_contact_counts else 0
        ),
        "robot_object_contact_count_sum": int(sum(robot_object_contact_counts)),
        "nearest_pair_examples": nearest_examples,
    }
    return progress.astype(np.float32), diagnostics


def process_episode(episode_dir: Path, beta: float, monotonic: bool, overwrite: bool) -> bool:
    semantic_path = episode_dir / "semantic_states.pkl"
    contact_path = episode_dir / "contact_records.pkl"
    center_path = episode_dir / "progress_center.npy"
    contact_progress_path = episode_dir / "progress_contact.npy"
    metadata_path = episode_dir / "progress_contact_metadata.json"
    center_metadata_path = episode_dir / "progress_center_metadata.json"

    if contact_progress_path.exists() and center_path.exists() and not overwrite:
        print(f"skip existing contact progress: {episode_dir}")
        return False
    if not semantic_path.is_file():
        print(f"skip missing semantic_states.pkl: {episode_dir}")
        return False
    if not contact_path.is_file():
        print(f"skip missing contact_records.pkl: {episode_dir}")
        return False

    try:
        semantic_states = _load_semantic_states(semantic_path)
        contact_records = _load_contact_records(contact_path)
        center_progress, center_diagnostics = compute_pick_place_progress(
            semantic_states=semantic_states, beta=beta, monotonic=monotonic
        )
        contact_progress, contact_diagnostics = compute_contact_progress(
            semantic_states=semantic_states,
            contact_records=contact_records,
            beta=beta,
            monotonic=monotonic,
        )
    except ValueError as exc:
        reason = {"supported": False, "reason": str(exc)}
        write_json(metadata_path, reason)
        print(f"skip unsupported contact progress: {episode_dir}: {exc}")
        return False

    np.save(center_path, center_progress)
    np.save(contact_progress_path, contact_progress)
    center_diagnostics["supported"] = True
    contact_diagnostics["supported"] = True
    write_json(center_metadata_path, center_diagnostics)
    write_json(metadata_path, contact_diagnostics)
    print(
        f"wrote {contact_progress_path} shape={contact_progress.shape} "
        f"center={center_progress[0]:.2f}->{center_progress[-1]:.2f} "
        f"contact={contact_progress[0]:.2f}->{contact_progress[-1]:.2f}"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute LIBERO Spatial contact-aware SOLE-R1 progress labels."
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
