from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def is_episode_dir(path: Path) -> bool:
    return (path / "images").is_dir() and (path / "actions.npy").is_file()


def find_episode_dirs(path: str | Path) -> list[Path]:
    root = Path(path).expanduser().resolve()
    if is_episode_dir(root):
        return [root]
    episodes = [p for p in root.rglob("episode_*") if p.is_dir() and is_episode_dir(p)]
    return sorted(episodes)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def episode_id_from_dir(episode_dir: Path) -> str:
    metadata = load_json(episode_dir / "metadata.json", default={}) or {}
    return str(metadata.get("episode_id") or episode_dir.name)


def task_description_from_episode(episode_dir: Path) -> str:
    task = load_json(episode_dir / "task.json", default={}) or {}
    return str(
        task.get("language_instruction")
        or task.get("task_description")
        or task.get("task_name")
        or ""
    )


def success_from_episode(episode_dir: Path) -> bool | None:
    metadata = load_json(episode_dir / "metadata.json", default={}) or {}
    if "success" in metadata:
        return bool(metadata["success"])
    return None
