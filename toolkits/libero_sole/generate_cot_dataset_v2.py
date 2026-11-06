from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image as PILImage

from toolkits.libero_sole import generate_cot_dataset as v1
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

DATA_SOURCE = "libero_spatial_rlinf_sole_cot_v5"
DATASET_VERSION = "v2_direction_from_zero"
DEFAULT_FROM_ZERO_RATIO = 0.3
DEFAULT_DROPOUT_SEED = 20260813
SAMPLED_IMAGE_DIR_SUFFIX = "v2"
SAMPLED_METADATA_NAME = "sole_image_sampled_metadata_v2.json"


@dataclass(frozen=True)
class V2CotSample(v1.CotSample):
    from_zero: bool = False


@dataclass(frozen=True)
class V2DebugItem:
    sample: V2CotSample
    qwen_reasoning: str
    row: dict[str, Any]


def _stable_unit_interval(key: str) -> float:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) / float(2**64)


def _use_from_zero(
    episode_id: str,
    timestep: int,
    dropout: float,
    seed: int,
) -> bool:
    if dropout <= 0:
        return False
    if dropout >= 1:
        return True
    key = f"{seed}:{episode_id}:{timestep}"
    return _stable_unit_interval(key) < dropout


def _make_grid(paths: list[Path], columns: int) -> PILImage.Image:
    images = [v1._load_rgb(path) for path in paths]
    ref = images[0]
    images = [v1._resize_like(img, ref) for img in images]
    rows = len(images) // columns
    grid = PILImage.new("RGB", (ref.width * columns, ref.height * rows))
    for idx, img in enumerate(images):
        row = idx // columns
        col = idx % columns
        grid.paste(img, (col * ref.width, row * ref.height))
    return grid


def build_sampled_sole_image_v2(
    episode_dir: Path,
    timestep: int,
    previous_timestep: int,
    sampling_name: str,
    from_zero: bool,
    overwrite: bool,
) -> Path:
    main_files = v1._image_files(episode_dir / "images" / "main")
    wrist_files = v1._image_files(episode_dir / "images" / "wrist")
    total = min(len(main_files), len(wrist_files))
    if timestep >= total or previous_timestep >= total:
        raise IndexError("sampled timestep exceeds image sequence length")

    out_dir = (
        episode_dir
        / "sole_images_sampled"
        / f"{sampling_name}_{SAMPLED_IMAGE_DIR_SUFFIX}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_fromzero" if from_zero else ""
    out_path = (
        out_dir / f"t{timestep:06d}_prev{previous_timestep:06d}{suffix}.png"
    )
    if out_path.exists() and not overwrite:
        return out_path.resolve()

    if from_zero:
        grid = _make_grid(
            [
                main_files[0],
                main_files[timestep],
                wrist_files[0],
                wrist_files[timestep],
            ],
            columns=2,
        )
    else:
        grid = _make_grid(
            [
                main_files[0],
                main_files[previous_timestep],
                main_files[timestep],
                wrist_files[0],
                wrist_files[previous_timestep],
                wrist_files[timestep],
            ],
            columns=3,
        )
    grid.save(out_path)
    return out_path.resolve()


def _reference_state_text(text: str, from_zero: bool) -> str:
    if not from_zero:
        return text
    return re.sub(
        r"^At the previous timestep",
        "At the first timestep",
        text,
        count=1,
    )


def _comparison_sentence(transition: dict[str, Any], from_zero: bool) -> str:
    events = transition["semantic_events"]
    reference = "first timestep" if from_zero else "previous timestep"
    gripper_motion = events["gripper_motion"]
    goal_motion = events["goal_motion"]
    contact_event = events["contact_event"]
    object_lift = events["object_lift"]

    clauses = []
    if "better position" in gripper_motion:
        clauses.append("the gripper has moved closer to the object")
    elif "farther" in gripper_motion:
        clauses.append("the gripper has moved farther from the object")
    else:
        clauses.append(
            "the gripper position has not clearly improved relative to the object"
        )

    if "establishes contact" in contact_event:
        clauses.append("contact with the object has been established")
    elif "maintains contact" in contact_event:
        clauses.append("contact with the object is maintained")
    elif "lost contact" in contact_event:
        clauses.append("contact with the object has been lost")

    if "lifted from" in object_lift or "remains lifted" in object_lift:
        clauses.append(object_lift)
    elif "closer to the target" in goal_motion:
        clauses.append("the object has moved closer to the target location")
    elif "farther from the target" in goal_motion:
        clauses.append("the object has moved farther from the target location")
    elif "not make a clear move" in goal_motion:
        clauses.append("the object has not clearly moved toward the target")

    return (
        f"Compared with the {reference}, the current timestep shows that "
        f"{v1._join_event_clauses(clauses)}."
    )


def _progress_direction(previous_progress: int, current_progress: int) -> str:
    if current_progress > previous_progress:
        return "increasing"
    if current_progress < previous_progress:
        return "decreasing"
    return "unchanged"


def _direction_sentence(
    previous_progress: int,
    current_progress: int,
    from_zero: bool,
) -> str:
    direction = _progress_direction(previous_progress, current_progress)
    if from_zero:
        if current_progress > 0:
            return (
                "Therefore, the task progress appears to be increasing "
                "from the initial state."
            )
        return (
            "Therefore, the task progress appears to be unchanged "
            "from the initial state."
        )
    if direction == "increasing":
        return "Therefore, the task progress appears to be increasing."
    if direction == "decreasing":
        return "Therefore, the task progress appears to be decreasing."
    return "Therefore, the task progress appears to be unchanged."


def _calibration_sentence(
    previous_progress: int,
    current_progress: int,
    from_zero: bool,
) -> str:
    direction = _progress_direction(previous_progress, current_progress)
    if from_zero:
        return (
            "Based on the change from the first timestep, I would estimate the "
            f"current task progress to be about {current_progress}%."
        )
    if direction == "increasing":
        return (
            f"Given that the previous task progress was {previous_progress}%, "
            "the current task progress seems to have increased to "
            f"{current_progress}%."
        )
    if direction == "decreasing":
        return (
            f"Given that the previous task progress was {previous_progress}%, "
            "the current task progress seems to have decreased to "
            f"{current_progress}%."
        )
    return (
        f"Given that the previous task progress was {previous_progress}%, "
        f"I would keep the current task progress at about {current_progress}%."
    )


def generate_template_reasoning_v2(
    task_description: str,
    transition: dict[str, Any],
    progress: float,
    from_zero: bool,
) -> str:
    del task_description
    previous_progress = 0 if from_zero else transition["progress"]["previous"]
    current_progress = int(round(float(progress)))
    remaining = transition["remaining_subgoal"]
    if remaining.startswith("no remaining subgoal"):
        remaining_sentence = "No further subgoal remains because the task is complete."
    else:
        remaining_sentence = (
            f"To complete the task, the robot still needs to {remaining}."
        )

    return " ".join(
        [
            _reference_state_text(transition["previous_state"], from_zero),
            transition["current_state"],
            _comparison_sentence(transition, from_zero),
            _direction_sentence(previous_progress, current_progress, from_zero),
            _calibration_sentence(previous_progress, current_progress, from_zero),
            remaining_sentence,
        ]
    )


def build_problem_prompt_v2(
    task_description: str,
    first_progress: float,
    previous_progress: float,
    from_zero: bool,
) -> str:
    if from_zero:
        return (
            "Here is an image containing multiple camera views of a robot attempting "
            "to complete a task. The views on the top are from an external camera. "
            "The views on the bottom are from the robot's wrist camera. The views "
            "from the very first timestep are shown to the left. The views from "
            "the current timestep are shown to the right. The task description is: "
            f"{task_description}. The task progress for the very first timestep is "
            f"{int(round(first_progress))}%. Predict the task progress for the "
            "current timestep."
        )
    return v1.build_problem_prompt(
        task_description=task_description,
        first_progress=first_progress,
        previous_progress=previous_progress,
    )


def build_qwen_instruction_v2(sample: V2CotSample) -> str:
    qwen_facts = {
        "prompt_mode": "from_zero" if sample.from_zero else "previous_progress",
        "timesteps": sample.transition["timesteps"],
        "previous_state": sample.transition["previous_state"],
        "current_state": sample.transition["current_state"],
        "semantic_events": sample.transition["semantic_events"],
        "progress": sample.transition["progress"],
        "task_state": sample.transition["task_state"],
    }
    transition_text = json.dumps(qwen_facts, ensure_ascii=False, indent=2)
    if sample.from_zero:
        mode_instruction = (
            "This is a from-zero training example. The image contains the first "
            "and current timesteps only, and the final user prompt does not "
            "provide previous progress. Compare the first timestep with the "
            "current timestep. Do not mention previous progress or a previous "
            "timestep."
        )
        progress_text = (
            f"Ground-truth current progress: {int(round(sample.progress))}%\n"
            "First timestep progress: 0%"
        )
    else:
        mode_instruction = (
            "Compare the previous sampled timestep with the current sampled "
            "timestep, using the previous progress as the numeric anchor."
        )
        progress_text = (
            f"Ground-truth current progress: {int(round(sample.progress))}%\n"
            f"Previous sampled progress: {int(round(sample.previous_progress))}%"
        )
    return (
        "Rewrite the template into fluent SOLE-R1 style robot reasoning. Use the "
        "image as visual context and the semantic events as ground-truth facts. "
        "Keep the factual content from the template, including the direction of "
        "progress. The reasoning should explicitly compare the relevant timesteps, "
        "state whether task progress appears increasing, decreasing, or unchanged, "
        "and mention the next remaining subgoal when one remains. Do not invent "
        "new objects, do not mention simulator variables, geometry names, "
        "coordinates, numeric distances, or internal phase labels such as "
        "approach, grasp, lift, transport, or place, and do not change the "
        "progress. Do not infer success unless it is stated in the semantic "
        "events. Return only <think>...</think>. Do not include <answer>.\n\n"
        f"{mode_instruction}\n\n"
        f"Task description:\n{sample.task_description}\n\n"
        f"{progress_text}\n\n"
        f"Semantic events:\n{transition_text}\n\n"
        f"Template reasoning:\n{sample.template_reasoning}"
    )


class QwenVlParaphraserV2(v1.QwenVlParaphraser):
    def _chat_prompt(self, sample: V2CotSample) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(sample.image_path)},
                    {"type": "text", "text": build_qwen_instruction_v2(sample)},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


class TemplateParaphraserV2(v1.BaseParaphraser):
    def generate(self, samples: list[V2CotSample]) -> list[str]:
        return [sample.template_reasoning for sample in samples]


def build_samples_for_episode_v2(
    episode_dir: Path,
    sampling: v1.SamplingConfig,
    progress_name: str,
    from_zero_ratio: float,
    dropout_seed: int,
    overwrite_images: bool,
) -> list[V2CotSample]:
    semantic_states = _load_semantic_states(episode_dir / "semantic_states.pkl")
    contact_records = v1._load_contact_records(episode_dir)
    progress = v1._load_progress(episode_dir, progress_name)
    image_total = min(
        len(v1._image_files(episode_dir / "images" / "main")),
        len(v1._image_files(episode_dir / "images" / "wrist")),
    )
    total_steps = min(len(semantic_states), len(progress), image_total)
    success_flags = v1._success_flags_from_states(semantic_states[:total_steps])
    first_success = v1._first_success_timestep(success_flags)
    timesteps = v1.sample_timesteps(
        total_steps,
        sampling,
        semantic_states=semantic_states[:total_steps],
        contact_records=contact_records[:total_steps] if contact_records else None,
    )
    timesteps = v1._truncate_timesteps_at_success(timesteps, success_flags)
    sampling_name = v1._sampling_name(sampling)
    episode_id = episode_id_from_dir(episode_dir)
    task_description = task_description_from_episode(episode_dir)
    success = success_from_episode(episode_dir)

    def make_sample(
        sample_idx: int,
        timestep: int,
        previous_timestep: int,
        from_zero: bool,
    ) -> V2CotSample:
        transition_previous_timestep = 0 if from_zero else previous_timestep
        image_previous_timestep = 0 if from_zero else previous_timestep
        image_path = build_sampled_sole_image_v2(
            episode_dir=episode_dir,
            timestep=timestep,
            previous_timestep=image_previous_timestep,
            sampling_name=sampling_name,
            from_zero=from_zero,
            overwrite=overwrite_images,
        )
        transition = v1.extract_state_transition(
            semantic_states=semantic_states,
            progress=progress,
            timestep=timestep,
            previous_timestep=transition_previous_timestep,
            contact_records=contact_records,
        )
        if from_zero:
            transition = dict(transition)
            transition["timesteps"] = dict(transition["timesteps"])
            transition["timesteps"]["previous_source"] = "first_frame"
        template = generate_template_reasoning_v2(
            task_description=task_description,
            transition=transition,
            progress=float(progress[timestep]),
            from_zero=from_zero,
        )
        return V2CotSample(
            episode_dir=episode_dir,
            episode_id=episode_id,
            timestep=timestep,
            previous_timestep=transition_previous_timestep,
            total_steps=total_steps,
            image_path=image_path,
            task_description=task_description,
            progress=float(progress[timestep]),
            previous_progress=float(progress[transition_previous_timestep]),
            success=success,
            transition=transition,
            template_reasoning=template,
            from_zero=from_zero,
        )

    samples: list[V2CotSample] = []
    sampled_metadata = []
    for sample_idx, timestep in enumerate(timesteps):
        previous_timestep = timesteps[sample_idx - 1] if sample_idx > 0 else 0
        if sample_idx > 0 and success_flags[previous_timestep]:
            continue
        if first_success is not None and timestep > first_success:
            continue

        normal_sample = make_sample(
            sample_idx=sample_idx,
            timestep=timestep,
            previous_timestep=previous_timestep,
            from_zero=False,
        )
        samples.append(normal_sample)
        sampled_metadata.append(
            {
                "episode_id": episode_id,
                "sample_index": sample_idx,
                "mode": "previous_progress",
                "timestep": timestep,
                "first_frame": 0,
                "previous_frame": previous_timestep,
                "current_frame": timestep,
                "previous_source": "sampled_sequence",
                "from_zero": False,
                "progress": float(progress[timestep]),
                "image_path": str(normal_sample.image_path),
            }
        )

        add_from_zero = timestep > 0 and _use_from_zero(
            episode_id=episode_id,
            timestep=timestep,
            dropout=from_zero_ratio,
            seed=dropout_seed,
        )
        if not add_from_zero:
            continue
        from_zero_sample = make_sample(
            sample_idx=sample_idx,
            timestep=timestep,
            previous_timestep=0,
            from_zero=True,
        )
        samples.append(from_zero_sample)
        sampled_metadata.append(
            {
                "episode_id": episode_id,
                "sample_index": sample_idx,
                "mode": "from_zero",
                "timestep": timestep,
                "first_frame": 0,
                "previous_frame": 0,
                "current_frame": timestep,
                "previous_source": "first_frame",
                "from_zero": True,
                "progress": float(progress[timestep]),
                "image_path": str(from_zero_sample.image_path),
            }
        )
    v1._write_json(
        episode_dir / SAMPLED_METADATA_NAME,
        {
            "dataset_version": DATASET_VERSION,
            "sampling": sampling.__dict__,
            "from_zero_ratio": from_zero_ratio,
            "dropout_seed": dropout_seed,
            "first_success_timestep": first_success,
            "samples": sampled_metadata,
        },
    )
    return samples


def build_dataset_row_v2(
    sample: V2CotSample,
    reasoning: str,
    include_debug: bool,
    data_source: str,
) -> dict[str, Any]:
    answer = v1._progress_answer(sample.progress)
    problem = build_problem_prompt_v2(
        task_description=sample.task_description,
        first_progress=sample.transition["progress"]["first"],
        previous_progress=sample.previous_progress,
        from_zero=sample.from_zero,
    )
    completion = f"<think>{reasoning.strip()}</think><answer>{answer}</answer>"
    image_suffix = "_from_zero" if sample.from_zero else ""
    row = {
        "image": str(sample.image_path),
        "problem": problem,
        "solution_orig": answer.rstrip("%"),
        "solution": completion,
        "prompt": v1.build_prompt_json(problem, completion=completion),
        "image_name": (
            f"{sample.episode_id}_timestep_{sample.timestep}{image_suffix}.png"
        ),
        "from_annotation": 0,
        "completion": completion,
        "data_source": data_source,
        "progress": float(sample.progress),
        "metadata": {
            "dataset_version": DATASET_VERSION,
            "episode_id": sample.episode_id,
            "episode_dir": str(sample.episode_dir),
            "image_path": str(sample.image_path),
            "timestep": sample.timestep,
            "previous_timestep": sample.previous_timestep,
            "previous_source": "from_zero" if sample.from_zero else "sampled_sequence",
            "from_zero": sample.from_zero,
            "total_steps": sample.total_steps,
            "success": sample.success,
            "task_phase": sample.transition["task_phase"],
        },
    }
    if include_debug:
        row["metadata"]["transition"] = sample.transition
        row["metadata"]["template_reasoning"] = sample.template_reasoning
    return row


def _row_key(row: dict[str, Any]) -> tuple[str, int, bool] | None:
    metadata = row.get("metadata") or {}
    episode_id = metadata.get("episode_id")
    timestep = metadata.get("timestep")
    if episode_id is None or timestep is None:
        return None
    return str(episode_id), int(timestep), bool(metadata.get("from_zero", False))


def write_debug_package(path: Path, debug_items: list[V2DebugItem], limit: int) -> None:
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
        v1._write_json(sample_dir / "task.json", task)
        v1._write_json(
            sample_dir / "progress.json",
            {
                "first": item.sample.transition["progress"]["first"],
                "previous": int(round(item.sample.previous_progress)),
                "current": int(round(item.sample.progress)),
                "raw_current": item.sample.progress,
                "from_zero": item.sample.from_zero,
            },
        )
        v1._write_json(
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
        v1._write_json(sample_dir / "dataset_row.json", item.row)
    readme = (
        "Debug package for RLinf LIBERO Spatial SOLE-R1 CoT v2.\n"
        "Samples keep the v1 timestep sampling and progress labels. About 30% "
        "extra rows use the official-style from-zero prompt/image by default; "
        "all original [first | previous sampled | current] rows are kept. CoT "
        "templates explicitly compare timesteps and state whether progress "
        "appears increasing, decreasing, or unchanged.\n"
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


def _chunks(items: list[V2CotSample], size: int) -> Iterable[list[V2CotSample]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SOLE-R1 style CoT SFT data from existing LIBERO rollout "
            "samples using the v2 prompt/COT recipe."
        )
    )
    parser.add_argument("path", help="Episode directory or trajectory root.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Defaults to <path>/sole_cot_dataset_v2.jsonl.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-json", default=None)
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
    parser.add_argument("--max-sample-gap", type=int, default=None)
    parser.add_argument("--no-include-last", action="store_true")
    parser.add_argument("--progress-name", default="progress_contact.npy")
    parser.add_argument(
        "--from-zero-ratio",
        "--previous-progress-dropout",
        dest="from_zero_ratio",
        type=float,
        default=DEFAULT_FROM_ZERO_RATIO,
        help=(
            "Deterministic fraction of sampled timesteps that also receive an "
            "additional official-style from-zero row. Normal previous-progress "
            "rows are always kept."
        ),
    )
    parser.add_argument("--dropout-seed", type=int, default=DEFAULT_DROPOUT_SEED)
    parser.add_argument(
        "--paraphraser",
        choices=["template", "qwen"],
        default="template",
    )
    parser.add_argument("--model-path", default=v1.DEFAULT_QWEN_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--include-debug", action="store_true")
    parser.add_argument("--write-debug-files", action="store_true")
    parser.add_argument("--hf-dataset-dir", default=None)
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument("--debug-package", default=None)
    parser.add_argument("--debug-limit", type=int, default=8)
    parser.add_argument("--data-source", default=DATA_SOURCE)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not 0 <= args.from_zero_ratio <= 1:
        raise ValueError("--from-zero-ratio must be in [0, 1]")

    root = Path(args.path).expanduser().resolve()
    episode_dirs = find_episode_dirs(root)
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories found under {args.path}")

    sampling = v1.SamplingConfig(
        strategy=args.sampling_strategy,
        interval=args.interval,
        num_samples=args.num_samples,
        max_event_samples=args.max_event_samples,
        include_last=not args.no_include_last,
        max_sample_gap=args.max_sample_gap,
    )
    samples: list[V2CotSample] = []
    for episode_dir in episode_dirs:
        episode_samples = build_samples_for_episode_v2(
            episode_dir=episode_dir,
            sampling=sampling,
            progress_name=args.progress_name,
            from_zero_ratio=args.from_zero_ratio,
            dropout_seed=args.dropout_seed,
            overwrite_images=args.overwrite_images,
        )
        samples.extend(episode_samples)
        print(f"sampled {len(episode_samples)} timesteps: {episode_dir}")

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else root / "sole_cot_dataset_v2.jsonl"
    )
    progress_path = (
        Path(args.progress_json).expanduser().resolve()
        if args.progress_json
        else output.with_suffix(".progress.json")
    )
    existing_rows = v1._read_jsonl(output) if args.resume else []
    existing_keys = {key for row in existing_rows if (key := _row_key(row)) is not None}
    if output.exists() and not args.resume:
        output.unlink()
    if progress_path.exists() and not args.resume:
        progress_path.unlink()
    if (
        args.write_debug_files
        and output.with_suffix(".debug.jsonl").exists()
        and not args.resume
    ):
        output.with_suffix(".debug.jsonl").unlink()
    if existing_keys:
        before = len(samples)
        samples = [
            sample
            for sample in samples
            if (
                sample.episode_id,
                sample.timestep,
                sample.from_zero,
            )
            not in existing_keys
        ]
        print(
            f"resume: skipped {before - len(samples)} existing samples; "
            f"remaining {len(samples)}",
            flush=True,
        )

    debug_rows = []
    debug_items: list[V2DebugItem] = []
    processed = len(existing_rows)
    total_samples = processed + len(samples)
    start_time = time.monotonic()
    v1._write_progress_json(
        progress_path, "sampled", processed, total_samples, start_time
    )
    print(
        "progress "
        + v1._format_progress_bar(processed, total_samples)
        + " status=sampled",
        flush=True,
    )
    if samples:
        print(
            f"initializing paraphraser={args.paraphraser} samples={len(samples)} "
            f"batch_size={args.batch_size} output={output}",
            flush=True,
        )
        if args.paraphraser == "qwen":
            paraphraser: v1.BaseParaphraser = QwenVlParaphraserV2(
                model_path=args.model_path,
                device=args.device,
                torch_dtype=args.torch_dtype,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            paraphraser = TemplateParaphraserV2()
        print("paraphraser initialized", flush=True)
    else:
        paraphraser = TemplateParaphraserV2()
        print("no remaining samples; skip paraphraser initialization", flush=True)

    for batch_idx, batch in enumerate(_chunks(samples, args.batch_size), start=1):
        first = batch[0]
        message = (
            f"batch={batch_idx} size={len(batch)} first_episode={first.episode_id} "
            f"first_timestep={first.timestep}"
        )
        print("generating " + message, flush=True)
        v1._write_progress_json(
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
            v1._write_progress_json(
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
            reasoning = v1._strip_think_tags(reasoning) or sample.template_reasoning
            row = build_dataset_row_v2(
                sample=sample,
                reasoning=reasoning,
                include_debug=args.include_debug,
                data_source=args.data_source,
            )
            batch_rows.append(row)
            if args.debug_package:
                debug_items.append(
                    V2DebugItem(sample=sample, qwen_reasoning=reasoning, row=row)
                )
            if args.write_debug_files:
                debug_rows.append(
                    {
                        "episode_id": sample.episode_id,
                        "timestep": sample.timestep,
                        "previous_timestep": sample.previous_timestep,
                        "from_zero": sample.from_zero,
                        "image": str(sample.image_path),
                        "progress": sample.progress,
                        "transition": sample.transition,
                        "template_reasoning": sample.template_reasoning,
                        "reasoning": reasoning,
                    }
                )
        v1._append_jsonl(output, batch_rows)
        if args.write_debug_files:
            v1._append_jsonl(output.with_suffix(".debug.jsonl"), debug_rows)
            debug_rows.clear()
        processed += len(batch_rows)
        v1._write_progress_json(
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
            + v1._format_progress_bar(processed, total_samples)
            + f" status=running wrote_batch_rows={len(batch_rows)}",
            flush=True,
        )

    rows = v1._read_jsonl(output)
    v1._write_progress_json(
        progress_path, "complete", len(rows), total_samples, start_time
    )
    print(
        (
            "progress "
            + v1._format_progress_bar(len(rows), total_samples)
            + " status=complete"
        ),
        flush=True,
    )
    if args.hf_dataset_dir:
        from datasets import Dataset, Image

        hf_rows = [{key: row[key] for key in v1.OFFICIAL_COLUMNS} for row in rows]
        dataset = Dataset.from_list(hf_rows).cast_column("image", Image())
        dataset.save_to_disk(str(Path(args.hf_dataset_dir).expanduser().resolve()))
    if args.debug_package:
        write_debug_package(
            Path(args.debug_package).expanduser().resolve(),
            debug_items,
            args.debug_limit,
        )
    from_zero_count = sum(
        bool((row.get("metadata") or {}).get("from_zero")) for row in rows
    )
    print(
        f"wrote CoT dataset samples={len(rows)} episodes={len(episode_dirs)} "
        f"from_zero={from_zero_count} "
        f"from_zero_ratio={args.from_zero_ratio} "
        f"paraphraser={args.paraphraser}: {output}",
        flush=True,
    )
    if args.hf_dataset_dir:
        print(f"wrote HF dataset: {Path(args.hf_dataset_dir).expanduser().resolve()}")
    if args.debug_package:
        print(f"wrote debug package: {Path(args.debug_package).expanduser().resolve()}")


if __name__ == "__main__":
    main()
