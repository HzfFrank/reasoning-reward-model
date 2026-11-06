from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from toolkits.libero_sole.common import find_episode_dirs, write_json
from toolkits.libero_sole.compute_contact_progress import (
    _load_contact_records,
    compute_contact_progress,
)
from toolkits.libero_sole.compute_progress import (
    _load_semantic_states,
    compute_pick_place_progress,
)

V3_BETA = 0.75
V3_PROGRESS_NAME = "progress_contact_v3.npy"
V3_CENTER_PROGRESS_NAME = "progress_center_v3.npy"
V3_METADATA_NAME = "progress_contact_v3_metadata.json"
V3_CENTER_METADATA_NAME = "progress_center_v3_metadata.json"


def process_episode(
    episode_dir: Path,
    beta: float,
    monotonic: bool,
    overwrite: bool,
) -> bool:
    semantic_path = episode_dir / "semantic_states.pkl"
    contact_path = episode_dir / "contact_records.pkl"
    contact_progress_path = episode_dir / V3_PROGRESS_NAME
    center_progress_path = episode_dir / V3_CENTER_PROGRESS_NAME
    metadata_path = episode_dir / V3_METADATA_NAME
    center_metadata_path = episode_dir / V3_CENTER_METADATA_NAME

    if (
        contact_progress_path.exists()
        and center_progress_path.exists()
        and not overwrite
    ):
        print(f"skip existing v3 contact progress: {episode_dir}")
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
            semantic_states=semantic_states,
            beta=beta,
            monotonic=monotonic,
        )
        contact_progress, contact_diagnostics = compute_contact_progress(
            semantic_states=semantic_states,
            contact_records=contact_records,
            beta=beta,
            monotonic=monotonic,
        )
    except ValueError as exc:
        reason = {
            "supported": False,
            "dataset_version": "v3_beta075",
            "reason": str(exc),
        }
        write_json(metadata_path, reason)
        print(f"skip unsupported v3 contact progress: {episode_dir}: {exc}")
        return False

    np.save(center_progress_path, center_progress)
    np.save(contact_progress_path, contact_progress)

    center_diagnostics.update(
        {
            "supported": True,
            "dataset_version": "v3_beta075",
            "progress_name": V3_CENTER_PROGRESS_NAME,
            "rationale": (
                "v3-A keeps the SOLE-R1 geometric progress formula but fixes "
                "the LIBERO Spatial task-tuned beta at 0.75."
            ),
        }
    )
    contact_diagnostics.update(
        {
            "supported": True,
            "dataset_version": "v3_beta075",
            "progress_name": V3_PROGRESS_NAME,
            "rationale": (
                "v3-A keeps the SOLE-R1 geometric progress formula and changes "
                "only the task-tuned beta from the v1/v2 default 0.5 to 0.75. "
                "This reduces over-credit for pre-grasp approach while keeping "
                "successful final states at 100."
            ),
        }
    )
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
        description=(
            "Compute LIBERO Spatial SOLE-R1 v3-A contact-aware progress labels. "
            "v3-A keeps the paper-style formula and fixes beta=0.75 by default."
        )
    )
    parser.add_argument("path", help="Episode directory or trajectory root.")
    parser.add_argument("--beta", type=float, default=V3_BETA)
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
    print(
        f"processed={len(episode_dirs)} wrote={written} "
        f"progress_name={V3_PROGRESS_NAME} beta={args.beta}"
    )


if __name__ == "__main__":
    main()
