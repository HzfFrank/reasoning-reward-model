from __future__ import annotations

import sys

from toolkits.libero_sole import generate_cot_dataset_v2 as v2
from toolkits.libero_sole.compute_contact_progress_v3 import V3_PROGRESS_NAME

DATA_SOURCE = "libero_spatial_rlinf_sole_cot_v6"
DATASET_VERSION = "v3_beta075_direction_from_zero"
DEFAULT_PROGRESS_NAME = V3_PROGRESS_NAME


def _has_option(argv: list[str], names: set[str]) -> bool:
    return any(
        arg == name or arg.startswith(f"{name}=")
        for arg in argv
        for name in names
    )


def main() -> None:
    argv = sys.argv[1:]
    if "--help" not in argv and "-h" not in argv:
        if not _has_option(argv, {"--progress-name"}):
            argv.extend(["--progress-name", DEFAULT_PROGRESS_NAME])
        if not _has_option(argv, {"--data-source"}):
            argv.extend(["--data-source", DATA_SOURCE])

    v2.DATA_SOURCE = DATA_SOURCE
    v2.DATASET_VERSION = DATASET_VERSION
    v2.SAMPLED_IMAGE_DIR_SUFFIX = "v3"
    v2.SAMPLED_METADATA_NAME = "sole_image_sampled_metadata_v3.json"
    sys.argv = [sys.argv[0], *argv]
    v2.main()


if __name__ == "__main__":
    main()
