from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

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


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def merge_jsonl(inputs: list[Path], output: Path) -> int:
    return _write_jsonl(output, (row for path in inputs for row in _read_jsonl(path)))


def save_hf_dataset(jsonl_path: Path, hf_dataset_dir: Path) -> int:
    from datasets import Dataset, Image

    rows = []
    for row in _read_jsonl(jsonl_path):
        rows.append({key: row[key] for key in OFFICIAL_COLUMNS})
    dataset = Dataset.from_list(rows).cast_column("image", Image())
    dataset.save_to_disk(str(hf_dataset_dir))
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge SOLE-R1 style CoT JSONL shards and optionally save HF dataset."
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="Shard JSONL files")
    parser.add_argument("--output", required=True, help="Merged JSONL output")
    parser.add_argument(
        "--hf-dataset-dir", default=None, help="Optional HF save_to_disk dir"
    )
    args = parser.parse_args()

    inputs = [Path(value).expanduser().resolve() for value in args.inputs]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing shard JSONL files: {missing}")
    output = Path(args.output).expanduser().resolve()
    rows = merge_jsonl(inputs, output)
    print(f"merged rows={rows}: {output}")
    if args.hf_dataset_dir:
        hf_dir = Path(args.hf_dataset_dir).expanduser().resolve()
        hf_rows = save_hf_dataset(output, hf_dir)
        print(f"wrote HF dataset rows={hf_rows}: {hf_dir}")


if __name__ == "__main__":
    main()
