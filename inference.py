"""V3 episode inference with the exact dataset prompt and collage layout."""

import argparse
import json
from pathlib import Path
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from toolkits.libero_sole.common import task_description_from_episode
from toolkits.libero_sole.generate_cot_dataset import _image_files, build_prompt_json
from toolkits.libero_sole.generate_cot_dataset_v2 import (
    _make_grid,
    build_problem_prompt_v2,
)
from stage3_tools.rlvr_core import parse_progress_answer_strict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["sequential", "from_zero"], default="sequential")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--max-queries", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    a = p.parse_args()
    if a.stride < 1:
        p.error("--stride must be positive")
    if a.output.exists():
        raise FileExistsError(a.output)
    main_frames = _image_files(a.episode / "images/main")
    wrist = _image_files(a.episode / "images/wrist")
    if not main_frames or len(main_frames) != len(wrist):
        raise ValueError("missing or mismatched camera frames")
    task = task_description_from_episode(a.episode)
    if not task:
        raise ValueError("missing task instruction")
    processor = AutoProcessor.from_pretrained(a.model, local_files_only=True)
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            a.model,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .to("cuda")
        .eval()
    )
    queries = list(range(a.stride, len(main_frames), a.stride))
    if not queries or queries[-1] != len(main_frames) - 1:
        queries.append(len(main_frames) - 1)
    if a.max_queries:
        queries = queries[: a.max_queries]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    prev = 0
    progress = 0
    fz = a.mode == "from_zero"
    with a.output.open("x") as f:
        for current in queries:
            indices = [0, current] if fz else [0, prev, current]
            image = _make_grid(
                [frames[i] for frames in (main_frames, wrist) for i in indices],
                columns=len(indices),
            )
            problem = build_problem_prompt_v2(task, 0, progress, fz)
            messages = json.loads(build_prompt_json(problem))
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[prompt],
                images=[image],
                return_tensors="pt",
                add_special_tokens=False,
            ).to("cuda")
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=a.max_new_tokens,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )
            text = processor.tokenizer.decode(
                output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
            parsed = parse_progress_answer_strict(text)
            record = {
                "previous": 0 if fz else prev,
                "current": current,
                "mode": a.mode,
                "previous_progress": None if fz else progress,
                "problem": problem,
                "completion": text,
                "answer": parsed.answer,
                "parse_ok": parsed.ok,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if not parsed.ok:
                raise RuntimeError(
                    f"invalid completion at frame {current}: {parsed.reason}"
                )
            prev = current
            progress = parsed.answer


if __name__ == "__main__":
    main()
