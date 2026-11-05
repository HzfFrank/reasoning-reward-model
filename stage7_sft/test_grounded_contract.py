#!/usr/bin/env python3
"""Local approved-prompt and export/inference consistency checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from inference import build_query, episode_frames, task_from_v3
from prompts import SYSTEM, SEQUENTIAL_USER, FROM_ZERO_USER, VERSION, TEMPLATE_SHA256
from protocol import check_progress_contract, load_samples, messages
from sampling import uniform_storyboard_frames

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'doc/LIBERO_SOLE_V7_APPROVED_COT_AND_INFERENCE_CN.md'
ORIGINAL = ROOT / 'stage7_artifacts/v7_grounded_contract_examples_20261012_r1/samples'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    args = parser.parse_args()
    document = DOC.read_text()
    sections = [document.split('### ' + title + '\n\n```text\n', 1)[1].split('\n```', 1)[0]
                for title in ('SYSTEM', 'SEQUENTIAL USER', 'FROM_ZERO USER')]
    if sections != [SYSTEM, SEQUENTIAL_USER, FROM_ZERO_USER]:
        raise AssertionError('student prompt differs from approved document')
    samples = load_samples(Path(args.data))
    root = ROOT / 'verified_datasets/rollout_trajectories_v3/episode_000041_rank0000_env0004_id000008_success'
    storyboard = uniform_storyboard_frames(episode_frames(root), 8)
    assert storyboard == [0, 14, 28, 42, 57, 71, 85, 99]
    source_task = task_from_v3(ROOT / 'verified_datasets/sft_dataset_v3/sole_cot_dataset.jsonl', root.name)
    rows = []
    for sample in samples:
        assert sample.task == source_task
        name = f'{sample.mode}_{sample.current}'
        original = json.loads((ORIGINAL / (name + '.json')).read_text())
        assert sample.completion == original['completion']
        rebuilt = build_query(root, sample.task, sample.mode, sample.current, sample.previous,
                              sample.previous_progress, storyboard)
        assert messages(rebuilt) == messages(sample)
        assert not check_progress_contract(sample.completion, sample.mode, sample.previous_progress)
        incorrect = sample.completion.replace(f'Current task progress is estimated at {original["source"]["answer"]}%.',
                                              'Current task progress is estimated at -77%.')
        assert check_progress_contract(incorrect, sample.mode, sample.previous_progress)
        rows.append({'sample_id': sample.sample_id, 'mode': sample.mode,
                     'images': len(sample.images), 'completion_sha256': hashlib.sha256(sample.completion.encode()).hexdigest()})
    print(json.dumps({'status': 'passed', 'prompt_version': VERSION, 'template_sha256': TEMPLATE_SHA256,
                      'storyboard': storyboard, 'samples': rows}, indent=2))


if __name__ == '__main__': main()
