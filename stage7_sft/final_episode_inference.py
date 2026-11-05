#!/usr/bin/env python3
"""Read-only final-model inference for one full offline rollout at fixed strides."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from inference import build_query, episode_frames, generate_one, task_from_v3
from intermediate_episode_inference import atomic_json, load_checked_model
from processing import load_processor
from prompts import VERSION, TEMPLATE_SHA256
from sampling import uniform_storyboard_frames


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_final_model(model_dir: Path) -> dict:
    index = json.loads((model_dir / 'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    if not shards or any(not (model_dir / name).is_file() for name in shards):
        raise ValueError('final_model is missing a listed weight shard')
    protocol = json.loads((model_dir / 'v7_protocol_manifest.json').read_text())
    if protocol['prompt_version'] != VERSION or protocol['template_sha256'] != TEMPLATE_SHA256:
        raise ValueError('final model protocol differs from shared v7 student prompt')
    if protocol['prompt_file_sha256'] != file_sha(Path(__file__).with_name('prompts.py')):
        raise ValueError('prompts.py bytes differ from training manifest')
    if protocol['image_pixels'] != 65536 or protocol['max_seq_length'] != 8192:
        raise ValueError('unexpected image/sequence configuration in final model')
    return {'protocol': protocol, 'model_dir': str(model_dir.resolve()),
            'index_sha256': file_sha(model_dir / 'model.safetensors.index.json'),
            'shards': [{'name': name, 'bytes': (model_dir / name).stat().st_size}
                       for name in shards], 'weight_keys': len(index['weight_map'])}


def query_frames(frames: list[int], stride: int) -> list[int]:
    if stride < 1:
        raise ValueError('stride must be positive')
    selected = frames[::stride]
    if selected[-1] != frames[-1]:
        selected.append(frames[-1])
    return selected


def export_csv(path: Path, rows: list[dict]):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['frame', 'status', 'progress', 'previous',
                            'previous_progress', 'previous_progress_provenance',
                            'generated_tokens', 'seconds'])
        writer.writeheader()
        for row in rows:
            attempt = row.get('attempts', [{}])[0] if row.get('attempts') else {}
            writer.writerow({'frame': row['current'], 'status': row['status'],
                'progress': row.get('progress'), 'previous': row.get('previous'),
                'previous_progress': row.get('previous_progress'),
                'previous_progress_provenance': row.get('previous_progress_provenance'),
                'generated_tokens': attempt.get('generated_tokens'),
                'seconds': attempt.get('seconds')})


def run_one(model, processor, a, frames: list[int], storyboard: list[int], task: str,
            mode: str, stride: int):
    queries = query_frames(frames, stride)
    sub = a.output_dir / f'stride_{stride}' / mode
    records = sub / 'per_query'
    records.mkdir(parents=True, exist_ok=True)
    config = {'kind': 'v7_final_model_episode_inference', 'mode': mode, 'stride': stride,
        'queries': queries, 'storyboard_frames': storyboard, 'task': task,
        'episode_dir': str(a.episode_dir.resolve()), 'model_dir': str(a.model_dir.resolve()),
        'prompt_version': VERSION, 'template_sha256': TEMPLATE_SHA256,
        'image_pixels': 65536, 'max_seq_length': 8192,
        'max_new_tokens': a.max_new_tokens, 'do_sample': False,
        'context_mode': 'offline_full_video', 'future_visual_context_available': True,
        'initial_frame': frames[0], 'device': a.device}
    config_path = sub / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError(f'existing run config conflict: {config_path}')
    atomic_json(config_path, config)
    previous = frames[0]
    previous_progress = 0
    blocked_by = None
    rows = []
    for current in queries:
        path = records / f'{current:06d}.json'
        if blocked_by is not None:
            row = {'mode': mode, 'stride': stride, 'current': current,
                   'status': 'blocked_by_previous_failure', 'blocked_by': blocked_by,
                   'attempts': []}
            if not path.exists(): atomic_json(path, row)
        elif path.exists():
            row = json.loads(path.read_text())
            if row.get('mode') != mode or row.get('stride') != stride or row.get('current') != current:
                raise ValueError(f'existing record identity conflict: {path}')
            if mode == 'sequential' and (row.get('previous') != previous or
                                         row.get('previous_progress') != previous_progress):
                raise ValueError(f'existing sequential feedback conflict: {path}')
        else:
            sample = build_query(a.episode_dir, task, mode, current,
                previous if mode == 'sequential' else None,
                previous_progress if mode == 'sequential' else None, storyboard)
            try:
                row = generate_one(model, processor, sample, 8192, a.max_new_tokens, a.device)
            except Exception as exc:
                row = {'mode': mode, 'current': current, 'previous': sample.previous,
                       'previous_progress': sample.previous_progress,
                       'status': 'engineering_failure', 'error': repr(exc), 'attempts': []}
            row['stride'] = stride
            export_manifest = a.model_dir / 'export_manifest.json'
            training_summary = a.model_dir.parent / 'train_summary.json'
            if export_manifest.is_file():
                row['model_kind'] = f"fsdp_export_step_{json.loads(export_manifest.read_text())['global_step']}"
            elif training_summary.is_file():
                row['model_kind'] = f"final_model_step_{json.loads(training_summary.read_text())['optimizer_steps']}"
            else:
                row['model_kind'] = 'final_model_step_unknown'
            row['storyboard_frames'] = storyboard
            if mode == 'sequential':
                row['previous_progress_provenance'] = ('initialized_zero_not_ground_truth'
                    if current == frames[0] else 'previous_model_prediction')
            atomic_json(path, row)
            attempt = row.get('attempts', [{}])[0] if row.get('attempts') else {}
            print(f'{mode} stride={stride} current={current} status={row["status"]} '
                  f'progress={row.get("progress")} seconds={attempt.get("seconds")}', flush=True)
        rows.append(row)
        if mode == 'sequential':
            if row['status'] != 'ok':
                blocked_by = current
            else:
                previous, previous_progress = current, row['progress']
        counts = {s: sum(r['status'] == s for r in rows) for s in sorted({r['status'] for r in rows})}
        atomic_json(sub / 'progress.json', {'processed': len(rows), 'total': len(queries),
                     'last_current': current, 'status_counts': counts})
    export_csv(sub / 'predictions.csv', rows)
    summary = {'mode': mode, 'stride': stride, 'queries': queries,
        'attempted': sum(r['status'] != 'blocked_by_previous_failure' for r in rows),
        'valid': sum(r['status'] == 'ok' for r in rows),
        'parse_failure': sum(r['status'] == 'parse_failure' for r in rows),
        'contract_failure': sum(r['status'] == 'contract_failure' for r in rows),
        'engineering_failure': sum(r['status'] == 'engineering_failure' for r in rows),
        'blocked': sum(r['status'] == 'blocked_by_previous_failure' for r in rows),
        'generated_tokens': sum(r['attempts'][0]['generated_tokens'] for r in rows if r.get('attempts')),
        'generation_seconds': round(sum(r['attempts'][0]['seconds'] for r in rows if r.get('attempts')), 3),
        'model_dir': str(a.model_dir.resolve()), 'prompt_version': VERSION}
    atomic_json(sub / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--episode-dir', type=Path, required=True)
    ap.add_argument('--v3-jsonl', type=Path)
    ap.add_argument('--task-file', type=Path,
                    help='environment task.json containing language_instruction; no progress labels')
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--mode', choices=['sequential', 'from_zero'], required=True)
    ap.add_argument('--strides', default='1,4')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--max-new-tokens', type=int, default=512)
    a = ap.parse_args()
    a.model_dir = a.model_dir.resolve()
    a.episode_dir = a.episode_dir.resolve()
    a.output_dir = a.output_dir.resolve()
    project = Path(__file__).resolve().parents[1]
    from portable_paths import check_output
    check_output(a.output_dir)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    model_manifest = verify_final_model(a.model_dir)
    atomic_json(a.output_dir / f'final_model_check_{a.mode}.json', model_manifest)
    frames = episode_frames(a.episode_dir)
    storyboard = uniform_storyboard_frames(frames, 8)
    if (a.task_file is None) == (a.v3_jsonl is None):
        raise ValueError('provide exactly one task source: --task-file or --v3-jsonl')
    if a.task_file is not None:
        task_file = a.task_file.resolve()
        if task_file != a.episode_dir / 'task.json':
            raise ValueError('task file must be the selected rollout task.json')
        task = json.loads(task_file.read_text())['language_instruction'].strip()
    else:
        task = task_from_v3(a.v3_jsonl, a.episode_dir.name)
    if not task:
        raise ValueError('empty task instruction')
    if frames[0] != 0 or frames != list(range(frames[-1] + 1)):
        raise ValueError('expected contiguous paired rollout frames starting at 0')
    processor = load_processor(str(a.model_dir), 65536)
    model, loading = load_checked_model(a.model_dir, a.device)
    atomic_json(a.output_dir / f'model_loading_{a.mode}.json', loading)
    strides = [int(x) for x in a.strides.split(',')]
    if len(strides) != len(set(strides)) or not strides:
        raise ValueError('strides must be unique')
    for stride in strides:
        run_one(model, processor, a, frames, storyboard, task, a.mode, stride)


if __name__ == '__main__': main()
