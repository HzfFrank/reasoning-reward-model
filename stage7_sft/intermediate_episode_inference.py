#!/usr/bin/env python3
"""Read-only exploratory inference from one exported FSDP intermediate model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from transformers import Qwen3VLForConditionalGeneration

from inference import build_query, episode_frames, generate_one, task_from_v3
from processing import load_processor
from prompts import VERSION, TEMPLATE_SHA256
from sampling import uniform_storyboard_frames


def atomic_json(path: Path, value: dict):
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def file_sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checked_model(export: Path, device: str):
    model, info = Qwen3VLForConditionalGeneration.from_pretrained(
        str(export), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='sdpa', low_cpu_mem_usage=True,
        output_loading_info=True)
    problems = {k: info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs') if info.get(k)}
    if problems:
        raise ValueError(f'export has missing/unexpected/mismatched weights: {problems}')
    model.to(device).eval()
    if any(p.device.type != 'cuda' for p in model.parameters()):
        raise ValueError('CPU/disk model offload is not permitted for this inference')
    return model, {'missing_keys': [], 'unexpected_keys': [], 'mismatched_keys': [],
                   'parameters': sum(p.numel() for p in model.parameters()),
                   'device': device, 'dtype': str(next(model.parameters()).dtype)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', type=Path, required=True)
    ap.add_argument('--snapshot', type=Path, required=True)
    ap.add_argument('--episode-dir', type=Path, required=True)
    ap.add_argument('--v3-jsonl', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--mode', choices=['sequential', 'from_zero'], required=True)
    ap.add_argument('--queries', required=True, help='comma-separated, strictly increasing frame IDs')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--max-new-tokens', type=int, default=512)
    args = ap.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((args.snapshot / 'v7_protocol_manifest.json').read_text())
    trainer_state = json.loads((args.snapshot / 'trainer_state.json').read_text())
    if protocol['prompt_version'] != VERSION or protocol['template_sha256'] != TEMPLATE_SHA256:
        raise ValueError('checkpoint protocol and current shared student prompt differ')
    if protocol['prompt_file_sha256'] != file_sha(Path(__file__).with_name('prompts.py')):
        raise ValueError('prompts.py differs from checkpoint protocol')
    if protocol['image_pixels'] != 65536 or protocol['max_seq_length'] != 8192:
        raise ValueError('unexpected checkpoint image or sequence budget')
    export_manifest = json.loads((args.export / 'export_manifest.json').read_text())
    if int(trainer_state['global_step']) != int(export_manifest['global_step']):
        raise ValueError('export and snapshot steps differ')
    frames = episode_frames(args.episode_dir)
    storyboard = uniform_storyboard_frames(frames, 8)
    task = task_from_v3(args.v3_jsonl, args.episode_dir.name)
    queries = [int(x) for x in args.queries.split(',')]
    if queries != sorted(set(queries)) or any(q not in frames for q in queries):
        raise ValueError('queries must be unique ordered paired rollout frame IDs')
    config = {'checkpoint_global_step': trainer_state['global_step'], 'export': str(args.export.resolve()),
              'episode_dir': str(args.episode_dir.resolve()), 'task': task,
              'mode': args.mode, 'queries': queries, 'storyboard_frames': storyboard,
              'prompt_version': VERSION, 'template_sha256': TEMPLATE_SHA256,
              'image_pixels': 65536, 'max_seq_length': 8192,
              'max_new_tokens': args.max_new_tokens, 'do_sample': False,
              'context_mode': 'offline_full_video', 'future_visual_context_available': True,
              'initial_frame': frames[0], 'device': args.device}
    config_path = output / f'{args.mode}_config.json'
    if config_path.exists():
        old = json.loads(config_path.read_text())
        # Resuming a pilot with a longer query list is allowed; all protocol
        # and model settings must remain identical.
        if {k:v for k,v in old.items() if k != 'queries'} != {k:v for k,v in config.items() if k != 'queries'}:
            raise ValueError('existing inference configuration conflicts with this run')
        if queries[:len(old['queries'])] != old['queries']:
            raise ValueError('resumed query list must extend the pilot prefix')
    atomic_json(config_path, config)
    processor = load_processor(str(args.export), 65536)
    model, loading = load_checked_model(args.export, args.device)
    atomic_json(output / f'{args.mode}_model_loading_check.json', loading)
    records = output / args.mode / 'per_query'
    records.mkdir(parents=True, exist_ok=True)
    previous = frames[0]
    previous_progress = 0
    blocked_by = None
    for current in queries:
        path = records / f'{current:06d}.json'
        if blocked_by is not None:
            record = {'mode':args.mode,'current':current,'status':'blocked_by_previous_failure',
                      'blocked_by':blocked_by,'attempts':[]}
            if not path.exists(): atomic_json(path, record)
            continue
        if path.exists():
            record = json.loads(path.read_text())
            if record.get('mode') != args.mode or record.get('current') != current:
                raise ValueError(f'existing query record conflicts: {path}')
        else:
            sample = build_query(args.episode_dir, task, args.mode, current,
                                 previous if args.mode == 'sequential' else None,
                                 previous_progress if args.mode == 'sequential' else None,
                                 storyboard)
            try:
                record = generate_one(model, processor, sample, 8192, args.max_new_tokens, args.device)
            except Exception as exc:
                record = {'mode':args.mode, 'current':current, 'previous':sample.previous,
                          'previous_progress':sample.previous_progress, 'status':'engineering_failure',
                          'error':repr(exc), 'attempts':[]}
            if args.mode == 'sequential':
                record['previous_progress_provenance'] = ('initialized_zero_not_ground_truth' if current == frames[0]
                                                           else 'previous_model_prediction')
            record['checkpoint_step'] = trainer_state['global_step']
            record['storyboard_frames'] = storyboard
            atomic_json(path, record)
            print(f'{args.mode} current={current} status={record["status"]} progress={record.get("progress")} '
                  f'seconds={record.get("attempts", [{}])[0].get("seconds") if record.get("attempts") else None}',
                  flush=True)
        if args.mode == 'sequential':
            if record['status'] != 'ok':
                blocked_by = current
            else:
                previous = current
                previous_progress = record['progress']

    rows = []
    for current in queries:
        record = json.loads((records / f'{current:06d}.json').read_text())
        rows.append(record)
    with (output / f'{args.mode}_predictions.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['frame','status','progress','previous','previous_progress','previous_progress_provenance','seconds'])
        writer.writeheader()
        for row in rows:
            writer.writerow({'frame':row['current'],'status':row['status'],'progress':row.get('progress'),
                             'previous':row.get('previous'),'previous_progress':row.get('previous_progress'),
                             'previous_progress_provenance':row.get('previous_progress_provenance'),
                             'seconds':row.get('attempts',[{}])[0].get('seconds') if row.get('attempts') else None})
    summary = {'mode':args.mode,'queries':queries,'attempted':sum(x['status']!='blocked_by_previous_failure' for x in rows),
               'valid':sum(x['status']=='ok' for x in rows),
               'parse_failures':sum(x['status']=='parse_failure' for x in rows),
               'contract_failures':sum(x['status']=='contract_failure' for x in rows),
               'engineering_failures':sum(x['status']=='engineering_failure' for x in rows),
               'blocked':sum(x['status']=='blocked_by_previous_failure' for x in rows),
               'last_current':rows[-1]['current'] if rows else None}
    atomic_json(output / f'{args.mode}_summary.json', summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
