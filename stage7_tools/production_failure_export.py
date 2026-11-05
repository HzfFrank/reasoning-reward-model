"""Export checked failure records using the exact approved success JSONL schema."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'stage7_sft'))
from prompts import VERSION, TEMPLATE_SHA256
from protocol import Sample, messages, parse_completion, check_progress_contract, validate_sample, load_samples


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def export(run: Path, source: Path, name: str) -> dict:
    baseline = json.loads((run / 'manifests/success_baseline.json').read_text())
    snapshot = Path(baseline['snapshot'])
    if digest(snapshot / 'manifest.json') != baseline['snapshot_manifest_sha256']:
        raise ValueError('success baseline manifest changed')
    reference = json.loads((snapshot / 'train.jsonl').open().readline())
    keys = list(reference)
    if keys != baseline['sequential_schema_keys']:
        raise ValueError('success sequential schema changed')
    if baseline['student_prompt_version'] != VERSION or baseline['student_template_sha256'] != TEMPLATE_SHA256:
        raise ValueError('shared student prompt differs from success baseline')
    selected, isolated = {x: [] for x in ('train', 'validation', 'test')}, []
    audit = []
    for file in sorted((source / 'preflight').glob('*.json')):
        item = json.loads(file.read_text())
        sid = item['sample_id']
        result_path = source / 'results' / (sid + '.json')
        if not result_path.is_file():
            isolated.append({'sample_id': sid, 'reason': 'no_result'})
            continue
        result = json.loads(result_path.read_text())
        if result.get('sample_id') != sid or result.get('status') != 'structural_pass':
            isolated.append({'sample_id': sid, 'reason': result.get('status', 'identity_mismatch')})
            continue
        if result.get('label_visual_conflict') or result.get('semantic_review') not in ('semantic_checked', 'accepted_for_export'):
            isolated.append({'sample_id': sid, 'reason': 'semantic_review_or_conflict'})
            continue
        group = f"libero_spatial:task{item['task_id']}:reset{item['reset_state_id']}"
        split = baseline['group_split'].get(group, item.get('split', 'unassigned'))
        if split not in selected:
            isolated.append({'sample_id': sid, 'reason': 'unassigned_or_conflicting_split', 'group_id': group})
            continue
        if item.get('student_prompt_version') != VERSION or item.get('student_template_sha256') != TEMPLATE_SHA256:
            raise ValueError(f'{sid}: prompt version mismatch')
        completion = result['completion']
        parsed = parse_completion(completion)
        if not parsed['ok'] or parsed['progress'] != item['answer'] or check_progress_contract(completion, 'sequential', item['previous_progress']):
            raise ValueError(f'{sid}: invalid XML/progress contract')
        sample = Sample(sid, 'sequential', item['task'], split, item['episode_id'],
                        item.get('initial', 0), item['previous'], item['current'],
                        item['previous_progress'], 'geometry_v2_reference', tuple(item['images']), completion)
        validate_sample(sample)
        student_messages = messages(sample)
        if item['student_messages'] != student_messages:
            raise ValueError(f'{sid}: student messages differ from shared protocol')
        row = dict(sample_id=sid, mode='sequential', task=item['task'], split=split,
                   episode_id=item['episode_id'], initial_frame=sample.initial,
                   previous_frame=sample.previous, current_frame=sample.current,
                   previous_progress=sample.previous_progress, answer=item['answer'],
                   source_jsonl_line=0, source_line_sha256=item['source_sha256'],
                   is_initialization=False, completion=completion, status='structural_pass',
                   images=item['images'], previous_progress_provenance='geometry_v2_reference',
                   prompt_version=VERSION, template_sha256=TEMPLATE_SHA256,
                   student_messages=student_messages)
        if list(row) != keys or any(type(row[k]) is not type(reference[k]) for k in keys):
            raise ValueError(f'{sid}: success schema/type mismatch')
        selected[split].append(row)
        audit.append({'sample_id': sid, 'group_id': group, 'source_preflight': str(file),
                      'source_preflight_sha256': digest(file), 'result': str(result_path),
                      'result_sha256': digest(result_path), 'trajectory': item['trajectory_dir'],
                      'progress_config_sha256': item['config_sha256'],
                      'event_evidence': item.get('event_evidence'), 'status': 'accepted_for_export'})
    all_ids = [r['sample_id'] for rows in selected.values() for r in rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError('duplicate sample IDs')
    stage = run / 'exports' / (name + '.partial')
    target = run / 'exports' / name
    if target.exists() or stage.exists():
        raise FileExistsError(target)
    stage.mkdir(parents=True)
    for split, rows in selected.items():
        with (stage / (split + '.jsonl')).open('w') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    with (stage / 'audit.jsonl').open('w') as f:
        for row in audit:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    atomic_json(stage / 'isolated.json', {'items': isolated})
    manifest = {'status': 'immutable_snapshot', 'partial': True, 'source': str(source),
                'success_baseline_snapshot': str(snapshot),
                'success_baseline_manifest_sha256': baseline['snapshot_manifest_sha256'],
                'prompt_version': VERSION, 'template_sha256': TEMPLATE_SHA256,
                'counts': {k: len(v) for k, v in selected.items()},
                'isolated_count': len(isolated), 'sample_ids': all_ids,
                'hashes': {p.name: digest(p) for p in stage.glob('*.jsonl')}}
    atomic_json(stage / 'manifest.json', manifest)
    # Run the actual approved loader before publishing immutable files.
    for split, rows in selected.items():
        if rows:
            loaded = load_samples(stage / (split + '.jsonl'), validate_images=True)
            if len(loaded) != len(rows):
                raise ValueError(f'{split}: loader count mismatch')
    os.replace(stage, target)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--name', required=True)
    a = p.parse_args()
    print(json.dumps(export(a.run.resolve(), a.source.resolve(), a.name), ensure_ascii=False))
