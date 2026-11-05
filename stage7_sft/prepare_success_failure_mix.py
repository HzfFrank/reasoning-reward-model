"""Freeze one globally ordered success/failure SFT exposure plan (no training)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'stage7_sft'))
from prompts import TEMPLATE_SHA256, VERSION
from protocol import load_samples

SUCCESS = ROOT / 'stage7_artifacts/v7_bulk_grounded_contract_20261012_r1/exports/snapshot_00014043'
FAILURE = ROOT / 'stage7_artifacts/v7_event_aligned_cot_retry_20261028_r1/exports/train.jsonl'
PREVIOUS_RUN = ROOT / 'stage7_artifacts/v7_sft_8gpu_grounded_20261018_r1'
FAILURE_AUDIT = FAILURE.parent / 'teacher_audit.jsonl'
SEED = 7
WORLD = 8
GRAD_ACCUM = 2


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def source_lines(path):
    raw = path.read_bytes().splitlines(keepends=True)
    rows = [json.loads(x) for x in raw if x.strip()]
    if len(rows) != len(raw): raise ValueError(f'{path}: blank lines')
    return raw, rows


def build(out):
    out = out.resolve()
    if not out.is_relative_to(ROOT / 'stage7_artifacts'):
        raise ValueError('output must be on project data disk')
    manifest = json.loads((PREVIOUS_RUN / 'protocol_manifest.json').read_text())
    baseline = json.loads((SUCCESS / 'manifest.json').read_text())
    assert manifest['data_path'] == str(SUCCESS / 'train.jsonl')
    assert manifest['data_sha256'] == sha(SUCCESS / 'train.jsonl')
    assert manifest['model_path'] == str((ROOT.parent / 'models/Qwen3-VL-8B-Instruct').resolve())
    assert manifest['prompt_version'] == VERSION and manifest['template_sha256'] == TEMPLATE_SHA256
    assert baseline['counts'] == {'train': 13364, 'validation': 336, 'test': 343}
    assert baseline['student_template_sha256'] == TEMPLATE_SHA256
    sbytes, success = source_lines(SUCCESS / 'train.jsonl')
    fbytes, failure = source_lines(FAILURE)
    assert len(success) == 13364 and len(failure) == 116
    assert len({x['sample_id'] for x in success}) == len(success)
    assert len({x['sample_id'] for x in failure}) == len(failure)
    assert not ({x['sample_id'] for x in success} & {x['sample_id'] for x in failure})
    assert all(x['split'] == 'train' for x in success + failure)
    assert all(x['status'] == 'structural_pass' for x in success + failure)
    assert all(x['mode'] == 'sequential' for x in failure)
    keys = list(success[0]); types = {k: type(v) for k, v in success[0].items()}
    for x in success + failure:
        # from_zero has null previous fields by contract; compare key/type
        # identity only against rows of the same mode.
        nullable = {'previous_frame', 'previous_progress', 'previous_progress_provenance'} if x['mode'] == 'from_zero' else set()
        if list(x) != keys or any(type(x[k]) is not types[k] for k in keys if k not in nullable):
            raise ValueError(f"success/failure schema mismatch: {x['sample_id']}")
        if x['prompt_version'] != VERSION or x['template_sha256'] != TEMPLATE_SHA256:
            raise ValueError(f"prompt mismatch: {x['sample_id']}")
    audit = {x['sample_id']: x for x in (json.loads(line) for line in FAILURE_AUDIT.read_text().splitlines() if line.strip())}
    if set(audit) != {x['sample_id'] for x in failure}:
        raise ValueError('failure audit identity mismatch')
    expected = {'miss': 16, 'drop': 27, 'wrong_place': 9, 'd1': 29, 'd2': 25, 'wrong_object': 10}
    assert Counter(x['failure_type'] for x in audit.values()) == expected
    # The existing loader validates the frozen student protocol and paths.
    if len(load_samples(SUCCESS / 'train.jsonl', validate_images=False)) != len(success):
        raise ValueError('success loader count mismatch')
    if len(load_samples(FAILURE, validate_images=False)) != len(failure):
        raise ValueError('failure loader count mismatch')
    n_success = len(success)
    n_failure = (n_success + 5) // 10
    success_order = list(range(n_success))
    random.Random(SEED).shuffle(success_order)
    failure_order = []
    rng = random.Random(SEED + 1)
    while len(failure_order) < n_failure:
        cycle = list(range(len(failure)))
        rng.shuffle(cycle)
        failure_order.extend(cycle[:n_failure - len(failure_order)])
    counts = Counter(failure_order)
    assert max(counts.values()) - min(counts.values()) <= 1
    q, rem = divmod(n_success, n_failure)
    plan = []
    s_cursor = 0
    def append(kind, source_index, padding=False):
        obj = (success if kind == 'success' else failure)[source_index]
        plan.append(dict(plan_index=len(plan), source=kind, source_index=source_index,
                         source_sample_id=obj['sample_id'], source_split='train',
                         exposure_ordinal=(exposure_counter[(kind, obj['sample_id'])] + 1),
                         padding=padding, failure_type=audit[obj['sample_id']]['failure_type'] if kind == 'failure' else None,
                         task_id=audit[obj['sample_id']]['task_id'] if kind == 'failure' else None))
        exposure_counter[(kind, obj['sample_id'])] += 1
    exposure_counter = Counter()
    for j, fi in enumerate(failure_order):
        # Four 11-success blocks are spread across the plan. All other gaps are 10.
        extra = ((j + 1) * rem // n_failure) - (j * rem // n_failure)
        for si in success_order[s_cursor:s_cursor + q + extra]: append('success', si)
        s_cursor += q + extra
        append('failure', fi)
    assert s_cursor == n_success and len(plan) == n_success + n_failure
    logical_length = len(plan)
    pad = (-logical_length) % (WORLD * GRAD_ACCUM)
    # Pad only failure IDs at the lower exposure count. Thus every success
    # remains exactly once and failure exposures still differ by at most one.
    lower = min(counts.values())
    pad_candidates = [i for i in range(len(failure)) if counts[i] == lower]
    rng.shuffle(pad_candidates)
    for fi in pad_candidates[:pad]: append('failure', fi, padding=True)
    assert len(plan) % (WORLD * GRAD_ACCUM) == 0
    assert Counter(x['source'] for x in plan if not x['padding']) == {'success': n_success, 'failure': n_failure}
    assert sum(x['source'] == 'success' for x in plan) == n_success
    fexposure = Counter(x['source_sample_id'] for x in plan if x['source'] == 'failure')
    assert max(fexposure.values()) - min(fexposure.values()) <= 1
    assert len(fexposure) == len(failure)
    # With one sample/rank/microbatch and accumulation=2, each step consumes
    # exactly 16 adjacent positions of this fixed global order.
    step_size = WORLD * GRAD_ACCUM
    half = (n_success + 1) // 2
    success_seen = 0
    checkpoints = {}
    for step in range(1, len(plan) // step_size + 1):
        segment = plan[(step - 1) * step_size:step * step_size]
        success_seen += sum(x['source'] == 'success' for x in segment)
        if '50_percent_success' not in checkpoints and success_seen >= half:
            checkpoints['50_percent_success'] = dict(step=step, success_exposures=success_seen,
                failure_exposures=sum(x['source'] == 'failure' and not x['padding'] for x in plan[:step * step_size]),
                padding_exposures=sum(x['padding'] for x in plan[:step * step_size]))
        if '100_percent_success' not in checkpoints and success_seen >= n_success:
            checkpoints['100_percent_success'] = dict(step=step, success_exposures=success_seen,
                failure_exposures=sum(x['source'] == 'failure' and not x['padding'] for x in plan[:step * step_size]),
                padding_exposures=sum(x['padding'] for x in plan[:step * step_size]))
    # CPU-level verification against Accelerate's actual BatchSamplerShard.
    from torch.utils.data import SequentialSampler, BatchSampler
    from accelerate.data_loader import BatchSamplerShard
    base_batches = BatchSampler(SequentialSampler(range(len(plan))), batch_size=1, drop_last=False)
    ranks = [list(i for b in BatchSamplerShard(base_batches, num_processes=WORLD, process_index=r,
             split_batches=False, even_batches=True) for i in b) for r in range(WORLD)]
    assert len({len(r) for r in ranks}) == 1
    assert len(ranks[0]) == len(plan) // WORLD
    rank_order = [ranks[r][i] for i in range(len(ranks[0])) for r in range(WORLD)]
    assert rank_order == list(range(len(plan)))
    assert len(set(rank_order)) == len(plan)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'data').mkdir(exist_ok=True)
    data_tmp = out / 'data/mixed_train.jsonl.partial'
    plan_tmp = out / 'data/exposure_plan.jsonl.partial'
    with data_tmp.open('wb') as data_file, plan_tmp.open('w') as plan_file:
        for p in plan:
            raw = (sbytes if p['source'] == 'success' else fbytes)[p['source_index']]
            data_file.write(raw if raw.endswith(b'\n') else raw + b'\n')
            plan_file.write(json.dumps(p, ensure_ascii=False) + '\n')
    os.replace(data_tmp, out / 'data/mixed_train.jsonl')
    os.replace(plan_tmp, out / 'data/exposure_plan.jsonl')
    by_type_unique = Counter(x['failure_type'] for x in audit.values())
    by_type_exposure = Counter(x['failure_type'] for x in plan if x['source'] == 'failure' and not x['padding'])
    by_task_unique = Counter(str(x['task_id']) for x in audit.values())
    by_task_exposure = Counter(str(x['task_id']) for x in plan if x['source'] == 'failure' and not x['padding'])
    record = dict(version='v7_success_failure_mixed_exposure_v1', seed=SEED,
        success_source=str(SUCCESS / 'train.jsonl'), success_sha256=sha(SUCCESS / 'train.jsonl'),
        success_validation=str(SUCCESS / 'validation.jsonl'), success_validation_sha256=sha(SUCCESS / 'validation.jsonl'),
        failure_source=str(FAILURE), failure_sha256=sha(FAILURE), failure_audit_sha256=sha(FAILURE_AUDIT),
        previous_run_manifest_sha256=sha(PREVIOUS_RUN / 'protocol_manifest.json'),
        model_path=manifest['model_path'], prompt_version=VERSION, template_sha256=TEMPLATE_SHA256,
        success_unique=n_success, failure_unique=len(failure), success_exposures=n_success,
        failure_exposures=n_failure, failure_padding_exposures=pad,
        logical_plan_length=logical_length, physical_plan_length=len(plan),
        effective_batch_size=step_size, optimizer_steps=len(plan)//step_size,
        checkpoints=checkpoints, failure_exposure_min=min(counts.values()),
        failure_exposure_max=max(counts.values()), physical_failure_exposure_min=min(fexposure.values()),
        physical_failure_exposure_max=max(fexposure.values()),
        failure_unique_by_type=dict(by_type_unique), failure_exposure_by_type=dict(by_type_exposure),
        failure_unique_by_task=dict(by_task_unique), failure_exposure_by_task=dict(by_task_exposure),
        mixed_train_sha256=sha(out / 'data/mixed_train.jsonl'), exposure_plan_sha256=sha(out / 'data/exposure_plan.jsonl'),
        rank_shard_check='Accelerate BatchSamplerShard 8 ranks: exactly-once over physical global plan',
        padding_source_ids=[x['source_sample_id'] for x in plan if x['padding']],
        training='not_started')
    atomic(out / 'data/mix_manifest.json', record)
    print(json.dumps({k: record[k] for k in ('success_unique','failure_unique','success_exposures','failure_exposures','failure_padding_exposures','physical_plan_length','optimizer_steps','checkpoints','rank_shard_check')}, ensure_ascii=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser();ap.add_argument('--output', type=Path, required=True)
    build(ap.parse_args().output)
