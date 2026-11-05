"""Resume-safe bulk generation for the immutable 116 event-aligned examples."""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'stage7_tools'), str(ROOT / 'stage7_sft')]
import run_short_failure_six_review as six
import run_v7_geometry_v2_cot_examples as base
from run_v7_grounded_contract import validate
from protocol import Sample, messages, validate_sample, load_samples, check_progress_contract, parse_completion
from prompts import VERSION as STUDENT_VERSION, TEMPLATE_SHA256

SOURCE = ROOT / 'stage7_artifacts/v7_event_aligned_six_review_20261027_r1'
TYPES = {'miss': 16, 'drop': 27, 'wrong_place': 9, 'd1': 29, 'd2': 25, 'wrong_object': 10}
MODEL = 'gpt-6.1-sol'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def source(x):
    return dict(mode='sequential', task=x['task'], previous=x['previous_frame'],
                current=x['current_frame'], previous_progress=x['previous_progress'],
                answer=x['current_progress'])


def prepare(x):
    assert x['mode'] == 'sequential' and len(x['images']) == 22
    for im in x['images']:
        if digest(im['path']) != im['sha256']:
            raise ValueError(f"image hash mismatch: {x['sample_id']}")
    s = source(x)
    states = pickle.load((Path(x['trajectory_dir']) / 'semantic_states.pkl').open('rb'))
    first = states[0]
    keys = ['object_name', 'goal_name', 'object_pos', 'goal_pos', 'eef_pos', 'gripper_qpos']
    facts = {role: {k: six.clean(states[f].get(k)) for k in keys}
             for role, f in [('PREVIOUS', s['previous']), ('CURRENT', s['current'])]}
    registry = {k: six.clean(first.get(k)) for k in
                ['object_name', 'goal_name', 'language_instruction', 'task_name']}
    registry['non_target_ramekin_identity'] = 'glazed_rim_porcelain_ramekin_1'
    user = (base.teacher_user(s, x['images']) + '\n\nTEACHER-ONLY OBJECT REGISTRY\n' +
            json.dumps(registry) + '\nTEACHER-ONLY LOCAL SUBPHASE\n' +
            x.get('event_subphase', '') +
            '\nTEACHER-ONLY ENDPOINT STATE (world coordinates in metres, not image left/right; no grasp inferred from proximity alone)\n' +
            json.dumps(facts))
    sample = Sample(x['sample_id'], 'sequential', x['task'], x['split'], x['episode_id'],
                    x['initial_frame'], x['previous_frame'], x['current_frame'],
                    x['previous_progress'], x['previous_progress_provenance'], tuple(x['images']))
    validate_sample(sample)
    student = messages(sample)
    if student != x['student_messages']:
        raise ValueError(f"student message mismatch: {x['sample_id']}")
    body = json.dumps(student, ensure_ascii=False)
    if f"Current reference progress: {x['current_progress']}%" in body or 'TEACHER-ONLY' in body:
        raise ValueError(f"teacher leak: {x['sample_id']}")
    return dict(sample_id=x['sample_id'], kind=x['failure_type'], source=s,
                source_candidate=x, teacher_user=user, images=x['images'],
                student_messages=student)


def request_one(client, e, out, tok):
    # The saved six-example request helper preserves the exact wire format,
    # strict schema, idempotency header, and zero SDK retries. It never retries.
    raw, rec = base.request(client, e, out, 'initial')
    if raw is None:
        result = dict(sample_id=e['sample_id'], status=rec['status'], errors=[rec.get('error_type', 'request failed')],
                      completion=None, structured=None, usage=rec.get('usage'), qwen_tokens=None,
                      label_visual_conflict=None, request=str(out / 'requests' / f"{e['sample_id']}_initial.json"))
    else:
        obj, errors, rendered = validate(raw, e['source'], e['images'], tok)
        conflict = obj.get('label_visual_conflict') if obj else None
        status = 'structural_pass' if not errors and not conflict else ('label_visual_conflict' if conflict else 'format_failure')
        result = dict(sample_id=e['sample_id'], status=status, errors=errors,
                      completion=rendered['completion'] if rendered else None, structured=obj,
                      usage=rec.get('usage'), qwen_tokens=rendered['tokens'] if rendered else None,
                      label_visual_conflict=conflict, request=str(out / 'requests' / f"{e['sample_id']}_initial.json"))
    atomic(out / 'results' / f"{e['sample_id']}.json", result)
    return result


def export(rows, results, out, baseline):
    ref = json.loads((Path(baseline['snapshot']) / 'train.jsonl').open().readline())
    keys = list(ref)
    assert keys == baseline['sequential_schema_keys']
    chosen = {s: [] for s in ('train', 'validation', 'test')}
    audit, pending = [], []
    for x in rows:
        sid = x['sample_id']
        r = results.get(sid)
        if not r or r['status'] != 'structural_pass':
            pending.append(dict(sample_id=sid, failure_type=x['failure_type'], status=r['status'] if r else 'not_submitted'))
            continue
        split = x['split']
        if split not in chosen:
            pending.append(dict(sample_id=sid, failure_type=x['failure_type'], status='unassigned_split'))
            continue
        comp = r['completion']
        if not parse_completion(comp)['ok'] or parse_completion(comp)['progress'] != x['current_progress'] or check_progress_contract(comp, 'sequential', x['previous_progress']):
            pending.append(dict(sample_id=sid, failure_type=x['failure_type'], status='completion_contract_failure'))
            continue
        row = dict(sample_id=sid, mode='sequential', task=x['task'], split=split,
                   episode_id=x['episode_id'], initial_frame=x['initial_frame'],
                   previous_frame=x['previous_frame'], current_frame=x['current_frame'],
                   previous_progress=x['previous_progress'], answer=x['current_progress'],
                   source_jsonl_line=0, source_line_sha256=x['_source_line_sha256'],
                   is_initialization=False, completion=comp, status='structural_pass',
                   images=x['images'], previous_progress_provenance=x['previous_progress_provenance'],
                   prompt_version=STUDENT_VERSION, template_sha256=TEMPLATE_SHA256,
                   student_messages=x['student_messages'])
        if list(row) != keys or any(type(row[k]) is not type(ref[k]) for k in keys):
            raise ValueError(f'{sid}: success schema mismatch')
        chosen[split].append(row)
        audit.append(dict(sample_id=sid, failure_type=x['failure_type'], task_id=x['task_id'],
                          source_candidate_sha256=x['_source_line_sha256'], source_trajectory=x['trajectory_dir'],
                          request=r['request'], completion_sha256=hashlib.sha256(comp.encode()).hexdigest(),
                          label_visual_conflict=False, semantic_review='pending_individual_visual_review'))
    for split, vals in chosen.items():
        p = out / 'exports' / f'{split}.jsonl'; p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + '.partial')
        with tmp.open('w') as f:
            for v in vals: f.write(json.dumps(v, ensure_ascii=False) + '\n')
        os.replace(tmp, p)
        if vals:
            loaded = load_samples(p, validate_images=False)
            if len(loaded) != len(vals): raise ValueError(f'{split}: loader mismatch')
    for name, vals in [('teacher_audit', audit), ('pending', pending)]:
        p = out / 'exports' / f'{name}.jsonl'
        with p.open('w') as f:
            for v in vals: f.write(json.dumps(v, ensure_ascii=False) + '\n')
    return chosen, pending


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    if not 1 <= args.workers <= 4: raise ValueError('workers must be 1..4')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if not out.is_relative_to(ROOT / 'stage7_artifacts'): raise ValueError('output outside project artifacts')
    lock = (out / 'coordinator.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lines = [l for l in (SOURCE / 'frozen_116_candidates.jsonl').read_bytes().splitlines() if l.strip()]
    rows = [json.loads(l) for l in lines]
    assert len(rows) == 116 and len({x['sample_id'] for x in rows}) == 116
    assert Counter(x['failure_type'] for x in rows) == TYPES
    for x, l in zip(rows, lines): x['_source_line_sha256'] = hashlib.sha256(l + b'\n').hexdigest()
    tmpl = json.loads((SOURCE / 'templates.json').read_text())
    for code in ('run_short_failure_six_review.py', 'v7_short_failure_teacher.py', 'run_v7_grounded_contract.py'):
        assert digest(ROOT / 'stage7_tools' / code) == digest(SOURCE / 'code_snapshot' / code), code + ' drift'
    assert tmpl['version'] == six.VERSION and tmpl['system'] == six.SYSTEM and tmpl['schema'] == six.SCHEMA
    assert tmpl['student_version'] == STUDENT_VERSION and tmpl['student_template_sha256'] == TEMPLATE_SHA256
    base.SYSTEM, base.SCHEMA, base.TEACHER_VERSION, base.MODEL = tmpl['system'], tmpl['schema'], tmpl['version'], MODEL
    baseline = json.loads((ROOT / 'stage7_artifacts/v7_spatial_failure_production_20261025_r1/manifests/success_baseline.json').read_text())
    assert digest(Path(baseline['snapshot']) / 'manifest.json') == baseline['snapshot_manifest_sha256']
    atomic(out / 'run_config.json', dict(source=str(SOURCE / 'frozen_116_candidates.jsonl'), source_sha256=digest(SOURCE / 'frozen_116_candidates.jsonl'),
                                         template_sha256=digest(SOURCE / 'templates.json'), model=MODEL, workers=args.workers,
                                         max_output_tokens=2500, sdk_max_retries=0, repairs=0, retry_uncertain=False,
                                         baseline_manifest_sha256=baseline['snapshot_manifest_sha256']))
    atomic(out / 'templates.json', tmpl)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(base.MODEL_PATH), local_files_only=True)
    entries = {x['sample_id']: prepare(x) for x in rows}
    results = {}
    for x in rows:
        sid = x['sample_id']; p = out / 'results' / f'{sid}.json'
        if p.is_file(): results[sid] = json.loads(p.read_text())
        elif (out / 'requests' / f'{sid}_initial.json').is_file():
            rec = json.loads((out / 'requests' / f'{sid}_initial.json').read_text())
            results[sid] = dict(sample_id=sid, status='uncertain' if rec['status'] == 'started' else rec['status'],
                                request=str(out / 'requests' / f'{sid}_initial.json'), usage=rec.get('usage'))
        else:
            match = [(name, json.loads(p.read_text())) for name in TYPES for p in [(SOURCE / 'results' / f'{name}.json')] if p.is_file() and json.loads(p.read_text())['sample_id'] == sid]
            if match:
                name, old = match[0]; pre = json.loads((SOURCE / 'preflight' / f'{name}.json').read_text())
                if pre['source'] != source(x) or pre['images'] != x['images'] or old['status'] != 'structural_pass' or old['completion'] is None:
                    raise ValueError(f'{sid}: approved result mismatch')
                req = SOURCE / 'requests' / f'{sid}_initial.json'
                if not req.is_file() or json.loads(req.read_text()).get('teacher_version') != tmpl['version'] or pre['teacher_user'] != entries[sid]['teacher_user']:
                    raise ValueError(f'{sid}: approved request/template mismatch')
                results[sid] = dict(sample_id=sid, status='structural_pass', completion=old['completion'],
                                    structured=old.get('structured'), usage=old.get('usage'), qwen_tokens=old.get('qwen_tokens'),
                                    label_visual_conflict=old.get('structured', {}).get('label_visual_conflict'),
                                    request=str(req), reused_from=str(SOURCE / 'results' / f'{name}.json'),
                                    reused_sha256=digest(SOURCE / 'results' / f'{name}.json'))
                atomic(p, results[sid])
    atomic(out / 'reuse_manifest.json', {'items': [dict(sample_id=s, source=r.get('reused_from'), sha256=r.get('reused_sha256')) for s, r in results.items() if r.get('reused_from')]})
    todo = [entries[x['sample_id']] for x in rows if x['sample_id'] not in results]
    print(json.dumps({'reused': len(results), 'new_pending': len(todo)}, ensure_ascii=False), flush=True)
    if todo:
        import tomli
        from openai import OpenAI
        from urllib.parse import urlparse
        from api_settings import load_api_settings
        cfg = load_api_settings()
        client = OpenAI(api_key=Path(cfg['auth']['args'][0]).read_text().strip(), base_url=cfg['base_url'], timeout=180, max_retries=0)
        errors = 0; submitted = 0; completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            running = {}
            while todo or running:
                while todo and len(running) < args.workers and errors < 3:
                    e = todo.pop(0); running[pool.submit(request_one, client, e, out, tok)] = e['sample_id']; submitted += 1
                if not running: break
                done, _ = concurrent.futures.wait(running, return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done:
                    sid = running.pop(fut)
                    try: r = fut.result()
                    except Exception as ex:
                        r = dict(sample_id=sid, status='uncertain', errors=[type(ex).__name__ + ': ' + str(ex)[:300]], request=str(out / 'requests' / f'{sid}_initial.json'), usage=None)
                        atomic(out / 'results' / f'{sid}.json', r)
                    results[sid] = r; completed += 1
                    errors = errors + 1 if r['status'] in ('uncertain', 'request_failed') else 0
                    atomic(out / 'status.json', dict(submitted_new=submitted, completed_new=completed, consecutive_service_errors=errors,
                                                     remaining_unsubmitted=len(todo), status_counts=dict(Counter(z['status'] for z in results.values())), updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
                    print(json.dumps({'completed_new': completed, 'submitted_new': submitted, 'remaining': len(todo), 'status': r['status'], 'errors_in_row': errors}), flush=True)
    chosen, pending = export(rows, results, out, baseline)
    usages = [r.get('usage') for r in results.values() if r.get('usage') is not None]
    sums = {k: sum(u.get(k, 0) or 0 for u in usages) for k in ('input_tokens', 'output_tokens', 'total_tokens')}
    sums['reasoning_tokens'] = sum((u.get('output_tokens_details') or {}).get('reasoning_tokens', 0) or 0 for u in usages)
    sums['cached_input_tokens'] = sum((u.get('input_tokens_details') or {}).get('cached_tokens', 0) or 0 for u in usages)
    report = dict(total=116, reused=sum(bool(r.get('reused_from')) for r in results.values()),
                  submitted_new=sum((out / 'requests' / f"{x['sample_id']}_initial.json").exists() for x in rows if x['sample_id'] in results and not results[x['sample_id']].get('reused_from')),
                  statuses=dict(Counter(r['status'] for r in results.values())), not_submitted=116-len(results),
                  exported={k: len(v) for k,v in chosen.items()}, exported_by_type=dict(Counter(a['failure_type'] for a in [json.loads(l) for l in (out/'exports/teacher_audit.jsonl').read_text().splitlines() if l.strip()])),
                  usage_known=sums, usage_unknown=sum(r.get('usage') is None for r in results.values()), pending=len(pending),
                  semantic_review='pending_individual_visual_review', no_training=True)
    atomic(out / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
