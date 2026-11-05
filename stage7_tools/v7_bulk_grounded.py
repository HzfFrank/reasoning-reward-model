#!/usr/bin/env python3
"""V3-aligned bulk generation with the approved grounded v7 contract."""
from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
import signal
import sys
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import v7_bulk_api as legacy
from v7_grounded_contract_templates import REPAIR, SCHEMA, SYSTEM, VERSION as TEACHER_VERSION

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'stage7_sft'))
from prompts import TEMPLATE_SHA256, VERSION as STUDENT_VERSION  # noqa: E402
from protocol import Sample, check_progress_contract, load_samples, messages, validate_sample  # noqa: E402
from run_v7_grounded_contract import validate as validate_teacher  # noqa: E402

VERSION = 'v7_bulk_grounded_contract_v1'
MODEL = 'gpt-6.1-sol'
MAX_OUTPUT_TOKENS = 2500
EXPECTED = 14046
STOP = threading.Event()


def source_id(row: dict) -> str:
    return f"{row['episode_id']}_{row['mode']}_{row['current_frame']}_line{row['source_jsonl_line']}"


def lean(row: dict, manifest: dict) -> dict:
    keys = ('source_jsonl_line', 'source_line_sha256', 'episode_id', 'mode', 'task',
            'current_frame', 'previous_frame', 'previous_progress', 'answer', 'split',
            'dataset_version', 'image_path', 'is_initialization')
    result = {key: row[key] for key in keys}
    result.update(sample_id=source_id(row), initial_frame=manifest['storyboard_frames'][0],
                  storyboard_frames=manifest['storyboard_frames'])
    return result


def scan(out: Path):
    rows, manifests, rejects = legacy.scan_source(out)
    totals = json.loads((out / 'source_scan.json').read_text())['totals']
    split = json.loads(legacy.SPLIT.read_text())['episode_splits']
    errors = []
    if (totals['rows'], totals['success_rows'], totals['sequential'], totals['from_zero']) != (14046, 14046, 10927, 3119):
        errors.append('source row or mode counts differ')
    if len(manifests) != 404 or len(rows) != EXPECTED or rejects:
        errors.append(f'episode/usable/rejected counts: {len(manifests)}/{len(rows)}/{len(rejects)}')
    if split.get(legacy.EP41) != 'test': errors.append('episode 41 split is not test')
    if any(manifests[r['episode_id']]['status'] != 'ok' or r['split'] != split[r['episode_id']] for r in rows):
        errors.append('rollout pairing or split mismatch')
    source_sha = legacy.sha(legacy.V3)
    checks = {'version': VERSION, 'source': str(legacy.V3), 'source_sha256': source_sha,
              'split_source': str(legacy.SPLIT), 'split_sha256': legacy.sha(legacy.SPLIT),
              'totals': totals, 'success_episodes': len(manifests), 'usable_rows': len(rows),
              'rejects': len(rejects), 'errors': errors, 'status': 'blocked' if errors else 'pass'}
    legacy.atomic_json(out / 'grounded_source_scan.json', checks)
    if errors: raise RuntimeError('; '.join(errors))
    return [lean(r, manifests[r['episode_id']]) for r in rows], manifests, checks


def manifest(out: Path, scan_report: dict) -> dict:
    prompt_file = ROOT / 'stage7_tools/v7_grounded_contract_templates.py'
    renderer_file = ROOT / 'stage7_tools/run_v7_grounded_contract.py'
    value = {'version': VERSION, 'model': MODEL, 'teacher_version': TEACHER_VERSION,
             'teacher_template_sha256': legacy.sha(prompt_file), 'teacher_system_sha256': hashlib.sha256(SYSTEM.encode()).hexdigest(),
             'schema_sha256': hashlib.sha256(json.dumps(SCHEMA, sort_keys=True).encode()).hexdigest(),
             'renderer_source_sha256': legacy.sha(renderer_file), 'student_prompt_version': STUDENT_VERSION,
             'student_template_sha256': TEMPLATE_SHA256, 'source_sha256': scan_report['source_sha256'],
             'source': scan_report['source'], 'split_source': scan_report['split_source'],
             'split_sha256': scan_report['split_sha256'], 'expected_rows': EXPECTED,
             'service_host': __import__('urllib.parse', fromlist=['urlparse']).urlparse(os.environ.get('V7_API_BASE_URL', '')).hostname, 'request': {'max_output_tokens': MAX_OUTPUT_TOKENS,
             'response_format': 'strict_json_schema', 'sdk_retries': 0, 'workers': 4},
             'images': {'storyboard': 'round(i*(N-1)/7), i=0..7 on paired frame IDs',
                        'order': 'GLOBAL main/wrist, INITIAL main/wrist, optional PREVIOUS main/wrist, CURRENT main/wrist',
                        'teacher': 'original rollout PNG bytes; no resize', 'student_image_pixels': 65536}}
    path = out / 'run_manifest.json'
    if path.exists():
        existing = json.loads(path.read_text())
        comparable_existing = dict(existing)
        comparable_new = dict(value)
        # The old mount is a symlink to the same project after the host move.
        # Preserve the original manifest while checking every non-path field exactly.
        for key in ('source', 'split_source'):
            comparable_existing[key] = str(Path(comparable_existing[key]).resolve(strict=True))
            comparable_new[key] = str(Path(comparable_new[key]).resolve(strict=True))
        if comparable_existing != comparable_new:
            raise ValueError('run manifest differs from this code/source/contract; use a new output directory')
    if not path.exists(): legacy.atomic_json(path, value)
    return value


def image_manifest(row: dict, episode: dict, cache: dict) -> list[dict]:
    ep = row['episode_id']
    if ep not in cache:
        base = Path(episode['rollout_dir']) / 'images'
        cache[ep] = {view: {legacy.frame_id(p): p for p in (base / view).glob('*.png')} for view in ('main', 'wrist')}
    known = {(im['frame'], im['view']): im for im in episode['images']}
    result = []
    def add(frame: int, role: str):
        for view in ('main', 'wrist'):
            path = cache[ep][view].get(frame)
            if path is None: raise FileNotFoundError(f'{ep} frame={frame} view={view}')
            expected = known.get((frame, view))
            digest = expected['sha256'] if expected else legacy.sha(path)
            result.append({'path': str(path.resolve()), 'sha256': digest, 'frame': frame,
                           'view': view, 'role': role.lower(),
                           'label': f'{role} frame={frame} view={view} role={role.lower()}'})
    for frame in episode['storyboard_frames']: add(frame, 'GLOBAL')
    add(row['initial_frame'], 'INITIAL')
    if row['mode'] == 'sequential': add(row['previous_frame'], 'PREVIOUS')
    add(row['current_frame'], 'CURRENT')
    if len(result) != (22 if row['mode'] == 'sequential' else 20): raise ValueError('image count mismatch')
    return result


def teacher_user(row: dict, images: list[dict]) -> str:
    text = (f"Mode: {row['mode'].upper()}\nTask instruction: {row['task']}\n"
            f"INITIAL frame: {row['initial_frame']}\nCURRENT frame: {row['current_frame']}\n")
    if row['mode'] == 'sequential': text += f"PREVIOUS frame: {row['previous_frame']}\n"
    text += '\nVISUAL INPUT\n' + '\n'.join(im['label'] for im in images)
    text += (f"\n\nOFFLINE ANNOTATION REFERENCE - NOT PART OF THE STUDENT INPUT\n"
             f"Initial progress baseline: 0%\nCurrent reference progress: {row['answer']}%\n")
    if row['mode'] == 'sequential':
        text += (f"Previous reference progress: {row['previous_progress']}%\n"
                 f"Reference change: {row['answer'] - row['previous_progress']} percentage points\n")
    return text + '\nGenerate the required JSON explanation.'


def preflight(rows: list[dict], manifests: dict, out: Path) -> list[dict]:
    by_key = {(r['episode_id'], r['mode'], r['current_frame']): r for r in rows}
    episode_rows = [r for r in rows if r['episode_id'] == legacy.EP41]
    initialization = next((r for r in episode_rows if r['mode'] == 'sequential' and r['is_initialization']), None)
    if initialization is None: raise ValueError('episode 41 sequential initialization missing')
    other = next(r for r in rows if r['episode_id'] != legacy.EP41 and r['mode'] == 'sequential' and r['current_frame'] > 10)
    other_zero = next(r for r in rows if r['episode_id'] == other['episode_id'] and r['mode'] == 'from_zero')
    chosen = [initialization] + [by_key[(legacy.EP41, mode, current)] for mode, current in
                                  (('sequential', 4), ('sequential', 50), ('sequential', 95),
                                   ('from_zero', 4), ('from_zero', 95))] + [other, other_zero]
    if len({r['sample_id'] for r in chosen}) != 8: raise ValueError('smoke selection has duplicates')
    cache = {}; checks = []
    for row in chosen:
        images = image_manifest(row, manifests[row['episode_id']], cache)
        sample = Sample(row['sample_id'], row['mode'], row['task'], row['split'], row['episode_id'],
                        row['initial_frame'], row['previous_frame'], row['current_frame'],
                        row['previous_progress'], 'source_previous_progress' if row['mode'] == 'sequential' else None,
                        tuple(images), is_initialization=row['is_initialization'])
        validate_sample(sample)
        user = json.dumps(messages(sample), ensure_ascii=False)
        if any(term in user for term in ('Current reference progress', 'Reference change', 'evidence_frames', 'label_visual_conflict')):
            raise ValueError('teacher field leaked into student messages')
        checks.append({'sample_id': row['sample_id'], 'source_jsonl_line': row['source_jsonl_line'],
                       'mode': row['mode'], 'current': row['current_frame'], 'images': len(images)})
    legacy.atomic_json(out / 'preflight.json', {'version': VERSION, 'status': 'pass', 'samples': checks})
    return chosen


def local_check(rows: list[dict], manifests: dict, out: Path):
    from transformers import AutoTokenizer
    from run_v7_grounded_contract import teacher_user as reference_user
    tokenizer = AutoTokenizer.from_pretrained(str(legacy.MODEL_PATH), local_files_only=True)
    lookup = {(r['episode_id'], r['mode'], r['current_frame']): r for r in rows}
    accepted_dir = ROOT / 'tests/fixtures/approved_success'
    checks = []; cache = {}
    for name, mode, current in (('sequential_50', 'sequential', 50), ('sequential_95', 'sequential', 95),
                                ('from_zero_4', 'from_zero', 4), ('from_zero_95', 'from_zero', 95)):
        row = lookup[(legacy.EP41, mode, current)]
        images = image_manifest(row, manifests[row['episode_id']], cache)
        prior = json.loads((accepted_dir / (name + '.json')).read_text())
        source = {'mode': mode, 'task': row['task'], 'previous': row['previous_frame'], 'current': current,
                  'previous_progress': row['previous_progress'], 'answer': row['answer']}
        if teacher_user(row, images) != reference_user(source, images):
            raise ValueError(f'{name}: teacher user differs from approved generator')
        if [{k:v for k,v in im.items() if k != 'path'} for im in images] != prior['images']:
            raise ValueError(f'{name}: images differ from approved generator')
        obj, errors, rendered = validate_teacher(prior['raw_response'], source, images, tokenizer)
        if errors or obj != prior['structured'] or rendered['completion'] != prior['completion']:
            raise ValueError(f'{name}: validation/rendering differs from approved example: {errors}')
        checks.append({'sample_id': row['sample_id'], 'mode': mode, 'images': len(images),
                       'completion_tokens': rendered['tokens'], 'status': 'pass'})
    value = {'version': VERSION, 'status': 'pass', 'checks': checks,
             'note': 'Existing approved responses were validated locally; no API call or result reuse.'}
    legacy.atomic_json(out / 'local_contract_check.json', value)
    return value


def api_client():
    import tomli
    from openai import OpenAI
    from api_settings import load_api_settings
    cfg = load_api_settings()
    client = OpenAI(api_key=Path(cfg['auth']['args'][0]).read_text().strip(),
                    base_url=cfg['base_url'], timeout=180, max_retries=0)
    available = {entry.id for entry in client.models.list().data}
    if MODEL not in available: raise RuntimeError(f'model {MODEL} unavailable; no fallback')
    return client


def wire(user: str, images: list[dict]) -> list[dict]:
    content = [{'type': 'input_text', 'text': user}]
    for im in images:
        path = Path(im['path'])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != im['sha256']: raise ValueError(f'image changed: {path}')
        content.extend(({'type': 'input_text', 'text': im['label']},
                        {'type': 'input_image', 'image_url': 'data:image/png;base64,' + base64.b64encode(raw).decode()}))
    return [{'role': 'system', 'content': [{'type': 'input_text', 'text': SYSTEM}]},
            {'role': 'user', 'content': content}]


def request(client, row: dict, images: list[dict], out: Path, kind: str,
            previous_raw: str | None = None, errors: list[str] | None = None,
            max_attempts: int = 2):
    user = teacher_user(row, images)
    extra = REPAIR.format(errors=json.dumps(errors, ensure_ascii=False), raw=previous_raw) if kind == 'repair' else None
    path = out / 'requests' / f"{row['sample_id']}_{kind}.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old.get('status') == 'response_saved': return old['raw_response'], old
        return None, old
    rec = {'sample_id': row['sample_id'], 'source_jsonl_line': row['source_jsonl_line'],
           'source_line_sha256': row['source_line_sha256'], 'kind': kind, 'model': MODEL,
           'teacher_version': TEACHER_VERSION, 'request_id': str(uuid.uuid4()),
           'messages_text': [{'role': 'system', 'text': SYSTEM}, {'role': 'user', 'text': user}]
           + ([{'role': 'user', 'text': extra}] if extra else []),
           'images': images, 'max_output_tokens': MAX_OUTPUT_TOKENS, 'attempts': [], 'status': 'started'}
    legacy.atomic_json(path, rec)
    content = wire(user, images)
    if extra: content.append({'role': 'user', 'content': [{'type': 'input_text', 'text': extra}]})
    fmt = {'type': 'json_schema', 'name': 'v7_grounded_contract', 'schema': SCHEMA, 'strict': True}
    for attempt in range(max_attempts):
        rec['attempts'].append({'number': attempt + 1, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        legacy.atomic_json(path, rec)
        start = time.monotonic()
        try:
            response = client.responses.create(model=MODEL, input=content, max_output_tokens=MAX_OUTPUT_TOKENS,
                                               text={'format': fmt}, extra_headers={'Idempotency-Key': rec['request_id']})
            usage = response.usage.model_dump() if getattr(response, 'usage', None) else None
            actual_model = getattr(response, 'model', None)
            if actual_model and actual_model != MODEL: raise RuntimeError('service returned a different model ID')
            raw = getattr(response, 'output_text', '') or ''
            rec['attempts'][-1].update(status='response', seconds=round(time.monotonic() - start, 3),
                                       response_id=getattr(response, 'id', None), usage=usage,
                                       returned_model=actual_model)
            rec.update(status='response_saved', raw_response=raw)
            legacy.atomic_json(path, rec)
            return raw, rec
        except Exception as exc:
            # A timeout or connection loss may occur after service execution; do not resend it.
            status = getattr(exc, 'status_code', None)
            definite_rejection = status == 429
            rec['attempts'][-1].update(status='error', seconds=round(time.monotonic() - start, 3),
                                       error_type=type(exc).__name__, http_status=status)
            rec['status'] = 'rate_limited' if definite_rejection else ('request_failed' if status and status < 500 else 'uncertain')
            legacy.atomic_json(path, rec)
            if definite_rejection and attempt + 1 < max_attempts:
                time.sleep(15 * (attempt + 1))
                rec['status'] = 'started'
                legacy.atomic_json(path, rec)
                continue
            return None, rec
    return None, rec


def process_one(client, row: dict, episode: dict, out: Path, tokenizer, cache: dict) -> dict:
    path = out / 'results' / f"{row['sample_id']}.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get('source_line_sha256') != row['source_line_sha256'] or existing.get('teacher_version') != TEACHER_VERSION:
            raise ValueError(f'result source/contract mismatch: {row["sample_id"]}')
        return existing
    images = image_manifest(row, episode, cache)
    original, req = request(client, row, images, out, 'initial')
    response = original
    repair = None
    validation_errors = []
    obj = None
    rendered = None
    if original is not None:
        obj, validation_errors, rendered = validate_teacher(original, {
            'mode': row['mode'], 'current': row['current_frame'], 'previous_progress': row['previous_progress'],
            'answer': row['answer']}, images, tokenizer)
        if rendered and not validation_errors:
            validation_errors += check_progress_contract(rendered['completion'], row['mode'], row['previous_progress'])
        if validation_errors and req['status'] == 'response_saved':
            repair_raw, repair = request(client, row, images, out, 'repair', original, validation_errors,
                                         max_attempts=max(0, 3 - len(req['attempts'])))
            if repair_raw is not None:
                response = repair_raw
                obj, validation_errors, rendered = validate_teacher(repair_raw, {
                    'mode': row['mode'], 'current': row['current_frame'], 'previous_progress': row['previous_progress'],
                    'answer': row['answer']}, images, tokenizer)
                if rendered and not validation_errors:
                    validation_errors += check_progress_contract(rendered['completion'], row['mode'], row['previous_progress'])
    if original is None:
        status = 'uncertain' if req['status'] == 'uncertain' or req['status'] == 'started' else 'request_failed'
    elif repair is not None and repair.get('status') != 'response_saved':
        status = 'uncertain' if repair['status'] in ('uncertain', 'started') else 'request_failed'
    elif validation_errors:
        status = 'rejected'
    elif obj.get('label_visual_conflict'):
        status = 'pending_review'
    else:
        status = 'accepted'
    record = {**row, 'status': status, 'teacher_version': TEACHER_VERSION,
              'teacher_template_sha256': legacy.sha(ROOT / 'stage7_tools/v7_grounded_contract_templates.py'),
              'structured_teacher_output': obj, 'completion': rendered['completion'] if rendered else None,
              'qwen_completion_tokens': rendered['tokens'] if rendered else None,
              'validation_errors': validation_errors, 'label_visual_conflict': bool(obj and obj.get('label_visual_conflict')),
              'semantic_review': 'pending_manual_review', 'image_manifest': images,
              'initial_request': str(out / 'requests' / f"{row['sample_id']}_initial.json"),
              'repair_request': str(out / 'requests' / f"{row['sample_id']}_repair.json") if repair else None,
              'raw_response': response, 'request_attempts': len(req['attempts']) + (len(repair['attempts']) if repair else 0),
              'request_usage': [a['usage'] for item in (req, repair) if item for a in item['attempts'] if a.get('usage')],
              'returned_models': [a['returned_model'] for item in (req, repair) if item for a in item['attempts'] if a.get('returned_model')]}
    legacy.atomic_json(path, record)
    return record


def usage_stats(out: Path):
    totals = {'initial_requests': 0, 'transport_retries': 0, 'repairs': 0,
              'input_tokens': 0, 'cached_input_tokens': 0, 'output_tokens': 0,
              'reasoning_tokens': 0, 'unknown_usage': 0}
    for path in (out / 'requests').glob('*.json'):
        rec = json.loads(path.read_text())
        totals['initial_requests' if rec['kind'] == 'initial' else 'repairs'] += int(bool(rec['attempts']))
        totals['transport_retries'] += max(0, len(rec['attempts']) - 1)
        for attempt in rec['attempts']:
            usage = attempt.get('usage')
            if usage is None:
                totals['unknown_usage'] += 1
                continue
            totals['input_tokens'] += usage.get('input_tokens', 0)
            totals['output_tokens'] += usage.get('output_tokens', 0)
            totals['cached_input_tokens'] += (usage.get('input_tokens_details') or {}).get('cached_tokens', 0)
            totals['reasoning_tokens'] += (usage.get('output_tokens_details') or {}).get('reasoning_tokens', 0)
    return totals


def result_stats(out: Path):
    result = Counter()
    for path in (out / 'results').glob('*.json'):
        try: result[json.loads(path.read_text())['status']] += 1
        except (ValueError, KeyError): result['invalid_result'] += 1
    result['pending'] = EXPECTED - sum(result.values())
    return dict(result)


def progress(out: Path, started: float, new_count: int, phase: str):
    counts = result_stats(out)
    legacy.atomic_json(out / 'progress.json', {'version': VERSION, 'phase': phase,
        'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'counts': counts, 'usage': usage_stats(out), 'new_results_this_process': new_count,
        'new_results_per_hour': round(new_count / max((time.monotonic() - started) / 3600, 1e-6), 2)})


def export_snapshot(out: Path, rows: list[dict], manifests: dict) -> Path:
    accepted = []
    for path in (out / 'results').glob('*.json'):
        rec = json.loads(path.read_text())
        if rec['status'] == 'accepted': accepted.append(rec)
    accepted.sort(key=lambda x: x['source_jsonl_line'])
    name = f'snapshot_{len(accepted):08d}'
    final = out / 'exports' / name
    if final.exists(): return final
    building = out / 'exports' / (name + '.building')
    building.mkdir(parents=True, exist_ok=False)
    by_split = {split: [] for split in ('train', 'validation', 'test')}
    audit = []
    for rec in accepted:
        sample = Sample(rec['sample_id'], rec['mode'], rec['task'], rec['split'], rec['episode_id'],
            rec['initial_frame'], rec['previous_frame'], rec['current_frame'], rec['previous_progress'],
            'source_previous_progress' if rec['mode'] == 'sequential' else None,
            tuple(rec['image_manifest']), rec['completion'], str(building / (rec['split'] + '.jsonl')),
            rec['is_initialization'])
        validate_sample(sample)
        if check_progress_contract(sample.completion, sample.mode, sample.previous_progress):
            raise ValueError(f'completion contract mismatch in accepted sample {sample.sample_id}')
        student = messages(sample)
        row = {key: rec[key] for key in ('sample_id', 'mode', 'task', 'split', 'episode_id',
            'initial_frame', 'previous_frame', 'current_frame', 'previous_progress', 'answer',
            'source_jsonl_line', 'source_line_sha256', 'is_initialization', 'completion')}
        row.update(status='structural_pass', images=rec['image_manifest'],
                   previous_progress_provenance='source_previous_progress' if sample.mode == 'sequential' else None,
                   prompt_version=STUDENT_VERSION, template_sha256=TEMPLATE_SHA256,
                   student_messages=student + [{'role': 'assistant', 'content': [{'type': 'text', 'text': sample.completion}]}])
        by_split[rec['split']].append(row)
        audit.append({key: rec.get(key) for key in ('sample_id', 'source_jsonl_line', 'source_line_sha256',
            'structured_teacher_output', 'initial_request', 'repair_request', 'raw_response',
            'validation_errors', 'label_visual_conflict', 'qwen_completion_tokens', 'semantic_review',
            'request_usage', 'returned_models', 'request_attempts')})
    for split, split_rows in by_split.items():
        (building / (split + '.jsonl')).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in split_rows))
    (building / 'audit.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in audit))
    meta = {'version': VERSION, 'teacher_version': TEACHER_VERSION, 'student_prompt_version': STUDENT_VERSION,
            'student_template_sha256': TEMPLATE_SHA256, 'source_sha256': legacy.sha(legacy.V3),
            'split_source': str(legacy.SPLIT), 'partial': len(accepted) != EXPECTED,
            'expected': EXPECTED, 'accepted': len(accepted), 'counts': {k: len(v) for k,v in by_split.items()},
            'episodes': {k: len({r['episode_id'] for r in v}) for k,v in by_split.items()},
            'sample_ids': [r['sample_id'] for r in accepted], 'semantic_review': 'pending_manual_review'}
    legacy.atomic_json(building / 'manifest.json', meta)
    os.replace(building, final)
    legacy.atomic_json(out / 'latest_snapshot.json', {'path': str(final), 'accepted': len(accepted), 'counts': meta['counts']})
    return final


def check_smoke_export(path: Path):
    from processing import encode_sample, load_processor
    samples = []
    for split in ('train', 'validation', 'test'):
        file = path / (split + '.jsonl')
        if file.stat().st_size: samples.extend(load_samples(file))
    modes = {mode: next((s for s in samples if s.mode == mode), None) for mode in ('sequential', 'from_zero')}
    if any(sample is None for sample in modes.values()): raise RuntimeError('smoke export lacks an accepted sample in either mode')
    processor = load_processor(str(legacy.MODEL_PATH), 65536)
    summary = {}
    for mode, sample in modes.items():
        encoded = encode_sample(sample, processor, 8192, training=True)
        prompt_len = encoded['prompt_tokens']; labels = encoded['tensors']['labels'][0]
        if not labels[:prompt_len].eq(-100).all() or not labels[prompt_len:].ne(-100).all():
            raise RuntimeError(f'{mode} label mask mismatch')
        if encoded['images'] != (22 if mode == 'sequential' else 20):
            raise RuntimeError(f'{mode} image count mismatch')
        summary[mode] = {'sample_id': sample.sample_id, 'images': encoded['images'],
                         'grid_count': int(encoded['tensors']['image_grid_thw'].shape[0]),
                         'prompt_tokens': prompt_len, 'completion_tokens': encoded['completion_tokens']}
    return summary


def choose_smoke(rows: list[dict]) -> list[dict]:
    by_key = {(r['episode_id'], r['mode'], r['current_frame']): r for r in rows}
    ep = [r for r in rows if r['episode_id'] == legacy.EP41]
    init = next(r for r in ep if r['mode'] == 'sequential' and r['is_initialization'])
    other = next(r for r in rows if r['episode_id'] != legacy.EP41 and r['mode'] == 'sequential' and r['current_frame'] > 10)
    other_zero = next(r for r in rows if r['episode_id'] == other['episode_id'] and r['mode'] == 'from_zero')
    return [init] + [by_key[(legacy.EP41, mode, current)] for mode, current in
        (('sequential', 4), ('sequential', 50), ('sequential', 95), ('from_zero', 4), ('from_zero', 95))] + [other, other_zero]


def generate(out: Path, rows: list[dict], manifests: dict, client, tokenizer, workers: int):
    started = time.monotonic()
    smoke = choose_smoke(rows)
    cache = {}
    new_smoke = 0
    for row in smoke:
        if STOP.is_set(): break
        new_smoke += int(not (out / 'results' / (row['sample_id'] + '.json')).exists())
        process_one(client, row, manifests[row['episode_id']], out, tokenizer, cache)
        progress(out, started, new_smoke, 'smoke')
    if STOP.is_set(): return
    smoke_records = [json.loads((out / 'results' / (row['sample_id'] + '.json')).read_text()) for row in smoke]
    accepted = [r for r in smoke_records if r['status'] == 'accepted']
    if len(accepted) < 6 or {r['mode'] for r in accepted} != {'sequential', 'from_zero'}:
        legacy.atomic_json(out / 'smoke_check.json', {'status': 'blocked', 'counts': dict(Counter(r['status'] for r in smoke_records))})
        return
    snapshot = export_snapshot(out, rows, manifests)
    encoded = check_smoke_export(snapshot)
    legacy.atomic_json(out / 'smoke_check.json', {'status': 'pass', 'counts': dict(Counter(r['status'] for r in smoke_records)),
                                                 'snapshot': str(snapshot), 'processor': encoded})
    progress(out, started, new_smoke, 'bulk')

    remaining = (r for r in rows if not (out / 'results' / (r['sample_id'] + '.json')).exists())
    active = {}; new_count = 0; last_snapshot = result_stats(out).get('accepted', 0)
    accepted_total = last_snapshot; consecutive_errors = 0; repeated_contract_errors = 0
    last_error = None; inflated_tokens = 0
    smoke_inputs = [usage['input_tokens'] for rec in smoke_records for usage in rec.get('request_usage', [])
                    if isinstance(usage.get('input_tokens'), int)]
    baseline_input = sorted(smoke_inputs)[len(smoke_inputs)//2] if smoke_inputs else None
    def submit(pool):
        if STOP.is_set(): return False
        try: row = next(remaining)
        except StopIteration: return False
        future = pool.submit(process_one, client, row, manifests[row['episode_id']], out, tokenizer, cache)
        active[future] = row
        return True
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in range(workers): submit(pool)
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                row = active.pop(future)
                try: rec = future.result()
                except Exception as exc:
                    rec = {**row, 'status': 'request_failed', 'teacher_version': TEACHER_VERSION,
                           'validation_errors': [type(exc).__name__], 'semantic_review': 'pending_manual_review'}
                    legacy.atomic_json(out / 'results' / (row['sample_id'] + '.json'), rec)
                new_count += 1
                consecutive_errors = consecutive_errors + 1 if rec['status'] in ('request_failed', 'uncertain') else 0
                accepted_total += int(rec['status'] == 'accepted')
                error = tuple(rec.get('validation_errors', [])) if rec['status'] == 'rejected' else None
                repeated_contract_errors = repeated_contract_errors + 1 if error and error == last_error else int(bool(error))
                last_error = error
                observed_input = [u.get('input_tokens') for u in rec.get('request_usage', [])]
                inflated_tokens = inflated_tokens + 1 if baseline_input and any(
                    isinstance(n, int) and n > 3 * baseline_input for n in observed_input) else 0
                access_denied = False
                if rec['status'] == 'request_failed':
                    for key in ('initial_request', 'repair_request'):
                        request_path = rec.get(key)
                        if request_path and Path(request_path).exists():
                            request_record = json.loads(Path(request_path).read_text())
                            access_denied |= any(attempt.get('http_status') in (401, 403)
                                                 for attempt in request_record.get('attempts', []))
                if new_count % 50 == 0 or STOP.is_set(): progress(out, started, new_count + new_smoke, 'bulk')
                if accepted_total - last_snapshot >= 1000:
                    export_snapshot(out, rows, manifests)
                    last_snapshot = accepted_total
                if not STOP.is_set() and (access_denied or consecutive_errors >= 8 or
                                          repeated_contract_errors >= 8 or inflated_tokens >= 3):
                    STOP.set()
                    reason = ('api access denied (401/403)' if access_denied else
                              'repeated request errors' if consecutive_errors >= 8 else
                              'repeated contract errors' if repeated_contract_errors >= 8 else 'input token inflation')
                    legacy.atomic_json(out / 'auto_pause.json', {'reason': reason,
                                                                'last_sample_id': row['sample_id']})
                if not STOP.is_set(): submit(pool)
    progress(out, started, new_count + new_smoke, 'paused' if STOP.is_set() else 'complete')
    if not STOP.is_set(): export_snapshot(out, rows, manifests)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--scan-only', action='store_true')
    parser.add_argument('--check-model', action='store_true')
    parser.add_argument('--local-check', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 32: raise ValueError('workers must be 1..32')
    out = Path(args.output_dir).resolve()
    from portable_paths import check_output
    check_output(out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.scan_only and not args.check_model and not args.local_check and not args.status and not args.run:
        raise ValueError('choose --scan-only, --local-check, --check-model, --status, or --run')
    if args.status:
        lock = out / legacy.LOCK_NAME
        owner = json.loads(lock.read_text()) if lock.exists() else {}
        pid = owner.get('pid')
        try: os.kill(pid, 0); active = True
        except (TypeError, ProcessLookupError, PermissionError): active = False
        latest = json.loads((out / 'latest_snapshot.json').read_text()) if (out / 'latest_snapshot.json').exists() else None
        print(json.dumps({'active': active, 'pid': pid if active else None,
                          'counts': result_stats(out), 'usage': usage_stats(out), 'latest_snapshot': latest}, ensure_ascii=False))
        return
    if args.scan_only:
        rows, manifests, report = scan(out)
        preflight(rows, manifests, out)
        manifest(out, report)
        print(json.dumps({'status': report['status'], 'rows': len(rows), 'episodes': len(manifests)}))
        return
    if args.check_model:
        api_client()
        print(json.dumps({'status': 'available', 'model': MODEL, 'service_host': __import__('urllib.parse', fromlist=['urlparse']).urlparse(os.environ.get('V7_API_BASE_URL', '')).hostname}))
        return
    if args.local_check:
        rows, manifests, report = scan(out)
        preflight(rows, manifests, out)
        manifest(out, report)
        print(json.dumps(local_check(rows, manifests, out), ensure_ascii=False))
        return
    lock_fd = legacy.acquire_lock(out)
    def stop_handler(signum, frame): STOP.set()
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    try:
        rows, manifests, report = scan(out)
        preflight(rows, manifests, out)
        manifest(out, report)
        if not (out / 'local_contract_check.json').exists() or json.loads((out / 'local_contract_check.json').read_text()).get('status') != 'pass':
            raise RuntimeError('local contract check must pass before API generation')
        import transformers
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(legacy.MODEL_PATH), local_files_only=True)
        client = api_client()
        generate(out, rows, manifests, client, tokenizer, args.workers)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


if __name__ == '__main__': main()
