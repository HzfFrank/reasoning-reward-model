"""One authorized retry for the two uncertain frozen CoT candidates.

Writes a new run; never edits the original 116-candidate run or its requests.
"""
from __future__ import annotations

import fcntl
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'stage7_tools'), str(ROOT / 'stage7_sft')]
import run_event_aligned_116_cot_bulk as bulk
import run_short_failure_six_review as six
import run_v7_geometry_v2_cot_examples as base

OLD = ROOT / 'stage7_artifacts/v7_event_aligned_cot_bulk_20261027_r1'
OUT = ROOT / 'stage7_artifacts/v7_event_aligned_cot_retry_20261028_r1'
IDS = {'failure_short_ceaa07e8d0327b7e9019', 'failure_short_554b74fc28e9011e5dbe'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'coordinator.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = [json.loads(line) for line in (bulk.SOURCE / 'frozen_116_candidates.jsonl').read_text().splitlines() if line.strip()]
    assert len(rows) == 116 and {x['sample_id'] for x in rows if x['sample_id'] in IDS} == IDS
    old_config = json.loads((OLD / 'run_config.json').read_text())
    assert old_config['source_sha256'] == bulk.digest(bulk.SOURCE / 'frozen_116_candidates.jsonl')
    tmpl = json.loads((bulk.SOURCE / 'templates.json').read_text())
    assert tmpl['version'] == six.VERSION and tmpl['system'] == six.SYSTEM and tmpl['schema'] == six.SCHEMA
    assert old_config['template_sha256'] == bulk.digest(bulk.SOURCE / 'templates.json')
    base.SYSTEM, base.SCHEMA, base.TEACHER_VERSION, base.MODEL = tmpl['system'], tmpl['schema'], tmpl['version'], bulk.MODEL
    bulk.atomic(OUT / 'templates.json', tmpl)
    bulk.atomic(OUT / 'retry_manifest.json', {
        'source_run': str(OLD), 'source_input_sha256': old_config['source_sha256'],
        'source_template_sha256': old_config['template_sha256'], 'retry_ids': sorted(IDS),
        'reason': 'explicit user request to retry the two uncertain calls',
        'new_requests_per_id_limit': 1, 'sdk_max_retries': 0, 'max_output_tokens': 2500,
        'model': bulk.MODEL, 'possible_prior_unknown_usage': 2,
    })
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(base.MODEL_PATH), local_files_only=True)
    results = {x['sample_id']: json.loads((OLD / 'results' / f"{x['sample_id']}.json").read_text()) for x in rows}
    for sid in sorted(IDS):
        assert results[sid]['status'] == 'uncertain'
        req_path = OUT / 'requests' / f'{sid}_initial.json'
        result_path = OUT / 'results' / f'{sid}.json'
        if result_path.exists():
            results[sid] = json.loads(result_path.read_text())
            print(json.dumps({'sample_id': sid, 'status': results[sid]['status'], 'reused_retry_record': True}), flush=True)
            continue
        if req_path.exists():
            rec = json.loads(req_path.read_text())
            results[sid] = dict(sample_id=sid, status='uncertain', usage=rec.get('usage'),
                                request=str(req_path), error='retry request already started; no automatic resend')
            print(json.dumps({'sample_id': sid, 'status': 'uncertain', 'reused_started_record': True}), flush=True)
            continue
        x = next(x for x in rows if x['sample_id'] == sid)
        entry = bulk.prepare(x)
        import tomli
        from openai import OpenAI
        from urllib.parse import urlparse
        from api_settings import load_api_settings
        cfg = load_api_settings()
        client = OpenAI(api_key=Path(cfg['auth']['args'][0]).read_text().strip(),
                        base_url=cfg['base_url'], timeout=180, max_retries=0)
        r = bulk.request_one(client, entry, OUT, tok)
        results[sid] = r
        print(json.dumps({'sample_id': sid, 'status': r['status']}), flush=True)
    import hashlib
    for x, line in zip(rows, (bulk.SOURCE / 'frozen_116_candidates.jsonl').read_bytes().splitlines()):
        x['_source_line_sha256'] = hashlib.sha256(line + b'\n').hexdigest()
    baseline = json.loads((ROOT / 'stage7_artifacts/v7_spatial_failure_production_20261025_r1/manifests/success_baseline.json').read_text())
    chosen, pending = bulk.export(rows, results, OUT, baseline)
    usage = [results[s].get('usage') for s in IDS if results[s].get('usage') is not None]
    bulk.atomic(OUT / 'report.json', {
        'retry_submitted': sum((OUT / 'requests' / f'{s}_initial.json').exists() for s in IDS),
        'retry_statuses': {s: results[s]['status'] for s in sorted(IDS)},
        'retry_known_usage': {
            'input_tokens': sum(u.get('input_tokens', 0) or 0 for u in usage),
            'output_tokens': sum(u.get('output_tokens', 0) or 0 for u in usage),
            'reasoning_tokens': sum((u.get('output_tokens_details') or {}).get('reasoning_tokens', 0) or 0 for u in usage),
            'total_tokens': sum(u.get('total_tokens', 0) or 0 for u in usage),
        },
        'retry_unknown_usage': sum(results[s].get('usage') is None for s in IDS),
        'prior_unknown_usage': 2, 'merged_exported': {k: len(v) for k, v in chosen.items()},
        'merged_pending': len(pending), 'semantic_review': 'pending_individual_visual_review',
    })
    print((OUT / 'report.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
