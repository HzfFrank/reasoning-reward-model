"""Portable frozen-template generation from already selected sequential candidates.

Default is local preparation only. --generate submits at most --max-requests.
Existing uncertain/request records are never silently resubmitted.
"""
import argparse, fcntl, hashlib, json, os, re, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stage7_sft'))
from portable_paths import check_output
import run_event_aligned_116_cot_bulk as bulk
from v7_short_failure_teacher import SYSTEM, SCHEMA, VERSION
from api_settings import load_api_settings

def main():
 p=argparse.ArgumentParser();p.add_argument('--candidates',type=Path,required=True);p.add_argument('--success-snapshot',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--generate',action='store_true');p.add_argument('--max-requests',type=int,default=6);a=p.parse_args()
 if a.max_requests<1:raise ValueError('max-requests must be positive')
 out=check_output(a.output);out.mkdir(parents=True,exist_ok=True)
 lock=(out/'generation.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for name in ('preflight','results','requests','exports'):(out/name).mkdir(exist_ok=True)
 bulk.base.SYSTEM,bulk.base.SCHEMA,bulk.base.TEACHER_VERSION=SYSTEM,SCHEMA,VERSION
 contract={'candidate_sha256':bulk.digest(a.candidates),'teacher_version':VERSION,'teacher_sha256':hashlib.sha256(SYSTEM.encode()).hexdigest(),'student_sha256':bulk.TEMPLATE_SHA256,'model':bulk.MODEL}
 mf=out/'generation_manifest.json'
 if mf.exists() and json.loads(mf.read_text())!=contract:raise ValueError('generation input/template changed; use new output')
 bulk.atomic(mf,contract)
 rows=[];entries={};results={}
 for line in a.candidates.read_bytes().splitlines(keepends=True):
  x=json.loads(line);sid=x['sample_id']
  if not re.fullmatch(r'[A-Za-z0-9_.-]+',sid) or sid in entries:raise ValueError('invalid/duplicate sample ID')
  x['_source_line_sha256']=hashlib.sha256(line).hexdigest()
  rows.append(x);entries[sid]=bulk.prepare(x);bulk.atomic(out/'preflight'/f'{sid}.json',entries[sid])
  existing=out/'results'/f'{sid}.json'
  if existing.exists():results[sid]=json.loads(existing.read_text())
 if a.generate:
  from openai import OpenAI
  from transformers import AutoTokenizer
  cfg=load_api_settings();client=OpenAI(api_key=Path(cfg['auth']['args'][0]).read_text().strip(),base_url=cfg['base_url'],timeout=180,max_retries=0)
  tok=AutoTokenizer.from_pretrained(str(a.tokenizer),local_files_only=True)
  count=0
  for sid,e in entries.items():
   if sid in results or (out/'requests'/f'{sid}_initial.json').exists():continue
   if count>=a.max_requests:break
   results[sid]=bulk.request_one(client,e,out,tok);count+=1
 reference=json.loads((a.success_snapshot/'train.jsonl').open().readline())
 chosen,pending=bulk.export(rows,results,out,{'snapshot':str(a.success_snapshot),'sequential_schema_keys':list(reference)})
 print(json.dumps({'prepared':len(rows),'exported':{k:len(v) for k,v in chosen.items()},'pending':len(pending),'generated':a.generate}))
if __name__=='__main__':main()
