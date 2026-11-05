#!/usr/bin/env python3
"""Six review-only task-6 examples from frozen geometry v2 and approved v7 XML."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import pickle
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'stage7_sft'))
from prompts import TEMPLATE_SHA256, VERSION as STUDENT_VERSION
from protocol import Sample, check_progress_contract, messages, parse_completion, validate_sample
from sampling import uniform_storyboard_frames
from run_v7_grounded_contract import teacher_user, validate as validate_teacher
from v7_geometry_v2_cot_teacher import SYSTEM, SCHEMA, REPAIR, VERSION as TEACHER_VERSION

V2 = ROOT / 'stage7_artifacts/v7_failure_geometry_progress_v2_20261024_r4'
TASK = 'pick up the black bowl next to the cookie box and place it on the plate'
MODEL = 'gpt-6.1-sol'
MODEL_PATH = ROOT.parent / 'models/Qwen3-VL-8B-Instruct'
TARGETS = (
    ('review_01','sequential','miss_0',53,75),
    ('review_02','sequential','drop_0',80,88),
    ('review_03','sequential','drop_0',88,130),
    ('review_04','from_zero','wrong_place_1',None,239),
    ('review_05','from_zero','miss_0',None,239),
    ('review_06','from_zero','baseline_0',None,122),
)


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    os.replace(temp,path)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(z) for z in path.read_text().splitlines()]


def rounded(value: float) -> int:
    if not 0<=value<=100 or not math.isfinite(value): raise ValueError(f'invalid frozen v2 P: {value}')
    return max(0,min(100,math.floor(value+0.5)))


def paired_images(path: Path):
    result={}
    for view in ('main','wrist'):
        files={int(p.stem):p for p in (path/'images'/view).glob('*.png')}
        result[view]=files
    if not result['main'] or set(result['main'])!=set(result['wrist']):
        raise ValueError(f'missing paired raw images: {path}')
    return result


def prepare(out: Path) -> list[dict]:
    manifest={x['id']:x for x in json.loads((V2/'input_manifest.json').read_text())}
    frozen=json.loads((V2/'frozen_config.json').read_text())
    if frozen['beta']!=0.75 or frozen['y_start']!=0.3973929750757401 or not frozen['review_only']:
        raise ValueError('frozen v2 configuration changed')
    if frozen['version']!='task6_two_distance_shared_train_anchor_v2_review_only':
        raise ValueError('wrong progress version')
    prepared=[]; cache={}
    for sid,mode,tid,previous,current in TARGETS:
        source=manifest[tid];traj=Path(source['path']).resolve()
        if not traj.is_relative_to(ROOT) or not traj.is_dir():raise ValueError('source trajectory outside data disk')
        if source['reset_state_id'] not in (300,301,302):raise ValueError('wrong task-6 reset state')
        for filename,expected in source['source_sha256'].items():
            if digest(traj/filename)!=expected:raise ValueError(f'source hash changed: {tid}/{filename}')
        if tid not in cache:
            rows=read_rows(V2/'trajectories'/tid/'per_frame.jsonl')
            ims=paired_images(traj);frames=sorted(ims['main'])
            if frames!=list(range(source['frames'])) or len(rows)!=len(frames):raise ValueError(f'frame alignment: {tid}')
            story=uniform_storyboard_frames(frames,8)
            with (traj/'semantic_states.pkl').open('rb') as f:sem=pickle.load(f)
            if any(x['task_id']!=6 or x['object_name']!='akita_black_bowl_1' or x['goal_name']!='plate_1' for x in sem):
                raise ValueError(f'wrong task/object: {tid}')
            if sem[0]['language_instruction'].strip().rstrip('.').lower()!=TASK:
                raise ValueError(f'task text differs: {tid}')
            cache[tid]=(rows,ims,story,sem)
        rows,ims,story,sem=cache[tid]
        if not 0<=current<len(rows) or (mode=='sequential' and not 0<=previous<current):raise ValueError('wrong query')
        if rows[current]['frame']!=current or (previous is not None and rows[previous]['frame']!=previous):raise ValueError('score/frame mismatch')
        previous_float=rows[previous]['P'] if previous is not None else None
        images=[]
        def add(frame:int,role:str):
            for view in ('main','wrist'):
                path=ims[view][frame].resolve()
                images.append(dict(path=str(path),sha256=digest(path),frame=frame,view=view,
                                   role=role.lower(),label=f'{role} frame={frame} view={view} role={role.lower()}'))
        for frame in story:add(frame,'GLOBAL')
        add(0,'INITIAL')
        if mode=='sequential':add(previous,'PREVIOUS')
        add(current,'CURRENT')
        answer=rounded(rows[current]['P']);prev=rounded(previous_float) if previous is not None else None
        sample=Sample(sid,mode,TASK,'unassigned',source['episode_id'],0,previous,current,prev,
                      'geometry_v2_reference' if mode=='sequential' else None,tuple(images))
        validate_sample(sample)
        student=messages(sample)
        student_text='\n'.join(x['text'] for m in student for x in m['content'] if x['type']=='text')
        if any(secret in student_text for secret in (str(answer)+'%', 'Current reference progress', 'Reference change','label_visual_conflict','evidence_frames')):
            # The numeric current answer can legitimately equal the previous anchor; check
            # explicit current-label fields instead of rejecting that identical number.
            if any(secret in student_text for secret in ('Current reference progress','Reference change','label_visual_conflict','evidence_frames')):
                raise ValueError('teacher data leaked into student text')
        if mode=='from_zero' and ('PREVIOUS frame=' in student_text or 'Previous task progress:' in student_text):
            raise ValueError('from_zero has previous input')
        teacher_source=dict(mode=mode,task=TASK,previous=previous,current=current,previous_progress=prev,answer=answer)
        user=teacher_user(teacher_source,images)
        entry=dict(sample_id=sid,mode=mode,task=TASK,trajectory_id=tid,trajectory_dir=str(traj),
                   episode_id=source['episode_id'],split='unassigned',group_id=f'task6_reset_{source["reset_state_id"]}',
                   reset_state_id=source['reset_state_id'],initial=0,previous=previous,current=current,
                   previous_progress=prev,previous_progress_float=previous_float,
                   previous_progress_provenance='geometry_v2_reference' if mode=='sequential' else None,
                   current_progress_float=rows[current]['P'],answer=answer,
                   initial_geometry_progress_float=rows[0]['P'],initial_task_baseline=0,
                   current_success=rows[current]['success_current'],episode_success=bool(source['type']=='baseline' or json.loads((traj/'metadata.json').read_text())['success']),
                   formula_version=frozen['version'],frozen_config_sha256=digest(V2/'frozen_config.json'),
                   score_file=str(V2/'trajectories'/tid/'per_frame.jsonl'),score_file_sha256=digest(V2/'trajectories'/tid/'per_frame.jsonl'),
                   storyboard_frames=story,images=images,student_messages=student,teacher_user=user,
                   student_prompt_version=STUDENT_VERSION,student_template_sha256=TEMPLATE_SHA256,
                   source_manifest_sha256=digest(V2/'input_manifest.json'))
        if len(images)!=(22 if mode=='sequential' else 20):raise ValueError('image count')
        prepared.append(entry)
    atomic(out/'input_manifest.json',dict(status='preflight_pass',review_only=True,
        v2_config=str(V2/'frozen_config.json'),v2_config_sha256=digest(V2/'frozen_config.json'),
        teacher_version=TEACHER_VERSION,teacher_system_sha256=hashlib.sha256(SYSTEM.encode()).hexdigest(),
        student_version=STUDENT_VERSION,student_template_sha256=TEMPLATE_SHA256,
        samples=[{k:v for k,v in x.items() if k not in ('images','student_messages','teacher_user')} for x in prepared]))
    for x in prepared:
        atomic(out/'preflight'/f'{x["sample_id"]}.json',x)
    return prepared


def wire(user:str,images:list[dict],extra:str|None=None):
    content=[{'type':'input_text','text':user}]
    for im in images:
        raw=Path(im['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=im['sha256']:raise ValueError('image hash changed')
        content.extend(({'type':'input_text','text':im['label']},
                        {'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(raw).decode()}))
    result=[{'role':'system','content':[{'type':'input_text','text':SYSTEM}]},
            {'role':'user','content':content}]
    if extra:result.append({'role':'user','content':[{'type':'input_text','text':extra}]})
    return result


def request(client,entry,out,kind,extra=None):
    path=out/'requests'/f'{entry["sample_id"]}_{kind}.json'
    reqid=str(uuid.uuid4())
    rec=dict(sample_id=entry['sample_id'],kind=kind,model=MODEL,request_id=reqid,
             teacher_version=TEACHER_VERSION,system_text=SYSTEM,user_text=entry['teacher_user'],
             extra_text=extra,image_manifest=entry['images'],image_count=len(entry['images']),
             api_parameters=dict(max_output_tokens=2500,response_format='strict_json_schema',sdk_retries=0),
             status='started',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    atomic(path,rec)
    started=time.monotonic()
    try:
        r=client.responses.create(model=MODEL,input=wire(entry['teacher_user'],entry['images'],extra),
            max_output_tokens=2500,text={'format':{'type':'json_schema','name':'v7_review_rationale','schema':SCHEMA,'strict':True}},
            extra_headers={'Idempotency-Key':reqid})
        returned=getattr(r,'model',None)
        if returned and returned!=MODEL:raise RuntimeError('service returned unexpected model')
        raw=getattr(r,'output_text','') or ''
        rec.update(status='response_saved',raw_response=raw,response_id=getattr(r,'id',None),returned_model=returned,
                   usage=r.usage.model_dump() if getattr(r,'usage',None) else None,
                   response_status=getattr(r,'status',None),incomplete_details=str(getattr(r,'incomplete_details',None)),
                   seconds=round(time.monotonic()-started,3))
        atomic(path,rec);return raw,rec
    except Exception as e:
        status=getattr(e,'status_code',None)
        rec.update(status='uncertain' if status is None or status>=500 else 'request_failed',
                   error_type=type(e).__name__,http_status=status,seconds=round(time.monotonic()-started,3),usage=None)
        atomic(path,rec);return None,rec


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--preflight-only',action='store_true');args=ap.parse_args()
    out=args.output_dir.resolve()
    if not out.is_relative_to(ROOT/'stage7_artifacts'):
        raise ValueError('output must be under stage7_artifacts')
    if out.exists() and any(p.name not in ('tmp','cache','run.log') for p in out.iterdir()):
        raise ValueError('new output directory required; only precreated tmp/cache are allowed')
    out.mkdir(parents=True,exist_ok=True)
    prepared=prepare(out)
    atomic(out/'templates.json',dict(teacher_version=TEACHER_VERSION,teacher_system=SYSTEM,schema=SCHEMA,
       teacher_system_sha256=hashlib.sha256(SYSTEM.encode()).hexdigest(),student_version=STUDENT_VERSION,
       student_template_sha256=TEMPLATE_SHA256,approved_teacher_base_version='v7_grounded_contract_examples_v1',
       renderer='run_v7_grounded_contract.validate'))
    print(json.dumps(dict(preflight='pass',samples=[(x['sample_id'],x['previous'],x['current'],x['previous_progress'],x['answer'],len(x['images'])) for x in prepared])))
    if args.preflight_only:return
    import tomli
    from openai import OpenAI
    from transformers import AutoTokenizer
    from api_settings import load_api_settings
    cfg = load_api_settings()
    client=OpenAI(api_key=Path(cfg['auth']['args'][0]).read_text().strip(),base_url=cfg['base_url'],timeout=180,max_retries=0)
    models={m.id for m in client.models.list().data}
    if MODEL not in models:raise RuntimeError(f'{MODEL} unavailable; no fallback')
    tok=AutoTokenizer.from_pretrained(str(MODEL_PATH),local_files_only=True)
    results=[]
    for e in prepared:
        raw,request_rec=request(client,e,out,'initial')
        source=dict(mode=e['mode'],task=e['task'],previous=e['previous'],current=e['current'],
                    previous_progress=e['previous_progress'],answer=e['answer'])
        if raw is None:
            result=dict(sample_id=e['sample_id'],status=request_rec['status'],errors=[request_rec.get('error_type','request_failed')],
                        semantic_review='pending_user_review',request_count=1,repair_count=0)
        else:
            obj,errors,rendered=validate_teacher(raw,source,e['images'],tok)
            repair_count=0; repair_rec=None
            if errors:
                repair_count=1
                extra=REPAIR.format(errors=json.dumps(errors,ensure_ascii=False),raw=raw)
                repaired,repair_rec=request(client,e,out,'repair',extra)
                if repaired is not None:
                    obj,errors,rendered=validate_teacher(repaired,source,e['images'],tok)
                    raw=repaired
                else:errors.append('repair request '+repair_rec['status'])
            if rendered is not None and not errors:
                completion=rendered['completion']
                if parse_completion(completion)['progress']!=e['answer'] or check_progress_contract(completion,e['mode'],e['previous_progress']):
                    errors.append('XML/progress contract')
            result=dict(sample_id=e['sample_id'],status='structural_pass' if not errors else 'rejected',
                        errors=errors,structured_teacher_output=obj,raw_response=raw,
                        completion=rendered['completion'] if rendered else None,qwen_tokens=rendered['tokens'] if rendered else None,
                        label_visual_conflict=obj.get('label_visual_conflict') if obj else None,
                        conflict_reason=obj.get('conflict_reason') if obj else None,
                        semantic_review='pending_user_review',request_count=1+repair_count,repair_count=repair_count,
                        initial_request=str(out/'requests'/f'{e["sample_id"]}_initial.json'),
                        repair_request=str(out/'requests'/f'{e["sample_id"]}_repair.json') if repair_count else None)
        atomic(out/'results'/f'{e["sample_id"]}.json',result)
        results.append(result)
        atomic(out/'progress.json',dict(completed=len(results),total=len(prepared),request_count=sum(r['request_count'] for r in results),
            structural_pass=sum(r['status']=='structural_pass' for r in results),rejected=sum(r['status']=='rejected' for r in results),
            uncertain=sum(r['status']=='uncertain' for r in results)))
        print(json.dumps(dict(sample=e['sample_id'],status=result['status'],tokens=result.get('qwen_tokens'),errors=result['errors'])),flush=True)
    atomic(out/'report.json',dict(model=MODEL,teacher_version=TEACHER_VERSION,student_version=STUDENT_VERSION,
        initial_requests=len(prepared),actual_requests=sum(r['request_count'] for r in results),
        repairs=sum(r['repair_count'] for r in results),structural_pass=sum(r['status']=='structural_pass' for r in results),
        rejected=sum(r['status']=='rejected' for r in results),uncertain=sum(r['status']=='uncertain' for r in results),
        review_only=True,samples=[dict(sample_id=r['sample_id'],status=r['status'],tokens=r.get('qwen_tokens'),
                 conflict=r.get('label_visual_conflict'),errors=r['errors']) for r in results]))

if __name__=='__main__':main()
