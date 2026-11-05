#!/usr/bin/env python3
"""Offline full-video sequential/from-zero Qwen3-VL inference."""
from __future__ import annotations
import argparse, hashlib, json, math, re, time
from pathlib import Path
import torch
from processing import encode_sample, load_processor
from protocol import Sample, parse_completion, check_progress_contract, messages
from runtime import load_model
from prompts import VERSION as PROMPT_VERSION, TEMPLATE_SHA256
from sampling import uniform_storyboard_frames


def task_from_v3(path: Path, episode_id: str) -> str:
    tasks=set()
    with path.open() as source:
        for line in source:
            row=json.loads(line)
            meta=row.get('metadata',{})
            if meta.get('episode_id') != episode_id or not meta.get('success'): continue
            match=re.search(r'task description is:\s*(.*?)\s+The task progress',row['problem'],re.I|re.S)
            if not match: raise ValueError(f'v3 task text missing for {episode_id}')
            tasks.add(match.group(1).strip())
    if len(tasks)!=1: raise ValueError(f'expected one v3 task for {episode_id}, got {len(tasks)}')
    return tasks.pop()


def image_entry(root: Path, frame: int, view: str, role: str) -> dict:
    path = (root/'images'/view/f'{frame:06d}.png').resolve()
    if not path.is_file(): raise FileNotFoundError(path)
    return {'type':'image_ref','path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'frame':frame,'view':view,'label':f'{role} frame={frame} view={view} role={role.lower()}'}


def episode_frames(root: Path) -> list[int]:
    main={int(p.stem) for p in (root/'images'/'main').glob('*.png')}
    wrist={int(p.stem) for p in (root/'images'/'wrist').glob('*.png')}
    if main != wrist:
        raise ValueError(f'paired observation mismatch: main_only={sorted(main-wrist)[:8]}, wrist_only={sorted(wrist-main)[:8]}')
    frames=sorted(main)
    if len(frames)<8: raise ValueError('fewer than eight paired observations')
    return frames


def build_query(root: Path, task: str, mode: str, current: int, previous: int | None,
                previous_progress: int | None, storyboard: list[int]) -> Sample:
    images=[]
    for f in storyboard:
        for view in ('main','wrist'): images.append(image_entry(root,f,view,'GLOBAL'))
    for view in ('main','wrist'): images.append(image_entry(root,storyboard[0],view,'INITIAL'))
    if mode=='sequential':
        for view in ('main','wrist'): images.append(image_entry(root,previous,view,'PREVIOUS'))
    for view in ('main','wrist'): images.append(image_entry(root,current,view,'CURRENT'))
    return Sample(f'{mode}_{current}',mode,task,'inference',root.name,storyboard[0],previous,current,
                  previous_progress,'model_prediction_or_explicit_initialization' if mode=='sequential' else None,
                  tuple(images),is_initialization=(mode=='sequential' and previous==current==storyboard[0]))


def generate_one(model, processor, sample: Sample, max_seq_length: int, max_new_tokens: int, device: str):
    item=encode_sample(sample,processor,max_seq_length,training=False)
    batch={k:v.to(device) for k,v in item['tensors'].items()}
    start=time.monotonic()
    with torch.inference_mode():
        ids=model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id)
    generated=ids[0,batch['input_ids'].shape[1]:]
    raw=processor.tokenizer.decode(generated,skip_special_tokens=True).strip()
    parsed=parse_completion(raw)
    contract_errors=check_progress_contract(raw,sample.mode,sample.previous_progress) if parsed['ok'] else []
    attempt={'attempt':1,'raw':raw,'parse':parsed,'contract_errors':contract_errors,
             'generated_tokens':int(generated.numel()),'stopped_at_max_new_tokens':int(generated.numel())>=max_new_tokens,
             'seconds':round(time.monotonic()-start,3)}
    status='ok' if parsed['ok'] and not contract_errors else ('contract_failure' if parsed['ok'] else 'parse_failure')
    chat=messages(sample)
    return {'sample_id':sample.sample_id,'mode':sample.mode,'current':sample.current,'previous':sample.previous,
            'previous_progress':sample.previous_progress,'previous_progress_provenance':sample.previous_progress_provenance,
            'image_count':item['images'],'images':list(sample.images),'prompt_tokens':item['prompt_tokens'],
            'prompt_version':PROMPT_VERSION,'template_sha256':TEMPLATE_SHA256,
            'system_text':chat[0]['content'][0]['text'],'user_text':chat[1]['content'][0]['text'],
            'attempts':[attempt], 'status':status,
            'progress':parsed['progress'] if status=='ok' else None}


def run(model, processor, root: Path, task: str, mode: str, queries: list[int],
        max_seq_length: int, max_new_tokens: int, device: str, record_dir: Path | None = None) -> dict:
    frames=episode_frames(root)
    storyboard=uniform_storyboard_frames(frames,8)
    if any(q not in frames or q<=frames[0] for q in queries): raise ValueError('query frame missing or initial')
    if queries != sorted(set(queries)): raise ValueError('queries must be strictly increasing')
    previous=frames[0]; previous_progress=0; results=[]
    for current in queries:
        sample=build_query(root,task,mode,current,previous if mode=='sequential' else None,
                           previous_progress if mode=='sequential' else None,storyboard)
        result=generate_one(model,processor,sample,max_seq_length,max_new_tokens,device)
        if mode=='sequential' and len(results)==0: result['previous_progress_provenance']='initialized_zero_not_ground_truth'
        results.append(result)
        if record_dir is not None:
            record_dir.mkdir(parents=True,exist_ok=True)
            (record_dir/f'{mode}_{current}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        if mode=='sequential':
            if result['status']!='ok':
                for blocked in queries[len(results):]:
                    results.append({'mode':mode,'current':blocked,'status':'blocked_by_previous_failure',
                                    'blocked_by':current,'attempts':[]})
                break
            previous=current; previous_progress=result['progress']
    return {'mode':mode,'prompt_version':PROMPT_VERSION,'template_sha256':TEMPLATE_SHA256,
            'context_mode':'offline_full_video','future_visual_context_available':True,
            'storyboard_frames':storyboard,'requested_queries':queries,'queries':results,
            'parse_failures':sum(x['status']=='parse_failure' for x in results),
            'contract_failures':sum(x['status']=='contract_failure' for x in results),
            'blocked':sum(x['status']=='blocked_by_previous_failure' for x in results)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model-path',required=True); ap.add_argument('--checkpoint'); ap.add_argument('--episode-dir',required=True)
    ap.add_argument('--task'); ap.add_argument('--task-source-v3'); ap.add_argument('--mode',choices=['sequential','from_zero'],required=True)
    ap.add_argument('--queries',required=True,help='comma-separated increasing current frame indices'); ap.add_argument('--output-dir',required=True)
    ap.add_argument('--image-pixels',type=int,default=65536); ap.add_argument('--max-seq-length',type=int,default=8192)
    ap.add_argument('--max-new-tokens',type=int,default=256); ap.add_argument('--device',default='cuda:0'); args=ap.parse_args()
    out=Path(args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    if args.checkpoint:
        meta=json.loads((Path(args.checkpoint)/'metadata.json').read_text())
        protocol=meta.get('protocol',{}); expected=protocol.get('image_pixels')
        if expected is not None and expected != args.image_pixels: raise ValueError(f'checkpoint image_pixels={expected} conflicts with CLI={args.image_pixels}')
        if protocol.get('prompt_version') != PROMPT_VERSION or protocol.get('template_sha256') != TEMPLATE_SHA256:
            raise ValueError('checkpoint student prompt differs from current shared prompt')
        processor=load_processor(str(Path(args.checkpoint)/'processor'),args.image_pixels) if (Path(args.checkpoint)/'processor').is_dir() else load_processor(args.model_path,args.image_pixels)
    else: processor=load_processor(args.model_path,args.image_pixels)
    model=load_model(args.model_path,args.device,args.checkpoint).eval()
    task=task_from_v3(Path(args.task_source_v3),Path(args.episode_dir).name) if args.task_source_v3 else args.task
    if not task: raise ValueError('provide --task or --task-source-v3')
    if args.task and args.task_source_v3 and args.task != task: raise ValueError('CLI task differs from v3 source task')
    result=run(model,processor,Path(args.episode_dir),task,args.mode,[int(x) for x in args.queries.split(',')],args.max_seq_length,args.max_new_tokens,args.device,out/'per_query')
    result['task_source']=str(Path(args.task_source_v3).resolve()) if args.task_source_v3 else 'explicit_cli'
    (out/f'{args.mode}_inference.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
