"""Exactly six initial requests; no retries, bulk switch changes or training."""
import argparse, json, pathlib, pickle, hashlib, shutil
import run_v7_geometry_v2_cot_examples as base
from run_v7_grounded_contract import validate
from v7_short_failure_teacher import SYSTEM, SCHEMA, VERSION
ROOT=pathlib.Path(__file__).resolve().parents[1]
SOURCE=ROOT/'stage7_artifacts/v7_spatial_failure_production_20261025_r1/balanced_failure_20261026_r1/short_interval_corrected_20261027_r2/train_candidates.jsonl'
KINDS=['miss','drop','wrong_place','d1','d2','wrong_object']
def clean(v):
 if hasattr(v,'tolist'):return v.tolist()
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [clean(x) for x in v]
 return v

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--generate',action='store_true');a=ap.parse_args();out=a.output.resolve()
 assert out.is_relative_to(ROOT/'stage7_artifacts')
 out.mkdir(parents=True,exist_ok=True)
 assert not list((out/'requests').glob('*.json')), 'Existing request records: refuse duplicate submissions'
 xs=[json.loads(l) for l in SOURCE.open()]
 chosen=[next(x for x in xs if x['failure_type']==k and not x.get('is_stall',False)) for k in KINDS]
 base.SYSTEM,base.SCHEMA,base.TEACHER_VERSION=SYSTEM,SCHEMA,VERSION
 base.atomic(out/'templates.json',dict(version=VERSION,system=SYSTEM,schema=SCHEMA,renderer='run_v7_grounded_contract.validate',system_sha256=hashlib.sha256(SYSTEM.encode()).hexdigest(),student_version=base.STUDENT_VERSION,student_template_sha256=base.TEMPLATE_SHA256,request_limit=6,repairs=0,model=base.MODEL))
 for name in ['v7_short_failure_teacher.py','v7_spatial_def_teacher.py','v7_grounded_contract_templates.py','run_v7_grounded_contract.py','run_short_failure_six_review.py']:
  (out/'code_snapshot').mkdir(exist_ok=True);shutil.copyfile(ROOT/'stage7_tools'/name,out/'code_snapshot'/name)
 entries=[]
 for x in chosen:
  src=dict(mode='sequential',task=x['task'],previous=x['previous_frame'],current=x['current_frame'],previous_progress=x['previous_progress'],answer=x['current_progress'])
  assert len(x['images'])==22
  for im in x['images']:assert base.digest(pathlib.Path(im['path']))==im['sha256']
  states=pickle.load((pathlib.Path(x['trajectory_dir'])/'semantic_states.pkl').open('rb'))
  first=states[0];keys=['object_name','goal_name','object_pos','goal_pos','eef_pos','gripper_qpos']
  facts={role:{k:clean(states[f].get(k)) for k in keys} for role,f in [('PREVIOUS',src['previous']),('CURRENT',src['current'])]}
  registry={k:clean(first.get(k)) for k in ['object_name','goal_name','language_instruction','task_name']}
  registry['non_target_ramekin_identity']='glazed_rim_porcelain_ramekin_1'
  user=base.teacher_user(src,x['images'])+'\n\nTEACHER-ONLY OBJECT REGISTRY\n'+json.dumps(registry)+'\nTEACHER-ONLY LOCAL SUBPHASE\n'+x.get('event_subphase','')+'\nTEACHER-ONLY ENDPOINT STATE (world coordinates in metres, not image left/right; no grasp inferred from proximity alone)\n'+json.dumps(facts)
  e=dict(sample_id=x['sample_id'],kind=x['failure_type'],source=src,source_candidate=x,teacher_user=user,images=x['images'],student_messages=x['student_messages'])
  entries.append(e);base.atomic(out/'preflight'/f'{e["kind"]}.json',e)
 base.atomic(out/'selection.json',dict(source=str(SOURCE),source_sha256=base.digest(SOURCE),selection='first non-stall source row per type, no label-sign selection',samples=[dict(kind=e['kind'],sample_id=e['sample_id'],**e['source']) for e in entries]))
 print(json.dumps({'prepared':len(entries),'selection':[(e['kind'],e['source']['previous'],e['source']['current']) for e in entries]}),flush=True)
 if not a.generate:return
 import tomli
 from openai import OpenAI
 from transformers import AutoTokenizer
 from api_settings import load_api_settings
 cfg = load_api_settings()
 from urllib.parse import urlparse
 client=OpenAI(api_key=pathlib.Path(cfg['auth']['args'][0]).read_text().strip(),base_url=cfg['base_url'],timeout=180,max_retries=0)
 tok=AutoTokenizer.from_pretrained(str(base.MODEL_PATH),local_files_only=True)
 results=[]
 for e in entries:
  raw,rec=base.request(client,e,out,'initial')
  obj,errors,rendered=validate(raw,e['source'],e['images'],tok) if raw else (None,['no returned completion'],None)
  result=dict(kind=e['kind'],sample_id=e['sample_id'],source=e['source'],status='structural_pass' if not errors else rec['status'] if raw is None else 'rejected',errors=errors,structured=obj,completion=rendered['completion'] if rendered else None,qwen_tokens=rendered['tokens'] if rendered else None,usage=rec.get('usage'),semantic_status='pending_user_review')
  base.atomic(out/'results'/f'{e["kind"]}.json',result);results.append(result)
  print(json.dumps({'kind':e['kind'],'status':result['status'],'tokens':result['qwen_tokens']}),flush=True)
 base.atomic(out/'report.json',dict(model=base.MODEL,requests=len(results),repairs=0,results=results))
 display='# 六类失败短窗 CoT 真实样例\n\n固定模板 v7_short_failure_teacher_v1；仅六次首次请求，不重试、不启动批量。全部待用户审阅。\n\n'
 for r in results:
  s=r['source'];display+=f'## {r["kind"]}：{s["previous"]}→{s["current"]}，{s["previous_progress"]}%→{s["answer"]}%\n\n```xml\n{r["completion"] or "无返回结果"}\n```\n\n'
 (out/'DISPLAY.md').write_text(display)
if __name__=='__main__':main()
