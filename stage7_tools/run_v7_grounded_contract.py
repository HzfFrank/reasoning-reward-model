import argparse,base64,hashlib,json,os,re,sys,time,uuid
from pathlib import Path
from xml.etree import ElementTree
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'stage7_sft'))
from protocol import Sample,messages as student_messages,validate_sample,parse_completion
from v7_grounded_contract_templates import VERSION,SYSTEM,SCHEMA,REPAIR
V3=ROOT/'verified_datasets/sft_dataset_v3/sole_cot_dataset.jsonl'; EP='episode_000041_rank0000_env0004_id000008_success'; ROLL=ROOT/'verified_datasets/rollout_trajectories_v3'/EP; TOK=ROOT.parent/'models/Qwen3-VL-8B-Instruct'; STORY=[0,14,28,42,57,71,85,99]
TARGET=[('sequential',50,48,48),('sequential',95,91,95),('from_zero',4,None,2),('from_zero',95,None,95)]
def atomic(p,x):
 p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n'); os.replace(q,p)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def frame(p):
 m=re.search(r'(?:t|frame)?[_-]?(\d+)',Path(p).stem); return int(m.group(1)) if m else None
def sources():
 found={}
 for n,line in enumerate(V3.open('rb'),1):
  if EP.encode() not in line: continue
  x=json.loads(line); m=x.get('metadata',{}); mode='from_zero' if m.get('from_zero') else 'sequential'; key=(mode,m.get('timestep'))
  if m.get('episode_id')!=EP or not m.get('success') or key not in {(a,b) for a,b,_,_ in TARGET}: continue
  exp=next(z for z in TARGET if z[0]==mode and z[1]==m['timestep']); prev=exp[2]; ans=int(x['solution_orig'])
  if ans!=exp[3] or (mode=='sequential' and m.get('previous_timestep')!=prev): raise RuntimeError(f'source mismatch line {n}')
  task=re.search(r'task description is:\s*(.*?)\s+The task progress',x['problem'],re.I|re.S).group(1).strip(); pm=re.findall(r'previous timestep is\s*(-?\d+)\s*%',x['problem'],re.I)
  found[key]={'mode':mode,'previous':None if mode=='from_zero' else prev,'current':m['timestep'],'previous_progress':None if mode=='from_zero' else int(pm[-1]),'answer':ans,'task':task,'line':n,'sha256':hashlib.sha256(line).hexdigest()}
 if len(found)!=4: raise RuntimeError(f'missing source rows {found.keys()}')
 return [found[(a,b)] for a,b,_,_ in TARGET]
def imgs(s):
 main={frame(p):p for p in (ROLL/'images/main').glob('*.png')}; wrist={frame(p):p for p in (ROLL/'images/wrist').glob('*.png')}; assert set(main)==set(wrist)
 allf=sorted(main); assert [allf[round(i*(len(allf)-1)/7)] for i in range(8)]==STORY
 out=[]
 def add(f,v,r):
  p=(main if v=='main' else wrist)[f]; out.append({'path':str(p.resolve()),'sha256':sha(p),'frame':f,'view':v,'role':r,'label':f'{r.upper()} frame={f} view={v} role={r.lower()}'})
 for f in STORY:
  add(f,'main','global');add(f,'wrist','global')
 add(0,'main','initial');add(0,'wrist','initial')
 if s['mode']=='sequential': add(s['previous'],'main','previous');add(s['previous'],'wrist','previous')
 add(s['current'],'main','current');add(s['current'],'wrist','current'); assert len(out)==(22 if s['mode']=='sequential' else 20); return out
def teacher_user(s,im):
 gm='\n'.join(x['label'] for x in im[:16]); qm='\n'.join(x['label'] for x in im[16:]); text=f'Mode: {s["mode"].upper()}\nTask instruction: {s["task"]}\nINITIAL frame: 0\nCURRENT frame: {s["current"]}\n'
 if s['mode']=='sequential': text+=f'PREVIOUS frame: {s["previous"]}\n'
 text+=f'\nVISUAL INPUT\n{gm}\n{qm}\n\nOFFLINE ANNOTATION REFERENCE - NOT PART OF THE STUDENT INPUT\nInitial progress baseline: 0%\nCurrent reference progress: {s["answer"]}%\n'
 if s['mode']=='sequential': text+=f'Previous reference progress: {s["previous_progress"]}%\nReference change: {s["answer"]-s["previous_progress"]} percentage points\n'
 return text+'\nGenerate the required JSON explanation.'
def wire(user,im):
 c=[{'type':'input_text','text':user}]
 for x in im:
  c += [{'type':'input_text','text':x['label']},{'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(Path(x['path']).read_bytes()).decode()}]
 return [{'role':'system','content':[{'type':'input_text','text':SYSTEM}]},{'role':'user','content':c}]
def validate(raw,s,im,tok):
 import jsonschema
 try: x=json.loads(raw);jsonschema.validate(x,SCHEMA)
 except Exception as e:return None,[f'schema {e}'],None
 e=[]; mode=s['mode']; starts=('At the previous timestep, ','At the current timestep, ','Comparing the two observations, '); vals=[x['update'][k] for k in ('reference_observation','current_observation','comparison')]
 if x['mode']!=mode:e.append('mode')
 if not vals[0].startswith(starts[0] if mode=='sequential' else 'At the initial timestep, '):e.append('reference prefix')
 if not vals[1].startswith(starts[1]):e.append('current prefix')
 if not vals[2].startswith(starts[2]):e.append('comparison prefix')
 fields=vals+[x['state'][k] for k in ('global_context','established','remaining')]+[x['progress']['reason']]
 if any(not v.strip() or '<' in v or '>' in v for v in fields):e.append('empty/XML text')
 if re.search(r'\b(?:frame\s*\d+|source label|annotation|teacher|API|current target|reference label)\b',' '.join(fields),re.I):e.append('forbidden wording')
 if any('%' in v for v in fields):e.append('percent in non-rendered fields')
 allowed={z['frame'] for z in im}; ev=x['evidence_frames']
 if not ev or any(z not in allowed or z>s['current'] for z in ev):e.append('evidence')
 base=s['previous_progress'] if mode=='sequential' else 0
 progress=('Previous task progress was '+str(base)+'%. '+x['progress']['reason']+' Current task progress is estimated at '+str(s['answer'])+'%.' if mode=='sequential' else 'Initial task progress was 0%. '+x['progress']['reason']+' Current task progress is estimated at '+str(s['answer'])+'%.')
 comp=f'<think><update>{" ".join(vals)}</update><state>{x["state"]["global_context"]} {x["state"]["established"]} {x["state"]["remaining"]}</state><progress>{progress}</progress></think><answer>{s["answer"]}%</answer>'; n=len(tok.encode(comp,add_special_tokens=False)); parsed=parse_completion(comp)
 if not parsed['ok'] or parsed['progress']!=s['answer']:e.append('XML answer')
 return x,e,{'completion':comp,'tokens':n}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',required=True);ap.add_argument('--preflight-only',action='store_true');a=ap.parse_args();out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
 if not out.is_relative_to(ROOT/'stage7_artifacts'):raise SystemExit('bad output')
 ss=sources(); prep=[]
 for s in ss:
  im=imgs(s); sample=Sample(f'{EP}_{s["mode"]}_{s["current"]}',s['mode'],s['task'],'test',EP,0,s['previous'],s['current'],s['previous_progress'],'source_previous_progress' if s['mode']=='sequential' else None,tuple(im));validate_sample(sample); sm=student_messages(sample); dump=json.dumps(sm); assert 'Current reference progress' not in dump and 'Reference change' not in dump
  prep.append({'source':s,'images':im,'teacher_user':teacher_user(s,im),'student_messages':sm})
 atomic(out/'config.json',{'version':VERSION,'model':'gpt-6.1-sol','episode':EP,'split':'test','storyboard':STORY,'schema':SCHEMA,'student_prompt_source':'stage7_sft.prompts','bulk_started':False});atomic(out/'preflight.json',{'samples':[{'mode':p['source']['mode'],'previous':p['source']['previous'],'current':p['source']['current'],'previous_progress':p['source']['previous_progress'],'answer':p['source']['answer'],'images':len(p['images']),'line':p['source']['line']} for p in prep]})
 print(json.dumps(json.loads((out/'preflight.json').read_text()),ensure_ascii=False));
 if a.preflight_only:return
 import tomli;from openai import OpenAI;from transformers import AutoTokenizer
 from api_settings import load_api_settings
 cfg = load_api_settings()
 tok=AutoTokenizer.from_pretrained(str(TOK),local_files_only=True);fmt={'type':'json_schema','name':'v7_grounded_contract','schema':SCHEMA,'strict':True};results=[]
 for p in prep:
  s=p['source'];name=f'{s["mode"]}_{s["current"]}'; raw=None;usage=None;rid=str(uuid.uuid4());attempts=[]
  for i in range(2):
   try:
    start=time.monotonic();r=client.responses.create(model='gpt-6.1-sol',input=wire(p['teacher_user'],p['images']),max_output_tokens=2500,text={'format':fmt},extra_headers={'Idempotency-Key':rid});raw=getattr(r,'output_text','') or '';usage=r.usage.model_dump() if getattr(r,'usage',None) else None;attempts.append({'attempt':i+1,'status':'ok','seconds':round(time.monotonic()-start,3),'usage':usage});break
   except Exception as ex:attempts.append({'attempt':i+1,'status':'error','error':type(ex).__name__+': '+str(ex)[:300]});time.sleep(2**i)
  obj,err,cnt=validate(raw,s,p['images'],tok) if raw else (None,['request failed'],None); repairs=0
  if err and raw:
   repairs=1;repair=REPAIR.format(errors=json.dumps(err),raw=raw);r=client.responses.create(model='gpt-6.1-sol',input=wire(p['teacher_user'],p['images'])+[{'role':'user','content':[{'type':'input_text','text':repair}]}],max_output_tokens=2500,text={'format':fmt},extra_headers={'Idempotency-Key':str(uuid.uuid4())});raw2=getattr(r,'output_text','') or '';obj,err,cnt=validate(raw2,s,p['images'],tok);raw=raw2;usage=r.usage.model_dump() if getattr(r,'usage',None) else usage
  final={'sample_id':f'{EP}_{s["mode"]}_{s["current"]}','source':s,'images':p['images'],'teacher_user':p['teacher_user'],'student_messages':p['student_messages'],'raw_response':raw,'structured':obj,'completion':cnt['completion'] if cnt else None,'qwen_tokens':cnt['tokens'] if cnt else None,'errors':err,'status':'structural_pass' if not err else 'rejected','label_visual_conflict':obj.get('label_visual_conflict') if obj else None,'usage':usage,'attempts':attempts,'repairs':repairs};atomic(out/'samples'/(name+'.json'),final);results.append(final)
 atomic(out/'report.json',{'version':VERSION,'model':'gpt-6.1-sol','requests':len(results),'repairs':sum(z['repairs'] for z in results),'accepted':sum(z['status']=='structural_pass' for z in results),'rejected':sum(z['status']=='rejected' for z in results),'samples':[{'id':z['sample_id'],'status':z['status'],'tokens':z['qwen_tokens'],'conflict':z['label_visual_conflict'],'errors':z['errors']} for z in results]});print(json.dumps({'accepted':sum(z['status']=='structural_pass' for z in results),'rejected':sum(z['status']=='rejected' for z in results)},ensure_ascii=False))
if __name__=='__main__':main()
