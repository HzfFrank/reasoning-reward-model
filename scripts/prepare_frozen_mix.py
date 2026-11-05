"""Rebuild the exact published exposure order with relocatable dataset paths."""
import argparse, hashlib, json, pathlib, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'stage7_sft'))
from portable_paths import check_output
from protocol import load_samples

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load_dataset(path):
    path=path.resolve(); result={}
    def resolve(x):
        if isinstance(x,dict):return {k:resolve(v) for k,v in x.items()}
        if isinstance(x,list):return [resolve(v) for v in x]
        if isinstance(x,str) and x.startswith('images/'):
            target=(path.parent/x).resolve()
            if not target.is_relative_to(path.parent):raise ValueError('image escaped dataset directory')
            return str(target)
        return x
    for line in path.read_text().splitlines():
        row=resolve(json.loads(line))
        if row['split']!='train' or row['sample_id'] in result:raise ValueError('non-train or duplicate source')
        result[row['sample_id']]=row
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--success',type=pathlib.Path,required=True);p.add_argument('--failure',type=pathlib.Path,required=True);p.add_argument('--model',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
    out=check_output(a.output);sources={'success':load_dataset(a.success),'failure':load_dataset(a.failure)}
    if (len(sources['success']),len(sources['failure']))!=(13364,116):raise ValueError('this frozen plan requires the released 13364 + 116 training examples')
    plan_file=ROOT/'provenance/exposure_plan.jsonl';plan=[json.loads(x) for x in plan_file.read_text().splitlines()]
    rows=[sources[x['source']][x['source_sample_id']] for x in plan]
    manifest=json.loads((ROOT/'provenance/mix_manifest.json').read_text())
    if sha(plan_file)!=manifest['exposure_plan_sha256']:raise ValueError('exposure plan changed')
    if not (a.model/'config.json').is_file():raise FileNotFoundError('base model config missing')
    out.mkdir(parents=True,exist_ok=False);data=out/'data';data.mkdir()
    (data/'mixed_train.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    (data/'exposure_plan.jsonl').write_bytes(plan_file.read_bytes())
    manifest.update(model_path=str(a.model.resolve()),success_source=str(a.success.resolve()),failure_source=str(a.failure.resolve()),success_sha256=sha(a.success),failure_sha256=sha(a.failure),mixed_train_sha256=sha(data/'mixed_train.jsonl'),training='not_started')
    (data/'mix_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    samples=load_samples(data/'mixed_train.jsonl',validate_images=False)
    print(json.dumps({'rows':len(samples),'steps':manifest['optimizer_steps'],'output':str(out)}))
if __name__=='__main__':main()
