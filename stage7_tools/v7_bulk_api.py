#!/usr/bin/env python3
"""V3-aligned v7 bulk teacher construction with a source-scan-only phase."""
from __future__ import annotations

import argparse, base64, fcntl, hashlib, json, os, re, sys, time, uuid
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
V3 = Path(os.environ.get('V7_V3_SOURCE', str(ROOT / "verified_datasets/sft_dataset_v3/sole_cot_dataset.jsonl")))
ROLLOUTS = Path(os.environ.get('V7_ROLLOUTS', str(ROOT / "verified_datasets/rollout_trajectories_v3")))
SPLIT = Path(os.environ.get('V7_SPLIT_MANIFEST', str(ROOT / "provenance/success_episode_splits.json")))
MODEL_PATH = Path(os.environ.get('V7_BASE_MODEL', str(ROOT.parent / "models/Qwen3-VL-8B-Instruct")))
EP41 = "episode_000041_rank0000_env0004_id000008_success"
VERSION = "v7_bulk_api_v3_aligned_v1"
LOCK_NAME = "bulk_run.lock"

SYSTEM = r"""You are generating a training rationale for a robotics video-progress reward model.

You receive a task instruction, a fixed global storyboard, and query observations. The global storyboard provides visual context for the task, target identity, ordering of major stages, and milestones that may already be established. Some global frames may occur after the query frame.

Future global frames may be used to understand the task and object identities, but they are not evidence that a later event has already happened at the query frame. Every assertion about the current state must be supported by the current observation or by observations at or before the current frame.

There are two modes.

In SEQUENTIAL mode:
- INITIAL, PREVIOUS, and CURRENT observations are provided.
- A previous progress value from the source training example is provided as a noisy numerical anchor.
- Describe the local visible change from PREVIOUS to CURRENT.
- Use the global storyboard to identify the current task stage and completed milestones.
- Explain whether the current visual change supports retaining, increasing, or decreasing the previous estimate.
- Do not blindly copy the previous value.

In FROM_ZERO mode:
- INITIAL and CURRENT observations are provided.
- PREVIOUS is absent.
- No previous progress value is provided.
- Describe cumulative visual change from INITIAL to CURRENT.
- Use the global storyboard, initial state, and current state to estimate absolute progress.

Return exactly one JSON object with these fields:
mode
update.change
update.anchor
state.established
state.remaining
progress.stage
progress.reason
evidence_frames
uncertainty

Each textual field must be one complete English sentence.
The progress.reason field must explain the basis of the absolute progress estimate by combining global task stage, completed milestones, current visual change, and remaining conditions. In SEQUENTIAL mode it must also explain the relation to the previous progress anchor. In FROM_ZERO mode it must explain the initial baseline and current state.
Do not output a numeric percentage in JSON. Do not mention source labels, ground-truth progress, v3 completion text, hidden simulator state, external plans, stage IDs, or future success. Do not infer contact, grasp, lift, transport, release, or stable placement without visual support. Do not use frame number or elapsed time as a progress estimate. Keep sentences specific and complementary."""

SCHEMA = {"type":"object", "additionalProperties":False, "required":["mode","update","state","progress","evidence_frames","uncertainty"], "properties":{
    "mode":{"type":"string","enum":["sequential","from_zero"]},
    "update":{"type":"object","additionalProperties":False,"required":["change","anchor"],"properties":{"change":{"type":"string"},"anchor":{"type":"string"}}},
    "state":{"type":"object","additionalProperties":False,"required":["established","remaining"],"properties":{"established":{"type":"string"},"remaining":{"type":"string"}}},
    "progress":{"type":"object","additionalProperties":False,"required":["stage","reason"],"properties":{"stage":{"type":"string"},"reason":{"type":"string"}}},
    "evidence_frames":{"type":"array","items":{"type":"integer"}}, "uncertainty":{"type":"string"}}}

def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)

def sha(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""): h.update(b)
    return h.hexdigest()

def frame_id(path: Path):
    m=re.search(r"(?:t|frame)[_-]?(\d+)", path.stem, re.I)
    if m: return int(m.group(1))
    m=re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else None

def source_task(problem):
    m=re.search(r"task description is:\s*(.*?)\s+The task progress", problem, re.I|re.S)
    return m.group(1).strip() if m else None

def prev_progress(problem):
    matches=re.findall(r"previous timestep is\s*(-?\d+)\s*%", problem, re.I)
    return int(matches[-1]) if matches else None

def scan_source(out: Path):
    split=json.loads(SPLIT.read_text()).get("episode_splits", {})
    rows=[]; rejects=[]; episodes={}; totals={"rows":0,"success_rows":0,"sequential":0,"from_zero":0}
    with V3.open("rb") as f:
        for line_no,line in enumerate(f,1):
            totals["rows"]+=1; digest=hashlib.sha256(line).hexdigest()
            try: obj=json.loads(line); meta=obj.get("metadata") or {}
            except Exception as e:
                rejects.append({"line":line_no,"reason":"invalid_json","detail":str(e)}); continue
            if not meta.get("success",False): continue
            totals["success_rows"]+=1
            required=["episode_id","timestep","previous_timestep","from_zero","image_path","dataset_version"]
            missing=[k for k in required if k not in meta] + (["solution_orig"] if "solution_orig" not in obj else [])
            if missing:
                rejects.append({"line":line_no,"reason":"rejected_missing_source","missing":missing}); continue
            mode="from_zero" if bool(meta["from_zero"]) else "sequential"
            totals[mode]+=1
            task=source_task(obj.get("problem", ""))
            if task is None:
                rejects.append({"line":line_no,"reason":"rejected_missing_source","missing":["task"]}); continue
            ep=meta["episode_id"]
            cur=int(meta["timestep"]); prv=None if mode=="from_zero" else int(meta["previous_timestep"])
            row={"source_jsonl_line":line_no,"source_line_sha256":digest,"episode_id":ep,"mode":mode,"task":task,"current_frame":cur,"previous_frame":prv,"from_zero":bool(meta["from_zero"]),"previous_progress":None if mode=="from_zero" else prev_progress(obj.get("problem", "")),"answer":int(obj["solution_orig"]),"dataset_version":meta["dataset_version"],"image_path":meta["image_path"],"completion":obj.get("completion"),"problem":obj.get("problem","")}
            row["is_initialization"] = mode == "sequential" and prv == cur
            if mode == "sequential" and not row["is_initialization"] and not (prv is not None and prv < cur):
                rejects.append({"line":line_no,"reason":"rejected_query_order","previous":prv,"current":cur}); continue
            if mode=="sequential" and row["previous_progress"] is None:
                rejects.append({"line":line_no,"reason":"rejected_missing_source","missing":["previous_progress"]}); continue
            if ep not in split:
                rejects.append({"line":line_no,"reason":"rejected_missing_split","episode_id":ep}); continue
            row["split"]=split[ep]; episodes.setdefault(ep,[]).append(row); rows.append(row)
    ep_manifests={}; usable=[]
    for ep, ep_rows in episodes.items():
        d=ROLLOUTS/ep; main_dir=d/"images/main"; wrist_dir=d/"images/wrist"
        mains={frame_id(p):p for p in main_dir.glob("*.png") if frame_id(p) is not None} if main_dir.is_dir() else {}
        wrists={frame_id(p):p for p in wrist_dir.glob("*.png") if frame_id(p) is not None} if wrist_dir.is_dir() else {}
        paired=sorted(set(mains)&set(wrists)); missing=sorted(set(mains)^set(wrists))
        em={"episode_id":ep,"rollout_dir":str(d),"num_paired_frames":len(paired),"paired_frame_ids":paired,"missing_unpaired_frame_ids":missing,"status":"ok"}
        if not main_dir.is_dir() or not wrist_dir.is_dir() or missing or len(paired)<8:
            em["status"]="rejected_missing_paired_frame"; rejects.append({"episode_id":ep,"reason":"rejected_missing_paired_frame","main_count":len(mains),"wrist_count":len(wrists),"missing":missing}); ep_manifests[ep]=em; continue
        indices=[round(i*(len(paired)-1)/7) for i in range(8)]; sb=[paired[i] for i in indices]
        if len(set(sb))!=8 or sb[0]!=paired[0] or sb[-1]!=paired[-1] or any(a>=b for a,b in zip(sb,sb[1:])):
            em["status"]="rejected_storyboard_sampling"; rejects.append({"episode_id":ep,"reason":"rejected_storyboard_sampling","frames":sb}); ep_manifests[ep]=em; continue
        em["storyboard_frames"]=sb; em["images"]=[]
        for fr in sb:
            for view,p in (("main",mains[fr]),("wrist",wrists[fr])):
                em["images"].append({"frame":fr,"view":view,"path":str(p),"sha256":sha(p),"size":list(_image_size(p))})
        bad=[]
        for r in ep_rows:
            needed=[r["current_frame"]]+([] if r["mode"]=="from_zero" else [r["previous_frame"]])
            if any(fr not in mains or fr not in wrists for fr in needed): bad.append(r["source_jsonl_line"])
            r["storyboard_frames"]=sb; r["storyboard_images"]=em["images"]; r["rollout_dir"]=str(d)
        if bad:
            em["status"]="rejected_missing_query_frame"; rejects.append({"episode_id":ep,"reason":"rejected_missing_query_frame","lines":bad})
        else: usable.extend(ep_rows)
        ep_manifests[ep]=em
    atomic_json(out/"episode_index.json", {"episodes":episodes,"usable_rows":len(usable)})
    atomic_json(out/"storyboards/index.json", ep_manifests)
    atomic_json(out/"source_scan.json", {"version":VERSION,"source":str(V3),"totals":totals,"usable_rows":len(usable),"usable_episodes":len({r['episode_id'] for r in usable}),"reject_count":len(rejects),"reject_reasons":_counts(rejects),"rejects":rejects[:1000],"expected":{"rows":14046,"success_episodes":404,"sequential":10927,"from_zero":3119},"status":"pass" if totals["rows"]==14046 and totals["success_rows"]==14046 and len(usable)==14046 and not rejects else "blocked"})
    return usable, ep_manifests, rejects

def _image_size(path):
    try:
        from PIL import Image
        with Image.open(path) as im: return im.size
    except Exception: return [None,None]
def _counts(items):
    out={}
    for x in items: out[x["reason"]]=out.get(x["reason"],0)+1
    return out

def _messages(row, ep_manifest, tokenizer):
    mode=row["mode"]; current=row["current_frame"]; previous=row["previous_frame"]
    sb=ep_manifest["images"]; frames=ep_manifest["storyboard_frames"]
    by={(x["frame"],x["view"]):x for x in sb}
    def ref(fr, view, role):
        d=by.get((fr,view))
        if d is None:
            p=Path(ep_manifest["rollout_dir"])/"images"/view
            matches=[x for x in p.glob("*.png") if frame_id(x)==fr]
            if len(matches)!=1: raise FileNotFoundError(f"missing paired frame={fr} view={view}")
            path=matches[0]; d={"frame":fr,"view":view,"path":str(path),"sha256":sha(path),"size":list(_image_size(path))}
        return {"path":d["path"],"sha256":d["sha256"],"frame":fr,"view":view,"role":role}
    imgs=[]
    for fr in frames:
        imgs.extend([ref(fr,"main","global"),ref(fr,"wrist","global")])
    initial=frames[0]
    imgs.extend([ref(initial,"main","initial"),ref(initial,"wrist","initial")])
    if mode=="sequential": imgs.extend([ref(previous,"main","previous"),ref(previous,"wrist","previous")])
    imgs.extend([ref(current,"main","current"),ref(current,"wrist","current")])
    if len(imgs)!=(22 if mode=="sequential" else 20): raise ValueError("image count mismatch")
    labels="\n".join(f"{i+1}. {x['role'].upper()} frame={x['frame']} view={x['view']}" for i,x in enumerate(imgs))
    allowed=sorted(set(frames+[initial,current]+([] if mode=="from_zero" else [previous])))
    if mode=="sequential":
        user=f"""Mode: SEQUENTIAL

Task instruction:
{row['task']}

Initial frame: {initial}
Previous frame: {previous}
Current frame: {current}
Previous progress anchor: {row['previous_progress']}%

The global storyboard contains eight uniformly sampled time points from the complete rollout. Each point has main and wrist views. Some global frames may occur after CURRENT. They may provide task context but cannot prove that a later event has already happened at CURRENT.

The image order is:
GLOBAL main/wrist images first;
INITIAL main/wrist;
PREVIOUS main/wrist;
CURRENT main/wrist.

Image labels:
{labels}

The only legal evidence_frames are integers from this supplied set: {allowed}. Every evidence frame must be no later than CURRENT.

Generate exactly the required JSON object."""
    else:
        user=f"""Mode: FROM_ZERO

Task instruction:
{row['task']}

Initial frame: {initial}
Current frame: {current}

No previous image and no previous progress value are provided.

The global storyboard contains eight uniformly sampled time points from the complete rollout. Each point has main and wrist views. Some global frames may occur after CURRENT. They may provide task context but cannot prove that a later event has already happened at CURRENT.

The image order is:
GLOBAL main/wrist images first;
INITIAL main/wrist;
CURRENT main/wrist.

Image labels:
{labels}

The only legal evidence_frames are integers from this supplied set: {allowed}. Every evidence frame must be no later than CURRENT.

Describe cumulative progress from INITIAL to CURRENT and generate exactly the required JSON object."""
    return [{"role":"system","content":SYSTEM},{"role":"user","content":user}], imgs

def _wire_messages(messages, imgs):
    content=[{"type":"input_text","text":messages[0]["content"]}]
    for x in imgs:
        p=Path(x["path"])
        if sha(p)!=x["sha256"]: raise RuntimeError(f"image changed: {p}")
        content += [{"type":"input_text","text":f"{x['role'].upper()} frame={x['frame']} view={x['view']}"},{"type":"input_image","image_url":"data:image/png;base64,"+base64.b64encode(p.read_bytes()).decode("ascii")}]
    content += [{"type":"input_text","text":messages[1]["content"]}]
    return [{"role":"user","content":content}]

def _output_text(response):
    if getattr(response,"output_text",None): return response.output_text
    return "".join(getattr(c,"text","") for item in getattr(response,"output",[]) for c in getattr(item,"content",[]) or [])

def _sentence(s):
    return isinstance(s,str) and bool(s.strip()) and not any(t in s for t in ("<",">","%")) and s.strip()[-1] in ".!?"

def _validate(obj,row,tokenizer):
    errors=[]
    if not isinstance(obj,dict) or set(obj)!=set(SCHEMA["required"]): errors.append("schema keys mismatch"); return errors,None
    if obj.get("mode")!=row["mode"]: errors.append("mode mismatch")
    fields=[obj.get("update",{}).get("change"),obj.get("update",{}).get("anchor"),obj.get("state",{}).get("established"),obj.get("state",{}).get("remaining"),obj.get("progress",{}).get("stage"),obj.get("progress",{}).get("reason"),obj.get("uncertainty")]
    if any(not _sentence(x) for x in fields): errors.append("all text fields must be one sentence")
    if len(set(fields[:6]))<6: errors.append("repeated semantic fields")
    provided=set(row["storyboard_frames"]+[row["current_frame"],row["initial_frame"]]+([] if row["mode"]=="from_zero" else [row["previous_frame"]]))
    ev=obj.get("evidence_frames")
    if not isinstance(ev,list) or not ev or any(not isinstance(x,int) or x not in provided for x in ev): errors.append("evidence not supplied")
    if any(x>row["current_frame"] for x in ev if isinstance(x,int)): errors.append("future evidence")
    text=" ".join(fields)
    if re.search(r"(?:source label|ground[- ]truth|v3 completion|external plan|hidden simulator|future success|plan_text)",text,re.I): errors.append("forbidden source information")
    out=f"<think><update>{fields[0]} {fields[1]}</update><state>{fields[2]} {fields[3]}</state><progress>{fields[4]} {fields[5]}</progress></think><answer>{row['answer']}%</answer>"
    n=len(tokenizer.encode(out,add_special_tokens=False))
    if n>190: errors.append(f"token limit {n}>190")
    try:
        root=ElementTree.fromstring("<root>"+out+"</root>"); tags=[x.tag for x in root]
        if tags!=["think","answer"] or [x.tag for x in root[0]]!=["update","state","progress"] or root[1].text!=f"{row['answer']}%": errors.append("xml/answer mismatch")
    except Exception as e: errors.append("xml parse: "+str(e))
    return errors,{"completion":out,"qwen_completion_tokens":n}

def _load_api():
    import tomli
    from openai import OpenAI
    from api_settings import load_api_settings
    cfg = load_api_settings()
    host=cfg["base_url"]
    key=Path(cfg["auth"]["args"][0]).read_text().strip()
    client=OpenAI(api_key=key,base_url=host,timeout=180,max_retries=0)
    available={m.id for m in client.models.list().data}
    model="gpt-6-astra" if "gpt-6-astra" in available else ("gpt-6-sol" if "gpt-6-sol" in available else None)
    if model is None: raise RuntimeError("neither authorized model is available")
    return client,model,host.split("/")[2]

def _one_api(client,model,row,ep_manifest,out,tokenizer,repair=False,raw_previous=None,errors=None):
    messages,imgs=_messages(row,ep_manifest,tokenizer)
    text=messages[1]["content"]
    if repair:
        text += "\n\nYour previous response failed these checks: " + json.dumps(errors,ensure_ascii=False) + "\nReturn a corrected JSON object with the same schema and visual input. Do not invent evidence or use source labels. Previous response:\n" + str(raw_previous)
        messages=[messages[0],{"role":"user","content":text}]
    wire=_wire_messages(messages,imgs)
    name=row["sample_id"]+"_"+("repair" if repair else "initial")
    rec={"sample_id":row["sample_id"],"model":model,"repair":repair,"messages_text":messages,"image_manifest":imgs,"attempts":[],"status":"started","request_id":str(uuid.uuid4())}
    path=out/"requests"/(name+".json"); atomic_json(path,rec)
    fmt={"type":"json_schema","name":"v7_progress_rationale","schema":SCHEMA,"strict":True}
    for attempt in range(3):
        started=time.monotonic(); a={"attempt":attempt+1}; rec["attempts"].append(a); atomic_json(path,rec)
        try:
            response=client.responses.create(model=model,input=wire,max_output_tokens=2000,text={"format":fmt},extra_headers={"Idempotency-Key":rec["request_id"]})
            raw=_output_text(response); usage=getattr(response,"usage",None)
            a.update(status="ok",elapsed_seconds=round(time.monotonic()-started,3),response_id=getattr(response,"id",None),usage=usage.model_dump() if usage else None)
            rec.update(status="response_saved",raw_response=raw); atomic_json(path,rec); return raw,rec
        except Exception as e:
            a.update(status="error",elapsed_seconds=round(time.monotonic()-started,3),error={"type":type(e).__name__,"message":str(e)[:500]}); atomic_json(path,rec)
            if attempt<2: time.sleep(min(2**attempt,4))
    rec["status"]="request_failed"; atomic_json(path,rec); return None,rec

def _process_one(r, by_ep, client, model, out, tokenizer):
    r=dict(r); r["initial_frame"]=by_ep[r["episode_id"]]["storyboard_frames"][0]; r["storyboard_frames"]=by_ep[r["episode_id"]]["storyboard_frames"]; r["sample_id"]=f"{r['episode_id']}_{r['mode']}_{r['current_frame']}_line{r['source_jsonl_line']}"
    target=out/"results"/(r["sample_id"]+".json")
    if target.exists():
        old=json.loads(target.read_text())
        if old.get('status') in ('structural_pass','rejected','request_failed','uncertain'): return old
    initial_log=out/'requests'/(r['sample_id']+'_initial.json')
    if initial_log.exists():
        log=json.loads(initial_log.read_text())
        rec={**r,'status':'uncertain','validation_errors':['request was started before result commit; saved response requires explicit recovery'],'request_record':str(initial_log)}
        atomic_json(target,rec); return rec
    started=time.monotonic(); raw,request=_one_api(client,model,r,by_ep[r["episode_id"]],out,tokenizer); obj=None; errs=[]; counts=None; raw_repair=None
    if raw is None: status="request_failed"
    else:
        try: obj=json.loads(raw)
        except Exception as e: errs=["json parse: "+str(e)]
        if obj is not None: errs,counts=_validate(obj,r,tokenizer)
        if errs:
            fixed,repair_req=_one_api(client,model,r,by_ep[r["episode_id"]],out,tokenizer,True,raw,errs)
            if fixed:
                try: obj2=json.loads(fixed); errs2,counts2=_validate(obj2,r,tokenizer)
                except Exception as e: errs2=["json parse repair: "+str(e)]; obj2=None; counts2=None
                if not errs2: obj,counts,errs=obj2,counts2,[]; raw_repair=fixed
                else: raw_repair=fixed; errs=errs2
        status="structural_pass" if not errs and obj is not None else "rejected"
    rec={**r,"status":status,"structured_teacher_output":obj,"completion":counts["completion"] if counts else None,"qwen_completion_tokens":counts["qwen_completion_tokens"] if counts else None,"validation_errors":errs,"semantic_review":"pending_manual_review","request_elapsed_seconds":round(time.monotonic()-started,3),"request_record":str(out/"requests"/(r["sample_id"]+"_initial.json"))}
    if raw_repair is not None: rec["repair_response"]=raw_repair
    atomic_json(target,rec); return rec

def api_phase(out:Path, smoke_only:bool=False, workers:int=4):
    import transformers
    from transformers import AutoTokenizer
    rows, manifests, rejects=scan_source(out)
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL_PATH),local_files_only=True,trust_remote_code=True)
    client,model,host=_load_api()
    by_ep={e:manifests[e] for e in manifests}
    by_key={(r["episode_id"],r["mode"],r["current_frame"]):r for r in rows}
    smoke=[by_key[(EP41,"sequential",4)],by_key[(EP41,"sequential",50)],by_key[(EP41,"sequential",95)],by_key[(EP41,"from_zero",4)],by_key[(EP41,"from_zero",95)]]
    other=next(r for r in rows if r["mode"]=="sequential" and r["episode_id"]!=EP41); smoke.append(other)
    selected=smoke if smoke_only else rows
    atomic_json(out/"run_manifest.json",{"version":VERSION,"model":model,"service_host":host,"tokenizer":str(MODEL_PATH),"phase":"smoke" if smoke_only else "bulk","selected_count":len(selected),"workers":workers,"no_plan":True,"context_mode":"offline_full_video"})
    result_dir=out/"results"; result_dir.mkdir(exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results=[]
    with ThreadPoolExecutor(max_workers=max(1,workers)) as pool:
        futures={pool.submit(_process_one,r,by_ep,client,model,out,tokenizer):r for r in selected}
        for i,fut in enumerate(as_completed(futures),1):
            try: rec=fut.result()
            except Exception as e:
                src=futures[fut]; sid=f"{src['episode_id']}_{src['mode']}_{src['current_frame']}_line{src['source_jsonl_line']}"
                rec={"sample_id":sid,"episode_id":src['episode_id'],"source_jsonl_line":src['source_jsonl_line'],"source_line_sha256":src['source_line_sha256'],"status":"request_failed","validation_errors":[type(e).__name__+": "+str(e)]}
                atomic_json(out/'results'/(sid+'.json'),rec)
            results.append(rec)
            if i%10==0 or i==len(selected): atomic_json(out/"progress.json",{"observed_results":i,"total":len(selected),"counts":_counts([{"reason":x.get("status","request_failed")} for x in results]),"pending":len(selected)-i})
    results.sort(key=lambda x:x.get("source_jsonl_line",10**12))
    good=[x for x in results if x["status"]=="structural_pass"]
    with (out/"v7_student.jsonl").open("w") as sf, (out/"v7_teacher_audit.jsonl").open("w") as af:
        for x in good:
            student={k:x[k] for k in ("sample_id","mode","task","answer","split","episode_id","source_jsonl_line","source_line_sha256","completion")}; student["image_references"]=[{"frame":fr,"view":v,"role":role} for fr in x["storyboard_frames"] for v,role in (("main","global"),("wrist","global"))]; student["student_user_prompt"]="mode-specific student input uses the same image order; teacher audit fields are excluded"; sf.write(json.dumps(student,ensure_ascii=False)+"\n")
            audit={k:x.get(k) for k in ("sample_id","source_jsonl_line","source_line_sha256","episode_id","mode","initial_frame","previous_frame","previous_progress","answer","task","split","storyboard_frames","structured_teacher_output","completion","qwen_completion_tokens","status","semantic_review","validation_errors","request_record")}; af.write(json.dumps(audit,ensure_ascii=False)+"\n")
    atomic_json(out/"final_report.json",{"model":model,"phase":"smoke" if smoke_only else "bulk","selected":len(selected),"structural_pass":len(good),"rejected":len(results)-len(good),"rejected_reasons":_counts([{"reason":x.get("status")} for x in results if x.get("status")!="structural_pass"]),"semantic_review":"pending_manual_review"})

def finalize_partial(out: Path):
    recs=[]
    for p in sorted((out/"results").glob("*.json")):
        try: recs.append(json.loads(p.read_text()))
        except Exception: continue
    recs.sort(key=lambda x:x.get("source_jsonl_line",10**12)); good=[x for x in recs if x.get("status")=="structural_pass"]
    export_snapshot(out, good)
    atomic_json(out/"final_report.json",{"model":"gpt-6-astra","phase":"bulk_partial","expected":14046,"processed":len(recs),"structural_pass":len(good),"rejected_or_failed":len(recs)-len(good),"remaining":14046-len(recs),"semantic_review":"pending_manual_review","complete":len(recs)==14046})

def export_snapshot(out: Path, good: list[dict]):
    sys.path.insert(0, str(ROOT/'stage7_sft'))
    from protocol import Sample, messages, validate_sample, check_progress_contract
    from prompts import VERSION as STUDENT_VERSION, TEMPLATE_SHA256
    incompatible=[x['sample_id'] for x in good if check_progress_contract(x['completion'],x['mode'],x.get('previous_progress'))]
    if incompatible:
        raise ValueError(f'{len(incompatible)} legacy CoT rows do not match {STUDENT_VERSION}; explicit migration required, first={incompatible[0]}')
    stamp=time.strftime('%Y%m%d_%H%M%S',time.gmtime()); root=out/'exports'/('snapshot_'+stamp); root.mkdir(parents=True,exist_ok=False)
    by_split={s:[] for s in ('train','validation','test')}; audit=[]; init=0
    manifest_index=json.loads((out/'storyboards/index.json').read_text()); path_cache={}
    for x in sorted(good,key=lambda z:z.get('source_jsonl_line',10**12)):
        epman=manifest_index[x['episode_id']]
        if x['episode_id'] not in path_cache:
            path_cache[x['episode_id']]={v:{frame_id(q):q for q in (Path(epman['rollout_dir'])/'images'/v).glob('*.png') if frame_id(q) is not None} for v in ('main','wrist')}
        by={(im['frame'],im['view']):im for im in epman['images']}; imgs=[]
        def add(fr,view,role):
            q=path_cache[x['episode_id']].get(view,{}).get(fr)
            if q is None: raise ValueError(f'missing image {x["sample_id"]} {fr} {view}')
            cached=by.get((fr,view)); digest=cached['sha256'] if cached else sha(q)
            imgs.append({'path':str(q.resolve()),'frame':fr,'view':view,'role':role,'sha256':digest,'label':f'{role.upper()} frame={fr} view={view} role={role.lower()}'})
        for fr in x['storyboard_frames']:
            add(fr,'main','GLOBAL'); add(fr,'wrist','GLOBAL')
        add(x['initial_frame'],'main','INITIAL'); add(x['initial_frame'],'wrist','INITIAL')
        if x['mode']=='sequential': add(x['previous_frame'],'main','PREVIOUS'); add(x['previous_frame'],'wrist','PREVIOUS')
        add(x['current_frame'],'main','CURRENT'); add(x['current_frame'],'wrist','CURRENT')
        is_init=x['mode']=='sequential' and x['previous_frame']==x['current_frame']==x['initial_frame']
        row={"sample_id":x['sample_id'],"mode":x['mode'],"task":x['task'],"split":x['split'],"episode_id":x['episode_id'],"initial_frame":x['initial_frame'],"previous_frame":x['previous_frame'],"current_frame":x['current_frame'],"previous_progress":x['previous_progress'],"previous_progress_provenance":"source_previous_progress" if x['mode']=='sequential' else None,"is_initialization":is_init,"images":imgs,"completion":x['completion'],"answer":x['answer'],"source_jsonl_line":x['source_jsonl_line'],"source_line_sha256":x['source_line_sha256'],"status":"structural_pass","prompt_version":STUDENT_VERSION,"template_sha256":TEMPLATE_SHA256}
        sample=Sample(row['sample_id'],row['mode'],row['task'],row['split'],row['episode_id'],row['initial_frame'],row['previous_frame'],row['current_frame'],row['previous_progress'],row['previous_progress_provenance'],tuple(imgs),row['completion'],str(root/(row['split']+'.jsonl')),row['is_initialization']); validate_sample(sample)
        row['student_messages']=messages(sample)+[{'role':'assistant','content':[{'type':'text','text':row['completion']}]}]
        by_split[x['split']].append(row); audit.append({k:x.get(k) for k in ('sample_id','source_jsonl_line','source_line_sha256','episode_id','mode','initial_frame','previous_frame','current_frame','previous_progress','answer','task','split','storyboard_frames','structured_teacher_output','completion','qwen_completion_tokens','status','semantic_review','validation_errors','request_record')})
        init += int(row['is_initialization'])
    for split,rows in by_split.items():
        tmp=root/(split+'.jsonl.tmp'); tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)); os.replace(tmp,root/(split+'.jsonl'))
    (root/'audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audit))
    manifest={'version':VERSION,'source':str(V3),'template_version':STUDENT_VERSION,'template_sha256':TEMPLATE_SHA256,'snapshot':root.name,'partial':len(good)!=14046,'counts':{s:len(v) for s,v in by_split.items()},'accepted':len(good),'expected':14046,'initialization_rows':init,'split_source':str(SPLIT),'student_protocol':'shared stage7_sft protocol/messages','episodes':{s:sorted({r['episode_id'] for r in rows}) for s,rows in by_split.items()},'sample_ids':[r['sample_id'] for r in good]}
    atomic_json(root/'manifest.json',manifest); atomic_json(out/'latest_snapshot.json',manifest)
    return root

def acquire_lock(out: Path):
    lock=out/LOCK_NAME; payload={'pid':os.getpid(),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'cmd':' '.join(sys.argv),'output_dir':str(out)}
    fd=os.open(lock,os.O_CREAT|os.O_RDWR,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        try: old=json.loads(lock.read_text())
        except Exception: old={'raw':lock.read_text()}
        os.close(fd); raise SystemExit(json.dumps({'status':'already_running','lock':str(lock),'owner':old}))
    os.ftruncate(fd,0); os.write(fd,(json.dumps(payload)+'\n').encode()); os.fsync(fd)
    return fd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",required=True); ap.add_argument("--scan-only",action="store_true"); ap.add_argument("--smoke",action="store_true"); ap.add_argument("--finalize-only",action="store_true"); ap.add_argument("--workers",type=int,default=4); args=ap.parse_args()
    out=Path(args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    if not str(out).startswith(str(ROOT/"stage7_artifacts")): raise SystemExit("output must be under stage7_artifacts")
    usable, manifests, rejects=scan_source(out)
    print(json.dumps({"status":json.loads((out/'source_scan.json').read_text())["status"],"usable_rows":len(usable),"episodes":len(manifests),"rejects":len(rejects),"output":str(out)},ensure_ascii=False))
    if args.finalize_only: finalize_partial(out); return
    if args.scan_only: return
    if rejects: raise SystemExit("source scan is blocked; API phase not started")
    lock_fd=acquire_lock(out)
    try:
        api_phase(out, smoke_only=getattr(args,"smoke",False), workers=getattr(args,"workers",4))
        recs=[]
        for p in (out/'results').glob('*.json'):
            try: recs.append(json.loads(p.read_text()))
            except Exception: pass
        export_snapshot(out,[x for x in recs if x.get('status')=='structural_pass'])
    finally:
        fcntl.flock(lock_fd,fcntl.LOCK_UN); os.close(lock_fd)
if __name__=='__main__': main()
