#!/usr/bin/env python3
"""Conservative, event-first resampling of frozen failure trajectories. No models or API."""
from __future__ import annotations
import collections, csv, hashlib, json, math, os, pathlib, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=pathlib.Path('/project_c20250301/251503016/reasoning')
sys.path.insert(0,str(ROOT/'stage7_tools'))
import split_failure_short_sequential_corrected as prior
BASE=prior.BASE
OLD=BASE/'short_interval_corrected_20261027_r2'
OUT=pathlib.Path(os.environ.get('FAILURE_EVENT_ALIGNED_OUT',str(BASE/'failure_event_aligned_sampling_v1')))
KINDS=('miss','drop','wrong_place','d1','d2','wrong_object')
DISPLAY={'miss':'A 抓空','drop':'B 掉落','wrong_place':'C 盘外放置','d1':'D1 错误接近','d2':'D2 错误运输','wrong_object':'E 错误对象'}

def rows(p):return list(prior.jsonline(p))
def dump(p,x):prior.put(p,x)
def dump_rows(p,x):prior.putl(p,x)
def image_change(path,p,c,view):
    a=path/'images'/view/f'{p:06d}.png';b=path/'images'/view/f'{c:06d}.png'
    with Image.open(a) as z:aa=np.asarray(z.convert('RGB'),dtype=np.int16)
    with Image.open(b) as z:bb=np.asarray(z.convert('RGB'),dtype=np.int16)
    return float(np.abs(aa-bb).mean())
def opening(s):return float(sum(abs(float(z)) for z in s['gripper_qpos']))
def valid_score(t,p,c):return t.rows[p]['P'] is not None and t.rows[c]['P'] is not None
def sign(a,b):return 'decrease' if b<a else 'increase' if b>a else 'equal'
def source_for_trajectory(xs):
    x=xs[0];r=dict(task_id=x['task_id'],task=x['task'],trajectory_dir=x['trajectory_dir'],failure_type=x['failure_type'],
       reset_state_id=int(x['group_id'].rsplit('reset',1)[1]),group_id=x['group_id'],split=x['split'],episode_id=x['episode_id'],
       per_frame_progress_file=x['per_frame_progress_file'],per_frame_progress_sha256=x['per_frame_progress_sha256'],
       progress_config=x['progress_config'],progress_config_sha256=x['progress_config_sha256'],event_file=x['evidence_sources'][-1])
    return r
def in_interval(t,p,c):
    return (0<=p<c<t.n and 2<=c-p<=8 and p>=max(0,t.verified_start-8) and c<=min(t.n-1,t.verified_end+8))
def physical(t,p,c):
    """Return stage and physical salience; never read progress here."""
    if not in_interval(t,p,c):return None
    o=float(np.linalg.norm(t.obj[c]-t.obj[p]));e=float(np.linalg.norm(t.eef[c]-t.eef[p]));
    gp=float(t.gap[c]-t.gap[p]);gd=float(t.gdist[c]-t.gdist[p]);r0,r1=int(t.rc[p]),int(t.rc[c]);
    q0,q1=opening(t.sem[p]),opening(t.sem[c])
    if t.kind=='miss':
        if t.close is None or not p<=t.close<=c: return None
        if t.support[c] is False or o>.022 or t.obj[c,2]-t.obj[0,2]>.012:return None
        if c<t.close+3 or q0-q1<.02 or min(int(t.rc[j]) for j in range(max(p,t.close),c+1))>=2:return None
        if any(all(t.rc[k]>=2 for k in range(j,min(j+3,t.n))) for j in range(p,c+1)):return None
        return 'visible_gripper_closure_without_target_lift',q0-q1+e*.2,{'target_motion_m':o,'opening_change_m':q1-q0,'support_current':bool(t.support[c]),'target_contacts':[r0,r1]}
    if t.kind=='drop':
        loss=t.loss
        if loss is None or not p<loss<=c:return None
        if t.recovery is not None and c>=t.recovery:return None
        if min(t.rc[max(0,p-2):p+1])<2 or r1>=2:return None
        # Prior hold, a local lost hold, and either a visible descent or new support.
        dz=float(t.obj[p,2]-t.obj[c,2]);
        if t.support[p] or (dz<.012 and not t.support[c]):return None
        return 'held_to_lost_or_descending',max(0,dz)+max(0,gp)*.2,{'loss_frame':loss,'landing_frame':t.landing,'descent_m':dz,'contacts':[r0,r1],'support':[bool(t.support[p]),bool(t.support[c])]}
    if t.kind=='wrong_place':
        release=t.release
        if release is None or not p<release<=c or r0<2 or r1>=2:return None
        if t.support[p] or not t.support[c] or t.plate_contact(c):return None
        xy=float(np.linalg.norm(t.obj[c,:2]-t.goal[c,:2]));
        if xy<=.055 or t.rows[c]['success_current'] is not False:return None
        return 'held_to_off_plate_support',xy,{'release_frame':release,'goal_xy_m':xy,'contacts':[r0,r1],'support':[bool(t.support[p]),bool(t.support[c])],'plate_contact_current':False}
    if t.kind=='d1':
        if o>.012 or max(r0,r1)>=2 or gp<.018 or e<.015:return None
        if p<t.trigger-5 or c>t.verified_end:return None
        return 'unheld_gripper_recedes_from_target',gp,{'gripper_target_gap_increase_m':gp,'target_motion_m':o,'eef_motion_m':e,'contacts':[r0,r1]}
    if t.kind=='d2':
        if min(t.rc[p:c+1])<2 or gd<.018 or o<.018 or abs(gp)>.035:return None
        if p<t.trigger-5 or c>t.verified_end:return None
        return 'held_target_moves_away_from_goal',gd,{'target_goal_distance_increase_m':gd,'target_motion_m':o,'contacts':[r0,r1],'gripper_target_gap_change_m':gp}
    if t.kind=='wrong_object':
        if min(t.wc[max(p,c-2):c+1])<2 or o>.015:return None
        if e<.025 and abs(q1-q0)<.012:return None
        if c<t.verified_start or p>t.verified_end:return None
        return 'wrong_object_contact_with_gripper_motion',e+abs(q1-q0),{'wrong_object_contacts':[int(t.wc[p]),int(t.wc[c])],'eef_motion_m':e,'opening_change_m':q1-q0,'correct_target_motion_m':o}
    return None

def event_candidates(t):
    a=max(0,t.verified_start-8);b=min(t.n-1,t.verified_end+8)
    found=[]
    for p in range(a,b):
        for c in range(p+2,min(b,p+8)+1):
            f=physical(t,p,c)
            if f is None:continue
            stage,salience,proof=f
            found.append((p,c,stage,salience,proof,0.0,0.0))
    found.sort(key=lambda z:rank_candidate(t,z))
    visible_found=[]
    for p,c,stage,salience,proof,_,_ in found[:12]:
        # This checks observation availability and obvious visual change, not semantic truth.
        main=image_change(t.path,p,c,'main');wrist=image_change(t.path,p,c,'wrist')
        if max(main,wrist)>=1.0:visible_found.append((p,c,stage,salience,proof,main,wrist))
    return visible_found
def rank_candidate(t,z):
    p,c,stage,salience,proof,main,wrist=z
    anchor={'miss':t.close,'drop':t.loss,'wrong_place':t.release,'d1':t.trigger,'d2':t.trigger,'wrong_object':t.verified_start}[t.kind]
    # Prefer clear physical change and a concise interval; no progress labels are used.
    return (abs(p-int(anchor))+abs(c-int(anchor)), -salience, abs((c-p)-4),p,c)
def pair_key(x):return (x['trajectory_dir'],x['previous_frame'],x['current_frame'])
def render_sheet(samples,kind):
    outdir=OUT/'contact_sheets';outdir.mkdir(exist_ok=True)
    index=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    # One card per pair: 2x2 unmodified full frames, 256 px each.
    for page in range(0,len(samples),9):
        chunk=samples[page:page+9];cardw,cardh=560,565
        sheet=Image.new('RGB',(cardw*3,cardh*3),'white');draw=ImageDraw.Draw(sheet)
        for pos,x in enumerate(chunk):
            bx=(pos%3)*cardw;by=(pos//3)*cardh
            draw.text((bx+8,by+4),x['sample_id']+'  '+str(x['previous_frame'])+'->'+str(x['current_frame']),fill='black',font=font)
            for k,(role,view) in enumerate((('previous','main'),('current','main'),('previous','wrist'),('current','wrist'))):
                img=next(z for z in x['images'] if z['role']==role and z['view']==view)
                with Image.open(img['path']) as f:im=f.convert('RGB')
                sheet.paste(im,(bx+8+(k%2)*264,by+34+(k//2)*264))
            index.append(dict(sample_id=x['sample_id'],page=page//9+1,cell=pos+1,previous=x['previous_frame'],current=x['current_frame'],image_sha256={z['role']+'_'+z['view']:z['sha256'] for z in x['images'] if z['role'] in ('previous','current')}))
        path=outdir/f'{kind}_{page//9+1:02d}.png';sheet.save(path)
    dump(outdir/f'{kind}_index.json',index)

def main():
    assert prior.old.readjson(prior.API)['api_submission_enabled'] is False
    if OUT.exists() and any(OUT.iterdir()):
        partial={p.name for p in OUT.iterdir()}
        assert partial=={'holdout_trajectories.json','excluded_training_groups.json'},f'output already exists: {OUT}'
        for name in partial:
            assert prior.old.sha(OUT/name)==prior.old.sha(OLD/name)
    OUT.mkdir(parents=True,exist_ok=True)
    old_rows=rows(OLD/'train_candidates.jsonl')+rows(OLD/'validation_candidates.jsonl')
    bypath=collections.defaultdict(list)
    for x in old_rows:bypath[x['trajectory_dir']].append(x)
    ev={x['trajectory_dir']:x for x in rows(OLD/'resolved_event_evidence.jsonl')}
    hold=prior.old.readjson(OLD/'holdout_trajectories.json');groups=prior.old.readjson(OLD/'excluded_training_groups.json')
    holdgroups={x['group_id'] for x in groups}
    assert not any(x['group_id'] in holdgroups for x in old_rows)
    dump(OUT/'holdout_trajectories.json',hold);dump(OUT/'excluded_training_groups.json',groups)
    mainrows=[];persistent=[];quarantine=[];excluded=[];audit=[];chosen_from_old=set()
    per_type=collections.defaultdict(lambda:collections.Counter())
    for path,xs in sorted(bypath.items()):
        kind=xs[0]['failure_type'];per_type[kind]['old_windows']+=len(xs)
        source=source_for_trajectory(xs);r=ev[path]
        resolved={'source_rows':[{'previous_frame':z['previous_frame'],'current_frame':z['current_frame']} for z in xs], 'revision_evidence':r.get('revision_evidence')}
        t=prior.Traj(source,resolved)
        assert t.kind==kind and all(x['group_id'] not in holdgroups for x in xs)
        candidates=event_candidates(t)
        # event selection is complete before the first progress value is examined.
        candidates.sort(key=lambda z:rank_candidate(t,z))
        pick=candidates[0] if candidates else None
        score_fallback=None
        if pick is not None and not valid_score(t,pick[0],pick[1]):
            nearby=next((z for z in candidates[1:] if abs(z[0]-pick[0])<=2 and abs(z[1]-pick[1])<=2 and z[2]==pick[2] and valid_score(t,z[0],z[1])),None)
            if nearby is not None:
                score_fallback={'first_event_pair':[pick[0],pick[1]],'selected_time_nearby_scored_pair':[nearby[0],nearby[1]],'rule':'same event stage; endpoint shift <=2 frames; only score availability checked, not score value'}
                pick=nearby
        oldpairs=[(z['previous_frame'],z['current_frame']) for z in xs]
        if pick is None:
            for x in xs:
                quarantine.append(dict(trajectory_dir=path,failure_type=kind,old_sample_id=x['sample_id'],original_pair=[x['previous_frame'],x['current_frame']],reason='no_short_window_with_required_physical_transition_and_visible_image_change',status='machine_checked_pending_visual'))
            per_type[kind]['event_not_confirmed_windows']+=len(xs)
            audit.append(dict(trajectory_dir=path,failure_type=kind,original_pairs=oldpairs,selected_pair=None,reason='no event-first pair met conservative physical and visual-change checks',event_file=source['event_file'],resolved_event_status=r['resolved_status']))
            continue
        p,c,stage,salience,proof,mainchange,wristchange=pick
        per_type[kind]['clear_event_trajectories']+=1
        previous=t.rows[p]['P'];current=t.rows[c]['P']
        if previous is None or current is None:
            disposition='quarantine';reason='missing_frozen_v2_score_at_event_aligned_endpoint';direction='unknown'
        else:
            pi=int(math.floor(float(previous)+.5));ci=int(math.floor(float(current)+.5));direction=sign(pi,ci)
            per_type[kind]['physically_selected_'+direction]+=1
            if kind in ('drop','d1','d2'):
                disposition='main' if direction=='decrease' else 'quarantine'
                reason='event_label_consistent_decrease' if direction=='decrease' else ('frozen_integer_label_insensitive_to_regression' if direction=='equal' else 'event_and_frozen_label_direction_conflict')
            else:
                disposition='main' if direction in ('equal','decrease') else 'quarantine'
                reason='event_label_consistent_nonincrease' if disposition=='main' else 'event_and_frozen_label_direction_conflict'
        auditrow=dict(trajectory_dir=path,failure_type=kind,original_pairs=oldpairs,selected_pair=[p,c],critical_frame={'miss':t.close,'drop':t.loss,'wrong_place':t.release,'d1':t.trigger,'d2':t.trigger,'wrong_object':t.verified_start}[kind],stage=stage,selection_reason='earliest clear causal transition with physical change and paired-image change; score read only afterward',score_availability_time_fallback=score_fallback,physical_evidence=proof,visual_screen={'main_mean_rgb_change':mainchange,'wrist_mean_rgb_change':wristchange,'visual_status':'machine_checked_pending_independent_visual_review'},score={'previous_float':previous,'current_float':current,'previous_integer':int(math.floor(float(previous)+.5)) if previous is not None else None,'current_integer':int(math.floor(float(current)+.5)) if current is not None else None,'direction':direction},decision=disposition,decision_reason=reason,event_file=source['event_file'],resolved_event_status=r['resolved_status'],progress_source=source['per_frame_progress_file'],progress_config=source['progress_config'])
        audit.append(auditrow)
        if disposition=='main':
            sourceids=sorted({sid for x in xs for sid in x['source_sample_ids']})
            z=t.make(p,c,stage,source['split'],sourceids)
            z['review_status']='machine_checked_pending_independent_visual_review';z['event_aligned_sampling_version']='v1';z['event_aligned_audit_file']=str(OUT/'selection_audit.jsonl')
            z['physical_evidence']=proof;z['visual_screen']=auditrow['visual_screen'];z['original_short_pairs']=oldpairs
            assert z['assistant_completion'] is None and len(z['images'])==22
            mainrows.append(z);per_type[kind]['main_kept']+=1
            if (p,c) in oldpairs:chosen_from_old.add((path,p,c))
        else:
            quarantine.append(dict(trajectory_dir=path,failure_type=kind,original_pairs=oldpairs,selected_pair=[p,c],reason=reason,physical_evidence=proof,visual_screen=auditrow['visual_screen'],score=auditrow['score'],event_file=source['event_file']))
            per_type[kind]['quarantined_selected']+=1
        for x in xs:
            if (path,x['previous_frame'],x['current_frame']) not in chosen_from_old:
                excluded.append(dict(trajectory_dir=path,failure_type=kind,old_sample_id=x['sample_id'],original_pair=[x['previous_frame'],x['current_frame']],reason='superseded_by_event_aligned_pair_or_quarantine',selected_pair=[p,c],event_stage=stage))
    # No persistent samples by default: a separate causal context and visual check is required.
    no_event_old=sum('old_sample_id' in x for x in quarantine)
    assert len(excluded)+len(chosen_from_old)+no_event_old==len(old_rows)
    assert len({(x['trajectory_dir'],x['previous_frame'],x['current_frame']) for x in mainrows})==len(mainrows)
    dump_rows(OUT/'train_candidates.jsonl',mainrows)
    dump_rows(OUT/'persistent_no_progress_candidates.jsonl',persistent)
    dump_rows(OUT/'quarantined_windows.jsonl',quarantine)
    dump_rows(OUT/'excluded_windows.jsonl',excluded)
    dump_rows(OUT/'selection_audit.jsonl',audit)
    counts=[]
    for kind in KINDS:
        q=per_type[kind]
        counts.append(dict(type=DISPLAY[kind],old_windows=q['old_windows'],event_clear_trajectories=q['clear_event_trajectories'],physically_selected_decrease=q['physically_selected_decrease'],physically_selected_equal=q['physically_selected_equal'],physically_selected_increase=q['physically_selected_increase'],main_kept=q['main_kept'],quarantined_selected=q['quarantined_selected'],event_not_confirmed_old_windows=q['event_not_confirmed_windows']))
        render_sheet([x for x in mainrows if x['failure_type']==kind],kind)
    with (OUT/'counts_by_type.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(counts[0]));w.writeheader();w.writerows(counts)
    dump(OUT/'summary.json',dict(old_windows=len(old_rows),old_trajectories=len(bypath),main_kept=len(mainrows),main_kept_original_pair=len(chosen_from_old),main_kept_new_pair=len(mainrows)-len(chosen_from_old),persistent_kept=len(persistent),quarantine_records=len(quarantine),excluded_old_windows=len(excluded),old_rows_accounted=len(excluded)+len(chosen_from_old)+no_event_old,counts_by_type=counts,api_requests=0,rollouts=0,cot=0,training=0))
    print(json.dumps({'main_kept':len(mainrows),'quarantine':len(quarantine),'excluded_old':len(excluded),'counts':counts},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
