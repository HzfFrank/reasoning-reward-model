#!/usr/bin/env python3
"""Event-aware, review-only short sequential candidates from the frozen 255 rows."""
from __future__ import annotations
import collections,hashlib,json,math,os,pathlib,pickle,sys
import numpy as np
ROOT=pathlib.Path('/project_c20250301/251503016/reasoning')
sys.path.insert(0,str(ROOT/'stage7_tools'))
import split_failure_short_sequential as old
BASE=old.BASE; SOURCE=old.SOURCE; CAPACITY=old.CAPACITY; API=old.API; SUCCESS=old.SUCCESS
PREVIOUS=BASE/'short_interval_20261027_r3'
REVISION=BASE/'revision_20261026_r1'
OUT=pathlib.Path(os.environ.get('SHORT_FAILURE_CORRECTED_OUTPUT',str(BASE/'short_interval_corrected_20261027_r1')))

def jsonline(p):
    with open(p) as f:
        for s in f:
            if s.strip():yield json.loads(s)
def putl(p,rows):
    with p.open('w') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def put(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def key(d):return (d['trajectory_dir'],int(d['previous_frame']),int(d['current_frame']))
def nearest_pair(traj,p,c):
    """Nearest causal time endpoints only; never inspect or optimize score values."""
    if traj.rows[p]['P'] is not None and traj.rows[c]['P'] is not None:return p,c,None
    choices=[]
    for pp in range(max(0,p-2),min(traj.n-1,p+2)+1):
        for cc in range(max(pp+1,c-2),min(traj.n-1,c+2)+1):
            if not 2<=cc-pp<=5:continue
            if traj.rows[pp]['P'] is None or traj.rows[cc]['P'] is None:continue
            choices.append((abs(pp-p)+abs(cc-c),abs(pp-p),abs(cc-c),pp,cc))
    if not choices:return None,None,'no_nearby_scored_pair'
    _,_,_,pp,cc=min(choices)
    return pp,cc,'nearest_scored_observations_after_restore_mismatch'

class Traj(old.Traj):
    def __init__(self,src,resolved):
        self.src=src;self.path=pathlib.Path(src['trajectory_dir']);self.kind=src['failure_type'];self.resolved=resolved
        with (self.path/'semantic_states.pkl').open('rb') as f:self.sem=pickle.load(f)
        with (self.path/'contact_records.pkl').open('rb') as f:self.contacts=pickle.load(f)
        self.actions=np.load(self.path/'executed_actions.npy')
        self.rows=list(jsonline(src['per_frame_progress_file']))
        self.n=len(self.sem)
        assert self.n==len(self.contacts)==len(self.actions)==len(self.rows)
        assert all(x['frame']==i for i,x in enumerate(self.rows))
        self.target=self.sem[0]['object_name'];assert self.target=='akita_black_bowl_1'
        assert int(self.sem[0]['task_id'])==src['task_id'] and int(self.sem[0]['reset_state_id'])==src['reset_state_id']
        assert src['group_id']==f"libero_spatial:task{src['task_id']}:reset{src['reset_state_id']}"
        self.eef=np.asarray([x['eef_pos'] for x in self.sem],float)
        self.obj=np.asarray([x['object_pos'] for x in self.sem],float)
        self.goal=np.asarray([x['goal_pos'] for x in self.sem],float)
        self.gap=np.linalg.norm(self.eef-self.obj,axis=1)
        self.gdist=np.linalg.norm(self.obj-self.goal,axis=1)
        self.rc=np.asarray([old.robot_contacts(c,self.target) for c in self.contacts])
        self.wc=np.asarray([old.wrong_contacts(c) for c in self.contacts]) if self.kind=='wrong_object' else None
        self.support=np.asarray([old.target_support(c,self.target) for c in self.contacts])
        self.event=old.readjson(src['event_file'])
        self.frames=sorted(int(x.stem) for x in (self.path/'images/main').glob('*.png'))
        assert self.frames==list(range(self.n))==sorted(int(x.stem) for x in (self.path/'images/wrist').glob('*.png'))
        self.story=old.uniform_storyboard_frames(self.frames,8);self.images_cache={}
        self.verified_start=min(x['previous_frame'] for x in resolved['source_rows'])
        self.verified_end=max(x['current_frame'] for x in resolved['source_rows'])
        self.trigger=int(self.event.get('trigger',self.verified_start))
        evidence=resolved.get('revision_evidence') or {}
        self.loss=evidence.get('loss') if self.kind=='drop' else None
        if self.kind=='drop' and self.loss is None:
            for i in range(max(3,self.trigger-2),min(self.n-2,self.verified_end+2)):
                if all(self.rc[j]>=2 for j in range(i-3,i)) and self.rc[i]<2 and self.rc[i+1]<2:
                    self.loss=i;break
        self.landing=None;self.recovery=None
        if self.kind=='drop' and self.loss is not None:
            for i in range(self.loss+1,min(self.n-2,self.verified_end+13)):
                if self.rc[i]<2 and self.support[i] and all(self.rc[j]<2 for j in range(i,min(i+3,self.n))):
                    self.landing=i;break
            for i in range((self.landing or self.loss)+1,min(self.n-3,self.verified_end+13)):
                if all(self.rc[j]>=2 for j in range(i,i+3)) and self.obj[i+2,2]-self.obj[i-1,2]>.01:
                    self.recovery=i;break
        self.close=None
        if self.kind=='miss':
            for i in range(max(0,self.trigger-2),min(self.n,self.verified_end+1)):
                if self.actions[i,-1]>.8:self.close=i;break
        self.release=evidence.get('release') if self.kind=='wrong_place' else None
        if self.kind=='wrong_place' and self.release is None:
            for i in range(max(1,self.verified_start),min(self.n,self.verified_end+1)):
                if self.rc[i-1]>=2 and self.rc[i]<2:self.release=i;break
        self.static_kept=0
    def plate_contact(self,i):
        return any(self.target in (s:=str(c.get('body1',''))+' '+str(c.get('body2',''))) and 'plate' in s.lower() for c in self.contacts[i]['contacts'])
    def stage(self,p,c):
        if not 0<=p<c<self.n or not 2<=c-p<=5:return None,'other_invalid_gap'
        if p<self.verified_start-1 or c>min(self.n-1,self.verified_end+12):return None,'other_outside_verified_event'
        move=float(np.linalg.norm(self.eef[c]-self.eef[p]));omove=float(np.linalg.norm(self.obj[c]-self.obj[p]));gap=float(self.gap[c]-self.gap[p]);gd=float(self.gdist[c]-self.gdist[p]);r0,r1=int(self.rc[p]),int(self.rc[c])
        if self.kind=='miss':
            if self.close is None:return None,'latest_event_evidence_insufficient_no_closure'
            if c<self.close:return None,'ordinary_approach_before_grasp_attempt'
            if any(all(self.rc[j]>=2 for j in range(i,i+3)) and self.obj[i+2,2]-self.obj[0,2]>.025 for i in range(max(0,self.close),min(c+1,self.n-2))):return None,'true_recovery_grasp_established'
            if omove>.035 and self.obj[c,2]-self.obj[0,2]>.015:return None,'other_target_push_or_lift_ambiguous'
            if p<=self.close<=c:return 'miss_closure_without_stable_hold',None
            if move>.004 or r0!=r1 or abs(gap)>.004:return 'empty_gripper_departure_or_attempt',None
            if c<=self.verified_end+8:return 'miss_short_unrecovered_aftermath_static',None
            return None,'duplicate_static_or_long_aftermath'
        if self.kind=='drop':
            if self.loss is None:return None,'latest_event_evidence_insufficient_no_loss'
            if c<self.loss:return None,'ordinary_held_transport_before_loss'
            if self.recovery is not None and c>=self.recovery:return None,'true_recovery_regrasp'
            if p<=self.loss<=c:return 'loss_of_hold',None
            if self.landing is None or p<self.landing:
                dz=float(self.obj[p,2]-self.obj[c,2])
                if abs(dz)>.001 or r0!=r1 or self.support[p]!=self.support[c] or omove>.003:return 'falling_or_landing',None
                return 'drop_short_unrecovered_aftermath_static',None
            if move>.004 or omove>.003 or r0!=r1:return 'post_drop_unrecovered_motion',None
            if c<=self.landing+12:return 'drop_short_unrecovered_aftermath_static',None
            return None,'duplicate_static_or_long_aftermath'
        if self.kind=='wrong_place':
            if self.release is None:return None,'latest_event_evidence_insufficient_no_release'
            if c<self.release:return None,'ordinary_transport_before_wrong_release'
            if self.rows[c]['success_current'] is not False:return None,'other_success_or_unknown_current'
            if self.plate_contact(c):return None,'other_plate_edge_or_plate_contact_ambiguous'
            if not self.support[c] or r1>=2:return None,'other_no_stable_off_plate_support'
            xy=float(np.linalg.norm(self.obj[c,:2]-self.goal[c,:2]))
            if xy<=.045:return None,'other_goal_region_ambiguous'
            if p<=self.release<=c:return 'wrong_place_release_and_support',None
            if c<=self.release+8:return 'wrong_place_short_unrecovered_aftermath_static',None
            return None,'duplicate_static_or_long_aftermath'
        if self.kind=='d1':
            if omove>.018 or r0>=2 or r1>=2:return None,'other_target_contact_or_motion'
            broader=min(self.n-1,c+2);older=max(0,p-2)
            continuous=float(self.gap[broader]-self.gap[older])
            if gap>.002 and move>.003:return 'moving_away_unheld',None
            if continuous>.008 and move>.002:return 'continued_wrong_direction_small_local_step',None
            return None,'ordinary_no_wrong_direction_or_static'
        if self.kind=='d2':
            if min(r0,r1)<2:return None,'true_recovery_or_loss_of_hold'
            broader=min(self.n-1,c+2);older=max(0,p-2)
            continuous=float(self.gdist[broader]-self.gdist[older])
            if abs(gap)>.045:return None,'other_possible_loss_of_hold'
            if gd>.002 and omove>.003:return 'held_wrong_direction_transport',None
            if continuous>.008 and omove>.002:return 'continued_held_wrong_direction_small_local_step',None
            return None,'ordinary_no_wrong_transport_or_static'
        if self.kind=='wrong_object':
            w0,w1=int(self.wc[p]),int(self.wc[c])
            if w1<2:return None,'ordinary_before_wrong_object_contact'
            if omove>.018:return None,'other_correct_target_moved'
            if w0<2:return 'wrong_object_grasp_onset',None
            if move>.003:return 'wrong_object_lift_or_transport',None
            return 'wrong_object_short_unrecovered_aftermath_static',None
        return None,'other_unsupported_type'
    def convert(self,p,c,stage,split,sourceids,original_pair):
        row=self.make(p,c,stage,split,sourceids)
        row['is_stall']='static' in stage
        row['review_status']='resolved_event_short_window_machine_checked_pending_visual_review'
        row['resolved_event_evidence_path']=str(OUT/'resolved_event_evidence.jsonl')
        row['original_capacity_pair']=list(original_pair)
        row['endpoint_adjustment']='nearest_valid_time_observation' if (p,c)!=original_pair else None
        return row

def main():
    assert old.readjson(API)['api_submission_enabled'] is False
    assert not OUT.exists(),f'output already exists: {OUT}'
    OUT.mkdir(parents=True)
    source=list(jsonline(SOURCE));assert len(source)==255
    success=old.readjson(SUCCESS)['group_split']
    diagnostics={ (x['trajectory'],x['kind']):x for x in jsonline(REVISION/'diagnostics.jsonl') }
    bytraj=collections.defaultdict(list);identity=[]
    for raw in source:
        r=dict(raw);group_reset=int(r['group_id'].rsplit('reset',1)[1])
        if r['reset_state_id']!=group_reset:
            identity.append(dict(sample_id=r['sample_id'],old_reset_field=r['reset_state_id'],actual_reset_from_group=group_reset,trajectory=r['trajectory_dir'],resolution='checked_against_saved_semantic_state_when_loaded'))
            r['reset_state_id']=group_reset
        bytraj[r['trajectory_dir']].append(r)
    # Resolve later physical revision against actual event identity, path and exact evidence.
    resolved={};evidence_rows=[]
    for path,rs in sorted(bytraj.items()):
        r=rs[0];ev=old.readjson(r['event_file']);old_valid=ev.get('event_valid')
        revision=r.get('revision_evidence');diag=diagnostics.get((path,r['failure_type']))
        status='original_event_valid';why='original event was accepted in source event file'
        if old_valid is False:
            good=(isinstance(revision,dict) and diag is not None and diag.get('audit_result')=='candidate_added' and diag.get('evidence')==revision and
                  diag.get('event_file')==r['event_file'] and diag.get('trajectory')==path and diag.get('kind')==r['failure_type'] and
                  (r['previous_frame'],r['current_frame'])==(revision.get('previous'),revision.get('current')))
            status='revision_physical_check_accepted' if good else 'unresolved_evidence_conflict'
            why='same-trajectory, same-type causal check admitted this exact frame pair' if good else 'old event rejected and later evidence identity/facts did not fully match'
        elif old_valid is None and ev.get('valid') is True:
            window=ev.get('window') or {}
            good=(ev.get('trajectory')==path and ev.get('kind')==r['failure_type'] and ev.get('attempt_status')=='completed' if 'attempt_status' in ev else ev.get('trajectory')==path and ev.get('kind')==r['failure_type'])
            good=good and isinstance(window,dict) and any((x['previous_frame'],x['current_frame'])==(window.get('previous'),window.get('current')) for x in rs)
            status='alternate_schema_physical_check_accepted' if good else 'unresolved_evidence_conflict'
            why='alternate valid=true event check matches trajectory, type and exact reviewed frame pair' if good else 'alternate check did not match reviewed source identity/window'
            if good and revision is None:revision=window
        elif old_valid is None:
            good=(r['failure_type']=='wrong_object' and ev.get('machine_check')=='passed_trajectory_and_score_alignment' and ev.get('trajectory')==path and ev.get('kind')=='wrong_object')
            status='new_wrong_object_physical_check_accepted' if good else 'unresolved_evidence_conflict'
            why='actual wrong-object motion/contact verified in E supplement' if good else 'no explicit event-valid fact'
        resolved[path]=dict(status=status,why=why,source_rows=rs,revision_evidence=revision,diagnostic_file=str(REVISION/'diagnostics.jsonl') if diag else None)
        evidence_rows.append(dict(trajectory_dir=path,task_id=r['task_id'],reset_state_id=r['reset_state_id'],failure_type=r['failure_type'],original_event_file=r['event_file'],original_event_sha256=old.sha(r['event_file']),original_event_valid=old_valid,original_event_reason=ev.get('event_reason'),revision_diagnostic_file=str(REVISION/'diagnostics.jsonl') if diag else None,revision_diagnostic_sha256=old.sha(REVISION/'diagnostics.jsonl') if diag else None,revision_audit_result=diag.get('audit_result') if diag else None,revision_evidence=revision,resolved_status=status,decision_reason=why,source_sample_ids=[x['sample_id'] for x in rs]))
    putl(OUT/'resolved_event_evidence.jsonl',evidence_rows)
    # Reuse the existing full-trajectory choices; do not generate any holdout pair or 22-image input.
    holds=old.readjson(PREVIOUS/'holdout_trajectories.json')
    holdgroups={h['group_id'] for h in holds}
    for h in holds:
        h.pop('short_window_count',None)
        if h['failure_type']=='f_block':h['isolation_status']='reference_only_unassigned'
        elif h['success_data_group_split']=='train':h['isolation_status']='evaluation_reference_with_overlap'
        else:h['isolation_status']='independent_holdout'
        h['holdout_unit']='complete_trajectory';h['no_windows_generated_this_run']=True
    put(OUT/'holdout_trajectories.json',holds)
    excluded_groups=old.readjson(PREVIOUS/'excluded_training_groups.json')
    put(OUT/'excluded_training_groups.json',excluded_groups)
    prior_train=list(jsonline(PREVIOUS/'train_candidates.jsonl'))
    old_train_keys={key(x) for x in prior_train}
    proposals=list(jsonline(CAPACITY));assert len(proposals)==831
    train=[];val=[];excluded=[];outcomes=[];cache={};seen_actual=set();source_pair_seen=set();score_audit_cache={}
    for z in proposals:
        path=z['trajectory_dir'];p=int(z['previous_frame']);c=int(z['current_frame']);basekey=(path,p,c)
        rs=bytraj.get(path);kind=rs[0]['failure_type'] if rs else z['failure_type']
        outcome=dict(trajectory_dir=path,failure_type=kind,previous_frame=p,current_frame=c,source_sample_ids=z.get('source_sample_ids',[]),old_event_valid=old.readjson(rs[0]['event_file']).get('event_valid') if rs else None,original_null_endpoint=False)
        if rs:
            sf=rs[0]['per_frame_progress_file']
            if sf not in score_audit_cache:score_audit_cache[sf]={r['frame']:r for r in jsonline(sf)}
            outcome['original_null_endpoint']=score_audit_cache[sf][p]['P'] is None or score_audit_cache[sf][c]['P'] is None
        def excluded_reason(reason):
            excluded.append(dict(**outcome,reason=reason));outcome['disposition']='excluded';outcome['reason']=reason
        if basekey in source_pair_seen:excluded_reason('other_duplicate_original_capacity_pair');outcomes.append(outcome);continue
        source_pair_seen.add(basekey)
        if rs is None:excluded_reason('other_not_in_frozen_source');outcomes.append(outcome);continue
        src=rs[0]
        # Full-trajectory group holdout is applied before loading or scoring any proposed pair.
        if src['group_id'] in holdgroups:
            outcome.update(disposition='holdout_group_removed_before_slicing',reason='test_group_isolation');outcomes.append(outcome);continue
        if src['split']=='test':excluded_reason('test_group_reserve_not_selected');outcomes.append(outcome);continue
        if resolved[path]['status']=='unresolved_evidence_conflict':excluded_reason('latest_event_evidence_still_unsupported');outcomes.append(outcome);continue
        if success.get(src['group_id'],'unknown')!=src['split']:
            excluded_reason('other_source_success_split_conflict');outcomes.append(outcome);continue
        if path not in cache:cache[path]=Traj(src,resolved[path])
        traj=cache[path]
        outcome['original_null_endpoint']=traj.rows[p]['P'] is None or traj.rows[c]['P'] is None
        pp,cc,adjust=nearest_pair(traj,p,c)
        if pp is None:excluded_reason('missing_progress_no_nearby_scored_observation');outcomes.append(outcome);continue
        if (path,pp,cc) in seen_actual:excluded_reason('other_duplicate_after_endpoint_adjustment');outcomes.append(outcome);continue
        stage,reason=traj.stage(pp,cc)
        if reason:
            if outcome['original_null_endpoint']:outcome['neighbor_pair_checked']=[pp,cc]
            excluded_reason(reason);outcomes.append(outcome);continue
        if 'static' in stage and traj.static_kept>=2:
            excluded_reason('duplicate_static_stage_cap');outcomes.append(outcome);continue
        row=traj.convert(pp,cc,stage,src['split'],z['source_sample_ids'],(p,c))
        row['resolved_event_status']=resolved[path]['status']
        row['resolved_event_decision_reason']=resolved[path]['why']
        if adjust:row['endpoint_adjustment_reason']=adjust
        seen_actual.add((path,pp,cc))
        if 'static' in stage:traj.static_kept+=1
        if src['split']=='train':train.append(row)
        elif src['split']=='validation':val.append(row)
        else:raise AssertionError(src['split'])
        outcome.update(disposition=src['split'],actual_previous_frame=pp,actual_current_frame=cc,stage=stage,endpoint_adjustment=adjust)
        outcomes.append(outcome)
    assert len(outcomes)==831
    putl(OUT/'train_candidates.jsonl',train);putl(OUT/'validation_candidates.jsonl',val)
    putl(OUT/'excluded_windows.jsonl',excluded);putl(OUT/'capacity_pair_outcomes.jsonl',outcomes)
    # Keep C reserve and historical F. No test sample file is written.
    reserves=old.readjson(PREVIOUS/'reserve_trajectories.json');put(OUT/'reserve_trajectories.json',reserves)
    put(OUT/'source_identity_corrections.json',identity)
    newkeys={key(x) for x in train}
    changes=[]
    for x in prior_train:
        changes.append(dict(action='retained' if key(x) in newkeys else 'excluded_or_changed',trajectory_dir=x['trajectory_dir'],failure_type=x['failure_type'],previous_frame=x['previous_frame'],current_frame=x['current_frame'],sample_id=x['sample_id']))
    for x in train:
        if key(x) not in old_train_keys:changes.append(dict(action='new_or_adjusted',trajectory_dir=x['trajectory_dir'],failure_type=x['failure_type'],previous_frame=x['previous_frame'],current_frame=x['current_frame'],sample_id=x['sample_id'],original_capacity_pair=x['original_capacity_pair']))
    putl(OUT/'changes_vs_r3.jsonl',changes)
    bykind=collections.Counter(x['failure_type'] for x in train)
    summary=dict(source_windows=255,source_trajectories=len(bytraj),capacity_pairs=831,train_windows=len(train),validation_windows=len(val),holdout_full_trajectories=len(holds),holdout_windows_generated=0,train_by_type=dict(bykind),train_by_task=dict(collections.Counter(str(x['task_id']) for x in train)),gap_histogram=dict(collections.Counter(str(x['current_frame']-x['previous_frame']) for x in train+val)),excluded_reasons=dict(collections.Counter(x['reason'] for x in excluded)),test_group_pairs_removed=sum(x['disposition']=='holdout_group_removed_before_slicing' for x in outcomes),old_false_pairs=dict(collections.Counter(x['disposition'] for x in outcomes if x['old_event_valid'] is False)),old_false_accepted=sum(x['disposition'] in ('train','validation') for x in outcomes if x['old_event_valid'] is False),old_false_total=sum(x['old_event_valid'] is False for x in outcomes),null_endpoint_pairs=dict(collections.Counter(x['disposition'] for x in outcomes if x['original_null_endpoint'])),null_endpoint_accepted=sum(x['disposition'] in ('train','validation') for x in outcomes if x['original_null_endpoint']),original_r3_train_windows=len(prior_train),retained_r3_train_pairs=len(old_train_keys&newkeys),new_or_adjusted_train_pairs=len(newkeys-old_train_keys),removed_r3_train_pairs=len(old_train_keys-newkeys),API_requests=0,CoT_generated=0,rollouts=0,training_runs=0)
    put(OUT/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
