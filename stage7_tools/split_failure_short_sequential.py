#!/usr/bin/env python3
"""Review-only short sequential windows from frozen, event-verified failure candidates."""
from __future__ import annotations
import collections, csv, hashlib, json, math, os, pathlib, pickle, sys
import numpy as np
ROOT=pathlib.Path('/project_c20250301/251503016/reasoning')
sys.path.insert(0,str(ROOT/'stage7_sft'))
from sampling import uniform_storyboard_frames
from prompts import VERSION,TEMPLATE_SHA256
from protocol import Sample, messages, validate_sample
BASE=ROOT/'stage7_artifacts/v7_spatial_failure_production_20261025_r1/balanced_failure_20261026_r1'
SOURCE=BASE/'ef_final_supplement_20261027_r1/snapshot_20261027_r1/candidates.jsonl'
CAPACITY=BASE/'sampling_capacity_20261027_r1/proposed_pairs.jsonl'
SUCCESS=ROOT/'stage7_artifacts/v7_spatial_failure_production_20261025_r1/manifests/success_baseline.json'
API=BASE/'API_DISABLED.json'
OUT=pathlib.Path(os.environ.get('SHORT_FAILURE_OUTPUT',str(BASE/'short_interval_20261027_r3')))
KINDS=('miss','drop','wrong_place','d1','d2','wrong_object')
TRAIN_KINDS={'miss','drop','d1','d2','wrong_object'}
REP_KIND={'miss':'A','drop':'B','wrong_place':'C','d1':'D1','d2':'D2','wrong_object':'E'}

def readjson(p):return json.loads(pathlib.Path(p).read_text())
def readjsonl(p):
    with open(p) as f:
        for line in f:
            if line.strip():yield json.loads(line)
def dump(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def dumpjsonl(p,ls):
    with p.open('w') as f:
        for x in ls:f.write(json.dumps(x,ensure_ascii=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def norm(v):return np.asarray(v,float)
def robot_contacts(rec,target):
    return sum(bool(x.get('is_robot_object')) for x in rec['contacts'] if target in (str(x.get('body1',''))+' '+str(x.get('body2',''))))
def wrong_contacts(rec):
    return sum(('ramekin' in (n:=str(x.get('body1','')).lower()+' '+str(x.get('body2','')).lower())) and ('gripper' in n or 'finger' in n or 'robot' in n) for x in rec['contacts'])
def target_support(rec,target):
    for x in rec['contacts']:
        n=(str(x.get('body1',''))+' '+str(x.get('body2',''))).lower()
        if target in n and any(s in n for s in ('table','plate','cabinet','ramekin','stove','burner','cookie')):return True
    return False

class Traj:
    def __init__(self,src):
        self.src=src;self.path=pathlib.Path(src['trajectory_dir']);self.kind=src['failure_type']
        self.sem=pickle.load(open(self.path/'semantic_states.pkl','rb'))
        self.contacts=pickle.load(open(self.path/'contact_records.pkl','rb'))
        self.actions=np.load(self.path/'executed_actions.npy')
        self.rows=list(readjsonl(src['per_frame_progress_file']))
        self.n=len(self.sem);assert self.n==len(self.contacts)==len(self.actions)==len(self.rows)
        assert all(x['frame']==i for i,x in enumerate(self.rows))
        self.target=self.sem[0]['object_name']
        assert self.target=='akita_black_bowl_1'
        assert self.sem[0]['task_id']==src['task_id'] and self.sem[0]['reset_state_id']==src['reset_state_id']
        assert src['group_id']==f"libero_spatial:task{src['task_id']}:reset{src['reset_state_id']}"
        self.eef=np.asarray([x['eef_pos'] for x in self.sem],float)
        self.obj=np.asarray([x['object_pos'] for x in self.sem],float)
        self.goal=np.asarray([x['goal_pos'] for x in self.sem],float)
        self.gap=np.linalg.norm(self.eef-self.obj,axis=1)
        self.gdist=np.linalg.norm(self.obj-self.goal,axis=1)
        self.rc=np.asarray([robot_contacts(c,self.target) for c in self.contacts])
        self.wc=np.asarray([wrong_contacts(c) for c in self.contacts]) if self.kind=='wrong_object' else None
        self.support=np.asarray([target_support(c,self.target) for c in self.contacts])
        self.event=readjson(src['event_file'])
        assert self.event.get('event_valid',True) is True
        self.frames=sorted(int(x.stem) for x in (self.path/'images/main').glob('*.png'))
        assert self.frames==list(range(self.n))
        assert self.frames==sorted(int(x.stem) for x in (self.path/'images/wrist').glob('*.png'))
        self.story=uniform_storyboard_frames(self.frames,8)
        self.images_cache={}
    def image(self,frame,view,role):
        key=(frame,view,role)
        if key not in self.images_cache:
            path=(self.path/'images'/view/f'{frame:06d}.png').resolve();assert path.is_file()
            self.images_cache[key]=dict(path=str(path),sha256=sha(path),frame=frame,view=view,role=role.lower(),label=f'{role} frame={frame} view={view} role={role.lower()}')
        return self.images_cache[key]
    def selected(self,p,c):
        """Causal local evidence, never the integer progress direction."""
        if not 0<=p<c<self.n or c-p>5:return None,'invalid_or_long_pair'
        if self.rows[p]['P'] is None or self.rows[c]['P'] is None:return None,'missing_frozen_v2_score_at_endpoint'
        es=self.event;trigger=int(es.get('trigger',self.src['previous_frame']))
        end=int(es.get('evaluated_frame',es.get('current',self.src['current_frame'])))
        # Original reviewed windows delimit the event; no full-rollout slicing.
        if p<min(trigger-2,self.src['previous_frame']) or c>max(end,self.src['current_frame']):return None,'outside_verified_event_interval'
        rc0,rc1=int(self.rc[p]),int(self.rc[c])
        eefmove=float(np.linalg.norm(self.eef[c]-self.eef[p]))
        objmove=float(np.linalg.norm(self.obj[c]-self.obj[p]))
        gd=float(self.gdist[c]-self.gdist[p]);gap=float(self.gap[c]-self.gap[p])
        if self.kind=='miss':
            close=np.any(self.actions[max(p,trigger):c+1,-1]>.8)
            if not close:return None,'normal_approach_before_closure'
            if objmove>.018 or self.obj[c,2]-self.obj[0,2]>.025:return None,'target_moved_or_lifted_not_clear_miss'
            if rc1>=2:return None,'possible_grasp_not_clear_miss'
            if eefmove<.003 and rc0==rc1:return None,'repeated_stationary_empty_gripper'
            return ('closure_without_hold' if rc0<2 else 'lost_transient_contact'),None
        if self.kind=='drop':
            if p<trigger-2:return None,'normal_held_transport_before_drop'
            held=rc0>=2 or any(self.rc[max(0,p-3):p+1]>=2)
            descent=float(self.obj[p,2]-self.obj[c,2])
            if not held and descent<.008:return None,'post_drop_stationary_or_recovery'
            if rc0>=2 and rc1>=2 and descent<.008:return None,'still_held_before_drop'
            if rc0<2 and rc1>=2:return None,'regrasp_recovery'
            if descent<.006 and not (rc0>=2 and rc1<2):return None,'no_loss_or_descent'
            return ('loss_of_hold' if rc0>=2 and rc1<2 else 'descent_or_landing'),None
        if self.kind=='d1':
            if objmove>.016:return None,'target_moved_not_wrong_approach'
            if rc0>=2 or rc1>=2:return None,'target_contact_or_grasp'
            if gap<.005 or eefmove<.006:return None,'no_substantial_wrong_direction_motion'
            return 'moving_away_unheld',None
        if self.kind=='d2':
            if min(rc0,rc1)<2:return None,'hold_not_sustained'
            if gd<.004 or objmove<.004:return None,'not_transporting_away_from_goal'
            if abs(self.gap[c]-self.gap[p])>.035:return None,'possible_loss_of_hold'
            return 'held_wrong_direction_transport',None
        if self.kind=='wrong_object':
            w0,w1=int(self.wc[p]),int(self.wc[c])
            if w1<2:return None,'no_sustained_wrong_object_contact'
            if objmove>.015:return None,'correct_target_moved'
            if eefmove<.006 and w0==w1:return None,'wrong_object_static_repeat'
            return ('wrong_object_grasp_onset' if w0<2 else 'wrong_object_lift_or_transport'),None
        return None,'type_not_for_short_training'
    def make(self,p,c,stage,split,source_ids):
        refs=[]
        for f in self.story:
            for v in ('main','wrist'):refs.append(self.image(f,v,'GLOBAL'))
        for role,f in (('INITIAL',0),('PREVIOUS',p),('CURRENT',c)):
            for v in ('main','wrist'):refs.append(self.image(f,v,role))
        sid='failure_short_'+hashlib.sha256(f'{self.path}|{self.kind}|{p}|{c}|{sha(SOURCE)}'.encode()).hexdigest()[:20]
        pp=int(math.floor(float(self.rows[p]['P'])+.5));cp=int(math.floor(float(self.rows[c]['P'])+.5))
        task=self.src['task'];ep=self.src['episode_id']
        sm=Sample(sid,'sequential',task,split,ep,0,p,c,pp,'geometry_v2',tuple(refs))
        validate_sample(sm,verify_hash=True)
        msg=messages(sm)
        # Audit only actual text blocks; image filesystem paths are image block data.
        visible=' '.join(z['text'] for m in msg for z in m['content'] if z['type']=='text').lower()
        for forbidden in ('failure_type','wrong_object','f_block','geometry_v2','simulator','event_label','teacher json'):
            assert forbidden not in visible
        assert len(refs)==22 and sum(z['type']=='image' for z in msg[1]['content'])==22
        return dict(sample_id=sid,status='review_only_pending_cot',mode='sequential',task_id=self.src['task_id'],task=task,
            failure_type=self.kind,event_subphase=stage,window_reason=f'{stage}: local change supported by saved action/contact/pose within verified event',
            evidence_sources=[str(self.path/'semantic_states.pkl'),str(self.path/'contact_records.pkl'),str(self.path/'executed_actions.npy'),self.src['event_file']],
            is_stall=False,trajectory_dir=str(self.path),episode_id=ep,group_id=self.src['group_id'],reset_state_id=self.src['reset_state_id'],
            source_split=self.src['split'],split=split,source_sample_ids=source_ids,initial_frame=0,previous_frame=p,current_frame=c,
            previous_progress_float=float(self.rows[p]['P']),current_progress_float=float(self.rows[c]['P']),previous_progress=pp,current_progress=cp,
            previous_progress_provenance='geometry_v2',current_progress_provenance='geometry_v2',
            previous_success_current=self.rows[p]['success_current'],current_success_current=self.rows[c]['success_current'],
            progress_config=self.src['progress_config'],progress_config_sha256=self.src['progress_config_sha256'],
            per_frame_progress_file=self.src['per_frame_progress_file'],per_frame_progress_sha256=self.src['per_frame_progress_sha256'],
            prompt_version=VERSION,template_sha256=TEMPLATE_SHA256,storyboard_frames=self.story,images=refs,student_messages=msg,
            review_status='short_window_machine_checked_pending_visual_review',assistant_completion=None)

def main():
    assert readjson(API)['api_submission_enabled'] is False
    assert not OUT.exists() or not any(OUT.iterdir()),f'output exists: {OUT}'
    OUT.mkdir(exist_ok=True)
    source=list(readjsonl(SOURCE));assert len(source)==255
    src_hash=sha(SOURCE);success=readjson(SUCCESS)['group_split']
    bytraj=collections.defaultdict(list);metadata_corrections=[]
    for raw in source:
        x=dict(raw);expected=int(x['group_id'].rsplit('reset',1)[1])
        if x['reset_state_id']!=expected:
            metadata_corrections.append(dict(sample_id=x['sample_id'],source_reset_state_id=x['reset_state_id'],group_reset_state_id=expected,trajectory_dir=x['trajectory_dir']))
            x['reset_state_id']=expected
        bytraj[x['trajectory_dir']].append(x)
    invalid_paths={path for path,rs in bytraj.items() if readjson(rs[0]['event_file']).get('event_valid',True) is not True}
    proposals=list(readjsonl(CAPACITY)); assert len(proposals)==831
    proposed_by_path=collections.defaultdict(list)
    for pair in proposals:proposed_by_path[pair['trajectory_dir']].append(pair)
    eligibility_cache={}
    def has_usable_short_pair(x):
        path=x['trajectory_dir']
        if path in eligibility_cache:return eligibility_cache[path]
        if path in invalid_paths:return False
        traj=Traj(x)
        eligibility_cache[path]=any(traj.selected(z['previous_frame'],z['current_frame'])[0] is not None for z in proposed_by_path[path])
        return eligibility_cache[path]
    # Select entire trajectories before any window enters train.
    hold=[];holdgroups=set();used_traj=set()
    for kind in KINDS:
        options=[]
        for path,rs in bytraj.items():
            if rs[0]['failure_type']!=kind:continue
            x=rs[0];event=readjson(x['event_file']);valid=event.get('event_valid',True)
            if valid is not True:continue
            if kind!='wrong_place' and not has_usable_short_pair(x):continue
            options.append(x)
        picked=[]
        for x in sorted(options,key=lambda x:(x['split']!='test',x['split']=='train',x['task_id'],x['reset_state_id'])):
            if len(picked)>=2:break
            if x['trajectory_dir'] in used_traj:continue
            if picked and x['task_id']==picked[0]['task_id'] and any(y['task_id']!=picked[0]['task_id'] for y in options if y['trajectory_dir'] not in used_traj):continue
            picked.append(x);used_traj.add(x['trajectory_dir']);holdgroups.add(x['group_id'])
        for x in picked:
            s=success.get(x['group_id'],'unknown')
            status=('original_test' if x['split']=='test' and s=='test' else 'evaluation_holdout_success_train_group' if s=='train' else 'reference_only_split_unknown' if s=='unknown' else 'evaluation_holdout_'+s)
            hold.append(dict(type=REP_KIND[kind],failure_type=kind,task_id=x['task_id'],reset_state_id=x['reset_state_id'],group_id=x['group_id'],trajectory_dir=x['trajectory_dir'],source_split=x['split'],success_data_group_split=s,isolation_status=status,
                shares_task_reset_with_success_train=s=='train',must_exclude_success_group_for_independent_test=s=='train',images_main=str(pathlib.Path(x['trajectory_dir'])/'images/main'),images_wrist=str(pathlib.Path(x['trajectory_dir'])/'images/wrist'),
                recommended_test_interval=[min(z['previous_frame'] for z in bytraj[x['trajectory_dir']]),max(z['current_frame'] for z in bytraj[x['trajectory_dir']])],source_event_file=x['event_file']))
    # Historical F natural failure remains reference_only/unassigned.
    refs=readjson(BASE/'ef_final_supplement_20261027_r1/snapshot_20261027_r1/natural_reference_manifest.json')
    fref=next(x for x in refs if x['kind']=='f_block')
    hold.append(dict(type='F',failure_type='f_block',task_id=fref['task'],reset_state_id=fref['reset_state_id'],group_id=fref['group_id'],trajectory_dir=fref['trajectory'],source_split=fref['split'],success_data_group_split=success.get(fref['group_id'],'unknown'),isolation_status='reference_only_unassigned',shares_task_reset_with_success_train=success.get(fref['group_id'])=='train',must_exclude_success_group_for_independent_test=success.get(fref['group_id'])=='train',images_main=str(pathlib.Path(fref['trajectory'])/'images/main'),images_wrist=str(pathlib.Path(fref['trajectory'])/'images/wrist'),recommended_test_interval=[fref['previous'],fref['current']],source_event_file=None,provenance='historical_natural_failure_user_review'))
    excluded_groups=[dict(group_id=g,reason='full_trajectory_evaluation_holdout',success_data_split=success.get(g,'unknown'),exclude_from_new_failure_train=True,exclude_success_train_group_for_independent_test=success.get(g)=='train',trajectories=[h['trajectory_dir'] for h in hold if h['group_id']==g]) for g in sorted(holdgroups)]
    # The F reference is never in the new train pool, but retain its group exclusion request.
    excluded_groups.append(dict(group_id=fref['group_id'],reason='F_reference_only_unassigned',success_data_split=success.get(fref['group_id'],'unknown'),exclude_from_new_failure_train=True,exclude_success_train_group_for_independent_test=success.get(fref['group_id'])=='train',trajectories=[fref['trajectory']]))
    train=[];val=[];testwindows=[];excluded=[];seen=set();cache={}
    for pair in proposals:
        path=pair['trajectory_dir'];rset=bytraj.get(path)
        if not rset:
            excluded.append(dict(trajectory_dir=path,previous_frame=pair['previous_frame'],current_frame=pair['current_frame'],reason='not_in_frozen_source'));continue
        src=rset[0];kind=src['failure_type'];p=pair['previous_frame'];c=pair['current_frame'];key=(path,p,c)
        if key in seen:
            excluded.append(dict(trajectory_dir=path,previous_frame=p,current_frame=c,reason='duplicate_pair'));continue
        seen.add(key)
        if kind=='wrong_place':
            excluded.append(dict(trajectory_dir=path,previous_frame=p,current_frame=c,reason='C_reserve_only_no_new_windows'));continue
        if path in invalid_paths:
            excluded.append(dict(trajectory_dir=path,failure_type=kind,previous_frame=p,current_frame=c,reason='source_event_not_valid'));continue
        if path not in cache:cache[path]=Traj(src)
        stage,why=cache[path].selected(p,c)
        if why:
            excluded.append(dict(trajectory_dir=path,failure_type=kind,previous_frame=p,current_frame=c,reason=why,source_sample_ids=pair['source_sample_ids']));continue
        group=src['group_id'];orig=src['split'];actual=success.get(group,'unknown')
        if actual!=orig:
            excluded.append(dict(trajectory_dir=path,failure_type=kind,previous_frame=p,current_frame=c,reason='source_success_split_conflict',source_split=orig,success_group_split=actual));continue
        if group in holdgroups:
            row=cache[path].make(p,c,stage,'evaluation_holdout' if orig=='train' else orig,pair['source_sample_ids'])
            testwindows.append(row);continue
        if orig=='test':
            excluded.append(dict(trajectory_dir=path,failure_type=kind,previous_frame=p,current_frame=c,reason='original_test_reserve_not_selected_holdout'));continue
        if orig=='validation':val.append(cache[path].make(p,c,stage,'validation',pair['source_sample_ids']));continue
        if orig=='train':train.append(cache[path].make(p,c,stage,'train',pair['source_sample_ids']));continue
        excluded.append(dict(trajectory_dir=path,failure_type=kind,previous_frame=p,current_frame=c,reason='unassigned_source_split'))
    # C is test-only: reuse its already reviewed source pair, without creating C training slices.
    for h in hold:
        if h['failure_type']!='wrong_place':continue
        path=h['trajectory_dir'];src=bytraj[path][0]
        if path not in cache:cache[path]=Traj(src)
        for old in bytraj[path]:
            p,c=old['previous_frame'],old['current_frame']
            if cache[path].rows[p]['P'] is None or cache[path].rows[c]['P'] is None:
                excluded.append(dict(trajectory_dir=path,failure_type='wrong_place',previous_frame=p,current_frame=c,reason='C_holdout_original_pair_missing_v2_score'));continue
            row=cache[path].make(p,c,'confirmed_off_plate_reference','evaluation_holdout',[old['sample_id']])
            row['window_reason']='C test-only: original reviewed wrong-place pair; no extra C slicing or training window'
            testwindows.append(row)
    # Include selected full holdout trajectories without valid short event subwindows in test manifest anyway.
    for h in hold:
        if h['failure_type']=='f_block':continue
        h['short_window_count']=sum(x['trajectory_dir']==h['trajectory_dir'] for x in testwindows)
    assert all(x['group_id'] not in holdgroups for x in train)
    assert all(x['failure_type'] in TRAIN_KINDS for x in train+val)
    allrows=train+val+testwindows
    assert len({x['sample_id'] for x in allrows})==len(allrows)
    assert all(x['assistant_completion'] is None for x in allrows)
    assert all(x['current_progress']==int(math.floor(x['current_progress_float']+.5)) for x in allrows)
    dumpjsonl(OUT/'train_candidates.jsonl',train)
    dumpjsonl(OUT/'validation_candidates.jsonl',val)
    dumpjsonl(OUT/'holdout_windows.jsonl',testwindows)
    dumpjsonl(OUT/'excluded_windows.jsonl',excluded)
    dump(OUT/'holdout_trajectories.json',hold)
    dump(OUT/'source_metadata_corrections.json',metadata_corrections)
    dump(OUT/'excluded_training_groups.json',excluded_groups)
    reserve=[dict(trajectory_dir=p,task_id=rs[0]['task_id'],reset_state_id=rs[0]['reset_state_id'],failure_type=rs[0]['failure_type'],group_id=rs[0]['group_id'],source_split=rs[0]['split'],reason='C_reserve_only' if rs[0]['failure_type']=='wrong_place' else 'original_test_not_chosen' if rs[0]['split']=='test' and p not in used_traj else 'source_retained') for p,rs in bytraj.items() if rs[0]['failure_type']=='wrong_place' and p not in used_traj or rs[0]['split']=='test' and p not in used_traj]
    reserve.extend(dict(trajectory_dir=x['trajectory'],task_id=x['task'],reset_state_id=x['reset_state_id'],failure_type=x['kind'],group_id=x['group_id'],source_split=x['split'],reason='historical_natural_reference_only') for x in refs)
    dump(OUT/'reserve_trajectories.json',reserve)
    gap=collections.Counter(x['current_frame']-x['previous_frame'] for x in allrows)
    bykind=collections.Counter(x['failure_type'] for x in train);bytask=collections.Counter(x['task_id'] for x in train)
    exreason=collections.Counter(x['reason'] for x in excluded)
    summary=dict(source=str(SOURCE),source_sha256=src_hash,capacity_reference=str(CAPACITY),capacity_pairs=831,source_windows=255,source_trajectories=len(bytraj),invalid_source_event_trajectories=len(invalid_paths),source_reset_metadata_corrections=len(metadata_corrections),holdout_trajectories=len(hold),train_trajectories=len({x['trajectory_dir'] for x in train}),train_windows=len(train),validation_windows=len(val),holdout_windows=len(testwindows),train_by_type=dict(bykind),train_by_task={str(k):v for k,v in sorted(bytask.items())},gap_histogram={str(k):v for k,v in sorted(gap.items())},excluded_reasons=dict(exreason),excluded_total=len(excluded),API_requests=0,CoT_generated=0,training_runs=0,student_prompt_version=VERSION,student_template_sha256=TEMPLATE_SHA256,api_disabled_file=str(API))
    dump(OUT/'summary.json',summary)
    with (OUT/'counts_by_task_type.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['task','type','source_trajectories','holdout_trajectories','train_trajectories','train_windows','validation_windows','holdout_windows'])
        for task in range(10):
            for kind in KINDS:
                srcpaths={p for p,rs in bytraj.items() if rs[0]['task_id']==task and rs[0]['failure_type']==kind}
                if not srcpaths:continue
                filt=lambda rows:[x for x in rows if x['task_id']==task and x['failure_type']==kind]
                w.writerow([task,kind,len(srcpaths),len(filt(hold)),len({x['trajectory_dir'] for x in filt(train)}),len(filt(train)),len(filt(val)),len(filt(testwindows))])
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
