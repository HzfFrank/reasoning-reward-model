"""Review-only shared two-distance v2 calibration for selected Spatial tasks.

Only current-state official success in saved train observations supplies goal anchors.
Uncalibrated tasks remain explicitly unscored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from geometry_progress_v1 import ROOT, load_trajectory
from geometry_progress_v2 import distance_rows, score_rows, save_rows

SOURCE=ROOT/'verified_datasets/rollout_trajectories_v3'
SNAPSHOT=ROOT/'stage7_artifacts/v7_bulk_grounded_contract_20261012_r1/exports/snapshot_00014003'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
    tmp.replace(path)


def train_ids():
    seen=set()
    with (SNAPSHOT/'train.jsonl').open() as f:
        for line in f:
            x=json.loads(line);e=x['episode_id']
            if e not in seen:seen.add(e);yield e


def select(task_ids):
    groups={k:[] for k in task_ids}
    for e in sorted(train_ids()):
        p=SOURCE/e
        if not p.is_dir():continue
        task=json.loads((p/'task.json').read_text())
        if task['task_id'] in groups and e.endswith('_success'):
            groups[task['task_id']].append(e)
    return groups


def current_success(t):
    """Same saved observation, official predicate; record MuJoCo forward drift."""
    import pickle
    from libero.libero.envs.env_wrapper import ControlEnv
    env=ControlEnv(t['sem'][0]['bddl_file'],use_camera_obs=False,has_offscreen_renderer=False)
    result=[];max_pos_error=0.0
    try:
        env.reset()
        for i,p in enumerate(t['sim']):
            with p.open('rb') as f:state=np.asarray(pickle.load(f))
            env.set_state(state);env.sim.forward()
            if i in (0,t['n']-1) and not np.array_equal(env.get_sim_state(),state):
                raise ValueError(f'sim state restore mismatch: {t["path"]} frame {i}')
            actual=np.asarray(env.sim.data.body_xpos[env.env.obj_body_id['akita_black_bowl_1']])
            expected=np.asarray(t['sem'][i]['object_pos'])
            err=float(np.linalg.norm(actual-expected));max_pos_error=max(max_pos_error,err)
            if err>.003:
                raise ValueError(f'restored target pos mismatch {err:.4f}m: {t["path"]} frame {i}')
            result.append(dict(success=bool(env.check_success())))
    finally:env.env.close()
    return result,max_pos_error


def calibrate_task(task_id,episodes,out):
    if len(episodes)<3:
        return dict(status='blocked_fewer_than_three_train_success_references',task_id=task_id,available=len(episodes))
    checked=[]; beta=None; start=[]; anchor=[]
    for order,e in enumerate(episodes[:13]):
        p=SOURCE/e
        meta_path=p/'progress_contact_v3_metadata.json'
        m=json.loads(meta_path.read_text())
        if m.get('object_name')!='akita_black_bowl_1' or m.get('goal_name')!='plate_1':
            raise ValueError(f'wrong target/goal in {meta_path}')
        b=m.get('beta')
        if not isinstance(b,(int,float)) or not 0<=b<=1:raise ValueError(f'invalid beta {meta_path}')
        if beta is None:beta=float(b)
        if b!=beta:raise ValueError(f'inconsistent task beta at {meta_path}')
        t=load_trajectory(p,require_task=False)
        if t['sem'][0]['task_id']!=task_id:raise ValueError(f'task ID mismatch {e}')
        d=distance_rows(t,beta)
        if order<3:
            start.append(d[0]['y'])
            if abs(d[0]['d_contact_m']-m['y_contact_first'])>1e-9 or abs(d[0]['d_goal_3d_m']-m['y_goal_first'])>1e-9:
                raise ValueError(f'distance mismatch against original metadata: {e}')
        s,max_pos_error=current_success(t)
        matched=[dict(frame=i,y=d[i]['y'],source='same_frame_restored_official_check_success',
                      sim_state_sha256=sha(t['sim'][i])) for i,r in enumerate(s) if r['success']]
        record=dict(episode_id=e,split='train',order=order,beta=float(b),
                    metadata_sha256=sha(meta_path),num_saved_frames=t['n'],
                    initial_y=d[0]['y'] if order<3 else None,
                    current_success_anchors=matched,max_restored_position_error_m=max_pos_error,
                    median_goal_y=float(np.median([r['y'] for r in matched])) if matched else None)
        checked.append(record)
        write(out/f'task{task_id}_calibration.json',dict(status='checking',task_id=task_id,
              checked=checked,initial_reference_episodes=episodes[:3],extra_check_limit=10))
        if matched:anchor.append(record)
        if order>=2 and len(anchor)>=2:break
    y_start=float(np.median(start))
    if len(anchor)<2:
        result=dict(status='blocked_insufficient_current_success_anchors',task_id=task_id,beta=beta,
                    y_start=y_start,y_goal_anchor=None,checked=checked,goal_anchor_count=len(anchor),
                    initial_reference_episodes=episodes[:3],extra_check_limit=10)
    else:
        y_goal=float(np.median([r['median_goal_y'] for r in anchor]))
        result=dict(status='calibrated' if y_start>y_goal+1e-6 else 'blocked_degenerate_anchors',
                    task_id=task_id,beta=beta,y_start=y_start,y_goal_anchor=y_goal,
                    checked=checked,goal_anchor_count=len(anchor),
                    initial_reference_episodes=episodes[:3],extra_check_limit=10,
                    formula='y=(1-beta)*d_contact+beta*d_goal; P_distance=100*clip((y_start-y)/(y_start-y_goal_anchor),0,1); P=100 if same-frame official success else min(P_distance,99)',
                    review_only=True)
    write(out/f'task{task_id}_calibration.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tasks',type=int,nargs='+',required=True);p.add_argument('--trajectory',type=Path,nargs='*',default=[])
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(k not in range(10) for k in a.tasks):raise ValueError('Spatial tasks only')
    groups=select(a.tasks)
    configs={}
    for task in a.tasks:
        configs[task]=calibrate_task(task,groups[task],out/'calibration')
        print(task,configs[task]['status'],flush=True)
    for path in a.trajectory:
        t=load_trajectory(path,require_task=False);task=t['sem'][0]['task_id'];cfg=configs[task]
        if cfg['status']!='calibrated':continue
        d=distance_rows(t,cfg['beta']);s,_=current_success(t)
        rows=score_rows(d,s,cfg)
        dest=out/'progress'/Path(path).name
        save_rows(dest,rows)
        write(dest/'provenance.json',dict(source=str(path.resolve()),source_semantic_sha256=sha(path/'semantic_states.pkl'),
                                    source_contact_sha256=sha(path/'contact_records.pkl'),
                                    config_path=str(out/'calibration'/f'task{task}_calibration.json'),
                                    config_sha256=sha(out/'calibration'/f'task{task}_calibration.json'),review_only=True))
    write(out/'calibration/summary.json',configs)


if __name__=='__main__':main()
