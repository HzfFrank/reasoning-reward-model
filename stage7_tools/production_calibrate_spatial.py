"""Review-only per-task v2 calibration and old-success-scale screening for production."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from calibrate_spatial_failure_coverage import SOURCE, current_success, select, sha, write
from geometry_progress_v1 import load_trajectory
from geometry_progress_v2 import distance_rows, score_rows

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / 'stage7_artifacts/v7_spatial_failure_coverage_pilot_20261024_r1/calibration'
TASK4 = ROOT / 'stage7_artifacts/v7_spatial_def_sequential_cot_20261025_r1/calibration/task4_calibration.json'
TASK6 = ROOT / 'stage7_artifacts/v7_failure_geometry_progress_v2_20261024_r4/frozen_config.json'


def new_calibration(task, episodes, dest):
    if len(episodes) < 3:
        return dict(status='blocked_fewer_than_three_train_references', task_id=task, available=len(episodes))
    starts=[]; checked=[]; anchored=[]; beta=None
    for order, episode in enumerate(episodes):
        path=SOURCE/episode
        meta_path=path/'progress_contact_v3_metadata.json'
        m=json.loads(meta_path.read_text())
        if m.get('object_name')!='akita_black_bowl_1' or m.get('goal_name')!='plate_1':
            raise ValueError(f'wrong task object/goal {episode}')
        b=m.get('beta')
        if not isinstance(b,(int,float)) or not 0<=b<=1 or (beta is not None and b!=beta):
            raise ValueError(f'missing/inconsistent beta {episode}')
        beta=float(b)
        t=load_trajectory(path,require_task=False)
        if t['sem'][0]['task_id']!=task:raise ValueError(f'task mismatch {episode}')
        dist=distance_rows(t,beta)
        if order<3:
            starts.append(dist[0]['y'])
            if abs(dist[0]['d_contact_m']-m['y_contact_first'])>1e-9 or abs(dist[0]['d_goal_3d_m']-m['y_goal_first'])>1e-9:
                raise ValueError(f'distance/old metadata mismatch {episode}')
        success,err=current_success(t)
        frames=[dict(frame=i,y=dist[i]['y'],source='same_frame_restored_official_check_success',
                     sim_state_sha256=sha(t['sim'][i])) for i,s in enumerate(success) if s['success']]
        record=dict(episode_id=episode,split='train',order=order,beta=beta,metadata_sha256=sha(meta_path),
                    num_saved_frames=t['n'],initial_y=dist[0]['y'] if order<3 else None,
                    current_success_anchors=frames,max_restored_position_error_m=err,
                    median_goal_y=float(np.median([f['y'] for f in frames])) if frames else None)
        checked.append(record)
        if frames:anchored.append(record)
        write(dest,dict(status='checking',task_id=task,beta=beta,checked=checked,
                        goal_anchor_count=len(anchored),remaining_train_episodes=len(episodes)-len(checked)))
        print(json.dumps(dict(task=task,episode=episode,anchors=len(anchored),remaining=len(episodes)-len(checked))),flush=True)
        if order>=2 and len(anchored)>=2:break
    if len(anchored)<2:
        return dict(status='blocked_insufficient_current_success_anchors',task_id=task,beta=beta,
                    y_start=float(np.median(starts)),y_goal_anchor=None,checked=checked,
                    goal_anchor_count=len(anchored),review_only=True)
    start=float(np.median(starts));goal=float(np.median([r['median_goal_y'] for r in anchored]))
    return dict(status='calibrated' if start>goal else 'blocked_degenerate_anchors',task_id=task,beta=beta,
                y_start=start,y_goal_anchor=goal,checked=checked,goal_anchor_count=len(anchored),
                initial_reference_episodes=episodes[:3],calibration_quality='multi_episode',review_only=True,
                formula='y=(1-beta)*d_contact+beta*d_goal; P_distance=100*clip((y_start-y)/(y_start-y_goal_anchor),0,1); P=100 if same-frame official success else min(P_distance,99)')


def screen(task, cfg, refs):
    residuals=[]; episode_reports=[]
    for e in refs[:3]:
        p=SOURCE/e;t=load_trajectory(p,require_task=False)
        dist=distance_rows(t,cfg['beta'])
        success,_=current_success(t)
        scores=score_rows(dist,success,cfg)
        old=np.load(p/'progress_contact_v3.npy')
        if len(old)!=len(scores):raise ValueError(f'old score length mismatch {e}')
        new=np.array([r['P'] for r in scores],dtype=float)
        delta=new-old
        residuals.extend(delta.tolist())
        episode_reports.append(dict(episode=e,frames=len(old),mae=float(np.mean(np.abs(delta))),
                                    p90=float(np.percentile(np.abs(delta),90)),
                                    mean_signed=float(np.mean(delta)),
                                    early=dict(frame=0,old=float(old[0]),new=float(new[0])),
                                    middle=dict(frame=len(old)//2,old=float(old[len(old)//2]),new=float(new[len(old)//2])),
                                    late=dict(frame=len(old)-1,old=float(old[-1]),new=float(new[-1]))))
    arr=np.array(residuals)
    mae=float(np.mean(np.abs(arr)));p90=float(np.percentile(np.abs(arr),90))
    return dict(task_id=task,calibration_only_train_references=refs[:3],frame_count=len(arr),
                mae=mae,median_abs=float(np.median(np.abs(arr))),p90_abs=p90,
                mean_signed=float(np.mean(arr)),scale_review_required=mae>10 or p90>20,
                screening_thresholds=dict(mae=10,p90=20),episodes=episode_reports)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    groups=select(list(range(10)))
    report={}
    for task in range(10):
        target=out/f'task{task}_calibration.json'
        if target.exists() and json.loads(target.read_text()).get('status')=='calibrated':
            cfg=json.loads(target.read_text());source='resumed_local_calibration'
        elif task==4:
            cfg=json.loads(TASK4.read_text());source=str(TASK4)
        elif task==6:
            cfg=json.loads(TASK6.read_text());cfg=dict(cfg,task_id=6,status='calibrated',calibration_quality='multi_episode');source=str(TASK6)
        elif (PRIOR/f'task{task}_calibration.json').exists() and json.loads((PRIOR/f'task{task}_calibration.json').read_text()).get('status')=='calibrated':
            source=str(PRIOR/f'task{task}_calibration.json');cfg=json.loads(Path(source).read_text())
        else:
            source='computed_from_sorted_train_success_references';cfg=new_calibration(task,groups[task],target)
        cfg['task_id']=task
        cfg['review_only']=True
        cfg['calibration_source']=source
        cfg['calibration_source_sha256']=sha(source) if Path(source).is_file() else None
        write(target,cfg)
        if cfg['status']=='calibrated':
            comparison=screen(task,cfg,groups[task])
        else:
            comparison=dict(task_id=task,status=cfg['status'],scale_review_required=True)
        comparison['calibration_path']=str(target);comparison['calibration_sha256']=sha(target)
        report[str(task)]=comparison
        write(out/'scale_screen.json',report)
        print(json.dumps(dict(task=task,status=cfg['status'],mae=comparison.get('mae'),p90=comparison.get('p90_abs'),
                              scale_review_required=comparison['scale_review_required'])),flush=True)


if __name__=='__main__':main()
