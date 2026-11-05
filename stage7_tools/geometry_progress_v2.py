#!/usr/bin/env python3
"""Task-6-only shared-distance progress prototype; never fits failure rollouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from geometry_progress_v1 import (
    ROOT, REFS, KEYS, load_trajectory, restore_success, sha, dump,
)
from toolkits.libero_sole.compute_contact_progress import (
    _select_robot_points, _select_object_points, _nearest_sum,
)

V1 = ROOT / 'stage7_artifacts/v7_failure_geometry_progress_pilot_20261024_r3'
FORMULA_VERSION = 'task6_two_distance_shared_train_anchor_v2_review_only'


def distance_rows(t, beta):
    """Same two distances as the old implementation; no old normalization."""
    result=[]
    for i,(s,c) in enumerate(zip(t['sem'],t['con'])):
        rn,rp,rr=_select_robot_points(c)
        on,op,orr=_select_object_points(c,s)
        dc,nearest,individual=_nearest_sum(rp,op)
        obj=np.asarray(s['object_pos'],dtype=np.float64)
        goal=np.asarray(s['goal_pos'],dtype=np.float64)
        if obj.shape!=(3,) or goal.shape!=(3,) or not np.isfinite(obj).all() or not np.isfinite(goal).all():
            raise ValueError(f'nonfinite/missing target position at {t["path"]} frame={i}')
        dg=float(np.linalg.norm(obj-goal))
        y=(1-beta)*dc+beta*dg
        result.append(dict(frame=i,d_contact_m=dc,d_goal_3d_m=dg,y=y,
                           robot_points=rn,object_points=on,matched_object_points=[on[j] for j in nearest],
                           nearest_distances_m=individual,robot_point_count=len(rn),object_point_count=len(on),
                           robot_point_source=rr,object_point_source=orr))
    return result


def score_rows(distances, successes, anchors):
    start=anchors['y_start'];goal=anchors['y_goal_anchor']
    if not start>goal+1e-6:
        raise ValueError(f'degenerate or reversed fixed anchor: {start=} {goal=}')
    if len(distances)!=len(successes):raise ValueError('distance/success length mismatch')
    out=[]
    for d,s in zip(distances,successes):
        pd=float(100*np.clip((start-d['y'])/(start-goal),0,1))
        out.append({**d,'P_distance':pd,'P':100.0 if s['success'] else min(pd,99.0),
                    'success_current':bool(s['success']),
                    'success_source':'same_frame_restored_sim_state_official_check_success',
                    'saved_frame_success_field_not_used':True})
    return out


def save_rows(directory, rows):
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'per_frame.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    np.save(directory/'progress_geometry_v2.npy',np.asarray([r['P'] for r in rows],dtype=np.float64))


def plot_curve(directory, rows, v1, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if len(rows)!=len(v1):raise ValueError('v1/v2 length mismatch')
    x=np.arange(len(rows));fig,ax=plt.subplots(figsize=(11,3.6))
    ax.plot(x,[r['P'] for r in rows],label='v2 final',color='#225da8')
    ax.plot(x,[r['P_distance'] for r in rows],label='v2 distance',color='#62a5cf',alpha=.75)
    ax.plot(x,[r['P'] for r in v1],label='v1',color='#888888',alpha=.75)
    ax.scatter([r['frame'] for r in rows if r['success_current']],[r['P'] for r in rows if r['success_current']],s=20,color='#bc2626',label='official current success')
    ax.set(xlabel='saved observation frame',ylabel='progress %',ylim=(-2,102),title=title)
    ax.grid(alpha=.2);ax.legend(ncol=4,fontsize=8);fig.tight_layout()
    fig.savefig(directory/'curve_v1_v2.png',dpi=130);plt.close(fig)


def metrics(old,new):
    old=np.asarray(old,dtype=np.float64);new=np.asarray(new,dtype=np.float64)
    if old.shape!=new.shape:raise ValueError('comparison length mismatch')
    delta=new-old;absolute=np.abs(delta)
    return dict(count=len(old),mae=float(absolute.mean()),median_ae=float(np.median(absolute)),
                p90_ae=float(np.percentile(absolute,90)),mean_signed_difference=float(delta.mean()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--additional-anchor-search',type=Path,default=None)
    args=ap.parse_args();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if any(x.name not in ('cache','tmp','run.log') for x in out.iterdir()):
        raise ValueError(f'output already contains results: {out}')
    refs=json.loads(REFS.read_text());refids=list(dict.fromkeys(r['episode_id'] for r in refs['train_references']))
    if len(refids)!=3 or any(r['split']!='train' for r in refs['train_references']):
        raise ValueError('three fixed same-task train references required')
    split_check=json.loads((V1/'train_reference_split_audit.json').read_text())
    if any(split_check['membership'].get(e)!=['train'] for e in refids):raise ValueError('reference is not verified train-only')
    manifest=json.loads((V1/'input_manifest.json').read_text())
    if len(manifest)!=11 or sum(x['type']=='baseline' for x in manifest)!=3:
        raise ValueError('expected v1 selection of three baselines and eight valid interventions')
    for x in manifest:
        p=Path(x['path'])
        for name,digest in x['source_sha256'].items():
            if sha(p/name)!=digest:raise ValueError(f'source changed: {p/name}')
    dump(out/'input_manifest.json',manifest)
    beta_sources=[]
    for e in refids:
        meta=ROOT/'verified_datasets/rollout_trajectories_v3'/e/'progress_contact_v3_metadata.json'
        contents=json.loads(meta.read_text())
        if contents.get('dataset_version')!='v3_beta075' or contents.get('object_name')!='akita_black_bowl_1' or contents.get('goal_name')!='plate_1':
            raise ValueError(f'wrong beta provenance or task in {meta}')
        beta_sources.append(dict(episode_id=e,path=str(meta),sha256=sha(meta),beta=contents.get('beta'),dataset_version=contents.get('dataset_version')))
    betas=[x['beta'] for x in beta_sources]
    if any(not isinstance(x,(int,float)) or not 0<=x<=1 for x in betas) or len(set(betas))!=1:
        raise ValueError(f'inconsistent or missing actual source beta: {beta_sources}')
    beta=float(betas[0])
    reference_records=[];prepared=[]
    for e in refids:
        t=load_trajectory(ROOT/'verified_datasets/rollout_trajectories_v3'/e)
        distance=distance_rows(t,beta)
        old_meta=json.loads((t['path']/'progress_contact_v3_metadata.json').read_text())
        first=distance[0]
        dc_error=abs(first['d_contact_m']-old_meta['y_contact_first'])
        dg_error=abs(first['d_goal_3d_m']-old_meta['y_goal_first'])
        if dc_error>1e-9 or dg_error>1e-9:
            raise ValueError(f'v2 distance definitions differ from old metadata at {e}: contact={dc_error}, goal={dg_error}')
        success=restore_success(t)
        anchors=[dict(frame=i,y=distance[i]['y'],success_source='same_frame_restored_sim_state_official_check_success') for i,s in enumerate(success) if s['success']]
        rec=dict(episode_id=e,num_saved_observations=t['n'],y_start=distance[0]['y'],
                 success_candidates=anchors,episode_goal_median=float(np.median([a['y'] for a in anchors])) if anchors else None,
                 first_d_contact_m=first['d_contact_m'],first_d_goal_3d_m=first['d_goal_3d_m'],
                 first_robot_point_names=first['robot_points'],first_robot_point_source=first['robot_point_source'],
                 first_object_point_count=first['object_point_count'],first_object_point_source=first['object_point_source'],
                 first_distance_matches_old_metadata=dict(contact_abs_error=dc_error,goal_abs_error=dg_error),
                 old_progress_path=str(t['path']/'progress_contact_v3.npy'),beta_metadata=next(x for x in beta_sources if x['episode_id']==e))
        reference_records.append(rec);prepared.append((e,t,distance,success))
    y_start=float(np.median([x['y_start'] for x in reference_records]))
    goal_episodes=[x for x in reference_records if x['episode_goal_median'] is not None]
    extra_records=[]
    if len(goal_episodes)<2 and args.additional_anchor_search is not None:
        search=json.loads(args.additional_anchor_search.read_text())
        if len(search['selected'])>10 or search['selected']!=sorted(search['selected']):
            raise ValueError('additional anchor search must be sorted and capped at ten')
        if any(r.get('split')!='train' or r.get('episode_id') not in search['selected'] for r in search['records']):
            raise ValueError('additional anchor is not verified in source train snapshot')
        for rec in search['records']:
            if rec['status']!='checked' or not rec['saved_current_success_frames']:continue
            e=rec['episode_id'];t=load_trajectory(ROOT/'verified_datasets/rollout_trajectories_v3'/e)
            d=distance_rows(t,beta);s=restore_success(t)
            found=[i for i,x in enumerate(s) if x['success']]
            if found!=rec['saved_current_success_frames']:
                raise ValueError(f'additional anchor success frames changed: {e}')
            anchors=[dict(frame=i,y=d[i]['y'],success_source='same_frame_restored_sim_state_official_check_success',
                          sim_state_path=str(t['sim'][i]),sim_state_sha256=sha(t['sim'][i])) for i in found]
            extra=dict(episode_id=e,split='train',num_saved_observations=t['n'],success_candidates=anchors,
                       episode_goal_median=float(np.median([a['y'] for a in anchors])),
                       anchor_origin='original_saved_observation',source_snapshot=search['records'][0]['source_snapshot'])
            extra_records.append(extra);goal_episodes.append(extra)
    calibration=dict(status='pending',formula_version=FORMULA_VERSION,beta=beta,beta_sources=beta_sources,
                     y_start=y_start,y_goal_anchor=None,reference_episodes=reference_records,
                     additional_goal_anchor_episodes=extra_records,
                     additional_anchor_search_path=str(args.additional_anchor_search) if args.additional_anchor_search else None,
                     additional_anchor_search_sha256=sha(args.additional_anchor_search) if args.additional_anchor_search else None,
                     required_goal_anchor_episodes=2,available_goal_anchor_episodes=len(goal_episodes),
                     source_split_snapshot=refs['snapshot'],source_split_manifest_sha256=refs['snapshot_manifest_sha256'],
                     event_labels_used_for_calibration=False,old_progress_used_for_calibration=False)
    if len(goal_episodes)<2:
        calibration['status']='blocked_insufficient_saved_current_success_anchors'
        calibration['blocking_reason']=f'only {len(goal_episodes)} of three fixed train reference trajectories have saved observations where restored same-frame official check_success is true; at least two required'
        dump(out/'calibration.json',calibration)
        dump(out/'preflight_config.json',dict(status=calibration['status'],formula_version=FORMULA_VERSION,beta=beta,
                                           y_start=y_start,y_goal_anchor=None,score_computed=False))
        (out/'REVIEW_CN.md').write_text('# v7 简化几何 progress v2：校准阻塞\n\n'
            f'三条指定同任务 train 参考的 beta 均为 {beta}；首帧 y 的中位数 y_start={y_start:.9f}。'
            f'但只有 {len(goal_episodes)} 条参考具有已保存观测对应的官方即时成功帧，要求至少 2 条。'
            '因此 y_goal_anchor 未定义，未计算 11 条试验轨迹的 v2 分数、曲线或同步视频。'
            '旧标签末帧 100%、episode 最终 success、success_once 和动作后 success.npy 均未充当锚点。'
            '参见 calibration.json 的逐条候选帧和元数据来源。\n')
        print(json.dumps(dict(status=calibration['status'],beta=beta,y_start=y_start,goal_anchor_episodes=len(goal_episodes))))
        raise SystemExit(2)
    y_goal=float(np.median([x['episode_goal_median'] for x in goal_episodes]))
    if not y_start>y_goal+1e-6:
        calibration.update(status='blocked_degenerate_anchor',y_goal_anchor=y_goal)
        dump(out/'calibration.json',calibration);raise SystemExit(2)
    calibration.update(status='calibrated',y_goal_anchor=y_goal)
    dump(out/'calibration.json',calibration)
    frozen=dict(version=FORMULA_VERSION,review_only=True,beta=beta,y_start=y_start,y_goal_anchor=y_goal,
                formula='y=(1-beta)*sum_nearest_finger_pad_to_correct_bowl_geom_center_3d+beta*norm(correct_bowl_pos-goal_pos)_3d; P_distance=100*clip((y_start-y)/(y_start-y_goal_anchor),0,1); P=100 if same_frame_official_success else min(P_distance,99)',
                calibration_sha256=sha(out/'calibration.json'),v1_input_manifest_sha256=sha(V1/'input_manifest.json'))
    dump(out/'frozen_config.json',frozen)
    scores={}
    for x in manifest:
        t=load_trajectory(x['path']);d=distance_rows(t,beta);s=restore_success(t)
        rs=score_rows(d,s,frozen);scores[x['id']]=rs
        dest=out/'trajectories'/x['id'];save_rows(dest,rs)
        old=[json.loads(z) for z in (V1/'trajectories'/x['id']/'per_frame.jsonl').read_text().splitlines()]
        plot_curve(dest,rs,old,x['id'])
    comparison=[]
    for e,t,d,s in prepared:
        rs=score_rows(d,s,frozen);dest=out/'train_references'/e;save_rows(dest,rs)
        old=np.load(t['path']/'progress_contact_v3.npy')
        v1=[json.loads(z) for z in (V1/'train_reference_comparison'/e/'per_frame.jsonl').read_text().splitlines()]
        if not len(old)==len(v1)==len(rs):raise ValueError(f'train comparison length mismatch: {e}')
        comparison.append(dict(episode_id=e,old_vs_v2=metrics(old,[r['P'] for r in rs]),
                               old_vs_v1=metrics(old,[r['P'] for r in v1])))
        plot_curve(dest,rs,v1,e)
    dump(out/'train_reference_comparison.json',comparison)
    key=json.loads(KEYS.read_text());lookup={str(Path(x['path']).resolve()):x['id'] for x in manifest};keyrows=[]
    for label,rec in key.items():
        tid=lookup[str(Path(rec['trajectory']).resolve())]
        v1=[json.loads(z) for z in (V1/'trajectories'/tid/'per_frame.jsonl').read_text().splitlines()]
        for query in rec['queries']:
            i=query['frame'];r=scores[tid][i]
            keyrows.append(dict(candidate=label,trajectory_id=tid,frame=i,v1=v1[i]['P'],v2=r['P'],
                                P_distance=r['P_distance'],d_contact_m=r['d_contact_m'],d_goal_3d_m=r['d_goal_3d_m'],
                                y=r['y'],success_current=r['success_current'],images=query['images']))
    dump(out/'keyframe_comparison.json',keyrows)
    print(json.dumps(dict(status='computed',trajectories=len(scores),frames=sum(len(x) for x in scores.values()),beta=beta,y_start=y_start,y_goal_anchor=y_goal)))

if __name__=='__main__':main()
