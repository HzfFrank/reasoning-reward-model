#!/usr/bin/env python3
"""Review-only deterministic progress for LIBERO spatial task 6."""
import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/project_c20250301/251503016/reasoning')
PILOT = ROOT / 'stage7_artifacts/v7_perturbed_rollout_pilot_20261023_r1'
KEYS = ROOT / 'stage7_artifacts/v7_failure_keyframe_progress_20261023_r1/selection/keyframes.json'
REFS = ROOT / 'stage7_artifacts/v7_failure_keyframe_progress_20261023_r1/calibration/references.json'
RLINF = Path('/project_c20250301/251503016/RLinf')
sys.path.insert(0, str(RLINF))
from toolkits.libero_sole.compute_contact_progress import _select_robot_points, _select_object_points


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def load_trajectory(path, require_task=True):
    path = Path(path).resolve()
    meta = json.loads((path / 'metadata.json').read_text())
    with open(path / 'semantic_states.pkl', 'rb') as f:
        sem = pickle.load(f)
    with open(path / 'contact_records.pkl', 'rb') as f:
        con = pickle.load(f)
    main = sorted((path / 'images/main').glob('*.png'))
    wrist = sorted((path / 'images/wrist').glob('*.png'))
    sim = sorted((path / 'sim_states').glob('*.pkl'))
    n = len(main)
    if not (n == len(wrist) == len(sim) == len(sem) == len(con)):
        raise ValueError(f'unaligned source lengths {path}: {n}, {len(wrist)}, {len(sim)}, {len(sem)}, {len(con)}')
    if [int(x.stem) for x in main] != list(range(n)) or [int(x.stem) for x in wrist] != list(range(n)) or [int(x.stem) for x in sim] != list(range(n)):
        raise ValueError(f'nonconsecutive or mismatched frame IDs: {path}')
    if require_task and any(s.get('task_id') != 6 or s.get('object_name') != 'akita_black_bowl_1' or s.get('goal_name') != 'plate_1' for s in sem):
        raise ValueError(f'unsupported task/object/destination: {path}')
    return dict(path=path, meta=meta, sem=sem, con=con, main=main, wrist=wrist, sim=sim, n=n)


def contact_flags(record, target):
    left = right = table = plate = False
    for c in record['contacts']:
        g1, g2 = c['geom1'], c['geom2']
        if not (target in g1 or target in g2 or 'black_bowl' in g1 or 'black_bowl' in g2):
            continue
        names = (g1, g2)
        left |= any('finger1_pad' in g for g in names)
        right |= any('finger2_pad' in g for g in names)
        table |= any('table_collision' in g for g in names)
        plate |= any('plate' in g for g in names)
    return bool(left), bool(right), bool(table), bool(plate)


def hysteresis(raw, confirm, release):
    state = False
    yes = no = 0
    result = []
    for x in raw:
        yes = yes + 1 if x else 0
        no = no + 1 if not x else 0
        if not state and yes >= confirm:
            state = True
        if state and no >= release:
            state = False
        result.append(state)
    return result


def raw_features(t):
    rows = []
    for frame, (s, c) in enumerate(zip(t['sem'], t['con'])):
        rn, rp, rs = _select_robot_points(c)
        on, op, os = _select_object_points(c, s)
        d = np.linalg.norm(rp[:, None, :] - op[None, :, :], axis=2).min(axis=1)
        left, right, table, plate = contact_flags(c, s['object_name'])
        obj = np.asarray(s['object_pos'], dtype=float)
        goal = np.asarray(s['goal_pos'], dtype=float)
        rows.append(dict(frame=frame, d_A_m=float(d.mean()), d_xy_m=float(np.linalg.norm(obj[:2]-goal[:2])), object_z_m=float(obj[2]), goal_z_m=float(goal[2]), left_finger_contact=left, right_finger_contact=right, table_contact=table, plate_contact=plate, robot_points=rn, object_points=on, robot_point_source=rs, object_point_source=os))
    return rows


def restore_success(t):
    sys.path.insert(0, str(RLINF / '.venv/libero'))
    from libero.libero.envs.env_wrapper import ControlEnv
    env = ControlEnv(t['sem'][0]['bddl_file'], use_camera_obs=False, has_offscreen_renderer=False)
    out = []
    try:
        env.reset()
        for frame, p in enumerate(t['sim']):
            with open(p, 'rb') as f:
                state = np.asarray(pickle.load(f))
            env.set_state(state)
            env.sim.forward()
            if frame == 0 or frame == t['n']-1:
                if not np.array_equal(env.get_sim_state(), state):
                    raise ValueError(f'restored state mismatch at {t["path"]} frame={frame}')
                obj = np.asarray(t['sem'][frame]['object_pos'])
                actual = env.sim.data.body_xpos[env.env.obj_body_id['akita_black_bowl_1']]
                # Saved semantic positions precede an explicit MuJoCo forward; the
                # forward geometry can shift a few tenths of a millimetre.
                if np.linalg.norm(obj-actual) > 1e-3:
                    raise ValueError(f'restored bowl position mismatch at {t["path"]} frame={frame}: {obj}, {actual}')
            contacts=[]
            for j in range(env.sim.data.ncon):
                c=env.sim.data.contact[j]
                contacts.append(dict(geom1=env.sim.model.geom_id2name(c.geom1),geom2=env.sim.model.geom_id2name(c.geom2)))
            left,right,table,plate=contact_flags(dict(contacts=contacts),t['sem'][frame]['object_name'])
            out.append(dict(success=bool(env.check_success()),left=left,right=right,table=table,plate=plate))
    finally:
        env.env.close()
    return out


def compute(t, f, cfg, success):
    d0 = f[0]['d_A_m']; xy0 = f[0]['d_xy_m']; near = cfg['d_Anear_m']; goal = cfg['r_goal_m']
    if d0-near <= 0.005 or xy0-goal <= 0.005:
        raise ValueError(f'invalid denominator for {t["path"]}: d_A0={d0}, near={near}, d_xy0={xy0}, goal={goal}')
    grasp_raw=[q['left'] and q['right'] for q in success]
    grasps = hysteresis(grasp_raw, cfg['grasp_confirm_observations'], cfg['grasp_release_observations'])
    support_raw = [q['plate'] and r['d_xy_m'] < goal and r['goal_z_m'] <= r['object_z_m'] for r,q in zip(f,success)]
    supports = hysteresis(support_raw, cfg['support_confirm_observations'], cfg['support_release_observations'])
    out = []
    for i,r in enumerate(f):
        h = r['object_z_m']-f[0]['object_z_m']
        A = float(np.clip((d0-r['d_A_m'])/(d0-near),0,1))
        L = float(np.clip((h-cfg['h_clear_m'])/(cfg['h_lift_m']-cfg['h_clear_m']),0,1))
        # Table support may coexist with a brief two-finger touch; never award full lift then.
        H = (0.5+0.5*L) if grasps[i] else 0.0
        if success[i]['table'] and H > 0.5:
            H = 0.5
        D = float(np.clip((xy0-r['d_xy_m'])/(xy0-goal),0,1))
        S = float(supports[i])
        ar = max(A,D)
        raw = 100*(cfg['approach_weight']*ar+cfg['grasp_lift_weight']*H+cfg['transport_weight']*D+cfg['correct_support_weight']*S)
        p = 100.0 if success[i]['success'] else min(raw,99.0)
        row={**{k:v for k,v in r.items() if k not in ('robot_points','object_points')}, 'height_m':h, 'grasp_valid':bool(grasps[i]), 'support_valid':bool(supports[i]), 'success_current':bool(success[i]['success']), 'success_source':'restored_same_frame_official_check_success', 'A':A,'H':H,'D':D,'S':S,'A_retained':ar,'P_raw':raw,'P':p,'uncertainty':[]}
        for k in ('left','right','table','plate'):
            row[f'restored_{k}_contact']=bool(success[i][k])
        row['saved_restored_contact_disagreement']=any(row[f'restored_{k}_contact'] != row[dict(left='left_finger_contact',right='right_finger_contact',table='table_contact',plate='plate_contact')[k]] for k in ('left','right','table','plate'))
        out.append(row)
    return out


def write_rows(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    with (path/'per_frame.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+'\n')
    np.save(path/'progress_geometry_v1.npy', np.asarray([r['P'] for r in rows], dtype=np.float64))


def plot(path, rows, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = [r['frame'] for r in rows]
    fig, ax = plt.subplots(figsize=(11,3.5))
    ax.plot(x,[r['P'] for r in rows],label='P',color='black')
    for name in ['A','H','D','S']:
        ax.plot(x,[100*r[name] for r in rows],label=name,alpha=.7)
    ax.set(xlabel='saved observation frame',ylabel='percent / component ×100',ylim=(-2,102),title=title)
    ax.grid(alpha=.2); ax.legend(ncol=5,fontsize=8)
    fig.tight_layout(); fig.savefig(path/'curve_components.png',dpi=130); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); args=ap.parse_args()
    out=Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        # The shell wrapper precreates only cache and tmp.
        if any(p.name not in ('cache','tmp','run.log') for p in out.iterdir()):
            raise ValueError(f'nonempty output: {out}')
    out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'stage7_tools/geometry_progress_v1.json').read_text())
    key=json.loads(KEYS.read_text())
    refs=json.loads(REFS.read_text())['train_references']
    refids=list(dict.fromkeys(r['episode_id'] for r in refs))
    if len(refids)!=3 or any(r['split']!='train' for r in refs):
        raise ValueError('expected exactly three same-task train references')
    snapshot=Path(json.loads(REFS.read_text())['snapshot'])
    split_membership={rid:[] for rid in refids}
    for split in ('train','validation','test'):
        with (snapshot/f'{split}.jsonl').open() as fh:
            for line in fh:
                eid=json.loads(line).get('episode_id')
                if eid in split_membership and split not in split_membership[eid]:
                    split_membership[eid].append(split)
    if any(split_membership[rid]!=['train'] for rid in refids):
        raise ValueError(f'train reference split mismatch: {split_membership}')
    dump(out/'train_reference_split_audit.json',dict(snapshot=str(snapshot),membership=split_membership))
    trajectories=[]
    bdir=PILOT/'trajectories/baseline'
    for p in sorted(bdir.glob('*')):
        if (p/'metadata.json').exists(): trajectories.append((f'baseline_{len(trajectories)}',p,'baseline'))
    for name,typ in [('miss','miss'),('drop','drop'),('wrong_place_offset12','wrong_place')]:
        for p in sorted((PILOT/'trajectories'/name).glob('*')):
            ev=p/'event_summary.json'
            if ev.exists() and json.loads(ev.read_text()).get('event_valid'):
                trajectories.append((f'{typ}_{len([x for x in trajectories if x[2]==typ])}',p,typ))
    if len(trajectories)!=11 or len([x for x in trajectories if x[2]=='baseline'])!=3:
        raise ValueError(f'expected 3 baselines + 8 valid perturbations, found {len(trajectories)}')
    prepared=[]; manifest=[]
    for tid,p,typ in trajectories:
        t=load_trajectory(p); f=raw_features(t)
        pair=json.loads((p/'pair_manifest.json').read_text()) if (p/'pair_manifest.json').exists() else None
        ev=json.loads((p/'event_summary.json').read_text()) if (p/'event_summary.json').exists() else None
        manifest.append(dict(id=tid,episode_id=t['meta']['episode_id'],path=str(t['path']),frames=t['n'],reset_state_id=t['sem'][0]['reset_state_id'],type=typ,pair_manifest=pair,event_valid=ev.get('event_valid') if ev else None,source_sha256={q:sha(p/q) for q in ['metadata.json','semantic_states.pkl','contact_records.pkl']},first_main_sha256=sha(t['main'][0]),last_wrist_sha256=sha(t['wrist'][-1])))
        prepared.append((tid,t,f,typ))
    dump(out/'input_manifest.json',manifest)
    # Calibrate using train references only; no old progress values enter this step.
    near_obs=[]; train_prepared=[]
    for rid in refids:
        t=load_trajectory(ROOT/'verified_datasets/rollout_trajectories_v3'/rid)
        f=raw_features(t)
        g=hysteresis([r['left_finger_contact'] and r['right_finger_contact'] for r in f],3,2)
        vals=[r['d_A_m'] for i,r in enumerate(f) if g[i] and r['object_z_m']-f[0]['object_z_m']>=cfg['h_clear_m'] and not r['table_contact']]
        if not vals: raise ValueError(f'no confirmed grasped-and-lifted train reference frames: {rid}')
        near_obs.extend(vals); train_prepared.append((rid,t,f))
    cfg['d_Anear_m']=float(np.median(near_obs))
    cfg['d_Anear_reference_episodes']=refids
    cfg['d_Anear_reference_frame_count']=len(near_obs)
    dump(out/'frozen_config.json',cfg)
    results={}
    for tid,t,f,typ in prepared:
        succ=restore_success(t)
        rows=compute(t,f,cfg,succ)
        d=out/'trajectories'/tid; write_rows(d,rows); plot(d,rows,tid)
        results[tid]=rows
    # Old success scale: exact same saved observations and frames, read only for evaluation.
    comparisons=[]
    for rid,t,f in train_prepared:
        old=np.load(t['path']/'progress_contact_v3.npy')
        if len(old)!=t['n']: raise ValueError(f'old progress frame mismatch: {rid}')
        new=compute(t,f,cfg,restore_success(t))
        d=out/'train_reference_comparison'/rid; write_rows(d,new); plot(d,new,rid)
        for i,r in enumerate(new): comparisons.append(dict(episode=rid,frame=i,old=float(old[i]),new=r['P'],abs_error=abs(r['P']-float(old[i])),A=r['A'],H=r['H'],D=r['D'],S=r['S']))
    dump(out/'train_reference_comparison/metrics.json',dict(episodes=refids,frames=len(comparisons),mae=float(np.mean([x['abs_error'] for x in comparisons])),median_ae=float(np.median([x['abs_error'] for x in comparisons])),p90_ae=float(np.percentile([x['abs_error'] for x in comparisons],90))))
    with (out/'train_reference_comparison/per_frame.jsonl').open('w') as fh:
        for x in comparisons: fh.write(json.dumps(x)+'\n')
    lookup={str(m['path']):m['id'] for m in manifest}
    keys=[]
    for label,rec in key.items():
        tid=lookup[str(Path(rec['trajectory']).resolve())]
        for q in rec['queries']:
            row=results[tid][q['frame']]
            keys.append(dict(candidate=label,trajectory_id=tid,frame=q['frame'],P=row['P'],rounded=round(row['P']),A=row['A'],H=row['H'],D=row['D'],S=row['S'],grasp_valid=row['grasp_valid'],table_contact=row['table_contact'],plate_contact=row['plate_contact'],success_current=row['success_current'],main=q['images'][0]['path'],wrist=q['images'][1]['path']))
    dump(out/'keyframe_comparison.json',keys)
    dump(out/'summary.json',dict(trajectory_count=len(manifest),total_frames=sum(x['frames'] for x in manifest),trajectory_ids=[x['id'] for x in manifest],keyframes=len(keys),config_sha256=sha(out/'frozen_config.json')))
    print(json.dumps(dict(trajectories=len(manifest),frames=sum(x['frames'] for x in manifest),keyframes=len(keys),old_scale_mae=float(np.mean([x['abs_error'] for x in comparisons])))))

if __name__=='__main__': main()
