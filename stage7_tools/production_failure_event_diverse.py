"""Conservative event check and failure-only window selection for new rollouts."""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
from production_failure_score import score
from production_failure_export import atomic_json, digest


def distance(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def locate(run: Path, task: int, trial: int, kind: str, tag: str) -> Path:
    suffix = f'_{tag}' if tag else ''
    matches = []
    for p in (run / 'trajectories').glob(f'task{task}_{kind}{suffix}*/episode_*'):
        source = json.loads((p / 'task.json').read_text())
        if source['task_id'] == task and source['reset_state_id'] == task * 50 + trial:
            matches.append(p)
    if len(matches) != 1:
        raise ValueError(f'expected one exact trajectory for {task}/{trial}/{kind}/{tag}, found {len(matches)}')
    return matches[0]


def evaluate(run: Path, task: int, trial: int, kind: str, tag: str,
             revision: str = '', reuse_score: bool = False,
             score_progress: bool = True, baseline_run: Path | None = None) -> dict:
    suffix = f'_{tag}' if tag else ''
    prefix = f'task{task}_{kind}'
    source = run / 'logs' / f'{prefix}_trial{trial:02d}{suffix}_steps.json'
    summary = run / 'logs' / f'{prefix}_trial{trial:02d}{suffix}_summary.json'
    if not source.is_file() or not summary.is_file():
        raise FileNotFoundError(f'missing step/summary logs: {source}')
    steps = json.loads(source.read_text())
    meta = json.loads(summary.read_text())
    traj = locate(run, task, trial, kind, tag)
    pair = json.loads((traj / 'pair_manifest.json').read_text())
    with (traj / 'semantic_states.pkl').open('rb') as f:
        sem = pickle.load(f)
    with (traj / 'contact_records.pkl').open('rb') as f:
        contacts = pickle.load(f)
    n = len(sem)
    for path in (traj / 'images/main', traj / 'images/wrist', traj / 'sim_states'):
        if len(list(path.iterdir())) != n:
            raise ValueError(f'obs alignment failed: {path}')
    if n != len(steps) or n != len(contacts) or n != len(np.load(traj / 'actions.npy')):
        raise ValueError('obs/action/semantic/contact lengths differ')
    t = meta['trigger_step']
    if not isinstance(t, int) or t < 3 or t + 3 >= n:
        raise ValueError('invalid trigger or short trajectory')
    baseline = locate(baseline_run or run, task, trial, 'baseline', '')
    with (baseline / 'semantic_states.pkl').open('rb') as f:
        baseline_sem = pickle.load(f)
    if len(baseline_sem) <= t or pair['full_prefix_sim_state_max_abs'] > 1e-8:
        raise ValueError('paired baseline or full prefix mismatch')
    if not np.array_equal(np.load(traj / 'actions.npy')[:t], np.load(baseline / 'actions.npy')[:t]):
        raise ValueError('executed action prefix mismatch')
    if not np.array_equal(np.load(traj / 'actions.npy'), np.load(traj / 'executed_actions.npy')):
        raise ValueError('collector actions differ from executed actions')
    if tag == 's1':
        prior = run / 'logs' / f'task{task}_{kind}_trial{trial:02d}_s0_steps.json'
        if prior.is_file():
            first = json.loads(prior.read_text())
            a = np.asarray([x['executed_action'] for x in first if x['state_machine_phase'] == 'INTERVENE'])
            b = np.asarray([x['executed_action'] for x in steps if x['state_machine_phase'] == 'INTERVENE'])
            if a.shape == b.shape and np.array_equal(a, b):
                raise ValueError('second strength produced identical executed intervention actions after clipping')
    if not meta['success'] and any(steps[i]['terminated'] for i in range(n)):
        raise ValueError('termination flag contradicts episode summary')
    initial = np.asarray(sem[0]['object_pos'], float)
    pre = sem[t - 1]
    end = min(t + (15 if kind == 'f_block' else 20), n - 1)
    current = sem[end]
    p_obj, c_obj = np.asarray(pre['object_pos'], float), np.asarray(current['object_pos'], float)
    p_eef, c_eef = np.asarray(pre['eef_pos'], float), np.asarray(current['eef_pos'], float)
    goal = np.asarray(current['goal_pos'], float)
    pre_contact = steps[t - 1]['before']['robot_object_contacts']
    end_contact = steps[end]['before']['robot_object_contacts']
    facts = dict(target_move_m=distance(p_obj, c_obj), target_height_before_m=float(p_obj[2] - initial[2]),
                 target_height_current_m=float(c_obj[2] - initial[2]),
                 gripper_target_gap_before_m=distance(p_eef, p_obj),
                 gripper_target_gap_current_m=distance(c_eef, c_obj),
                 target_goal_before_m=distance(p_obj, goal), target_goal_current_m=distance(c_obj, goal),
                 target_contacts_before=pre_contact, target_contacts_current=end_contact,
                 intervention_steps=[t, min(t + (15 if kind == 'f_block' else 20) - 1, n - 1)])
    reason = None
    if kind == 'miss':
        closure_steps = sum(steps[i]['executed_action'][-1] > .8 for i in range(t, end + 1))
        facts['attempted_closure_steps'] = closure_steps
        valid = (facts['target_move_m'] < .025 and facts['target_height_current_m'] < .025
                 and closure_steps >= 3
                 and max(steps[i]['before']['robot_object_contacts'] for i in range(t, end + 1)) < 2)
        reason = 'target remains at support after attempted closure without sustained two-contact hold'
    elif kind == 'drop':
        valid = (facts['target_height_before_m'] > .05 and pre_contact >= 2
                 and facts['target_height_current_m'] < facts['target_height_before_m'] - .035
                 and end_contact < 2)
        reason = 'previously lifted target loses hold and descends'
    elif kind == 'wrong_place':
        xy = distance(c_obj[:2], goal[:2])
        # The starting support may be a drawer/cabinet. Check actual transport
        # and loss of hold outside the goal, without a universal initial-height band.
        initial_goal = distance(initial, goal)
        valid = (pre_contact >= 2 and end_contact < 2 and
                 facts['target_goal_current_m'] < initial_goal - .04 and
                 .045 < xy < .25 and not steps[end]['obs_success_current'])
        facts['target_goal_xy_current_m'] = xy
        reason = 'transported target released near but outside target region'
    elif kind == 'd1':
        valid = (facts['target_move_m'] < .02 and
                 facts['gripper_target_gap_current_m'] > facts['gripper_target_gap_before_m'] + .045
                 and facts['target_height_current_m'] < .02)
        reason = 'gripper moves away from stationary correct target'
    elif kind == 'd2':
        stable = sum(steps[i]['before']['robot_object_contacts'] >= 2 for i in range(t, end + 1))
        valid = (facts['target_height_before_m'] > .04 and facts['target_height_current_m'] > .04
                 and stable >= (end - t + 1) // 2 and
                 facts['target_goal_current_m'] > facts['target_goal_before_m'] + .05)
        facts['held_contact_frames'] = stable
        reason = 'target remains elevated with grasp contact while moving farther from goal'
    elif kind == 'wrong_object':
        touched = [i for i in range(t, end + 1) if steps[i]['before'].get('non_target_robot_contacts', 0) >= 2]
        first_other = steps[t]['before'].get('non_target_contact_pos')
        last_other = steps[end]['before'].get('non_target_contact_pos')
        other_delta = distance(first_other, last_other) if first_other is not None and last_other is not None else None
        facts['non_target_two_contact_frames'] = len(touched)
        facts['non_target_object_pose_displacement_m'] = other_delta
        valid = (facts['target_move_m'] < .02 and len(touched) >= 3
                 and other_delta is not None and other_delta > .025)
        reason = 'non-target object is contacted and displaced while the specified bowl remains stationary'
    elif kind == 'f_block':
        if end >= len(baseline_sem):
            valid = False
            baseline_move = None
        else:
            baseline_move = distance(baseline_sem[t - 1]['object_pos'], baseline_sem[end]['object_pos'])
            held = sum(steps[i]['before']['robot_object_contacts'] >= 2 for i in range(t, end + 1))
            drawer_contact = sum(any(
                'cabinet' in (x.get('body1', '') + x.get('body2', '')).lower()
                and 'akita_black_bowl' in (x.get('body1', '') + x.get('body2', '')).lower()
                for x in contacts[i].get('contacts', [])) for i in range(t, end + 1))
            facts.update(baseline_target_motion_m=baseline_move, sustained_target_contact_frames=held,
                         target_cabinet_contact_frames=drawer_contact)
            valid = (baseline_move > .055 and facts['target_move_m'] < .04
                     and held >= 5 and drawer_contact >= 5)
        reason = ('paired baseline extracts the target while the perturbed branch retains target contacts '
                  'near drawer support; precise mechanical obstruction remains a visual-review question')
    else:
        raise ValueError(kind)
    evidence = dict(kind=kind, task_id=task, trial_id=trial, trajectory=str(traj),
                    trajectory_sha256={k: digest(traj / k) for k in ('task.json', 'metadata.json', 'semantic_states.pkl', 'contact_records.pkl')},
                    baseline=str(baseline), trigger=t, evaluated_frame=end, n_frames=n,
                    physical_checks=facts, event_valid=bool(valid), event_reason=reason,
                    event_review='pending_stratified_visual_review' if valid or kind == 'f_block' else 'physical_check_failed',
                    final_episode_success=bool(meta['success']), split_group=f'libero_spatial:task{task}:reset{task * 50 + trial}')
    if revision and (not revision.isalnum() or len(revision) > 24):
        raise ValueError('invalid event revision')
    version_suffix = f'_{revision}' if revision else ''
    path = run / ('events' if score_progress else 'events_light') / f'{prefix}_trial{trial:02d}{suffix}{version_suffix}.json'
    if path.exists():
        raise FileExistsError(path)
    if not score_progress:
        # Production collection only: do not restore every saved sim state or
        # compute labels. E and F need a later stratified visual check before
        # they can be counted as confirmed events.
        evidence['physical_candidate'] = bool(valid)
        if valid and kind in ('wrong_object', 'f_block'):
            evidence['event_valid'] = False
            evidence['event_review'] = 'needs_stratified_visual_review'
        evidence['progress_status'] = 'deferred_until_collection_complete'
        atomic_json(path, evidence)
        return evidence
    if valid:
        try:
            if reuse_score:
                scored = run / 'progress' / traj.parent.name / traj.name
                score_quality = json.loads((scored / 'provenance.json').read_text())
                cfg = run / 'calibration' / f'task{task}_calibration.json'
                if (score_quality['source_semantic_sha256'] != digest(traj / 'semantic_states.pkl')
                    or score_quality['source_contact_sha256'] != digest(traj / 'contact_records.pkl')
                    or score_quality['calibration_sha256'] != digest(cfg)):
                    raise ValueError('existing score source/config hash mismatch')
            else:
                scored, score_quality = score(traj, run)
        except (ValueError, FileExistsError) as exc:
            # In particular, same-frame sim restoration can disagree with the
            # saved semantic pose during a dynamic impact. Keep the physical
            # event evidence but do not assign a progress label or API sample.
            evidence['event_valid'] = False
            evidence['event_review'] = 'score_blocked'
            evidence['score_error'] = f'{type(exc).__name__}: {exc}'
            atomic_json(path, evidence)
            return evidence
        evidence['score_quality'] = {k: score_quality[k] for k in ('unknown_frames', 'total_frames', 'unknown_ratio')}
        rows = [json.loads(line) for line in (scored / 'per_frame.jsonl').read_text().splitlines()]
        if len(rows) != n or any(rows[i]['frame'] != i for i in range(n)):
            raise ValueError('score observation alignment failed')
        if score_quality['unknown_ratio'] > .05 or rows[t - 1]['P'] is None or rows[end]['P'] is None:
            evidence['event_valid'] = False
            evidence['event_review'] = 'score_blocked_unreliable_query_or_many_unknown_frames'
        elif rows[end]['success_current']:
            evidence['event_valid'] = False
            evidence['event_review'] = 'invalid_current_task_success'
        else:
            candidates = [(t - 1, end)]
            middle = t + (end - t) // 2
            # A second sample needs a distinct CURRENT and an already visible
            # failure state, never a dense duplicate of the same end frame.
            if middle > t + 2 and middle < end - 3 and abs(rows[middle]['P'] - rows[t - 1]['P']) >= 3:
                mid = sem[middle]
                mid_obj = np.asarray(mid['object_pos'], float)
                mid_eef = np.asarray(mid['eef_pos'], float)
                visible = ((kind == 'drop' and mid_obj[2] < p_obj[2] - .025)
                           or (kind == 'd1' and distance(mid_eef, mid_obj) > distance(p_eef, p_obj) + .025)
                           or (kind == 'd2' and distance(mid_obj, goal) > distance(p_obj, goal) + .03))
                if visible:
                    candidates.insert(0, (t - 1, middle))
            windows = []
            for previous, current in candidates:
                if current >= n or current <= previous or rows[current]['success_current']:
                    continue
                windows.append(dict(previous=previous, current=current,
                                    previous_progress_float=rows[previous]['P'], current_progress_float=rows[current]['P'],
                                    selection_reason=reason, event_evidence=str(path),
                                    score_file=str(scored / 'per_frame.jsonl')))
            evidence['windows'] = windows
            evidence['score_dir'] = str(scored)
    atomic_json(path, evidence)
    return evidence


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--task-id', type=int, required=True)
    p.add_argument('--trial', type=int, required=True)
    p.add_argument('--kind', choices=['miss', 'drop', 'wrong_place', 'd1', 'd2', 'wrong_object', 'f_block'], required=True)
    p.add_argument('--tag', default='')
    p.add_argument('--revision', default='')
    p.add_argument('--reuse-score', action='store_true')
    p.add_argument('--light-only', action='store_true')
    p.add_argument('--baseline-run', type=Path)
    a = p.parse_args()
    print(json.dumps(evaluate(a.run.resolve(), a.task_id, a.trial, a.kind, a.tag,
                              revision=a.revision, reuse_score=a.reuse_score,
                              score_progress=not a.light_only,
                              baseline_run=a.baseline_run.resolve() if a.baseline_run else None), ensure_ascii=False))
