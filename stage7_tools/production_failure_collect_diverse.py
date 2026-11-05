"""Isolated bounded LIBERO-Spatial failure production collection.

This entry intentionally uses the evaluation model, LiberoEnv and CollectEpisode
without starting a Ray cluster. Interventions are applied to prepared 7-D env
actions immediately before ``LiberoEnv.step``. No progress labels are computed.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import pickle
import traceback
import hashlib
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.libero.libero_env import LiberoEnv
from rlinf.envs.wrappers.collect_episode import CollectEpisode
from rlinf.models.embodiment.openpi import get_model

RLINF = Path(os.environ['V7_COLLECTION_REPO']).expanduser().resolve()


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str))
    temp.replace(path)


def _scalar(value):
    if torch.is_tensor(value):
        return bool(value.detach().cpu().numpy().any())
    return bool(np.asarray(value).any())


def _state(obs, raw=None):
    s = obs['semantic_states'][0]
    c = obs['contact_records'][0]
    contacts = c.get('contacts', [])
    # The complete contact record also contains non-target contacts. For task 7
    # only, table/ramekin contacts give an approximate non-target location;
    # actual event validity is checked again against physical contact and video.
    other = [x for x in contacts if 'glazed_rim_porcelain_ramekin_1' in
             (x.get('body1', '') + x.get('body2', ''))]
    support = [x for x in other if 'table' in (x.get('body1'), x.get('body2'))]
    other_points = support or other
    other_pos = (np.asarray(raw['glazed_rim_porcelain_ramekin_1_pos']).tolist()
                 if raw is not None and 'glazed_rim_porcelain_ramekin_1_pos' in raw else None)
    other_robot = sum(any(name in (x.get('body1', '') + x.get('body2', ''))
                          for name in ('gripper', 'finger')) for x in other)
    return {
        'eef_pos': s.get('eef_pos'), 'object_pos': s.get('object_pos'),
        'goal_pos': s.get('goal_pos'), 'gripper_qpos': s.get('gripper_qpos'),
        'robot_object_contacts': sum(bool(x.get('is_robot_object')) for x in contacts),
        'non_target_contact_pos': other_pos,
        'non_target_pose_source': 'raw_observation' if other_pos is not None else 'unknown',
        'non_target_robot_contacts': other_robot,
        'semantic_success_once': s.get('success'),
    }


def _distance(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _raw_objects(raw):
    return {k[:-4]: np.asarray(v).tolist() for k, v in raw.items()
            if k.endswith('_pos') and not k.startswith('robot') and np.asarray(v).shape == (3,)}


def _save_sidecars(directory, terminal_obs, log, initial_semantic, raw_objects, split):
    terminal = directory / 'terminal'
    terminal.mkdir(exist_ok=False)
    for view, key in [('main', 'main_images'), ('wrist', 'wrist_images')]:
        image = CollectEpisode._to_image_uint8(terminal_obs[key][0].detach().cpu().numpy())
        Image.fromarray(image).save(terminal / f'{view}.png')
    with (terminal / 'sim_state.pkl').open('wb') as stream:
        pickle.dump(terminal_obs['sim_states'][0], stream)
    _json(terminal / 'semantic.json', terminal_obs['semantic_states'][0])
    _json(terminal / 'contacts.json', terminal_obs['contact_records'][0])
    _json(terminal / 'manifest.json', dict(role='post_last_recorded_action_observation',
        is_original_saved_frame=False, action_index=len(log)-1, observation_index=len(log),
        success_current=log[-1]['post_action_success_current'],
        success_source='LiberoEnv.step terminations; auto_reset=False'))
    with (directory / 'success_alignment.jsonl').open('w') as stream:
        for row in log:
            stream.write(json.dumps(dict(obs_t=row['step'], action_t=row['step'],
                obs_success_current=row['obs_success_current'],
                post_action_success_current=row['post_action_success_current'],
                source='official LiberoEnv.step termination; initial observation unknown'))+'\n')
    _json(directory / 'collection_sidecar.json', dict(split=split,
        obs_action_alignment='obs_t -> action_t -> obs_t+1; main/wrist PNG and sim_states are obs_t',
        image_fps=4, control_step_unit='one LiberoEnv.step per action; playback FPS is not control frequency',
        action_space='prepared 7-D OSC_POSE action; translation target delta normalized to 0.05 m; gripper sign per controller',
        task=initial_semantic, objects_initial_m=raw_objects,
        distance_status='pending_reconstruction', distance_source='geometry_progress_v2.py: d_contact finger-pad nearest geom sum; d_goal 3D object-to-goal distance',
        coord_units='simulator meters; world coordinate frame',
        policy_actions='policy_actions.npy', executed_actions='actions.npy and executed_actions.npy'))


def _candidate(kind, baseline):
    initial_z = baseline[0]['before']['object_pos'][2]
    for row in baseline:
        state = row['before']
        e, obj, goal = (np.asarray(state[k], dtype=float) for k in
                        ('eef_pos', 'object_pos', 'goal_pos'))
        lifted = obj[2] - initial_z
        contact = state['robot_object_contacts']
        if kind in ('miss', 'control') and _distance(e, obj) < .075 and lifted < .02 and row['policy_action'][-1] < 0:
            return row['step']
        if kind == 'd1' and row['step'] > 5 and _distance(e, obj) > .09 and lifted < .02 and contact == 0:
            prior = baseline[max(0, row['step']-3):row['step']]
            if len(prior) == 3 and _distance(prior[0]['before']['eef_pos'], prior[0]['before']['object_pos']) > _distance(e,obj)+.015:
                return row['step']
        if kind == 'wrong_object' and state.get('non_target_contact_pos') is not None and row['step'] > 5 and lifted < .02 and contact == 0:
            other = np.asarray(state['non_target_contact_pos'], dtype=float)
            if .07 < _distance(e, other) < .50 and _distance(other, obj) > .08:
                return row['step']
        if kind == 'drop' and lifted > .035 and contact >= 2 and _distance(obj[:2], goal[:2]) > .10:
            if all(baseline[j]['before']['robot_object_contacts'] >= 2 for j in range(row['step']-3, row['step'])):
                return row['step']
        if kind == 'wrong_place' and row['step'] > 10 and contact >= 2 and _distance(obj[:2], goal[:2]) < .09 and _distance(obj, goal) < _distance(np.asarray(baseline[0]['before']['object_pos']), goal) - .04:
            return row['step']
        if kind == 'd2' and lifted > .035 and contact >= 2 and _distance(obj[:2], goal[:2]) > .10:
            prior=baseline[max(0,row['step']-3):row['step']]
            if len(prior)==3 and all(q['before']['robot_object_contacts']>=2 and
                                     q['before']['object_pos'][2]-initial_z>.045 for q in prior):
                return row['step']
        if kind == 'hover' and lifted > .055 and contact >= 2 and .025 < _distance(obj[:2],goal[:2]) < .05:
            prior=baseline[max(0,row['step']-3):row['step']]
            if len(prior)==3 and all(q['before']['robot_object_contacts']>=2 for q in prior):
                return row['step']
        if kind == 'f_block' and row['step'] > 25 and .0 <= lifted < .025 and contact >= 2:
            prior=baseline[max(0,row['step']-3):row['step']]
            if len(prior)==3 and all(q['before']['robot_object_contacts']>=2 for q in prior):
                return row['step']
    raise RuntimeError(f'No physically supported {kind} trigger in successful baseline')


def _candidate_final(kind, baseline):
    """Late, visually meaningful trigger used only by the final supplement."""
    if kind == 'd1':
        for row in baseline:
            s = row['before']
            gap = _distance(s['eef_pos'], s['object_pos'])
            prior = baseline[max(0, row['step']-4):row['step']]
            if (len(prior) == 4 and .075 < gap < .115
                    and s['robot_object_contacts'] == 0
                    and _distance(prior[0]['before']['eef_pos'], prior[0]['before']['object_pos']) > gap + .025):
                return row['step']
        raise RuntimeError('No near-object D1 approach in successful baseline')
    return _candidate(kind, baseline)


def _candidate_ef(kind, baseline):
    """Select a real early wrong-object opportunity or held drawer extraction."""
    if kind == 'wrong_object':
        for row in baseline:
            s = row['before']
            other = s.get('non_target_contact_pos')
            if (5 <= row['step'] <= 12 and other is not None
                    and _distance(other, s['object_pos']) > .12
                    and s['robot_object_contacts'] == 0):
                return row['step']
        raise RuntimeError('no separate reachable non-target object at E trigger')
    if kind == 'f_block':
        first_z = baseline[0]['before']['object_pos'][2]
        for row in baseline:
            t = row['step']; s = row['before']
            if t < 50 or t + 28 >= len(baseline):
                continue
            prior = baseline[t-3:t]
            held = all(q['before']['robot_object_contacts'] >= 3 for q in prior)
            near_support = s['object_pos'][2] - first_z < .035
            future_move = _distance(baseline[t+28]['before']['object_pos'], s['object_pos']) > .07
            if held and near_support and future_move:
                return t
        raise RuntimeError('no held target starting extraction inside drawer in successful baseline')
    raise ValueError(kind)


def _perturb_ef(kind, age, before, trigger_state, phase, strength):
    """Feedback controller for opt-in E/F; only yields legal prepared env actions."""
    out = np.zeros(7, dtype=np.float32)
    e = np.asarray(before['eef_pos'], float)
    if kind == 'wrong_object':
        other = before.get('non_target_contact_pos')
        if other is None:
            raise RuntimeError('actual non-target pose unavailable during E control')
        other = np.asarray(other, float)
        # Offset measured from the task-7 wrong-ramekin reference grasp, in
        # the world frame. The feedback target follows the physical object.
        grasp_e = other + np.array([-.01, -.03, .022])
        if phase.get('grasp_age') is None:
            target = grasp_e.copy()
            if np.linalg.norm(e[:2] - target[:2]) > .032:
                target[2] += .09
                phase['stage'] = 'approach_above_non_target'
            elif _distance(e, target) < .035:
                phase['grasp_age'] = age
                phase['stage'] = 'close_on_non_target'
            else:
                phase['stage'] = 'descend_to_non_target'
            out[:3] = np.clip((target - e) / .05, -.8, .8)
            out[-1] = -1.0
        else:
            close_age = age - phase['grasp_age']
            out[-1] = 1.0
            if close_age < 10:
                target = grasp_e
                phase['stage'] = 'close_on_non_target'
            else:
                target = grasp_e + np.array([.06, 0, .12])
                phase['stage'] = 'lift_and_carry_non_target'
            out[:3] = np.clip((target - e) / .05, -.8, .8)
    elif kind == 'f_block':
        start_e = np.asarray(trigger_state['eef_pos'], float)
        # The target remains in the opened drawer at trigger. Move diagonally
        # against its side during a real closed-gripper extraction attempt.
        # No object or simulator state is edited; saved contacts decide validity.
        # The first trial's target remained *inside* the open drawer, so a
        # stationary hold there was not evidence of obstructed extraction.
        # This revision commands a point beyond the opening in +Y while the
        # side offset tests whether the real cabinet constrains that route.
        target = start_e + np.array([strength, .22, .08])
        out[:3] = np.clip((target - e) / .05, -.7, .7)
        out[-1] = 1.0
        phase['stage'] = 'closed_gripper_diagonal_extraction_attempt'
        phase['target_eef_m'] = target.tolist()
    else:
        raise ValueError(kind)
    if not np.isfinite(out).all() or np.max(np.abs(out)) > 1.00001:
        raise RuntimeError('invalid bounded E/F intervention action')
    return out


def _perturb_final(kind, age, before, trigger_state, strength, phase, landing_axis='y_minus'):
    """Bounded feedback actions; all changes pass through the normal env step."""
    out = np.zeros(7, dtype=np.float32)
    e = np.asarray(before['eef_pos'], float)
    obj = np.asarray(before['object_pos'], float)
    goal = np.asarray(before['goal_pos'], float)
    start_e = np.asarray(trigger_state['eef_pos'], float)
    if kind == 'wrong_place':
        # Strength is a physical lateral offset from the plate center. The
        # target is away from the plate rim; release only after the real bowl
        # reaches the lower off-plate location. No object state is changed.
        target_obj = goal.copy()
        if landing_axis == 'x_plus':
            target_obj[0] += strength
        elif landing_axis == 'y_plus':
            target_obj[1] += strength
        elif landing_axis == 'y_minus':
            target_obj[1] -= strength
        else:
            raise ValueError('unrecognized off-plate landing axis')
        target_obj[2] = goal[2] + .015
        grip_offset = start_e - np.asarray(trigger_state['object_pos'], float)
        target_e = target_obj + grip_offset
        phase.setdefault('landing_target_obj_m', target_obj.tolist())
        if phase.get('release_age') is None:
            # Clear the plate in the horizontal plane while the bowl remains
            # high. Lowering and crossing the rim together caused real rim
            # contact in task 1 and 6, so use a measured horizontal stage.
            close_xy = np.linalg.norm(obj[:2] - target_obj[:2]) < .045
            if phase.get('reached_xy_age') is None:
                if close_xy:
                    phase['reached_xy_age'] = age
                else:
                    high_e = target_e.copy()
                    high_e[2] = max(start_e[2], goal[2] + .14)
                    out[:3] = np.clip((high_e-e)/.05, -.8, .8)
                    out[-1] = 1.0
                    return out
            # The bowl center remains a few centimetres above the tabletop
            # while grasped. Check the physical bowl position before release.
            low = obj[2] < target_obj[2] + .055
            if close_xy and low:
                phase['release_age'] = age
            else:
                out[:3] = np.clip((target_e-e)/.05, -.8, .8)
                out[-1] = 1.0
                return out
        release_age = age - phase['release_age']
        out[-1] = -1.0
        if release_age >= 10:
            lift_e = target_e.copy(); lift_e[2] += .08
            out[:3] = np.clip((lift_e-e)/.05, -.7, .7)
    elif kind == 'd1':
        away = start_e - np.asarray(trigger_state['object_pos'], float)
        away[2] = 0
        norm = np.linalg.norm(away)
        if norm < .02:
            raise RuntimeError('D1 retreat direction undefined')
        target = start_e.copy(); target[:2] += strength * away[:2] / norm
        out[:3] = np.clip((target-e)/.05, -.8, .8)
        out[-1] = -1.0
    elif kind == 'drop':
        out[-1] = -1.0
    elif kind == 'miss':
        away = start_e[:2] - np.asarray(trigger_state['object_pos'], float)[:2]
        norm = np.linalg.norm(away)
        if norm < .015:
            raise RuntimeError('miss offset direction undefined')
        # Clear the bowl vertically before crossing its footprint. Lower at
        # the offset location, close there, then lift the empty gripper.
        target = start_e.copy()
        if age < 8:
            target[2] += .07
        else:
            target[:2] += strength * away / norm
            if age < 24 or age >= 44:
                target[2] += .07
        out[:3] = np.clip((target-e)/.05, -.8, .8)
        out[-1] = -1.0 if age < 34 else 1.0
    else:
        raise ValueError(kind)
    if not np.isfinite(out).all() or np.max(np.abs(out)) > 1.00001:
        raise RuntimeError('invalid bounded intervention action')
    return out


def _perturb(kind, age, before, trigger_state, strength, initial_object_z, goal_bowl_z=None):
    """Return a bounded env action after policy unnormalization and mapping.

    OSC_POSE maps a normalized translation command of 1 to a 0.05 m *target
    delta*, not guaranteed physical displacement. Feedback uses measured EEF.
    """
    out = np.zeros(7, dtype=np.float32)
    e = np.asarray(before['eef_pos'], dtype=float)
    obj = np.asarray(before['object_pos'], dtype=float)
    goal = np.asarray(before['goal_pos'], dtype=float)
    if kind == 'miss':
        baseline_e = np.asarray(trigger_state['eef_pos'], dtype=float)
        side = np.array([0.0, -1.0, 0.0])
        target = baseline_e + side * strength
        target[2] = baseline_e[2]
        out[:3] = np.clip((target-e)/.05, -.8, .8)
        out[-1] = -1.0 if age < 7 else 1.0
    elif kind == 'd1':
        away = e - obj
        away[2] = 0
        norm = float(np.linalg.norm(away))
        if norm < .03:
            raise RuntimeError('D1 direction undefined at target')
        target = np.asarray(trigger_state['eef_pos'], float).copy()
        target[:2] += strength * away[:2] / norm
        out[:3] = np.clip((target-e)/.05, -.8, .8)
        out[-1] = -1.0
    elif kind == 'wrong_object':
        if trigger_state.get('non_target_contact_pos') is None:
            raise RuntimeError('non-target position unavailable')
        target = np.asarray(trigger_state['non_target_contact_pos'], float).copy()
        target[2] += .065
        out[:3] = np.clip((target-e)/.05, -.8, .8)
        out[-1] = 1.0 if age < 11 else -1.0
    elif kind == 'drop':
        out[-1] = -1.0
    elif kind == 'wrong_place':
        # Place beside the plate, never by teleporting the bowl.
        offset = np.asarray(trigger_state['eef_pos'], dtype=float) - np.asarray(trigger_state['object_pos'], dtype=float)
        target_obj = goal.copy()
        target_obj[1] -= strength
        # An early release from height would be a drop, not placement.
        table_bowl_center_z = goal_bowl_z if goal_bowl_z is not None else initial_object_z
        target_obj[2] = table_bowl_center_z + .025
        target = target_obj + offset
        if age < 16 or obj[2] - table_bowl_center_z > .06:
            out[:3] = np.clip((target-e)/.05, -.8, .8)
            out[-1] = 1.0
        else:
            out[-1] = -1.0
    elif kind == 'd2':
        # OSC_POSE translation action maps to a 5 cm *target* delta.
        # The physical motion is checked against saved object positions later.
        direction = obj[:2]-goal[:2]
        norm = float(np.linalg.norm(direction))
        if norm < .03:
            raise RuntimeError('D2 direction undefined near goal')
        target = np.asarray(trigger_state['eef_pos'],float).copy()
        target[:2] += strength*direction/norm
        out[:3] = np.clip((target-e)/.05,-.8,.8)
        out[-1] = 1.0
    elif kind == 'hover':
        target = np.asarray(trigger_state['eef_pos'],float).copy()
        out[:3] = np.clip((target-e)/.05,-.5,.5)
        out[-1] = 1.0
    elif kind == 'f_block':
        # Bounded lateral extraction bias. The event is validated from cabinet
        # contacts and observed displacement, not inferred from this command.
        target=np.asarray(trigger_state['eef_pos'],float).copy()
        target[0]-=.035
        target[1]+=strength
        target[2]+=.035
        out[:3]=np.clip((target-e)/.05,-.6,.6)
        out[-1]=1.0
    else:
        raise ValueError(kind)
    assert np.isfinite(out).all() and np.max(np.abs(out)) <= 1.00001
    return out


class PersistentRuntime:
    """Keep Pi0 resident; recreate LIBERO only when the task changes."""

    def __init__(self):
        self.model = None
        self.model_path = None
        self.env = None
        self.task_id = None
        self.model_load_count = 0
        self.env_build_count = 0

    def resources(self, args, cfg):
        if self.model is None:
            model_cfg = OmegaConf.load(RLINF / 'examples/embodiment/config/model/pi0.yaml')
            model_cfg.model_path = args.model
            model_cfg.num_action_chunks = 5
            model_cfg.precision = None
            self.model = get_model(model_cfg).to('cuda:0').eval()
            assert self.model.config.action_env_dim == 7
            self.model_path = args.model
            self.model_load_count += 1
        elif self.model_path != args.model:
            raise ValueError('persistent worker cannot change model weights')
        if self.env is None or self.task_id != args.task_id:
            if self.env is not None:
                self.env.close()
            self.env = LiberoEnv(cfg, num_envs=1, seed_offset=0,
                                 total_num_processes=1, worker_info=None)
            self.task_id = args.task_id
            self.env_build_count += 1
        return self.model, self.env

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


def run(args, runtime: PersistentRuntime | None = None):
    out = Path(args.output).resolve()
    assert str(out).startswith('/project_c20250301/251503016/reasoning/stage7_artifacts/')
    task_id = int(args.task_id)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = OmegaConf.load(RLINF / 'examples/embodiment/config/env/libero_spatial.yaml')
    cfg.total_num_envs = 1
    cfg.is_eval = True
    cfg.auto_reset = False
    cfg.ignore_terminations = False
    cfg.max_episode_steps = 240
    cfg.record_sim_state = True
    cfg.record_semantic_state = True
    cfg.record_contact_state = True
    cfg.use_fixed_reset_state_ids = True
    cfg.use_ordered_reset_state_ids = True
    cfg.task_id_filter = [task_id]
    cfg.specific_reset_id = None
    cfg.video_cfg.save_video = False
    cfg.seed = args.seed

    if runtime is None:
        runtime = PersistentRuntime()
        owns_runtime = True
    else:
        owns_runtime = False
    model, env = runtime.resources(args, cfg)
    task_name=env.task_suite.get_task(task_id).name
    assert task_name.startswith('pick_up_the_black_bowl_') and task_name.endswith('_and_place_it_on_the_plate')
    first_reset_id = int(env.cumsum_trial_id_bins[task_id - 1]) if task_id else 0
    assert len(env.task_suite.get_task_init_states(task_id)) >= max(args.trials)+1
    tag = f'_{args.attempt_tag}' if args.attempt_tag else ''
    prefix=f'task{task_id}_{args.kind}'
    if len(args.trials) != 1:
        raise ValueError('production collector requires exactly one reset trial per isolated output directory')
    episode_root = out / 'trajectories' / f'{prefix}{tag}_trial{args.trials[0]:02d}'
    if episode_root.exists() and any(episode_root.iterdir()):
        raise FileExistsError(f'refusing to overwrite existing trajectory: {episode_root}')
    recorder = CollectEpisode(env, save_dir=str(episode_root), rank=0, num_envs=1,
                              export_format='trajectory', fps=4, only_success=False,
                              stop_on_success=True)
    records = []
    for trial in args.trials:
        reset_id = first_reset_id + trial
        env.specific_reset_id = reset_id
        # LiberoEnv._reconfigure updates task/trial but its public reset-state
        # array is otherwise stale on a second manual reset. Set both so the
        # recorded provenance matches the actual fixed initialization.
        env.reset_state_ids = np.asarray([reset_id], dtype=np.int64)
        obs, _ = recorder.reset()
        initial_semantic = dict(obs['semantic_states'][0])
        initial_objects = _raw_objects(env.current_raw_obs[0])
        assert int(obs['task_ids'][0]) == task_id
        assert int(obs['reset_state_ids'][0]) == reset_id
        log = []
        baseline = None
        online = getattr(args, 'profile', '') == 'ef_online_v1'
        if online and args.kind != 'f_block':
            raise ValueError('online fallback is limited to task4 extraction')
        source = None
        baseline_root = Path(getattr(args, 'baseline_output', out) or out).resolve()
        # Final supplements may cite a successful baseline from the immutable
        # parent production run. Resolve symlinks, then keep the trust boundary
        # at that run; a sibling experiment or arbitrary filesystem path is
        # still rejected. The task/reset and trajectory are checked below.
        production_root = Path('/project_c20250301/251503016/reasoning/stage7_artifacts/v7_spatial_failure_production_20261025_r1').resolve()
        if not out.is_relative_to(production_root) or not baseline_root.is_relative_to(production_root):
            raise ValueError('baseline and output must remain inside the same production run')
        trigger = None
        trigger_state = None
        if args.kind != 'baseline' and not online:
            baseline_tag = getattr(args, 'baseline_tag', '')
            if baseline_tag and (not baseline_tag.isalnum() or len(baseline_tag) > 24):
                raise ValueError('invalid explicit baseline tag')
            baseline_suffix = f'_{baseline_tag}' if baseline_tag else ''
            source = baseline_root / 'logs' / f'task{task_id}_baseline_trial{trial:02d}{baseline_suffix}_steps.json'
            if not source.is_file():
                raise FileNotFoundError(f'Paired baseline actions missing: {source}')
            baseline = json.loads(source.read_text())
            baseline_summary_path = baseline_root / 'logs' / f'task{task_id}_baseline_trial{trial:02d}{baseline_suffix}_summary.json'
            if not baseline_summary_path.is_file():
                raise FileNotFoundError('Paired baseline summary missing')
            baseline_summary = json.loads(baseline_summary_path.read_text())
            if int(baseline_summary['task_id']) != task_id or int(baseline_summary['trial_id']) != trial or int(baseline_summary['reset_state_id']) != reset_id:
                raise RuntimeError('Paired baseline task/reset identity mismatch')
            if not baseline_summary['success']:
                raise RuntimeError('Cannot perturb a non-successful baseline')
            trigger = (_candidate_ef(args.kind, baseline) if getattr(args, 'profile', '') == 'ef_final_v1'
                       else _candidate_final(args.kind, baseline) if getattr(args, 'profile', '') == 'final_supplement_v1'
                       else _candidate(args.kind, baseline))
            trigger_state = baseline[trigger]['before']
            print(f'TRIGGER_CANDIDATE {args.kind} trial={trial} step={trigger}', flush=True)
        chunk = None
        chunk_pos = 0
        chunk_start_step = None
        replay_error = None
        full_prefix_error = None
        success = False
        previous_success_current = None
        terminal_obs = None
        intervention_limit=(80 if args.kind == 'wrong_object' else 40) if getattr(args, 'profile', '') in ('ef_final_v1', 'ef_online_v1') else ((80 if args.kind == 'wrong_place' else 50 if args.kind == 'miss' else 30 if args.kind == 'drop' else 20) if getattr(args, 'profile', '') == 'final_supplement_v1' else (15 if args.kind=='f_block' else 20))
        final_phase = {}
        online_prior_states = []
        t0 = time.monotonic()
        for step in range(240):
            if baseline is None or step >= trigger:
                if chunk is None or chunk_pos >= 5:
                    with torch.inference_mode():
                        policy_obs = dict(obs, extra_view_images=None)
                        action_chunk, _ = model.predict_action_batch(policy_obs, mode='eval')
                    chunk = prepare_actions(action_chunk, env_type='libero', model_type='openpi',
                                            num_action_chunks=5, action_dim=7)
                    assert chunk.shape == (1, 5, 7), chunk.shape
                    assert np.isfinite(chunk).all()
                    chunk_pos = 0
                    chunk_start_step = step
                policy_action = np.array(chunk[0, chunk_pos], dtype=np.float32, copy=True)
                chunk_pos += 1
            else:
                policy_action = np.asarray(baseline[step]['policy_action'], dtype=np.float32)
                chunk_start_step = baseline[step]['chunk_index'] * 5
            executed_action = policy_action.copy()
            before = _state(obs, env.current_raw_obs[0])
            if online and trigger is None:
                prior = online_prior_states[-3:]
                initial_z = float(initial_semantic['object_pos'][2])
                if (step >= 50 and len(prior) == 3
                        and all(x['robot_object_contacts'] >= 3 for x in prior)
                        and before['robot_object_contacts'] >= 3
                        and before['object_pos'][2] - initial_z < .035
                        and policy_action[-1] > .5):
                    trigger = step
                    trigger_state = before
                    print(f'ONLINE_TRIGGER f_block trial={trial} step={step}', flush=True)
            phase = 'ONLINE_POLICY' if online and trigger is None else 'BASELINE' if baseline is None else 'REPLAY_PREFIX'
            if baseline is not None and step < trigger:
                executed_action = np.asarray(baseline[step]['executed_action'], dtype=np.float32)
            elif trigger is not None and step < trigger + intervention_limit:
                if step == trigger:
                    if baseline is not None:
                        replay_error = max(_distance(before[k], trigger_state[k]) for k in ('eef_pos', 'object_pos', 'goal_pos'))
                        print('PREFIX_REPLAY_MAX_M', replay_error, flush=True)
                        if replay_error > .01:
                            raise RuntimeError(f'Unsafe prefix replay mismatch {replay_error:.4f} m')
                        matching_baselines=[p for p in (baseline_root/'trajectories').glob(f'task{task_id}_baseline*/episode_*')
                                            if json.loads((p/'task.json').read_text())['reset_state_id']==reset_id]
                        if len(matching_baselines)!=1:
                            raise RuntimeError('matching baseline trajectory missing')
                        base_dir=matching_baselines[0]
                        with (base_dir/'sim_states'/f'{step:06d}.pkl').open('rb') as f:
                            base_state=np.asarray(pickle.load(f),dtype=float)
                        current_state=np.asarray(obs['sim_states'][0],dtype=float)
                        if base_state.shape!=current_state.shape:
                            raise RuntimeError('full prefix sim state shape mismatch')
                        full_prefix_error=float(np.max(np.abs(base_state-current_state)))
                        if full_prefix_error>1e-8:
                            raise RuntimeError(f'full prefix sim state mismatch {full_prefix_error}')
                if args.kind == 'control':
                    executed_action = np.asarray(baseline[step]['executed_action'], dtype=np.float32)
                    phase = 'CONTROL_REPLAY'
                else:
                    goal_bowl_z = float(np.median([q['before']['object_pos'][2] for q in baseline[-5:]])) if baseline is not None else None
                    executed_action = (_perturb_ef(args.kind, step-trigger, before, trigger_state,
                                                    final_phase, args.strength)
                                       if getattr(args, 'profile', '') in ('ef_final_v1', 'ef_online_v1')
                                       else _perturb_final(args.kind, step-trigger, before, trigger_state,
                                                       args.strength, final_phase,
                                                       getattr(args, 'landing_axis', 'y_minus'))
                                       if getattr(args, 'profile', '') == 'final_supplement_v1'
                                       else _perturb(args.kind, step-trigger, before, trigger_state,
                                                     args.strength, baseline[0]['before']['object_pos'][2], goal_bowl_z))
                    phase = 'INTERVENE'
            elif trigger is not None:
                phase = 'RELEASE_TO_POLICY'
                if step == trigger + intervention_limit:
                    # Force a fresh Pi0 request from the *new* observation.
                    with torch.inference_mode():
                        policy_obs = dict(obs, extra_view_images=None)
                        action_chunk, _ = model.predict_action_batch(policy_obs, mode='eval')
                    chunk = prepare_actions(action_chunk, env_type='libero', model_type='openpi',
                                            num_action_chunks=5, action_dim=7)
                    policy_action = np.array(chunk[0, 0], dtype=np.float32, copy=True)
                    chunk_pos = 1
                    chunk_start_step = step
                    executed_action = policy_action.copy()
            obs, _reward, term, trunc, infos = recorder.step(executed_action[None, :])
            terminal_obs = obs
            after = _state(obs, env.current_raw_obs[0])
            if phase=='INTERVENE' and _distance(after['object_pos'],before['object_pos'])>.08:
                raise RuntimeError(f'abnormal object jump during intervention: step {step}')
            row = {'step': step, 'obs_t': step, 'obs_t_plus_1': step + 1,
                   'chunk_index': step // 5, 'policy_chunk_start_step': chunk_start_step,
                   'policy_action': policy_action.tolist(),
                   'executed_action': executed_action.tolist(),
                   'state_machine_phase': phase, 'intervention_kind': args.kind,
                   'before': before, 'after': after,
                   'terminated': _scalar(term), 'truncated': _scalar(trunc)}
            row['obs_success_current'] = previous_success_current
            row['post_action_success_current'] = _scalar(term)
            previous_success_current = _scalar(term)
            log.append(row)
            if online:
                online_prior_states.append(before)
            if step % 20 == 0:
                print(f'{args.kind} trial={trial} step={step} eef={after["eef_pos"]} object={after["object_pos"]}', flush=True)
            if _scalar(term) or _scalar(trunc):
                success = _scalar(term)
                break
        record = {'task_id': task_id, 'task_name': task_name, 'trial_id': trial,
                  'reset_state_id': reset_id, 'seed': args.seed, 'success': success,
                  'steps': len(log), 'seconds': time.monotonic()-t0,
                  'policy_model': args.model, 'num_action_chunks': 5,
                  'trigger_step': trigger, 'prefix_replay_max_m': replay_error,
                  'full_prefix_sim_state_max_abs':full_prefix_error,
                  'kind': args.kind, 'strength_m': args.strength,
                  'pairing': 'unpaired_online_intervention' if online else 'strict_paired_prefix' if baseline is not None else 'baseline'}
        if getattr(args, 'profile', ''):
            record['intervention_profile'] = args.profile
            record['intervention_phase'] = final_phase
            record['landing_axis'] = getattr(args, 'landing_axis', 'y_minus')
        if args.kind != 'baseline':
            arr = np.asarray([r['policy_action'] for r in log], dtype=np.float32)
            ex = np.asarray([r['executed_action'] for r in log], dtype=np.float32)
            # CollectEpisode's actions.npy already stores the executed actions.
            recorder._wait_futures()
            trajectory_dirs = sorted(episode_root.glob('episode_*'), key=lambda p: p.stat().st_mtime_ns)
            if trajectory_dirs:
                d = trajectory_dirs[-1]
                np.save(d / 'policy_actions.npy', arr)
                np.save(d / 'executed_actions.npy', ex)
                _json(d / 'pair_manifest.json', {'baseline_log': str(source) if source else None, 'reset_state_id': reset_id,
                                                 'trigger_step': trigger, 'prefix_replay_max_m': replay_error,
                                                 'full_prefix_sim_state_max_abs':full_prefix_error,
                                                 'pairing': record['pairing'],
                                                 'pre_intervention_saved_state': str(d / 'sim_states' / f'{trigger:06d}.pkl') if trigger is not None else None,
                                                 'obs_action_alignment': 'obs_t -> action_t -> obs_t+1; exported images contain obs_t'})
                with (d / 'interventions.jsonl').open('w') as f:
                    for r in log:
                        if r['state_machine_phase'] == 'INTERVENE':
                            f.write(json.dumps(r) + '\n')
            _json(out / 'logs' / f'{prefix}_trial{trial:02d}{tag}_event_summary.json',
                  {'intended_event': args.kind, 'observed_event': 'pending_visual_review',
                   'event_valid': None, 'recovered': None, 'success_once': success,
                   'success_at_end': success, 'trigger_step': trigger,
                   'intervention_steps': [trigger, min(trigger+intervention_limit-1, len(log)-1)] if trigger is not None else None})
        recorder._wait_futures()
        trajectory_dirs = sorted(episode_root.glob('episode_*'), key=lambda p: p.stat().st_mtime_ns)
        if len(trajectory_dirs) != 1 or terminal_obs is None:
            raise RuntimeError('expected one complete trajectory and terminal observation')
        _save_sidecars(trajectory_dirs[0], terminal_obs, log, initial_semantic,
                       initial_objects, getattr(args, 'split', 'unassigned'))
        _json(out / 'logs' / f'{prefix}_trial{trial:02d}{tag}_steps.json', log)
        _json(out / 'logs' / f'{prefix}_trial{trial:02d}{tag}_summary.json', record)
        records.append(record)
        print('EPISODE_DONE', json.dumps(record), flush=True)
    # CollectEpisode.close() closes its env. The long-lived worker owns that
    # env, so detach it only after all writes have been flushed.
    if not owns_runtime:
        recorder.env = None
    recorder.close()
    if owns_runtime:
        runtime.close()
    _json(out / 'logs' / f'{prefix}_trial{args.trials[0]:02d}{tag}_batch_summary.json', records)
    return records


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--output')
    p.add_argument('--model', default='/project_c20250301/251503016/models/RLinf-Pi0-PPO-LIBERO-spatial')
    p.add_argument('--task-id', type=int, choices=range(10))
    p.add_argument('--kind', choices=['baseline', 'miss', 'drop', 'wrong_place', 'd1', 'd2', 'wrong_object', 'f_block'], default='baseline')
    p.add_argument('--strength', type=float, default=.03)
    p.add_argument('--attempt-tag', default='')
    p.add_argument('--baseline-output')
    p.add_argument('--trials', type=int, nargs='+', default=[0])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--serve', action='store_true')
    return p


def serve():
    runtime = PersistentRuntime()
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request.get('command') == 'stop':
                break
            args = argparse.Namespace(**request['args'])
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    records = run(args, runtime)
                reply = dict(status='completed', records=records,
                             model_load_count=runtime.model_load_count,
                             env_build_count=runtime.env_build_count)
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                # A failed episode may leave the simulator in an unknown
                # state. Keep the model, but rebuild the environment next time.
                runtime.close()
                reply = dict(status='failed', error=f'{type(exc).__name__}: {exc}',
                             model_load_count=runtime.model_load_count,
                             env_build_count=runtime.env_build_count)
            print(json.dumps(reply, ensure_ascii=False), flush=True)
    finally:
        runtime.close()


if __name__ == '__main__':
    args = parser().parse_args()
    if args.serve:
        serve()
    else:
        if args.output is None or args.task_id is None:
            raise ValueError('--output and --task-id are required')
        run(args)
