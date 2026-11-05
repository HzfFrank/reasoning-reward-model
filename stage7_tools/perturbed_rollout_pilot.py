"""Isolated single-environment RLinf/LIBERO/Pi0 rollout pilot.

This entry intentionally uses the evaluation model, LiberoEnv and CollectEpisode
without starting a Ray cluster. Interventions are applied to prepared 7-D env
actions immediately before ``LiberoEnv.step``. No progress labels are computed.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.libero.libero_env import LiberoEnv
from rlinf.envs.wrappers.collect_episode import CollectEpisode
from rlinf.models.embodiment.openpi import get_model

RLINF = Path('/project_c20250301/251503016/RLinf')
TASK_NAME = 'pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate'
TASK_ID = 6


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str))
    temp.replace(path)


def _scalar(value):
    if torch.is_tensor(value):
        return bool(value.detach().cpu().numpy().any())
    return bool(np.asarray(value).any())


def _state(obs):
    s = obs['semantic_states'][0]
    c = obs['contact_records'][0]
    contacts = c.get('contacts', [])
    return {
        'eef_pos': s.get('eef_pos'), 'object_pos': s.get('object_pos'),
        'goal_pos': s.get('goal_pos'), 'gripper_qpos': s.get('gripper_qpos'),
        'robot_object_contacts': sum(bool(x.get('is_robot_object')) for x in contacts),
        'semantic_success_once': s.get('success'),
    }


def _distance(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


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
        if kind == 'drop' and lifted > .07 and contact >= 2 and _distance(obj[:2], goal[:2]) > .12:
            if all(baseline[j]['before']['robot_object_contacts'] >= 2 for j in range(row['step']-3, row['step'])):
                return row['step']
        if kind == 'wrong_place' and .09 < lifted < .12 and contact >= 2 and _distance(obj[:2], goal[:2]) < .06:
            return row['step']
    raise RuntimeError(f'No physically supported {kind} trigger in successful baseline')


def _perturb(kind, age, before, trigger_state, strength, initial_object_z):
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
    elif kind == 'drop':
        out[-1] = -1.0
    elif kind == 'wrong_place':
        # Place beside the plate, never by teleporting the bowl.
        offset = np.asarray(trigger_state['eef_pos'], dtype=float) - np.asarray(trigger_state['object_pos'], dtype=float)
        target_obj = goal.copy()
        target_obj[1] -= strength
        # An early release from height would be a drop, not placement.
        table_bowl_center_z = initial_object_z
        target_obj[2] = table_bowl_center_z + .025
        target = target_obj + offset
        if age < 16 or obj[2] - table_bowl_center_z > .06:
            out[:3] = np.clip((target-e)/.05, -.8, .8)
            out[-1] = 1.0
        else:
            out[-1] = -1.0
    else:
        raise ValueError(kind)
    assert np.isfinite(out).all() and np.max(np.abs(out)) <= 1.00001
    return out


def run(args):
    out = Path(args.output).resolve()
    assert str(out).startswith('/project_c20250301/251503016/reasoning/stage7_artifacts/')
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
    cfg.task_id_filter = [TASK_ID]
    cfg.specific_reset_id = None
    cfg.video_cfg.save_video = False
    cfg.seed = args.seed

    model_cfg = OmegaConf.load(RLINF / 'examples/embodiment/config/model/pi0.yaml')
    model_cfg.model_path = args.model
    model_cfg.num_action_chunks = 5
    model_cfg.precision = None
    model = get_model(model_cfg)
    model = model.to('cuda:0').eval()
    assert model.config.action_env_dim == 7

    env = LiberoEnv(cfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None)
    assert env.task_suite.get_task(TASK_ID).name == TASK_NAME
    first_reset_id = int(env.cumsum_trial_id_bins[TASK_ID - 1])
    assert len(env.task_suite.get_task_init_states(TASK_ID)) >= 6
    tag = f'_{args.attempt_tag}' if args.attempt_tag else ''
    episode_root = out / 'trajectories' / f'{args.kind}{tag}'
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
        assert int(obs['task_ids'][0]) == TASK_ID
        assert int(obs['reset_state_ids'][0]) == reset_id
        log = []
        baseline = None
        trigger = None
        trigger_state = None
        if args.kind != 'baseline':
            source = out / 'logs' / f'baseline_trial{trial:02d}_steps.json'
            if not source.is_file():
                raise FileNotFoundError(f'Paired baseline actions missing: {source}')
            baseline = json.loads(source.read_text())
            if not (out / 'logs' / f'baseline_trial{trial:02d}_summary.json').is_file():
                raise FileNotFoundError('Paired baseline summary missing')
            if not json.loads((out / 'logs' / f'baseline_trial{trial:02d}_summary.json').read_text())['success']:
                raise RuntimeError('Cannot perturb a non-successful baseline')
            trigger = _candidate(args.kind, baseline)
            trigger_state = baseline[trigger]['before']
            print(f'TRIGGER_CANDIDATE {args.kind} trial={trial} step={trigger}', flush=True)
        chunk = None
        chunk_pos = 0
        chunk_start_step = None
        replay_error = None
        success = False
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
            before = _state(obs)
            phase = 'BASELINE' if baseline is None else 'REPLAY_PREFIX'
            if baseline is not None and step < trigger:
                executed_action = np.asarray(baseline[step]['executed_action'], dtype=np.float32)
            elif baseline is not None and step < trigger + 20:
                if step == trigger:
                    replay_error = max(_distance(before[k], trigger_state[k]) for k in ('eef_pos', 'object_pos', 'goal_pos'))
                    print('PREFIX_REPLAY_MAX_M', replay_error, flush=True)
                    if replay_error > .01:
                        raise RuntimeError(f'Unsafe prefix replay mismatch {replay_error:.4f} m')
                if args.kind == 'control':
                    executed_action = np.asarray(baseline[step]['executed_action'], dtype=np.float32)
                    phase = 'CONTROL_REPLAY'
                else:
                    executed_action = _perturb(args.kind, step-trigger, before, trigger_state,
                                               args.strength, baseline[0]['before']['object_pos'][2])
                    phase = 'INTERVENE'
            elif baseline is not None:
                phase = 'RELEASE_TO_POLICY'
                if step == trigger + 20:
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
            after = _state(obs)
            row = {'step': step, 'obs_t': step, 'obs_t_plus_1': step + 1,
                   'chunk_index': step // 5, 'policy_chunk_start_step': chunk_start_step,
                   'policy_action': policy_action.tolist(),
                   'executed_action': executed_action.tolist(),
                   'state_machine_phase': phase, 'intervention_kind': args.kind,
                   'before': before, 'after': after,
                   'terminated': _scalar(term), 'truncated': _scalar(trunc)}
            log.append(row)
            if step % 20 == 0:
                print(f'{args.kind} trial={trial} step={step} eef={after["eef_pos"]} object={after["object_pos"]}', flush=True)
            if _scalar(term) or _scalar(trunc):
                success = _scalar(term)
                break
        record = {'task_id': TASK_ID, 'task_name': TASK_NAME, 'trial_id': trial,
                  'reset_state_id': reset_id, 'seed': args.seed, 'success': success,
                  'steps': len(log), 'seconds': time.monotonic()-t0,
                  'policy_model': args.model, 'num_action_chunks': 5,
                  'trigger_step': trigger, 'prefix_replay_max_m': replay_error,
                  'kind': args.kind, 'strength_m': args.strength}
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
                _json(d / 'pair_manifest.json', {'baseline_log': str(source), 'reset_state_id': reset_id,
                                                 'trigger_step': trigger, 'prefix_replay_max_m': replay_error,
                                                 'obs_action_alignment': 'obs_t -> action_t -> obs_t+1; exported images contain obs_t'})
                with (d / 'interventions.jsonl').open('w') as f:
                    for r in log:
                        if r['state_machine_phase'] == 'INTERVENE':
                            f.write(json.dumps(r) + '\n')
            _json(out / 'logs' / f'{args.kind}_trial{trial:02d}{tag}_event_summary.json',
                  {'intended_event': args.kind, 'observed_event': 'pending_visual_review',
                   'event_valid': None, 'recovered': None, 'success_once': success,
                   'success_at_end': success, 'trigger_step': trigger,
                   'intervention_steps': [trigger, min(trigger+19, len(log)-1)]})
        _json(out / 'logs' / f'{args.kind}_trial{trial:02d}{tag}_steps.json', log)
        _json(out / 'logs' / f'{args.kind}_trial{trial:02d}{tag}_summary.json', record)
        records.append(record)
        print('EPISODE_DONE', json.dumps(record), flush=True)
    recorder.close()
    _json(out / 'logs' / f'{args.kind}_summary.json', records)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--model', default='/project_c20250301/251503016/models/RLinf-Pi0-PPO-LIBERO-spatial')
    p.add_argument('--kind', choices=['baseline', 'control', 'miss', 'drop', 'wrong_place'], default='baseline')
    p.add_argument('--strength', type=float, default=.03)
    p.add_argument('--attempt-tag', default='')
    p.add_argument('--trials', type=int, nargs='+', default=[0])
    p.add_argument('--seed', type=int, default=0)
    run(p.parse_args())
