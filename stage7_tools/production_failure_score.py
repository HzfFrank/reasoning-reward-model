"""Production v2 scorer: same formula, explicit null for unreliable restored frames."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from geometry_progress_v1 import load_trajectory
from geometry_progress_v2 import distance_rows
from production_failure_export import atomic_json, digest


def score(path: Path, run: Path) -> tuple[Path, dict]:
    from libero.libero.envs.env_wrapper import ControlEnv
    trajectory = load_trajectory(path, require_task=False)
    task = trajectory['sem'][0]['task_id']
    config_path = run / 'calibration' / f'task{task}_calibration.json'
    config = json.loads(config_path.read_text())
    if config['status'] != 'calibrated' or config['task_id'] != task:
        raise ValueError('task-specific frozen calibration unavailable')
    y_start, y_goal = config['y_start'], config['y_goal_anchor']
    if not y_start > y_goal + 1e-6:
        raise ValueError('degenerate fixed calibration')
    distances = distance_rows(trajectory, config['beta'])
    if len(distances) != trajectory['n']:
        raise ValueError('distance/observation length mismatch')
    env = ControlEnv(trajectory['sem'][0]['bddl_file'], use_camera_obs=False,
                     has_offscreen_renderer=False)
    success = []
    restore_errors = []
    try:
        env.reset()
        for frame, sim_path in enumerate(trajectory['sim']):
            with sim_path.open('rb') as f:
                original = np.asarray(pickle.load(f))
            env.set_state(original)
            env.sim.forward()
            restored = np.asarray(env.get_sim_state())
            if frame in (0, trajectory['n']-1) and not np.array_equal(restored, original):
                success.append(None)
                restore_errors.append(dict(frame=frame, reason='full_sim_state_mismatch',
                                           sim_path=str(sim_path)))
                continue
            actual = np.asarray(env.sim.data.body_xpos[env.env.obj_body_id['akita_black_bowl_1']])
            expected = np.asarray(trajectory['sem'][frame]['object_pos'])
            error = float(np.linalg.norm(actual - expected))
            if error > .003:
                success.append(None)
                restore_errors.append(dict(frame=frame, reason='target_position_mismatch',
                                           error_m=error, allowed_m=.003, sim_path=str(sim_path)))
                continue
            success.append(bool(env.check_success()))
    finally:
        env.env.close()
    if len(success) != len(distances):
        raise ValueError('same-frame success length mismatch')
    rows = []
    for point, value in zip(distances, success):
        raw = float(100 * np.clip((y_start - point['y']) / (y_start - y_goal), 0, 1))
        progress = None if value is None else (100.0 if value else min(raw, 99.0))
        rows.append({**point, 'P_distance': raw, 'P': progress,
                     'success_current': value,
                     'success_source': 'same_frame_restored_sim_state_official_check_success'
                                       if value is not None else 'unknown_restore_mismatch',
                     'saved_frame_success_field_not_used': True})
    source_dir = path.parent.name
    dest = run / 'progress' / source_dir / path.name
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(dest)
    dest.mkdir(parents=True)
    with (dest / 'per_frame.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    np.save(dest / 'progress_geometry_v2.npy',
            np.asarray([np.nan if row['P'] is None else row['P'] for row in rows], dtype=np.float64))
    quality = dict(unknown_frames=len(restore_errors), total_frames=len(rows),
                   unknown_ratio=len(restore_errors) / len(rows), restore_errors=restore_errors,
                   calibration_path=str(config_path), calibration_sha256=digest(config_path),
                   source_semantic_sha256=digest(path / 'semantic_states.pkl'),
                   source_contact_sha256=digest(path / 'contact_records.pkl'),
                   formula='100*clip((y_start-y)/(y_start-y_goal_anchor),0,1); official current success=>100; else min(raw,99)',
                   review_only=True)
    atomic_json(dest / 'provenance.json', quality)
    return dest, quality
