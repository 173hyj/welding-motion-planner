"""Hash-bound plan loading and right-continuous rigid-pose interpolation."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SOURCE_TO_MUJOCO = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


def pair_at(times, time):
    times = np.asarray(times, dtype=float)
    if not len(times):
        raise ValueError('empty timeline')
    i = max(0, min(len(times) - 1, int(np.searchsorted(times, time, side='right')) - 1))
    j = min(i + 1, len(times) - 1)
    alpha = 0. if i == j else float(np.clip((time - times[i]) / (times[j] - times[i]), 0., 1.))
    return i, j, alpha


def audited_plan(root, path):
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    raw = path.read_bytes()
    decoded = gzip.decompress(raw) if path.suffix == '.gz' else raw
    audit = json.loads((root / 'build/eight_hour_v17/audit/plan_overrides.json').read_text())
    entry = audit['plans'].get(path.name)
    registry = root / 'build/eight_hour_v17/audit/plan_overrides.json'
    if entry is None:
        for directory in ['welding_29_staged_v2', 'welding_29_relaxed_v1']:
            registry = root / 'build' / directory / 'audit.json'
            if registry.exists():
                entry = json.loads(registry.read_text())['plans'].get(path.name)
            if entry is not None:
                break
        if entry is None or (root / entry.get('relative_path', '')).resolve() != path.resolve():
            raise ValueError(f'plan path not registered in audit: {path}')
    digest = hashlib.sha256(decoded).hexdigest()
    if digest != entry['sha256']:
        raise ValueError(f'plan hash mismatch: {path.name}')
    plan = json.loads(decoded)
    frames = plan.get('frames', [])
    if not frames:
        raise ValueError('plan has no frames')
    times = np.array([frame['time_s'] for frame in frames], dtype=float)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0) or times[-1] <= 0:
        raise ValueError('invalid plan timestamps')
    metadata = dict(path=str(path), audit_registry=str(registry), decoded_sha256=digest,
                    file_sha256=hashlib.sha256(raw).hexdigest(), audit=entry,
                    frame_count=len(frames), duration_s=float(times[-1]))
    return plan, metadata


def value_at_path(frame, path):
    for key in path:
        frame = frame[key]
    return frame


def slerp_many(a, b, alpha):
    b = b.copy()
    dot = np.sum(a * b, axis=1)
    b[dot < 0] *= -1
    theta = np.arccos(np.clip(np.abs(dot), 0., 1.))
    sine = np.sin(theta)
    result = a * (1. - alpha) + b * alpha
    keep = sine > 1e-7
    result[keep] = ((a[keep] * np.sin((1. - alpha) * theta[keep, None]) +
                     b[keep] * np.sin(alpha * theta[keep, None])) / sine[keep, None])
    return result / np.linalg.norm(result, axis=1, keepdims=True)


class Timeline:
    def __init__(self, frames, bindings):
        self.frames = frames
        self.times = np.array([f['time_s'] for f in frames], dtype=float)
        self.bindings = bindings
        positions, quaternions, sources = [], [], []
        c = SOURCE_TO_MUJOCO
        for binding in bindings:
            matrices = np.array([value_at_path(frame, binding['path']) for frame in frames]).reshape(-1, 4, 4)
            rotations = c @ matrices[:, :3, :3] @ c.T
            if (not np.all(np.isfinite(matrices)) or
                    not np.allclose(matrices[:, 3], [0, 0, 0, 1], atol=1e-8) or
                    not np.allclose(rotations @ np.swapaxes(rotations, -1, -2), np.eye(3), atol=2e-6) or
                    not np.allclose(np.linalg.det(rotations), 1., atol=2e-6)):
                raise ValueError(f'nonrigid source transform: {binding["name"]}')
            positions.append(matrices[:, :3, 3] @ c.T / 1000.)
            quaternions.append(np.roll(Rotation.from_matrix(rotations).as_quat(), 1, axis=1))
            sources.append(matrices)
        self.positions = np.stack(positions, axis=1)
        self.quaternions = np.stack(quaternions, axis=1)
        self.sources = sources
        self.duration = float(self.times[-1])

    def apply(self, model, data, time):
        import mujoco
        i, j, alpha = pair_at(self.times, time)
        ids = [b['mocap_id'] for b in self.bindings]
        data.mocap_pos[ids] = self.positions[i] * (1. - alpha) + self.positions[j] * alpha
        data.mocap_quat[ids] = slerp_many(self.quaternions[i], self.quaternions[j], alpha)
        data.time = float(time)
        mujoco.mj_forward(model, data)
        return i, j, alpha
