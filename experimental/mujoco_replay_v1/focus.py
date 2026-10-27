"""Actual source motion inspection and fixed-camera viewing regions."""
import numpy as np
from timeline import SOURCE_TO_MUJOCO


class MotionFocus:
    def __init__(self, frames):
        self.frames = frames
        data = {}
        for rid in frames[0]['robots']:
            q = np.array([f['robots'][rid]['q_rad'] for f in frames])
            tcp = np.array([f['robots'][rid]['tcp_matrix'] for f in frames]).reshape(-1, 4, 4)
            data[rid] = (q, tcp)
        self.robot_id = max(data, key=lambda rid: np.linalg.norm(np.ptp(data[rid][0], axis=0)))
        self.q, matrices = data[self.robot_id]
        self.tcp = matrices[:, :3, 3] @ SOURCE_TO_MUJOCO.T / 1000.
        self.center = (self.tcp.min(axis=0) + self.tcp.max(axis=0)) / 2.
        self.summary = dict(robot_id=self.robot_id, tcp_range_mm=np.ptp(matrices[:, :3, 3], axis=0).tolist(),
                            joint_range_deg=np.rad2deg(np.ptp(self.q, axis=0)).tolist(),
                            fixed_robot_ids=[rid for rid, (q, _) in data.items() if np.max(np.ptp(q, axis=0)) < 1e-8])
        # Every source frame is a source-FK TCP sample. Never connect distant
        # welds or skip intermediate bends just to reduce visual geometry.
        self.path = self.tcp.copy()
        self.path_pairs=np.array([(i,i+1) for i in range(len(frames)-1)
                                  if np.linalg.norm(self.path[i+1]-self.path[i])>=1e-8],dtype=int).reshape(-1,2)
        self.path_colors=np.array([
            [.15,.85,1.,.85] if frames[int(a)].get('phase') in ('ingress','retreat')
            else ([.8,.45,1.,.8] if frames[int(a)].get('phase')=='loop_return' else [1.,.6,.05,.8])
            for a,_ in self.path_pairs],dtype=np.float32).reshape(-1,4)
        self.summary['trajectory_source']='all saved source-FK TCP samples; no sparse weld-point chords'
        self.summary['trajectory_sample_count']=len(frames)
        self.distance = max(1.3, float(np.linalg.norm(np.ptp(self.tcp, axis=0))) * 1.8 + .7)

    def current(self, i, j, alpha):
        position = self.tcp[i] * (1 - alpha) + self.tcp[j] * alpha
        q = self.q[i] * (1 - alpha) + self.q[j] * alpha
        return dict(position=position, q_deg=np.rad2deg(q),
                    joint_delta_deg=np.rad2deg(q - self.q[0]),
                    tcp_displacement_mm=float(np.linalg.norm(position - self.tcp[0]) * 1000.))
