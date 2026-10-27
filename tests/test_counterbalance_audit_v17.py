"""Independent closure and base-rotation covariance of passive source links."""
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('counterbalance_audit',ROOT/'scripts/audit_counterbalance_fk_v17.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class CounterbalanceAuditTests(unittest.TestCase):
    def test_closure_and_base_rotation_covariance(self):
        scene=json.loads((ROOT/'build/scene.json').read_text());a=np.array(scene['joints'][6]['axis1']);b=np.array(scene['joints'][8]['axis1']);axis=np.array(scene['joints'][7]['axis2'])-np.array(scene['joints'][7]['axis1'])
        for q2 in (-.8,0.,.8):
            q=np.zeros(6);q[1]=q2;_,home,_,_=audit.counterbalance(q,scene)
            for q1 in (-2.,-.8,0.,.8,1.5,2.):
                q[0]=q1;main,t,pa,pb=audit.counterbalance(q,scene);rz=np.eye(4);rz[:3,:3]=Rotation.from_euler('z',q1).as_matrix()
                for key in ('246','247'):np.testing.assert_allclose(t[key],rz@home[key],atol=1e-10)
                np.testing.assert_allclose(t['246'][:3,:3]@a+t['246'][:3,3],pa,atol=1e-10)
                np.testing.assert_allclose(t['247'][:3,:3]@b+t['247'][:3,3],pb,atol=1e-10)
                directed=t['246'][:3,:3]@axis
                # Source rounded axes differ by approximately 8e-7 radians.
                self.assertLess(np.linalg.norm(np.cross(directed/np.linalg.norm(directed),(pa-pb)/np.linalg.norm(pa-pb))),1e-6)
                np.testing.assert_allclose(t['246'][:3,:3].T@t['246'][:3,:3],np.eye(3),atol=1e-12)

if __name__=='__main__':unittest.main()
