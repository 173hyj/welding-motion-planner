"""Independence checks for raw XML coupling and Rodrigues registration."""
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('release_audit',ROOT/'scripts/audit_contact_release_v17.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class SourceFKAuditTests(unittest.TestCase):
    def test_source_coupling_factor_is_not_silently_ignored(self):
        raw=json.loads((ROOT/'build/robot.json').read_text());joints=audit.joints_from_raw(raw)[:6]
        q=np.zeros(6);q[1]=.4
        actual=audit.flange_source(q,joints)
        independently_expected=audit.source_rotation(joints[1],.4)@audit.source_rotation(joints[2],.4)
        np.testing.assert_allclose(actual,independently_expected,atol=1e-12)
        uncoupled=audit.source_rotation(joints[1],.4)
        self.assertGreater(np.max(abs(actual-uncoupled)),100.)

    def test_source_fk_agrees_archived_world_flange(self):
        scene=json.loads((ROOT/'build/scene.json').read_text());plan=json.loads((ROOT/'build/welding_plan_three_alternative.json').read_text());joints=audit.joints_from_raw(json.loads((ROOT/'build/robot.json').read_text()))[:6]
        for robot in (a for a in scene['assets']if a['kind']=='robot'):
            base=np.array(robot['transform']).reshape(4,4)
            for frame in plan['frames'][::20]:
                state=frame['robots'][robot['id']];actual=base@audit.flange_source(state['q_rad'],joints)
                np.testing.assert_allclose(actual,np.array(state['links']['245']).reshape(4,4),atol=1e-8,rtol=0)


if __name__=='__main__':unittest.main()
