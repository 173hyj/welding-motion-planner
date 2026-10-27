import unittest
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from welding_kinematics import axis_matrix
from experimental.welding_candidate_catalog_analytic_v1.analytic import FanucAnalytic


def model_fixture():
    points = [[0,0,0],[312,0,670],[312,0,1745],[2000,0,1970],[1592,0,1970],[2000,0,1970]]
    axes = [[0,0,1],[0,1,0],[0,-1,0],[-1,0,0],[0,-1,0],[-1,0,0]]
    joints = [dict(key=str(i),axis1=p,axis2=(np.array(p)+a).tolist(),
                   leadingJoints=['1'] if i==2 else [],leadingFactors=[1.] if i==2 else [])
              for i,(p,a) in enumerate(zip(points,axes))]
    base = np.eye(4); base[:3,:3] = Rotation.from_euler('z',.4).as_matrix(); base[:3,3]=[40,-80,210]
    home = np.eye(4); home[:3,3]=[2952,2,1950]
    model = SimpleNamespace(joints=joints,base=base,tcp_home=home,q_initial=np.zeros(6),
                            bounds=np.array([[-3.3,-1.1,-2.4,-6.5,-2.3,-9.5],[3.3,1.4,1.6,6.5,2.3,9.5]]))
    model.limits_valid=lambda q: bool(np.all(q>=model.bounds[0]) and np.all(q<=model.bounds[1]))
    def tcp(q):
        t=np.eye(4)
        for i,j in enumerate(joints):
            p=np.array(j['axis1']); a=np.array(j['axis2'])-p
            t=t@axis_matrix(p,a,q[i]+(q[1] if i==2 else 0))
        return base@t@home
    model.tcp=tcp
    return model


class FanucAnalyticTests(unittest.TestCase):
    def test_random_forward_targets_recover_full_pose(self):
        model=model_fixture(); solver=FanucAnalytic(model)
        rng=np.random.default_rng(731)
        for _ in range(40):
            q=rng.uniform(model.bounds[0],model.bounds[1])
            solutions,stats=solver.solve(model.tcp(q))
            self.assertTrue(solutions, (q,stats))
            self.assertLess(min(np.linalg.norm(np.asarray(s['q_rad'])-q) for s in solutions),1e-6)
            self.assertTrue(all(s['position_error_mm']<1e-6 and s['rotation_error_rad']<1e-8 for s in solutions))

    def test_singular_wrist_is_deterministic_and_valid(self):
        model=model_fixture(); solver=FanucAnalytic(model)
        target=model.tcp(np.array([.2,-.3,.6,.8,0.,-.9]))
        a,_=solver.solve(target); b,_=solver.solve(target)
        self.assertTrue(a); self.assertEqual(a,b)

    def test_unreachable_and_wrong_geometry_are_rejected(self):
        model=model_fixture(); solver=FanucAnalytic(model)
        target=model.tcp(np.zeros(6)); target[0,3]+=10000
        self.assertEqual(solver.solve(target)[0],[])
        model.joints[2]['leadingFactors']=[-1.]
        with self.assertRaises(ValueError): FanucAnalytic(model)

    def test_cartesian_winding_expansion_covers_joint_four_and_six_together(self):
        model=model_fixture(); solver=FanucAnalytic(model)
        q=np.array([.2,-.3,.6,5.5,.7,8.])
        solutions,_=solver.solve(model.tcp(q))
        self.assertLess(min(np.linalg.norm(np.asarray(s['q_rad'])-q) for s in solutions),1e-6)

    def test_singular_family_includes_regular_five_degree_joint_four_grid(self):
        model=model_fixture(); solver=FanucAnalytic(model)
        q=np.array([.2,-.3,.6,np.deg2rad(40),0.,-.9])
        solutions,stats=solver.solve(model.tcp(q))
        self.assertTrue(stats['singular_family_sampled'])
        self.assertLess(min(np.linalg.norm(np.asarray(s['q_rad'])-q) for s in solutions),1e-6)

    def test_shoulder_singularity_is_recorded_and_sampled(self):
        model=model_fixture(); solver=FanucAnalytic(model)
        q=np.array([np.deg2rad(30),np.arcsin(-87/1075),np.pi/2,.3,.7,-.9])
        solutions,stats=solver.solve(model.tcp(q))
        self.assertTrue(stats['shoulder_singular_family'])
        self.assertLess(min(np.linalg.norm(np.asarray(s['q_rad'])-q) for s in solutions),1e-6)

    def test_singular_candidates_have_no_near_duplicate_actuator_states(self):
        model=model_fixture()
        model.bounds[0,3]=-np.deg2rad(210)-2e-8
        model.bounds[1,3]=np.deg2rad(210)+2e-8
        solver=FanucAnalytic(model)
        solutions,stats=solver.solve(model.tcp(np.array([.2,-.3,.6,np.deg2rad(40),0.,-.9])))
        self.assertTrue(stats['singular_family_sampled'])
        q=np.array([s['q_rad'] for s in solutions])
        distances=np.linalg.norm(q[:,None,:]-q[None,:,:],axis=2)
        np.fill_diagonal(distances,np.inf)
        self.assertGreaterEqual(float(distances.min()),1e-4)

    def test_matrix_classifies_exact_singular_wrist_despite_euler_roundoff(self):
        model=model_fixture(); solver=FanucAnalytic(model)
        q=np.array([.2,-.3,.6,.8,0.,-.9])
        solutions,stats=solver.solve(model.tcp(q))
        self.assertTrue(stats['singular_family_sampled'])
        # The same pose has the exact J4=40-degree representative below.
        equivalent=q.copy(); equivalent[3]=np.deg2rad(40); equivalent[5]=q[3]+q[5]-equivalent[3]
        self.assertLess(min(np.linalg.norm(np.asarray(s['q_rad'])-equivalent) for s in solutions),1e-6)

    def test_singular_pi_wrist_uses_correct_dependent_joint_six_sign(self):
        model=model_fixture(); model.bounds[0,4]=-3.3; model.bounds[1,4]=3.3
        solver=FanucAnalytic(model)
        q=np.array([.2,-.3,.6,.8,np.pi,-.9])
        solutions,stats=solver.solve(model.tcp(q))
        self.assertTrue(stats['singular_family_sampled'])
        equivalent=q.copy(); equivalent[3]=np.deg2rad(40); equivalent[5]=q[5]-q[3]+equivalent[3]
        self.assertLess(min(np.linalg.norm(np.asarray(s['q_rad'])-equivalent) for s in solutions),1e-6)

    def test_near_singular_wrist_retains_two_independent_wrist_angles(self):
        model=model_fixture(); model.bounds[0,4]=-3.3; model.bounds[1,4]=3.3
        solver=FanucAnalytic(model)
        for q5 in (1e-8,1e-7,np.pi-1e-8,np.pi-1e-7):
            q=np.array([.2,-.3,.6,.8,q5,-.9])
            solutions,_=solver.solve(model.tcp(q))
            self.assertLess(min(np.linalg.norm(np.asarray(s['q_rad'])-q) for s in solutions),1e-4,str(q5))


if __name__=='__main__': unittest.main()
