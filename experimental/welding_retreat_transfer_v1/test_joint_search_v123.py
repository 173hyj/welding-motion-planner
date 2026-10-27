import pytest
import joint_search_v123 as joint


def test_closed_only_success_does_not_stop_next_solution():
    attempted=[]
    def evaluate(c):
        attempted.append(c)
        return {'closed_passed':True,'passed':c==2,'path':{'passed':True} if c==2 else None}
    result,records=joint.first_complete([0,1,2,3],evaluate)
    assert result['passed'] and attempted==[0,1,2] and len(records)==3


def test_all_retreats_failed_is_not_a_weld_success():
    result,records=joint.first_complete([0,1],lambda c:dict(closed_passed=True,passed=False,path=None))
    assert result is None and len(records)==2


def test_missing_or_failed_path_cannot_claim_complete_action():
    with pytest.raises(ValueError):joint.first_complete([0],lambda c:dict(passed=True,path=None))
    with pytest.raises(ValueError):joint.first_complete([0],lambda c:dict(passed=True,path={'passed':False}))


def test_stationary_barrier_uses_complete_actions_and_preserves_failed_points():
    rows=[dict(weld_id='a',relative_gun_candidates=1,stationary_done=False,passed=False),dict(weld_id='b',relative_gun_candidates=1,stationary_done=True,passed=False),dict(weld_id='c',relative_gun_candidates=0,stationary_done=True,passed=False)]
    assert joint.jobs_for_stage(rows,'stationary')==[0]
    with pytest.raises(ValueError):joint.jobs_for_stage(rows,'positioner')
    rows[0]['stationary_done']=True
    assert joint.jobs_for_stage(rows,'positioner')==[0,1]
    assert len(rows)==3


def test_finite_grid_schedule_contains_all_and_only_original_five_degree_angles():
    angles=joint.positioner_angles()
    assert len(angles)==71 and len(set(angles))==71
    assert set(angles)==set(range(5,181,5))|set(range(-175,0,5))
    assert all(x%5==0 for x in angles)


def test_source_mesh_paths_include_articulated_assets_without_top_level_url():
    from pathlib import Path
    scene=dict(assets=[dict(kind='robot',robot=dict(link_models={'a':dict(url='meshes/link.glb')})),dict(kind='product',url='meshes/product.glb')])
    assert joint.asset_mesh_paths(scene,Path('/project'))==[Path('/project/build/meshes/link.glb'),Path('/project/build/meshes/product.glb')]


def test_expired_budget_rejects_before_any_geometric_query():
    import time
    from types import SimpleNamespace
    class Model:
        def tcp(self,q):return __import__('numpy').eye(4)
        def limits_valid(self,q):return True
    engine=SimpleNamespace(world=SimpleNamespace(models={'a':Model(),'b':Model()}))
    result=joint.try_configuration(engine,{'id':'w'},dict(robot_id='a'),[0.]*6,{'b':[[0.]*6]},dict(deadline=time.perf_counter()-1),{})
    assert not result['passed'] and result['reason']=='cooperative_budget_expired'
