import json
from types import SimpleNamespace
import pytest
import run_joint_pipeline_v129 as pipeline


@pytest.mark.parametrize('failure_stage', [None, 'joint_actions', 'transfers', 'partial_route'])
def test_pipeline_times_both_commands_and_does_not_accept_partial_route(monkeypatch, tmp_path, failure_stage):
    out = tmp_path/'new_run'
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    monkeypatch.setattr(pipeline.sys, 'argv', ['pipeline', '--out', str(out)])
    calls = []
    def run(cmd, **kwargs):
        folder = pipeline.Path(cmd[cmd.index('--out')+1])
        stage = folder.name
        calls.append(cmd)
        if stage == failure_stage:
            return SimpleNamespace(returncode=2)
        folder.mkdir()
        summary = dict(all_required_points_route_planned=failure_stage != 'partial_route')
        (folder/'report.json').write_text(json.dumps(dict(summary=summary)))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(pipeline.subprocess, 'run', run)
    if failure_stage is None:
        pipeline.main()
    else:
        with pytest.raises(SystemExit) as error:
            pipeline.main()
        assert error.value.code == 2
    report = json.loads((out/'pipeline_report.json').read_text())
    assert report['complete'] == (failure_stage is None)
    assert len(calls) == (1 if failure_stage == 'joint_actions' else 2)
    assert report['command_wall_s'] >= sum(s['command_wall_s'] for s in report['stages'])
    assert not report['independent_cold_performed'] and not report['prior_pose_hints']
    if len(calls) == 2:
        assert '--retreat-online' in calls[1]
        assert '--retreat-cold' not in calls[1]


def test_finite_negative_stage_report_is_preserved(monkeypatch, tmp_path):
    out = tmp_path/'negative'
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    monkeypatch.setattr(pipeline.sys, 'argv', ['pipeline', '--out', str(out)])
    def run(cmd, **kwargs):
        folder = pipeline.Path(cmd[cmd.index('--out')+1])
        folder.mkdir()
        (folder/'report.json').write_text(json.dumps(dict(summary={'passed': 20, 'points': 21})))
        return SimpleNamespace(returncode=2)
    monkeypatch.setattr(pipeline.subprocess, 'run', run)
    with pytest.raises(SystemExit):
        pipeline.main()
    report = json.loads((out/'pipeline_report.json').read_text())
    assert not report['complete'] and len(report['stages']) == 1
    assert report['stages'][0]['summary']['passed'] == 20
    assert len(report['stages'][0]['report_sha256']) == 64
