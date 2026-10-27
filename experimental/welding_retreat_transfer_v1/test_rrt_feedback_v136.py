from pathlib import Path
import feedback_reselection_v136 as feedback


def test_changed_node_archives_old_passing_path_and_record(tmp_path):
    (tmp_path/'paths').mkdir()
    (tmp_path/'paths/ab.json').write_text('old endpoint path')
    (tmp_path/'edge_00_01.json').write_text('old passing record')
    (tmp_path/'edge_02_03.json').write_text('unchanged record')
    known={(0,1):{'path_file':'paths/ab.json'},(2,3):{}}
    feedback.archive_incident(tmp_path,known,1,1)
    assert not (tmp_path/'edge_00_01.json').exists()
    assert not (tmp_path/'paths/ab.json').exists()
    assert (tmp_path/'edge_02_03.json').is_file()
    assert (tmp_path/'invalidated/revision_01/edge_00_01.json').read_text()=='old passing record'
    assert (tmp_path/'invalidated/revision_01/paths/ab.json').read_text()=='old endpoint path'
