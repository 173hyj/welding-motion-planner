"""Preserve original candidate trajectories as locally readable exact bytes."""
import copy
import hashlib
import json
from pathlib import Path


def copy_candidate(weld_id,candidate,source,out,name):
    raw=(Path(source)/candidate['path_file']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=candidate['path_sha256']:
        raise ValueError('Retained candidate changed')
    path=json.loads(raw)
    if (path.get('weld_id',weld_id)!=weld_id or not path.get('passed') or
        path.get('length_mm')!=2000. or path.get('fixed_distance_mm')!=2000.):
        raise ValueError('Retained candidate weld/distance mismatch')
    target=Path(out)/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    result=copy.deepcopy(candidate);result['path_file']=name
    return result
