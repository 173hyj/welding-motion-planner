"""Explicit cold-bound local replay registry, without modifying old registries."""
import gzip,hashlib,json
from pathlib import Path
import numpy as np


def audited_plan(root,path):
    path=Path(path)
    if not path.is_absolute():path=root/path
    registry=path.parent/'audit.json'
    audit=json.loads(registry.read_text());entry=audit['plans'].get(path.name)
    if entry is None or (root/entry['relative_path']).resolve()!=path.resolve():raise ValueError('Plan path identity mismatch')
    for name,sha in audit['source_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha:raise ValueError('Bound source changed '+name)
    cold=json.loads((root/audit['cold_path']).read_text())
    if not cold['summary']['all_selected_passed'] or not cold['summary']['retained_subset_route_verified']:raise ValueError('No verified retained route')
    raw=path.read_bytes();decoded=gzip.decompress(raw) if path.suffix=='.gz' else raw
    digest=hashlib.sha256(decoded).hexdigest()
    if digest!=entry['sha256'] or hashlib.sha256(raw).hexdigest()!=entry['file_sha256']:raise ValueError('Plan hash mismatch')
    plan=json.loads(decoded);frames=plan.get('frames',[])
    times=np.array([f['time_s'] for f in frames])
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times)<=0.) or times[0]!=0. or times[-1]<=0.:raise ValueError('Invalid strictly increasing timeline')
    if not plan['summary']['conditional_cold_verified_subset']:raise ValueError('Unverified plan scope')
    return plan,dict(path=str(path),audit_registry=str(registry),decoded_sha256=digest,file_sha256=hashlib.sha256(raw).hexdigest(),audit=entry,frame_count=len(frames),duration_s=float(times[-1]),conditional_collision_cold_path=audit['cold_path'],source_scene=audit['source_scene'],source_scene_sha256=audit['source_scene_sha256'])


def bound_scene(root,requested,metadata):
    expected=(root/metadata['source_scene']).resolve()
    actual=expected if requested is None else Path(requested)
    if not actual.is_absolute():actual=root/actual
    if actual.resolve()!=expected:raise ValueError('scene differs from cold-bound placement')
    if hashlib.sha256(expected.read_bytes()).hexdigest()!=metadata['source_scene_sha256']:raise ValueError('scene hash changed')
    return expected
