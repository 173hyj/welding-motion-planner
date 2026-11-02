import numpy as np

def certify_source_faces(*, vertices, faces, transform, origin, tcp_xyz, contacts,
                         product_face_count, raw_collision, truncated,
                         electrode_is_source, designated_product):
    radius_mm,depth_mm,tcp_tol_mm,max_contacts,outward_epsilon_mm=25.,2.,.1,10000,1e-6
    reasons=[]
    try:
        v=np.asarray(vertices,dtype=float); f=np.asarray(faces)
        T=np.asarray(transform,dtype=float); o=np.asarray(origin,dtype=float); tcp=np.asarray(tcp_xyz,dtype=float)
    except Exception: return {'passed':False,'failure_reasons':['non_numeric_geometry']}
    if v.ndim!=2 or v.shape[1]!=3 or not np.all(np.isfinite(v)): reasons.append('invalid_vertices')
    if f.ndim!=2 or f.shape[1]!=3 or not np.issubdtype(f.dtype,np.integer): reasons.append('invalid_faces')
    if T.shape!=(4,4) or not np.all(np.isfinite(T)) or not np.allclose(T[3],[0,0,0,1],atol=1e-12,rtol=0) or np.linalg.norm(T[:3,:3].T@T[:3,:3]-np.eye(3))>1e-6 or abs(np.linalg.det(T[:3,:3])-1)>1e-6: reasons.append('invalid_transform')
    if o.shape!=(3,) or not np.all(np.isfinite(o)) or tcp.shape!=(3,) or not np.all(np.isfinite(tcp)): reasons.append('invalid_pose')
    if reasons: return {'passed':False,'failure_reasons':reasons}
    if np.linalg.norm(tcp-o)>=tcp_tol_mm: reasons.append('tcp_error_ge_0_1mm')
    if not electrode_is_source: reasons.append('not_source_electrode')
    if not designated_product: reasons.append('wrong_product')
    if not raw_collision: reasons.append('no_raw_collision')
    if truncated: reasons.append('contacts_truncated')
    if not contacts: reasons.append('empty_contacts')
    if len(contacts)>=max_contacts: reasons.append('contact_count_ge_10000')
    bounds=[]
    if not reasons:
        for i,c in enumerate(contacts):
            try:
                mf,pf=int(c['moving_face']),int(c['product_face']); d=float(c['penetration_depth_mm'])
                if c['moving_face']!=mf or c['product_face']!=pf: raise ValueError('nonintegral face')
                pos=np.asarray(c['contact_xyz_mm'],dtype=float)
                if pos.shape!=(3,) or not np.all(np.isfinite(pos)): raise ValueError('invalid raw cpos')
            except Exception: reasons.append(f'contact_{i}_invalid_fields'); continue
            if mf<0 or mf>=len(f): reasons.append(f'contact_{i}_moving_face_invalid')
            if pf<0 or pf>=product_face_count: reasons.append(f'contact_{i}_product_face_invalid')
            if not np.isfinite(d) or d<0 or d>depth_mm: reasons.append(f'contact_{i}_depth_invalid')
            if mf>=0 and mf<len(f):
                idx=f[mf]; tri=v[idx] if np.all(idx>=0) and np.all(idx<len(v)) else None
                if tri is None or not np.all(np.isfinite(tri)): reasons.append(f'contact_{i}_triangle_invalid'); continue
                world=tri@T[:3,:3].T+T[:3,3]; bound=float(np.max(np.linalg.norm(world-o,axis=1)))
                bounds.append({'contact_index':i,'source_face':mf,'world_triangle':world.tolist(),'maximum_vertex_radius_mm':bound,'outward_radius_bound_mm':bound+outward_epsilon_mm})
                if bound+outward_epsilon_mm>radius_mm: reasons.append(f'contact_{i}_source_face_radius_gt_25mm')
    return {'passed':not reasons,'failure_reasons':reasons,'source_face_bounds':bounds,
            'maximum_source_face_radius_mm':max((b['maximum_vertex_radius_mm'] for b in bounds),default=None),
            'outward_epsilon_mm':outward_epsilon_mm,'radius_threshold_mm':radius_mm,
            'raw_cpos_preserved':True,'radius_proof':'Norm convexity over the full source contact triangle'}

def filter_certified_pairs(pairs, *, robot_id, product_id, certificates, phase):
    if phase not in ('closed','opening','closing'): return {'remaining_pairs':list(pairs),'removed_pairs':[]}
    rem=[]; removed=[]
    for p in pairs:
        key=p.get('link'); cert=certificates.get(key,{})
        if p.get('checked_robot_id')==robot_id and p.get('asset_id')==product_id and key in certificates and cert.get('passed') and 'distance_mm' not in p:
            removed.append(p)
        else: rem.append(p)
    return {'remaining_pairs':rem,'removed_pairs':removed}
