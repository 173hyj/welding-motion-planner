"""Recover source world placements without changing frozen scene/plan evidence."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from build_scene import assemble, matrix, values
from import_project import xml_root


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source=ROOT/'source/psz/StandaloneStudy_PsState.xml'
    old_path=ROOT/'build/scene.json'
    plan_path=ROOT/'build/welding_29_staged_v2/welding_plan_29_staged_v2.json.gz'
    out=ROOT/'build/scene_placement_corrected_v1'
    out.mkdir(exist_ok=True)
    frozen={str(p.relative_to(ROOT)):sha(p) for p in [source,old_path,plan_path]}
    old=json.loads(old_path.read_text())
    recovered=assemble(export=False)
    objects={o.get('ExternalId'):o for o in xml_root(source.read_bytes()).find('Objects')}
    corrected=copy.deepcopy(old)
    recovered_assets={a['id']:a for a in recovered['assets']}
    rows=[]
    for asset in corrected['assets']:
        node=objects[objects[asset['id']].findtext('layout')]
        local=matrix(values(node.find('location')),values(node.find('rotation')))
        transform=np.array(recovered_assets[asset['id']]['transform']).reshape(4,4)
        chain=[node.get('ExternalId')]
        parent=objects.get(node.findtext('parent'))
        hierarchy_complete=node.findtext('parent')=='NULL' or parent is not None
        while parent is not None:
            if parent.tag!='PmLayout':raise ValueError('unexpected placement parent type')
            key=parent.get('ExternalId')
            if key in chain:raise ValueError('placement hierarchy cycle')
            chain.append(key)
            local=matrix(values(parent.find('location')),values(parent.find('rotation')))@local
            next_id=parent.findtext('parent')
            if next_id not in objects and next_id!='NULL':hierarchy_complete=False
            parent=objects.get(next_id)
        if hierarchy_complete:
            assert np.max(np.abs(local-transform))<1e-7
        previous=np.array(asset['transform']).reshape(4,4)
        delta=transform[:3,3]-previous[:3,3]
        asset['transform']=transform.ravel().tolist()
        rows.append(dict(asset_id=asset['id'],name=asset['name'],kind=asset['kind'],
                         source_layout_id=node.get('ExternalId'),source_parent_chain=chain,
                         hierarchy_complete=hierarchy_complete,
                         world_pose_source='PmLayout/NodeInfo/absoluteLocation',
                         before_world_matrix=previous.ravel().tolist(),after_world_matrix=transform.ravel().tolist(),
                         delta_source_xyz_mm=delta.tolist(),translation_change_mm=float(np.linalg.norm(delta)),
                         changed=bool(np.max(np.abs(transform-previous))>1e-8)))
    changed=[r for r in rows if r['changed']]
    assert len(changed)==2 and all(r['kind']=='fixture' for r in changed)
    assert corrected['welds']==old['welds']
    corrected['validation'].update(placement_corrected=True,
        world_pose_source='source PmLayout resolved absoluteLocation; independently composed complete parent chains',
        previous_scene_sha256=frozen[str(old_path.relative_to(ROOT))],
        old_collision_results_valid_for_this_scene=False,
        note='Only fixture placements corrected; historical robot trajectory may be replayed for diagnosis, but collision results must be recomputed')
    scene_path=ROOT/'build/scene_placement_corrected_v1.json'
    scene_path.write_text(json.dumps(corrected,ensure_ascii=False,indent=2))
    assert all(sha(ROOT/p)==v for p,v in frozen.items())
    report=dict(cause='Importer used relative layout pose as world pose and omitted parent assembly transform',
                parent_offset_source_xyz_mm=[0.,0.,4830.39435217581],
                scene=str(scene_path),corrected_scene_sha256=sha(scene_path),
                corrected_assets=changed,all_assets=rows,frozen_source_hashes=frozen,
                frozen_sources_unchanged=True,product_poses_unchanged=True,
                robot_poses_unchanged=True,gun_poses_unchanged=True,welds_unchanged=True,
                collision_evidence_stale_for_corrected_scene=True)
    (out/'placement_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    lines=['# 上航发工件与工作台位置核查','',
           '已确认导入错误：两个夹具遗漏父装配平移，工作台在原坐标Z方向错开4830.394mm；该方向在MuJoCo中是水平方向。',
           '按源XML的绝对位姿恢复，并用完整父布局变换链交叉验证。工件、机器人、焊枪、29个焊点位置不变。','',
           '修正版场景：`build/scene_placement_corrected_v1.json`。旧scene.json、源XML和封存轨迹文件哈希保持不变。',
           '修正场景后旧碰撞/RRT结果失效；现有轨迹只能用于诊断回放，不能据旧结果声称修正场景下规划成功。','',
           '|对象|原世界Z mm|正确世界Z mm|位移 mm|','|---|---:|---:|---:|']
    for row in changed:
        lines.append(f'|{row["name"]}|{row["before_world_matrix"][11]:.6f}|{row["after_world_matrix"][11]:.6f}|{row["translation_change_mm"]:.6f}|')
    (out/'位置核查报告.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('all_assets','corrected_assets')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
