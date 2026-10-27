"""Independent pose, mesh, material, hash and rendered-motion acceptance."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from replay import DEFAULT_PLAN, configure_camera, prepare
from scene import ROOT, source_matrix
from timeline import pair_at


def verify(model, data, timeline, builder, metadata, output):
    output.mkdir(parents=True, exist_ok=True)
    body_error = rotation_error = vertex_error = 0.
    body_ids = [b['body_id'] for b in builder.bindings]
    body_sources = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b['name']): source_matrix(b['source_matrix'])
                    for b in builder.bodies}
    for k in np.unique(np.linspace(0, len(timeline.frames) - 1, 31).astype(int)):
        i, _, _ = timeline.apply(model, data, float(timeline.times[k]))
        for n, binding in enumerate(builder.bindings):
            expected = source_matrix(timeline.sources[n][i])
            body_sources[binding['body_id']] = expected
        for bid, expected in body_sources.items():
            body_error = max(body_error, float(np.max(np.abs(data.xpos[bid] - expected[:3, 3]))))
            rotation_error = max(rotation_error, float(np.max(np.abs(data.xmat[bid].reshape(3, 3) - expected[:3, :3]))))
        for gid in range(model.ngeom):
            mid = model.geom_dataid[gid]
            if mid < 0:
                continue
            first, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            vertices = model.mesh_vert[first:first + count][::max(1, count // 7)].astype(float)
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, model.mesh_quat[mid])
            local = vertices @ rotation.reshape(3, 3).T + model.mesh_pos[mid]
            expected = body_sources[model.geom_bodyid[gid]]
            world = local @ expected[:3, :3].T + expected[:3, 3]
            actual = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
            vertex_error = max(vertex_error, float(np.max(np.abs(world - actual))))
    if body_error > 1e-10 or rotation_error > 1e-9 or vertex_error > 1e-9:
        raise AssertionError(f'pose mismatch {body_error}, {rotation_error}, {vertex_error}')
    material_error = 0.
    for material in builder.manifest['materials']:
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, material['mujoco_name'])
        material_error = max(material_error, float(np.max(np.abs(model.mat_rgba[idx] - material['effective_rgba']))))
    if material_error > 3e-8:
        raise AssertionError('material conversion mismatch')
    timeline.apply(model, data, 0.)
    first_positions = data.xpos[body_ids].copy()
    max_displacement = 0.
    for t in np.linspace(0, timeline.duration, 51):
        timeline.apply(model, data, t)
        max_displacement = max(max_displacement, float(np.max(np.linalg.norm(data.xpos[body_ids] - first_positions, axis=1))))
    camera = mujoco.MjvCamera()
    configure_camera(camera, builder)
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    hashes, snapshots, panels = [], [], []
    with mujoco.Renderer(model, height=900, width=1440) as renderer:
        for index, t in enumerate([0., timeline.duration / 2., timeline.duration]):
            i, _, _ = timeline.apply(model, data, t)
            renderer.update_scene(data, camera=camera, scene_option=option)
            rgb = renderer.render().copy()
            hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
            image = Image.fromarray(rgb)
            ImageDraw.Draw(image).text((16, 16), f'MuJoCo | t={t:.3f}s | {timeline.frames[i]["phase"]} | strict 0/29', fill='white')
            name = f'frame_{index}_{t:.3f}s.png'
            image.save(output / name)
            snapshots.append(name)
            panels.append(image.resize((720, 450)))
    if len(set(hashes)) < 2 or max_displacement < 1e-5:
        raise AssertionError('rendered scene did not move')
    sheet = Image.new('RGB', (720, 1350))
    for i, panel in enumerate(panels):
        sheet.paste(panel, (0, i * 450))
    sheet.save(output / 'motion_contact_sheet.png')
    (output / 'model.xml').write_text(builder.xml_text)
    report = dict(passed=True, mujoco_version=mujoco.__version__, plan=metadata,
                  scene=builder.manifest, sampled_frame_count=31,
                  position_max_error_m=body_error, rotation_max_error=rotation_error,
                  rendered_vertex_world_max_error_m=vertex_error, material_rgba_max_error=material_error,
                  max_body_displacement_m=max_displacement, rendered_frame_sha256=hashes,
                  snapshots=snapshots, strict_planned_task_count=0, system_safe=False,
                  physics_simulation=False, note='Exact archived matrix replay; does not certify collision safety')
    if hashlib.sha256(Path(metadata['path']).read_bytes()).hexdigest() != metadata['file_sha256']:
        raise AssertionError('source plan changed')
    for mesh in builder.mesh_manifest:
        if hashlib.sha256((ROOT / 'build' / mesh['url']).read_bytes()).hexdigest() != mesh['sha256']:
            raise AssertionError('source mesh changed')
    report['source_hashes_unchanged'] = True
    (output / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (output / '验证报告.md').write_text(
        '# 上航发 MuJoCo 精确回放验证\n\n'
        f'- 计划：`{Path(metadata["path"]).name}`；{metadata["frame_count"]} 帧，{timeline.duration:.6f} s。\n'
        f'- 解压后 SHA256：`{metadata["decoded_sha256"]}`。\n'
        f'- {model.nmocap} 个运动 body，{model.nmesh} 个网格资产，{model.nmeshface:,} 个原始三角面，无简化。\n'
        f'- 最大位置误差 {body_error:.3g} m，旋转误差 {rotation_error:.3g}，网格世界坐标误差 {vertex_error:.3g} m。\n'
        f'- 51 个运动时刻采样，最大 body 位移 {max_displacement:.6f} m；三个渲染时刻哈希不同。\n'
        '- 原 GLB 顶点 RGB 数值保留，默认 OPAQUE 显示规则保留；无 NORMAL 属性时使用逐面法线。\n'
        '- 源 GLB 没有原 JT 的完整 PBR/纹理数据；MuJoCo 的光照与浏览器不同，不能承诺像素一致。\n'
        '- 两台 FANUC、两组焊枪、三工件、两夹具均包含；外围厂房用 --layout 可选。\n'
        '- 此模型是 mocap 精确视觉回放，关闭碰撞和动力学驱动；strict 0/29、system_safe=false。\n'
        '- 原始网格及计划哈希再次核对未改变；不生成视频。\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('scene', 'plan')}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=DEFAULT_PLAN)
    parser.add_argument('--layout', action='store_true')
    parser.add_argument('--scene',help='explicit scene placement file; archived original is the default')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / 'build/mujoco_replay_v1' / Path(args.plan).name.replace('.json.gz', '').replace('.json', '')
    model, data, timeline, builder, metadata = prepare(args.plan, args.layout,args.scene)
    verify(model, data, timeline, builder, metadata, output)


if __name__ == '__main__':
    main()
