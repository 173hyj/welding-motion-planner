"""Publish digest-bound no-QP results, including skipped and reused points."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DESKTOP = ROOT.parent / '上航发焊枪_无QP直线回退_RRT对照_最新.md'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def check_hashes(base, hashes):
    for name, expected in hashes.items():
        if digest(base / name) != expected:
            raise RuntimeError('Evidence changed: ' + name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'build/welding_direct_rrt_v1')
    args = parser.parse_args()
    out = args.out.resolve()
    report_path = out / 'report.json'
    report = read(report_path)
    cold = read(out / 'cold_verification.json')
    if not report['summary']['complete'] or report['manifest']['qp_used']:
        raise RuntimeError('Completed no-QP report required')
    if cold['report_sha256'] != digest(report_path) or not cold['summary']['all_passed']:
        raise RuntimeError('Digest-bound passing cold verification required')
    count = report['summary']['additional_staged_passed']
    if count != cold['summary']['reverified_staged_paths'] or count != cold['summary']['selected_paths']:
        raise RuntimeError('Verified path count mismatch')
    for section in ('source_sha256', 'previous_code_sha256', 'code_sha256'):
        check_hashes(ROOT, report['manifest'][section])
    check_hashes(ROOT, cold['verification_code_sha256'])
    check_hashes(out, report['output_sha256'])
    parent_path = ROOT / 'build/welding_variable_open_v1/report.json'
    parent = read(parent_path)
    parent_cold = read(parent_path.parent / 'cold_verification.json')
    if parent_cold['report_sha256'] != digest(parent_path) or not parent_cold['summary']['all_passed']:
        raise RuntimeError('Passing archived straight-path verification required')
    check_hashes(parent_path.parent, parent['output_sha256'])
    original = {r['weld_id']: r for r in parent['rows']}
    reused = [r['weld_id'] for r in parent['rows'] if r['best']]
    if len(reused) != parent_cold['summary']['reverified_retreats']:
        raise RuntimeError('Archived straight-path count mismatch')
    verified = {r['weld_id']: r for r in cold['verified_rows']}
    fresh = []
    failed = []
    lines = ['# 上航发：无QP直线回退＋RRT对照', '',
        f'本轮重新规划剩余{len(report["rows"])}个难点，{count}个找到退出路径并通过独立冷复核；复用之前已复核的{len(reused)}条至少100 mm直线退枪路径，合计**{len(reused)+count}/{len(parent["rows"])}个几何合格点有退出方案**。源场景29点中另有8点未通过初始相对几何筛选，不计入退出规划成功分母。', '',
        '## 方法', '',
        '固定原始焊接构型、机器人、枪体初始朝向、变位机和另一臂停靠。沿用已验证的开枪幅度可用前缀；本轮在30/60/80/185 mm驱动幅度下重新生成原射线库候选，每幅度最多2条，不使用QP输出，也不调用QP优化器。', '',
        '先保持末端朝向不变进行直线回退。依据真实整臂首次失败位置取其前3 mm，并结合枪体粗几何失败位置前5 mm；终点要求至少0.1 mm工具净空。直线前缀不足5 mm的候选跳过。通过枪体分离及完整双臂前缀检查后，用双向关节空间RRT转移，碰撞边按0.05°检查；RRT阶段允许末端朝向改变。目标TCP距焊点至少200 mm，工具净空至少5 mm。保存轨迹为真实关节路径及其FK。', '',
        '| 焊点 | 开枪驱动幅度/mm | 直线前缀/mm | 退出终点距焊点/mm | 终点工具净空/mm | 冷复核 |',
        '|---|---:|---:|---:|---:|---|']
    for row in report['rows']:
        if row['qp_used'] or row['initial'] != original[row['weld_id']]['initial']:
            raise RuntimeError('QP use or archived initial changed')
        if not row['best']:
            failed.append(row['weld_id'])
            lines.append(f'| {row["weld_id"]} | — | 未满足直线前缀条件 | — | — | 未生成路径 |')
            continue
        best = row['best']
        check = verified[row['weld_id']]
        if not check['passed'] or check['planned_path_sha256'] != best['path_sha256']:
            raise RuntimeError('Verified path identity mismatch')
        path = read(out / best['path_file'])
        if path['endpoint_displacement_from_weld_mm'] < 200 or path['endpoint_tool_clearance_mm'] < 5:
            raise RuntimeError('Endpoint requirement failed')
        fresh.append(row['weld_id'])
        lines.append(f'| {row["weld_id"]} | {best["amplitude_mm"]:g} | {path["straight_prefix_mm"]:.2f} | {check["endpoint_displacement_mm"]:.2f} | {check["endpoint_tool_clearance_mm"]:.2f} | 通过 |')
    initial_time = read(ROOT / 'build/welding_positioner_static_v3/timing.json')['end_to_end_wall_s']
    plan_time = report['summary']['wall_s']
    cold_time = cold['summary']['wall_s']
    total = initial_time + parent['summary']['wall_s'] + parent_cold['summary']['wall_s'] + plan_time + cold_time
    qp_total = initial_time + parent['summary']['wall_s'] + parent_cold['summary']['wall_s']
    for name in ('welding_waypoint_qp_v2/report.json', 'welding_waypoint_qp_v5/report.json',
                 'welding_waypoint_qp_v5/cold_verification.json', 'welding_waypoint_qp_v7/report.json',
                 'welding_waypoint_qp_v7/cold_verification.json'):
        qp_total += read(ROOT / 'build' / name)['summary']['wall_s']
    lines += ['', '## 未通过点', '']
    for row in report['rows']:
        if row['best']:
            continue
        travels = [s['final']['last_accepted_travel_mm'] for s in row['original_seeds']]
        lines += [f'{row["weld_id"]}：{len(travels)}条原始候选方向最后接受的直线行程为{min(travels):.3f}–{max(travels):.3f} mm；按照当前碰撞前安全余量和5 mm最短前缀条件，全部在RRT前被跳过，实际RRT尝试数为{sum(len(t["rrt_trials"]) for t in row["trials"])}。这不证明RRT无解，也不证明该焊点几何不可行。已有QP方案将该点直线前缀改善至72.85 mm并找到退出路径。', '']
    lines += ['## 耗时与对比', '',
        f'本轮6点重新生成原方向、直线前缀和RRT共{plan_time:.2f} s（包括模型加载{report["summary"]["model_load_s"]:.2f} s），5条路径独立冷复核{cold_time:.2f} s，新增实验合计{plan_time+cold_time:.2f} s，约{int((plan_time+cold_time)//60)}分{round((plan_time+cold_time)%60)}秒。采用4个worker，单点耗时不能直接相加作为总墙钟时间。', '',
        '| 阶段 | 实测/s |', '|---|---:|',
        f'| 原姿态生成，21个初始姿态 | {initial_time:.2f} |',
        f'| 原20档开度＋完整直线筛选 | {parent["summary"]["wall_s"]:.2f} |',
        f'| 原15条直线路径冷复核 | {parent_cold["summary"]["wall_s"]:.2f} |',
        f'| 本轮6点无QP直线前缀＋RRT | {plan_time:.2f} |',
        f'| 本轮5条路径冷复核 | {cold_time:.2f} |',
        f'| 各阶段合计 | {total:.2f} |', '',
        f'沿用原20档开度筛选流程，各阶段实测相加约{int(total//60)}分{round(total%60)}秒，结果20/21。此前带QP工程流程约{int(qp_total//60)}分{round(qp_total%60)}秒，结果21/21。此次仅额外重跑6个难点，姿态及15条完整直线路径为封存结果复用；不是本轮21点从零串行重跑的单次计时。', '',
        '两轮候选预算与退出目标不同：本轮最多8条原射线、全部目标至少200 mm；历史QP分阶段方案最多3条改善方向，多数目标至少100 mm，并包含一次冷复核失败后的修复。可以比较工程耗时和本次成功数，不能作为严格同预算论文消融或宣称固定加速倍率。实践上可以先用无QP方案，对未找到出口的点再启用优化。', '',
        '## 复核范围', '',
        '新进程重建源夹具网格，用源XML独立FK复核开枪、直线前缀和RRT保存边中点，5/5通过。保留源关节限位、耦合约束及完整双臂碰撞检查；保留原装配接口豁免（DONG与整个JING及两个悬挂接口组件）。枪体平移采用分离区间认证，完整机械臂仍是加密离散验收，不是严格连续全系统安全证明。', '',
        '此处RRT连接的是退出构型，尚未证明焊点之间完整转移、变位机运动、源布局到起点的路径以及焊枪内部装配接口。驱动幅度不是两电极实际间隙。', '',
        '## 复现命令', '', '读取已完成结果（不重新规划）：', '', '```bash',
        f"cd '{ROOT}'", 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experimental/welding_direct_rrt_v1/run_direct.py --workers 4 --resume', '```', '',
        '独立冷复核：', '', '```bash', f"cd '{ROOT}'",
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experimental/welding_waypoint_qp_checks/verify_staged.py --out build/welding_direct_rrt_v1 --workers 4', '```', '',
        '重新规划时使用全新输出目录：', '', '```bash', f"cd '{ROOT}'",
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experimental/welding_direct_rrt_v1/run_direct.py --workers 4 --out build/welding_direct_rrt_fresh', '```', '',
        f'结果：`{report_path}`；冷复核：`{out / "cold_verification.json"}`；真实q/TCP路径：`{out / "paths"}`。', '',
        f'规划报告SHA256：`{digest(report_path)}`。', f'冷复核SHA256：`{digest(out / "cold_verification.json")}`。']
    text = '\n'.join(lines) + '\n'
    DESKTOP.write_text(text)
    (out / '结果.md').write_text(text)
    audit = dict(report_sha256=digest(report_path), cold_sha256=digest(out / 'cold_verification.json'),
                 qp_used=False, reused_straight_points=reused, new_verified_staged_points=fresh,
                 unresolved_points=failed, composed_stage_wall_s=total,
                 desktop_sha256=digest(DESKTOP), report_script_sha256=digest(Path(__file__)))
    (out / 'delivery_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print('DIRECT_RRT_DESKTOP', DESKTOP)
    print('DIRECT_RRT_VERIFIED', len(reused) + len(fresh), '/', len(parent['rows']))
    print('DIRECT_RRT_COMPOSED_WALL_S', round(total, 2))


if __name__ == '__main__':
    main()
