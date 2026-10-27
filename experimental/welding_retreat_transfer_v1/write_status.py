"""Write separate primary-distance and diagnostic progress to desktop Markdown."""
import json
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DESKTOP = ROOT.parent/'上航发焊接_指定距离退枪与点间规划_最新.md'


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def status(folder):
    report = read(folder/'report.json')
    if report:
        return report['summary'], report
    progress = read(folder/'progress.json')
    if progress:
        return dict(progress, running=True), None
    rows = [read(p) for p in folder.glob('point_*.json')]
    return dict(running=True, finished=len(rows), passed=sum(r.get('passed', False) for r in rows)), None


def cold_text(report):
    s = report['summary']
    return (f'退枪{s["passed_retreats"]}/{s["selected_retreats"]}、'
            f'转移{s["passed_transfers"]}/{s["selected_transfers"]}通过，'
            f'子集路线验证={s["retained_subset_route_verified"]}，'
            f'全部21点路线验证={s["full_route_verified"]}，耗时{s["wall_s"]:.2f}秒')


def matching_cold(path, retreat_path, transfer_path=None):
    """Never display verification belonging to a different inventory or route."""
    result = read(path)
    if not result or not retreat_path.exists():
        return None
    if result['retreat_report_sha256'] != hashlib.sha256(retreat_path.read_bytes()).hexdigest():
        return None
    if transfer_path is not None:
        if not transfer_path.exists() or result['transfer_report_sha256'] != hashlib.sha256(transfer_path.read_bytes()).hexdigest():
            return None
    return result


def main():
    primary = ROOT/'build/welding_retreat_200cm_v1'
    refinement = ROOT/'build/welding_retreat_200cm_v2'
    diagnostic = ROOT/'build/welding_retreat_200mm_diagnostic_v1'
    transfer = ROOT/'build/welding_transfer_200mm_chart_v11'
    first, first_report = status(primary)
    refined, refined_report = status(refinement)
    diag, diag_report = status(diagnostic)
    trans_report = read(transfer/'report.json')
    trans_progress = read(transfer/'progress.json')
    cold_diag = read(ROOT/'build/welding_retreat_200mm_diagnostic_cold_v1/cold_verification.json')
    cold_refined = read(ROOT/'build/welding_retreat_200cm_cold_v2/cold_verification.json')
    cold_transfer = read(ROOT/'build/welding_retreat_transfer_200mm_chart_cold_v11/cold_verification.json')
    fragment_cold = read(ROOT/'build/welding_transfer_200mm_fragment_cold_v8/cold_verification.json')
    chart = read(ROOT/'build/welding_retreat_200mm_chart_v1/report.json')
    chart_cold = read(ROOT/'build/welding_retreat_200mm_chart_cold_v1/cold_verification.json')
    legal_primary = read(ROOT/'build/welding_retreat_200cm_legal_chart_v6/report.json')
    full_directions = read(ROOT/'build/welding_retreat_200cm_full_directions_v7/report.json')
    inventory = read(ROOT/'build/welding_retreat_200cm_inventory_v8/report.json')
    inventory_progress = read(ROOT/'build/welding_retreat_200cm_inventory_v8/progress.json')
    analytic = read(ROOT/'build/welding_retreat_200cm_inventory_analytic_v10/report.json')
    analytic_progress = read(ROOT/'build/welding_retreat_200cm_inventory_analytic_v10/progress.json')
    ik_benchmark = read(ROOT/'build/welding_source_analytic_ik_benchmark_v2.json')
    analytic_positioner = read(ROOT/'build/welding_retreat_200cm_analytic_positioner_v11/report.json')
    other_park = read(ROOT/'build/welding_retreat_200cm_other_park_v12/report.json')
    other_park_progress = read(ROOT/'build/welding_retreat_200cm_other_park_v12/progress.json')
    park45 = read(ROOT/'build/welding_retreat_200cm_other_park_v15/report.json')
    coarse_diagnostic = read(ROOT/'build/welding_coarse_failure_detail_v16/report.json')
    neighbor = read(ROOT/'build/welding_retreat_200cm_neighbor_rays_v17/report.json')
    cell = read(ROOT/'build/welding_retreat_200cm_cell_inventory_v20/report.json')
    cell_progress = read(ROOT/'build/welding_retreat_200cm_cell_inventory_v20/progress.json')
    merged_cell = read(ROOT/'build/welding_retreat_200cm_merged_cell_v21/report.json')
    merged_positive = read(ROOT/'build/welding_retreat_200cm_merged_positive_v32/report.json')
    remaining_positive = read(ROOT/'build/welding_retreat_200cm_remaining_positive_wrist_v34/report.json')
    positive_wrist = read(ROOT/'build/welding_retreat_200cm_positive_wrist_v31/report.json')
    positive_wrist_progress = read(ROOT/'build/welding_retreat_200cm_positive_wrist_v31/progress.json')
    mixed_branch = read(ROOT/'build/welding_retreat_200cm_mixed_branch_baseline_v40/report.json')
    mixed_wrist = read(ROOT/'build/welding_retreat_200cm_mixed_wrist_v41/report.json')
    mixed_progress = read(ROOT/'build/welding_retreat_200cm_mixed_wrist_v41/progress.json')
    mixed_park_probe = read(ROOT/'build/welding_mixed_wrist_park_v40_probe.json')
    merged_mixed = read(ROOT/'build/welding_retreat_200cm_merged_mixed_v42/report.json')
    merged_mixed_cold = read(ROOT/'build/welding_retreat_200cm_merged_mixed_cold_v42/cold_verification.json')
    mixed_transfer = read(ROOT/'build/welding_transfer_200cm_merged_mixed_v42/report.json')
    mixed_transfer_cold = matching_cold(ROOT/'build/welding_retreat_transfer_200cm_merged_mixed_cold_v42/cold_verification.json', ROOT/'build/welding_retreat_200cm_merged_mixed_v42/report.json', ROOT/'build/welding_transfer_200cm_merged_mixed_v42/report.json')
    positive_wrist_cold = matching_cold(ROOT/'build/welding_retreat_200cm_positive_wrist_cold_v31/cold_verification.json', ROOT/'build/welding_retreat_200cm_positive_wrist_v31/report.json')
    expanded_angles = read(ROOT/'build/welding_retreat_200cm_positive_angles_v35/report.json')
    expanded_progress = read(ROOT/'build/welding_retreat_200cm_positive_angles_v35/progress.json')
    merged_angles = read(ROOT/'build/welding_retreat_200cm_merged_positive_angles_v36/report.json')
    remaining_angles = read(ROOT/'build/welding_retreat_200cm_remaining_positive_angles_v37/report.json')
    early_cell = read(ROOT/'build/welding_retreat_200cm_early_cell_v26/report.json')
    early_cell_cold = matching_cold(ROOT/'build/welding_retreat_200cm_early_cell_cold_v26/cold_verification.json', ROOT/'build/welding_retreat_200cm_early_cell_v26/report.json')
    recent_snapshot_version = 'v29' if (ROOT/'build/welding_retreat_200cm_early_cell_v29/report.json').exists() else 'v27'
    recent_snapshot = read(ROOT/f'build/welding_retreat_200cm_early_cell_{recent_snapshot_version}/report.json')
    recent_snapshot_cold = matching_cold(ROOT/f'build/welding_retreat_200cm_early_cell_cold_{recent_snapshot_version}/cold_verification.json', ROOT/f'build/welding_retreat_200cm_early_cell_{recent_snapshot_version}/report.json')
    gw1_neighbors = read(ROOT/'build/welding_retreat_200cm_neighbor_rays_v25/report.json')
    accelerated_transfer = read(ROOT/'build/welding_transfer_200cm_coarse_search_v30/report.json')
    accelerated_progress = read(ROOT/'build/welding_transfer_200cm_coarse_search_v30/progress.json')
    accelerated_superseded = read(ROOT/'build/welding_transfer_200cm_coarse_search_v30/superseded_resource_status.json')
    actual_fragments = read(ROOT/'build/welding_transfer_200cm_actual_fragments_v33/report.json')
    actual_fragments_cold = matching_cold(ROOT/'build/welding_retreat_transfer_200cm_actual_fragments_cold_v33/cold_verification.json', ROOT/'build/welding_retreat_200cm_early_cell_v29/report.json', ROOT/'build/welding_transfer_200cm_actual_fragments_v33/report.json')
    replay_summary = read(ROOT/'build/welding_verified_route_v23/summary.json')
    replay_gui = read(ROOT/'build/welding_verified_route_v23/gui_full_route_smoke.json')
    early_primary = read(refinement/'early_cold/point_16_verified.json')
    primary_transfer = read(ROOT/'build/welding_transfer_200cm_other_park_v12/report.json') or read(ROOT/'build/welding_transfer_200cm_inventory_analytic_v10/report.json') or read(ROOT/'build/welding_transfer_200cm_inventory_v8/report.json') or read(ROOT/'build/welding_transfer_200cm_full_directions_v7/report.json') or read(ROOT/'build/welding_transfer_200cm_legal_chart_v10/report.json') or read(ROOT/'build/welding_transfer_200cm_v5/report.json')
    latest_primary_cold=read(ROOT/'build/welding_retreat_transfer_200cm_other_park_cold_v12/cold_verification.json') or read(ROOT/'build/welding_retreat_transfer_200cm_inventory_analytic_cold_v10/cold_verification.json') or read(ROOT/'build/welding_retreat_transfer_200cm_inventory_cold_v8/cold_verification.json') or read(ROOT/'build/welding_retreat_transfer_200cm_full_directions_cold_v7/cold_verification.json') or read(ROOT/'build/welding_retreat_transfer_200cm_legal_chart_cold_v10/cold_verification.json')
    primary_retreat_folder = 'welding_retreat_200cm_other_park_v12'
    primary_transfer_folder = 'welding_transfer_200cm_other_park_v12'
    primary_cold_folder = 'welding_retreat_transfer_200cm_other_park_cold_v12'
    for retreat_folder, transfer_folder, cold_folder in [
        ('welding_retreat_200cm_early_cell_v26', 'welding_transfer_200cm_early_cell_v26', 'welding_retreat_transfer_200cm_early_cell_cold_v26'),
        ('welding_retreat_200cm_early_cell_v27', 'welding_transfer_200cm_early_cell_v27', 'welding_retreat_transfer_200cm_early_cell_cold_v27'),
        ('welding_retreat_200cm_early_cell_v29', 'welding_transfer_200cm_early_cell_v29', 'welding_retreat_transfer_200cm_early_cell_cold_v29'),
        ('welding_retreat_200cm_early_cell_v29', 'welding_transfer_200cm_coarse_search_v30', 'welding_retreat_transfer_200cm_coarse_search_cold_v30'),
        ('welding_retreat_200cm_merged_cell_v21', 'welding_transfer_200cm_merged_cell_v21', 'welding_retreat_transfer_200cm_merged_cell_cold_v21'),
        ('welding_retreat_200cm_remaining_positive_angles_v37', 'welding_transfer_200cm_remaining_positive_angles_v37', 'welding_retreat_transfer_200cm_remaining_positive_angles_cold_v37'),
        ('welding_retreat_200cm_merged_mixed_v42', 'welding_transfer_200cm_merged_mixed_v42', 'welding_retreat_transfer_200cm_merged_mixed_cold_v42')]:
        candidate = read(ROOT/'build'/transfer_folder/'report.json')
        if candidate and (not primary_transfer or candidate['summary']['retained_retreat_nodes'] >= primary_transfer['summary']['retained_retreat_nodes']):
            primary_transfer = candidate
            primary_retreat_folder, primary_transfer_folder, primary_cold_folder = retreat_folder, transfer_folder, cold_folder
            latest_primary_cold = matching_cold(ROOT/'build'/cold_folder/'cold_verification.json', ROOT/'build'/retreat_folder/'report.json', ROOT/'build'/transfer_folder/'report.json')
    combined_transfer = read(ROOT/'build/welding_transfer_200cm_merged_positive_v32/report.json')
    if combined_transfer and (not primary_transfer or combined_transfer['summary']['retained_retreat_nodes'] >= primary_transfer['summary']['retained_retreat_nodes']):
        primary_transfer = combined_transfer
        primary_retreat_folder = 'welding_retreat_200cm_merged_positive_v32'
        primary_transfer_folder = 'welding_transfer_200cm_merged_positive_v32'
        primary_cold_folder = 'welding_retreat_transfer_200cm_merged_positive_cold_v32'
        latest_primary_cold = matching_cold(ROOT/'build'/primary_cold_folder/'cold_verification.json', ROOT/'build'/primary_retreat_folder/'report.json', ROOT/'build'/primary_transfer_folder/'report.json')
    remaining_transfer = read(ROOT/'build/welding_transfer_200cm_remaining_positive_wrist_v34/report.json')
    if remaining_transfer and (not primary_transfer or remaining_transfer['summary']['retained_retreat_nodes'] >= primary_transfer['summary']['retained_retreat_nodes']):
        primary_transfer = remaining_transfer
        primary_retreat_folder = 'welding_retreat_200cm_remaining_positive_wrist_v34'
        primary_transfer_folder = 'welding_transfer_200cm_remaining_positive_wrist_v34'
        primary_cold_folder = 'welding_retreat_transfer_200cm_remaining_positive_wrist_cold_v34'
        latest_primary_cold = matching_cold(ROOT/'build'/primary_cold_folder/'cold_verification.json', ROOT/'build'/primary_retreat_folder/'report.json', ROOT/'build'/primary_transfer_folder/'report.json')
    alternate = read(ROOT/'build/welding_retreat_200cm_v4/report.json')
    positioner_alternate = read(ROOT/'build/welding_retreat_200cm_v5/report.json')
    now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    lines = ['# 上航发：指定距离退枪与回退点间规划', '', f'更新：{now}', '',
        '**主要求按用户指定的200厘米（2000毫米、2米）执行。200毫米为额外诊断对照，未代替主要求。**', '',
        '## 主要求：200厘米直线退枪', '',
        f'首轮全部21个初始几何合格点完成搜索，找到合格2米直线退枪路径 **{first.get("passed", 0)}/21**，耗时{first.get("wall_s", 0):.2f}秒。源场景共29点，另外8点维持初始几何筛除，不计入本轮退出搜索。首轮每个可用驱动幅度最多8条整臂候选；未找到路径不证明数学无解。', '',
        '首轮及扩大方向搜索保持原焊接构型、朝向、变位机和另一臂停靠。后续另行记录合法等价初始角度或焊接姿态重选。每条通过路径必须真实达到2米直线长度，枪体分离区间认证和源双臂离散碰撞/关节耦合检查同时通过。不会把短直线加RRT的终点位移当作2米直线退枪。', '']
    if latest_primary_cold:
        s = latest_primary_cold['summary']
        label = '当前已接通并冷验的子集路线' if s['retained_subset_route_verified'] else '最新批次冷验结果（子集尚未接通验收）'
        lead = (f'**{label}：{s["passed_retreats"]}/{s["selected_retreats"]}条完整2000mm退枪、'
                f'{s["passed_transfers"]}/{s["selected_transfers"]}条实际点间连接通过；全部21点路线验证={s["full_route_verified"]}。** '
                '新的跨机械臂及外轴搜索继续运行，历史批次记录列在下方。')
        lines[6:6] = [lead, '']
    if mixed_branch:
        s = mixed_branch['summary']
        lines[6:6] = [f'**最新完整2000mm候选快照：{s["passed"]}/21个焊点；快照保持全部21点身份，原搜索仍在运行。** 候选、已冷验片段和已接通路线分别记录，不能把候选数量称为已接通数量。', '']
    for snapshot, snapshot_cold, label in [(early_cell, early_cell_cold, '首个增量快照'), (recent_snapshot, recent_snapshot_cold, '最新增量快照')]:
        if snapshot:
            s = snapshot['summary']
            lines += [f'{label}保留全部21点身份，完整2000mm规划候选{s["passed"]}/21；此快照不代表原19失败点的完整搜索已经结束。'
                      + (f' 退枪独立冷验{snapshot_cold["summary"]["passed_retreats"]}/{snapshot_cold["summary"]["selected_retreats"]}通过，点间路线另行规划和复核。' if snapshot_cold else ' 该快照的退枪冷验待完成。'), '']
    if recent_snapshot or early_cell:
        lines += ['最新增量库存中的完整2米候选：', '',
                  '| 焊点 | 变位机角度/度 | 活动机械臂ID | 开枪驱动/mm |',
                  '|---|---:|---|---:|']
        snapshot_folder = f'welding_retreat_200cm_early_cell_{recent_snapshot_version}' if recent_snapshot else 'welding_retreat_200cm_early_cell_v26'
        for row in (recent_snapshot or early_cell)['rows']:
            if row['best']:
                path = read(ROOT/'build'/snapshot_folder/row['best']['path_file'])
                lines += [f'| {row["weld_id"].replace("OP10L-", "")} | {path["positioner_deg"]:g} | {path["robot_id"]} | {path["driver_mm"]:g} |']
        lines += ['', '七节点退枪已冷验7/7、四条实边已独立冷验；旧转移批次已封存，固定分支新方案继续。旧三节点重复批次因交换空间耗尽且已被完整库存方案替代而停止；其已完成边/失败证据封存，不声称未完成边通过。最新管线先合并全部19失败点搜索和连续末轴外轴扩展结果，再对新增负末轴路径扩展变位机角度，保留全部21点身份；已封存同域同角度有限失败跳过。冷验后只复用端点相同且已冷验的旧边，再规划和最终复核。旧等待任务已替换，没有重复启动计算。', '',
                  '查看最新增量候选及其验收状态（只读）：', '', '```bash', f"cd '{ROOT}'", "python3 - <<'PY'", 'import json', 'from pathlib import Path',
                  f"for name in ['welding_retreat_200cm_early_cell_{recent_snapshot_version}/report.json', 'welding_retreat_200cm_early_cell_cold_{recent_snapshot_version}/cold_verification.json', 'welding_transfer_200cm_early_cell_{recent_snapshot_version}/report.json', 'welding_retreat_transfer_200cm_early_cell_cold_{recent_snapshot_version}/cold_verification.json']:",
                  "    p=Path('build')/name", "    print(name, json.loads(p.read_text())['summary'] if p.exists() else '尚未完成')", 'PY', '```', '']
    if (ROOT/'build/welding_transfer_200cm_coarse_search_v30/manifest.json').exists():
        lines += ['最新RRT提速实验：探索边按关节1度、外轴1度、枪开度2mm检查完整源状态，仅生成候选；候选整条路径必须再按原关节0.05度、外轴0.25度、枪开度0.5mm细验，失败边拒绝并重试，最终仍独立冷验。未放宽成功路径的源约束。薄障碍遗漏、细验失败拒绝、正反失败边禁用和成功绕行等6项RED/GREEN测试通过；未把理论采样比例当作实测提速。', '']
        if accelerated_transfer:
            s = accelerated_transfer['summary']
            lines += [f'提速批次完成有限规划：{s["retained_retreat_nodes"]}节点，{s["selected_passed_edges"]}/{s["expected_edges"]}选定连接通过细验，子集接通={s["all_retained_nodes_route_planned"]}，本批耗时{s["wall_s"]:.2f}秒；复用边的本批0秒不代表原规划无耗时。独立最终冷验状态见上方主要求结果。', '']
        elif accelerated_superseded:
            lines += ['旧七节点提速批次已由新的构型/停靠选择方案替代，计算已停止；23个已有输出的哈希已封存，未完成边不记为失败。v33保留7条退枪和4条实际转移的独立冷验；原v20全库存搜索及新v41计算继续。', '']
        else:
            s = accelerated_progress or {}
            lines += [f'提速批次正在运行：已记录{s.get("known_edges", 2)}个方向连接，其中{s.get("passed_edges", 2)}个通过（包括同一条旧冷验边的正反复用）；新边与整条路线仍需计算和独立冷复核。', '']
        lines += ['提速批次实际日志：', '', '```bash', 'tail -n 30 /tmp/welding_transfer_200cm_coarse_search_v30.log', '```', '']
    if actual_fragments:
        s = actual_fragments['summary']
        lines += [f'实际连接片段快照：完整保留7个回退节点及顺序，封存{s["selected_passed_edges"]}/{s["expected_edges"]}条已完成且通过细验的实际边（包含1条旧冷验边逆序复用）。缺口与失败记录保留；快照complete不表示搜索结束或7节点接通。'
                  + (f' 独立冷复核：{cold_text(actual_fragments_cold)}。' if actual_fragments_cold else ' 片段独立冷复核正在运行。'), '',
                  '片段冷验日志：', '', '```bash', 'tail -n 30 /tmp/welding_retreat_transfer_200cm_actual_fragments_cold_v33.log', '```', '']
    if (ROOT/'build/welding_retreat_200cm_positive_wrist_v31/manifest.json').exists():
        s = (positive_wrist or {}).get('summary', positive_wrist_progress or {})
        lines += [f'连续末轴构型重选：七节点中4条SUB2退枪路径使用负末轴，与其他节点的正末轴分支衔接代价高。保持各已通过外轴角度，在原5度姿态和全方向库中重新搜索正末轴初态及完整2米退枪；已完成{s.get("finished", s.get("requested_points", 0))}/4，已重选{s.get("positive_wrist_reselected", 0)}点。仅缩窄搜索域，不改源限位、FK及耦合；失败保留原合格路径，新正例仍须完整源冷验及全新端点连接。4项源模型/解析IK回归通过，不在运动中wrap。', '',
                  '最新完整库存合并与构型重选管线日志：', '', '```bash',
                  'tail -n 30 /tmp/welding_retreat_200cm_positive_wrist_v31.log',
                  'tail -n 30 /tmp/welding_retreat_transfer_200cm_positive_angles_v36_pipeline.log', '```', '']
        if positive_wrist_cold:
            lines += [f'首轮末轴重选后的完整库存冷验：{cold_text(positive_wrist_cold)}。此证书只验证退枪，不含新端点的转移路线。', '']
    if (ROOT/'build/welding_retreat_200cm_positive_angles_v35/manifest.json').exists():
        s = (expanded_angles or {}).get('summary', expanded_progress or {})
        text = (f'残留负末轴点外轴扩展：已完成{s.get("finished", s.get("requested_points", 0))}/1，'
                f'已重选{s.get("positive_wrist_reselected", 0)}点；变位机原成功角度及0/±30/±60/±90/±120/±150/180度，'
                '优先接近原角度，焊接旋转仍为5度。仅复用源码/设置/原几何/输出哈希及点角度一致的已完成有限失败，'
                '跳过重复搜索不代表无解。未成功则保留旧正例；新正例仍需独立冷验和新端点实际连接。')
        lines += [text, '', '```bash', 'tail -n 30 /tmp/welding_retreat_200cm_positive_angles_v35.log',
                  'tail -n 30 /tmp/welding_retreat_transfer_200cm_positive_angles_v36_pipeline.log', '```', '']
    if merged_angles or remaining_angles:
        s = (remaining_angles or merged_angles)['summary']
        lines += [f'最新完整合并库存：{s["passed"]}/21个完整2000mm候选，保持全部21点身份。'
                  '转移使用经典多起点最近邻与开放路径2-opt，失败边禁用后重排；真实完整15轴连接及最终冷验仍需分别核查。', '']
    if refined_report:
        lines += [f'扩大方向搜索后的结果：{refined["passed"]}/21，新增计算{refined["newly_computed_points"]}点、复用封存负结果{refined["reused_negative_points"]}点；本次新增耗时{refined["wall_s"]:.2f}秒。只对SUB3 GW-4、GW-5、GW-6采用每幅度48条分散候选，其他点沿用首轮负结果。', '']
    else:
        lines += ['扩大搜索正在运行：SUB3 GW-4、GW-5、GW-6采用原方向库的最远点采样排序，每个幅度最多48条完整整臂候选。首轮其中GW-6最长已接受直线前缀约1844.81毫米，故继续尝试其他方向。另18点负结果来自首轮封存证据。', '']
    if early_primary:
        lines += [f'已完成单点独立冷复核：SUB3 GW-6通过={early_primary["passed"]}，实际直线行程{early_primary["final_travel_mm"]:.6f}毫米，终点工具净空{early_primary["final_clearance_mm"]:.2f}毫米；这不表示整个21点批次已通过。', '']
    if alternate:
        lines += [f'两困难点额外焊接姿态搜索（仍为2米）：SUB3 GW-4搜索66个、GW-5搜索71个5度几何允许旋转/轴符号候选，有限IK分支及每幅度36个方向。新增成功0个，批次耗时{alternate["summary"]["wall_s"]:.2f}秒；使用终点IK仅作快速拒绝，失败不证明无解。', '']
    if positioner_alternate:
        s = positioner_alternate['summary']
        lines += [f'额外变位机姿态搜索（2米）：两个困难点尝试±30、±60、±90度，每角前8个5度旋转/轴符号候选、每幅度36个方向；新增成功0个，仍为{s["passed"]}/21，耗时{s["wall_s"]:.2f}秒。有限搜索失败不证明无解。', '']
    else:
        lines += ['额外变位机姿态搜索正在运行：两个困难点尝试±30、±60、±90度，保持5度焊接旋转网格。', '']
    if cold_refined:
        lines += [f'扩大搜索的2米路径冷复核：{cold_text(cold_refined)}。空路径集合不算安全通过。', '']
    if legal_primary:
        s = legal_primary['summary']
        lines += [f'最新2米主要求的合法等价初始角度复算：{s["passed"]}/21找到完整路径，本次耗时{s["wall_s"]:.2f}秒。GW-4末轴−187.854度改为172.146度，保持同一末端姿态，只从合法等价初始构型重新计算整条路径，不在运动中瞬间wrap。', '']
    else:
        lines += ['最新2米主要求正在重算GW-4：合法等价初始末轴角度−187.854度改为172.146度，两者同末端姿态且均在源限位内；重新计算整条路径。旧构型曾在−205度限位处停止。', '']
    if full_directions:
        s = full_directions['summary']
        lines += [f'最新2米额外姿态搜索已覆盖GW-4/GW-5的全部5度几何允许姿态，以及各姿态/开度允许的原方向库（没有截取最接近GW-6的36方向）；本批{s["passed"]}/21通过，本次耗时{s["wall_s"]:.2f}秒。终点有限IK仅用于拒绝，正结果必须通过整条2米源追踪。其他点仍复用v2封存证据。', '']
    elif (ROOT/'build/welding_retreat_200cm_full_directions_v7/manifest.json').exists():
        lines += ['最新2米额外姿态搜索运行中：GW-4/GW-5覆盖全部5度允许姿态和各姿态/开度允许的原方向库；保留终点有限IK拒绝、完整2米源追踪和独立冷复核。不截取最接近GW-6的36方向。', '']
    if inventory:
        s = inventory['summary']
        lines += [f'其余点的2米姿态扩展完成：本批重新搜索{s["newly_searched_points"]}点，全部仍纳入21点库存，合计{s["passed"]}/21找到2米路径，本批耗时{s["wall_s"]:.2f}秒。保持各点原分配机械臂和变位机角度，搜索原5度允许姿态及全方向库；有限逆解分支搜索不构成无解证明。', '']
    elif (ROOT/'build/welding_retreat_200cm_inventory_v8/manifest.json').exists():
        progress = inventory_progress or {}
        lines += [f'其余点的2米姿态扩展正在运行：18点中已完成{progress.get("finished", 0)}点，本批当前合计{progress.get("passed", 1)}/21找到2米路径。保持原机械臂分配和变位机角度，搜索原5度允许姿态及全方向库，结束后自动冷复核并规划合格节点之间的转移。', '']
    if analytic:
        s = analytic['summary']
        lines += [f'源FANUC解析逆解扩展批次完成：重新搜索{s["newly_searched_points"]}点，累计{s["passed"]}/21找到完整2米路径，耗时{s["wall_s"]:.2f}秒。普通非奇异肩/肘/腕解析分支全部试验，奇异腕族仅保留参考代表；原机器人、外轴角度、姿态/方向库和源碰撞标准保持。', '']
    elif (ROOT/'build/welding_retreat_200cm_inventory_analytic_v10/manifest.json').exists():
        s = analytic_progress or {}
        lines += [f'源FANUC解析逆解扩展运行中：20点已完成{s.get("finished", 0)}点，累计{s.get("passed", 1)}/21找到2米路径。源耦合及合法等价角度严格检查，沿途最大关节变化5度、直线误差0.05mm、朝向误差0.01度、全源碰撞标准保持，不因解析提速放宽验收。', '']
    if ik_benchmark:
        b = ik_benchmark
        lines += [f'解析IK验证：两机器人100个随机合法源构型的同构型/源FK回归通过；独立20目标基准解析{b["analytic_passed"]}/20、原有限数值IK{b["numerical_passed"]}/20，解析{b["analytic_s"]:.4f}秒、数值{b["numerical_s"]:.4f}秒，约{b["speedup"]:.1f}倍。该数字仅为IK求解，不能当作完整规划提速倍数；此基准未检查碰撞。', '']
        lines += ['初版解析IK在q5接近0的1e-9至1e-7 rad范围存在数值漏解；封存v9保留为旧实验，不作最新结果。新analytic_ik_v2改用atan2及近奇异参考分解，实际RED/GREEN回归通过，36项单测通过；v10以新版本重新搜索完整20失败点。源FK验收及碰撞门槛保持。', '']
        lines += ['近奇异解析解仍依赖参考构型，无参考或相距较远时可能保守拒绝；不声称穷尽近奇异连续解族，不把失败当成不可达证明。', '']
    if analytic_positioner:
        s = analytic_positioner['summary']
        lines += [f'源解析IK外轴扩展已结束：GW-4/GW-5在±15、±45、±75、±105、±135、±165、180度搜索全部5度允许姿态/全方向库，批次累计{s["passed"]}/21找到完整2米路径，本批耗时{s["wall_s"]:.2f}秒；各成功结果另需冷复核，未自动与其他批次合并。', '']
    elif (ROOT/'build/welding_retreat_200cm_analytic_positioner_v11/manifest.json').exists():
        lines += ['源解析IK外轴扩展运行中：GW-4/GW-5尝试±15、±45、±75、±105、±135、±165、180度，全部5度允许姿态及全方向库，不截取最接近GW-6的方向。各成功点退枪冷复核后，再用完整源限位RRT连接并最终冷验。', '']
    if other_park:
        s = other_park['summary']
        lines += [f'另一臂停靠重选批次结束：GW5对已通过工具直线几何、曾整臂碰撞失败的{s["trials"]}个候选，改用已知GW6的另一臂停靠状态，并重新检查开枪、完整2米工具几何和源双臂运动。本批累计{s["passed"]}/21找到2米路径，耗时{s["wall_s"]:.2f}秒；新正例仍需独立冷复核和回退节点间完整状态转移。', '']
    elif (ROOT/'build/welding_retreat_200cm_other_park_v12/manifest.json').exists():
        s = other_park_progress or {}
        lines += [f'另一臂停靠重选正在运行：GW5原候选在约502mm处由活动臂link241碰到停靠另一臂gun391_body阻断，工具与工件检查当时通过。现重选另一臂为GW6已用合法停靠，并重算源双臂开枪与完整2米退枪；已尝试{s.get("finished", 0)}个候选，不作瞬移或完整路线声明。', '']
    completed_analytic = []
    for p in (ROOT/'build/welding_retreat_200cm_inventory_analytic_v10').glob('OP10L*.json'):
        if 'progress' not in p.name:
            completed_analytic.append(read(p))
    if completed_analytic:
        lines += ['最新解析批次的已完成单点证据（候选次数包含不同开度，不能相加当作唯一方向数；前置拒绝不构成数学无解证明）：', '',
                  '| 焊点 | 姿态候选 | 源合法初始IK候选 | 终点IK拒绝次数 | 工具沿途几何拒绝次数 | 完整2m通过 |',
                  '|---|---:|---:|---:|---:|---|']
        for row in sorted(completed_analytic, key=lambda r: r['weld_id']):
            poses = row.get('alternative_pose_trials', [])
            endpoint_rejected = geometry_rejected = 0
            for pose in poses:
                for trial in pose['trials']:
                    for ray in trial['directions']:
                        endpoint_rejected += ray.get('reason') == 'finite_endpoint_ik_prefilter_rejected'
                        geometry_rejected += ray.get('reason') == 'coarse_geometry_rejected'
            lines += [f'| {row["weld_id"].replace("OP10L-", "")} | {len(poses)} | {sum(p["starts"] for p in poses)} | {endpoint_rejected} | {geometry_rejected} | {"是，待冷验" if row["best"] else "未找到"} |']
        lines += ['']
    if primary_transfer:
        s = primary_transfer['summary']
        lines += [f'主要求点间规划：{s["retained_retreat_nodes"]}个合格回退节点，需要{s["expected_edges"]}条连接，选定通过{s["selected_passed_edges"]}条；子集接通={s["all_retained_nodes_route_planned"]}，全21点接通={s["all_required_points_route_planned"]}，耗时{s["wall_s"]:.2f}秒。只有一个节点时没有点间运动。', '',
                  '主要求访问顺序：'+' → '.join(primary_transfer['order_weld_ids']), '']
    if latest_primary_cold:
        lines += [f'最新2米退枪及转移独立冷复核：{cold_text(latest_primary_cold)}。', '']
    if park45:
        lines += [f'GW4在明确45度变位机及重选另一臂停靠状态下，重试6个完整几何候选，未新增2米路径，耗时{park45["summary"]["wall_s"]:.2f}秒。最长直线前缀1876.625mm，后续解析解需约360度关节跳变，被连续构型约束拒绝；未在运动中wrap。该前缀不计为合格2米回退。', '']
    if coarse_diagnostic:
        s = coarse_diagnostic['summary']
        lines += [f'提前几何拒绝诊断完成：在各点正确外轴状态下复现{s["points"]}点、{s["sampled"]}条原拒绝方向，耗时{s["wall_s"]:.2f}秒。96条在5mm处可按释放规则接受，但均未通过完整2000mm几何认证；不能据此声称获得新整臂路径。', '']
    if neighbor:
        s = neighbor['summary']
        lines += [f'GW4邻近球面方向搜索结束：方位/仰角均为5度网格，尝试{s["directions_tried"]}条，库存{s["passed"]}/21，本批耗时{s["wall_s"]:.2f}秒。新正例必须另行冷复核及重新规划完整状态转移。', '']
    elif (ROOT/'build/welding_retreat_200cm_neighbor_rays_v17/manifest.json').exists():
        lines += ['GW4邻近方向搜索运行中：明确45度外轴及源合法另一臂停靠，每个初始分支尝试原方向附近256个5度方位/仰角网格，补原方向库缺失的5度仰角。完整2米几何和双臂验收保持。', '']
    if gw1_neighbors:
        s = gw1_neighbors['summary']
        lines += [f'GW1在60度外轴、另一台机械臂的最长失败前缀1667.00mm处，复现活动臂link243与link245自碰撞，工具阻断对为空，源限位仍有效。随后补搜{s["directions_tried"]}个5度球面邻近方向，未新增完整2米路径，耗时{s["wall_s"]:.2f}秒；有限方向失败不证明无解。', '']
    lines += ['v13和v14是漏设外轴状态的无效诊断批次，已封存；其失败不作为正确角度下的负证据。v15/v16已明确设置实际角度并通过只读审查。此前v12正路径及组合冷验保持有效。', '']
    if cell:
        s = cell['summary']
        lines += [f'跨机械臂/外轴搜索完成：{s["requested_points"]}个原失败点搜索两台机械臂及原角、0/±30/±60/±90/±120/±150/180度，源5度姿态及原方向库；库存{s["passed"]}/21，本批增量{s["wall_s"]:.2f}秒。新增正例另行合并、冷验和实际RRT连接，不能仅凭找到候选称路线通过。', '']
    elif (ROOT/'build/welding_retreat_200cm_cell_inventory_v20/manifest.json').exists():
        s = cell_progress or {}
        lines += [f'跨机械臂/外轴搜索正在运行：19个失败点中已完成{s.get("finished", 0)}点，当前批次库存{s.get("passed", 2)}/21。两台机械臂和原角、0/±30/±60/±90/±120/±150/180度逐点有限搜索，源5度姿态/原方向库、显式另一臂停靠，成功一点后停止该点搜索。增加连续IK提前拒绝以节省完整碰撞计算；完整正路径验收保持。v18已因缓存键改进停止封存，当前v20键含机器人ID。', '']
    lines += ['主要求仍未形成覆盖21点的完整路线。只有距离和独立冷复核均通过的节点，才可进入点间规划。', '',
              '本轮各点的2米搜索记录（沿用其他点的封存结果；最长失败前缀仅供诊断，未作为合格退枪输出，也不是最大可达距离证明）：', '',
              '| 焊点 | 完整2000mm通过 | 整臂候选数 | 最长失败前缀/mm | 该前缀终止原因 |',
              '|---|---|---:|---:|---|']
    available_inventories = [x for x in (merged_mixed, mixed_wrist, mixed_branch, remaining_angles, merged_angles, expanded_angles, remaining_positive, merged_positive, merged_cell, positive_wrist, recent_snapshot, early_cell, cell, other_park, analytic, inventory, full_directions, legal_primary, refined_report, first_report) if x]
    current_inventory = max(available_inventories, key=lambda x: sum(bool(r.get('best')) for r in x.get('rows', [])), default={})
    for row in current_inventory.get('rows', []):
        trials = row.get('trials', [])
        failures = [t for t in trials if not (t.get('arm_passed') or t.get('passed'))]
        longest = max(failures, key=lambda t: t.get('last_accepted_travel_mm') or 0., default={})
        reason = longest.get('reason', '未进入整臂追踪；工具直线几何筛选未通过')
        reason = {'continuous_ik_failed': '连续逆解/限位条件未通过',
                  'full_retreat_blocked': '整臂或工具源碰撞未通过'}.get(reason, reason)
        wid = row['weld_id'].replace('OP10L-', '')
        lines += [f'| {wid} | {"是" if row["best"] else "否"} | {len(trials)} | {longest.get("last_accepted_travel_mm") or 0.:.2f} | {reason} |']
    lines += ['', '## 200毫米诊断对照', '']
    if diag_report:
        lines += [f'全部21点诊断完成：{diag["passed"]}/21找到严格200毫米直线退枪，耗时{diag["wall_s"]:.2f}秒。', '']
    else:
        lines += [f'诊断仍在运行，进度：{diag}。', '']
    if cold_diag:
        s = cold_diag['summary']
        lines += [f'新进程独立冷复核：{s["passed_retreats"]}/{s["selected_retreats"]}条退枪通过，耗时{s["wall_s"]:.2f}秒。', '']
    if trans_report:
        s = trans_report['summary']
        lines += [f'点间规划：保留{ s["retained_retreat_nodes"]}个已冷复核的诊断回退节点，要求{s["expected_edges"]}条转移边，选定已通过规划检查{s["selected_passed_edges"]}条；子集全部接通={s["all_retained_nodes_route_planned"]}；原21点全部接通={s["all_required_points_route_planned"]}。耗时{s["wall_s"]:.2f}秒。', '',
                  '访问顺序：'+' → '.join(trans_report['order_weld_ids']), '']
    elif trans_progress:
        lines += [f'点间规划正在运行，第{trans_progress.get("round", 0)+1}轮；已记录{trans_progress.get("known_edges", 0)}条有向边，其中{trans_progress.get("passed_edges", 0)}条通过，已运行{trans_progress.get("elapsed_s", 0):.2f}秒。这些边不一定构成一条完整路线。', '']
    else:
        lines += ['点间规划准备/运行中，尚无最终结果。', '']
    if fragment_cold:
        lines += [f'原9节点方案的封存片段冷复核：退枪9/9、实际转移6/6通过，耗时{fragment_cold["summary"]["wall_s"]:.2f}秒；缺两条连接，未形成完整路线。', '']
    if chart:
        lines += ['发现部分端点末轴正负分支切换经过源腕部自碰撞，已尝试合法等价旋转及另一腕部逆解分支：同一末端姿态/同一退枪直线的8/9条完整200毫米路径通过源检查，SUB1 GW-1改分支后失败并明确剔除；不是原9点全部成功。新8节点重新排序及转移规划。', '']
    if chart_cold:
        lines += [f'新腕部构型路径冷复核：{cold_text(chart_cold)}。', '']
    if cold_transfer:
        lines += [f'退枪及选定点间转移冷复核：{cold_text(cold_transfer)}。', '']
    if trans_report:
        lines += ['选定转移边（保存实际15轴轨迹和两TCP的逐帧FK，种子复用边的本次耗时为0，不代表最初规划无需耗时）：', '',
                  '| 起点 | 终点 | 保存帧数 | 本次规划秒数 |', '|---|---|---:|---:|']
        for edge in trans_report['selected_edges']:
            lines += [f'| {edge["from_weld_id"].replace("OP10L-", "")} | {edge["to_weld_id"].replace("OP10L-", "")} | {edge.get("saved_frames", 0)} | {edge.get("elapsed_s", 0.):.2f} |']
        lines += ['']
    if replay_summary and replay_gui:
        lines += ['## 已冷验两点路线的原色MuJoCo回放', '',
            f'GW5/GW6两点：{replay_summary["frame_count"]}个真实源FK帧；焊接停留示意→开枪→2000mm退枪→实际转移→沿原路径反向进枪/闭合→焊接停留示意→开枪及2000mm退枪。原色完整GLB，4,516,844个唯一三角面，未生成视频。', '',
            f'显示时间{replay_summary["duration_s"]:.3f}秒，仅按显示速度和示意停留生成，不作为真实焊接周期/速度/加速度验收。原生窗口全路线8倍速测试通过，实际到达{replay_gui["max_playback_time_s"]:.3f}秒，覆盖进枪、闭合、开枪、退枪、转移、示意停留6阶段；最终示意完成2点。', '',
            '源XML独立TCP最大矩阵误差3.64e-12，闭环机构误差1.82e-12；全部15轴接缝最大4.44e-16。入口绑定正确场景、全部36个实际网格及显示源码，拒绝切到旧场景仍使用新冷验标记。轨迹线来自全部保存的源FK采样，未用焊点间稀疏直线代替。', '',
            '此回放从GW5闭合姿态开始，初始部署到该姿态未认证；仍仅为2/21子集，保留原冷验的装配接口与加密离散检查边界。当前连接在完整状态直连中通过，RRT-Connect作为直连受阻时的后备规划器，没有把本条边说成RRT搜索生成。', '',
            '```bash', f"cd '{ROOT}'",
            "DISPLAY=:0 /home/ubuntu22/桌面/activenet_4070_offload_20260717/rl_sparse_qp_planner/.venv/bin/python experimental/welding_retreat_transfer_v1/replay_verified_route_v23.py --plan build/welding_verified_route_v23/welding_plan_verified_200cm_subset_v23.json.gz --scene build/scene_placement_corrected_v1.json --no-loop",
            '```', '', 'Space暂停/继续；R重置；N/B切换阶段；T显示实际轨迹线；F切换视角；+/-调速。默认正常速度播放一次；需要循环时去掉--no-loop。', '']
    if mixed_wrist or mixed_progress:
        s = (mixed_wrist or {}).get('summary', mixed_progress or {})
        lines += ['', '## 固定机器人末轴分支与一致停靠（v41）', '',
                  f'为减少末轴正负分支切换，v41为R2固定负末轴、R1固定正末轴，并为非工作臂使用同分支停靠构型；这批搜索不复用其他停靠/搜索域的有限负结果。当前已完成{s.get("finished", s.get("points", 0))}/21项，固定分支兼容{s.get("branch_compatible", 0)}项，保留完整2米候选{s.get("retained_complete_retreats", s.get("passed", 0))}项。',
                  '固定分支静态全源检查结果：0、±30、±60、±90、±120、±150、180度中，−120和180度发生变位机布局阻挡；其余采样角度通过静态检查。静态通过不代表运动通过，仍需完整退枪冷验和点间运动。',
                  'GW5/GW6/GW8三条热重算路径已独立检查全帧和中点：2000mm、真实15轴端点、另一臂停靠固定、源XML FK误差约3.64e−12；GW6保留原状态可由封存基线追溯，但热重算元数据未重复写入original_initial。', '']
        lines += ['当前计算日志（只读）：', '', '```bash',
                  'tail -n 30 /tmp/welding_retreat_200cm_mixed_wrist_v41.log',
                  'tail -n 30 /tmp/welding_retreat_transfer_200cm_mixed_v42_pipeline.log', '```', '',
                  '固定分支搜索复现（全新输出目录，保留21点；不覆盖当前计算）：', '', '```bash',
                  f"cd '{ROOT}'",
                  'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experimental/welding_retreat_transfer_v1/probe_mixed_wrist_v41.py --retreat-out build/welding_retreat_200cm_mixed_branch_baseline_v40 --out build/welding_retreat_200cm_mixed_wrist_fresh',
                  '```', '']
    if merged_mixed or mixed_transfer or mixed_transfer_cold:
        if merged_mixed:
            lines += [f'v42混合末轴/停靠合并库存：{merged_mixed["summary"]["passed"]}/21个完整2米候选，正在执行转移和最终冷验。', '']
        if mixed_transfer:
            lines += [f'v42经典顺序与真实转移：选定{mixed_transfer["summary"].get("selected_passed_edges", 0)}/{mixed_transfer["summary"].get("expected_edges", 0)}条边，完整路线={mixed_transfer["summary"].get("full_route_planned", False)}。', '']
        if mixed_transfer_cold:
            lines += [f'v42最终独立冷验：{cold_text(mixed_transfer_cold)}。', '']
    lines += ['## 顺序与转移算法', '',
        '采用经典多起点最近邻＋开放路径2-opt，以两臂实际关节、变位机角度、两枪开度变化建立初始代价，并加入机器人交接惩罚20（无量纲启发式，非实测时间）；用真实规划路径代价更新，失败边禁止使用并重新排序。该方法是启发式，不保证全局最优。', '',
        '最新200mm八节点诊断采用经典Held–Karp有向开放路径动态规划；未知边增加50的探索惩罚（无量纲），仅用于排序，不改真实路径长度。它精确优化给定代价矩阵，不代表未知完整物理运动图上的全局最优。新停靠库基于合法恒定腕部分支，2268/2493双臂组合在0/5/25度源模型静态通过；最大FK等价差1.82e-12，运动仍逐段检查。v11复用18条已完成源通过边并补足缺口，报告的51.75秒是本批增量规划耗时，未包含以前退枪、种子规划和冷验证。', '',
        '转移采用RRT-Connect，先尝试完整状态直连；共同停靠阶段让单臂关节与该枪开度一起运动，然后旋转变位机、部署下一构型，避免在工件旁强制闭枪，必要时联合搜索。200mm新腕部分支实验限定两臂末轴0.0001至205度（有限搜索域，冷复核仍用完整源限位）；最新2米全方向姿态搜索使用完整源末轴范围的转移连接器。不在运动中wrap。停靠候选的静态通过不代表运动通过，每一段仍须全源碰撞检查。全程保存15维状态（两臂12关节＋变位机角度＋两枪开度）及两TCP的真实FK轨迹。不会瞬移另一臂、枪口开度或变位机。正转移边按关节0.05度、外轴0.25度、开度0.5毫米采样；提前粗检查只拒绝失败边，不代替正路径加密检查。已通过源模型检查的静态转移路径可逆序复用，最终仍独立冷复核。', '',
        '## 证据与限制', '',
        '新进程重建原始夹具面片并采用源XML独立FK复核保存路径和边中点。保留原装配接口豁免、机器人底座安装接口以及焊枪内部装配未认证的边界；这仍是加密离散全系统验收，不是严格全程连续安全证明。焊点间转移的验证目标是回退节点之间；尚未额外认证完整焊接时间/速度参数。', '',
        '首轮退枪程序没有在启动时单独封存附件XML和复用方向路径。已补来源审计：当前15条方向路径与历史封存哈希全匹配，3个产品附件关系匹配，但不把事后记录说成运行全程哈希证明。改进版启动及结束均核验这些来源。', '',
        '## 命令', '', '查看首轮2米结果（读取封存结果，不重复规划）：', '', '```bash',
        f"cd '{ROOT}'", "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experimental/welding_retreat_transfer_v1/run_retreat.py --distance-mm 2000 --unit-confirmation 'literal user request: 200cm=2000mm; optional unit correction pending' --workers 4 --arm-candidates 8 --out build/welding_retreat_200cm_v1 --resume", '```', '',
        '读取/继续扩大2米方向搜索：', '', '```bash', f"cd '{ROOT}'",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experimental/welding_retreat_transfer_v1/run_retreat_v2.py --distance-mm 2000 --unit-confirmation 'literal user request 200cm=2000mm; optional correction pending' --workers 3 --arm-candidates 48 --weld-id OP10L-SUB3-POINT-GW-6 --weld-id OP10L-SUB3-POINT-GW-5 --weld-id OP10L-SUB3-POINT-GW-4 --reuse-result build/welding_retreat_200cm_v1 --out build/welding_retreat_200cm_v2 --resume", '```', '',
        '诊断点间规划复现（200毫米，非2米主要求；使用全新目录）：', '', '```bash', f"cd '{ROOT}'",
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experimental/welding_retreat_transfer_v1/run_transfer_v11.py --retreat-out build/welding_retreat_200mm_chart_v1 --retreat-cold build/welding_retreat_200mm_chart_cold_v1/cold_verification.json --seed-snapshot build/welding_transfer_200mm_chart_seed_snapshot_v11.json --out build/welding_transfer_200mm_chart_fresh --workers 1 --iterations 1200 --ordering-rounds 3', '```', '',
        '诊断整套独立冷复核（点间规划完成后执行）：', '', '```bash', f"cd '{ROOT}'",
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experimental/welding_retreat_transfer_v1/verify_v3.py --retreat-out build/welding_retreat_200mm_chart_v1 --transfer-out build/welding_transfer_200mm_chart_v11 --out build/welding_retreat_transfer_200mm_chart_cold_fresh --workers 2', '```', '',
        '主要求最新2米结果读取（不重复规划）：', '', '```bash', f"cd '{ROOT}'",
        "python3 - <<'PY'", 'import json', "from pathlib import Path", f"for p in [Path('build/{primary_retreat_folder}/report.json'), Path('build/{primary_transfer_folder}/report.json'), Path('build/{primary_cold_folder}/cold_verification.json')]:", "    print(p, json.loads(p.read_text())['summary'] if p.exists() else '尚未完成')", 'PY', '```', '',
        '主要求两点之间的规划复现（2米回退节点；新输出目录）：', '', '```bash', f"cd '{ROOT}'",
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experimental/welding_retreat_transfer_v1/run_transfer_v6.py --retreat-out build/welding_retreat_200cm_other_park_v12 --retreat-cold build/welding_retreat_200cm_other_park_cold_v12/cold_verification.json --out build/welding_transfer_200cm_other_park_fresh --workers 1 --iterations 2400 --ordering-rounds 5', '```', '',
        '主要求两点回退及实际连接独立冷验（新输出目录）：', '', '```bash', f"cd '{ROOT}'",
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experimental/welding_retreat_transfer_v1/verify_v3.py --retreat-out build/welding_retreat_200cm_other_park_v12 --transfer-out build/welding_transfer_200cm_other_park_v12 --out build/welding_retreat_transfer_200cm_other_park_cold_fresh --workers 2', '```', '',
        '其余18点2米姿态搜索的实时日志（只读）：', '', '```bash',
        'tail -n 30 /tmp/welding_retreat_200cm_inventory_v8.log', '```', '',
        '最新20点解析IK的2米姿态搜索复现（新的输出目录，完整2米，不会覆盖封存结果）：', '', '```bash', f"cd '{ROOT}'",
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experimental/welding_retreat_transfer_v1/probe_inventory_200cm_v10.py --out build/welding_retreat_200cm_inventory_analytic_fresh --workers 1', '```', '',
        '最新20点解析IK的2米姿态搜索实时日志（只读）：', '', '```bash',
        'tail -n 30 /tmp/welding_retreat_200cm_inventory_analytic_v10.log', '```', '',
        '源解析IK外轴扩展实时日志（只读）：', '', '```bash',
        'tail -n 30 /tmp/welding_retreat_200cm_analytic_positioner_v11.log', '```', '',
        '另一臂停靠重选实时日志（只读）：', '', '```bash',
        'tail -n 30 /tmp/welding_retreat_200cm_other_park_v12.log', '```', '',
        '跨机械臂及外轴扩展搜索实时日志（只读）：', '', '```bash',
        'tail -n 30 /tmp/welding_retreat_200cm_cell_inventory_v20.log', 'tail -n 30 /tmp/welding_retreat_transfer_200cm_positive_angles_v36_pipeline.log', '```', '',
        '增量三节点、最新七节点的冷验与实际连接日志（只读）：', '', '```bash',
        'tail -n 30 /tmp/welding_retreat_transfer_early_cell_v26_pipeline.log', 'tail -n 30 /tmp/welding_transfer_200cm_early_cell_v26.log',
        'tail -n 30 /tmp/welding_retreat_transfer_200cm_coarse_search_v30_pipeline.log', '```', '',
        '更新本Markdown进度：', '', '```bash', f"cd '{ROOT}'",
        '.venv/bin/python experimental/welding_retreat_transfer_v1/write_status.py', '```', '',
        f'主要求结果：`{primary}`、`{refinement}`。诊断退枪：`{diagnostic}`。诊断转移：`{transfer}`。未生成桌面视频。']
    DESKTOP.write_text('\n'.join(lines)+'\n')
    print('RETREAT_TRANSFER_DESKTOP', DESKTOP)


if __name__ == '__main__':
    main()
