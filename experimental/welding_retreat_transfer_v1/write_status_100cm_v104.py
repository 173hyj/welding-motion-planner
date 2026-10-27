"""Write current 1000mm evidence and reproducible commands to desktop Markdown."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DESKTOP = Path('/home/ubuntu22/桌面')
RETREAT = 'build/welding_retreat_100cm_diverse_v102'
TRANSFER = 'build/welding_transfer_100cm_full_v103'
COLD = 'build/welding_retreat_transfer_100cm_full_v103_cold/cold_verification.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    retreat = json.loads((ROOT / RETREAT / 'report.json').read_text())
    transfer = json.loads((ROOT / TRANSFER / 'report.json').read_text())
    own = json.loads((ROOT / (RETREAT + '_cold/cold_verification.json')).read_text())
    cold_path = ROOT / COLD
    cold = json.loads(cold_path.read_text()) if cold_path.exists() else None
    verified = False
    if cold:
        if (cold['retreat_report_sha256'] != sha(ROOT / RETREAT / 'report.json') or
                cold['transfer_report_sha256'] != sha(ROOT / TRANSFER / 'report.json')):
            raise ValueError('Final cold report binding changed')
        verified = bool(cold['summary']['full_route_verified'] and
            cold['summary']['passed_retreats'] == 21 and cold['summary']['passed_transfers'] == 20)
    state = ('**全部 21 个几何合格焊点的 1 米退枪、20 段实际转移和最终独立冷复核全部通过。**'
             if verified else '**21 条退枪冷复核通过，21点/20段转移已规划通过，最终整体验证仍在运行。**')
    cleanup = json.loads((ROOT / 'build/welding_disk_cleanup_1m_v97/summary.json').read_text())
    volume = os.statvfs(ROOT); free = volume.f_bavail * volume.f_frsize
    lines = ['# 上航发焊接：1 米退枪与点间规划', '', state, '',
        '距离按最新要求固定为 **1000 mm**；角度搜索保持 **5°**。直线退枪保持末端姿态，'
        '回退点之间用 RRT-Connect，访问顺序用多起点最近邻 + 开路径 2-opt。', '',
        '| 当前批次 | 结果 | 耗时 |', '|---|---|---|',
        f"| 补齐退枪候选 v102 | 21/21 | {retreat['summary']['wall_s']:.2f} s |",
        f"| 退枪独立冷复核 | {own['summary']['passed_retreats']}/21 | {own['summary']['wall_s']:.2f} s |",
        f"| 完整转移 v103 | 21点、20/20段 | {transfer['summary']['wall_s']:.2f} s |",
        (f"| 最终整体验证 | {cold['summary']['passed_retreats']}/21退枪、{cold['summary']['passed_transfers']}/20转移 | {cold['summary']['wall_s']:.2f} s |"
         if cold else '| 最终整体验证 | 正在运行 | 待完成 |'), '',
        '耗时是增量计算：保留已有20条1米路径，并从端点完全相同、已独立冷复核的1米路线继承19段转移，'
        '新增1段实际RRT连接；不是场景从零重建耗时。不复用2米转移。', '',
        '新增 GW6：R2、变位机45°、绕焊点轴10°、开枪幅度120 mm，1米终点净空655.20 mm。'
        '新路线只切换一次工作机械臂。', '',
        '## 查看结果和刷新此文件', '', '```bash', f"cd '{ROOT}'",
        '.venv/bin/python experimental/welding_retreat_transfer_v1/write_status_100cm_v104.py',
        f"xdg-open '{DESKTOP / '上航发焊接_1米退枪与点间规划_最新.md'}'", '```', '',
        '原始报告：', '', f'- 退枪：`{RETREAT}/report.json`',
        f'- 转移：`{TRANSFER}/report.json`', f'- 最终冷复核：`{COLD}`', '',
        '## 再次独立验证完整路线', '', '下面用时间戳创建新目录，不覆盖旧证据。', '',
        '```bash', f"cd '{ROOT}'", 'RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"',
        '.venv/bin/python experimental/welding_retreat_transfer_v1/verify_v3.py \\',
        f'  --retreat-out {RETREAT} \\', f'  --transfer-out {TRANSFER} \\',
        '  --out "build/welding_100cm_verify_${RUN_TAG}" --workers 3', '```', '',
        '## 重算实际转移和访问顺序', '',
        '保留已验证的1米退枪端点；下面重新计算转移，不传入旧转移种子。', '', '```bash',
        f"cd '{ROOT}'", 'RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"',
        '.venv/bin/python experimental/welding_retreat_transfer_v1/run_transfer_100cm_v99.py \\',
        f'  --retreat-out {RETREAT} \\',
        f'  --retreat-cold {RETREAT}_cold/cold_verification.json \\',
        '  --out "build/welding_100cm_transfer_${RUN_TAG}" \\',
        '  --workers 3 --iterations 1500 --ordering-rounds 5', '',
        '# 完成新规划后，复核刚生成的实际路线',
        '.venv/bin/python experimental/welding_retreat_transfer_v1/verify_v3.py \\',
        f'  --retreat-out {RETREAT} \\',
        '  --transfer-out "build/welding_100cm_transfer_${RUN_TAG}" \\',
        '  --out "build/welding_100cm_replanned_verify_${RUN_TAG}" --workers 3', '```', '',
        '## 焊点姿态与实际1米终点', '',
        '| 焊点 | 机器人 | 变位机角度 | 绕轴角度 | 开枪幅度 | 终点净空 |', '|---|---|---|---|---|---|']
    for row in retreat['rows']:
        path = json.loads((ROOT / RETREAT / row['best']['path_file']).read_text())
        robot = 'R2' if path['robot_id'].startswith('F461') else 'R1'
        wid = row['weld_id'].removeprefix('OP10L-POINT-').removeprefix('OP10L-')
        lines.append(f"| {wid} | {robot} | {path['positioner_deg']:g}° | {row['initial']['roll_deg']:g}° | "
            f"{abs(path['driver_mm']):g} mm | {path['endpoint_clearance_mm']:.2f} mm |")
    lines += ['', '实际访问顺序：', '', ' → '.join(w.removeprefix('OP10L-') for w in transfer['order_weld_ids']), '',
        '## 空间清理和证据范围', '',
        f"清理操作累计实测释放 {cleanup['cleanup_operation_measured_gain_bytes']:,} 字节（约936 MiB）；"
        f"当前可用 {free / 1024**3:.2f} GiB。系统同期写入会使当前净增量小于清理量。"
        '清理对象为可重建Python字节码、过期macOS安装包/下载残片和桌面搜索索引，'
        '项目源代码、场景和封存实验结果保留。详情：`build/welding_disk_cleanup_1m_v97/summary.json`。', '',
        '冷复核使用原始夹具网格、独立XML正运动学和轨迹中点；转移包含两台机械臂、变位机和两把焊枪的真实15轴状态。'
        '这是离散采样验证，不是连续碰撞或动力学安全证明。', '',
        '21点来自原29点中已几何筛选合格的集合。历史2米结果保留；历史早期49个JSON缺失仍如实记录。'
        'v98/v100与首个串行预探测批次中断，不当作失败结论。105项相关测试通过。', '',
        '本次没有生成视频或TXT；尚未导出支持此非零变位机路线的MuJoCo完整回放，旧回放不代表本次结果。', '']
    goal_path = ROOT / 'build/welding_100cm_delivery_v104/goal_record.json'
    if verified and goal_path.exists():
        goal = json.loads(goal_path.read_text())['goal_record']
        if goal['status'] == 'complete':
            seconds = goal['timeUsedSeconds']
            lines += [f'Goal已按最新1000mm要求完成，累计工具计时约{int(seconds//3600)}小时{int(seconds%3600//60)}分钟，'
                '并非本轮增量耗时。工具保留不可编辑的旧200cm目标标题，以用户最新要求和本次1000mm封存结果为准。', '']
    target = DESKTOP / '上航发焊接_1米退枪与点间规划_最新.md'
    temporary = target.with_suffix('.tmp'); temporary.write_text('\n'.join(lines)); temporary.replace(target)
    print('FULL_100CM_VERIFIED' if verified else 'FINAL_COLD_RUNNING', target)


if __name__ == '__main__': main()
