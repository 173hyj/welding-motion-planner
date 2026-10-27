"""Current desktop status with explicit historical dependency loss and fresh evidence."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from rebuild_evidence_v43 import field

ROOT=Path(__file__).resolve().parents[2]
DESKTOP=ROOT.parent/'上航发焊接_指定距离退枪与点间规划_最新.md'


def cold(name,retreat,transfer=None):
    path=ROOT/'build'/name/'cold_verification.json'
    if not path.exists():return None
    report=ROOT/'build'/retreat/'report.json'
    r=json.loads(path.read_text())
    if r['retreat_report_sha256']!=hashlib.sha256(report.read_bytes()).hexdigest():return None
    if transfer:
        report=ROOT/'build'/transfer/'report.json'
        if not report.exists() or r['transfer_report_sha256']!=hashlib.sha256(report.read_bytes()).hexdigest():return None
    return r['summary']


def main():
    report=ROOT/'build/welding_retreat_200cm_revalidated_inventory_v43/report.json'
    summary=field(report,'summary')
    audit=json.loads((report.parent/'dependency_loss_audit.json').read_text())
    c=cold('welding_retreat_200cm_revalidated_inventory_cold_v43','welding_retreat_200cm_revalidated_inventory_v43')
    p=ROOT/'build/welding_retreat_200cm_mixed_wrist_v44/progress.json'
    progress=json.loads(p.read_text()) if p.exists() else {}
    m=ROOT/'build/welding_retreat_200cm_mixed_wrist_v44/report.json'
    if m.exists():progress=field(m,'summary')
    final=cold('welding_retreat_transfer_200cm_revalidated_mixed_cold_v45',
               'welding_retreat_200cm_mixed_wrist_v44','welding_transfer_200cm_revalidated_mixed_v45')
    transfer=ROOT/'build/welding_transfer_200cm_revalidated_mixed_v45/report.json'
    t=field(transfer,'summary') if transfer.exists() else None
    lines=['# 上航发：200厘米退枪与回退点间规划', '',f'更新：{datetime.now().astimezone().isoformat(timespec="seconds")}', '',
       '**主要求：直线退枪200 cm = 2000 mm = 2米；回退点之间真实规划，经典算法决定访问顺序。**', '',
       '原场景29个焊点：此前按闭合焊枪相对几何筛选保留21个。本批保持全部21点身份。', '',
       f'现存完整2米路径候选：**{summary["passed"]}/21**。原v20有限跨臂/外轴搜索结束，报告时间7424.54秒；历史证据链存在下述缺失，不重新认证其负结果或耗时。', '',
       f'新独立源冷复核：**{c["passed_retreats"]}/{c["selected_retreats"]}通过，耗时{c["wall_s"]:.2f}秒**。只验证退枪，还没有新点间连接。' if c else '新独立源冷复核正在运行；尚不能把保留候选称为已重新验收。', '',
       f'固定机器人末轴分支新搜索：已完成{progress.get("finished",progress.get("points",0))}/21，分支兼容{progress.get("branch_compatible",0)}，保留完整路径{progress.get("retained_complete_retreats",progress.get("passed",14))}。R2固定负末轴、R1固定正末轴，非工作臂使用该机器人的同分支停靠；不在运动中wrap。', '',
       '## 经典顺序与真实连接', '',
       '多起点最近邻＋开放路径2-opt；以实际关节、外轴、开度变化和交接代价排序，失败边禁用后重新排序。真实转移先试完整状态直连/分阶段共同停靠，受阻时用RRT-Connect。探索候选随后按关节0.05°、外轴0.25°、开度0.5mm细验，最终新进程冷复核保存状态和中点。', '',
       '保存完整15轴（两臂12关节、变位机、两枪开度）以及双TCP实际FK，不用稀疏端点直线伪造轨迹，不瞬移另一臂。独立检查仍保留装配安装接口边界，是加密离散检查，并非连续安全证明。', '',
       f'最新转移批次：{t["selected_passed_edges"]}/{t["expected_edges"]}条选定连接，完整路线={t["full_route_planned"]}。' if t else '新版转移等待完整新库存和冷复核；尚未输出新完整路线。', '',
       f'最终冷复核：退枪{final["passed_retreats"]}/{final["selected_retreats"]}、连接{final["passed_transfers"]}/{final["selected_transfers"]}；全部21点路线={final["full_route_verified"]}。' if final else '最终新路线冷复核尚未完成。Goal保持active。', '',
       '## 清理事故与恢复边界', '',
       f'本轮清理误删了仍被封存清单引用的历史结果。这是执行错误。两份碰撞检查源码、v2全部21份单点及关键2米路径已按原SHA精确恢复；仍有{len(audit["missing"])}份历史报告/负候选/进度文件缺失，详见新库存dependency_loss_audit.json。旧manifest和seals未改，缺失项目不记通过。', '',
       '新链从现存原source资产、源码和路径字节重新绑定；现存文件必须匹配原sha。新链明确historical_chain_intact=False，重新执行原网格/XML FK冷复核，不复用旧证书或旧转移种子。事故前7条退枪＋4条连接曾冷验通过，但不能宣称旧证书现在完整可复现，也不能称七点接通。', '',
       '磁盘本轮约释放2GB，清理后约5.7GB可用。源场景、现存路径及新证据保留；今后清理先对全部活动与验收manifest做引用检查。桌面只生成Markdown，不生成视频。', '',
       '## 当前命令', '', '只读进度：', '', '```bash',
       f"cd '{ROOT}'", 'tail -n 30 /tmp/welding_retreat_200cm_mixed_wrist_v44.log',
       'tail -n 30 /tmp/welding_retreat_200cm_revalidated_inventory_cold_v43.log',
       'tail -n 30 /tmp/welding_retreat_transfer_200cm_revalidated_mixed_v45_pipeline.log', '```','',
       '重新核验保留14条退枪（全新输出目录）：', '', '```bash',f"cd '{ROOT}'",
       'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experimental/welding_retreat_transfer_v1/verify_v3.py --retreat-out build/welding_retreat_200cm_revalidated_inventory_v43 --out build/welding_retreat_200cm_revalidated_inventory_cold_fresh --workers 1','```','',
       '重新运行固定分支全21点搜索（全新输出目录）：', '', '```bash',f"cd '{ROOT}'",
       'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experimental/welding_retreat_transfer_v1/probe_mixed_wrist_v44.py --retreat-out build/welding_retreat_200cm_revalidated_inventory_v43 --out build/welding_retreat_200cm_mixed_wrist_fresh','```','',
       '更新本文件：', '', '```bash', f"cd '{ROOT}'",
       '.venv/bin/python experimental/welding_retreat_transfer_v1/write_recovery_status_v45.py','```','']
    DESKTOP.write_text('\n'.join(lines))
    print('RECOVERY_STATUS_DESKTOP',DESKTOP)


if __name__=='__main__':
    main()
