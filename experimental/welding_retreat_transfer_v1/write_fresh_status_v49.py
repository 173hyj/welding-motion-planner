"""Desktop Markdown for fresh 2m retreats and complete-state transfer batches."""
import json
from datetime import datetime
from pathlib import Path
from rebuild_evidence_v43 import field, digest

ROOT=Path(__file__).resolve().parents[2]
DESKTOP=ROOT.parent/'上航发焊接_指定距离退枪与点间规划_最新.md'


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def bound_cold(directory,retreat,transfer=None):
    p=ROOT/'build'/directory/'cold_verification.json';c=read(p)
    if not c:return None
    r=ROOT/'build'/retreat/'report.json'
    if not r.exists() or c['retreat_report_sha256']!=digest(r):raise ValueError('Cold/retreat binding changed')
    if transfer:
        t=ROOT/'build'/transfer/'report.json'
        if not t.exists() or c['transfer_report_sha256']!=digest(t):raise ValueError('Cold/transfer binding changed')
    return c['summary']


def main():
    base='welding_retreat_200cm_revalidated_inventory_v43'
    c=bound_cold(base.replace('inventory_v43','inventory_cold_v43'),base)
    snapshot='welding_retreat_200cm_retention_snapshot_v59'
    s=field(ROOT/'build'/snapshot/'report.json','summary')
    sc=bound_cold('welding_retreat_200cm_retention_expanded_v60_cold',snapshot)
    transfer='welding_transfer_200cm_retention_expanded_v60'
    tp=ROOT/'build'/transfer/'report.json'
    t=field(tp,'summary') if tp.exists() else read(tp.parent/'progress.json')
    order=field(tp,'order_weld_ids') if tp.exists() else []
    fc=bound_cold('welding_retreat_transfer_200cm_retention_expanded_v60_cold',snapshot,transfer)
    warm='welding_retreat_200cm_same_pose_v55'
    wp=ROOT/'build'/warm/'report.json'
    ws=field(wp,'summary') if wp.exists() else None
    wc=bound_cold('welding_retreat_200cm_same_pose_expanded_v56_cold',warm) if ws else None
    wt=ROOT/'build/welding_transfer_200cm_same_pose_expanded_v56/report.json'
    wts=field(wt,'summary') if wt.exists() else None
    wfc=bound_cold('welding_retreat_transfer_200cm_same_pose_expanded_v56_cold',warm,'welding_transfer_200cm_same_pose_expanded_v56') if ws else None
    live=ROOT/'build/welding_retreat_200cm_mixed_wrist_v44'
    ls=field(live/'report.json','summary') if (live/'report.json').exists() else read(live/'progress.json')
    ls=ls or {}
    lines=['# 上航发：200厘米退枪与回退点间规划','',
        f'更新：{datetime.now().astimezone().isoformat(timespec="seconds")}','',
        '**要求按字面执行：200 cm = 2000 mm = 2米直线退枪；无QP；退枪点之间规划真实运动；经典算法决定访问顺序。**','',
        '原场景29个焊点，经闭合焊枪相对几何筛选保留21个，本任务始终保留这21个身份。','',
        f'现存完整2米退枪候选14/21。新独立原网格/XML FK复核：{c["passed_retreats"]}/{c["selected_retreats"]}通过，{c["wall_s"]:.2f}秒。' if c else '14条保留路径尚待新独立冷复核。','',
        f'固定分支全21点搜索：完成{ls.get("finished",ls.get("points",0))}/21，分支兼容{ls.get("branch_compatible",0)}；完整2米路径保留{ls.get("retained_complete_retreats",ls.get("passed",14))}。R2固定负末轴，R1固定正末轴，非工作臂同分支停靠。','',
        f'已完成兼容正例快照选入{s["passed"]}条；原有14条路径另外保留，21行均保留。快照完成不代表全点搜索完成。', '',
        f'快照新冷复核：{sc["passed_retreats"]}/{sc["selected_retreats"]}，{sc["wall_s"]:.2f}秒。' if sc else '快照新冷复核尚未完成。','',
        f'原TCP姿态解析分支重选：相容候选{ws["passed"]}/21，{ws["wall_s"]:.2f}秒，新增SUB3 GW10；GW9在1938mm碰撞，未算2m成功。' if ws else '原TCP姿态分支重选尚在运行。','',
        f'新8候选冷复核：{wc["passed_retreats"]}/{wc["selected_retreats"]}，{wc["wall_s"]:.2f}秒。' if wc else '新增第8条退枪需新冷复核；不能把运动学候选当作验收。','',
        f'新8节点转移：{wts["selected_passed_edges"]}/{wts["expected_edges"]}；只复用刚通过v53新冷验证、完整15轴端点不变的边，最后仍全部新cold。' if wts else '8候选批次在运行新cold→当前新证书精确端点seed→点间规划→最终cold。','',
        f'新8节点最终冷复核：退枪{wfc["passed_retreats"]}/{wfc["selected_retreats"]}，连接{wfc["passed_transfers"]}/{wfc["selected_transfers"]}，子集={wfc["retained_subset_route_verified"]}，全21={wfc["full_route_verified"]}，{wfc["wall_s"]:.2f}秒。' if wfc else '新8节点完整路线尚未冷验完成；下述8节点路线仍在最终认证中。','',
        '## 点间规划状态','',
        f'当前转移状态：{json.dumps(t,ensure_ascii=False)}' if t else f'新{s["passed"]}节点转移等待快照冷复核，下面8节点路线已独立认证。','',
        f'最终独立冷复核：退枪{fc["passed_retreats"]}/{fc["selected_retreats"]}，转移{fc["passed_transfers"]}/{fc["selected_transfers"]}，子集路线={fc["retained_subset_route_verified"]}，全部21点路线={fc["full_route_verified"]}，{fc["wall_s"]:.2f}秒。' if fc else f'新增{s["passed"]}节点路线尚未最终冷验；8节点已有完整认证，全部21点仍未完成。','',
        '最新批次访问顺序：'+' → '.join(order) if order else ('已认证8点顺序：'+' → '.join(field(wt,'order_weld_ids')) if wfc else '最新访问顺序尚未落盘。'),'',
        '顺序采用多起点最近邻＋开放路径2-opt，代价包括实际关节/变位机/开度变化和机械臂交接。失败连接禁用后重新排序；这是经典启发式，不宣称全局最优。','',
        '转移先试完整状态直连和共同停靠阶段链，受阻时采用RRT-Connect。新入口从当前15轴状态和分支停靠生成候选，每个真实运动阶段都检查；候选本身不当作静态或运动认证。','',
        '所有通过候选仍按关节0.05°、外轴0.25°、枪开度0.5mm加密检查，保存完整15轴及双TCP实际FK；最终独立原网格/XML FK检查保存帧和中点。不会用端点直线假冒轨迹，不在中途wrap，不瞬移另一臂或变位机。','',
        '有限停靠诊断发现GW8/GW6往负第一轴停靠会撞外围或另一臂；扩展正向第一轴候选后，同一2→5边原fine验收通过，41.43秒。旧前两候选搜索累计388.54秒未找到；不据此宣称通用加速倍率。新整批与独立冷复核仍是最终依据。','',
        '完整批次完成后，新v60合并流水线重新封存相容库存→冷复核→真实转移→最终冷复核，任何非零退出即停止该流水线并保留失败记录。Goal保持active；有限失败不等于证明无解。','',
        '## 证据与清理边界','',
        '先前清理误删了仍被引用的历史结果，这是执行错误。两份碰撞源码及v2的21份单点、关键路径已按原SHA精确恢复；49份历史JSON仍缺失。新v43的dependency_loss_audit.json逐项列出，historical_chain_intact=False，旧seals没有改写。','',
        '14条原路径字节保留，并通过新原网格复核；不复用断链旧冷验证证书或旧转移种子。审查还发现旧共享停靠probe依赖一个缺失JSON，因此旧v45等待者已被替换。新v52入口不读取该旧probe，使用当前源模型逐阶段检查。','',
        '碰撞验收保留原装配安装接口豁免及枪内部接口边界；这是加密离散仿真验证，不等同于连续安全证明或实机运行授权。此前清理净释放约2GB，原源资产、路径和当前证据保留。桌面只写Markdown，不生成视频。','',
        '## 命令','',
        '查看进度：','', '```bash',f"cd '{ROOT}'",
        'tail -n 20 /tmp/welding_retreat_200cm_mixed_wrist_v44.log',
        'tail -n 20 /tmp/welding_fresh_retention_v60_pipeline.log',
        'tail -n 20 /tmp/welding_transfer_200cm_retention_expanded_v60.log',
        'tail -n 20 /tmp/welding_fresh_merged_v60_pipeline.log',
        'tail -n 20 /tmp/welding_fresh_same_pose_v56_pipeline.log','```','',
        '重做已封存兼容快照的完整流水线（使用新的输出tag，勿重复启动当前批次）：','',
        '```bash',f"cd '{ROOT}'",
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u experimental/welding_retreat_transfer_v1/finish_merged_transfer_v60.py --retreat-out build/welding_retreat_200cm_retention_snapshot_v59 --tag retention_reproduction_v60',
        '```','',
        '更新本Markdown：','', '```bash',f"cd '{ROOT}'",
        '.venv/bin/python experimental/welding_retreat_transfer_v1/write_fresh_status_v49.py','```','',
        '当前先交付可审计路径与规划结果。新的MuJoCo回放入口须绑定新路线最终冷复核后更新，旧断链回放不作为当前验收命令。','']
    DESKTOP.write_text('\n'.join(lines));print('FRESH_DESKTOP_MD',DESKTOP)


if __name__=='__main__':main()
