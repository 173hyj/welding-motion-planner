"""Deliver the staged-exit result without mixing it with all-straight results."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'welding_variable_open_v1'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'welding_tool_exit_v1'))
from run_geometry import digest,atomic
from tool_world import ROOT
DESKTOP=ROOT.parent/'上航发焊枪_QP直线回退_RRT转移_最新.md'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'build/welding_waypoint_qp_v6');args=parser.parse_args()
    out=args.out.resolve();rp=out/'report.json';report=json.loads(rp.read_text());cold=json.loads((out/'cold_verification.json').read_text())
    if not report['summary']['complete'] or cold['report_sha256']!=digest(rp) or not cold['summary']['all_passed']:raise RuntimeError('Complete digest-bound passing cold result required')
    if report['summary']['additional_staged_passed']!=cold['summary']['reverified_staged_paths']:raise RuntimeError('Cold path count mismatch')
    for section in ('source_sha256','previous_code_sha256','code_sha256'):
        for n,v in report['manifest'][section].items():
            if digest(ROOT/n)!=v:raise RuntimeError('Source/code changed: '+n)
    for n,v in cold['verification_code_sha256'].items():
        if digest(ROOT/n)!=v:raise RuntimeError('Cold code changed')
    for n,v in report['output_sha256'].items():
        if digest(out/n)!=v:raise RuntimeError('Output changed')
    original=json.loads((ROOT/'build/welding_variable_open_v1/report.json').read_text());old={r['weld_id']:r for r in original['rows']}
    verified={r['weld_id']:r for r in cold['verified_rows']};new=[];lines=['# 上航发：QP直线退枪＋RRT转移','',
       f"剩余6个点中，**{report['summary']['additional_staged_passed']}/6** 找到分阶段退出路径并通过加密冷复核。原来的15个点仍保留完整至少100 mm直线退枪方案；新6个点采用短直线释放后RRT转移。因此当前21个几何筛选合格点都有退出方案，不能称21/21全部100 mm直线退枪。",'',
       '## 流程','',
       '1. 固定原焊接构型、枪体初始朝向、机器人、变位机和另一臂停靠，按合适幅度开枪。','2. 用多轮waypoint QP调整共享回退直线方向，提高可退行程。','3. 根据完整整臂首次碰撞或源构型失效位置，取其前3 mm；结合枪体粗检查障碍前5 mm，重新验证直线前缀。','4. 前缀终点已经与工件分离，再用双向关节空间RRT连接到有效转移构型。RRT阶段允许末端位置和朝向随避障改变；每个样本仍检查源限位、耦合和完整双臂碰撞。','5. 终点TCP离原焊点至少100 mm，枪体与环境净空至少5 mm。保存的轨迹线来自真实关节样本的FK，不是直线连点。','',
       '| 焊点 | 开枪驱动幅度/mm | 直线前缀/mm | 转移终点离焊点/mm | 终点枪体净空/mm | 冷复核 |','|---|---:|---:|---:|---:|---|']
    for row in report['rows']:
        if row['initial']!=old[row['weld_id']]['initial']:raise RuntimeError('Archived initial changed')
        if not row['best']:continue
        best=row['best'];path=json.loads((out/best['path_file']).read_text());check=verified[row['weld_id']]
        if check['planned_path_sha256']!=best['path_sha256'] or not check['passed']:raise RuntimeError('Cold path identity mismatch')
        new.append(row['weld_id'])
        lines.append(f"| {row['weld_id']} | {best['amplitude_mm']:g} | {path['straight_prefix_mm']:.2f} | {check['endpoint_displacement_mm']:.2f} | {check['endpoint_tool_clearance_mm']:.2f} | 通过 |")
    v2=json.loads((ROOT/'build/welding_waypoint_qp_v2/report.json').read_text());v5=json.loads((ROOT/'build/welding_waypoint_qp_v5/report.json').read_text())
    replanning_step_deg=report['manifest']['settings']['rrt_edge_step_deg']
    lines+=['','## 耗时','',
        f"多轮QP改善6点：{v2['summary']['wall_s']:.2f} s（193次局部QP，仍没有完整100 mm直线成功）。首次双向RRT分阶段扫描：{v5['summary']['wall_s']:.2f} s，6条原离散检查通过，其中1条未通过冷复核。针对这1条加密重规划：{report['summary']['wall_s']:.2f} s，其他5条使用已冷复核原路径；最终6条再次冷复核：{cold['summary']['wall_s']:.2f} s。各次包含重复模型加载，不能当成集成一次运行的最佳耗时。",'',
        '## 验证范围','',
        f'首版RRT的0.25°采样漏掉了SUB2 GW-3边中点的电极碰撞，已拒绝该路径并重新规划。此次交付中GW-3重规划使用{replanning_step_deg:g}°关节间隔；其余5条保留原0.25°离散规划并已冷复核通过的路径。最终全部6条再用新进程重建原始夹具BVH、源XML独立FK、0.5 mm开枪和保存q边中点复核，通过后交付。','',
        '源关节与耦合限制没有放宽。RRT内部伸展可产生被拒候选，只有完整检查过的边进入保存路径。局部直线前缀终点允许接近障碍，保留至少0.1 mm工具净空；RRT最终目标仍要求5 mm。此结果属于加密离散验收，不是严格全程连续系统安全证明。','',
        '保留既有装配接口豁免：DONG–整个JING，DONG–两整个悬挂接口组件；枪与这些环境仍检查。没有验证变位机旋转路径、焊点之间完整转移、焊枪内部装配接口或源初始布局到当前起点的运动。','',
        '第三版消元QP未形成有效方向更新：18/18 OSQP达到30000内部迭代上限。它是数值求解失败，不作为几何不可行的证据；最终分阶段方案使用第二版已改善的方向。','',
        '## 命令','', '```bash',f"cd '{ROOT}'",f"OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experimental/{out.name}/run_staged.py --workers 1 --resume",'```','',
        '加密冷复核：','', '```bash',f"cd '{ROOT}'",f"OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experimental/welding_waypoint_qp_checks/verify_staged.py --out '{out}' --workers 4",'```','',
        f"完整规划与冷复核：`{out/'report.json'}`、`{out/'cold_verification.json'}`。真实q/TCP轨迹：`{out/'paths'}`；未生成桌面视频。",'',
        f"报告SHA256：`{digest(rp)}`。",f"冷复核SHA256：`{digest(out/'cold_verification.json')}`。"]
    text='\n'.join(lines)+'\n';DESKTOP.write_text(text);(out/'结果.md').write_text(text)
    atomic(out/'delivery_audit.json',dict(report_sha256=digest(rp),cold_sha256=digest(out/'cold_verification.json'),
          initial_fields_locked=True,new_staged_successes=new,all_straight_100mm_successes=15,staged_additional_successes=len(new)))
    print('STAGED_DESKTOP',DESKTOP)


if __name__=='__main__':main()
