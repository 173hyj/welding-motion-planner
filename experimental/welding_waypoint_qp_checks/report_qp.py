"""Evidence-bound Markdown report of collision-waypoint QP experiments."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'welding_variable_open_v1'))
from run_geometry import digest,atomic
from tool_world import ROOT
DESKTOP=ROOT.parent/'上航发焊枪_碰撞waypoint_QP优化_最新.md'


def load_sealed(out):
    report=json.loads((out/'report.json').read_text())
    if not report['summary']['complete']:raise RuntimeError('Experiment incomplete')
    for section in ('source_sha256','previous_code_sha256','code_sha256'):
        for name,value in report['manifest'].get(section,{}).items():
            if digest(ROOT/name)!=value:raise RuntimeError('Frozen source/code changed: '+name)
    for name,value in report['output_sha256'].items():
        if digest(out/name)!=value:raise RuntimeError('Saved output changed: '+name)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--final-out',type=Path,default=ROOT/'build/welding_waypoint_qp_v3');args=parser.parse_args()
    out=args.final_out.resolve();final=load_sealed(out)
    v1=load_sealed(ROOT/'build/welding_waypoint_qp_v1');v2=load_sealed(ROOT/'build/welding_waypoint_qp_v2')
    parent=json.loads((ROOT/'build/welding_variable_open_v1/report.json').read_text());parent_rows={r['weld_id']:r for r in parent['rows']}
    priorcold=json.loads((ROOT/'build/welding_variable_open_v1/cold_verification.json').read_text())
    if priorcold['report_sha256']!=digest(ROOT/'build/welding_variable_open_v1/report.json') or not priorcold['summary']['all_passed']:raise RuntimeError('Previous15 paths not bound to passing cold check')
    newcold=json.loads((out/'cold_verification.json').read_text())
    if newcold['report_sha256']!=digest(out/'report.json') or not newcold['summary']['all_passed']:raise RuntimeError('Final QP cold report missing,failed or digest mismatch')
    for name,value in newcold['verification_code_sha256'].items():
        if digest(ROOT/name)!=value:raise RuntimeError('Cold code changed')
    if newcold['summary']['reverified_retreats']!=final['summary']['additional_retreat_passed']:raise RuntimeError('Selected/new cold count mismatch')
    v2rows={r['weld_id']:r for r in v2['rows']};new=[]
    for row in final['rows']:
        if row['initial']!=parent_rows[row['weld_id']]['initial']:raise RuntimeError('Archived original welding posture changed')
        if row['retreat_passed']:new.append(row['weld_id'])
    total=parent['summary']['any_amplitude_passed']+len(new)
    lines=['# 上航发：碰撞waypoint驱动的QP直线回退优化','',
           f"对上一轮未通过的6个点完成了单次QP、局部多轮QP和消元方向QP三组尝试。最终新增完整通过 **{len(new)}/6**，合并原15个点为 **{total}/21**。",'',
           '所有实验保留原始焊接q0、枪体朝向、机器人、原变位机角度和另一臂停靠。优化变量包含共享的回退直线方向；没有用折线路径绕障碍，没有重选焊接初始逆解。','',
           '| 实验 | 实际QP求解次数 | 新增完整通过/6 | 耗时/s（含模型加载） |','|---|---:|---:|---:|']
    for title,r in [('单次QP对照',v1),('实际受阻点局部多轮QP',v2),('方向消元QP＋整线非线性merit',final)]:
        lines.append(f"| {title} | {r['summary']['actual_qp_solves']} | {r['summary']['additional_retreat_passed']} | {r['summary']['wall_s']:.2f} |")
    lines+=['','## 剩余点的结果','',
            '下表的中间行程是已通过离散整臂检查的前缀，不代表已经生成合格退枪路径。完整通过仍要求至少100 mm、枪体终点净空≥5 mm、精确工具几何验收、源关节及耦合有效、实际q插值碰撞与直线/姿态验收。','',
            '| 焊点 | 第二版最好候选：优化前行程/mm | 第二版最好行程/mm | 最终完整结果 |','|---|---:|---:|---|']
    for row in final['rows']:
        old=v2rows[row['weld_id']];best=max(old['trials'],key=lambda t:t['final']['last_accepted_travel_mm'])
        lines.append(f"| {row['weld_id']} | {best['baseline']['last_accepted_travel_mm']:.3f} | {best['final']['last_accepted_travel_mm']:.3f} | {'完整通过且冷复核通过' if row['retreat_passed'] else '仍未通过'} |")
    lines+=['','## 具体做法','',
            '单次对照：在直线上的失败附近和远端waypoint线性化碰撞代理、末端位姿误差和源构型约束，一次QP更新共享方向，再完整重算原q0出发的直线。没有新增成功，所以进一步改进。','',
            '局部多轮：记录实际失败q、最后通过q及行程；每轮围绕当前受阻处重新线性化，只有位姿足够精确的状态查询mesh梯度。每候选最多12轮，四档幅度30/60/80/185 mm、每档最多两条原几何出口。','',
            '消元方向QP：通过末端位姿Jacobian消去关节增量，剩下2维共享方向变量。沿数学连续IK分支计算全线碰撞/限位违约，基于非线性merit下降接受更新，每个初始化最多20轮。内部数学构型可以超出源限位，仅用于构造优化约束和计算违约；最终路径必须回到原严格限位与完整验收。','',
            '碰撞代理：分离时用FCL原始三角mesh距离；穿透时用最多100个接触中最大穿透深度的负值。后者是局部、不光滑代理，不是完整物体的可靠全局signed distance。有限差分梯度和QP松弛只能引导优化，不能认证通过。源非凸依赖多边形按真实多边形signed distance处理，没有替换为凸包。','',
            '## 验证与限制','',
            f"新增选中路径冷复核：{newcold['summary']['reverified_retreats']}/{newcold['summary']['new_selected_paths']}。只在有新增完整正路径时重建原始夹具并进行源XML FK、0.5 mm开枪和实际关节边中点复核；无正路径时不把空集合检查称为路径安全验证。",'',
            '保留此前DONG–JING和DONG–两个悬挂接口组件装配豁免，枪与环境仍检查。没有验证变位机运动、点间转移或整个系统的严格连续碰撞安全。未通过只表明本轮有限初始化和优化未找到路径，不证明不可行。','',
            '## 复现命令','',
            '已完成的扫描只读验证：','', '```bash',f"cd '{ROOT}'",
            f"OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experimental/welding_waypoint_qp_v3/run_reduced.py --workers 4 --resume",'```','',
            '新增路径冷复核：','', '```bash',f"cd '{ROOT}'",
            f"OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experimental/welding_waypoint_qp_checks/verify_qp.py --out '{out}' --workers 4",'```','',
            f"完整证据：`{out/'report.json'}`；单次与局部多轮对照在`build/welding_waypoint_qp_v1`和`build/welding_waypoint_qp_v2`。未生成桌面视频。",'',
            f"最终报告SHA256：`{digest(out/'report.json')}`。"]
    text='\n'.join(lines)+'\n';DESKTOP.write_text(text);(out/'结果.md').write_text(text)
    atomic(out/'delivery_audit.json',dict(final_report_sha256=digest(out/'report.json'),new_successes=new,total_with_previous=total,
          original_welding_initial_fields_match=True,cold_report_sha256=digest(out/'cold_verification.json')))
    print('QP_DESKTOP',DESKTOP);print('QP_DELIVERY',json.dumps(dict(new=new,total=total)))


if __name__=='__main__':main()
