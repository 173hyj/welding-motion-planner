"""Publish a completed and freshly verified static-posture result as Markdown."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PYTHON='/home/ubuntu22/桌面/activenet_4070_offload_20260717/rl_sparse_qp_planner/.venv/bin/python'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--folder',type=Path,required=True)
    args=parser.parse_args();out=args.folder.resolve()
    report_path=out/'report.json';report=json.loads(report_path.read_text())
    verified=json.loads((out/'selected_independent_verification.json').read_text())
    assert len(report['rows'])==29
    assert verified['report_sha256']==hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert verified['all_selected_static_states_passed']
    selected=[r for r in report['rows'] if r['passed']]
    assert len(selected)==verified['selected_count']
    old_manifest=json.loads((ROOT/'build/welding_endpoints_fixed_normal_v2/input_manifest.json').read_text())
    # The verification CLI gains a folder argument. All scene meshes, source
    # kinematics and collision rules must remain byte-identical to the old run.
    unchanged={name:sha for name,sha in old_manifest.items() if not name.startswith('experimental/')}
    for name,expected in unchanged.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    for path in (ROOT/'experimental/welding_endpoints_v1').glob('*.py'):
        unchanged[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'input_manifest.json').write_text(json.dumps(unchanged,ensure_ascii=False,indent=2))
    categories=collections.Counter();details=[]
    for row in report['rows']:
        if row['passed']:category='通过'
        elif row['gun_clear_placements']==0:category='本轮绕轴角度采样均有枪体干涉'
        elif row['ik_states_checked']==0:category='枪体可避开，尚未找到合法机械臂逆解'
        else:category='存在逆解，完整碰撞/接触检查仍拒绝'
        categories[category]+=1;details.append((row,category))
    ik_stats=[x['ik_stats'] for r in report['rows'] for x in r['gun_screen'] if 'ik_stats' in x]
    summary=dict(report['summary'],failure_categories=dict(categories),
        gun_placements_checked=sum(r['gun_placements_checked'] for r in report['rows']),
        ik_seed_optimizations=sum(x['starts'] for x in ik_stats),
        full_simultaneous_state_checks=sum(r['ik_states_checked'] for r in report['rows']),
        previous_passed_welds=7,new_passed_welds=len(selected)-7,
        fresh_verification_passed=True,scene_and_collision_inputs_unchanged=True)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    postures=[dict(weld_id=r['weld_id'],normal_world=r['normal_world'],**r['selected']) for r in selected]
    (out/'selected_postures.json').write_text(json.dumps(dict(scene_sha256=report['scene_sha256'],
        scope='static welding postures only',postures=postures),ensure_ascii=False,indent=2))
    lines=['# 上航发29焊点：固定法向、绕轴角度搜索','',
        f"**本轮完成：{len(selected)}/29 点找到并复核通过静态焊接姿态，{29-len(selected)}/29 点尚未通过。**",'',
        '仅处理焊接瞬间姿态。进枪、退枪、转移及RRT规划暂停。','',
        '保留机器人原安装位置，使用修正夹具父布局的源场景和源三角网格。焊点位置固定，TCP Z轴即电极夹紧轴固定在焊接面法向上；只改变绕轴角度。',
        '分别尝试轴线的正、反两种焊枪朝向，每种朝向只进行滚转，不允许倾斜轴线。原投影法向已与真实工件三角面法向交叉检查。','',
        '本轮对未通过点细化到0.25°；用源候选与邻近构型、64个满足原关节限制的Sobol初值扩大逆解分支搜索。用原螺旋轴及关节耦合的解析雅可比加速求解，已与独立FK有限差分交叉验证。',
        f"检查枪体放置 {summary['gun_placements_checked']} 次；逆解初值优化 {summary['ik_seed_optimizations']} 次；完整双方机器人状态复核 {summary['full_simultaneous_state_checks']} 次。",'',
        '判定条件：原关节边界/J2-J3耦合限制满足，TCP误差<0.1mm、法向误差<0.01°；机械臂、枪体、夹具、非目标工件及另一机器人不得有非预期碰撞。',
        '电极与指定工件的预期接触独立核验：原电极接触三角面位于焊点25mm范围内，离散接触深度≤2mm。该条件不豁免枪体和夹具碰撞。设计装配接口不当作自由物体间碰撞；不认证焊接电流或热过程。',
        '另一机器人用另行检查的静态停放状态；未验证原初态到停放点的运动。所选姿态由新建FCL场景及独立原始源关节FK再次检查。','',
        '**未找到不等于证明无解。** 角度为有限采样，逆解为多初值数值求解；不声称连续角度或全部逆解被穷尽。','',
        '|焊点|结果/原因|执行机器人|绕轴角度°|','|---|---|---|---:|']
    for row,category in details:
        selection=row['selected'];tag=angle='—'
        if selection:
            tag='R1' if selection['robot_id'].startswith('F769') else 'R2'
            angle=selection['roll_deg']
        lines.append(f"|{row['weld_id']}|{category}|{tag}|{angle}|")
    lines+=['','## 静态复核命令','', '```bash',f"cd '{ROOT}'",
        f"'{PYTHON}' experimental/welding_endpoints_v1/verify_selected.py --folder '{out}'",'```','',
        f'结果目录：`{out}`。',
        '`report.json`保留全部采样和拒绝证据；`selected_postures.json`保存通过构型；`selected_independent_verification.json`保存二次核验；`input_manifest.json`保存输入哈希。']
    content='\n'.join(lines)+'\n'
    (out/'结果报告.md').write_text(content)
    (ROOT.parent/'上航发焊点_静态姿态核验_最新.md').write_text(content)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
