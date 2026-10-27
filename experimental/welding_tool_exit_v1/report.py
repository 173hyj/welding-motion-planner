"""Read-only evidence summary, desktop Markdown only; no videos."""
import json
from pathlib import Path
from tool_world import ROOT
from run_initial import OUT
from run_geometry import digest


def main():
    initial_path=OUT/'initial_report.json';initial=json.loads(initial_path.read_text())
    cold=json.loads((OUT/'initial_cold_verification.json').read_text())
    if cold['initial_report_sha256']!=digest(initial_path) or not cold['summary']['all_agree']:
        raise RuntimeError('Current initial report lacks matching passed cold verification')
    if not initial['summary']['complete']:raise RuntimeError('Initial test incomplete')
    rows=initial['rows'];summary=initial['summary'];lines=[
        '# 上航发：已有初始构型直接开枪与直线退枪',
        '',
        f"已完成21个上一轮保存初始构型的直接试验。开枪成功 **{summary['opening_passed']}/21**；继续完成至少100 mm直线退枪成功 **{summary['open_and_retreat_passed']}/21**。",
        '',
        '每点保持v3已选机器人、六关节初值、焊枪滚转角、变位机角度及另一机器人的停靠构型，不重新搜索焊接姿态。退枪沿途以前一步构型为初值求连续逆解。每个初始构型尝试至多两条已经通过枪体几何退出检查、相互至少20°不同的直线方向。',
        '',
        '| 初始解类别 | 数量 | 开枪成功 | 开枪＋至少100 mm退枪成功 |',
        '|---|---:|---:|---:|']
    for zero,label in [(True,'变位机0°'),(False,'沿用原先已选非零变位机角度')]:
        subset=[r for r in rows if (r['initial']['positioner_deg']==0)==zero]
        lines.append(f"| {label} | {len(subset)} | {sum(r['opening_passed'] for r in subset)} | {sum(r['retreat_passed'] for r in subset)} |")
    lines += ['', '失败分类：2个初始姿态开枪过程中碰撞；其余已开枪点中，4个在连续逆解/原关节及耦合限制上失败，3个退枪途中出现完整机器人或焊枪碰撞，1个未找到满足本轮退出长度和净空条件的枪体直线出口。失败只对应当前方向集与100 mm要求，不证明无解。', '',
        '以下是逐点结果。变位机角度是该初始构型原本对应的角度，本轮没有调整它。', '',
        '| 焊点 | 原变位机角度 | 开枪 | 100 mm退枪 | 失败原因 |',
        '|---|---:|---|---|---|']
    for row in rows:
        short=row['weld_id'].replace('OP10L-','').replace('-POINT-',' ')
        reason='—'
        if not row['opening_passed']:reason='开枪碰撞'
        elif not row['retreat_passed']:
            if row['reason']=='no_tool_exit_found':reason='未找到本轮枪体直线出口'
            else:
                types={trial['reason'] for trial in row['exit_trials']}
                reason='途中碰撞' if types=={'full_retreat_blocked'} else '连续IK/原构型限制'
        lines.append(f"| {short} | {row['initial']['positioner_deg']}° | {'通过' if row['opening_passed'] else '失败'} | {'通过' if row['retreat_passed'] else '失败'} | {reason} |")
    lines += ['',
        f"冷启动复核：重新从原始夹具数据构建BVH，使用独立源XML正运动学和原始PMI工具坐标，开枪驱动步长加密到0.5 mm，对保存的实际关节插值增加中点检查。21点开枪分类全部一致，{cold['summary']['reverified_retreats']}条成功退枪路径全部复核通过。",
        '',
        f"本轮初始解试验端到端 {summary['wall_s']:.2f} s（含模型加载 {summary['model_load_s']:.2f} s）；冷复核 {cold['summary']['wall_s']:.2f} s。使用4个隔离FCL进程。",
        '',
        '原始闭合两电极均有源面片证书，未使用“<0.1 mm近接但无接触证书”作为这些21个初始解的放行依据。只允许指定电极与指定工件的局部工艺接触：源接触面半径25 mm、离散深度上限2 mm；枪体及其余碰撞不豁免。',
        '',
        '验收范围：原始三角网格、完整双机器人碰撞及实际关节插值离散检查。开合、最初局部脱离和整臂运动没有严格连续碰撞证明。保留此前静态场景的DONG–整个JING，以及DONG–主动/从动悬挂两个整组件装配豁免；没有逐面认证这些装配接口、枪内部装配接口或变位机旋转运动。成功是上述条件下的仿真验证，不能据此称全系统安全。',
        '',
        '修正前build/welding_tool_initial_exit_v1中的退枪方向对非零变位机角度重复旋转，旧10/21不作为最终统计。代码快照和旧结果保留；本报告采用修正后的v2及其匹配冷复核。',
        '',
        '## 结果文件与命令',
        '',
        f"- 初始解直接试验：`{OUT/'initial_report.json'}`",
        f"- 冷复核：`{OUT/'initial_cold_verification.json'}`",
        f"- 每条成功路径的实际关节与TCP点保存在各`initial_point_XX.json`的`path.frames`，可用于后续MuJoCo回放。没有生成视频。",
        '',
        '读取已完成结果（保护原始报告与计时，不重复跑）：',
        '',
        '```bash',
        f"cd '{ROOT}'",
        '.venv/bin/python experimental/welding_tool_exit_v1/run_safe.py --stage initial --resume',
        '```',
        '',
        '重新冷复核现有轨迹：',
        '',
        '```bash',
        f"cd '{ROOT}'",
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python experimental/welding_tool_exit_v1/verify_initial.py --workers 4',
        '```',
        '',
        '完整重跑到另一个新目录（不覆盖本轮结果）：',
        '',
        '```bash',
        f"cd '{ROOT}'",
        '.venv/bin/python experimental/welding_tool_exit_v1/run_safe.py --stage initial --workers 4 --out build/welding_tool_initial_exit_recheck_20261008',
        '.venv/bin/python experimental/welding_tool_exit_v1/verify_initial.py --workers 4 --out build/welding_tool_initial_exit_recheck_20261008',
        '```',
        '',
        '新入口检查旧v2几何缓存并封存实际源网格与导入依赖。缓存不匹配时拒绝隐式重建历史v2。已完成续跑只读返回，保护原始计时。',
        '']
    geometry=json.loads((ROOT/'build/welding_tool_exit_v1/geometry_report.json').read_text())
    arm=json.loads((ROOT/'build/welding_tool_exit_v1/arm_report.json').read_text())
    lines += ['## 同轮扩大候选的试验', '',
        f"全部第一轮1,192个5°滚转候选已经检查，21个保留焊点都有至少一个枪体开枪＋直线退出方案；{geometry['summary']['counts']['tool_exit_passed']}个角度候选通过，48个开枪受阻，12个闭合接触未认证，4个未找到本轮直线出口。几何阶段耗时{geometry['summary']['total_wall_s']:.2f} s（含加载）。",
        '',
        f"允许重新选择焊接构型/滚转候选，在0°已有可达点或原先逐点找到的变位机角度上尝试整臂实现，得到{arm['summary']['passed_points']}/21条离散验收通过的退枪路径。这与本报告主要回答的固定初始解11/21口径不同；该12条尚未逐条进行上述更密冷复核。",
        '',
        '旧几何/整臂阶段原manifest仅保护部分依赖，不能将其source_unchanged字段解读为全部实际输入的完整封存证明。固定初始解v2另行封存全部实际网格和关键导入依赖，前后hash一致，且正结果已冷复核。',
        '',
        f"完整几何记录：`{ROOT/'build/welding_tool_exit_v1/geometry_report.json'}`。扩大候选整臂记录：`{ROOT/'build/welding_tool_exit_v1/arm_report.json'}`。",
        '']
    content='\n'.join(lines)
    dest=ROOT.parent/'上航发焊枪_初始解直接退枪_最新.md'
    dest.write_text(content,encoding='utf-8')
    (OUT/'结果与命令.md').write_text(content,encoding='utf-8')
    print(dest)


if __name__=='__main__':main()
