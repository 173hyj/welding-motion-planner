"""Publish only a completed search with a matching fresh verification snapshot."""
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from world import OUT, sha


def failure_reason(row):
    if not row['relative_gun_candidates']:
        return '全部5°样本与随动工件/夹具干涉，转动变位机不能改变该相对关系'
    trials = row['angle_trials']
    if all(t['blocking_pairs'] for t in trials):
        return '采样变位机姿态均被固定工位干涉拒绝'
    if not any(t['gun_clear_candidates'] for t in trials):
        return '相对几何有候选，但固定工位挡住全部本轮焊枪姿态'
    if not any(t['ik_states'] for t in trials):
        return '存在几何候选，有限多初值搜索未找到合法逆解'
    blockers = Counter(p['asset_name'] for check in row['full_checks'] for p in check['remaining_pairs'])
    if blockers:
        return '逆解存在，完整检查拒绝：'+', '.join(name for name, count in blockers.most_common(3))
    if not row['full_checks']:
        return '逆解存在，但未找到另一机器人无碰撞停放状态'
    return '逆解存在，电极接触或精度检查不合格'


def main():
    report_path = OUT/'report.json'
    report = json.loads(report_path.read_text())
    verification = json.loads((OUT/'selected_independent_verification.json').read_text())
    assert report['phase'] == 'complete' and len(report['rows']) == 29
    assert verification['report_sha256'] == sha(report_path), 'Verification must match the final search snapshot'
    assert verification['all_selected_passed']
    assert verification['selected_count'] == report['summary']['total_passed']
    summary = report['summary']; geometry = json.loads((OUT/'geometry_audit.json').read_text())
    lines = ['# 上航发29焊点：5°相对几何优先静态姿态检查', '',
        f"更新于 {datetime.now(timezone.utc).isoformat(timespec='seconds')}。本轮有限搜索已结束。", '',
        f"**在声明的装配接口组件豁免假设下，机械臂/闭合焊枪静态条件合格：当前变位机不动 {summary['stationary_passed']}/29；变位机改变后新增 {summary['additional_positioner_passed']} 个；合计 {summary['total_passed']}/29。剩余 {29-summary['total_passed']} 个未找到。**", '',
        f"**仅相对几何筛选：{sum(r['relative_gun_candidates'] > 0 for r in report['rows'])}/29有枪体候选，{sum(r['relative_gun_candidates'] == 0 for r in report['rows'])}点在全部5°样本中被随动几何拒绝。该数量不包含可达性、固定工位及电极接触验收。**", '',
        '这些结果是机械臂/闭合焊枪的静态姿态检查。变位机主动、从动悬挂及DONG/JING之间采用下述装配接口分组；自锁机构释放状态、变位机转动过程、进退枪和点间转移没有验证，不能据此宣称整套工艺已无碰撞。', '',
        '**焊枪开合核对：**本轮全部几何筛选、完整检查和独立复核均固定在原闭合状态（开合驱动0mm）。两把枪原始源模型在0mm时电极间隙0mm，在−185mm打开时电极间隙约187.18mm；闭合几何与源网格一致，未用打开状态代替焊接姿态。核对证据见同目录`gun_closure_audit.json`。', '',
        '## 执行顺序', '',
        '1. TCP固定在源焊点，夹紧轴保持源投影法向轴线，两轴方向均尝试；滚转只取0、5、…、355°。',
        '2. 先枚举焊枪与工件、随动夹具的相对几何，拒绝枪体及非指定工件干涉。每点最多72×2轴方向×2机器人=288个候选。',
        '3. 全部29点先在变位机0°进行固定工位筛选、逆解和双方机器人完整检查。通过点保持当前变位机姿态。',
        '4. 仅剩余有相对几何候选的点进入变位机搜索。按±5、±10、…、±175、180°逐步尝试，共覆盖一个完整周期。',
        '5. 每次变位机变化，同步变换工件、焊点、法向及动夹具。相对几何不变，因此缓存复用；固定工位干涉、可达性及完整碰撞重新检查。',
        '6. 有逆解后检查原关节边界、J2/J3耦合、双方机器人、枪体、夹具、其他工件及电极接触；通过即保留该点姿态，不继续强行改变变位机。', '',
        '## 检查标准和装配处理', '',
        'TCP误差<0.1mm；目标整体旋转误差<0.01°；原关节限制和耦合有效；所有机械臂/焊枪非预期碰撞为空。两个原电极与指定工件的预期接触按原三角面证书核验：接触源面位于焊点25mm范围，离散预期接触深度≤2mm。此接触容差不适用于枪体、夹具或其他工件。', '',
        f"外围布局共有{geometry['original_layout_faces']:,}个三角面，沿用原碰撞检查的工作区裁剪后有{geometry['cropped_layout_faces']:,}面。只去除逐三角面三顶点匹配的夹具副本{geometry['duplicate_faces']:,}面（最大允许差0.02mm），实体仍由原动/静夹具保留。其余外部布局{geometry['retained_external_faces']:,}面和悬挂接口{geometry['retained_interface_faces']:,}面均保留供机械臂/焊枪检查。", '',
        '装配接口白名单仅包含：DONG–整个JING，以及DONG–外围布局节点14768“主动侧悬挂”和14808“从动悬挂”两个组件。此处是整组件对的豁免假设，没有逐接触面认证这些接口接触，也没有忽略整个waiwei。工件与上述固定接口、机械臂/焊枪与这些接口仍检查；DONG与白名单之外固定布局的碰撞仍拒绝。该分组依据原部件名称和父子装配结构，尚未证明自锁释放后的变位机运行安全。', '',
        '独立复核重新读取原机器人关节做FK，并从原fixtureL原始网格重建两个夹具FCL对象，以局部源网格+外部刚体变换复核，避免重复施加夹具父变换。重新检查机械臂/焊枪，以及动夹具/所有工件对固定环境的碰撞；装配接口未认证碰撞记录在复核文件中。所有选中姿态均再次满足上述静态条件标准。', '',
        f"角度筛选及逆解搜索耗时{summary['elapsed_s']:.2f}s（不含首次源几何去重、模型加载及独立复核）。每个几何可行姿态使用源/已有/相邻解初值和16个合法Sobol初值。未找到不等于证明无解。", '',
        '## 逐点结果', '',
        '|焊点|结果|机器人|变位机°|滚转°|说明|', '|---|---|---|---:|---:|---|']
    for row in report['rows']:
        name = row['name'].removeprefix('AP31-LZ-OP10L-')
        if row['passed']:
            selected = row['selected']
            robot = 'R1' if selected['robot_id'] == 'F7690943-8A45-40E4-9AD3-F92856F982FA' else 'R2'
            label = '当前姿态条件合格' if row['stationary_passed'] else '改变变位机后条件合格'
            lines.append(f"|{name}|{label}|{robot}|{selected['positioner_deg']}|{selected['roll_deg']}|独立复核满足声明条件|")
        else:
            lines.append(f"|{name}|未找到|—|—|—|{failure_reason(row)}|")
    lines += ['', '## 复核与查看命令', '', '```bash',
        "cd '/home/ubuntu22/桌面/上航发_焊接规划_20261006'",
        '.venv/bin/python experimental/welding_positioner_v2/verify.py',
        '.venv/bin/python experimental/welding_positioner_v2/report.py', '```', '',
        '```bash', "xdg-open '/home/ubuntu22/桌面/上航发焊点_静态姿态核验_最新.md'", '```', '',
        '仅复查当前已保存结果，不播放旧运动轨迹。搜索断点续跑命令（已完成时不会增加新角度）：', '',
        '```bash', "cd '/home/ubuntu22/桌面/上航发_焊接规划_20261006'",
        '.venv/bin/python experimental/welding_positioner_v2/search.py --resume', '```', '',
        f'完整候选、失败碰撞见证和选中关节构型：`{OUT}/report.json`。',
        f'独立复核：`{OUT}/selected_independent_verification.json`。',
        f'源网格去重及装配分组证据：`{OUT}/geometry_audit.json`。', '']
    contents = '\n'.join(lines)
    (OUT/'结果报告.md').write_text(contents)
    latest = ROOT.parent/'上航发焊点_静态姿态核验_最新.md'
    archive = OUT/'上一版桌面报告.md'
    if latest.exists() and not archive.exists():
        shutil.copy2(latest, archive)
    latest.write_text(contents)
    selected = [dict(weld_id=r['weld_id'], name=r['name'], gun_driver_mm=0., gun_phase='closed', **r['selected'])
                for r in report['rows'] if r['passed']]
    (OUT/'selected_postures.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2))
    print('PUBLISHED', latest, summary)


if __name__ == '__main__': main()
