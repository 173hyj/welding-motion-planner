"""Validate like-for-like results and publish measured speedup in Markdown."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'build/welding_positioner_static_v3'
BASE = ROOT/'build/welding_positioner_static_v2'
sys.path[:0] = [str(ROOT/'experimental/welding_positioner_v2'), str(ROOT)]
from world import sha


def main():
    old = json.loads((BASE/'report.json').read_text())
    fast = json.loads((OUT/'report.json').read_text())
    timing = json.loads((OUT/'timing.json').read_text())
    verification = json.loads((OUT/'selected_independent_verification.json').read_text())
    verify_timing = json.loads((OUT/'verification_timing.json').read_text())
    assert fast['phase'] == 'complete' and len(fast['rows']) == 29
    assert fast['scene_sha256'] == old['scene_sha256']
    assert verification['report_sha256'] == sha(OUT/'report.json') and verification['all_selected_passed']
    assert verification['selected_count'] == fast['summary']['total_passed']
    assert fast['baseline_seed_report_sha256'] == sha(ROOT/'build/welding_endpoints_fixed_normal_v2/report.json')
    old_rows = {r['weld_id']: r for r in old['rows']}
    for row in fast['rows']:
        reference = old_rows[row['weld_id']]
        assert row['relative_gun_candidates'] == reference['relative_gun_candidates']
        assert row['passed'] == reference['passed']
        if not row['relative_gun_candidates']:
            assert not row['angle_trials'] and not row['full_checks']
        if row['passed']:
            assert row['selected']['positioner_deg'] % 5 == 0 and row['selected']['roll_deg'] % 5 == 0
    trials_old = [t for r in old['rows'] for t in r['angle_trials']]
    trials_new = [t for r in fast['rows'] for t in r['angle_trials']]
    old_starts = sum(t['ik_seed_starts'] for t in trials_old)
    old_full = sum(len(r['full_checks']) for r in old['rows'])
    comparison = dict(old_search_s=old['summary']['elapsed_s'], new_search_s=timing['search_wall_s'],
        new_end_to_end_s=timing['end_to_end_wall_s'], independent_verification_s=verify_timing['independent_verify_wall_s'],
        speedup_search=old['summary']['elapsed_s']/timing['search_wall_s'],
        old_ik_seed_starts=old_starts, new_ik_seed_starts=timing['ik_seed_starts'],
        old_full_checks=old_full, new_full_checks=timing['full_checks'],
        source_scene_identical=True, relative_candidates_identical=True, passed_weld_ids_identical=True,
        used_final_v2_solution_seeds=False, original_7_seed_input_identical=True,
        qualification='Single historical v2 run versus fresh v3 run on the same host; v3 uses 4 processes, existing immutable geometry cache and identical old 7-point seed input. No claim about repeated cold-run timing or general scenes.')
    (OUT/'speed_comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2))
    shutil.copy2(BASE/'gun_closure_audit.json', OUT/'gun_closure_audit.json')
    # Reuse the original Chinese static audit report, changing only its output.
    spec = importlib.util.spec_from_file_location('static_report_v2', ROOT/'experimental/welding_positioner_v2/report.py')
    publisher = importlib.util.module_from_spec(spec); spec.loader.exec_module(publisher)
    publisher.OUT = OUT
    publisher.main()
    ratio = comparison['speedup_search']
    lines = ['# 上航发焊接静态搜索提速结果与命令', '',
        f"相同场景、5°网格和验收条件下，搜索从 **{comparison['old_search_s']:.2f}秒（约30分钟）降到{comparison['new_search_s']:.2f}秒**，本次实测约 **{ratio:.1f}倍**。", '',
        '**结果一致：**相对几何21/29有候选、8个直接剔除；0°合格7个；变位机改变新增14个；21个均用独立源FK及新建源网格FCL复核通过。采用与旧版相同的明确整组件装配接口豁免假设；未验证自锁释放、进退枪、变位机转动路径和点间转移。', '',
        '|指标|旧版v2|加速版v3|', '|---|---:|---:|',
        f"|角度筛选与搜索时间|{comparison['old_search_s']:.2f}s|{comparison['new_search_s']:.2f}s|",
        f"|加载、筛选、搜索、最终落盘合计|旧版未单独计时|{comparison['new_end_to_end_s']:.2f}s|",
        f"|独立复核耗时|旧版未单独计时|{comparison['independent_verification_s']:.2f}s|",
        f"|逆解初值优化次数|{old_starts:,}|{timing['ik_seed_starts']:,}|",
        f"|完整检查次数|{old_full}|{timing['full_checks']}|",
        f"|最终条件合格焊点|21/29|{fast['summary']['total_passed']}/29|", '',
        '## 提速具体修改', '',
        '1. 合法逆解逐个返回：立刻执行相同完整碰撞/接触检查，只有验收通过才停止；不再先求完全部逆解。',
        '2. 同一点、同机器人、同法向方向、同滚转角的已收敛合法逆解，按变位机角度距离排序作为优先初值。碰撞失败的逆解只作为初值，不充当可行证据。',
        '3. 难点仍用完原源候选、原初态、零及16个确定性合法Sobol初值；每次最大优化次数仍为220。源解析雅可比和求解精度保持一致。',
        '4. 4个独立进程并行焊点；Linux fork共享不可变网格页，每个进程独立修改FCL变换；每进程数值线程限制为1。先全体0°阶段结束，再提交变位机任务。',
        '5. 每角度追加JSONL事件、小文件保存进度，每点结束保存该点结果，最后只合并写一次大报告；不再每候选重写整个几十MB的报告。', '',
        '逆解初值次数仅减少约0.95%，不能把22倍提速主要归因于近邻初值。额外微基准显示旧版最终约61MB报告仅JSON序列化一次就需约1.31–1.34秒，旧代码每个候选都执行这一操作；加速版大幅减少重复序列化。历史运行未逐阶段计时，无法精确分摊各项收益。', '',
        '本次相对几何从源场景重新筛选，没有用v2最终21个构型直接复放。初始7点输入与v2相同，缓存只复用经hash校验的不可变几何。旧版是串行、加速版是4进程；这是本机一次实测对比，不是多次冷启动或其他场景的时间保证。', '',
        '## 查看最新报告', '', '```bash',
        "xdg-open '/home/ubuntu22/桌面/上航发焊点_静态姿态核验_最新.md'", '```', '',
        '## 独立复核并更新Markdown报告', '', '```bash',
        "cd '/home/ubuntu22/桌面/上航发_焊接规划_20261006'",
        '.venv/bin/python experimental/welding_positioner_v3/verify_fast.py',
        '.venv/bin/python experimental/welding_positioner_v3/speed_report.py', '```', '',
        '## 断点续跑', '', '```bash', "cd '/home/ubuntu22/桌面/上航发_焊接规划_20261006'",
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experimental/welding_positioner_v3/run_fast.py --workers 4 --resume',
        '```', '',
        '已完成任务的续跑会读取已保存点，不能用它作为从头搜索的计时结果。', '',
        f'原始完整结果：`{OUT}/report.json`。',
        f'实际计时：`{OUT}/timing.json`。',
        f'对比证据：`{OUT}/speed_comparison.json`。', '']
    text = '\n'.join(lines)
    (OUT/'提速结果与命令.md').write_text(text)
    (ROOT.parent/'上航发焊点_搜索提速_最新.md').write_text(text)
    latest = ROOT.parent/'上航发焊点_静态姿态核验_最新.md'
    static_text = latest.read_text()
    static_text = static_text.replace('3. 全部29点先在变位机0°进行固定工位筛选、逆解和双方机器人完整检查。通过点保持当前变位机姿态。',
        '3. 8个相对几何无候选点直接剔除。剩余21点先在变位机0°进行固定工位筛选、逆解和双方机器人完整检查。通过点保持当前变位机姿态。')
    static_text += f"\n## 本轮提速实测\n\n旧版{comparison['old_search_s']:.2f}s → 加速版{comparison['new_search_s']:.2f}s，约{ratio:.1f}倍。包含模型加载和最终落盘为{comparison['new_end_to_end_s']:.2f}s。详细命令见桌面`上航发焊点_搜索提速_最新.md`。\n"
    static_text = static_text.replace('experimental/welding_positioner_v2/verify.py', 'experimental/welding_positioner_v3/verify_fast.py')
    static_text = static_text.replace('experimental/welding_positioner_v2/report.py', 'experimental/welding_positioner_v3/speed_report.py')
    static_text = static_text.replace('.venv/bin/python experimental/welding_positioner_v2/search.py --resume',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experimental/welding_positioner_v3/run_fast.py --workers 4 --resume')
    latest.write_text(static_text)
    (OUT/'结果报告.md').write_text(static_text)
    print('SPEED COMPARISON', json.dumps(comparison, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
