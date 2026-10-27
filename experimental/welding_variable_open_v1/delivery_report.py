"""Digest-bound Chinese Markdown delivery of the opening-amplitude comparison."""
import argparse
import json
from pathlib import Path
from variable_world import VariableToolWorld
from run import ROOT, OUT
from run_geometry import digest, atomic

DESKTOP=ROOT.parent/'上航发焊枪_可变开度试验_最新.md'


def validate_delivery(report,cold,report_digest):
    if not report['summary']['complete'] or not report['summary']['full_opening_baseline_matches']:
        raise RuntimeError('Grid must be complete and full-open baseline must match')
    if cold['report_sha256']!=report_digest:
        raise RuntimeError('Cold verification is bound to another report')
    if not cold['summary']['all_passed'] or cold['summary']['reverified_retreats']!=report['summary']['any_amplitude_passed']:
        raise RuntimeError('Every selected path must pass cold verification')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args()
    out=args.out.resolve();report_path=out/'report.json';report=json.loads(report_path.read_text())
    cold=json.loads((out/'cold_verification.json').read_text());validate_delivery(report,cold,digest(report_path))
    for name,value in {**report['manifest']['source_sha256'],**report['manifest']['code_sha256'],**cold['verification_code_sha256']}.items():
        if digest(ROOT/name)!=value:raise RuntimeError('Tested source/code changed: '+name)
    for name,value in report['output_sha256'].items():
        if digest(out/name)!=value:raise RuntimeError('Saved evidence changed: '+name)
    old=json.loads((ROOT/'build/welding_tool_initial_exit_v2/initial_report.json').read_text())
    olds={r['weld_id']:r for r in old['rows']};new=[];lost=[];initial_matches=True
    for row in report['rows']:
        prior=olds[row['weld_id']]
        for key,value in prior['initial'].items():
            if row['initial'][key]!=value:initial_matches=False
        if row['best'] and not prior['retreat_passed']:new.append(row['weld_id'])
        if prior['retreat_passed'] and not row['best']:lost.append(row['weld_id'])
    if not initial_matches or lost:raise RuntimeError('Initial state or baseline success was lost')
    counts=report['by_amplitude'];best_fixed=max(counts,key=lambda x:x['open_and_retreat_passed'])
    tied_best=', '.join(f"{c['amplitude_mm']:g}" for c in counts if c['open_and_retreat_passed']==best_fixed['open_and_retreat_passed'])
    stationary=[r for r in report['rows'] if r['initial']['positioner_deg']==0.]
    other=[r for r in report['rows'] if r['initial']['positioner_deg']!=0.]
    metrics=dict(report_sha256=digest(report_path),cold_sha256=digest(out/'cold_verification.json'),
            initial_fields_match_old_baseline=initial_matches,new_successes=new,lost_successes=lost,
            stationary_successes=sum(r['best'] is not None for r in stationary),prior_positioner_successes=sum(r['best'] is not None for r in other))
    atomic(out/'delivery_audit.json',metrics)
    lines=['# 上航发焊枪：可变开度与直接直线退枪', '',
           f"固定全部原始焊接构型，20档幅度完整比较。全开185 mm基准仍为 **11/21**；逐点选择本轮成功幅度后为 **{report['summary']['any_amplitude_passed']}/21**，最终选出的路径全部通过冷复核。", '',
           f"统一使用一种幅度的最佳结果：驱动幅度 {tied_best} mm，均为{best_fixed['open_and_retreat_passed']}/21；逐点选幅度的合并结果不能视为这一档的成绩。", '',
           f"变位机0°的原7个初始点成功 {metrics['stationary_successes']}/7；原非零角度的14个初始点成功 {metrics['prior_positioner_successes']}/14。没有搜索新初始逆解或新的变位机角度。", '',
           f"新增成功点：{', '.join(new) if new else '无'}。", '',
           '幅度为源开枪驱动行程绝对值：闭合 driver=0，开枪 driver为负数，185 mm为全开。它不等于两枚电极的真实最小间距；下面逐点另列原始面片的实际间隙。', '',
           '| 驱动幅度/mm | 完整开枪通过/21 | 开枪＋至少100 mm退枪通过/21 |', '|---:|---:|---:|']
    for c in counts:lines.append(f"| {c['amplitude_mm']:g} | {c['opening_passed']} | {c['open_and_retreat_passed']} |")
    lines += ['', '## 各焊点测试网格最小成功幅度', '',
              '这里只能称本轮测试网格最小，不是连续参数空间的最小开度。', '',
              '| 焊点 | 原变位机角度/° | 最小成功幅度/mm | 电极实际间隙/mm | 全开退枪 |', '|---|---:|---:|---:|---|']
    for row in report['rows']:
        best=row['best'];prior=olds[row['weld_id']]
        lines.append(f"| {row['weld_id']} | {row['initial']['positioner_deg']:g} | {best['amplitude_mm'] if best else '未通过'} | {format(best['electrode_mesh_gap_mm'],'.3f') if best else '—'} | {'通过' if prior['retreat_passed'] else '未通过'} |")
    lines += ['', '## 未通过点', '']
    for row in report['rows']:
        if row['best']:continue
        reasons={}
        details=set()
        for trial in row['trials']:
            reasons[trial['reason']]=reasons.get(trial['reason'],0)+1
            for t in trial['exit_trials']:
                details.add(t['reason']+':'+str(t.get('failure',{}).get('reason','')))
        lines.append(f"- {row['weld_id']}：{json.dumps(reasons,ensure_ascii=False)}；退枪细分原因 {', '.join(sorted(details)) or '未找到几何出口'}。")
    lines += ['', '## 验收与边界', '',
        '每点开枪过程1 mm采样，一旦途中碰撞，更大幅度直接剔除；每幅度最多保留两条几何出口，再从原关节构型连续求解直线退出。保留源关节限位、J2/J3耦合限制和双臂检查。真实关节插值横向误差≤0.05 mm、姿态误差≤0.01°，退出至少100 mm，枪体终点净空≥5 mm。', '',
        '最终每点最小成功幅度另用新进程重新构建原始夹具面片，以源XML正运动学、0.5 mm开枪和实际关节边中点复核。其他成功幅度是原规划离散验收结果，没有逐条做此冷复核；失败只代表本轮有限方向搜索未找到成功路径。', '',
        '保留既有装配接口豁免：DONG–整个JING，DONG–两整个悬挂接口组件；枪与这些环境仍检查。本轮不验证变位机转动过程、点间转移或完整系统连续碰撞安全。', '',
        f"计算耗时：幅度扫描 {report['summary']['wall_s']:.2f} s（含模型加载 {report['summary']['model_load_s']:.2f} s）；最终选择冷复核 {cold['summary']['wall_s']:.2f} s。", '',
        '## 命令（Markdown，无视频）', '',
        '已完成实验的续跑命令只读取并验证完成报告，不重复计算、不覆盖原耗时：', '',
        '```bash',f"cd '{ROOT}'",'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experimental/welding_variable_open_v1/run.py --workers 4 --resume','```', '',
        '重新进行冷复核：', '',
        '```bash',f"cd '{ROOT}'",'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experimental/welding_variable_open_v1/verify.py --workers 4','```','',
        '查看完整结果：', '', '```bash',f"xdg-open '{out/'report.json'}'",f"xdg-open '{out/'cold_verification.json'}'",'```', '',
        f"真实关节/TCP轨迹保存在 `{out/'paths'}`，各点最终选用文件由 report.json 的 best.path_file 标明；未生成桌面视频。", '',
        f"报告 SHA256：`{digest(report_path)}`。",f"冷复核 SHA256：`{digest(out/'cold_verification.json')}`。"]
    text='\n'.join(lines)+'\n';DESKTOP.write_text(text);(out/'结果.md').write_text(text)
    print('DESKTOP_MARKDOWN',DESKTOP)
    print('DELIVERY',json.dumps(metrics,ensure_ascii=False))


if __name__=='__main__':main()
