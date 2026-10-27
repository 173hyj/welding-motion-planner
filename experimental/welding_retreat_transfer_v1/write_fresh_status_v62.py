"""Publish separate candidate, retreat and complete-route evidence in Markdown."""
import json
from datetime import datetime
from pathlib import Path
from rebuild_evidence_v43 import field, digest

ROOT = Path(__file__).resolve().parents[2]
DESKTOP = ROOT.parent / '上航发焊接_指定距离退枪与点间规划_最新.md'
BATCHES = [
    ('8点', 'welding_retreat_200cm_same_pose_v55', 'same_pose_expanded_v56'),
    ('10点', 'welding_retreat_200cm_retention_snapshot_v59', 'retention_expanded_v60'),
    ('11点扩展', 'welding_retreat_200cm_expanded_snapshot_v61', 'expanded_v61'),
    ('13点扩展', 'welding_retreat_200cm_expanded_snapshot_v66', 'expanded_v66'),
    ('外轴角度细化', 'welding_retreat_200cm_refined_angles_v69', 'refined_v69'),
    ('近终点方向细化', 'welding_retreat_200cm_near_ray_v72', 'near_ray_v72'),
    ('完整球面方向', 'welding_retreat_200cm_full_sphere_v73', 'full_sphere_v73'),
    ('局部外轴细化', 'welding_retreat_200cm_local_external_v74', 'local_external_v74'),
    ('14点扩展', 'welding_retreat_200cm_expanded_snapshot_v75', 'expanded_v75'),
    ('65°/85°外轴扩展', 'welding_retreat_200cm_local_external_v76', 'local_external_v76'),
    ('55°/95°外轴扩展', 'welding_retreat_200cm_local_external_v77', 'local_external_v77'),
    ('剩余点完整球面', 'welding_retreat_200cm_deferred_sphere_v79', 'deferred_sphere_v79'),
    ('新构型与开度扩展', 'welding_retreat_200cm_deferred_sphere_v81', 'deferred_sphere_v81'),
    ('未查构型＋紧凑证据', 'welding_retreat_200cm_deferred_sphere_v83', 'deferred_sphere_v83'),
    ('非工作臂肩肘停靠重选', 'welding_retreat_200cm_folded_idle_v86', 'folded_idle_v86'),
    ('剩余4点新外轴域', 'welding_retreat_200cm_deferred_external_v87', 'deferred_external_v87'),
    ('小开度1–4mm', 'welding_retreat_200cm_small_openings_v89', 'small_openings_v89'),
    ('源合法大开度90–185mm', 'welding_retreat_200cm_large_openings_v90', 'large_openings_v90'),
    ('已有连接优先的经典排序', 'welding_retreat_200cm_large_openings_v90', 'certified_ordering_v92'),
    ('大开度新构型', 'welding_retreat_200cm_large_new_seeds_v93', 'large_new_seeds_v93'),
    ('实际肩肘过渡构型', 'welding_retreat_200cm_large_new_seeds_v93', 'endpoint_folded_v95'),
    ('最后两身份新大开度构型', 'welding_retreat_200cm_remaining_large_v96', 'remaining_large_v96'),
    ('全库存汇总', 'welding_retreat_200cm_revalidated_merged_v60', 'revalidated_merged_v60'),
]


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def certificate(directory, retreat, transfer=None):
    path = ROOT / 'build' / directory / 'cold_verification.json'
    cold = read(path)
    if cold is None:
        return None
    if cold['retreat_report_sha256'] != digest(retreat / 'report.json'):
        raise ValueError('Cold certificate retreat binding changed: ' + str(path))
    if transfer and cold['transfer_report_sha256'] != digest(transfer / 'report.json'):
        raise ValueError('Cold certificate transfer binding changed: ' + str(path))
    for name, sha in cold['verification_code_sha256'].items():
        if digest(ROOT / name) != sha:
            raise ValueError('Verification code changed: ' + name)
    report = read(retreat / 'report.json')
    selected = {r['weld_id']: r for r in report['rows'] if r['best']}
    if set(selected) != {r['weld_id'] for r in cold['retreat_rows']}:
        raise ValueError('Cold certificate retreat scope differs')
    for row in cold['retreat_rows']:
        best = selected[row['weld_id']]['best']
        if row['planned_path_sha256'] != best['path_sha256'] or digest(retreat / best['path_file']) != best['path_sha256']:
            raise ValueError('Cold certificate retreat path changed')
    if transfer:
        report = read(transfer / 'report.json')
        selected = {(e['from_weld_id'], e['to_weld_id']): e for e in report['selected_edges']}
        if set(selected) != {(e['from_weld_id'], e['to_weld_id']) for e in cold['transfer_rows']}:
            raise ValueError('Cold certificate transfer scope differs')
        for row in cold['transfer_rows']:
            edge = selected[(row['from_weld_id'], row['to_weld_id'])]
            if row['planned_path_sha256'] != edge['path_sha256'] or digest(transfer / edge['path_file']) != edge['path_sha256']:
                raise ValueError('Cold certificate transfer path changed')
    return cold['summary']


def main():
    lines = ['# 上航发：200厘米直线退枪与点间规划', '',
        '更新：' + datetime.now().astimezone().isoformat(timespec='seconds'), '',
        '**200 cm = 2000 mm = 2米。采用直线退枪＋点间转移，不使用QP。Goal保持active，全部21点尚未完成。**', '',
        '原场景29个焊点中，闭合焊枪相对几何筛选保留21个；后续批次始终保留全部21个身份和14条原始完整2米候选。', '']
    base = ROOT / 'build/welding_retreat_200cm_revalidated_inventory_v43'
    cold = certificate('welding_retreat_200cm_revalidated_inventory_cold_v43', base)
    lines += [f'原14条候选经新独立原网格/XML FK复核：{cold["passed_retreats"]}/{cold["selected_retreats"]}通过，{cold["wall_s"]:.2f}秒。这不代表14条已连成路线。', '']
    live = ROOT / 'build/welding_retreat_200cm_mixed_wrist_v44'
    progress = field(live / 'report.json', 'summary') if (live / 'report.json').exists() else read(live / 'progress.json')
    if progress:
        lines += [f'全21点固定分支搜索：完成{progress.get("finished", progress.get("points", 0))}/21；兼容候选{progress["branch_compatible"]}；原完整候选仍保留。R2负末轴、R1正末轴，非工作臂同分支停靠；有限失败不证明无解。', '']
    refined = ROOT / 'build/welding_retreat_200cm_refined_angles_v69'
    refined_progress = field(refined / 'report.json', 'summary') if (refined / 'report.json').exists() else read(refined / 'progress.json')
    if refined_progress:
        lines += [f'新角度细化批次v69：完成{refined_progress.get("finished", refined_progress.get("points", 0))}/21，兼容候选{refined_progress["branch_compatible"]}，原候选保留{refined_progress.get("retained_original_candidates", 14)}。已认证13条原样保留，仅搜索剩余身份；优先SUB1 GW8，新增±15/45/75/105/135/165°变位机角度，焊枪姿态仍5°网格。通过候选仍须新cold和真实连接，尚非全21完成。', '']
    near = ROOT / 'build/welding_retreat_200cm_near_ray_v72'
    near_summary = read(near / 'report.json')
    near_progress = read(near / 'progress.json')
    if near_summary:
        summary = near_summary['summary']
        lines += [f'近终点方向细化v72：有限批次完成，新正例={summary["new_positive"]}，候选{summary["passed"]}/21，耗时{summary["wall_s"]:.2f}秒；是否通过退枪与整路线复核见下表。', '']
    elif near_progress:
        lines += [f'近终点方向细化v72正在运行，已尝试{near_progress["attempts"]}组方向/开度组合；保持SUB1 GW1初始姿态和75°外轴，按5°球面方向筛选释放关系，先检查完整端点自碰撞再做完整trace。', '']
    sphere = ROOT / 'build/welding_retreat_200cm_full_sphere_v73'
    sphere_report = read(sphere / 'report.json')
    sphere_progress = read(sphere / 'progress.json')
    if sphere_report:
        ss = sphere_report['summary']
        lines += [f'完整5°球面批次v73已结束，新正例={ss["new_positive"]}，耗时{ss["wall_s"]:.2f}秒；有限失败不证明全局无解。', '']
    elif (sphere / 'manifest.json').exists():
        text = f'已记录{sphere_progress["attempts"]}组尝试' if sphere_progress else '正在加载或执行第一组开度'
        lines += [f'完整5°球面方向v73正在运行：{text}。使用相同v72代码，将方向预算扩为2522，依然先按释放支持过滤；原姿态、75°外轴、源限位和碰撞条件不变。', '']
    local = ROOT / 'build/welding_retreat_200cm_local_external_v74'
    local_summary = read(local / 'report.json')
    local_progress = read(local / 'progress.json')
    if local_summary:
        ls = local_summary['summary']
        lines += [f'局部外轴细化v74有限批次完成：候选{ls["branch_compatible"]}/21，耗时{ls["wall_s"]:.2f}秒。新增正例才自动执行自己的退枪及整路线复核。', '']
    elif local_progress:
        lines += [f'局部外轴细化v74正在运行：完成{local_progress["finished"]}/21，候选{local_progress["branch_compatible"]}。保持完整21身份/14原候选，优先SUB1 GW1，在初始角之外尝试70°、80°外轴；焊枪姿态仍5°。', '']
    expanded = ROOT / 'build/welding_retreat_200cm_local_external_v76'
    expanded_report = read(expanded / 'report.json')
    expanded_progress = read(expanded / 'progress.json')
    if expanded_report:
        es = expanded_report['summary']
        lines += [f'65°/85°外轴批次v76完成：候选{es["branch_compatible"]}/21，耗时{es["wall_s"]:.2f}秒；新候选需自己的完整复核流程。', '']
    elif (expanded / 'manifest.json').exists():
        ep = expanded_progress or dict(finished=15, branch_compatible=15)
        lines += [f'65°/85°外轴批次v76继续：完成{ep["finished"]}/21，候选{ep["branch_compatible"]}；从完整v74库存原样保留15条候选、14条原candidate和全部21身份，仅搜索剩余6点，优先SUB1 GW8。', '']
    farther = ROOT / 'build/welding_retreat_200cm_local_external_v77'
    farther_report = read(farther / 'report.json')
    farther_progress = read(farther / 'progress.json')
    if farther_report:
        fs = farther_report['summary']
        lines += [f'55°/95°外轴批次v77完成：候选{fs["branch_compatible"]}/21，耗时{fs["wall_s"]:.2f}秒；有限失败不证明不可行。', '']
    elif (farther / 'manifest.json').exists():
        fp = farther_progress or dict(finished=15, branch_compatible=15)
        lines += [f'55°/95°外轴批次v77运行：完成{fp["finished"]}/21，候选{fp["branch_compatible"]}；原15条路径和14条原候选均保留，只搜索剩余6点的新外轴域。', '']
    deferred_sphere = ROOT / 'build/welding_retreat_200cm_deferred_sphere_v79'
    sphere_inventory = read(deferred_sphere / 'report.json')
    sphere_live = read(deferred_sphere / 'progress.json')
    if sphere_inventory:
        ds = sphere_inventory['summary']
        lines += [f'剩余点完整球面批次v79结束：候选{ds["passed"]}/21，新增{ds["new_positives"]}，耗时{ds["wall_s"]:.2f}秒；新正例仍须独立cold及实际连接。', '']
    elif sphere_live:
        lines += [f'剩余点完整球面v79运行：完成{sphere_live["finished"]}/21，候选{sphere_live["compatible"]}；当前{sphere_live.get("weld_id", "点间切换")}。每个剩余点最多两个已尝试源姿态，补查完整5°俯仰/方位角球面；旧负例只排序，重新检查全部原opening/geometry/2000mm trace。', '']
    augmented = ROOT / 'build/welding_retreat_200cm_deferred_sphere_v81'
    augmented_report = read(augmented / 'report.json')
    augmented_live = read(augmented / 'progress.json')
    if augmented_report:
        au = augmented_report['summary']
        lines += [f'新构型与开度扩展v81结束：候选{au["passed"]}/21，新增{au["new_positives"]}，耗时{au["wall_s"]:.2f}秒。有限实验不证明不可行，正例另需本批cold与真实转移。', '']
    elif (augmented / 'manifest.json').exists():
        ap = augmented_live or dict(finished=15, compatible=15)
        lines += [f'新构型与开度扩展v81运行：完成{ap["finished"]}/21，候选{ap["compatible"]}；每个剩余点最多4个尚未在v79试过的exact源构型，完整5°球面，新增5/20/40/50/70mm开度并保留原10/30/60/80mm。源opening/geometry/lift/limits验收不变。', '']
    compact = ROOT / 'build/welding_retreat_200cm_deferred_sphere_v83'
    compact_report = read(compact / 'report.json')
    compact_live = read(compact / 'progress.json')
    if compact_report:
        cs = compact_report['summary']
        lines += [f'未查构型v83完成：候选{cs["passed"]}/21，新增{cs["new_positives"]}，搜索{cs["wall_s"]:.2f}秒。GW8新增R1/30°外轴/295°滚转/80mm开度候选；独立复核和真实转移状态见下表。继承搜索字段引用原report＋SHA，旧文件保留，执行字段与本地路径完整。', '']
    elif compact_live:
        lines += [f'未查构型v83运行：完成{compact_live["finished"]}/21，候选{compact_live["compatible"]}；每点最多4个未在v79/v81搜索的精确初始构型。', '']
    folded = ROOT / 'build/welding_retreat_200cm_folded_idle_v86'
    folded_report = read(folded / 'report.json')
    folded_live = read(folded / 'progress.json')
    if folded_report:
        fs = folded_report['summary']
        lines += [f'GW7非工作臂肩肘停靠重选v86完成：候选{fs["passed"]}/21，新增{fs["new_positives"]}，{fs["wall_s"]:.2f}秒。只重新检查真实other_robot失败射线；每条退枪内park固定，正例仍需自己的cold和真实15axis转移。', '']
    elif (folded / 'manifest.json').exists():
        fp = folded_live or dict(attempts=0)
        lines += [f'GW7非工作臂肩肘停靠重选v86运行：已记录{fp["attempts"]}次尝试。沿用原活跃臂姿态与2000mm直线，重选合法非工作臂q1/q2/q3；稀疏source检查只拒绝，原完整opening/geometry/lift验收不变。', '']
    external = ROOT / 'build/welding_retreat_200cm_deferred_external_v87'
    external_report = read(external / 'report.json')
    external_live = read(external / 'progress.json')
    if external_report:
        es = external_report['summary']
        lines += [f'剩余4点新外轴域v87完成：候选{es["passed"]}/21，新增{es["new_positives"]}，{es["wall_s"]:.2f}秒；有限未通过不证明不可行。', '']
    elif (external / 'manifest.json').exists():
        ep = external_live or dict(finished=17, compatible=17)
        lines += [f'剩余4点新外轴域v87运行：完成{ep["finished"]}/21，候选{ep["compatible"]}。GW6(SUB1)100/110°、GW4/GW6(SUB2)140/160°、GW14(SUB3)130/140°；17旧路径按各自实际park原样保留，不改回旧全局park。', '']
    small_report = read(ROOT / 'build/welding_retreat_200cm_small_openings_v89/report.json')
    large_report = read(ROOT / 'build/welding_retreat_200cm_large_openings_v90/report.json')
    newlarge = ROOT / 'build/welding_retreat_200cm_large_new_seeds_v93'
    newlarge_report = read(newlarge / 'report.json')
    if small_report:
        lines += ['v89小开度1/2/3/4mm有限搜索56.09秒，无新增，实际31850组；6个seed的1mm被释放支持过滤后没有方向trial，不声称每seed都测试1225方向。', '']
    if large_report:
        ls = large_report['summary']
        lines += [f'v90源合法大开度搜索完成：候选{ls["passed"]}/21，新增{ls["new_positives"]}，耗时{ls["wall_s"]:.2f}秒。新增SUB2 GW6：R1、140°外轴、230°roll、90mm开度；原opening前缀必须通过，source maximum185mm不放宽。', '']
    if newlarge_report:
        ns = newlarge_report['summary']
        lines += [f'v93大开度新构型批次完成：候选{ns["passed"]}/21，新增{ns["new_positives"]}，耗时{ns["wall_s"]:.2f}秒。', '']
    elif (newlarge / 'manifest.json').exists():
        lines += ['v93继续搜索剩余3身份，每点4个未在v90大开度域尝试的精确源构型，原90–185mm开度/5°球面/完整source2000mm trace保持。', '']
    remaining_report = read(ROOT / 'build/welding_retreat_200cm_remaining_large_v96/report.json')
    if remaining_report:
        rs = remaining_report['summary']
        lines += [f'v96最后两身份新构型搜索完成：候选{rs["passed"]}/21，新增{rs["new_positives"]}，{rs["wall_s"]:.2f}秒。', '']
    elif (ROOT / 'build/welding_retreat_200cm_remaining_large_v96/manifest.json').exists():
        lines += ['v96继续SUB1 GW6／SUB3 GW14，每点4个未在v90/v93大开度域试过的exactseed，原source/5°sphere/90–185mm完整opening/2000mm trace不变。', '']
    lines += ['v92独立排序实验使用原nearest-neighbour＋open2opt，未试边有限加价、已通过边真实长度，启发式偏好已有连接（不保证最少新边数或全局最优）；没有旧seed incident的新节点先尝试一个相同外轴角节点。v95继承此排序，把真实端点肩肘构型、源park和新foldedpark组合为中间proposal；每段仍原source/RRT/fine验收，不能瞬移。结果见下表。', '']
    latest_snapshot = read(ROOT / 'build/welding_retreat_200cm_expanded_snapshot_v75/report.json')
    if latest_snapshot:
        inventory = remaining_report or newlarge_report or large_report or small_report or external_report or folded_report or compact_report or augmented_report or sphere_inventory or farther_report or expanded_report or local_summary or latest_snapshot
        deferred = [r['weld_id'].replace('OP10L-', '').replace('-POINT-', ' ')
                    for r in inventory['rows'] if not r['best']]
        lines += ['最新完整库存尚未纳入的身份：' + '、'.join(deferred) + '。这些点仍保留在完整任务库存中，不把有限搜索失败标为不可行证明。', '']
    lines += ['| 批次 | 候选数 | 退枪独立复核 | 实际转移规划 | 最终整路线独立复核 |',
              '|---|---:|---|---|---|']
    details = []
    for label, retreat_name, tag in BATCHES:
        retreat = ROOT / 'build' / retreat_name
        if not (retreat / 'report.json').exists():
            continue
        summary = field(retreat / 'report.json', 'summary')
        retreat_cold_tag = {'certified_ordering_v92':'large_openings_v90', 'endpoint_folded_v95':'large_new_seeds_v93'}.get(tag,tag)
        rc = certificate('welding_retreat_200cm_' + retreat_cold_tag + '_cold', retreat)
        transfer = ROOT / 'build' / ('welding_transfer_200cm_' + tag)
        ts = field(transfer / 'report.json', 'summary') if (transfer / 'report.json').exists() else None
        fc = certificate('welding_retreat_transfer_200cm_' + tag + '_cold', retreat, transfer)
        no_new_positive = (summary.get('new_positive') is False or summary.get('new_positives') == 0 or
            (tag in ('local_external_v76', 'local_external_v77') and summary['passed'] <= 15))
        retreat_text = f'{rc["passed_retreats"]}/{rc["selected_retreats"]}，{rc["wall_s"]:.2f} s' if rc else ('无新正例，未启动本批复核' if no_new_positive else '进行中')
        transfer_text = f'{ts["selected_passed_edges"]}/{ts["expected_edges"]}，{ts["wall_s"]:.2f} s' if ts else ('未启动' if no_new_positive else '等待退枪复核')
        final_text = f'退枪{fc["passed_retreats"]}/{fc["selected_retreats"]}＋转移{fc["passed_transfers"]}/{fc["selected_transfers"]}，{fc["wall_s"]:.2f} s；子集={fc["retained_subset_route_verified"]}，全21={fc["full_route_verified"]}' if fc else ('未新增路线认证' if no_new_positive else '尚未完成')
        lines.append(f'| {label} | {summary["passed"]}/21 | {retreat_text} | {transfer_text} | {final_text} |')
        if ts:
            order = field(transfer / 'report.json', 'order_weld_ids')
            edges = field(transfer / 'report.json', 'selected_edges')
            new = sum(not e.get('seed_path_sha256') for e in edges)
            details += [f'{label}顺序：' + ' → '.join(order), '',
                f'本批新增{new}条边，复用{len(edges)-new}条已通过新冷复核且完整15轴端点一致的边。{ts["wall_s"]:.2f}秒为增量批次规划时间（含模型加载{ts["model_load_s"]:.2f}秒），不代表从零计算总时间。', '']
    lines += ['', *details,
        '顺序使用多起点最近邻＋开放路径2-opt，代价包含关节、变位机、开度变化与机械臂交接；受阻边触发重新排序，不宣称全局最优。', '',
        '转移先试完整状态直连与共同停靠阶段链，受阻时采用RRT-Connect。点间转移按关节0.05°、变位机0.25°、开度0.5mm加密验收，保存完整15轴及双TCP实际FK。直线退枪另用源实现的关节插值（最大0.25°）、实际TCP直线误差0.05mm/姿态误差0.01°约束；独立复核检查保存帧和中点。不做中途wrap、不瞬移非工作臂或变位机、不用端点连线冒充运动轨迹。', '',
        '11点快照新增SUB3 GW4：30°变位机、130°焊枪滚转、60mm开度，完整2000mm。新13点快照再纳入SUB3 GW7（90°变位机/260°滚转/10mm开度）与GW9（150°变位机/150°滚转/10mm开度）。每个扩展批次均须自己的独立退枪及整路线复核；已有11点证书不能认证13点。', '',
        '预筛顺序实验保持原coarse检查，先做连续运动学拒绝，再做原fine几何/完整trace。4条已知负例gate微基准20.01秒→2.15秒；GW4同一前30姿态/原完整方向库的整点搜索859.62秒→243.33秒（约3.53倍），原成功路径SHA逐字节相同，模型加载时间排除。只报告这组微基准和单点对照，不推广为全部场景加速。新入口v65已实现原候选保留，后续加入方向及seed排序/开度扩展测试，本轮93项测试通过；旧批次自然结束，v81按新构型/开度域继续；没有因elapsed重启。', '',
        'v80对12个源姿态各前32条粗工具几何失败方向定位首次阻碍，33.77秒：部分5mm附近电极/工件或枪体/夹具碰撞，另有15–1955mm沿途阻碍。只是有限诊断，没有认定粗筛误拒、没有放宽原释放接触/碰撞判据、没有新增路线认证。', '',
        'GW8诊断：固定旧停靠时3条失败ray在556/578/596mm出现两臂连杆碰撞。改变非工作臂q1的有限15次尝试仍遇自碰撞/两臂碰撞，未得到新正例（218.51秒）。原限位下225个q6=0构型样本全部有碰撞（30.78秒），其中每个都有源自碰撞pair；这只是有限诊断，不证明所有连续分支转换都不可行，也未更改碰撞豁免或源关节限制。', '',
        'SUB1 GW1近终点诊断v71：75°外轴、355°滚转、60mm开度的旧失败射线回退约1917mm后出现R2连杆243/245自碰撞，工具blocking为空。30.92秒的源fullprobe仅诊断失败附近，不是完整退枪认证。v72优先尝试最近256个满足释放支持的5°球面方向；端点预筛仅拒绝，正例仍要求原完整opening/geometry/lift及新cold。', '',
        'v74新增SUB1 GW1候选：70°变位机、350°滚转、60mm开度，原完整trace通过2000mm；该角度单点搜索266.90秒。14点快照v75保持全21身份和14条原candidate，自己的退枪cold→精确端点seed→真实转移→finalcold自动执行。候选通过不等于冷复核通过，状态见表。', '',
        'v74再新增SUB1 GW9候选：70°变位机、55°滚转、80mm开度，R1原完整trace通过2000mm；该角度单点搜索25.76秒。live库存达到15候选；v74完整批次结束后另执行自己的退枪cold和整路线cold，不由14点证书代替认证。', '',
        '## 进度和结果命令', '', '```bash', f"cd '{ROOT}'",
        'tail -n 20 /tmp/welding_retreat_200cm_mixed_wrist_v44.log',
        'tail -n 20 /tmp/welding_retreat_200cm_refined_angles_v69.log',
        'tail -n 20 /tmp/welding_refined_v69_pipeline.log',
        'tail -n 20 /tmp/welding_retreat_200cm_near_ray_v72.log',
        'tail -n 20 /tmp/welding_near_ray_v72_pipeline.log',
        'tail -n 20 /tmp/welding_retreat_200cm_full_sphere_v73.log',
        'tail -n 20 /tmp/welding_full_sphere_v73_pipeline.log',
        'tail -n 20 /tmp/welding_retreat_200cm_local_external_v74.log',
        'tail -n 20 /tmp/welding_local_external_v74_pipeline.log',
        'tail -n 20 /tmp/welding_expanded_v75_pipeline.log',
        'tail -n 20 /tmp/welding_retreat_200cm_expanded_v75_cold.log',
        'tail -n 20 /tmp/welding_transfer_200cm_expanded_v75.log',
        'tail -n 20 /tmp/welding_retreat_transfer_200cm_expanded_v75_cold.log',
        'tail -n 20 /tmp/welding_retreat_200cm_local_external_v76.log',
        'tail -n 20 /tmp/welding_local_external_v76_pipeline.log',
        'tail -n 20 /tmp/welding_retreat_200cm_local_external_v77.log',
        'tail -n 20 /tmp/welding_local_external_v77_pipeline.log',
        'tail -n 20 /tmp/welding_retreat_200cm_deferred_sphere_v79.log',
        'tail -n 20 /tmp/welding_deferred_sphere_v79_pipeline.log',
        'tail -n 20 /tmp/welding_coarse_release_diagnostic_v80.log',
        'tail -n 20 /tmp/welding_retreat_200cm_deferred_sphere_v81.log',
        'tail -n 20 /tmp/welding_deferred_sphere_v81_pipeline.log',
        'tail -n 20 /tmp/welding_retreat_200cm_deferred_sphere_v83.log',
        'tail -n 20 /tmp/welding_deferred_sphere_v83_pipeline.log',
        'tail -n 20 /tmp/welding_retreat_200cm_folded_idle_v86.log',
        'tail -n 20 /tmp/welding_folded_idle_v86_pipeline.log',
        'tail -n 20 /tmp/welding_retreat_200cm_deferred_external_v87.log',
        'tail -n 20 /tmp/welding_retreat_200cm_small_openings_v89.log',
        'tail -n 20 /tmp/welding_retreat_200cm_large_openings_v90.log',
        'tail -n 20 /tmp/welding_large_openings_v90_pipeline.log',
        'tail -n 20 /tmp/welding_certified_ordering_v92_pipeline.log',
        'tail -n 20 /tmp/welding_retreat_200cm_large_new_seeds_v93.log',
        'tail -n 20 /tmp/welding_endpoint_folded_v95_pipeline.log',
        'tail -n 20 /tmp/welding_retreat_200cm_remaining_large_v96.log',
        'tail -n 20 /tmp/welding_fresh_retention_v60_pipeline.log',
        'tail -n 20 /tmp/welding_retreat_200cm_expanded_v61_cold.log',
        'tail -n 20 /tmp/welding_transfer_200cm_expanded_v61.log',
        'tail -n 20 /tmp/welding_retreat_transfer_200cm_expanded_v61_cold.log',
        'tail -n 20 /tmp/welding_expanded_v66_pipeline.log',
        'tail -n 20 /tmp/welding_retreat_200cm_expanded_v66_cold.log',
        'tail -n 20 /tmp/welding_transfer_200cm_expanded_v66.log',
        'tail -n 20 /tmp/welding_retreat_transfer_200cm_expanded_v66_cold.log',
        'tail -n 20 /tmp/welding_fresh_merged_v60_pipeline.log', '```', '',
        '更新本Markdown（逐项核对证书与报告、路径、复核代码SHA）：', '', '```bash', f"cd '{ROOT}'",
        '.venv/bin/python experimental/welding_retreat_transfer_v1/write_fresh_status_v62.py', '```', '',
        '重新执行13点快照的完整流水线（使用新的输出tag，勿重复启动当前计算；通用入口使用v56/v53已验证种子，计时不可直接视为当前v61/v60种子的复现）：', '', '```bash', f"cd '{ROOT}'",
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u experimental/welding_retreat_transfer_v1/finish_merged_transfer_v60.py --retreat-out build/welding_retreat_200cm_expanded_snapshot_v66 --tag expanded_reproduction_v66', '```', '',
        '完整单点预筛对照复跑（新目录，勿重复启动同名结果）：', '', '```bash', f"cd '{ROOT}'",
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u experimental/welding_retreat_transfer_v1/probe_ordered_witness_v64.py --out build/welding_rejection_order_full_witness_v64_reproduction', '```', '',
        '变位机细化批次复跑（全21身份保留；使用新输出目录）：', '', '```bash', f"cd '{ROOT}'",
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u experimental/welding_retreat_transfer_v1/probe_mixed_wrist_v65.py --retreat-out build/welding_retreat_200cm_expanded_snapshot_v66 --out build/welding_retreat_200cm_refined_angles_v69_reproduction --priority-weld OP10L-SUB1-POINT-GW-8 --angles 15 45 75 105 135 165 -15 -45 -75 -105 -135 -165', '```', '',
        '完整5°球面方向实验复跑（保持GW1原姿态和75°外轴；新输出目录）：', '', '```bash', f"cd '{ROOT}'",
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u experimental/welding_retreat_transfer_v1/probe_near_retreat_v72.py --out build/welding_retreat_200cm_full_sphere_v73_reproduction --directions 2522', '```', '',
        '局部70°/80°外轴实验复跑（全21身份保留；新输出目录）：', '', '```bash', f"cd '{ROOT}'",
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u experimental/welding_retreat_transfer_v1/probe_mixed_wrist_v65.py --retreat-out build/welding_retreat_200cm_refined_angles_v69 --out build/welding_retreat_200cm_local_external_v74_reproduction --priority-weld OP10L-SUB1-POINT-GW-1 --angles 70 80', '```', '',
        '剩余六点完整5°球面复跑（新输出目录，每点2个初始姿态）：', '', '```bash', f"cd '{ROOT}'",
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u experimental/welding_retreat_transfer_v1/probe_deferred_sphere_v79.py --out build/welding_retreat_200cm_deferred_sphere_v79_reproduction --seeds 2', '```', '',
        '新构型与开度扩展复跑（新输出目录，4个新初始构型/点）：', '', '```bash', f"cd '{ROOT}'",
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u experimental/welding_retreat_transfer_v1/probe_deferred_sphere_v81.py --out build/welding_retreat_200cm_deferred_sphere_v81_reproduction --seeds 4', '```', '',
        '## 未查构型＋紧凑证据v83复现', '', '```bash',
        'cd ' + str(ROOT),
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u experimental/welding_retreat_transfer_v1/probe_deferred_sphere_v83.py --out build/welding_retreat_200cm_deferred_sphere_v83_reproduction --seeds 4', '```', '',
        '## 非工作臂肩肘停靠重选v86复现', '', '```bash',
        'cd ' + str(ROOT),
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u experimental/welding_retreat_transfer_v1/probe_folded_idle_v86.py --out build/welding_retreat_200cm_folded_idle_v86_reproduction', '```', '',
        '## 剩余4点外轴域v87复现', '', '```bash',
        'cd ' + str(ROOT),
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u experimental/welding_retreat_transfer_v1/probe_deferred_external_v87.py --out build/welding_retreat_200cm_deferred_external_v87_reproduction', '```', '',
        '## 大开度v90复现', '', '```bash',
        'cd ' + str(ROOT),
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u experimental/welding_retreat_transfer_v1/probe_large_openings_v90.py --out build/welding_retreat_200cm_large_openings_v90_reproduction --seeds 2', '```', '',
        '## 大开度新构型v93复现', '', '```bash',
        'cd ' + str(ROOT),
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u experimental/welding_retreat_transfer_v1/probe_large_new_seeds_v93.py --out build/welding_retreat_200cm_large_new_seeds_v93_reproduction --seeds 4', '```', '',
        '## 最后两身份v96复现', '', '```bash',
        'cd ' + str(ROOT),
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u experimental/welding_retreat_transfer_v1/probe_remaining_large_v96.py --out build/welding_retreat_200cm_remaining_large_v96_reproduction --seeds 4', '```', '',
        '当前交付规划路径JSON与可审计结果。新路线MuJoCo入口尚未更新，旧入口不作为这些非零变位机路线的回放命令。', '',
        '## 证据与清理边界', '',
        '先前清理误删了仍被引用的历史结果，是执行错误。两份碰撞源码、v2的21份单点及关键路径已按原SHA精确恢复；49份历史JSON仍缺失，原seals未改写。新v43 dependency_loss_audit.json记录完整缺失清单，historical_chain_intact=False。新批次不复用断链旧证书或旧转移种子，原14条路径以保留字节重新复核；历史负搜索与耗时不重新认证。', '',
        '碰撞检查保留源装配安装接口豁免和枪内部接口边界；结果是加密离散仿真复核，不能视为连续全系统安全证明。此前清理净释放约2GB；本轮v78对20个已完成封存、逐字节相同的路径用硬链接合并，实测增加可用空间14.344192MB，全部文件名/内容/SHA保留、没有路径删除。继续保留原源资产与当前证据；桌面只生成Markdown，不生成视频。', '']
    dedup = read(ROOT / 'build/welding_exact_path_dedup_v88/report.json')
    if dedup and dedup.get('complete'):
        lines += [f'v88对{len(dedup["entries"])}个完成封存的精确重复路径执行硬链接合并，实测可用空间变化{dedup["measured_free_change_bytes"]/1000000:.6f}MB（测量期间其他任务亦在写入）；原文件名、内容与SHA全部保留。', '']
    lines += ['v85停靠实验入口曾在候选字段合并处抛TypeError，未生成完成结果，不能算负搜索结论。v86隔离修正该入口并增加真实回归测试，旧v85封存文件保留。', '']
    temporary = DESKTOP.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines))
    temporary.replace(DESKTOP)
    print('FRESH_DESKTOP_MD', DESKTOP)


if __name__ == '__main__':
    main()
