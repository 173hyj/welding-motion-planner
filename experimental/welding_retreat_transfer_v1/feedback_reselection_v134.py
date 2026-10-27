"""Generate a replacement complete action only after an RRT edge failed."""
import copy
import json
import sys
import time
from pathlib import Path
import run_joint_weld_retreat_v134 as search
from integrated_retreat_v117 import online_retreat_ids
from run_geometry import atomic, digest

ROOT = Path(__file__).resolve().parents[2]


def invalidate_incident(known, index):
    return {edge: value for edge, value in known.items() if index not in edge}


class Reselection:
    def __init__(self, seconds, directions=64):
        self.seconds, self.directions = seconds, directions
        self.engine = None
        self.excluded = {}

    def generate(self, source, index, out):
        started = time.perf_counter()
        original = json.loads((source/'report.json').read_text())
        old = original['rows'][index]
        self.excluded.setdefault(old['weld_id'], []).append(copy.deepcopy(old['initial']))
        scene = json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
        if self.engine is None:
            self.engine = search.load_raw_world(scene, search.SIGNS)
        out.mkdir(parents=True, exist_ok=False)
        search.ENGINE, search.OUT, search.HINTS = self.engine, out, {}
        search.EXCLUDED = self.excluded
        # Current failed angle is feedback from this run, not a saved good pose.
        angles = search.positioner_angles()
        angle = old['initial']['positioner_deg']
        search.SETTINGS = dict(directions=self.directions, poses=48, configurations=4,
            point_seconds=self.seconds, amplitudes=[10.,30.,60.,80.,120.,140.,180.],
            feedback_angles=[angle]+[a for a in angles if a != angle])
        row = dict(copy.deepcopy(old), passed=False, initial=None, best=None,
                   online_acceptance=None, angle_trials=[], positioner_done=False)
        weld = next(w for w in scene['welds'] if w['id'] == old['weld_id'])
        _, replacement = search.search_point(index, row, weld, 'positioner')
        atomic(out/'reselection.json',dict(weld_id=old['weld_id'], passed=replacement['passed'],
            generation_wall_s=time.perf_counter()-started, no_historical_pose_hints=True,
            failed_configurations_excluded=len(self.excluded[old['weld_id']]),
            budget_semantics='Cooperative posture deadline; setup/load and atomic final online trace may extend wall time'))
        if not replacement['passed']:
            return None, None
        # Preserve the other actions as unchanged nodes of this same experiment.
        # Materialize only their bound point/path evidence in a new report.
        rows = copy.deepcopy(original['rows'])
        rows[index] = replacement
        for i, r in enumerate(rows):
            if i != index:
                name = f'paths/retreat_{i:02d}.json'
                path = json.loads((source/r['best']['path_file']).read_text())
                atomic(out/name,path)
                r['best']['path_file'] = name
                assert digest(out/name) == r['best']['path_sha256']
            atomic(out/f'point_{i:02d}.json',r)
        manifest = copy.deepcopy(original['manifest'])
        manifest['source_sha256'][str((source/'report.json').relative_to(ROOT))] = digest(source/'report.json')
        for module in list(sys.modules.values()):
            name = getattr(module,'__file__',None)
            if name:
                f=Path(name).resolve()
                if f.suffix == '.py' and f.is_relative_to(ROOT) and '.venv' not in f.parts:
                    manifest['code_sha256'][str(f.relative_to(ROOT))] = digest(f)
        manifest['settings']['feedback_reselection'] = dict(index=index,seconds=self.seconds,
            no_historical_pose_hints=True, excluded_current_failed_configurations=True)
        outputs = {str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}
        report = dict(manifest=manifest,rows=rows,output_sha256=outputs,
            summary=dict(original['summary'], passed=sum(r['passed'] for r in rows),
                wall_s=time.perf_counter()-started, timing_scope='Replacement action and same-run unchanged-node evidence only'),
            scope='Failed-edge feedback replacement; unchanged actions retained from this same experiment. No historical good configuration candidate injected.')
        online_retreat_ids(report,out)
        atomic(out/'report.json',report)
        return report, replacement
