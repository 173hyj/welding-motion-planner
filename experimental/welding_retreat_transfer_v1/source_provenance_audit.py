"""Supplement sealed v1 runs with an explicit audit of source read dependencies."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'experimental/welding_variable_open_v1'),
               str(ROOT/'experimental/welding_tool_exit_v1')]
from run_geometry import digest, atomic
from run_initial import assert_frozen_geometry_cache
from positioner import SourcePositioner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert_frozen_geometry_cache()
    parent_dir = ROOT/'build/welding_variable_open_v1'
    parent = json.loads((parent_dir/'report.json').read_text())
    seals = {}
    for row in parent['rows']:
        if not row['best']:
            continue
        name, expected = row['best']['path_file'], row['best']['path_sha256']
        actual = digest(parent_dir/name)
        if actual != expected or parent['output_sha256'][name] != expected:
            raise RuntimeError('Archived direction source path changed')
        seals[str((parent_dir/name).relative_to(ROOT))] = actual
    source_xml = ROOT/'source/psz/EngData/engData_ver2.xml'
    start_hash = digest(source_xml)
    scene = json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
    positioner = SourcePositioner(scene)
    products = sorted(a['id'] for a in scene['assets'] if a['kind']=='product')
    attached = sorted(positioner.attached_product_ids)
    if attached != products or digest(source_xml) != start_hash:
        raise RuntimeError('Attachment evidence mismatch or changed during audit')
    result = dict(archived_direction_source_path_sha256=seals,
                  attachment_xml_sha256=start_hash, source_xml_path=str(source_xml.relative_to(ROOT)),
                  products=products, attached_product_ids=attached,
                  positioner_source_metadata=positioner.metadata,
                  audit_script_sha256=digest(Path(__file__)),
                  scope='Post-run source provenance cross-check. All archived direction files match parent sealed identities; attachment XML matches all source products. V1 did not hash attachment XML or direction files at experiment start, so this audit does not retroactively prove their hashes throughout that interval.')
    atomic(args.out.resolve(), result)
    print('SOURCE_PROVENANCE_AUDIT', len(seals), 'archived paths matched;', len(products), 'product attachments matched')


if __name__ == '__main__':
    main()
