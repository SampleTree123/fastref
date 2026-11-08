#!/usr/bin/env python3
"""Verify the four-group campaign, reuse empty-reference outputs, export its page.

Experimental owner: experiments/inference/final_camera_comparison_20261108/README.md.
No model invocation or external publication. The final fastref transfer is separate.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from project_layout import ROOT, add_script_paths
add_script_paths()
from build_fixed_case_gallery import checked_comparison, read_cases
from run_fixed_cases_parallel import collect_results, sha_file, write_status
from export_guidance_gallery import export_folder, referenced_media


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    folder = args.campaign.resolve().parent
    plan = json.loads(args.campaign.read_text())
    state = json.loads((folder / 'execution_state.json').read_text())
    if state['status'] != 'generation_completed_pending_reuse_merge_and_gallery':
        raise ValueError('Wait for all four generation groups to finish')
    runs = {r['name']: r for r in plan['runs']}
    all_cases, rows, _ = read_cases(Path(plan['runs'][0]['bundle']), Path(plan['cases']))
    source = runs[plan['reuse_no_reference_from']]
    target = runs[plan['reuse_no_reference_to']]
    source_path, target_path = Path(source['comparison']), Path(target['comparison'])
    source_value = json.loads(source_path.read_text())
    target_value = json.loads(target_path.read_text())
    selected = plan['no_optional_reference_uids']
    if selected != [c['uid'] for c in all_cases if not rows[c['uid']][0]['references']]:
        raise ValueError('Empty-reference selection changed')
    source_results, _, _ = checked_comparison(source_path, 'step768', all_cases,
                                               Path(source['bundle']), 768)
    for field in ('seed', 'steps', 'cfg', 'bundle', 'bundle_sha256', 'config_sha256',
                  'checkpoint', 'checkpoint_sha256'):
        if source_value['comparison_contract'][field] != target_value['comparison_contract'][field]:
            raise ValueError('Reuse inputs differ: ' + field)
    for field in ('cfg_scale', 'stg_scale', 'guidance_rescale', 'negative_prompt',
                  'stg_blocks', 'guidance_batch_size'):
        if source['guidance'][field] != target['guidance'][field]:
            raise ValueError('Reuse guidance differs: ' + field)
    if source['guidance'].get('ref_cfg', 1) != 1 or target['guidance'].get('ref_cfg') != 3:
        raise ValueError('Unexpected reuse comparison')
    if not target_value.get('empty_reference_reuse'):
        subset = [c for c in all_cases if c['uid'] not in selected]
        checked_comparison(target_path, 'step768', subset, Path(target['bundle']), 768)
        write_status(target_path.with_name('generated_subset_comparison.json'), target_value)
        reuse = target_path.parent / 'reused_without_optional_reference'
        reuse.mkdir(exist_ok=False)
        donor = json.loads((Path(source_value['jobs'][0]['output']) / 'generation.json').read_text())
        donor.update(target_value['comparison_contract'])
        donor.update(status='completed', sample_uids=selected,
                     sample_splits={c['uid']: c.get('split', 'development') for c in all_cases if c['uid'] in selected},
                     results=[], generation_mode='reused_equivalent_no_optional_reference',
                     effective_guidance=source['guidance'], reused_from=str(source_path),
                     started_unix=time.time(), finished_unix=time.time())
        for i, uid in enumerate(selected):
            entry = deepcopy(source_results[uid])
            original = source_path.parent / entry['output']
            dest = reuse / f'case_{i:02d}.mp4'
            if sha_file(original) != entry['sha256']:
                raise ValueError('Donor video changed')
            os.link(original, dest)
            entry.update(output=dest.name, seconds=0,
                         original_generation_seconds=entry['seconds'],
                         original_denoise_peak_allocated_bytes=entry.get('denoise_peak_allocated_bytes'),
                         original_decode_peak_allocated_bytes=entry.get('decode_peak_allocated_bytes'),
                         denoise_peak_allocated_bytes=0, decode_peak_allocated_bytes=0,
                         reused_from=str(original), reuse_reason='same model and inputs; no optional reference, so reference CFG is inactive')
            donor['results'].append(entry)
        write_status(reuse / 'generation.json', donor)
        target_value['jobs'].append({'gpu': None, 'sample_uids': selected,
                                    'sample_splits': donor['sample_splits'], 'output': str(reuse),
                                    'command': [], 'generation_mode': donor['generation_mode']})
        target_value['cases'] = all_cases
        target_value['results'] = collect_results(target_value['jobs'], target_path.parent,
                                                  target_value['comparison_contract'])
        target_value['empty_reference_reuse'] = {'count': len(selected), 'from': str(source_path), 'uids': selected}
        write_status(target_path, target_value)
    # Validate every displayed group using its own text CFG, including baseline CFG=1.
    result_map = []
    for run in plan['runs']:
        cases = [{**c, 'cfg_scale': run['guidance']['cfg_scale']} for c in all_cases]
        path = Path(run['comparison'])
        results, _, _ = checked_comparison(path, 'step768', cases, Path(run['bundle']), 768)
        for case in cases:
            entry = results[case['uid']]
            result_map.append({'group': run['name'], 'label': run['label'], 'uid': case['uid'],
                               'split': case.get('split', 'development'), 'checkpoint': run['checkpoint'],
                               'guidance': run['guidance'], 'sha256': entry['sha256'], 'bytes': entry['bytes'],
                               'display_media': 'media/' + entry['sha256'] + '.mp4',
                               'reused': bool(entry.get('reused_from'))})
    if len(result_map) != 108:
        raise ValueError('Expected 27 cases x 4 groups')
    manifest = {**plan, 'guidance': plan['runs'][0]['guidance']}
    manifest_path = folder / 'gallery_manifest.json'
    write_status(manifest_path, manifest)
    subprocess.run(['bash', str(ROOT / 'scripts/common/run_trainer_python.sh'),
                    str(ROOT / 'scripts/inference/build_experiment_comparison.py'),
                    '--manifest', str(manifest_path), '--output', str(folder / 'gallery')], check=True)
    result = export_folder(folder / 'gallery', folder / 'display')
    display = folder / 'display'
    for row in result_map:
        path = display / row['display_media']
        if path.stat().st_size != row['bytes'] or sha_file(path) != row['sha256']:
            raise ValueError('Exported video differs from generation result')
    write_status(folder / 'display_results.json', result_map)
    write_status(folder / 'export_audit.json', {'status': 'passed', 'cases': 27, 'groups': 4,
                 'results': 108, 'new_generations': 100, 'equivalent_reuse': 8,
                 'export': result, 'embedded_media_files': len(referenced_media((display / 'index.html').read_text()))})
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
