#!/usr/bin/env python3
"""Save or verify the exact local code/config/data used by a formal run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from queue_client import atomic_json

ROOT = Path(__file__).resolve().parents[2]
LTX = ROOT.parent / 'ltx2.5'


def files():
    paths = list((ROOT / 'scripts/editing_agent').glob('*.py'))
    paths += list((ROOT / 'scripts/editing_agent').glob('*.sbatch'))
    paths += [ROOT / 'scripts/run_gpu_python.sh']
    paths += list((ROOT / 'scripts/evaluation').glob('*.py'))
    for directory in ['configs/editing_agent', 'datasets/editing_agent_v1']:
        paths += [p for p in (ROOT / directory).rglob('*') if p.is_file()]
    paths += [ROOT / 'configs/evaluation/actual_id/high.json', ROOT / 'configs/model_revision.json']
    paths += [ROOT / 'reports/weights_verified.json']
    paths += list((ROOT / 'models/Qwen3.8-27B').glob('*.json'))
    paths += list((ROOT / 'models/Qwen3.8-27B').glob('*.jinja'))
    for directory in ['scripts/common', 'scripts/inference']:
        paths += list((LTX / directory).glob('*.py'))
    paths += list((LTX / 'scripts/common').glob('*.sh'))
    paths += list((LTX / 'runtime/repos/LTX-2/packages').rglob('*.py'))
    paths += [LTX / 'runtime/models/native_model_revision.json',
              LTX / 'runtime/verification/native_weights_verified.json',
              LTX / 'runtime/verification/trainer_environment.json',
              LTX / 'configs/data/synthetic_original_resolution/media.json']
    paths += [LTX / 'scripts/common/run_trainer_python.sh', LTX / 'configs/training/outfit_camera_first_frame_sequential_time_visual_dropout_768/train.yaml']
    return sorted(set(p for p in paths if '__pycache__' not in p.parts))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    manifest = a.output / 'freeze_manifest.json'
    current = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files()}
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old['files'] != current:
            differences = sorted(path for path in set(old['files']) | set(current) if old['files'].get(path) != current.get(path))
            raise ValueError('Frozen input/code changed; use a new formal version: ' + repr(differences))
        print('Frozen code, configurations and input lists match.')
        return
    if a.verify:
        raise ValueError('No freeze_manifest.json: freeze only after development validation')
    snapshot = a.output / 'frozen_sources'
    for name in current:
        source = Path(name)
        target = snapshot / source.relative_to(ROOT.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    atomic_json(manifest, {'files': current, 'snapshot': str(snapshot.resolve()),
                          'note': 'Workers use verified local files. Frozen copies retain reproducibility; weights and original media remain referenced.'})
    print('Saved', len(current), 'code/config/input files.')


if __name__ == '__main__':
    main()
