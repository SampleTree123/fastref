#!/usr/bin/env python3
"""Snapshot dataset, evaluator and config before submission; existing freezes verify."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review-flags', type=Path, default=ROOT / 'reports/history_selection_v3/independent_review_flags.json')
    args = parser.parse_args()
    output = args.output.resolve()
    frozen = output / 'frozen'
    manifest_path = output / 'freeze_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['source_dataset'] != str(args.dataset.resolve()):
            raise ValueError('This output already pins another dataset')
        for name, digest in manifest['sha256'].items():
            if sha(frozen / name) != digest:
                raise ValueError('Frozen file changed: ' + name)
        print(json.dumps({'status': 'verified_existing_freeze', 'manifest': str(manifest_path)}))
        return
    if frozen.exists():
        raise ValueError('Uncommitted partial freeze exists; inspect it before retrying')
    frozen.mkdir(parents=True)
    for relative in ('scripts/evaluation', 'configs/evaluation'):
        shutil.copytree(ROOT / relative, frozen / relative, ignore=shutil.ignore_patterns('__pycache__'))
    for relative in ('scripts/run_gpu_python.sh', 'configs/model_revision.json'):
        target = frozen / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    target = frozen / 'datasets/history_selection_v3/eval.jsonl'
    target.parent.mkdir(parents=True)
    shutil.copy2(args.dataset, target)
    review_source = args.review_flags.resolve() if args.review_flags.exists() else None
    if review_source is not None:
        shutil.copy2(review_source, target.parent / 'independent_review_flags.json')
    for name in ('models', 'envs', 'cache', 'reports'):
        (frozen / name).symlink_to(ROOT / name, target_is_directory=True)
    paths = [p for p in frozen.rglob('*') if p.is_file() and not p.is_symlink()]
    # rglob does not recurse through the large shared-model/cache symlinks.
    manifest = {'schema_version': 1, 'selection_contract': 'actual_id_v1',
                'source_dataset': str(args.dataset.resolve()), 'frozen_root': str(frozen),
                'source_review_flags': str(review_source) if review_source is not None else None,
                'shared_runtime': ['models', 'envs', 'cache', 'reports'],
                'sha256': {str(p.relative_to(frozen)): sha(p) for p in sorted(paths)}}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': 'created', 'manifest': str(manifest_path), 'files': len(paths)}))


if __name__ == '__main__':
    main()
