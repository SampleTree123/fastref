"""Workspace location helpers; no mutation of historical receipts or identities.

Owner: experiments/data_preparation/notes/contact_sheet_package_merge_20261102.md. The relocation registry
maps old project-local locations to their current locations only at file access.
Paths outside this project are never redirected.
"""
from functools import lru_cache
import json
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'configs').is_dir() and (p / 'README.md').is_file())


@lru_cache(maxsize=1)
def registry():
    path = ROOT / 'runtime/path_relocations.json'
    if not path.exists():
        return {'schema': 'ltx_project_path_relocations_v1', 'paths': {}, 'scripts': {}}
    data = json.loads(path.read_text())
    if data.get('schema') != 'ltx_project_path_relocations_v1':
        raise ValueError('Unknown path relocation registry')
    for old, new in data['paths'].items():
        if any(Path(p).is_absolute() or '..' in Path(p).parts for p in (old, new)):
            raise ValueError('Relocation paths must remain inside the project')
    return data


@lru_cache(maxsize=1)
def _relocation_prefixes():
    # The registry is fixed for this process. Avoid sorting and reconstructing
    # the same prefix Paths for every cache receipt or media lookup.
    return tuple((Path(old).as_posix(), Path(new)) for old, new in
                 sorted(registry()['paths'].items(), key=lambda item: len(Path(item[0]).parts), reverse=True))


@lru_cache(maxsize=65536)
def resolve_path(value):
    """Resolve a project path without changing its record; cache within this process.

    Directory relocation happens while workers are stopped. Start a new process
    after a relocation instead of reusing pre-move resolution results.
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    try:
        relative = path.relative_to(ROOT)
    except ValueError:
        return path
    relative_text = relative.as_posix()
    for old, new in _relocation_prefixes():
        # Preserve directory boundaries and longest-prefix ordering without
        # constructing Paths and raising hundreds of exceptions per cache file.
        if relative_text == old:
            return ROOT / new
        if relative_text.startswith(old + '/'):
            return ROOT / new / relative_text[len(old) + 1:]
    return path


def script_path(name):
    if Path(name).name != name:
        raise ValueError('Script name must be a basename')
    relative = registry().get('scripts', {}).get(name)
    if relative:
        return ROOT / relative
    for directory in ('common', 'data', 'training', 'inference', 'validation', 'setup', 'workspace', ''):
        path = ROOT / 'scripts' / directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(name)


def add_script_paths():
    """Preserve shared imports while task entry files live in separate folders."""
    for directory in reversed(('common', 'data', 'training', 'inference', 'validation', 'setup', 'workspace')):
        path = ROOT / 'scripts' / directory
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


add_script_paths()


def verify_recorded_code(expected_sha256, current_path):
    """Check unchanged code, or one explicitly recorded path-only relocation.

    Historical receipts keep their original digest. A relocation is separate
    evidence: the exact old file is archived, and the exact current file must
    match the reviewed transition. Later source edits are not implicitly allowed.
    """
    import hashlib
    current = resolve_path(current_path)
    actual = hashlib.sha256(current.read_bytes()).hexdigest()
    if actual == expected_sha256:
        return False
    entry = registry().get('code_relocations', {}).get(current.name)
    if not isinstance(entry, dict) or entry.get('scope') != 'path_and_import_relocation_only':
        raise ValueError('Code differs without an approved path relocation: ' + current.name)
    if entry.get('old_sha256') != expected_sha256 or entry.get('new_sha256') != actual:
        raise ValueError('Recorded code relocation does not match old/current SHA: ' + current.name)
    relative = Path(entry['archive_path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Code archive must use a project-relative path')
    archive = resolve_path(relative)
    if not archive.is_relative_to(ROOT) or archive == current:
        raise ValueError('Invalid original-code archive location')
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('Archived original code SHA mismatch: ' + current.name)
    return True


def resolve_cli_paths(namespace):
    """Normalize execution addresses only, never loaded identity dictionaries."""
    for name, value in vars(namespace).items():
        if isinstance(value, Path):
            setattr(namespace, name, resolve_path(value))
        elif isinstance(value, list) and any(isinstance(item, Path) for item in value):
            setattr(namespace, name, [resolve_path(item) if isinstance(item, Path) else item for item in value])
    return namespace


def main():
    """Use the same path registry from Bash entrypoints, without loading models."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('resolve', 'script', 'root'))
    parser.add_argument('value', nargs='?')
    args = parser.parse_args()
    if args.operation != 'root' and args.value is None:
        parser.error('resolve/script requires one path or script basename')
    if args.operation == 'root' and args.value is not None:
        parser.error('root does not take a value')
    print(ROOT if args.operation == 'root' else
          resolve_path(args.value) if args.operation == 'resolve' else script_path(args.value))


if __name__ == '__main__':
    main()
