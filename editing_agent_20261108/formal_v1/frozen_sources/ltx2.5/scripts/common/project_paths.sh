#!/usr/bin/env bash
# Shared workspace paths. Owner: experiments/data_preparation/notes/contact_sheet_package_merge_20261102.md.
# Source this file; resolve paths through the Python registry, never duplicate it.
LTX_SCRIPT_HELPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$LTX_SCRIPT_HELPER_DIR"
while [[ ! -d "$ROOT/configs" || ! -f "$ROOT/README.md" ]]; do
  if [[ "$ROOT" == / ]]; then
    printf '%s\n' 'Cannot find the LTX project root.' >&2
    return 1 2>/dev/null || exit 1
  fi
  ROOT="$(dirname -- "$ROOT")"
done
LTX_LAYOUT_HELPER="$LTX_SCRIPT_HELPER_DIR/project_layout.py"
[[ -f "$LTX_LAYOUT_HELPER" ]] || { printf '%s\n' 'Missing project_layout.py beside project_paths.sh.' >&2; return 1 2>/dev/null || exit 1; }
# The helper uses only the Python standard library. This intentionally precedes
# selecting the isolated trainer interpreter and does not import user packages.
LTX_LAYOUT_PYTHON="$(command -v python3)"
ltx_path() { "$LTX_LAYOUT_PYTHON" -I "$LTX_LAYOUT_HELPER" resolve "$1"; }
ltx_script() { "$LTX_LAYOUT_PYTHON" -I "$LTX_LAYOUT_HELPER" script "$1"; }
