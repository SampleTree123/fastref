#!/usr/bin/env bash
# Compatibility alias after Diffusers retirement; owner: README.md.
# All preparation and training now use the isolated official trainer environment.
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$script_dir/run_trainer_python.sh" "$@"
