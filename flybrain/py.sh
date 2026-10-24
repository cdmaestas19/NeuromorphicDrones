#!/usr/bin/env bash
# Run a flybrain script in the brain's own Python environment: bash py.sh <script.py> [args]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../flypilot/env.sh"
[ -x "$SNN_VENV_DIR/bin/python" ] || { echo "no brain environment at $SNN_VENV_DIR: run flypilot\\setup.cmd first" >&2; exit 1; }
cd "$HERE"
exec "$SNN_VENV_DIR/bin/python" "$@"
