#!/usr/bin/env bash
# Run a flypilot Python script with the project venv: bash py.sh camera_view.py [args...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "Python venv missing: run setup.cmd (Windows) or setup.sh (Linux) first." >&2
    exit 1
fi
cd "$HERE"
exec "$VENV_DIR/bin/python" "$@"
