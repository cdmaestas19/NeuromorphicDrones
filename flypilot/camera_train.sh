#!/usr/bin/env bash
# Train the camera's person / vehicle classifier on recorded flights (vision_train.py; camera_train.cmd
# runs this). The first run creates its Python environment (PyTorch, CPU build, ~200 MB).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
if [ ! -x "$VISION_VENV_DIR/bin/python" ] || ! "$VISION_VENV_DIR/bin/python" -c 'import torch' 2>/dev/null; then
    echo "== Camera-classifier Python environment -> $VISION_VENV_DIR (first run only)"
    [ -x "$VISION_VENV_DIR/bin/python" ] || python3 -m venv "$VISION_VENV_DIR"
    "$VISION_VENV_DIR/bin/python" -m pip install --quiet --upgrade pip
    "$VISION_VENV_DIR/bin/python" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
    "$VISION_VENV_DIR/bin/python" -m pip install --quiet -r "$HERE/vision-requirements.txt"
fi
cd "$HERE"
exec "$VISION_VENV_DIR/bin/python" vision_train.py "$@"
