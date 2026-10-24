#!/usr/bin/env bash
# Fly-brain side of the setup (flypilot/setup.sh runs it at the end; safe to re-run):
#   - the whole-brain LIF model of Shiu et al. 2024 with its FlyWire connectome data (~200 MB)
#   - the FlyWire cell-type annotations (Schlegel et al. 2024), to find neurons by name
#   - a Python environment of its own with Brian2 (the flight stack's venv stays separate)
# Both repos are pinned to the commits in flypilot/env.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../flypilot/env.sh"

python3 -c 'import os, sysconfig; assert os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h"))' \
    2>/dev/null || { echo "missing python3-dev (Brian2 compiles C++): run flypilot/system_deps.sh as root first" >&2; exit 1; }

fetch() {  # url ref dir: a shallow checkout of exactly that commit
    local url=$1 ref=$2 dir=$3
    if [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null)" = "$ref" ]; then
        return
    fi
    if [ ! -d "$dir/.git" ]; then
        git init -q "$dir"
        git -C "$dir" remote add origin "$url"
    fi
    git -C "$dir" fetch -q --depth 1 origin "$ref"
    git -C "$dir" -c advice.detachedHead=false checkout -q FETCH_HEAD
}

echo "== Brain model -> $BRAIN_MODEL_DIR @ ${BRAIN_MODEL_REF:0:7}"
fetch https://github.com/sazulai/Drosophila_brain_model.git "$BRAIN_MODEL_REF" "$BRAIN_MODEL_DIR"
echo "== FlyWire annotations -> $ANNOTATIONS_DIR @ ${ANNOTATIONS_REF:0:7}"
fetch https://github.com/flyconnectome/flywire_annotations.git "$ANNOTATIONS_REF" "$ANNOTATIONS_DIR"

echo "== Brain Python environment -> $SNN_VENV_DIR"
[ -x "$SNN_VENV_DIR/bin/python" ] || python3 -m venv "$SNN_VENV_DIR"
"$SNN_VENV_DIR/bin/python" -m pip install --quiet --upgrade pip
"$SNN_VENV_DIR/bin/python" -m pip install --quiet -r "$HERE/requirements.txt"
"$SNN_VENV_DIR/bin/python" - <<'EOF'
import brian2, numpy
print(f'   brian2 {brian2.__version__}, numpy {numpy.__version__}')
EOF
echo "Brain setup complete. Check it with: flypilot\\brain_reproduce.cmd (see flybrain/README.md)"
