#!/usr/bin/env bash
# Re-runnable setup inside WSL/Linux: CrazySim checkout, Python venv, SITL firmware build.
# Needs the packages from system_deps.sh first.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

for tool in git cmake make g++ pkg-config; do
    command -v "$tool" >/dev/null || { echo "missing '$tool': run system_deps.sh as root first" >&2; exit 1; }
done
python3 -c 'import ensurepip' 2>/dev/null || { echo "missing python3-venv: run system_deps.sh as root first" >&2; exit 1; }

echo "== CrazySim -> $CRAZYSIM_DIR @ ${CRAZYSIM_REF:0:7}"
if [ ! -d "$CRAZYSIM_DIR/.git" ]; then
    git clone https://github.com/gtfactslab/CrazySim.git "$CRAZYSIM_DIR"
fi
if ! git -C "$CRAZYSIM_DIR" cat-file -e "$CRAZYSIM_REF^{commit}" 2>/dev/null; then
    git -C "$CRAZYSIM_DIR" fetch origin
fi
git -C "$CRAZYSIM_DIR" -c advice.detachedHead=false checkout --quiet "$CRAZYSIM_REF"
# Local fixes live in patches/<submodule>/. Undo them so the pinned submodule checkout is clean,
# then re-apply, which keeps this script safe to re-run. If an older version of a patch is applied
# (it can't be reversed), the files the patch touches are restored to the pinned version instead.
PATCH_TARGETS=(
    "crazyflie-firmware:$FIRMWARE_DIR"
    "crazyflie-simulation:$FIRMWARE_DIR/tools/crazyflie-simulation"
)
for t in "${PATCH_TARGETS[@]}"; do
    repo="${t#*:}"
    [ -e "$repo/.git" ] || continue
    for p in "$HERE/patches/${t%%:*}"/*.patch; do
        if git -C "$repo" apply --reverse --check "$p" 2>/dev/null; then
            git -C "$repo" apply --reverse "$p"
        else
            files=$(git -C "$repo" apply --numstat "$p" | cut -f3)
            if ! git -C "$repo" diff --quiet -- $files; then
                echo "   restoring $files (an older flypilot patch was applied)"
                git -C "$repo" checkout -- $files
            fi
        fi
    done
done
# Only the firmware submodule is needed; crazyswarm2 (ROS 2) is skipped.
git -C "$CRAZYSIM_DIR" submodule update --init --recursive crazyflie-firmware

echo "== Patching CrazySim (flypilot/patches)"
for t in "${PATCH_TARGETS[@]}"; do
    for p in "$HERE/patches/${t%%:*}"/*.patch; do
        echo "   ${t%%:*}/$(basename "$p")"
        git -C "${t#*:}" apply "$p"
    done
done

echo "== Python venv -> $VENV_DIR"
[ -x "$VENV_DIR/bin/python" ] || python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install -r "$HERE/requirements.txt"

echo "== SITL firmware"
mkdir -p "$FIRMWARE_DIR/sitl_make/build"
cd "$FIRMWARE_DIR/sitl_make/build"
cmake ..  # the venv is active, so the build's bare `python` call resolves
# Only the firmware binary: `make all` also builds the Gazebo plugin, which fails without Gazebo installed.
make -j"$(nproc)" cf2

echo "== Fly brain (flybrain/setup_brain.sh)"
bash "$HERE/../flybrain/setup_brain.sh"

echo
echo "Setup complete. Next: launch.cmd (Windows) or 'bash py.sh launch.py' (Linux); see run.txt."
