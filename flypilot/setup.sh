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
# Only the firmware submodule is needed; crazyswarm2 (ROS 2) is skipped.
git -C "$CRAZYSIM_DIR" submodule update --init --recursive crazyflie-firmware

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

echo
echo "Setup complete. Start the sim with run_sim.cmd (Windows) or 'bash run_sim.sh' (Linux)."
