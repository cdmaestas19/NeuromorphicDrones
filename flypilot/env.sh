# Shared paths for the flypilot scripts (sourced, not run). Export any of these first to override.
CRAZYSIM_DIR="${CRAZYSIM_DIR:-$HOME/CrazySim}"
CRAZYSIM_REF="${CRAZYSIM_REF:-3ec8b55da4bff887da542a9f314da825460e65be}"  # tested CrazySim commit (2026-04-03)
VENV_DIR="${VENV_DIR:-$HOME/.venvs/flybrain}"

FIRMWARE_DIR="$CRAZYSIM_DIR/crazyflie-firmware"
MUJOCO_DIR="$FIRMWARE_DIR/tools/crazyflie-simulation/simulator_files/mujoco"

# Under WSL, Mesa defaults to the llvmpipe CPU renderer, which drags the sim to ~2% of real time
# (and the firmware with it). Route OpenGL to the Windows GPU through Mesa's D3D12 driver instead.
if [ -e /usr/lib/wsl/lib/libd3d12.so ]; then
    export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
fi
