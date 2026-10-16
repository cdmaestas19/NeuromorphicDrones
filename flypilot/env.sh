# Shared paths for the flypilot scripts (sourced, not run). Export any of these first to override.
CRAZYSIM_DIR="${CRAZYSIM_DIR:-$HOME/CrazySim}"
CRAZYSIM_REF="${CRAZYSIM_REF:-3ec8b55da4bff887da542a9f314da825460e65be}"  # tested CrazySim commit (2026-04-03)
VENV_DIR="${VENV_DIR:-$HOME/.venvs/flybrain}"

FIRMWARE_DIR="$CRAZYSIM_DIR/crazyflie-firmware"
MUJOCO_DIR="$FIRMWARE_DIR/tools/crazyflie-simulation/simulator_files/mujoco"
