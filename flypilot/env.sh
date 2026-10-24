# Shared paths for the flypilot scripts (sourced, not run). Export any of these first to override.
CRAZYSIM_DIR="${CRAZYSIM_DIR:-$HOME/CrazySim}"
CRAZYSIM_REF="${CRAZYSIM_REF:-3ec8b55da4bff887da542a9f314da825460e65be}"  # tested CrazySim commit (2026-04-03)
VENV_DIR="${VENV_DIR:-$HOME/.venvs/flybrain}"

FIRMWARE_DIR="$CRAZYSIM_DIR/crazyflie-firmware"
MUJOCO_DIR="$FIRMWARE_DIR/tools/crazyflie-simulation/simulator_files/mujoco"

# The fly-brain side (flybrain/): the whole-brain LIF model with its FlyWire connectome data, the FlyWire
# cell-type annotations, and a Python environment of its own (Brian2).
export BRAIN_MODEL_DIR="${BRAIN_MODEL_DIR:-$HOME/Drosophila_brain_model}"
BRAIN_MODEL_REF="${BRAIN_MODEL_REF:-1503efb4dcdbb245c6e28f05094f4ffa2e55eef5}"  # sazulai/Drosophila_brain_model (2026-09-14)
export ANNOTATIONS_DIR="${ANNOTATIONS_DIR:-$HOME/flywire_annotations}"
ANNOTATIONS_REF="${ANNOTATIONS_REF:-a83b2776d60d5764cef36b927f5f9679c16c47a2}"  # flyconnectome/flywire_annotations
SNN_VENV_DIR="${SNN_VENV_DIR:-$HOME/.venvs/flybrain-snn}"
# Training the camera's person / vehicle classifier (camera_train.sh): PyTorch, in its own environment.
VISION_VENV_DIR="${VISION_VENV_DIR:-$HOME/.venvs/flybrain-vision}"

# Under WSL, Mesa defaults to the llvmpipe CPU renderer, which drags the sim to ~2% of real time
# (and the firmware with it). Route OpenGL to the Windows GPU through Mesa's D3D12 driver instead.
if [ -e /usr/lib/wsl/lib/libd3d12.so ]; then
    export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
fi
