#!/usr/bin/env bash
# Launch CrazySim on the MuJoCo backend with the AI-deck camera stream and the obstacle scene.
# Extra args pass through to CrazySim's sitl_camera.sh (e.g. -m cf2x_L250, --sensor-noise).
# Override the scene with SCENE=scene_walls.xml. Close the MuJoCo window or Ctrl-C to stop.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

if [ ! -x "$FIRMWARE_DIR/sitl_make/build/cf2" ] || [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "CrazySim is not set up yet: run setup.cmd (Windows) or setup.sh (Linux) first." >&2
    exit 1
fi

# env.sh routes OpenGL to the GPU under WSL. The patched FPV camera renders on a worker thread; EGL contexts are thread-safe where GLFW
# windows are not. Only mujoco.Renderer reads this; the 3D viewer window keeps its own GLFW context.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# sitl_camera.sh calls bare `python3`, so it needs the venv on PATH. Unbuffered so logs/sim.log is live
# (launch.py watches it for the camera bridge's "Client connected").
source "$VENV_DIR/bin/activate"
export PYTHONUNBUFFERED=1
cd "$FIRMWARE_DIR"
# stdbuf line-buffers the firmware console, so sitl_make/build/0/out.log fills live instead of in 4 KB chunks.
exec stdbuf -oL -eL bash "$MUJOCO_DIR/launch/sitl_camera.sh" -s "${SCENE:-scene_obstacles.xml}" "$@"
