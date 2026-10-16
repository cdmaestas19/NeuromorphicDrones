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

# sitl_camera.sh calls bare `python3`, so it needs the venv on PATH.
source "$VENV_DIR/bin/activate"
cd "$FIRMWARE_DIR"
exec bash "$MUJOCO_DIR/launch/sitl_camera.sh" -s "${SCENE:-scene_obstacles.xml}" "$@"
