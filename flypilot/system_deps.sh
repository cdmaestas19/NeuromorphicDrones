#!/usr/bin/env bash
# System packages for building the CrazySim SITL firmware and opening MuJoCo/OpenCV windows.
# Run as root: `sudo bash system_deps.sh` (setup.cmd does this for you via `wsl -u root`).
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "run as root: sudo bash $0" >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    git ca-certificates cmake build-essential pkg-config \
    python3 python3-venv \
    libgl1 libegl1 libgl1-mesa-dri libglib2.0-0 \
    libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6 libxkbcommon-x11-0 \
    mesa-utils
