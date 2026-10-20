#!/usr/bin/env bash
# Kill every CrazySim process (firmware, simulator, camera bridge) and any flypilot viewers left behind
# (dashboard, camera view). Safe to run any time.
pkill -x cf2 && echo "stopped firmware (cf2)" || true
pkill -f '/crazysim_cpx\.py' && echo "stopped camera bridge" || true
pkill -f '/crazysim\.py' && echo "stopped simulator" || true
pkill -f 'python.* dashboard\.py' && echo "stopped dashboard" || true
pkill -f 'python.* camera_view\.py' && echo "stopped camera view" || true
rm -f /dev/shm/crazysim_overlay.rgb /dev/shm/crazysim_overlay.rgb.tmp
echo "CrazySim stopped."
