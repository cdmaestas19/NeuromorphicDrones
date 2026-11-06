#!/usr/bin/env bash
# Kill every CrazySim process (firmware, simulator, camera bridge) and any flypilot viewers left behind
# (dashboard, camera view). Safe to run any time. A video being drawn after a flight (crazysim.py
# --render-video, launch.cmd --video) is left to finish.
pkill -x cf2 && echo "stopped firmware (cf2)" || true
pkill -f '/crazysim_cpx\.py' && echo "stopped camera bridge" || true
stopped=0
for pid in $(pgrep -f '/crazysim\.py'); do
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q -- '--render-video' && continue
    kill "$pid" 2>/dev/null && stopped=1
done
[ "$stopped" = 1 ] && echo "stopped simulator"
pkill -f 'python.* dashboard\.py' && echo "stopped dashboard" || true
pkill -f 'python.* camera_view\.py' && echo "stopped camera view" || true
pkill -f 'python.* brain_server\.py' && echo "stopped fly brain" || true
rm -f /dev/shm/crazysim_overlay.rgb /dev/shm/crazysim_overlay.rgb.tmp
echo "CrazySim stopped."
