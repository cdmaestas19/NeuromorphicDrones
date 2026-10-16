#!/usr/bin/env bash
# Kill every CrazySim process (firmware, simulator, camera bridge). Safe to run any time.
pkill -x cf2 && echo "stopped firmware (cf2)" || true
pkill -f '/crazysim_cpx\.py' && echo "stopped camera bridge" || true
pkill -f '/crazysim\.py' && echo "stopped simulator" || true
echo "CrazySim stopped."
