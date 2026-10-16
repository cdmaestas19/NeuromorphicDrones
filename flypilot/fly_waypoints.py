"""Baseline flight: take off, fly a short waypoint route through open floor, land. No avoidance.

Run with launch.cmd waypoints (starts the sim too), or with the sim already up: bash py.sh fly_waypoints.py
Ctrl-C lands the drone where it is; a second Ctrl-C cuts the motors.
"""
import logging
import math
import sys
import time

from flight import Pilot, fly_mission, setup_logging

log = logging.getLogger('flypilot')

ALTITUDE_M = 0.5
# Room frame of scene_obstacles.xml, metres. Every leg keeps >= 0.45 m from the nearest obstacle.
ROUTE = [(1.0, 0.0), (1.0, -0.8), (0.0, -0.8), (0.0, 0.0)]


def mission(pilot: Pilot):
    pilot.takeoff(ALTITUDE_M)
    pilot.hover(1.0)
    for i, (x, y) in enumerate(ROUTE, 1):
        t0 = time.monotonic()
        pilot.goto(x, y, ALTITUDE_M)
        s = pilot.tm.state()
        nearest = min((d for d in s.ranges.items() if d[0] != 'up'), key=lambda d: d[1])
        clearance = f'{nearest[1]:.2f} m ({nearest[0]})' if math.isfinite(nearest[1]) else 'none within 4 m'
        log.info('waypoint %d/%d (%.1f, %.1f) reached in %.1f s; nearest obstacle %s',
                 i, len(ROUTE), x, y, time.monotonic() - t0, clearance)
        pilot.hover(1.0)
    pilot.land()


if __name__ == '__main__':
    setup_logging()
    sys.exit(fly_mission(mission))
