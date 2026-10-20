"""Baseline flight: take off, fly the "square" route through open floor on planned paths, land. No avoidance.

Every flight plans its path (planner.py). This one has no avoidance layer to fall back on, so it always
plans on the full map, whatever the route's map says.

Run with launch.cmd waypoints (starts the sim too), or with the sim already up: bash py.sh fly_waypoints.py
Ctrl-C lands the drone where it is; a second Ctrl-C cuts the motors.
"""
import logging
import math
import sys
import time

from flight import GEOFENCE, Pilot, SafetyStop, fly_mission, setup_logging
from planner import Planner, load_obstacles
from routes import ALTITUDE_M, ROUTES

log = logging.getLogger('flypilot')

ROUTE = ROUTES['square']


def mission(pilot: Pilot):
    planner = Planner(load_obstacles(), GEOFENCE, ALTITUDE_M, 'all')
    s = pilot.tm.state()
    waypoints = ROUTE.stops + [(s.x, s.y)]  # ...and back to where it took off
    pilot.takeoff(ALTITUDE_M)
    pilot.hover(1.0)
    for i, (x, y) in enumerate(waypoints, 1):
        t0 = time.monotonic()
        s = pilot.tm.state()
        path = planner.plan((s.x, s.y), (x, y))
        if path is None:
            raise SafetyStop(f'no path to waypoint {i} ({x:.1f}, {y:.1f})')
        for px, py in path[1:]:
            pilot.goto(px, py, ALTITUDE_M)
        s = pilot.tm.state()
        nearest = min((d for d in s.ranges.items() if d[0] != 'up'), key=lambda d: d[1])
        clearance = f'{nearest[1]:.2f} m ({nearest[0]})' if math.isfinite(nearest[1]) else 'none within 4 m'
        log.info('waypoint %d/%d (%.1f, %.1f) reached in %.1f s; nearest obstacle %s',
                 i, len(waypoints), x, y, time.monotonic() - t0, clearance)
        pilot.hover(1.0)
    pilot.land()


if __name__ == '__main__':
    setup_logging()
    sys.exit(fly_mission(mission))
