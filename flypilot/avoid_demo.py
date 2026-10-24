"""Flight to a list of stops with path planning and obstacle avoidance.

Every flight is stops on an OrderBook (orders.py): a test route from routes.py, stops given with --at,
or stops added during the flight. Each leg follows a path planned on a map (planner.py) and is
re-planned after every dodge. Obstacles that are not on the map are left to the avoidance layer
(avoidance.py): a policy chosen by name (policies.py) turns sensors into an avoid signal and a dodge,
and the supervisor's state machine decides when to leave the path and dodge. Every control tick is
streamed to dashboard.py.

    launch.cmd avoid                     route "blocked" (crate3 not on the map: avoidance must act)
    launch.cmd avoid --route square      everything mapped, open floor: avoidance should never trigger
    launch.cmd avoid --record recordings/blocked.jsonl
                                         also save every decision's inputs and outputs for bench.py
    launch.cmd deliver                   (= --route live) no stops yet: add them in the Stops window or by
                                         double-clicking the floor
    launch.cmd deliver --at 2.5,0 --at 2.5,-0.9 --return
                                         stops known at take-off; return home and land after the last
    launch.cmd deliver --map building    the planner doesn't know the crates: the avoidance must handle them
    launch.cmd avoid --route headon      moving obstacles (street scene): headon, hover, crossing, cart, lane, dropoff
    launch.cmd deliver --scene street    live stops among a walking person and a driving cart
With the sim and the dashboard already running:  bash py.sh avoid_demo.py [--route R] [--policy P]
Ctrl-C lands the drone where it is; a second Ctrl-C cuts the motors.
"""
import argparse
import json
import logging
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np

from avoidance import (CLEAR_OUT_MPS, HORIZONTAL_RAYS, AvoidancePolicy, AvoidanceSupervisor, CostMeter, Mode, Sensors,
                       clear_out, make_sensors)
from dashboard import DashboardFeed
from eye import CollisionCourse, EyeSource, clear_below, clear_out_ring, points_near, ring
from eye import clearance as eye_clearance
from flight import Pilot, SafetyStop, fly_mission, setup_logging
from frames import FrameSource
from orders import CANCELLED, DELIVERED, SKIPPED, OrderBook, obstacle_gap
from movers import MoverRunner, load_movers
from planner import MAPS, SCENES, geofence, load_obstacles, path_length
from policies import POLICIES
from routes import ALTITUDE_M, LIVE_CRUISE_M, LIVE_MOVERS, ROUTES

log = logging.getLogger('flypilot')

CRUISE_MPS = 0.4
ARRIVE_M = 0.1
SLOWDOWN_PER_S = 1.0      # nominal speed = min(cruise, this x distance to the waypoint)
MAX_EPISODE_S = 20.0      # an avoidance episode longer than this counts as a failure
DASHBOARD_PERIOD_S = 0.1  # 10 Hz is all the panel redraws; JSON encoding is the costliest thing per tick
DROP_OFF_S = 1.0          # hover at each delivery stop (stands in for releasing the parcel)
# Planned paths: intermediate points are passed, not stopped at.
PASS_M = 0.15             # next path point once this close
CORNER_MIN_MPS = 0.1      # speed through a turn of 90 degrees or more
CORNER_SLOWDOWN_PER_S = 0.5  # approaching a corner: speed = corner speed + this x distance
FENCE_KEEP_M = 0.1        # the setpoint stays this far inside the geofence (= orders.FENCE_MARGIN_M)
MAP_TTC_S = 2.5           # never command a time-to-collision below this with a mapped obstacle
                          # (the policy's threat would be 1/2.5 = 0.4, under its 0.5 enter threshold)
MAP_CORRIDOR_M = 0.25     # ...also along lines this far either side of the path (policies.GeometricTTC360Policy
                          # counts what is within 0.3 m of the path)
DODGE_HALF_WIDTH_M = 0.15  # a dodge slows for what the eye sees within this of its line (drone 0.07 m + margin)...
DODGE_GAP_M = 0.2          # ...so as to stop this short of it...
DODGE_TTC_S = 0.8          # ...taking at least this long to get there
SEEN_RANGE_M = 1.2         # after an episode, static things the eye sees within this go on the planner's map
CLIMB_MPS = 1.0            # up to the route's altitude at most this fast (cruise: 4 m in ~5 s)...
DESCENT_MPS = 1.0          # ...down at most this fast,
DESCENT_SLOW_MPS = 0.4     # and this slow lower than SLOW_BELOW_M
SLOW_BELOW_M = 1.5
DESCENT_CLEAR_M = 0.3      # ...and down only while what is under the drone is this far below where it is going
FLOOR_M = 0.1              # (eye.clear_below); anything lower than this is the floor
MIN_Z_M = 0.3              # a dodge never takes the setpoint lower than this...
CEILING_KEEP_M = 0.4       # ...nor closer than this to the geofence's ceiling
CEILING_BRAKE_PER_S = 1.5  # a climb slows to this x the room left below that (0.6 m/s 0.4 m short of it)
ALT_TOL_M = 0.05           # at a cruise height or drop-off height once this close
LAND_HANDOFF_M = MIN_Z_M   # landing: the avoidance stays on down to this height, the pilot's touchdown takes over
# Careful landing (from cruise height, _come_down): first down to HOLD_M, above people's heads, to look at the
# drop-off point with the eye. Taken while anything within LZ_RADIUS_M of it, at least LZ_TALL_M high, isn't part
# of an obstacle in the scene file (a person, a parked cart, something new): then wait above it, and come down
# once it has been clear for LZ_CLEAR_S. Taken for LZ_MOVE_S (parked there): come down at the nearest clear
# spot on LZ_RINGS_M round it instead. On the way down, the descent pauses while the point is taken.
HOLD_M = 2.5
LZ_RADIUS_M = 0.7
LZ_TALL_M = 0.25
LZ_MIN_POINTS = 2
LZ_CLEAR_S = 1.0
LZ_MOVE_S = 6.0
LZ_RINGS_M = (0.9, 1.3, 1.8, 2.4)
LZ_MARGIN_M = 0.3          # ...clear this much further out (one 0.7 m from the cart's edge flickered taken / clear)
LZ_GIVE_UP_S = 40.0        # no clear spot for this long: skip the stop


def body_to_world(v_body: tuple, yaw_deg: float) -> tuple:
    yaw = yaw_deg * (math.pi / 180.0)
    c, s = math.cos(yaw), math.sin(yaw)
    return (v_body[0] * c - v_body[1] * s, v_body[0] * s + v_body[1] * c, v_body[2])


def _corner_speeds(path: list) -> list:
    """The speed to pass each intermediate path point at: lower the sharper the turn there."""
    out = [CRUISE_MPS] * len(path)
    for i in range(1, len(path) - 1):
        (ax, ay), (bx, by), (cx, cy) = path[i - 1], path[i], path[i + 1]
        turn = abs((math.atan2(cy - by, cx - bx) - math.atan2(by - ay, bx - ax) + math.pi) % (2 * math.pi) - math.pi)
        out[i] = max(CORNER_MIN_MPS, CRUISE_MPS * math.cos(turn)) if turn < math.pi / 2 else CORNER_MIN_MPS
    return out


def _timeout(path: list) -> float:
    return path_length(path) / CRUISE_MPS * 3.0 + MAX_EPISODE_S


class AvoidFlight:
    """Flies the stops of an OrderBook on planned paths, handing control to the policy's dodge whenever
    the supervisor says AVOID.

    Knows nothing about how a policy decides: it only reads AvoidCommand.signal and .dodge.
    """

    def __init__(self, route_name: str, policy: AvoidancePolicy, camera: FrameSource | None, feed: DashboardFeed,
                 orders: OrderBook, movers: MoverRunner, record: Path | None = None, return_when_done: bool = False,
                 dwell_s: float = DROP_OFF_S, scene: str = 'obstacles', eye: EyeSource | None = None,
                 cruise_m: float | None = None):
        self.route_name, self.scene = route_name, scene
        self.cruise_z = max(cruise_m or ALTITUDE_M, ALTITUDE_M)  # between stops; stops are flown at ALTITUDE_M
        self.alt_target = ALTITUDE_M                             # the altitude _step holds (outside AVOID)
        self._zone = None                                        # the landing spot a descent watches (_come_down)
        self.orders, self.movers, self.return_when_done, self.dwell_s = orders, movers, return_when_done, dwell_s
        self.policy, self.camera, self.feed, self.eye = policy, camera, feed, eye
        self._ring = None  # (azimuth relative to the heading, nearest distance) all round, from the eye
        self._objects, self._objects_reading = CollisionCourse(), None  # the things around, tracked
        self._below = math.inf  # how far down the eye sees room (eye.clear_below)
        self.supervisor = AvoidanceSupervisor()
        self.decision_cost, self.telemetry_cost = CostMeter(), CostMeter()
        self._cost_summary, self._cost_summary_t = {}, 0.0
        self._last_publish = 0.0
        self._record = None
        self._record_path, self._frames, self._frame_t, self._last_frame = record, [], [], None
        self._eye_log, self._last_eye = [], None
        if record is not None:
            record.parent.mkdir(parents=True, exist_ok=True)
            self._record = open(record, 'w')
        self.pilot: Pilot | None = None
        self.cmd = None
        self.v_cmd = (0.0, 0.0, 0.0)
        self.clearance = math.inf
        self.phase = 'connecting'
        self._replan = False
        self.t0 = time.monotonic()
        self._camera_counts = deque(maxlen=40)  # (time, frames received): ~2 s window for the fps readout

    def run(self, pilot: Pilot):
        self.pilot = pilot
        pilot.on_tick = self._on_tick
        try:
            st = pilot.tm.state()
            self.orders.home = (st.x, st.y)
            self.phase = 'taking off'
            pilot.takeoff(ALTITUDE_M)
            pilot.hover(1.0)
            self.policy.reset()
            self._deliver()
            done = f'{self.orders.count(DELIVERED)} delivered, {self.orders.count(SKIPPED)} skipped'
            self._land()
            self.phase = f'landed: {done}, {self.supervisor.episodes} avoidance episode(s)'
            log.info('%s with %d avoidance episode(s)', done, self.supervisor.episodes)
        except BaseException as e:
            self.phase = f'SAFETY STOP ({"Ctrl-C" if isinstance(e, KeyboardInterrupt) else e}), landing'
            raise
        finally:
            if self.movers.walks:
                log.info('%s', self.movers.summary())
            if hasattr(self.policy, 'summary'):
                log.info('%s', self.policy.summary())
            self._log_costs()
            if self._record:
                self._record.close()
                if self._frames:  # the camera frames, for camera policies in bench.py and for tuning eye.py
                    np.savez_compressed(self._record_path.with_suffix('.frames.npz'),
                                        t=np.array(self._frame_t), frames=np.stack(self._frames))
                if self._eye_log:  # the compound eye's readings, for tuning eye.py offline
                    cols = list(zip(*self._eye_log))
                    np.savez_compressed(self._record_path.with_suffix('.eye.npz'), t=np.array(cols[0]),
                                        sim_t=np.array(cols[1]), yaw=np.array(cols[2]),
                                        pos=np.array(cols[3:6]).T, dist=np.stack(cols[6]))

    def _on_tick(self):
        st = self.pilot.tm.state()
        self.orders.drone = (st.x, st.y)
        self.orders.poll()
        self.movers.update(time.monotonic(), (st.x, st.y), st.z)
        self.publish()

    def _deliver(self):
        """Fly to each stop as it comes in, on a planned path; with none waiting, hover and wait (or go
        home if asked)."""
        book = self.orders
        while True:
            order = book.next()
            if order is None:
                if book.return_home or self.return_when_done:
                    break
                self.phase = 'waiting for stops: click the Stops map or double-click the floor'
                self._hover(0.2)
                continue
            t_leg = time.monotonic()
            self._set_altitude(self.cruise_z, 'climbing to cruise height')
            if (self._fly_to((order.x, order.y), f'stop {order.n}', give_up=True,
                             stop_if=lambda o=order: o.state == CANCELLED)
                    and self._come_down((order.x, order.y), f'stop {order.n}') is not None):
                log.info('stop %d (%.1f, %.1f) reached in %.1f s', order.n, order.x, order.y, time.monotonic() - t_leg)
                book.finish(order, DELIVERED)
                self.phase = f'stop {order.n}: dropping off'
                self._hover(self.dwell_s)
            else:
                book.finish(order, SKIPPED)  # no-op if it was cancelled
        self._set_altitude(self.cruise_z, 'climbing to cruise height')
        self._fly_to(book.home, 'home')
        self._come_down(book.home, 'home')
        log.info('back home')

    def _come_down(self, spot: tuple, name: str) -> tuple | None:
        """From cruise height down to ALTITUDE_M at `spot`, carefully (see HOLD_M). Returns where it came down
        (beside the spot if that was taken), or None if no clear spot turned up within LZ_GIVE_UP_S (it then
        climbs back to cruise height). Without a cruise height, plain _set_altitude (nothing to do)."""
        if self.cruise_z <= ALTITUDE_M + 1e-6:
            return spot
        hold = min(HOLD_M, self.cruise_z)
        self._set_altitude(hold, f'{name}: coming down to look at the drop-off point')
        t0 = time.monotonic()
        t_taken = t_clear = None
        said = False
        while True:
            now = time.monotonic()
            if self._spot_taken(spot, hold - 0.3):
                t_clear, t_taken = None, t_taken or now
                if not said:
                    log.info('%s: the drop-off point is taken: waiting above it', name)
                    said = True
            else:
                t_taken, t_clear = None, t_clear or now
                if now - t_clear >= LZ_CLEAR_S:
                    break
            if t_taken is not None and now - t_taken >= LZ_MOVE_S:
                beside = self._clear_spot_near(spot, hold)
                if beside is not None:
                    log.info('%s: still taken after %.0f s: coming down beside it, at (%.2f, %.2f)', name, LZ_MOVE_S,
                             *beside)
                    self._fly_to(beside, f'{name}: beside the drop-off point')
                    spot, t_taken, said = beside, None, False
                    continue
            if now - t0 > LZ_GIVE_UP_S:
                log.warning('%s: no clear spot to come down within %.0f s: skipping it', name, LZ_GIVE_UP_S)
                self._set_altitude(self.cruise_z, 'climbing to cruise height')
                return None
            self.phase = f'{name}: the drop-off point is taken: waiting above it'
            s = self._decide(self.pilot.tm.state(), (0.0, 0.0, 0.0))
            if self.supervisor.mode is Mode.AVOID:
                self._dodge(s)
            else:
                self._step((0.0, 0.0, 0.0), None)
        self._zone = spot
        try:
            self._set_altitude(ALTITUDE_M, f'{name}: coming down')
        finally:
            self._zone = None
        return spot

    def _spot_taken(self, spot: tuple, z_top: float, radius: float = LZ_RADIUS_M) -> bool:
        """Whether the eye sees something standing on `spot` (within `radius`, between LZ_TALL_M and z_top)
        that isn't part of an obstacle in the scene file: a person, a parked cart, anything nobody mapped."""
        reading = self.eye.latest() if self.eye else None
        if reading is None or z_top <= LZ_TALL_M:
            return False
        pts = points_near(reading, spot[0], spot[1], radius, LZ_TALL_M, z_top)
        unknown = [p for p in pts if obstacle_gap(p[0], p[1], p[2], self.orders.obstacles)[0] > 0.1]
        return len(unknown) >= LZ_MIN_POINTS

    def _clear_spot_near(self, spot: tuple, z: float) -> tuple | None:
        """The nearest spot on LZ_RINGS_M round `spot` that a stop could be at (orders' checks), reachable in a
        straight line at height z, and not taken (as far as the eye can see from here)."""
        st = self.pilot.tm.state()
        for r in LZ_RINGS_M:
            for k in range(16):
                a = 2.0 * math.pi * k / 16
                c = (spot[0] + r * math.cos(a), spot[1] + r * math.sin(a))
                if self.orders._unsuitable(*c) is not None:
                    continue
                steps = max(int(math.dist((st.x, st.y), c) / 0.1), 1)
                if any(obstacle_gap(st.x + (c[0] - st.x) * i / steps, st.y + (c[1] - st.y) * i / steps, z,
                                    self.orders.obstacles)[0] < 0.3 for i in range(steps + 1)):
                    continue
                if not self._spot_taken(c, z - 0.3, LZ_RADIUS_M + LZ_MARGIN_M):
                    return c
        return None

    def _land(self, timeout_s: float = 30.0):
        """Down to LAND_HANDOFF_M with the avoidance still on (someone walking onto the landing spot is dodged,
        and the descent waits while something is under the drone), then the pilot's own touchdown."""
        self.alt_target, self.phase = LAND_HANDOFF_M, 'landing'
        st0 = self.pilot.tm.state()
        self._zone = (st0.x, st0.y)  # the descent pauses while something stands on the landing spot
        t_end = time.monotonic() + timeout_s
        while time.monotonic() < t_end:
            st = self.pilot.tm.state()
            s = self._decide(st, (0.0, 0.0, 0.0))
            if self.supervisor.mode is Mode.AVOID:
                self._dodge(s)
                continue
            if st.z < LAND_HANDOFF_M + ALT_TOL_M:
                break
            self._step((0.0, 0.0, 0.0), None)
        self._zone = None
        self.pilot.land()

    def _set_altitude(self, z: float, phase: str, timeout_s: float = 30.0) -> bool:
        """Climb or descend to z where the drone is, with the avoidance on (a dodge may interrupt it). A descent
        waits while something is under the drone (_step). Nothing to do without a cruise height."""
        if self.cruise_z <= ALTITUDE_M + 1e-6:
            return True
        self.alt_target, self.phase = z, phase
        t_end = time.monotonic() + timeout_s
        while time.monotonic() < t_end:
            st = self.pilot.tm.state()
            s = self._decide(st, (0.0, 0.0, 0.0))
            if self.supervisor.mode is Mode.AVOID:
                self._dodge(s)
                continue
            if abs(st.z - z) < ALT_TOL_M and abs(self.pilot.setpoint[2] - z) < 0.02:
                return True
            self._step((0.0, 0.0, 0.0), None)
        log.warning('%s: not at %.1f m after %.0f s (something stayed under the drone?)', phase, z, timeout_s)
        return False

    def _log_costs(self):
        d, t = self.decision_cost.summary(), self.telemetry_cost.summary()
        if d.get('count'):
            log.info('decision cost (sensors + policy + supervisor): mean %.1f us, p99 %.1f us, max %.1f us '
                     'over %d decisions; the 20 Hz loop allows 50000 us', d['mean_us'], d['p99_us'], d['max_us'], d['count'])
        if t.get('count'):
            log.info('dashboard telemetry cost (not part of the decision): mean %.1f us per update at 10 Hz',
                     t['mean_us'])

    def _fly_to(self, goal: tuple, name: str, stop_if: Callable[[], bool] | None = None,
                give_up: bool = False) -> bool:
        """Fly to goal (x, y) at ALTITUDE_M along the planner's path, dodging whenever the supervisor says
        AVOID. The path is re-planned after every avoidance episode (the dodge moved the drone) and every
        map change. Intermediate path points are passed without stopping.

        True on arrival. False when stop_if() turns true, or with give_up on a timeout or when there is
        no path. Those are only checked outside AVOID, so a dodge always finishes first. Without give_up
        a timeout or a missing path is a SafetyStop.
        """
        pilot, book = self.pilot, self.orders
        path = self._path(goal, name)
        if path is None:
            if give_up:
                return False
            raise SafetyStop(f'no path to {name}')
        corner = _corner_speeds(path)
        self.cmd, self.v_cmd = None, (0.0, 0.0, 0.0)
        st = pilot.tm.state()
        dx, dy = path[1][0] - st.x, path[1][1] - st.y
        if math.hypot(dx, dy) > 0.2:
            self.phase = f'turning toward {name}'
            pilot.turn_to(math.degrees(math.atan2(dy, dx)))
        self.phase = f'flying to {name}'
        t_end = time.monotonic() + _timeout(path)
        map_version = book.map_version
        self._replan, i = False, 1
        try:
            while True:
                st = pilot.tm.state()
                if self.supervisor.mode is Mode.NOMINAL and (self._replan or book.map_version != map_version):
                    self._replan, map_version = False, book.map_version
                    path = self._path(goal, name)
                    if path is None:
                        if give_up:
                            pilot.hold_position()
                            return False
                        raise SafetyStop(f'no path to {name}')
                    corner, i = _corner_speeds(path), 1
                    t_end = max(t_end, time.monotonic() + _timeout(path))
                while i < len(path) - 1 and math.hypot(path[i][0] - st.x, path[i][1] - st.y) < PASS_M:
                    i += 1
                    book.set_leg([(st.x, st.y)] + path[i:])
                final = i == len(path) - 1
                dx, dy = path[i][0] - st.x, path[i][1] - st.y
                dist = math.hypot(dx, dy)
                speed = SLOWDOWN_PER_S * dist if final else corner[i] + CORNER_SLOWDOWN_PER_S * dist
                k = min(CRUISE_MPS, speed) / dist if dist > 1e-6 else 0.0
                v_route = self._map_safe(st, (dx * k, dy * k, 0.0))

                s = self._decide(st, v_route, dist if final else math.inf)
                if self.supervisor.mode is Mode.AVOID:
                    self._dodge(s)
                    continue
                if final and dist < ARRIVE_M:
                    return True
                if stop_if is not None and stop_if():
                    pilot.hold_position()
                    return False
                if time.monotonic() > t_end:
                    if not give_up:
                        raise SafetyStop(f'{name} not reached in time')
                    log.warning('%s not reached in time: skipping it', name)
                    pilot.hold_position()
                    return False
                self._step(v_route, math.degrees(math.atan2(dy, dx)) if dist > 0.25 else None)
        finally:
            self.cmd, self.v_cmd = None, (0.0, 0.0, 0.0)
            book.set_leg([])

    def _path(self, goal: tuple, name: str) -> list | None:
        st = self.pilot.tm.state()
        planner = self.orders.planner
        path = planner.plan((st.x, st.y), goal)
        if path is None:
            log.warning('%s: no path on the "%s" map', name, planner.map_name)
        else:
            planner.cost_field(goal)  # which way round is shorter, for the avoidance (Sensors.route_cost)
            self.orders.set_leg(path)
            if len(path) > 2:
                log.info('%s: planned %d legs, %.1f m (straight line %.1f m)', name, len(path) - 1,
                         path_length(path), math.hypot(goal[0] - st.x, goal[1] - st.y))
        return path

    def _decide(self, st, v_route: tuple, stop_m: float = 0.0) -> Sensors:
        """One decision: sensors -> policy -> supervisor. Everything in here is what CostMeter times.
        stop_m: how far ahead the route will stop the drone (0: holding position; inf: flying on)."""
        t_decide = time.perf_counter_ns()
        planner = self.orders.planner
        s = make_sensors(time.monotonic(), (st.x, st.y, st.z), st.yaw, (st.vx, st.vy, st.vz), v_route,
                         st.ranges, self.supervisor.mode, self.eye.latest() if self.eye else None, stop_m,
                         planner.route_cost if planner.field_goal is not None else None)
        frame = self.camera.latest() if self.camera else None
        self.cmd = self.policy.step(s, frame)
        if s.eye is not None and s.eye is not self._objects_reading:  # what is around, static or moving
            self._objects_reading = s.eye
            self._objects.update(s.eye)
        if s.eye is not None:  # the nearest thing ahead and aside, not just on three rays; and the room below
            self._ring = ring(s.eye, s.yaw_deg)
            self.clearance = eye_clearance(*self._ring)
            self._below = clear_below(s.eye)
        else:
            self._ring, self._below = None, math.inf
            self.clearance = min(s.ranges['front'], s.ranges['left'], s.ranges['right'])
        changed = self.supervisor.update(s.t, self.cmd.signal, self.clearance)
        self.decision_cost.add(time.perf_counter_ns() - t_decide)
        if self._record:
            self._write_record(s, v_route)
            if frame is not None and frame is not self._last_frame:  # each new camera frame, half resolution
                self._last_frame = frame
                self._frames.append(frame[::2, ::2].copy())
                self._frame_t.append(s.t - self.t0)
            reading = s.eye
            if reading is not None and reading is not self._last_eye:  # each new compound-eye reading, as decided on
                self._last_eye = reading
                self._eye_log.append((s.t - self.t0, reading[0], s.yaw_deg, *reading[1], reading[2].astype(np.float16)))
        if changed:
            self._on_mode_change(s)
        return s

    def _map_safe(self, st, v: tuple) -> tuple:
        """Slow down so that no MAPPED obstacle is less than MAP_TTC_S away at the commanded speed: along
        each of the four range rays, and anywhere in the corridor the drone's path sweeps (what the
        360-degree policy checks). The planner already routes around those; without this the avoidance
        would read the approach to a corner beside a wall as a collision course and dodge an obstacle that
        is no threat. Unmapped obstacles are untouched: they stay the avoidance's job."""
        planner, scale = self.orders.planner, 1.0
        yaw = math.radians(st.yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        rays = [(bf * c - bl * s, bf * s + bl * c, 0.0) for bf, bl in HORIZONTAL_RAYS.values()]
        speed = math.hypot(v[0], v[1])
        if speed > 0.05:
            ux, uy = v[0] / speed, v[1] / speed
            rays += [(ux, uy, off) for off in (-MAP_CORRIDOR_M, 0.0, MAP_CORRIDOR_M)]
        for ux, uy, off in rays:
            closing = v[0] * ux + v[1] * uy
            if closing <= 0.05:
                continue
            reach = closing * MAP_TTC_S
            d = planner.distance_ahead(st.x - uy * off, st.y + ux * off, ux, uy, reach)
            if d < reach:
                scale = min(scale, d / reach)
        return (v[0] * scale, v[1] * scale, v[2])

    def _dodge(self, s: Sensors):
        """One tick of the policy's dodge (heading frozen), plus the clear-out when the policy is done
        but the drone is too close to something for the supervisor to resume (avoidance.clear_out)."""
        sup = self.supervisor
        if sup.time_in_mode(s.t) > MAX_EPISODE_S:
            raise SafetyStop(f'avoidance did not resolve within {MAX_EPISODE_S:.0f} s')
        dodge = self.cmd.dodge
        if sup.gate['signal_ok'] and not sup.gate['clear_ok']:
            out = (clear_out_ring(*self._ring, sup.clearance_m, CLEAR_OUT_MPS) if self._ring is not None
                   else clear_out(s.ranges, sup.clearance_m))
            dodge = (dodge[0] + out[0], dodge[1] + out[1], dodge[2])
        self._step(body_to_world(self._dodge_safe(dodge), s.yaw_deg), None)

    def _dodge_safe(self, dodge: tuple) -> tuple:
        """Slow a dodge down rather than fly it into something: what the compound eye sees within
        DODGE_HALF_WIDTH_M of the dodge's direction must stay DODGE_TTC_S away at the dodge's speed
        (beyond DODGE_GAP_M). Policy-independent, like the speed cap near mapped obstacles: a dodge away
        from one thing must not fly into another (an escape into clutter makes the next alarm)."""
        f, l, up = dodge
        speed = math.hypot(f, l)
        if self._ring is None or speed < 0.02:
            return dodge
        az, near = self._ring
        near = np.where(np.isfinite(near), near, 1e3)  # nothing there: far away (keeps the arithmetic finite)
        rel = np.radians(az) - math.atan2(l, f)
        along, across = near * np.cos(rel), near * np.sin(rel)
        hit = (along > 0.0) & (np.abs(across) < DODGE_HALF_WIDTH_M)
        if not hit.any():
            return dodge
        allowed = max(float(along[hit].min()) - DODGE_GAP_M, 0.0) / DODGE_TTC_S
        k = min(1.0, allowed / speed)
        return (f * k, l * k, up)

    def _step(self, v: tuple, yaw: float | None):
        """One tick in velocity mode. The setpoint never goes within FENCE_KEEP_M of the geofence (the
        flight's safety tripwire): the policy doesn't know where it is, and the setpoint leads the drone."""
        vx, vy, vz = v
        sp, dt = self.pilot.setpoint, self.pilot.dt
        (x0, x1), (y0, y1), (_, z1) = self.orders.fence['x'], self.orders.fence['y'], self.orders.fence['z']
        vx = max((x0 + FENCE_KEEP_M - sp[0]) / dt, min((x1 - FENCE_KEEP_M - sp[0]) / dt, vx))
        vy = max((y0 + FENCE_KEEP_M - sp[1]) / dt, min((y1 - FENCE_KEEP_M - sp[1]) / dt, vy))
        if self.supervisor.mode is not Mode.AVOID:  # the route's altitude (a dodge may have climbed)
            down = DESCENT_MPS if sp[2] > SLOW_BELOW_M else DESCENT_SLOW_MPS
            pull = min(CLIMB_MPS, max(-down, 2.0 * (self.alt_target - sp[2])))
            if pull < 0.0 and self._zone is not None and self._spot_taken(self._zone, sp[2] - 0.3):
                pull = 0.0  # someone (something) on the landing spot: wait
            under = sp[2] - self._below  # how tall what is under the drone is (the floor: ~0)
            if pull < 0.0 and under > max(self.alt_target - DESCENT_CLEAR_M, FLOOR_M):
                pull = 0.0  # something under the drone (a person, a cart): wait until it has gone
            vz += pull
        # Up to the ceiling's keep-out at most, braking on the way in proportion to the room left, from the
        # setpoint and from where the drone really is: stopping the setpoint dead at 1 m/s let the drone
        # overshoot it by 0.5 m, through the geofence (fs_blocked, back-to-back escape climbs)
        z_top = z1 - CEILING_KEEP_M
        z_now = self.pilot.tm.state().z
        vz = min(vz, max(CEILING_BRAKE_PER_S * (z_top - sp[2]), 0.0), max(CEILING_BRAKE_PER_S * (z_top - z_now), 0.0)) \
            if vz > 0.0 else vz
        vz = max((MIN_Z_M - sp[2]) / dt, min((z_top - sp[2]) / dt, vz))
        self.v_cmd = (vx, vy, vz)
        self.pilot.step_velocity(self.v_cmd, yaw)

    def _hover(self, seconds: float):
        """Hold position with the avoidance still on: it decides every tick and can dodge."""
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            s = self._decide(self.pilot.tm.state(), (0.0, 0.0, 0.0))
            if self.supervisor.mode is Mode.AVOID:
                self._dodge(s)
            else:
                self._step((0.0, 0.0, 0.0), None)  # holds position, and the route's altitude

    def _on_mode_change(self, s: Sensors):
        sup, info = self.supervisor, self.cmd.info
        if sup.mode is Mode.AVOID:
            if self.policy.brake_on_avoid:
                self.pilot.hold_position()  # stop here instead of finishing the setpoint's lead
            if 'gf_left' in info:  # the fly brain: its giant fibers and what the eyes saw
                log.info('AVOID   episode %d: signal %.2f >= %.2f (giant fibers L %.0f R %.0f Hz; eye drive L %.2f R %.2f)',
                         sup.episodes, self.cmd.signal, sup.enter, info['gf_left'], info['gf_right'],
                         info['drive_left'], info['drive_right'])
            else:
                ray = info.get('worst_ray', '?')
                log.info('AVOID   episode %d: signal %.2f >= %.2f (%s ray %.2f m, TTC %.1f s)',
                         sup.episodes, self.cmd.signal, sup.enter, ray,
                         s.ranges.get(ray, math.nan), info.get('ttc_worst', math.nan))
        else:
            log.info('NOMINAL resumed after %.1f s in AVOID', sup.last_episode_s)
            self.policy.reset()
            self._map_what_was_dodged(s)
            self._replan = True  # the dodge moved the drone off its path

    def _map_what_was_dodged(self, s: Sensors):
        """After an avoidance episode, put the static things the eye sees nearby that aren't on the map yet on
        the planner's map, so the re-planned path goes round them instead of straight back into them. Only
        things never seen moving: a person or a cart moves on, a crate stays."""
        if self._objects_reading is None:
            return
        planner = self.orders.planner
        x, y = s.position[0], s.position[1]
        points = [p for obj in self._objects.objects if not obj.moved for p in obj.points
                  if math.hypot(p[0] - x, p[1] - y) < SEEN_RANGE_M and planner.free(p[0], p[1])]
        if points and planner.add_seen(points):
            log.info('map: added an obstacle the avoidance met (%d points within %.1f m; not on the "%s" map)',
                     len(points), SEEN_RANGE_M, planner.map_name)

    def _write_record(self, s: Sensors, v_route_world: tuple):
        """One JSON line per decision: the raw inputs and the decision taken (replayed by bench.py), plus, for
        analysis only, the policy's own readout (info) and where the movers were."""
        finite = lambda v: v if math.isfinite(v) else None  # noqa: E731
        info = {k: (finite(float(v)) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                for k, v in self.cmd.info.items() if v is None or isinstance(v, (int, float, str, bool))}
        self._record.write(json.dumps({
            't': s.t - self.t0, 'position': [float(v) for v in s.position], 'yaw_deg': s.yaw_deg,
            'velocity_world': [float(v) for v in s.velocity_world],
            'intended_velocity_world': [float(v) for v in v_route_world], 'mode_before': s.mode.value,
            'ranges': {k: finite(v) for k, v in s.ranges.items()},
            'signal': self.cmd.signal, 'dodge': [float(v) for v in self.cmd.dodge],
            'mode_after': self.supervisor.mode.value, 'info': info, 'route_stop_m': finite(s.route_stop_m),
            'movers': {n: [round(p[0], 3), round(p[1], 3)] for n, p in self.movers.pose.items()},
        }) + '\n')

    def publish(self, force: bool = False):
        now = time.monotonic()
        if not force and now - self._last_publish < DASHBOARD_PERIOD_S:
            return
        self._last_publish = now
        t_pub = time.perf_counter_ns()
        self._publish()
        self.telemetry_cost.add(time.perf_counter_ns() - t_pub)

    def _publish(self):
        st, sup, cmd, now = self.pilot.tm.state(), self.supervisor, self.cmd, time.monotonic()
        if now - self._cost_summary_t > 1.0:
            self._cost_summary, self._cost_summary_t = self.decision_cost.summary(), now
        camera_fps = None
        if self.camera:
            self._camera_counts.append((now, self.camera.frames_received))
            (t_a, n_a), (t_b, n_b) = self._camera_counts[0], self._camera_counts[-1]
            camera_fps = (n_b - n_a) / (t_b - t_a) if t_b - t_a > 0.5 else None
        self.feed.send({
            't': now - self.t0, 'phase': self.phase,
            'route_name': self.route_name,
            'policy': {'name': self.policy.name, 'inputs': list(self.policy.inputs),
                       'notes': getattr(self.policy, 'input_notes', {}), 'description': self.policy.description},
            'camera_streaming': bool(self.camera and self.camera.connected), 'camera_fps': camera_fps,
            'mode': sup.mode.value, 'time_in_mode': sup.time_in_mode(now), 'episodes': sup.episodes,
            'thresholds': {'enter': sup.enter, 'exit': sup.exit, 'dwell_s': sup.dwell_s,
                           'clearance_m': sup.clearance_m},
            'gate': sup.gate, 'clearance': self.clearance,
            'signal': cmd.signal if cmd else None, 'info': cmd.info if cmd else {},
            'v_cmd': self.v_cmd,
            'state': {'x': st.x, 'y': st.y, 'z': st.z, 'yaw': st.yaw, 'vx': st.vx, 'vy': st.vy},
            'ranges': st.ranges, 'decision_cost': self._cost_summary,
            'eye_ring': None if self._ring is None else  # what is around the drone: [world azimuth deg, m]
            [[round(float(a) + st.yaw, 1), round(float(d), 2)] for a, d in zip(*self._ring) if d < 4.0],
            'seen': [[round(x, 2), round(y, 2)] for x, y in self.orders.planner.seen_points[-300:]],
            'orders': self.orders.snapshot(), 'scene': self.scene, 'movers': self.movers.snapshot(),
        })


def xy(s: str) -> tuple:
    try:
        x, y = (float(v) for v in s.split(','))
    except ValueError:
        raise argparse.ArgumentTypeError(f'expected X,Y in metres, like 1.5,-0.4 (got {s!r})') from None
    return x, y


def _join_at(argv: list) -> list:
    """'--at -0.7,1.6' -> '--at=-0.7,1.6': argparse reads a value starting with '-' as an option."""
    out, args = [], iter(argv)
    for a in args:
        out.append(f'--at={next(args, "")}' if a == '--at' else a)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--route', choices=sorted(ROUTES) + ['live'], default='blocked',
                    help='a test route from routes.py (its stops and map; ends at home), or "live": '
                         'no stops until you add them, and it waits for more')
    ap.add_argument('--at', type=xy, action='append', default=[], metavar='X,Y',
                    help="an extra stop in metres, after the route's; repeat for more")
    ap.add_argument('--return', dest='return_when_done', action='store_true',
                    help='(live) return home and land after the last stop instead of waiting for more')
    ap.add_argument('--map', choices=MAPS, default=None,
                    help="what the path planner knows, instead of the route's (live: all; planner.py "
                         "explains): all, building (crates left to the avoidance), none (straight lines)")
    ap.add_argument('--scene', choices=sorted(SCENES), default=None,
                    help="(live) the scene, with what moves in it (routes.LIVE_MOVERS); a route brings its own. "
                         "launch.cmd starts the sim on the same scene")
    ap.add_argument('--policy', choices=sorted(POLICIES), default='geometric')
    ap.add_argument('--cruise', type=float, default=None, metavar='M',
                    help=f'fly between stops this high, descending to {ALTITUDE_M} m at each stop (default: the '
                         "route's; live flights in the street cruise at 4.5 m; 0: no cruise height)")
    ap.add_argument('--record', type=Path, metavar='FILE.jsonl',
                    help='save every decision (inputs + outputs), for replaying with bench.py')
    args = ap.parse_args(_join_at(sys.argv[1:]))
    setup_logging()

    route = ROUTES.get(args.route)
    if route and args.scene and args.scene != route.scene:
        ap.error(f'route "{args.route}" runs in the "{route.scene}" scene')
    scene = route.scene if route else args.scene or 'obstacles'
    fence = geofence(SCENES[scene])
    cruise = args.cruise if args.cruise is not None else route.cruise_m if route else LIVE_CRUISE_M.get(scene)
    cruise = cruise if cruise and cruise > ALTITUDE_M else None
    if cruise is not None and cruise > fence['z'][1] - CEILING_KEEP_M:
        ap.error(f'--cruise {cruise}: the "{scene}" scene\'s ceiling is {fence["z"][1]} m')
    orders = OrderBook(ALTITUDE_M, fence, load_obstacles(SCENES[scene]), args.map or (route.map if route else 'all'),
                       cruise=cruise)
    movers = MoverRunner(load_movers(SCENES[scene]), route.movers if route else LIVE_MOVERS.get(scene, ()))
    if movers.missing:
        log.warning('scene "%s" has no movers called %s', scene, ', '.join(movers.missing))
    for x, y in (route.stops if route else []) + args.at:
        if orders.add(x, y) is None:
            orders.close()
            ap.error(f'stop {x},{y}: {orders.last_event}')
    policy = POLICIES[args.policy]()
    if getattr(policy, 'brain_ok', True) is False:
        orders.close()
        raise SystemExit('the fly brain is not answering on udp://127.0.0.1:5320; launch.cmd starts it for '
                         '--policy flybrain (or: bash ../flybrain/py.sh brain_server.py)')
    log.info('avoidance test: route "%s" in the "%s" scene on the "%s" map, policy "%s" (reads: %s)%s%s',
             args.route, scene, orders.planner.map_name, policy.name, ', '.join(policy.inputs) or 'nothing',
             f'; moving: {", ".join(w.mover for w in movers.walks)}' if movers.walks else '',
             f'; cruising at {cruise:g} m, stops at {ALTITUDE_M:g} m' if cruise else '')
    # Connect the camera before cflib: when the camera bridge gets a client it briefly takes over the
    # sim's cflib port, which can stall a cflib handshake in progress.
    camera = FrameSource().start()
    try:
        camera.get_frame(timeout=10.0)
    except TimeoutError:
        camera.close()
        camera = None
        if getattr(policy, 'needs_camera', 'camera' in policy.inputs):
            orders.close()
            raise SystemExit(f'no camera stream, and policy "{policy.name}" sees through the camera')
        log.warning('no camera stream; flying without camera frames')
    # The compound eye: every policy gets it, and the supervisor's clearance comes from it.
    try:
        eye = EyeSource()
    except OSError as e:
        eye = None
        log.warning('compound eye off: udp port 5330 is busy (%s)', e)
    t_wait = time.monotonic() + 3.0
    while eye and eye.latest() is None and time.monotonic() < t_wait:
        time.sleep(0.05)
    if eye is None or eye.latest() is None:
        if getattr(policy, 'needs_eye', False):
            orders.close()
            raise SystemExit(f'no compound eye on udp://127.0.0.1:5330, and policy "{policy.name}" sees through '
                             'it: re-run setup.cmd (it is part of the CrazySim patch)')
        log.warning('no compound eye: deciding on the Multi-ranger rays alone')

    flight = AvoidFlight(args.route, policy, camera, DashboardFeed(), orders, movers, args.record,
                         return_when_done=args.return_when_done or route is not None,
                         dwell_s=route.dwell_s if route else DROP_OFF_S, scene=scene, eye=eye, cruise_m=cruise)
    try:
        code = fly_mission(flight.run, geofence=fence)
    finally:
        if camera:
            camera.close()
        if eye:
            eye.close()
        orders.close()
        movers.close()
    if flight.pilot:
        flight.publish(force=True)  # final state, so the dashboard shows how it ended
    return code


if __name__ == '__main__':
    sys.exit(main())
