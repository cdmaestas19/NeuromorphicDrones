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
    launch.cmd avoid --route headon      moving obstacles (street scene): headon, hover, crossing, cart
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

from avoidance import (HORIZONTAL_RAYS, AvoidancePolicy, AvoidanceSupervisor, CostMeter, Mode, Sensors, clear_out,
                       make_sensors)
from dashboard import DashboardFeed
from flight import GEOFENCE, Pilot, SafetyStop, fly_mission, setup_logging
from frames import FrameSource
from orders import CANCELLED, DELIVERED, SKIPPED, OrderBook
from movers import MoverRunner, load_movers
from planner import MAPS, SCENES, load_obstacles, path_length
from policies import POLICIES
from routes import ALTITUDE_M, LIVE_MOVERS, ROUTES

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
                 dwell_s: float = DROP_OFF_S, scene: str = 'obstacles'):
        self.route_name, self.scene = route_name, scene
        self.orders, self.movers, self.return_when_done, self.dwell_s = orders, movers, return_when_done, dwell_s
        self.policy, self.camera, self.feed = policy, camera, feed
        self.supervisor = AvoidanceSupervisor()
        self.decision_cost, self.telemetry_cost = CostMeter(), CostMeter()
        self._cost_summary, self._cost_summary_t = {}, 0.0
        self._last_publish = 0.0
        self._record = None
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
            self.phase = 'landing'
            pilot.land()
            self.phase = f'landed: {done}, {self.supervisor.episodes} avoidance episode(s)'
            log.info('%s with %d avoidance episode(s)', done, self.supervisor.episodes)
        except BaseException as e:
            self.phase = f'SAFETY STOP ({"Ctrl-C" if isinstance(e, KeyboardInterrupt) else e}), landing'
            raise
        finally:
            if self.movers.walks:
                log.info('%s', self.movers.summary())
            self._log_costs()
            if self._record:
                self._record.close()

    def _on_tick(self):
        st = self.pilot.tm.state()
        self.orders.drone = (st.x, st.y)
        self.orders.poll()
        self.movers.update(time.monotonic(), (st.x, st.y))
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
            if self._fly_to((order.x, order.y), f'stop {order.n}', give_up=True,
                            stop_if=lambda o=order: o.state == CANCELLED):
                log.info('stop %d (%.1f, %.1f) reached in %.1f s', order.n, order.x, order.y, time.monotonic() - t_leg)
                book.finish(order, DELIVERED)
                self.phase = f'stop {order.n}: dropping off'
                self._hover(self.dwell_s)
            else:
                book.finish(order, SKIPPED)  # no-op if it was cancelled
        self._fly_to(book.home, 'home')
        log.info('back home')

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

                s = self._decide(st, v_route)
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
            self.orders.set_leg(path)
            if len(path) > 2:
                log.info('%s: planned %d legs, %.1f m (straight line %.1f m)', name, len(path) - 1,
                         path_length(path), math.hypot(goal[0] - st.x, goal[1] - st.y))
        return path

    def _decide(self, st, v_route: tuple) -> Sensors:
        """One decision: sensors -> policy -> supervisor. Everything in here is what CostMeter times."""
        t_decide = time.perf_counter_ns()
        s = make_sensors(time.monotonic(), (st.x, st.y, st.z), st.yaw, (st.vx, st.vy, st.vz), v_route,
                         st.ranges, self.supervisor.mode)
        self.cmd = self.policy.step(s, self.camera.latest() if self.camera else None)
        self.clearance = min(s.ranges['front'], s.ranges['left'], s.ranges['right'])
        changed = self.supervisor.update(s.t, self.cmd.signal, self.clearance)
        self.decision_cost.add(time.perf_counter_ns() - t_decide)
        if self._record:
            self._write_record(s, v_route)
        if changed:
            self._on_mode_change(s)
        return s

    def _map_safe(self, st, v: tuple) -> tuple:
        """Slow down so that, along each of the four range rays, no MAPPED obstacle is less than
        MAP_TTC_S away at the commanded speed. The planner already routes around those; without this
        the avoidance would read the approach to a corner beside a wall as a collision course and dodge
        an obstacle that is no threat. Unmapped obstacles are untouched: they stay the avoidance's job."""
        planner, scale = self.orders.planner, 1.0
        yaw = math.radians(st.yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        for bf, bl in HORIZONTAL_RAYS.values():
            ux, uy = bf * c - bl * s, bf * s + bl * c
            closing = v[0] * ux + v[1] * uy
            if closing <= 0.05:
                continue
            reach = closing * MAP_TTC_S
            d = planner.distance_ahead(st.x, st.y, ux, uy, reach)
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
            out = clear_out(s.ranges, sup.clearance_m)
            dodge = (dodge[0] + out[0], dodge[1] + out[1], dodge[2])
        self._step(body_to_world(dodge, s.yaw_deg), None)

    def _step(self, v: tuple, yaw: float | None):
        """One tick in velocity mode. The setpoint never goes within FENCE_KEEP_M of the geofence (the
        flight's safety tripwire): the policy doesn't know where it is, and the setpoint leads the drone."""
        vx, vy, vz = v
        sp, dt = self.pilot.setpoint, self.pilot.dt
        (x0, x1), (y0, y1) = GEOFENCE['x'], GEOFENCE['y']
        vx = max((x0 + FENCE_KEEP_M - sp[0]) / dt, min((x1 - FENCE_KEEP_M - sp[0]) / dt, vx))
        vy = max((y0 + FENCE_KEEP_M - sp[1]) / dt, min((y1 - FENCE_KEEP_M - sp[1]) / dt, vy))
        vz += min(0.3, max(-0.3, 2.0 * (ALTITUDE_M - sp[2])))  # hold altitude
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
                self.v_cmd = (0.0, 0.0, 0.0)
                self.pilot.tick()

    def _on_mode_change(self, s: Sensors):
        sup, info = self.supervisor, self.cmd.info
        if sup.mode is Mode.AVOID:
            self.pilot.hold_position()  # stop here instead of finishing the setpoint's lead
            ray = info.get('worst_ray', '?')
            log.info('AVOID   episode %d: signal %.2f >= %.2f (%s ray %.2f m, TTC %.1f s)',
                     sup.episodes, self.cmd.signal, sup.enter, ray,
                     s.ranges.get(ray, math.nan), info.get('ttc_worst', math.nan))
        else:
            log.info('NOMINAL resumed after %.1f s in AVOID', sup.last_episode_s)
            self.policy.reset()
            self._replan = True  # the dodge moved the drone off its path

    def _write_record(self, s: Sensors, v_route_world: tuple):
        """One JSON line per decision: the raw inputs and the decision taken (replayed by bench.py)."""
        finite = lambda v: v if math.isfinite(v) else None  # noqa: E731
        self._record.write(json.dumps({
            't': s.t - self.t0, 'position': [float(v) for v in s.position], 'yaw_deg': s.yaw_deg,
            'velocity_world': [float(v) for v in s.velocity_world],
            'intended_velocity_world': [float(v) for v in v_route_world], 'mode_before': s.mode.value,
            'ranges': {k: finite(v) for k, v in s.ranges.items()},
            'signal': self.cmd.signal, 'dodge': [float(v) for v in self.cmd.dodge],
            'mode_after': self.supervisor.mode.value,
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
                       'description': self.policy.description},
            'camera_streaming': bool(self.camera and self.camera.connected), 'camera_fps': camera_fps,
            'mode': sup.mode.value, 'time_in_mode': sup.time_in_mode(now), 'episodes': sup.episodes,
            'thresholds': {'enter': sup.enter, 'exit': sup.exit, 'dwell_s': sup.dwell_s,
                           'clearance_m': sup.clearance_m},
            'gate': sup.gate, 'clearance': self.clearance,
            'signal': cmd.signal if cmd else None, 'info': cmd.info if cmd else {},
            'v_cmd': self.v_cmd,
            'state': {'x': st.x, 'y': st.y, 'z': st.z, 'yaw': st.yaw, 'vx': st.vx, 'vy': st.vy},
            'ranges': st.ranges, 'decision_cost': self._cost_summary,
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
    ap.add_argument('--record', type=Path, metavar='FILE.jsonl',
                    help='save every decision (inputs + outputs), for replaying with bench.py')
    args = ap.parse_args(_join_at(sys.argv[1:]))
    setup_logging()

    route = ROUTES.get(args.route)
    if route and args.scene and args.scene != route.scene:
        ap.error(f'route "{args.route}" runs in the "{route.scene}" scene')
    scene = route.scene if route else args.scene or 'obstacles'
    orders = OrderBook(ALTITUDE_M, GEOFENCE, load_obstacles(SCENES[scene]), args.map or (route.map if route else 'all'))
    movers = MoverRunner(load_movers(SCENES[scene]), route.movers if route else LIVE_MOVERS.get(scene, ()))
    if movers.missing:
        log.warning('scene "%s" has no movers called %s', scene, ', '.join(movers.missing))
    for x, y in (route.stops if route else []) + args.at:
        if orders.add(x, y) is None:
            orders.close()
            ap.error(f'stop {x},{y}: {orders.last_event}')
    policy = POLICIES[args.policy]()
    log.info('avoidance test: route "%s" in the "%s" scene on the "%s" map, policy "%s" (reads: %s)%s',
             args.route, scene, orders.planner.map_name, policy.name, ', '.join(policy.inputs) or 'nothing',
             f'; moving: {", ".join(w.mover for w in movers.walks)}' if movers.walks else '')
    # Connect the camera before cflib: when the camera bridge gets a client it briefly takes over the
    # sim's cflib port, which can stall a cflib handshake in progress.
    camera = FrameSource().start()
    try:
        camera.get_frame(timeout=10.0)
    except TimeoutError:
        log.warning('no camera stream; flying without camera frames')
        camera.close()
        camera = None

    flight = AvoidFlight(args.route, policy, camera, DashboardFeed(), orders, movers, args.record,
                         return_when_done=args.return_when_done or route is not None,
                         dwell_s=route.dwell_s if route else DROP_OFF_S, scene=scene)
    try:
        code = fly_mission(flight.run)
    finally:
        if camera:
            camera.close()
        orders.close()
        movers.close()
    if flight.pilot:
        flight.publish(force=True)  # final state, so the dashboard shows how it ended
    return code


if __name__ == '__main__':
    sys.exit(main())
