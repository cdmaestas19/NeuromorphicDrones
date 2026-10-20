"""Shared flight stack: cflib connection, telemetry, and a setpoint-streaming Pilot with safety stops.

The firmware's PID position controller keeps the drone stable; this module only decides where the
position setpoint goes. Flight scripts describe a mission as a function of a Pilot and hand it to
fly_mission(), which lands the drone on Ctrl-C, on any exception, and on any safety check failure.
"""
import logging
import math
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig

log = logging.getLogger('flypilot')

SIM_URI = 'udp://127.0.0.1:19850'
CACHE_DIR = Path(__file__).resolve().parent / '.cache'

# supervisor.info bits (crazyflie-firmware src/modules/src/supervisor.c)
SUP_CAN_FLY, SUP_IS_FLYING, SUP_TUMBLED, SUP_LOCKED, SUP_CRASHED = 0x08, 0x10, 0x20, 0x40, 0x80

RANGE_MAX_M = 4.0         # Multi-ranger (VL53L1x) max range; CrazySim reports 4.0 m when nothing is hit
RANGE_DIRECTIONS = ('front', 'back', 'left', 'right', 'up')

# scene_obstacles.xml is a 6 x 6 m room (walls at +-3 m); stay well inside it and below 1.4 m.
GEOFENCE = {'x': (-2.6, 2.6), 'y': (-2.6, 2.6), 'z': (-0.2, 1.4)}

CONNECT_ATTEMPTS = 3
TELEMETRY_PERIOD_MS = 50
TELEMETRY_STALE_S = 0.5
LEASH_M = 0.35            # how far the setpoint may run ahead of the drone before it waits


class SafetyStop(Exception):
    """Raised during a mission to abort it; fly_mission() then lands the drone."""


@dataclass
class State:
    """Latest estimate from the drone. Ranges are metres; math.inf means nothing within RANGE_MAX_M."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0      # degrees, 0 = +x, counter-clockwise positive
    vx: float = 0.0       # world-frame velocity [m/s]
    vy: float = 0.0
    vz: float = 0.0
    supervisor: int = 0
    ranges: dict = field(default_factory=lambda: {d: math.inf for d in RANGE_DIRECTIONS})
    stamp: float = 0.0    # time.monotonic() of the last pose update

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz])


def wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def sim_is_up(uri: str = SIM_URI, timeout: float = 0.3) -> bool:
    """True once CrazySim answers on its cflib port (it only does after the firmware has connected)."""
    u = urlparse(uri)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((u.hostname, u.port))
            s.send(b'\xff')
            return s.recv(64) == b'\xff'
        except OSError:
            return False


class Telemetry:
    """Streams pose, velocity, supervisor state and Multi-ranger distances from the drone at 20 Hz."""

    def __init__(self, cf: Crazyflie):
        self._lock = threading.Lock()
        self._state = State()
        # Three log blocks: each CRTP log packet carries at most 26 bytes of variables.
        pose = LogConfig('flypilot_pose', period_in_ms=TELEMETRY_PERIOD_MS)
        for var in ('stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z', 'stateEstimate.yaw'):
            pose.add_variable(var, 'float')
        pose.add_variable('supervisor.info', 'uint16_t')
        velocity = LogConfig('flypilot_velocity', period_in_ms=TELEMETRY_PERIOD_MS)
        for var in ('stateEstimate.vx', 'stateEstimate.vy', 'stateEstimate.vz'):
            velocity.add_variable(var, 'float')
        ranges = LogConfig('flypilot_ranges', period_in_ms=TELEMETRY_PERIOD_MS)
        for d in RANGE_DIRECTIONS:
            ranges.add_variable(f'range.{d}', 'uint16_t')
        for config, callback in ((pose, self._on_pose), (velocity, self._on_velocity), (ranges, self._on_ranges)):
            cf.log.add_config(config)
            config.data_received_cb.add_callback(callback)
            config.start()

    def state(self) -> State:
        with self._lock:
            return replace(self._state, ranges=dict(self._state.ranges))

    def _on_pose(self, _timestamp, data, _config):
        with self._lock:
            s = self._state
            s.x, s.y, s.z = data['stateEstimate.x'], data['stateEstimate.y'], data['stateEstimate.z']
            s.yaw = data['stateEstimate.yaw']
            s.supervisor = data['supervisor.info']
            s.stamp = time.monotonic()

    def _on_velocity(self, _timestamp, data, _config):
        with self._lock:
            s = self._state
            s.vx, s.vy, s.vz = data['stateEstimate.vx'], data['stateEstimate.vy'], data['stateEstimate.vz']

    def _on_ranges(self, _timestamp, data, _config):
        with self._lock:
            for d in RANGE_DIRECTIONS:
                metres = data[f'range.{d}'] / 1000.0
                self._state.ranges[d] = math.inf if metres >= RANGE_MAX_M else metres


class Link:
    """An open cflib connection with telemetry running and the position estimate converged."""

    def __init__(self, uri: str = SIM_URI, wait_for_sim_s: float = 60.0, connect_timeout_s: float = 10.0):
        self.uri = uri
        self.lost_reason = None
        cflib.crtp.init_drivers()
        logging.getLogger('cflib').setLevel(logging.WARNING)

        deadline = time.monotonic() + wait_for_sim_s
        while not sim_is_up(uri):
            if time.monotonic() > deadline:
                raise ConnectionError(f'no simulator answering at {uri}; is run_sim running?')
            log.info('waiting for the sim at %s ...', uri)
            time.sleep(2.0)

        CACHE_DIR.mkdir(exist_ok=True)
        for attempt in range(1, CONNECT_ATTEMPTS + 1):
            log.info('connecting to %s', uri)
            self.cf, error = self._open(uri, connect_timeout_s)
            if self.cf is not None:
                break
            log.warning('connection attempt %d/%d failed: %s', attempt, CONNECT_ATTEMPTS, error)
        else:
            raise ConnectionError(f'could not connect to {uri}: {error}')

        self.telemetry = Telemetry(self.cf)
        self._reset_estimator()
        self._wait_until_can_fly()

    def close(self):
        self.cf.close_link()

    def __enter__(self) -> 'Link':
        return self

    def __exit__(self, *exc):
        self.close()

    def _open(self, uri: str, timeout: float) -> tuple[Crazyflie | None, str | None]:
        """One connection attempt, bounded in time: cflib itself waits forever if a reply is lost.

        CrazySim forwards replies only to whoever last sent to its cflib port, so a camera client
        connecting mid-handshake (its bridge registers on that port) can swallow a TOC reply.
        """
        cf = Crazyflie(rw_cache=str(CACHE_DIR))
        done, failure = threading.Event(), []
        cf.fully_connected.add_callback(lambda _uri: done.set())
        cf.connection_failed.add_callback(lambda _uri, msg: (failure.append(msg), done.set()))
        cf.open_link(uri)
        if done.wait(timeout) and not failure:
            cf.connection_lost.add_callback(self._on_lost)
            return cf, None
        cf.close_link()
        return None, failure[0] if failure else f'no complete reply within {timeout:.0f} s'

    def _on_lost(self, _uri, msg):
        self.lost_reason = msg
        log.error('link lost: %s', msg)

    def _reset_estimator(self, timeout: float = 10.0):
        """Reset the Kalman filter and wait for its position variance to settle (Bitcraze's standard check)."""
        var = LogConfig('flypilot_kalman_var', period_in_ms=100)
        names = ('kalman.varPX', 'kalman.varPY', 'kalman.varPZ')
        for n in names:
            var.add_variable(n, 'float')
        history = {n: deque(maxlen=10) for n in names}
        settled = threading.Event()

        def on_var(_timestamp, data, _config):
            for n in names:
                history[n].append(data[n])
            if all(len(h) == h.maxlen and max(h) - min(h) < 0.001 for h in history.values()):
                settled.set()

        self.cf.log.add_config(var)
        var.data_received_cb.add_callback(on_var)
        self.cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        self.cf.param.set_value('kalman.resetEstimation', '0')
        var.start()
        try:
            if not settled.wait(timeout):
                raise ConnectionError('position estimate did not converge')
        finally:
            var.delete()
        s = self.telemetry.state()
        log.info('estimator converged at (%.2f, %.2f, %.2f), yaw %.0f deg', s.x, s.y, s.z, s.yaw)

    def _wait_until_can_fly(self, timeout: float = 10.0):
        deadline = time.monotonic() + timeout
        while not self.telemetry.state().supervisor & SUP_CAN_FLY:
            if time.monotonic() > deadline:
                raise ConnectionError(f'supervisor reports not ready to fly (info=0x{self.telemetry.state().supervisor:02x})')
            time.sleep(0.1)


class Pilot:
    """Streams position setpoints at a fixed rate and moves them smoothly toward goals.

    Every control tick runs the safety checks (link, telemetry freshness, supervisor, geofence) and
    raises SafetyStop when one fails. `on_tick`, if set, is called after every tick (e.g. to feed a
    dashboard), including during takeoff, turns and landing.
    """

    def __init__(self, link: Link, rate_hz: float = 20.0):
        self.link = link
        self.cf = link.cf
        self.tm = link.telemetry
        self.dt = 1.0 / rate_hz
        self.flying = False
        self.on_tick: Callable[[], None] | None = None
        s = self.tm.state()
        self.setpoint = [s.x, s.y, s.z, s.yaw]  # x, y, z [m], yaw [deg]
        self._next_tick = time.monotonic()
        self._safety_landing = False

    # -- mission primitives -------------------------------------------------------------------

    def takeoff(self, height: float, speed: float = 0.3):
        s = self.tm.state()
        log.info('taking off to %.2f m', height)
        self.setpoint = [s.x, s.y, s.z, s.yaw]
        self.flying = True
        while self.setpoint[2] < height:
            self.setpoint[2] = min(height, self.setpoint[2] + speed * self.dt)
            self.tick()
        self._settle(lambda st: abs(st.z - height) < 0.05, 'takeoff altitude', timeout=5.0)

    def turn_to(self, yaw_deg: float, rate: float = 90.0, tol: float = 5.0):
        target = wrap_deg(yaw_deg)
        while True:
            err = wrap_deg(target - self.setpoint[3])
            if abs(err) < 1e-3:
                break
            self.setpoint[3] = wrap_deg(self.setpoint[3] + math.copysign(min(abs(err), rate * self.dt), err))
            self.tick()
        self._settle(lambda st: abs(wrap_deg(target - st.yaw)) < tol, f'heading {target:.0f} deg', timeout=5.0)

    def goto(self, x: float, y: float, z: float, speed: float = 0.4, tol: float = 0.1, face_travel: bool = True):
        """Fly in a straight line to (x, y, z), first turning to face the direction of travel."""
        target = np.array([x, y, z])
        carrot = np.array(self.setpoint[:3])
        dx, dy = target[:2] - carrot[:2]
        if face_travel and math.hypot(dx, dy) > 0.2:
            self.turn_to(math.degrees(math.atan2(dy, dx)))
        timeout = np.linalg.norm(target - carrot) / speed * 2.0 + 5.0
        t_end = time.monotonic() + timeout
        while True:
            pos = self.tm.state().position
            to_go = target - carrot
            dist = np.linalg.norm(to_go)
            if dist > 1e-6:
                step = carrot + to_go / dist * min(dist, speed * self.dt)
                if np.linalg.norm(step - pos) <= LEASH_M:  # don't run away from a drone that lags behind
                    carrot = step
            self.setpoint[:3] = carrot.tolist()
            self.tick()
            if dist <= 1e-6 and np.linalg.norm(pos - target) < tol:
                return
            if time.monotonic() > t_end:
                raise SafetyStop(f'waypoint ({x:.2f}, {y:.2f}, {z:.2f}) not reached within {timeout:.0f} s')

    def hover(self, seconds: float):
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            self.tick()

    def step_velocity(self, velocity, yaw_deg: float | None = None, yaw_rate: float = 90.0):
        """One tick in velocity mode: move the setpoint by velocity * dt (world frame, m/s) and turn
        toward yaw_deg (None holds the heading). The setpoint stays leashed to the drone."""
        pos = self.tm.state().position
        step = np.array(self.setpoint[:3]) + np.asarray(velocity, float) * self.dt
        if np.linalg.norm(step - pos) <= LEASH_M:
            self.setpoint[:3] = step.tolist()
        if yaw_deg is not None:
            err = wrap_deg(yaw_deg - self.setpoint[3])
            limit = yaw_rate * self.dt
            self.setpoint[3] = wrap_deg(self.setpoint[3] + max(-limit, min(limit, err)))
        self.tick()

    def hold_position(self):
        """Pull the setpoint back to where the drone is now, so it stops instead of finishing the
        lead the setpoint had over it."""
        s = self.tm.state()
        self.setpoint[:3] = [s.x, s.y, self.setpoint[2]]

    def land(self, speed: float = 0.25):
        log.info('landing at (%.2f, %.2f)', self.setpoint[0], self.setpoint[1])
        while self.setpoint[2] > 0.05:
            self.setpoint[2] = max(0.05, self.setpoint[2] - speed * self.dt)
            self.tick()
        try:
            self._settle(lambda st: st.z < 0.08, 'touchdown', timeout=3.0)
        finally:
            self.motors_off()
        log.info('landed')

    def motors_off(self):
        self.cf.commander.send_stop_setpoint()
        self.cf.commander.send_notify_setpoint_stop()
        self.flying = False

    # -- control loop ---------------------------------------------------------------------------

    def tick(self):
        """Check safety, send the current setpoint, and wait for the next control period."""
        self._check_safety()
        x, y, z, yaw = self.setpoint
        self.cf.commander.send_position_setpoint(x, y, z, yaw)
        if self.on_tick is not None:
            try:
                self.on_tick()
            except Exception:
                log.exception('on_tick hook failed; disabling it')
                self.on_tick = None
        self._next_tick += self.dt
        delay = self._next_tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            self._next_tick = time.monotonic()  # fell behind (e.g. a GC pause): don't burst to catch up

    def safety_land(self):
        """Land straight down from where the drone is now. Used by fly_mission() after a safety stop."""
        if not self.flying:
            return
        if self.link.lost_reason:
            log.error('cannot land: link is down. The firmware will level off and cut the motors '
                      'once setpoints stop arriving.')
            return
        s = self.tm.state()
        self.setpoint = [s.x, s.y, self.setpoint[2], self.setpoint[3]]
        self._safety_landing = True
        self.land()

    def _settle(self, done: Callable[[State], bool], what: str, timeout: float):
        t_end = time.monotonic() + timeout
        while not done(self.tm.state()):
            if time.monotonic() > t_end:
                raise SafetyStop(f'{what} not reached within {timeout:.0f} s')
            self.tick()

    def _check_safety(self):
        if self.link.lost_reason:
            raise SafetyStop(f'link lost: {self.link.lost_reason}')
        if self._safety_landing:
            return  # already landing; only a dead link should interrupt that
        s = self.tm.state()
        if time.monotonic() - s.stamp > TELEMETRY_STALE_S:
            raise SafetyStop(f'no telemetry for {time.monotonic() - s.stamp:.1f} s')
        if s.supervisor & (SUP_TUMBLED | SUP_CRASHED):
            raise SafetyStop(f'supervisor reports {"crashed" if s.supervisor & SUP_CRASHED else "tumbled"}')
        for axis, (lo, hi) in GEOFENCE.items():
            v = getattr(s, axis)
            if not lo <= v <= hi:
                raise SafetyStop(f'geofence: {axis}={v:.2f} m outside [{lo}, {hi}]')


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)-7s %(message)s',
                        datefmt='%H:%M:%S')


def fly_mission(mission: Callable[[Pilot], None], uri: str = SIM_URI) -> int:
    """Connect, run mission(pilot), and land on any interruption. Returns a process exit code.

    First Ctrl-C lands where the drone is; a second Ctrl-C during that landing cuts the motors.
    """
    try:
        link = Link(uri)
    except (ConnectionError, KeyboardInterrupt) as e:
        log.error('not flying: %s', e or 'interrupted')
        return 1
    with link:
        pilot = Pilot(link)
        try:
            mission(pilot)
            return 0
        except KeyboardInterrupt:
            log.warning('SAFETY STOP: Ctrl-C. Landing now (Ctrl-C again cuts the motors).')
        except SafetyStop as e:
            log.warning('SAFETY STOP: %s. Landing now.', e)
        except Exception:
            log.exception('SAFETY STOP: unexpected error. Landing now.')
        try:
            pilot.safety_land()
        except KeyboardInterrupt:
            log.error('SAFETY STOP: second Ctrl-C. Cutting motors.')
            pilot.motors_off()
        except SafetyStop as e:
            log.error('SAFETY STOP: landing failed (%s). Cutting motors.', e)
            pilot.motors_off()
        return 2
