"""The avoidance layer: the contract every avoidance policy implements, and the state machine that
decides when the drone is avoiding.

Once per control tick (20 Hz):
    Sensors + camera frame  -->  AvoidancePolicy.step()  -->  AvoidCommand(signal, dodge)
    signal + Multi-ranger clearance  -->  AvoidanceSupervisor  -->  NOMINAL or AVOID
    NOMINAL: the route navigator flies the drone.  AVOID: the drone flies AvoidCommand.dodge.

Any policy (the geometric rule now; a looming detector or the connectome later) implements
AvoidancePolicy and nothing else changes. The supervisor never looks inside a policy.
"""
import enum
import math
from abc import ABC, abstractmethod
from collections import deque
from typing import NamedTuple

import numpy as np

# The per-tick types are NamedTuples of plain floats: cheap to build 20x a second (no NumPy arrays or
# dataclass machinery on the hot path), and each maps one-to-one onto a C struct for an onboard port.
Vec3 = tuple  # (x, y, z) floats


class Mode(str, enum.Enum):
    NOMINAL = 'NOMINAL'
    AVOID = 'AVOID'


class Sensors(NamedTuple):
    """What a policy may read each tick (besides the camera frame)."""
    t: float                      # time.monotonic() [s]
    position: Vec3                # world x, y, z [m]
    yaw_deg: float                # heading, 0 = +x, counter-clockwise positive
    velocity_world: Vec3          # world vx, vy, vz [m/s]
    velocity_body: Vec3           # forward, left, up [m/s]
    intended_velocity_body: Vec3  # what the route navigator is asking for right now (body frame) [m/s]
    ranges: dict                  # Multi-ranger, body frame: front/back/left/right/up [m]; inf = nothing within 4 m
    mode: 'Mode'                  # supervisor mode on the previous tick, so a policy can run a maneuver


def make_sensors(t: float, position: Vec3, yaw_deg: float, velocity_world: Vec3, intended_velocity_world: Vec3,
                 ranges: dict, mode: Mode) -> Sensors:
    """Build one tick's Sensors, rotating the world-frame velocities into the body frame.
    Shared by the flight (avoid_demo.py) and the offline replay (bench.py), so both time the same code."""
    yaw = yaw_deg * (math.pi / 180.0)
    c, s = math.cos(yaw), math.sin(yaw)
    vx, vy, vz = velocity_world
    ix, iy, iz = intended_velocity_world
    return Sensors(t, position, yaw_deg, velocity_world, (vx * c + vy * s, vy * c - vx * s, vz),
                   (ix * c + iy * s, iy * c - ix * s, iz), ranges, mode)


class AvoidCommand(NamedTuple):
    signal: float                 # 0..1: how strongly the policy wants to avoid. The supervisor's only input from it.
    dodge: Vec3                   # body-frame velocity [m/s] (forward, left, up) to fly while in AVOID
    info: dict                    # diagnostics for the dashboard: numbers or short strings (not used to decide)


class AvoidancePolicy(ABC):
    """Decides WHEN and WHICH WAY to dodge. The firmware's controller keeps the drone stable."""

    name = 'unnamed policy'
    inputs: tuple = ()            # which sensors it reads: 'multiranger', 'camera'
    description = ''

    def reset(self):
        """Called before a flight and after every avoidance episode. Default: nothing to reset."""

    @abstractmethod
    def step(self, sensors: Sensors, frame: np.ndarray | None) -> AvoidCommand:
        """One control tick. `frame` is the newest AI-deck camera image (H x W uint8) or None."""


class AvoidanceSupervisor:
    """Explicit NOMINAL <-> AVOID state machine with hysteresis, a minimum dwell and a resume gate.

    NOMINAL -> AVOID    as soon as signal >= enter.
    AVOID -> NOMINAL    only once signal < exit AND clearance > clearance_m have both held, without
                        interruption, for dwell_s; and never sooner than dwell_s after entering AVOID.

    With a single threshold, a signal hovering near it flips the mode on every noisy sample. The gap
    between enter and exit, plus the dwell, is what stops that chatter. Clearance comes from the
    Multi-ranger whichever policy is deciding, as a policy-independent safety gate on resuming.
    """

    def __init__(self, enter: float = 0.5, exit: float = 0.25, dwell_s: float = 0.4, clearance_m: float = 0.5):
        if not exit < enter:
            raise ValueError('exit threshold must be below the enter threshold (that gap is the hysteresis)')
        self.enter, self.exit, self.dwell_s, self.clearance_m = enter, exit, dwell_s, clearance_m
        self.mode = Mode.NOMINAL
        self.mode_since: float | None = None
        self.episodes = 0
        self.last_episode_s = 0.0
        self.gate = {'signal_ok': True, 'clear_ok': True, 'held_s': 0.0}
        self._ok_since: float | None = None

    def update(self, t: float, signal: float, clearance: float) -> bool:
        """Advance one tick. Returns True when the mode changed."""
        if self.mode_since is None:
            self.mode_since = t
        signal_ok, clear_ok = signal < self.exit, clearance > self.clearance_m
        if self.mode is Mode.NOMINAL:
            self.gate = {'signal_ok': signal_ok, 'clear_ok': clear_ok, 'held_s': 0.0}
            if signal >= self.enter:
                self._switch(Mode.AVOID, t)
                self.episodes += 1
                return True
            return False

        if signal_ok and clear_ok:
            if self._ok_since is None:
                self._ok_since = t
        else:
            self._ok_since = None
        held = 0.0 if self._ok_since is None else t - self._ok_since
        self.gate = {'signal_ok': signal_ok, 'clear_ok': clear_ok, 'held_s': held}
        if t - self.mode_since >= self.dwell_s and held >= self.dwell_s:
            self.last_episode_s = t - self.mode_since
            self._switch(Mode.NOMINAL, t)
            return True
        return False

    def time_in_mode(self, t: float) -> float:
        return 0.0 if self.mode_since is None else t - self.mode_since

    def _switch(self, mode: Mode, t: float):
        self.mode, self.mode_since, self._ok_since = mode, t, None


# Body-frame unit vectors (forward, left) of the four horizontal Multi-ranger rays.
HORIZONTAL_RAYS = {'front': (1.0, 0.0), 'back': (-1.0, 0.0), 'left': (0.0, 1.0), 'right': (0.0, -1.0)}
CLEAR_OUT_MPS = 0.15


def clear_out(ranges: dict, clearance_m: float) -> tuple:
    """Body-frame velocity (forward, left, 0) that backs the drone away from whatever is inside the
    supervisor's resume clearance.

    The flight adds it to the dodge while a policy reports no threat (signal below exit) but the
    supervisor cannot resume because something is closer than clearance_m. Without it, a policy that
    finishes its maneuver next to a wall holds there, the gate never opens, and the episode times out.
    It belongs to the avoidance layer, not to any policy, so every policy gets it. Pinned between two
    things on opposite sides (a gap too narrow for the clearance), it slides out lengthwise instead,
    toward the more open end.
    """
    f = l = 0.0
    close = {ray: ranges[ray] < clearance_m + 0.1 for ray in HORIZONTAL_RAYS}
    for ray, (bf, bl) in HORIZONTAL_RAYS.items():
        if close[ray]:
            f, l = f - bf, l - bl
    if f == 0.0 and l == 0.0 and any(close.values()):
        if close['left'] and close['right']:
            f = 1.0 if ranges['front'] >= ranges['back'] else -1.0
        else:
            l = 1.0 if ranges['left'] >= ranges['right'] else -1.0
    n = (f * f + l * l) ** 0.5
    return (0.0, 0.0, 0.0) if n == 0.0 else (CLEAR_OUT_MPS * f / n, CLEAR_OUT_MPS * l / n, 0.0)


class CostMeter:
    """Wall-clock cost of each avoidance decision (building Sensors + policy + supervisor), so every
    feature added to the decision path shows what it costs."""

    def __init__(self, keep: int = 4000):
        self._recent = deque(maxlen=keep)
        self.count = 0
        self._total_us = 0.0
        self.max_us = 0.0

    def add(self, ns: int):
        us = ns / 1000.0
        self._recent.append(us)
        self.count += 1
        self._total_us += us
        self.max_us = max(self.max_us, us)

    def summary(self) -> dict:
        """mean / median / p99 / max in microseconds. Sorts the recent window: call it ~1 Hz, not per tick."""
        if not self._recent:
            return {'count': 0}
        s = sorted(self._recent)
        return {'count': self.count, 'mean_us': self._total_us / self.count, 'median_us': s[len(s) // 2],
                'p99_us': s[min(len(s) - 1, int(0.99 * len(s)))], 'max_us': self.max_us}
