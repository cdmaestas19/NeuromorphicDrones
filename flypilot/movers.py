"""Moving obstacles: people and a cart that walk and drive through the scene during a flight.

The scene file holds the movers as mocap bodies (scenes/street.xml). The patched simulator moves them as
this module commands over UDP MOVERS_PORT (crazysim.py MoverDriver), and its range sensors and camera
see them like anything else. They are never on the planner's map: dealing with them is the avoidance
layer's job, which is what the fly brain is for.

A scenario (routes.py) gives each mover a Walk: a path, a speed, when to start, and whether to go back
and forth. Starting a walk when the drone comes near a point makes the encounter happen in the same
place on every run. MoverRunner also measures how close the drone came to each mover: the safety
metric for people.
"""
import json
import math
import socket
import xml.etree.ElementTree as ET
from typing import NamedTuple

MOVERS_PORT = 5312       # flight -> simulator (crazysim.py MOVER_PORT)
DRONE_RADIUS_M = 0.07    # Crazyflie 2.x out to the propeller tips


class Walk(NamedTuple):
    mover: str                       # body name in the scene file
    path: list                       # [(x, y), ...]; the mover waits at path[0] until it starts
    speed: float                     # m/s
    start_near: tuple | None = None  # ((x, y), radius): start once the drone comes this close; None: at take-off
    start_after_s: float = 0.0       # ...and this long after that
    loop: bool = False               # back and forth along the path, pausing pause_s at each end
    pause_s: float = 1.5


class Shape(NamedTuple):
    name: str
    kind: str        # 'person', 'cart' or 'mover' (from the name)
    z: float         # body height in the scene file (kept while moving)
    half: tuple      # footprint half extents (x, y) in the body frame; a round footprint has radius = half[0]
    round: bool


def load_movers(path) -> dict:
    """The scene's mocap bodies and their footprints, from the first geom of each (box or round)."""
    try:
        world = ET.parse(path).getroot().find('worldbody')
    except (OSError, ET.ParseError):
        return {}
    out = {}
    for body in world.findall('body') if world is not None else []:
        geoms = body.findall('geom')
        if body.get('mocap') != 'true' or not geoms:
            continue
        name = body.get('name', '')
        z = float(body.get('pos', '0 0 0').split()[2])
        size = [float(v) for v in geoms[0].get('size', '0.1').split()]
        is_box = geoms[0].get('type') == 'box'
        kind = 'person' if name.startswith('person') else 'cart' if name.startswith('cart') else 'mover'
        out[name] = Shape(name, kind, z, (size[0], size[1]) if is_box else (size[0], size[0]), not is_box)
    return out


def gap_to(shape: Shape, x: float, y: float, yaw_deg: float, px: float, py: float) -> float:
    """Horizontal distance from point (px, py) to the edge of the mover's footprint (0 when inside)."""
    dx, dy = px - x, py - y
    if shape.round:
        return max(math.hypot(dx, dy) - shape.half[0], 0.0)
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    lx, ly = dx * c + dy * s, -dx * s + dy * c
    return math.hypot(max(abs(lx) - shape.half[0], 0.0), max(abs(ly) - shape.half[1], 0.0))


class MoverRunner:
    """Plays the scenario's walks, streams the movers' motion to the simulator, and keeps the
    closest-approach and contact counts. update() is called every control tick (20 Hz)."""

    def __init__(self, shapes: dict, walks):
        self.shapes = shapes
        self.walks = [w for w in walks if w.mover in shapes]
        self.missing = [w.mover for w in walks if w.mover not in shapes]
        self._t_go = {w.mover: None for w in self.walks}       # when each walk started moving
        self._t_trigger = {w.mover: None for w in self.walks}
        self.pose = {}                                           # name -> (x, y, yaw_deg, vx, vy)
        self.min_gap = {w.mover: math.inf for w in self.walks}
        self.contacts = {w.mover: 0 for w in self.walks}
        self._touching = {w.mover: False for w in self.walks}
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def update(self, t: float, drone_xy: tuple):
        cmd = {}
        for w in self.walks:
            if self._t_go[w.mover] is None:
                if self._t_trigger[w.mover] is None and (
                        w.start_near is None or math.dist(drone_xy, w.start_near[0]) <= w.start_near[1]):
                    self._t_trigger[w.mover] = t
                if self._t_trigger[w.mover] is not None and t - self._t_trigger[w.mover] >= w.start_after_s:
                    self._t_go[w.mover] = t
            x, y, yaw, vx, vy = self._where(w, None if self._t_go[w.mover] is None else t - self._t_go[w.mover])
            self.pose[w.mover] = (x, y, yaw, vx, vy)
            shape = self.shapes[w.mover]
            cmd[w.mover] = {'p': [x, y, shape.z], 'v': [vx, vy, 0.0],
                            'yaw': yaw - 90.0 if shape.kind == 'cart' else yaw}  # a cart's long axis is its y
            gap = gap_to(shape, x, y, cmd[w.mover]['yaw'], *drone_xy) - DRONE_RADIUS_M
            self.min_gap[w.mover] = min(self.min_gap[w.mover], gap)
            touching = gap <= 0.0
            if touching and not self._touching[w.mover]:
                self.contacts[w.mover] += 1
            self._touching[w.mover] = touching
        if cmd:
            try:
                self._sock.sendto(json.dumps({'movers': cmd}).encode(), ('127.0.0.1', MOVERS_PORT))
            except OSError:
                pass

    def snapshot(self) -> list:
        """For the dashboards: [name, kind, x, y, body yaw (deg), half_x, half_y, round]."""
        out = []
        for name, (x, y, yaw, _, _) in self.pose.items():
            s = self.shapes[name]
            out.append([name, s.kind, x, y, yaw - 90.0 if s.kind == 'cart' else yaw, s.half[0], s.half[1], s.round])
        return out

    def summary(self) -> str:
        if not self.walks:
            return ''
        gaps = ', '.join(f'{n} {g:.2f} m' if math.isfinite(g) else f'{n} -' for n, g in self.min_gap.items())
        hits = sum(self.contacts.values())
        touched = ', '.join(f'{n} x{c}' for n, c in self.contacts.items() if c)
        return f'closest approach (drone edge to mover edge): {gaps}; contacts: {touched if hits else "none"}'

    def close(self):
        self._sock.close()

    @staticmethod
    def _where(w: Walk, elapsed: float | None) -> tuple:
        """(x, y, heading_deg, vx, vy) of a walk `elapsed` seconds after it started (None: not yet)."""
        pts = w.path
        legs = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
        total = sum(legs)
        if elapsed is None or total == 0.0:
            return (*pts[0], _heading(pts[0], pts[1]) if len(pts) > 1 else 0.0, 0.0, 0.0)
        travel = total / w.speed
        direction, s, moving = 1, w.speed * elapsed, True
        if w.loop:
            cycle = 2.0 * (travel + w.pause_s)
            phase = elapsed % cycle
            if phase < travel:
                s = w.speed * phase
            elif phase < travel + w.pause_s:
                s, moving = total, False
            elif phase < 2.0 * travel + w.pause_s:
                s, direction = total - w.speed * (phase - travel - w.pause_s), -1
            else:
                s, moving = 0.0, False  # back at the start, facing the way it goes next
        if s >= total:
            s, moving = total, False
        for (a, b), length in zip(zip(pts, pts[1:]), legs):
            if s <= length or b == pts[-1]:
                f = 0.0 if length == 0.0 else min(s / length, 1.0)
                x, y = a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
                ux, uy = ((b[0] - a[0]) / length, (b[1] - a[1]) / length) if length else (0.0, 0.0)
                v = w.speed * direction if moving else 0.0
                heading = _heading(a, b) if direction > 0 else _heading(b, a)
                return x, y, heading, ux * v, uy * v
            s -= length
        return (*pts[-1], 0.0, 0.0, 0.0)


def _heading(a: tuple, b: tuple) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
