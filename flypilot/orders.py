"""Delivery orders that arrive during the flight: the stops are not known before take-off.

A real delivery drone gets its next stop from a customer's app. Here the customer uses the Stops window
(dashboard.py --stops) or double-clicks in the MuJoCo window; the patched simulator forwards each clicked
point over UDP. Stops can also be given at launch (avoid_demo.py --at X,Y). OrderBook keeps the queue,
checks every request, and keeps the planned route (planner.py) for display; the flight
(avoid_demo.py --route live, i.e. launch.cmd deliver) asks it for the next stop.

    click / double-click the floor      new delivery stop
    click a stop                        remove it (a stop not done yet is cancelled)
    click H (the take-off spot)         return home and land once the queue is done (again: undo)

Requests arrive as JSON on UDP ORDERS_PORT, from anything: {"cmd": "pick", "x", "y", "z", "eye"} (a click;
"eye" = camera position, so a click on a marker ball selects that stop), "add" {x, y}, "remove" {n},
"home", "cancel_all", "clear_finished", "map" {name}.

The book sends markers back to the simulator, drawn only in the 3D view: the drone's camera and range
sensors never see them.
"""
import json
import logging
import math
import socket
import time
from dataclasses import dataclass

from planner import MAPS, Planner

log = logging.getLogger('flypilot')

ORDERS_PORT = 5310   # anyone -> flight: requests (crazysim.py PICK_PORT sends double-clicks here)
MARKERS_PORT = 5311  # flight -> simulator: markers to draw (crazysim.py MARKER_PORT)
MARKERS_REFRESH_S = 1.0  # the simulator drops markers that are not refreshed for 3 s

FLOOR_MAX_Z = 0.05      # a click higher than this landed on an obstacle, not the floor
SELECT_RADIUS_M = 0.25  # a click this close to home or to a stop selects it
MARKER_HIT_M = 0.12     # a click ray passing this close to a stop's marker ball selects that stop
KEEP_OUT_M = 0.35       # room a stop needs from any obstacle or wall, to arrive and hover there
FENCE_MARGIN_M = 0.1    # how far inside the flight geofence (a safety tripwire) a stop must be

QUEUED, ACTIVE, DELIVERED, SKIPPED, CANCELLED, REMOVED = 'queued', 'active', 'delivered', 'skipped', 'cancelled', 'removed'
OPEN = (QUEUED, ACTIVE)
SHOWN = (QUEUED, ACTIVE, DELIVERED, SKIPPED)
RGB = {QUEUED: (1.0, 0.55, 0.1), ACTIVE: (1.0, 0.9, 0.15), DELIVERED: (0.3, 0.85, 0.35), SKIPPED: (0.9, 0.25, 0.25)}
HOME_RGB = (0.25, 0.55, 1.0)


@dataclass
class Order:
    n: int
    x: float
    y: float
    state: str = QUEUED


def obstacle_gap(x: float, y: float, z: float, obstacles) -> tuple[float, str]:
    """Horizontal distance from (x, y) to the nearest obstacle spanning height z, and its name.
    `obstacles` as from dashboard.load_obstacles: (kind, pos, size, name), unrotated boxes and cylinders."""
    best, name = math.inf, ''
    for kind, pos, size, label in obstacles:
        half_h = size[2] if kind == 'box' else size[1]
        if not pos[2] - half_h <= z <= pos[2] + half_h:
            continue
        if kind == 'box':
            gap = math.hypot(max(abs(x - pos[0]) - size[0], 0.0), max(abs(y - pos[1]) - size[1], 0.0))
        else:
            gap = max(math.hypot(x - pos[0], y - pos[1]) - size[0], 0.0)
        if gap < best:
            best, name = gap, label
    return best, name


def _ray_gap(eye, hit, p) -> float:
    """Distance from point p to the segment eye -> hit."""
    d = [h - e for e, h in zip(eye, hit)]
    dd = sum(c * c for c in d)
    t = 0.0 if dd == 0 else max(0.0, min(1.0, sum((pc - e) * c for pc, e, c in zip(p, eye, d)) / dd))
    return math.dist(p, [e + t * c for e, c in zip(eye, d)])


class OrderBook:
    """The delivery queue: takes requests, hands out stops in order, plans the route on show."""

    def __init__(self, altitude: float, fence: dict, obstacles=(), map_name: str = 'all',
                 cruise: float | None = None):
        """altitude: where stops are checked and flown to (the drop-off height); cruise: the height the
        planner plans the paths between stops at (default: the same)."""
        self.altitude, self.fence, self.obstacles = altitude, fence, list(obstacles)
        self.planner = Planner(self.obstacles, fence, cruise or altitude, map_name)
        self.map_version = 0     # bumped on every map change, so the flight knows to re-plan
        self.home = (0.0, 0.0)
        self.drone = (0.0, 0.0)  # updated by the flight each tick
        self.orders: list[Order] = []
        self.return_home = False
        self.leg = []            # the path the drone is flying now (set by the flight)
        self.last_event = 'click the map or double-click the floor to add a stop'
        self._count = 0
        self._preview: list = []
        self._rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._rx.bind(('127.0.0.1', ORDERS_PORT))
        except OSError as e:
            raise SystemExit(f'cannot listen for orders on udp port {ORDERS_PORT} ({e}); is another flight running?')
        self._rx.setblocking(False)
        self._tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._dirty, self._markers_t = True, -math.inf  # markers need resending
        self._preview_dirty = True                        # later legs need re-planning

    # -- requests ---------------------------------------------------------------------------------

    def add(self, x: float, y: float) -> Order | None:
        why = self._unsuitable(x, y)
        if why:
            self._event(f'refused ({x:.1f}, {y:.1f}): {why}')
            return None
        self._count += 1
        order = Order(self._count, x, y)
        self.orders.append(order)
        self._event(f'stop {order.n} added at ({x:.1f}, {y:.1f})')
        return order

    def pick(self, x: float, y: float, z: float = 0.0, eye=None):
        """A click: on H, toggle return home; on a stop, remove it; on the floor, add a stop."""
        shown = [o for o in self.orders if o.state in SHOWN]
        if eye is not None:  # a 3D click: the ray may pass through a stop's marker ball before hitting something
            on_ray = [o for o in shown if _ray_gap(eye, (x, y, z), (o.x, o.y, self.altitude)) < MARKER_HIT_M]
            if on_ray:
                return self.remove(min(on_ray, key=lambda o: math.dist(eye, (o.x, o.y, self.altitude))).n)
        if z > FLOOR_MAX_Z:
            self._event(f'that hit an obstacle ({z:.1f} m up): click the floor')
            return
        if math.hypot(x - self.home[0], y - self.home[1]) < SELECT_RADIUS_M:
            return self.toggle_home()
        near = [o for o in shown if math.hypot(x - o.x, y - o.y) < SELECT_RADIUS_M]
        if near:
            return self.remove(min(near, key=lambda o: math.hypot(x - o.x, y - o.y)).n)
        self.add(x, y)

    def remove(self, n: int):
        for o in self.orders:
            if o.n == n and o.state in SHOWN:
                o.state = CANCELLED if o.state in OPEN else REMOVED
                self._event(f'stop {n} {"cancelled" if o.state == CANCELLED else "removed"}')
                return

    def toggle_home(self):
        self.return_home = not self.return_home
        self._event('will return home when the queue is done' if self.return_home else 'return home undone')

    def cancel_all(self):
        n = 0
        for o in self.orders:
            if o.state in OPEN:
                o.state, n = CANCELLED, n + 1
        self._event(f'{n} open stop(s) cancelled')

    def clear_finished(self):
        n = 0
        for o in self.orders:
            if o.state in (DELIVERED, SKIPPED):
                o.state, n = REMOVED, n + 1
        self._event(f'{n} finished stop(s) cleared')

    def set_map(self, name: str):
        if name not in MAPS or name == self.planner.map_name:
            return
        self.planner.set_map(name)
        self.map_version += 1
        self._event(f'planner map: {name}')

    def _unsuitable(self, x: float, y: float) -> str | None:
        (x0, x1), (y0, y1) = self.fence['x'], self.fence['y']
        m = FENCE_MARGIN_M
        if not (x0 + m <= x <= x1 - m and y0 + m <= y <= y1 - m):
            return 'outside the flight area'
        gap, name = obstacle_gap(x, y, self.altitude, self.obstacles)  # every obstacle, mapped or not
        if gap < KEEP_OUT_M:
            return f'{gap:.2f} m from {name or "an obstacle"}'
        if not self.planner.reachable(self.home, (x, y)):
            return f'no way there on the "{self.planner.map_name}" map'
        return None

    # -- the flight's side ------------------------------------------------------------------------

    def next(self) -> Order | None:
        """The oldest queued stop, now marked active; None when nothing is waiting."""
        for order in self.orders:
            if order.state == QUEUED:
                order.state = ACTIVE
                self._dirty = self._preview_dirty = True
                return order
        return None

    def finish(self, order: Order, state: str):
        if order.state == ACTIVE:
            order.state = state
            self._event(f'stop {order.n} {state}')

    def set_leg(self, path: list):
        self.leg, self._dirty = list(path), True

    def count(self, *states: str) -> int:
        return sum(o.state in states for o in self.orders)

    def poll(self):
        """Apply requests that arrived, and refresh the simulator's markers. Called every control tick."""
        while True:
            try:
                raw = self._rx.recv(4096)
            except OSError:  # nothing more queued
                break
            try:
                self._handle(json.loads(raw))
            except (ValueError, KeyError, TypeError, AttributeError):
                log.warning('ignored a malformed order message: %r', raw[:80])
        now = time.monotonic()
        if self._preview_dirty:
            self._preview, self._preview_dirty = self._plan_preview(), False
        if self._dirty or now - self._markers_t > MARKERS_REFRESH_S:
            self._send({'markers': self._markers()})
            self._dirty, self._markers_t = False, now

    def _handle(self, msg: dict):
        cmd = msg.get('cmd')
        if cmd == 'pick':
            eye = msg.get('eye')
            self.pick(float(msg['x']), float(msg['y']), float(msg.get('z', 0.0)),
                      [float(c) for c in eye] if eye else None)
        elif cmd == 'add':
            self.add(float(msg['x']), float(msg['y']))
        elif cmd == 'remove':
            self.remove(int(msg['n']))
        elif cmd == 'home':
            self.toggle_home()
        elif cmd == 'cancel_all':
            self.cancel_all()
        elif cmd == 'clear_finished':
            self.clear_finished()
        elif cmd == 'map':
            self.set_map(str(msg['name']))
        else:
            raise ValueError(cmd)

    def route(self) -> list:
        """The rest of the route as legs (lists of points): the one being flown, then the planned ones."""
        return ([self.leg] if self.leg else []) + self._preview

    def snapshot(self) -> dict:
        """For the dashboards."""
        return {'home': list(self.home), 'return_home': self.return_home, 'event': self.last_event,
                'map': self.planner.map_name, 'route': self.route(),
                'orders': [[o.n, o.x, o.y, o.state] for o in self.orders if o.state in SHOWN]}

    def close(self):
        self._send({'markers': []})
        self._rx.close()
        self._tx.close()

    # -- internals --------------------------------------------------------------------------------

    def _event(self, what: str):
        self.last_event, self._dirty, self._preview_dirty = what, True, True
        log.info('orders: %s', what)

    def _plan_preview(self) -> list:
        """Planned legs after the current one: queued stops in order, then home if asked (a few ms each)."""
        active = [o for o in self.orders if o.state == ACTIVE]
        frm = (active[0].x, active[0].y) if active else self.drone
        legs = []
        targets = [(o.x, o.y) for o in self.orders if o.state == QUEUED] + ([self.home] if self.return_home else [])
        for to in targets:
            path = self.planner.plan(frm, to)
            legs.append(path or [frm, to])
            frm = to
        return legs

    def _send(self, msg: dict):
        try:
            self._tx.sendto(json.dumps(msg).encode(), ('127.0.0.1', MARKERS_PORT))
        except OSError:
            pass

    def _markers(self) -> list:
        z = self.altitude
        hx, hy = self.home
        out = [{'kind': 'disc', 'pos': [hx, hy, 0.004], 'size': 0.15, 'label': 'H',
                'rgba': [*HOME_RGB, 0.85 if self.return_home else 0.45]}]
        for o in self.orders:
            if o.state not in SHOWN:
                continue
            rgb = RGB[o.state]
            out += [{'kind': 'disc', 'pos': [o.x, o.y, 0.004], 'size': 0.1, 'rgba': [*rgb, 0.5]},
                    {'kind': 'line', 'from': [o.x, o.y, 0.0], 'to': [o.x, o.y, z], 'width': 1, 'rgba': [*rgb, 0.8]},
                    {'kind': 'sphere', 'pos': [o.x, o.y, z], 'size': 0.035, 'rgba': [*rgb, 0.95], 'label': str(o.n)}]
        for k, leg in enumerate(self.route()):
            rgba = [1.0, 0.9, 0.2, 0.9] if k == 0 and self.leg else [1.0, 0.6, 0.15, 0.55]
            for (ax, ay), (bx, by) in zip(leg, leg[1:]):
                out.append({'kind': 'line', 'from': [ax, ay, z], 'to': [bx, by, z], 'width': 2, 'rgba': rgba})
        return out
