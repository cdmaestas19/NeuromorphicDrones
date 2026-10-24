"""Avoidance policies. Each implements avoidance.AvoidancePolicy; flight scripts pick one by name
from POLICIES and never depend on a particular one.

    geometric     time-to-collision on the four horizontal Multi-ranger rays (the default): what a real
                  Crazyflie with the Multi-ranger deck can run
    geometric360  EXPERIMENTAL: the same rule in every direction, from the 360-degree compound eye. Not
                  ready: it mistakes oblique static surfaces for movers and its maneuver re-triggers in
                  clutter (the square and blocked routes time out)
    flybrain      360-degree compound eye -> looming per eye -> the fly connectome's escape circuit (flybrain/)
"""
import json
import math
import socket
import time
from collections import Counter, deque

import numpy as np

from avoidance import AvoidCommand, AvoidancePolicy, Mode, Sensors

_ZERO = (0.0, 0.0, 0.0)
_RAYS = ('front', 'back', 'left', 'right')


def _direction(forward: float, left: float) -> str:
    """'left', 'back right', ... for a body-frame direction (the dashboard's maneuver text)."""
    fb = 'forward' if forward > 0.38 else 'back' if forward < -0.38 else ''
    lr = 'left' if left > 0.38 else 'right' if left < -0.38 else ''
    return ' '.join(w for w in (fb, lr) if w) or 'aside'


PERSON_CLEAR_M = 1.5   # never climb with anything the eye takes for a person this close (whatever the threat is:
                       # a person walking past a bench can read as one vehicle-sized mover for seconds)
MAP_MARGIN_M = 0.3     # one way round must be this much shorter on the map for the map to choose it when the
                       # policy's own sense can't tell (the brain's left-right is weak, both rays are free)...
MAP_OVERRULE_M = 1.0   # ...and this much shorter to overrule a clear choice (the brain's reflex away from the
                       # looming side: kept unless it means a long way back round)


def _cost(c: float) -> float | str:
    """A map cost for the decision record: metres, or 'blocked' (inf) / '-' (no map)."""
    return round(c, 2) if math.isfinite(c) else 'blocked' if math.isinf(c) else '-'


def _map_choice(cost_a: float, cost_b: float, margin: float = MAP_MARGIN_M) -> int:
    """+1 if the map says way a is the shorter way on to the stop (Sensors.route_cost), -1 if way b, 0 if it
    can't tell: no map, both run into something mapped, or within `margin` of each other. A way that runs
    into something mapped (inf) loses to one that doesn't."""
    if math.isnan(cost_a) or math.isnan(cost_b) or (math.isinf(cost_a) and math.isinf(cost_b)):
        return 0
    if cost_a + margin < cost_b:
        return 1
    if cost_b + margin < cost_a:
        return -1
    return 0


def _camera_look(camera, looming, s: Sensors, frame) -> tuple | None:
    """The camera's sure verdict (kind, p) on the threat the eye is tracking (aimed at its points' centre), or
    None: no model, no frame, the threat out of view, or the network unsure (camera_kind.CameraKind)."""
    pts = looming.threat_points
    if camera is None or not len(pts):
        if camera is not None:
            camera.last = None
        return None
    return camera.classify(frame, s.position, s.yaw_deg, pts.mean(axis=0))


def _camera_text(camera) -> str:
    """What the camera made of the threat, for the dashboard: 'person 0.97', or '' (not looking at one)."""
    if camera is None or camera.last is None:
        return ''
    kind, p, _ = camera.last
    return f'{kind} {p:.2f}'


class _ThreatKinds:
    """What an avoidance episode escapes from, as the compound eye classifies it (eye.SeenObject.kind:
    'static', 'person', 'vehicle', 'unknown'), and whether that allows going up. Up over a vehicle: it is
    too wide and fast to get out of the way of in time (the cart's lane is 1.2 m, the alarm comes ~1.3 s
    before it arrives). Up from static things too. Never up beside a person: round them, not over them (in
    most places flying over people is restricted). 'unknown' (moving, but wider than any vehicle) decides
    nothing either way: in testing these were walls whose visible stretch slid along as the drone moved
    close to them; a person merged with a wall reads as a person whenever they are apart from it. The
    eye's readings are counted over the episode: a vehicle only if they say so VEHICLE_ODDS times as often
    as person. The first readings of a cart can be a sliver of it, which looks like a person, but they are
    few; and a bench that seems to move for a moment is vehicle-sized (fo_headon: it outvoted a second of
    "person" readings, and the drone climbed 1 m from the person)."""

    PERSON_LIKE = frozenset(('person',))
    VEHICLE_ODDS = 2.0

    def __init__(self):
        self.eye = Counter()       # the eye's readings of the threat's kind (its size of it), per tick
        self.camera = Counter()    # the camera's sure verdicts ('person' / 'vehicle'), when it saw the threat

    @property
    def seen(self) -> set:
        return set(self.eye)

    def add(self, kind: str):
        if kind:
            self.eye[kind] += 1

    def add_camera(self, kind: str):
        if kind in ('person', 'vehicle'):
            self.camera[kind] += 1

    CAMERA_MIN = 3   # sure camera readings in an episode before its majority counts: on held-out flights one
                     # stray "vehicle" would have taken the drone up beside a person (r1_headon)

    def verdict(self) -> str:
        """'vehicle', 'person' or '' (neither: static, or nothing known). The camera's majority when it saw the
        threat often enough (it can tell what a thing is; the eye only how big it is), else the eye's sizes."""
        if sum(self.camera.values()) >= self.CAMERA_MIN:
            return 'vehicle' if self.camera['vehicle'] > self.camera['person'] else 'person'
        if self.eye['vehicle'] > self.VEHICLE_ODDS * self.eye['person']:
            return 'vehicle'
        return 'person' if self.eye['person'] else 'vehicle' if self.eye['vehicle'] else ''

    def climb_ok(self) -> bool:
        return self.verdict() != 'person'

    def label(self) -> str:
        v = self.verdict()
        what = ('a vehicle' if v == 'vehicle' else 'a person' if v == 'person' else
                'an unsized mover' if 'unknown' in self.seen else 'static' if self.seen else '')
        return f'{what} (camera)' if what and sum(self.camera.values()) >= self.CAMERA_MIN else what


class GeometricTTCPolicy(AvoidancePolicy):
    """Time-to-collision on the Multi-ranger rays, plus an escape maneuver.

    Signal: for each horizontal ray, TTC = distance / closing speed along that ray, and the signal is
    TTC_REF_S / TTC of the worst ray, capped at 1 (0.5 means impact in 2 s). Anything closer than
    NEAR_M is a threat regardless. The closing speed has two parts:
      - the drone's own motion along the ray: the smaller of the measured and the commanded speed, so
        braking into a planned stop in front of a wall is not a threat but flying on toward it is;
      - the obstacle's own motion toward the drone: how fast the range shrinks beyond what the drone's
        motion explains (range rate over RATE_WINDOW_S). This is what makes a person walking at a
        hovering drone a threat. It is a time-to-contact from a looming distance, the quantity a fly's
        looming neurons estimate from image expansion.

    Why time-to-collision rather than a potential field: it turns the four distances into exactly
    the one "how soon do I hit something" number the supervisor's thresholds need, and it is the
    quantity a fly's looming circuit estimates from vision, so this baseline and the fly-brain
    policy produce comparable signals. A potential field gives a push direction, not a threat
    level, and stalls in front of an obstacle dead ahead on the route.

    Maneuver while the supervisor is in AVOID (heading frozen): get out of the line of approach.
      - threat ahead: sidestep toward the freer side (the other one, once, if that is blocked) until
        the front ray clears, keep going for MARGIN_S (the drone is wider than the rays), fly forward
        past where the front ray hit (sidestepping again if the obstacle shows up ahead), then hold;
      - threat from the side: move forward (or back, if that is freer) across its line of approach;
      - threat from behind: sidestep toward the freer side.
    The signal is held at MANEUVER_SIGNAL until the maneuver is done, because once the drone stops
    approaching, TTC alone says "safe" while the obstacle is still in the way.
    From a vehicle it also climbs, as the fly brain does (_ThreatKinds): what the threat is comes from the
    compound eye when the flight has one (the rays can't tell a cart from a person). Without the eye, or
    from anything else, it never climbs.

    Cost per decision: ~8 divisions, ~80 multiply/add/compare operations, 9 floats of maneuver state and a
    5-sample range history. Plain scalar code that maps directly onto C.
    """

    name = 'Geometric TTC'
    inputs = ('multiranger',)
    needs_camera = False      # the camera only tells it what a threat is (camera_kind.py); it flies without
    description = 'time-to-collision on the 4 horizontal ToF rays (own motion + range rate); get out of the way'

    TTC_REF_S = 1.0
    NEAR_M = 0.2              # anything closer is a full threat, whatever the speed
    MIN_CLOSING = 0.05        # m/s; slower approaches don't count
    MANEUVER_SIGNAL = 0.8
    V_SIDESTEP = 0.3          # m/s
    V_PASS = 0.3
    V_BACK = 0.15
    V_ESCAPE = 0.4            # crossing the line of approach of something coming from the side
    FRONT_CLEAR_M = 1.0       # past this, the front ray no longer sees the obstacle
    FRONT_BLOCKED_M = 0.6     # passing, but something this close ahead again: back to sidestepping
    SIDE_BLOCKED_M = 0.4      # sidestepping toward something this close: switch sides (once)
    MARGIN_S = 0.5            # (the drone's lag behind its setpoint adds ~0.2 m of drift on top)
    MARGIN_MOVING_S = 1.0     # a moving obstacle ahead: keep stepping aside this long after it clears
    PASS_BEYOND_M = 0.6       # beyond the front ray's hit point: obstacle depth + margin (a geometric guess)
    SIDE_CLEAR_M = 0.6        # keep passing while anything is this close beside us
    KEEP_OFF_M = 0.3          # push away from anything closer than this beside us
    ESCAPE_MIN_S = 1.0        # cross the line of approach for at least this long
    ESCAPE_FREE_M = 0.8       # a direction is free to escape into if its ray is clear this far
    SIDE_PASS_M = 0.5         # the map's pass point for a sidestep: this far aside (and past the front ray's hit)
    # Range rate (moving obstacles). The rays update at 10 Hz, the decisions at 20 Hz, so a reading is
    # dated when it first shows up and the rate is a straight-line fit over the last few new readings.
    RATE_SPAN_S = 0.25        # fit over readings spanning at least this long (3+ readings)
    RATE_FIT_TOL_M = 0.04     # every reading within this of the line, or it is a jump between surfaces
    MIN_OBJECT_MPS = 0.3      # slower is timing jitter or the drone's own sway (people walk 0.8-1.4 m/s)
    MAX_OBJECT_MPS = 2.5      # faster is not a person or a cart
    MAX_SIDEWAYS_MPS = 0.15   # a ray is trusted only while the drone moves along it, not across: moving
                              # across, the ray slides over surfaces and corners and its range jumps
    MAX_YAW_RATE_DPS = 25.0   # while turning faster, the rays sweep across surfaces: no range rate
    MAX_FIT_TURN_DEG = 3.0    # ...and no more than this much turn over the readings a rate is fitted to
    V_CLIMB = 1.0             # m/s up while escaping a vehicle, while the up ray sees UP_CLEAR_M of room,
    UP_CLEAR_M = 1.0          # at most CLIMB_MAX_M per episode (the fly brain's numbers)
    CLIMB_MAX_M = 1.4

    def __init__(self):
        from eye import Looming
        self._readings = {ray: deque(maxlen=4) for ray in _RAYS}  # (t first seen, range)
        self._motion = deque(maxlen=12)                           # (t, yaw, v_forward, v_left), ~0.6 s
        self._eye = Looming()      # only to tell what a threat is (_ThreatKinds), never when to avoid it
        self._kind_reading = None
        from camera_kind import CameraKind
        self._camera = CameraKind()
        if self._camera.net is None:
            self._camera = None
        self.inputs = type(self).inputs + (('camera',) if self._camera else ()) + ('map',)
        self.input_notes = {'camera': 'person / vehicle', 'map': 'which side is shorter'}
        self.reset()

    def reset(self):
        self._phase = None
        self._kinds, self._z0, self._climbing = _ThreatKinds(), 0.0, False
        self._side_from, self._map_cost = '', (math.nan, math.nan)
        for q in self._readings.values():
            q.clear()
        self._motion.clear()

    def _watch(self, s: Sensors, frame):
        """Keep the eye's object tracks going every tick (a mover's kind needs its motion), and note what
        looms most while avoiding: its kind by the eye's size of it, and by the camera when it sees it."""
        if s.eye is not None and s.eye is not self._kind_reading:
            self._kind_reading = s.eye
            self._eye.update(s.eye, s.yaw_deg, s.route_stop_m)
        seen = _camera_look(self._camera, self._eye, s, frame) if s.eye is not None else None
        if s.mode is Mode.AVOID and s.eye is not None:
            self._kinds.add(self._eye.threat_kind)
            if seen:
                self._kinds.add_camera(seen[0])

    def _climb(self, s: Sensors) -> float:
        """Up while escaping a vehicle (not once the maneuver is done)."""
        self._climbing = (self._kinds.verdict() == 'vehicle' and self._phase != 'done'
                          and self._eye.nearest_person(s.position) > PERSON_CLEAR_M
                          and s.ranges.get('up', math.inf) > self.UP_CLEAR_M
                          and s.position[2] < self._z0 + self.CLIMB_MAX_M)
        return self.V_CLIMB if self._climbing else 0.0

    def _label(self) -> str:
        if self._phase is None:
            return '-'
        by_map = ' (shorter way on: map)' if self._side_from == 'map' and self._phase in ('sidestep', 'margin', 'escape') else ''
        return self._labels[self._phase] + by_map + (', up over it (a vehicle)' if self._climbing else '')

    def _threat(self, distance: float, closing: float) -> float:
        """TTC_REF / TTC for one ray, written as TTC_REF * closing / distance (one division)."""
        if distance < self.NEAR_M:
            return 1.0
        if closing <= self.MIN_CLOSING:
            return 0.0
        t = self.TTC_REF_S * closing / distance  # inf distance -> 0
        return t if t < 1.0 else 1.0

    def _object_closing(self, s: Sensors) -> tuple:
        """Each ray's obstacle speed toward the drone [m/s], beyond what the drone's own motion explains.
        0 when unknown: too few readings, turning, moving across the ray, out of range, or not a straight-
        line approach (a jump between surfaces)."""
        r, t = s.ranges, s.t
        vf, vl = s.velocity_body[0], s.velocity_body[1]
        m = self._motion
        if m and (t - m[-1][0] > 0.5 or abs((s.yaw_deg - m[-1][1] + 180.0) % 360.0 - 180.0)
                  > self.MAX_YAW_RATE_DPS * max(t - m[-1][0], 0.05)):
            self.reset_rates()  # a gap in the decisions (a turn in place), or turning: start over
        m.append((t, s.yaw_deg, vf, vl))
        out = [0.0, 0.0, 0.0, 0.0]
        for k, ray in enumerate(_RAYS):
            d, q = r[ray], self._readings[ray]
            if d == math.inf:
                q.clear()
                continue
            if not q or d != q[-1][1] or t - q[-1][0] >= 0.15:  # a new reading, or the range really is steady
                q.append((t, d))
            sideways = abs(vl) if k < 2 else abs(vf)
            if len(q) < 3 or q[-1][0] - q[0][0] < self.RATE_SPAN_S or sideways > self.MAX_SIDEWAYS_MPS:
                continue
            (t0, d0), (t1, d1) = q[0], q[-1]
            slope = (d1 - d0) / (t1 - t0)
            if any(abs(d0 + slope * (tk - t0) - dk) > self.RATE_FIT_TOL_M for tk, dk in q):
                continue
            n = sf = sl = 0.0  # the drone's mean motion over the same span
            turned = 0.0
            for tm, yaw_m, mf, ml in m:
                if t0 <= tm <= t1:
                    n, sf, sl = n + 1.0, sf + mf, sl + ml
                    turned = max(turned, abs((s.yaw_deg - yaw_m + 180.0) % 360.0 - 180.0))
            if turned > self.MAX_FIT_TURN_DEG:
                continue  # the ray swept across surfaces while the drone turned
            own = (sf if k < 2 else sl) / n if n else 0.0
            obj = -slope - (own if k in (0, 2) else -own)
            if self.MIN_OBJECT_MPS <= obj <= self.MAX_OBJECT_MPS:
                out[k] = obj
        return tuple(out)

    def reset_rates(self):
        for q in self._readings.values():
            q.clear()
        self._motion.clear()

    def step(self, s: Sensors, frame: np.ndarray | None) -> AvoidCommand:
        self._watch(s, frame)
        r = s.ranges
        vf, vl = s.velocity_body[0], s.velocity_body[1]
        jf, jl = s.intended_velocity_body[0], s.intended_velocity_body[1]
        o_f, o_b, o_l, o_r = self._object_closing(s)
        # Closing speed along each ray: own motion (the smaller of measured and intended) + the obstacle's.
        c_f, c_b = min(vf, jf) + o_f, -max(vf, jf) + o_b
        c_l, c_r = min(vl, jl) + o_l, -max(vl, jl) + o_r
        t_f, t_b = self._threat(r['front'], c_f), self._threat(r['back'], c_b)
        t_l, t_r = self._threat(r['left'], c_l), self._threat(r['right'], c_r)
        worst, ttc_signal, closing = 'front', t_f, c_f
        if t_b > ttc_signal:
            worst, ttc_signal, closing = 'back', t_b, c_b
        if t_l > ttc_signal:
            worst, ttc_signal, closing = 'left', t_l, c_l
        if t_r > ttc_signal:
            worst, ttc_signal, closing = 'right', t_r, c_r
        self._threats = {'front': t_f, 'back': t_b, 'left': t_l, 'right': t_r}

        if s.mode is not Mode.AVOID:
            self._phase, self._kinds = None, _ThreatKinds()
            signal, dodge = ttc_signal, _ZERO
        else:
            if self._phase is None:
                self._start(s, worst, {'front': o_f, 'back': o_b, 'left': o_l, 'right': o_r}[worst] > 0.0)
            dodge = self._maneuver(s)
            signal = ttc_signal if self._phase == 'done' else max(ttc_signal, self.MANEUVER_SIGNAL)
        moving = max(o_f, o_b, o_l, o_r)
        info = {'ttc_signal': ttc_signal, 'worst_ray': worst,
                'ttc_worst': r[worst] / closing if closing > self.MIN_CLOSING else math.inf,
                'threat_front': t_f, 'threat_back': t_b, 'threat_left': t_l, 'threat_right': t_r,
                'object_mps': moving,
                'object_ray': _RAYS[(o_f, o_b, o_l, o_r).index(moving)] if moving > 0.0 else None,
                'maneuver': self._label(), 'threat_kind': self._kinds.label(), 'side_from': self._side_from,
                'camera_kind': _camera_text(self._camera),
                'map_left': _cost(self._map_cost[0]), 'map_right': _cost(self._map_cost[1])}
        return AvoidCommand(signal, dodge, info)

    def _start(self, s: Sensors, worst: str, moving: bool):
        self._z0 = s.position[2]
        yaw = s.yaw_deg * (math.pi / 180.0)
        self._hx, self._hy = math.cos(yaw), math.sin(yaw)
        self._t_phase = s.t
        r = s.ranges
        side = 1.0 if r['left'] >= r['right'] else -1.0
        self._side_from = 'rays'
        if worst in ('front', 'back') and s.route_cost is not None:
            # Which way round is the shorter way on to the stop (the planner's map), unless the ray on that
            # side already sees something close (an unmapped crate: the map doesn't know it)
            ahead = (r['front'] if math.isfinite(r['front']) else 0.5) + self.PASS_BEYOND_M if worst == 'front' else 0.0
            x, y, lx, ly = s.position[0], s.position[1], -self._hy, self._hx
            self._map_cost = tuple(s.route_cost(x, y, x + self._hx * ahead + lx * k * self.SIDE_PASS_M,
                                                y + self._hy * ahead + ly * k * self.SIDE_PASS_M) for k in (1.0, -1.0))
            pick = _map_choice(*self._map_cost)
            if pick and r['left' if pick > 0 else 'right'] >= self.SIDE_BLOCKED_M:
                side, self._side_from = float(pick), 'map'
        if worst == 'front':
            # Something coming at us keeps coming: step aside faster and further, then let it pass
            # instead of flying on past it.
            self._moving = moving
            self._v_side = self.V_ESCAPE if moving else self.V_SIDESTEP
            self._margin_s = self.MARGIN_MOVING_S if moving else self.MARGIN_S
            self._side, self._flipped = side, False
            hit = r['front'] if math.isfinite(r['front']) else 0.5
            self._pass_until = self._hx * s.position[0] + self._hy * s.position[1] + hit + self.PASS_BEYOND_M
            self._phase = 'sidestep'
            self._side_labels()
            return
        self._threat_ray = worst
        if worst == 'back':
            self._escape = (0.0, side * self.V_ESCAPE)
            label = f'escape {"left" if side > 0 else "right"} (threat behind)'
        else:
            # Something is coming from the side: cross its line of approach, forward if that is free
            # and where we were heading, otherwise back.
            ahead_ok, back_ok = r['front'] > self.ESCAPE_FREE_M, r['back'] > self.ESCAPE_FREE_M
            forward = ahead_ok and (s.intended_velocity_body[0] >= 0.0 or not back_ok)
            self._escape = (self.V_ESCAPE if forward else -self.V_ESCAPE, 0.0)
            label = f'escape {"forward" if forward else "back"} (threat {worst})'
        self._phase = 'escape'
        self._labels = {'escape': label, 'done': 'clear, holding'}

    def _side_labels(self):
        name = 'left' if self._side > 0 else 'right'
        what = ', it is moving' if self._moving else ''
        self._labels = {'sidestep': f'sidestep {name}{what}', 'margin': f'sidestep {name} (margin){what}',
                        'pass': 'pass obstacle', 'done': 'clear, letting it pass' if self._moving else 'clear, holding'}

    def _maneuver(self, s: Sensors) -> tuple:
        r, t, phase = s.ranges, s.t, self._phase
        forward = lateral = 0.0
        if phase == 'pass' and r['front'] < self.FRONT_BLOCKED_M:
            phase = self._phase = 'sidestep'  # still in the way: the sidestep was not far enough
        if phase == 'sidestep':
            if not self._flipped and r['left' if self._side > 0 else 'right'] < self.SIDE_BLOCKED_M:
                self._side, self._flipped = -self._side, True  # that side is blocked: try the other, once
                self._side_labels()
            lateral = self._side * self._v_side
            if r['front'] > self.FRONT_CLEAR_M:
                self._phase, self._t_phase = 'margin', t
        elif phase == 'margin':
            lateral = self._side * self._v_side
            if t - self._t_phase >= self._margin_s:
                self._phase = 'done' if self._moving else 'pass'
        elif phase == 'pass':
            forward = self.V_PASS
            along = self._hx * s.position[0] + self._hy * s.position[1]
            if along >= self._pass_until and min(r['left'], r['right']) > self.SIDE_CLEAR_M:
                self._phase = 'done'
                forward = 0.0
        elif phase == 'escape':
            forward, lateral = self._escape
            if t - self._t_phase >= self.ESCAPE_MIN_S and self._threats[self._threat_ray] < 0.25:
                self._phase = 'done'
                forward = lateral = 0.0
        # In every phase: don't brush against anything beside or ahead.
        if r['left'] < self.KEEP_OFF_M:
            lateral -= self.V_SIDESTEP
        if r['right'] < self.KEEP_OFF_M:
            lateral += self.V_SIDESTEP
        if r['front'] < self.NEAR_M:
            forward = -self.V_BACK
        elif r['back'] < self.NEAR_M:
            forward = self.V_BACK
        return (forward, lateral, self._climb(s))


class GeometricTTC360Policy(GeometricTTCPolicy):
    """Time-to-collision all the way round, from the compound eye: the geometric rule with the same
    360-degree sense the fly brain gets, so nothing between four rays goes unseen.

    For every ommatidium column (5 degrees apart, at flying height: eye.ring) the time to collision is
    the shorter of:
      - own motion: the time until the drone reaches what that column sees, but only if its path passes
        within PATH_HALF_WIDTH_M of it. Something the drone flies past is not a threat, however close it
        gets. The drone's velocity counts at most at the commanded speed, so braking into a planned stop
        is not a threat either (as on the rays);
      - a mover: distance / (its speed toward the drone + the drone's toward it). Its speed toward the
        drone is how fast the column's range shrinks beyond what the drone's own motion explains, fitted
        over RATE_SPAN_S of eye readings (20 Hz). Something coming at the drone is taken to be on a
        collision course.
    Signal = TTC_REF_S / the worst column's TTC, capped at 1; anything within NEAR_M is a full threat.
    The maneuver is the four-ray policy's, on corridors instead of rays (eye.corridors): how far the
    drone could fly front / back / left / right before something within PATH_HALF_WIDTH_M of its path.
    Without the eye (it stopped) it decides on the Multi-ranger rays, as GeometricTTCPolicy.
    """

    name = 'Geometric TTC (360 eye)'
    inputs = ('360 eye',)
    description = 'time-to-collision in every direction (compound eye: own motion + range rate); get out of the way'

    PATH_HALF_WIDTH_M = 0.3   # the drone (0.07 m to its propeller tips) plus margin: what its path sweeps
    MIN_MOVER_MPS = 0.45      # faster toward the drone than its own motion explains: a mover. Higher than on
                              # the rays: an oblique surface's range changes a bit faster than a point's
    RATE_WINDOW_S = 0.4       # eye readings used for a range rate (at least RATE_SPAN_S of them)

    def __init__(self):
        super().__init__()
        self._eye_hist = deque(maxlen=12)  # (sim time, nearest distance per world-frame column)
        self._vel_hist = deque(maxlen=12)  # (t, vx, vy): the drone's motion over the same span
        self._eye_reading = None

    def reset(self):
        super().reset()
        if hasattr(self, '_eye_hist'):
            self._eye_hist.clear()
            self._vel_hist.clear()

    def step(self, s: Sensors, frame: np.ndarray | None) -> AvoidCommand:
        if s.eye is None:
            cmd = super().step(s, frame)
            cmd.info['maneuver'] = f'{cmd.info["maneuver"]} (no eye: Multi-ranger)'
            return cmd
        from eye import corridors, ring
        self._watch(s, frame)
        if s.eye is not self._eye_reading:
            self._eye_reading = s.eye
            self._eye_hist.append((s.eye[0], ring(s.eye, 0.0)[1]))
        self._vel_hist.append((s.t, s.velocity_world[0], s.velocity_world[1]))
        az_w, near = ring(s.eye, 0.0)  # world-frame columns: the eye doesn't turn with the drone
        near = np.where(np.isfinite(near), near, 1e3)  # nothing there: far away (keeps the arithmetic finite)
        a = np.radians(az_w)
        ux, uy = np.cos(a), np.sin(a)

        # The drone's motion, at most at the commanded speed: braking into a planned stop is no threat.
        yaw = math.radians(s.yaw_deg)
        jf, jl = s.intended_velocity_body[0], s.intended_velocity_body[1]
        jx, jy = jf * math.cos(yaw) - jl * math.sin(yaw), jf * math.sin(yaw) + jl * math.cos(yaw)
        vx, vy = s.velocity_world[0], s.velocity_world[1]
        speed, intended = math.hypot(vx, vy), math.hypot(jx, jy)
        k = min(1.0, intended / speed) if speed > 1e-6 else 0.0
        wx, wy = vx * k, vy * k
        w = speed * k

        ttc = np.full(near.shape, math.inf)
        if w > self.MIN_CLOSING:
            along = near * (ux * wx + uy * wy) / w           # how far ahead along the path
            miss = np.abs(near * (ux * wy - uy * wx) / w)    # how far beside the path
            hit = (along > 0.0) & (miss < self.PATH_HALF_WIDTH_M)
            ttc = np.where(hit, along / w, ttc)
        mover = self._mover_speeds(ux, uy)
        closing = mover + np.maximum(ux * wx + uy * wy, 0.0)
        ttc = np.minimum(ttc, np.where(mover > 0.0, near / np.maximum(closing, 1e-6), math.inf))
        threat = np.where(near < self.NEAR_M, 1.0, np.minimum(self.TTC_REF_S / ttc, 1.0))

        az_b = (az_w - s.yaw_deg + 180.0) % 360.0 - 180.0   # relative to the heading, + = left
        quadrant = np.where(np.abs(az_b) <= 45.0, 0, np.where(np.abs(az_b) >= 135.0, 1, np.where(az_b > 0.0, 2, 3)))
        self._threats = {ray: float(threat[quadrant == q].max(initial=0.0)) for q, ray in enumerate(_RAYS)}
        worst_k = int(np.argmax(threat))
        ttc_signal = float(threat[worst_k])
        worst = _RAYS[int(quadrant[worst_k])]
        corr = corridors(az_b, near, self.PATH_HALF_WIDTH_M)
        corr['up'] = s.ranges.get('up', math.inf)
        sc = s._replace(ranges=corr)  # the maneuver reads corridors where the four-ray policy read rays

        if s.mode is not Mode.AVOID:
            self._phase, self._kinds = None, _ThreatKinds()
            signal, dodge = ttc_signal, _ZERO
        else:
            if self._phase is None:
                self._start(sc, worst, bool(mover[worst_k] > 0.0))
            dodge = self._maneuver(sc)
            signal = ttc_signal if self._phase == 'done' else max(ttc_signal, self.MANEUVER_SIGNAL)
        m = int(np.argmax(mover))
        info = {'ttc_signal': ttc_signal, 'worst_ray': worst, 'worst_az': float(az_b[worst_k]),
                'ttc_worst': float(ttc[worst_k]),
                'threat_front': self._threats['front'], 'threat_back': self._threats['back'],
                'threat_left': self._threats['left'], 'threat_right': self._threats['right'],
                'object_mps': float(mover[m]), 'object_ray': _RAYS[int(quadrant[m])] if mover[m] > 0.0 else None,
                'maneuver': self._label(), 'threat_kind': self._kinds.label(), 'side_from': self._side_from,
                'camera_kind': _camera_text(self._camera),
                'map_left': _cost(self._map_cost[0]), 'map_right': _cost(self._map_cost[1])}
        return AvoidCommand(signal, dodge, info)

    def _mover_speeds(self, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
        """Per world-frame column: how fast what it sees approaches beyond the drone's own motion (m/s),
        0 where unknown or not a mover. A straight-line fit of the range over the last RATE_WINDOW_S of
        readings; a reading off the line (an edge passing, a jump between surfaces) makes it unknown."""
        h = self._eye_hist
        if len(h) < 3:
            return np.zeros(ux.shape)
        t_now = h[-1][0]
        span = [e for e in h if t_now - e[0] <= self.RATE_WINDOW_S]
        if len(span) < 3 or span[-1][0] - span[0][0] < self.RATE_SPAN_S:
            return np.zeros(ux.shape)
        T = np.array([e[0] for e in span])
        D = np.stack([e[1] for e in span])
        with np.errstate(invalid='ignore'):  # inf - inf where a column sees nothing
            slope = (D[-1] - D[0]) / (T[-1] - T[0])
            fits = np.isfinite(D).all(axis=0) & (np.abs(D - (D[0] + np.outer(T - T[0], slope))) <= self.RATE_FIT_TOL_M).all(axis=0)
        slope = np.where(fits, slope, 0.0)
        vel = [(vx, vy) for t, vx, vy in self._vel_hist if self._vel_hist[-1][0] - t <= self.RATE_WINDOW_S]
        mx, my = (sum(v[0] for v in vel) / len(vel), sum(v[1] for v in vel) / len(vel)) if vel else (0.0, 0.0)
        obj = np.where(fits, -slope - (ux * mx + uy * my), 0.0)
        return np.where((obj >= self.MIN_MOVER_MPS) & (obj <= self.MAX_OBJECT_MPS), obj, 0.0)


class FlyBrainPolicy(AvoidancePolicy):
    """The fly's own escape circuit decides: 360-degree compound eye -> connectome -> descending neurons -> dodge.

    Each tick: the compound eye (eye.py; the simulator's ring of ommatidia, all the way round) gives each eye's
    looming drive, 0..1. The drive goes to the brain server (flybrain/brain_server.py, its own process), which
    runs the FlyWire escape subcircuit of the Shiu et al. LIF model: the drive sets the Poisson rate of that
    eye's LC4/LPLC2 looming detectors, and the reply is the descending neurons' spike counts. The brain
    answers one tick late (~50 ms).

    Readout, when: signal = the faster giant fiber's (DNp01) rate / GF_FULL_HZ, capped at 1, smoothed over
    ~150 ms; the giant fiber is the fly's escape command. The supervisor's thresholds apply as for any policy.
    Readout, which way: away from the side whose looming-responsive descending neurons (DNp02, DNp04, DNp06,
    DNp11) fire more: these are strongly lateralized, the giant fibers much less so (flybrain/README.md).
    When they barely favour a side (a threat nearly straight ahead or behind, in both eyes), the side is the
    eye's: away from where the threat will pass (eye.Looming.pass_side).
    The dodge is a burst, repeated while the giant fibers stay alarmed, then a hold (see step()). Like a
    fly's escape takeoff it goes up as well as away, except beside a person: up over a vehicle or a static
    thing, round a person, never over them (_ThreatKinds, from the eye's size of the threat). Away from a static threat: sideways to that side, and
    forward or back away from where the looming is, keeping at least the speed it had unless the threat is
    ahead (eye.Looming.front_back). Away from a moving one (a person, the cart): straight out of its path,
    across its line of travel (_mover_escape). The brain gets one input channel per eye, so its outputs
    can't tell a threat ahead from one behind, nor where it is heading; those parts of the direction are the
    eye's, not the connectome's (a retinotopic input to the looming detectors would hand more to the brain).
    The compound eye is this policy's only sense (the flight's own rules around every policy use the range
    sensors and the eye too).
    """

    name = 'Fly brain (connectome)'
    inputs = ('360 eye', 'fly brain', 'multiranger')
    needs_eye = True         # no fallback: the eye is its sense of looming
    needs_camera = False     # the camera tells it what a threat is (camera_kind.py) when it is in view
    brake_on_avoid = False   # an escape keeps the drone's momentum (see step())
    EYE_GAIN = (1.35, 1.0)   # drive -> LC4/LPLC2 rate, left and right eye. The connectome's left pathway needs
                             # ~1.35x the looming to bring the giant fibers to 50 Hz (flybrain/README.md); without
                             # this, a threat seen by the left eye alone was flagged ~0.5 s later than on the right
    description = '360-degree looming -> FlyWire LC4/LPLC2 -> giant fibers + looming DNs (LIF, 3,118 neurons)'

    PORT = 5320              # flybrain/brain_server.py
    TICK_S = 0.05
    GF_FULL_HZ = 100.0       # giant-fiber rate that counts as a full threat (signal 1.0)
    SMOOTH = 0.35            # weight of the newest tick in the smoothed rates (~150 ms)
    STEER_DNS = ('DNp02', 'DNp04', 'DNp06', 'DNp11')  # looming-responsive, lateralized: which way
    V_ESCAPE = 0.4           # m/s sideways, away from the looming side; and up to this forward or back, away
                             # from the looming (or on at the current speed, see step()). The geometric
                             # policy's escape speed too.
    ESCAPE_S = 1.0           # one escape burst (~0.4 m sideways), then hold
    V_CLIMB = 1.0            # m/s up during a burst, as a fly's escape takeoff goes up and away, when the up
    UP_CLEAR_M = 1.0         # range sensor sees this much room. ~1.5 s clears a person or the cart from 0.5 m;
                             # the flight comes back down once clear below
    CLIMB_MAX_M = 1.4        # an episode climbs at most this far (back-to-back bursts would reach the ceiling)
    REARM_S = 0.0            # after a burst, another at once if still alarmed: holding still in a threat's path
                             # (a cart's lane) is the worst place to be
    SIDE_CLEAR = 0.5         # steering DNs this lateralized ((L - R) / (L + R)) pick the side; less, and the side
                             # the threat will pass on does (eye.Looming.pass_side)...
    SIDE_SETTLE_S = 0.4      # ...following the eye for this long into the burst: a mover that just set off
                             # (a person at the end of their walk) takes the tracker ~0.15 s to measure
    ON_ITS_LINE_M = 0.03     # a mover whose path passes this close to the drone's centre is coming straight at it
                             # (further off, the drone escapes on the side it will be on: less to go)
    MOMENTUM_S = 0.5         # ...where "will be" adds this much of its own motion across the mover's path: turning
                             # round from 0.4 m/s takes ~0.9 s and ~0.2 m the wrong way (a cart's lane is 1.2 m wide)
    FLIP_M = 0.12            # while settling, the chosen side changes only if the mover's path is clearly this far
                             # to the other side (the tracker's estimate of a path wobbles by ~0.1 m)
    V_ESCAPE_MAX = 0.8       # m/s: out of a mover's path faster than V_ESCAPE when that is what it takes to be out
    CLEAR_M = 0.15           # before it arrives (_escape_speed), this far past its edge. (The setpoint leash holds
    REACT_S = 0.5            # the drone to ~0.7 m/s.) The drone takes ~REACT_S to get going
    PASS_BESIDE_M = 0.35     # the map's pass points (which way round a standing threat is shorter): this far
    PASS_PAST_M = 0.2        # beside the eye's points of it, this far past its far end
    MAP_REACH_M = 0.6        # ...and for a mover, this far out of its path either way
    BURST_AGAIN = 0.5        # another burst if the giant fibers are still at alarm level (the supervisor's
                             # enter threshold): a threat that keeps coming, e.g. someone overtaking from behind
    REPLY_TIMEOUT_S = 0.5    # no answer for this long: the brain is down (the dashboard says so)

    def __init__(self, port: int = PORT):
        from camera_kind import CameraKind
        from eye import Looming
        self.looming = Looming()
        self._camera = CameraKind()
        if self._camera.net is None:
            self._camera = None
        self.inputs = type(self).inputs + (('camera',) if self._camera else ()) + ('map',)
        self.input_notes = {'multiranger': 'up: room to climb', 'camera': 'person / vehicle',
                            'map': 'which side is shorter'}
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(self.REPLY_TIMEOUT_S)
        self._addr = ('127.0.0.1', port)
        self._tick, self._reading = 0, None
        self.gf = [0.0, 0.0]     # smoothed giant-fiber rates, Hz (left, right)
        self.steer = [0.0, 0.0]  # smoothed steering-DN rates, Hz (left, right)
        self.brain_ms = None
        self.bursts = 0
        self.late_ticks = 0      # ticks the brain's newest answer was older than one tick (a slow brain)
        self.side_sources = Counter()  # per episode, what chose the side: brain, eye, map, motion (summary())
        self._stats = deque(maxlen=20 * 60 * 30)  # (drive L, drive R, GF L, GF R, brain ms) per tick, <= 30 min
        self.brain_ok = self._reset_brain()
        self.reset()

    def summary(self) -> str:
        """For the end-of-flight log: what the eyes and the giant fibers did, and what the brain cost."""
        if not self._stats:
            return 'fly brain: no decisions'
        a = np.array(self._stats)
        p = lambda c: f'median {np.median(a[:, c]):.2f} / 95% {np.percentile(a[:, c], 95):.2f} / max {a[:, c].max():.2f}'  # noqa: E731
        return (f'fly brain over {len(a)} ticks: eye drive L {p(0)}, R {p(1)}; giant fiber Hz L {p(2)}, R {p(3)}; '
                f'{self.bursts} escape bursts; brain compute {a[:, 4].mean():.1f} ms per 50 ms tick; '
                f'answers later than one tick: {self.late_ticks} tick(s); side chosen by: '
                + (', '.join(f'{k} {n}' for k, n in self.side_sources.most_common()) or 'no episodes'))

    def _reset_brain(self) -> bool:
        """Every neuron back at rest; False if the brain server doesn't answer."""
        self._sock.settimeout(5.0)
        try:
            self._sock.sendto(json.dumps({'type': 'reset'}).encode(), self._addr)
            self._sock.recv(4096)
            return True
        except OSError:
            return False
        finally:
            self._sock.setblocking(False)  # from now on the flight never waits for the brain
            self._last_reply = time.monotonic()

    def reset(self):
        if getattr(self, '_episode_t0', None) is not None:  # an episode ended: what chose its side
            self.side_sources[self._escape_from if self._escape_world is not None else self._side_from] += 1
        self._side = self._burst_end = None  # the brain itself is only reset per flight
        self._along = self._v_forward = 0.0
        self._side_from, self._escape_from, self._map_cost = 'brain', '', (math.nan, math.nan)
        self._out_of_path = self._escape_world = None
        self._v_out = self.V_ESCAPE
        self._will_be, self._n0 = [], None  # the drone's projected offsets from the mover's line, while choosing
        self._kinds = _ThreatKinds()
        self._episode_t0, self._episode_z0 = None, 0.0

    def _map_sides(self, s: Sensors) -> tuple:
        """(route on to the stop if the drone goes round the threat on its left, ... on its right), from the
        planner's map: via a point PASS_BESIDE_M beside the eye's points of the threat and PASS_PAST_M past
        their far end. The threat itself isn't on the map (else the planner would have gone round it):
        these points are what make going round its far side cost more. nan without a map or a threat."""
        pts = self.looming.threat_points
        if s.route_cost is None or not len(pts):
            return math.nan, math.nan
        x, y = s.position[0], s.position[1]
        yaw = math.radians(s.yaw_deg)
        fx, fy, lx, ly = math.cos(yaw), math.sin(yaw), -math.sin(yaw), math.cos(yaw)
        rel = pts - np.array((x, y))
        along, lat = rel @ np.array((fx, fy)), rel @ np.array((lx, ly))
        past = max(float(along.max()), 0.0) + self.PASS_PAST_M
        out = []
        for k in (1.0, -1.0):
            need = max(float((k * lat).max()), 0.0) + self.PASS_BESIDE_M
            out.append(s.route_cost(x, y, x + fx * past + lx * k * need, y + fy * past + ly * k * need))
        return tuple(out)

    def _mover_escape(self, s: Sensors, settling: bool) -> tuple | None:
        """For a moving threat, the way out of its path: across its line of travel, to the side of that line
        the drone will be on by the time it could turn round (where it is plus MOMENTUM_S of its own motion
        across the line: reversing in a cart's lane loses the time it has); right on the line and not moving
        across it, to the brain's side. Once chosen, the side is kept for the episode (only a clear reading
        while settling may change it): flipping sides between bursts leaves the drone in the way. The side
        follows the mean of the readings so far, not the newest: the line's offset at the drone is the mover's
        heading error times its distance (~0.4 m for 10 degrees on the cart 2.5 m off), and one bad reading
        must not turn the drone round.
        Returns (body-frame unit vector (forward, left), speed: _escape_speed); None if the threat stands
        still (the brain's side and the eye's front/back then make the escape)."""
        vx, vy = self.looming.threat_velocity
        speed = math.hypot(vx, vy)
        yaw = math.radians(s.yaw_deg)
        body = lambda nx, ny: (nx * math.cos(yaw) + ny * math.sin(yaw), -nx * math.sin(yaw) + ny * math.cos(yaw))  # noqa: E731
        prev = self._escape_world
        if speed == 0.0:
            if prev is None:
                return None
            return body(*prev), self._v_out  # the tracker lost its speed for a moment: keep going the way chosen
        nx, ny = -vy / speed, vx / speed
        mx, my = self.looming.threat_miss
        off = -(nx * mx + ny * my)  # the drone's offset from the mover's line, along n
        will_be = off + (nx * s.velocity_world[0] + ny * s.velocity_world[1]) * self.MOMENTUM_S
        if settling or prev is None:  # kept along the episode's first n (the mover may turn round: n flips)
            if self._n0 is None:
                self._n0 = (nx, ny)
            k = 1.0 if nx * self._n0[0] + ny * self._n0[1] >= 0.0 else -1.0
            self._will_be.append(k * will_be)
            will_be = k * sum(self._will_be) / len(self._will_be)
        x, y = s.position[0], s.position[1]
        if s.route_cost is not None:  # the way on to the stop after getting out of its path either way
            self._map_cost = tuple(s.route_cost(x, y, x + k * nx * self.MAP_REACH_M, y + k * ny * self.MAP_REACH_M)
                                   for k in (1.0, -1.0))
        pick = _map_choice(*self._map_cost)
        if prev is not None and not (settling and abs(will_be) > self.FLIP_M):
            sign = 1.0 if nx * prev[0] + ny * prev[1] >= 0.0 else -1.0  # the side already chosen
        elif abs(will_be) > self.ON_ITS_LINE_M:
            sign, self._escape_from = (1.0 if will_be > 0.0 else -1.0), 'eye'  # away from its line
        elif pick:  # a tie: the shorter way on to the stop (the map)
            sign, self._escape_from = float(pick), 'map'
        else:  # a tie the map can't break: on the way the drone is going across its line (no turning round),
            f, l = body(nx, ny)  # else the route's, else the brain's side (if the line runs that way at all)
            for along in (nx * s.velocity_world[0] + ny * s.velocity_world[1],
                          f * s.intended_velocity_body[0] + l * s.intended_velocity_body[1], l * self._side):
                if abs(along) > 0.05:
                    break
            sign, self._escape_from = (1.0 if along >= 0.0 else -1.0), 'motion'
        if (prev is None or settling) and abs(will_be) < self.FLIP_M and pick == -sign:
            # Nearly tied, and the map says the chosen way runs into something it knows (or is much longer):
            # crossing the mover's line this close to it costs little. Further off, out of its path comes first
            # (and the flight slows a dodge rather than fly it into something, avoid_demo._dodge_safe).
            sign, self._escape_from = -sign, 'map'
        nx, ny = nx * sign, ny * sign
        self._escape_world = (nx, ny)
        self._v_out = self._escape_speed(s, (vx / speed, vy / speed), speed, (nx, ny))
        return body(nx, ny), self._v_out

    def _escape_speed(self, s: Sensors, u: tuple, speed: float, n: tuple) -> float:
        """How fast to get out of a mover's path along n (world): from the drone out past the mover's far edge
        on that side (the eye's points of it, across its path; the side it doesn't show taken as wide as the
        one it does) plus CLEAR_M, in the time until its leading edge reaches the drone (moving along u at
        `speed`, less the drone's own motion along u), less REACT_S to get going. Between V_ESCAPE and
        V_ESCAPE_MAX: a person walking at the drone needs V_ESCAPE, the cart coming down its lane more."""
        pts = self.looming.threat_points
        if len(pts) == 0:
            return self.V_ESCAPE
        rel = pts - np.asarray(s.position[:2])
        across, along = rel @ np.asarray(n), rel @ np.asarray(u)
        centre = n[0] * self.looming.threat_miss[0] + n[1] * self.looming.threat_miss[1]  # its line, along n
        half = float(np.abs(across - centre).max())
        need = centre + half + self.CLEAR_M  # how far along n the drone has to go
        if need <= 0.0:
            return self.V_ESCAPE
        closing = speed - (u[0] * s.velocity_world[0] + u[1] * s.velocity_world[1])
        lead = float(along.max())  # its leading edge, along its path from the drone (< 0: still coming)
        t_arrive = -lead / closing if lead < 0.0 and closing > 0.05 else 0.0
        v = need / max(t_arrive - self.REACT_S, 0.25)
        return min(max(v, self.V_ESCAPE), self.V_ESCAPE_MAX)

    def _newest_reply(self) -> dict | None:
        """The newest of the brain's replies that have arrived (older ones are dropped), or None if none
        has since the last tick: the brain is slow this tick, and the policy keeps its last readout. Not
        answering for REPLY_TIMEOUT_S counts as the brain being down (brain_ok)."""
        newest = None
        while True:
            try:
                reply = json.loads(self._sock.recv(4096))
            except (BlockingIOError, ValueError):
                break
            except OSError:
                break
            if 'tick' in reply and (newest is None or reply['tick'] > newest['tick']):
                newest = reply
        now = time.monotonic()
        if newest is not None:
            self._last_reply, self.brain_ok = now, True
        elif now - self._last_reply > self.REPLY_TIMEOUT_S:
            self.brain_ok = False
        return newest

    def step(self, s: Sensors, frame: np.ndarray | None) -> AvoidCommand:
        reading = s.eye
        if reading is not None and reading is not self._reading:
            self._reading = reading
            self.looming.update(reading, s.yaw_deg, s.route_stop_m)  # with an efference copy of the route
        drive = tuple(min(d * g, 1.0) for d, g in zip(self.looming.drive(), self.EYE_GAIN))
        self._sock.sendto(json.dumps({'type': 'step', 'tick': self._tick, 'drive': drive}).encode(), self._addr)
        if self._tick > 0:  # usually the reply to the previous tick (the brain finished it ~25 ms ago)
            reply = self._newest_reply()
            if reply is not None:
                self.late_ticks += self._tick - 1 - reply['tick']
                self.brain_ms = reply.get('compute_ms')
                dn = reply.get('dn', {})
                steer = [sum(n for k, n in dn.items() if k.split(' ')[0] in self.STEER_DNS and side in k)
                         for side in ('(left)', '(right)')]
                for k in (0, 1):
                    self.gf[k] += self.SMOOTH * (reply['gf'][k] / self.TICK_S - self.gf[k])
                    self.steer[k] += self.SMOOTH * (steer[k] / self.TICK_S - self.steer[k])
        self._tick += 1

        threat = max(self.gf) / self.GF_FULL_HZ
        signal = threat if threat < 1.0 else 1.0
        dodge, maneuver = _ZERO, '-'
        seen = _camera_look(self._camera, self.looming, s, frame)  # what the threat is, if the camera sees it
        if s.mode is Mode.AVOID:
            # An escape is a burst, as a fly's is a jump: ESCAPE_S away from the looming side, then hold and let
            # the threat pass. Sliding on would make whatever the drone now moves toward loom, and keep the
            # giant fibers going. Another burst (side chosen afresh) only while they are still at alarm level.
            if self._burst_end is None or (s.t > self._burst_end + self.REARM_S and signal >= self.BURST_AGAIN):
                if self._side is None:  # once per episode: changing sides between bursts leaves it in the way
                    self._episode_t0, self._episode_z0 = s.t, s.position[2]
                    lateral = (self.steer[0] - self.steer[1]) / max(self.steer[0] + self.steer[1], 1e-6)
                    brain = -1.0 if self.steer[0] >= self.steer[1] else 1.0  # away from the stronger DNs
                    self._map_cost = self._map_sides(s)
                    pick = _map_choice(*self._map_cost)  # +1: round it on the left is shorter, -1: on the right
                    if abs(lateral) >= self.SIDE_CLEAR:
                        self._side, self._side_from = brain, 'brain'
                        if _map_choice(*self._map_cost, MAP_OVERRULE_M) == -brain:  # the brain's way is far longer
                            self._side, self._side_from = -brain, 'map'
                    elif pick:
                        # The brain's left-right is weak (nearly straight ahead or behind): the shorter way on
                        # to the stop, round the threat on that side (the planner's map)...
                        self._side, self._side_from = float(pick), 'map'
                    elif self.looming.pass_side:
                        # ...or, the map can't tell: away from the side it will pass on (the eye's collision course)
                        self._side, self._side_from = -float(self.looming.pass_side), 'eye'
                    else:
                        self._side, self._side_from = brain, 'brain'
                    if pick == -self._side and math.isinf(self._map_cost[0 if self._side > 0 else 1]):
                        self._side, self._side_from = -self._side, 'map'  # that way runs into something mapped
                ahead, behind = self.looming.front_back
                along = 2.0 * (behind - ahead) / max(ahead, behind) if max(ahead, behind) > 0.0 else 0.0
                self._along = min(max(along, -1.0), 1.0)  # +1: all of it behind, -1: all ahead
                # Unless the threat is ahead, don't brake: stopping in front of something coming from
                # behind or from the side (a cart down the road) only lets it close faster.
                self._v_forward = self._along * self.V_ESCAPE
                if self._along > -0.3:
                    self._v_forward = max(self._v_forward, s.velocity_body[0])
                self._burst_end = s.t + self.ESCAPE_S
                self.bursts += 1
            settling = s.t < self._episode_t0 + self.SIDE_SETTLE_S
            self._kinds.add(self.looming.threat_kind)
            if seen:
                self._kinds.add_camera(seen[0])
            if self._side_from == 'eye' and self.looming.pass_side and settling:
                self._side = -float(self.looming.pass_side)  # early on, follow the eye as its estimate settles
            self._out_of_path = self._mover_escape(s, settling)
            if s.t < self._burst_end:
                room = s.ranges.get('up', math.inf) > self.UP_CLEAR_M
                climb = (self.V_CLIMB if room and s.position[2] < self._episode_z0 + self.CLIMB_MAX_M
                         and self._kinds.climb_ok() and self.looming.nearest_person(s.position) > PERSON_CLEAR_M
                         else 0.0)
                what = self._kinds.label()
                up = (', up over it' if climb else ', not over it') if what.startswith(('a vehicle', 'a person')) else (
                    ', up' if climb else '')
                if self._out_of_path:  # a mover: get out of its path, across it
                    (f, l), v = self._out_of_path
                    dodge = (f * v, l * v, climb)
                    maneuver = f'escape out of its path ({_direction(f, l)}, {v:.1f} m/s){up} ({what or "it is moving"})'
                else:
                    dodge = (self._v_forward, self._side * self.V_ESCAPE, climb)
                    along = ' and forward' if self._along > 0.3 else ' and back' if self._along < -0.3 else ''
                    why = (f'looming on the {"right" if self._side > 0 else "left"}' if self._side_from == 'brain'
                           else 'the shorter way on: map' if self._side_from == 'map'
                           else f'it passes on the {"right" if self._side > 0 else "left"}')
                    maneuver = f'escape {"left" if self._side > 0 else "right"}{along}{up} ({why})'
            else:
                maneuver = 'holding, letting it pass'
        else:
            self._burst_end = None
        self._stats.append((drive[0], drive[1], self.gf[0], self.gf[1], self.brain_ms or 0.0))
        info = {'ttc_signal': signal, 'gf_left': self.gf[0], 'gf_right': self.gf[1],
                'steer_left': self.steer[0], 'steer_right': self.steer[1],
                'drive_left': drive[0], 'drive_right': drive[1], 'loom_left': self.looming.peak[0],
                'loom_right': self.looming.peak[1], 'loom_ahead': self.looming.front_back[0],
                'loom_behind': self.looming.front_back[1], 'brain_ms': self.brain_ms, 'brain_ok': self.brain_ok,
                'maneuver': maneuver, 'threat_kind': self._kinds.label(), 'camera_kind': _camera_text(self._camera),
                'side_from': (self._escape_from if self._escape_world is not None else self._side_from)
                if s.mode is Mode.AVOID else '',
                'map_left': _cost(self._map_cost[0]), 'map_right': _cost(self._map_cost[1])}
        return AvoidCommand(signal, dodge, info)


POLICIES = {
    'geometric': GeometricTTCPolicy,
    'geometric360': GeometricTTC360Policy,
    'flybrain': FlyBrainPolicy,
}
