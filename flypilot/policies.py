"""Avoidance policies. Each implements avoidance.AvoidancePolicy; flight scripts pick one by name
from POLICIES and never depend on a particular one.

Planned behind the same interface: a looming-detector policy and the connectome policy, both
reading the camera frame instead of the range sensors.
"""
import math
from collections import deque

import numpy as np

from avoidance import AvoidCommand, AvoidancePolicy, Mode, Sensors

_ZERO = (0.0, 0.0, 0.0)
_RAYS = ('front', 'back', 'left', 'right')


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

    Cost per decision: ~8 divisions, ~80 multiply/add/compare operations, 9 floats of maneuver state and a
    5-sample range history. Plain scalar code that maps directly onto C.
    """

    name = 'Geometric TTC'
    inputs = ('multiranger',)
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

    def __init__(self):
        self._readings = {ray: deque(maxlen=4) for ray in _RAYS}  # (t first seen, range)
        self._motion = deque(maxlen=12)                           # (t, yaw, v_forward, v_left), ~0.6 s
        self.reset()

    def reset(self):
        self._phase = None
        for q in self._readings.values():
            q.clear()
        self._motion.clear()

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
            self._phase = None
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
                'maneuver': '-' if self._phase is None else self._labels[self._phase]}
        return AvoidCommand(signal, dodge, info)

    def _start(self, s: Sensors, worst: str, moving: bool):
        yaw = s.yaw_deg * (math.pi / 180.0)
        self._hx, self._hy = math.cos(yaw), math.sin(yaw)
        self._t_phase = s.t
        r = s.ranges
        side = 1.0 if r['left'] >= r['right'] else -1.0
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
        return (forward, lateral, 0.0)


POLICIES = {
    'geometric': GeometricTTCPolicy,
}
