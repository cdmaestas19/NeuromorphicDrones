"""The drone's compound eye: what is around it, all the way round, and what is looming.

A fly sees almost all the way round through ~750 ommatidia per eye, ~5 degrees apart. In the simulator the
drone gets the same (crazysim.py CompoundEye): an ommatidium every 5 degrees of azimuth, and of elevation from
straight down to 30 degrees up, each measuring the distance to what it points at, 20 times a second, on a level
world-aligned frame (a stabilized eye, as a fly stabilizes its head; turning doesn't sweep it across the
scene). EyeSource receives that over UDP. The flight hands it to every policy (avoidance.Sensors.eye) and
uses it itself: ring() for the clearance all round, clear_below() before descending, CollisionCourse to
remember the static obstacles it met.

Looming turns two successive readings into, per ommatidium, the looming rate of what it sees: -d'/d =
1 / time-to-contact, the quantity the fly's looming detectors (LC4, LPLC2) respond to. Only what is on a
collision course counts (CollisionCourse: LPLC2 ignores an image that slides as it grows); what the drone
flies past, or level over (the floor), gives none. Each eye is the half of the world on its side of the
heading, with 15 degrees of frontal overlap as in flies, cut into receptive fields 30 degrees wide. The eye's
drive is its most looming field's looming rate (90th percentile) / LOOM_REF, 0..1: the brain makes it the
Poisson rate of that eye's LC4/LPLC2 neurons.

An ideal front end: looming from the simulator's geometry, not computed from images. Looming from the
AI-deck camera's images was tried first: its optic-flow estimate read ~1-2/s of "looming" while hovering
still (noise) and up to ~15/s in turns, which kept the giant fibers firing. Image-based looming (a T4/T5
motion model) is its own research step; this isolates what the connectome does with looming.
"""
import math
import socket
import struct
import threading
import time
from collections import deque
from typing import NamedTuple

import numpy as np

PORT = 5330           # crazysim.py EYE_PORT
MAGIC = b'CSE1'       # rows -30..+30 degrees (the first eye; old recordings)
MAGIC2 = b'CSE2'      # rows from straight down to +30 (crazysim.py EYE_ELEVATION_DEG): a fly sees below it too
ELEVATIONS = {13: (-30, 30), 25: (-90, 30)}  # rows -> (lowest, highest) elevation in degrees, 5 apart
MAX_RANGE_M = 4.0     # the simulator's "nothing there"
SELF_M = 0.12         # closer than this is the drone itself (propellers)
JUMP_M = 0.12         # a bigger change between readings (> 2.4 m/s) is an edge passing, not an approach
MIN_ELEVATION_DEG = -10  # ommatidia lower than this see mostly floor, which looms whenever the drone sinks
FIELD_DEG = 30.0      # receptive field width
OVERLAP_DEG = 15.0    # frontal overlap of the two eyes
LOOM_REF = 5.0        # 1/s: looming rate for full drive. The giant fibers reach the avoid threshold (50 Hz) at
                      # drive ~0.065 with both eyes, ~0.1 with either eye alone (the left calibrated, see
                      # policies.FlyBrainPolicy.EYE_GAIN): at ~3 and ~2 s to contact. (8 before CollisionCourse
                      # and the efference copy: what the drone flew past, or toward a stop, also loomed)


RING_ELEVATION_DEG = (-5.0, 15.0)  # the rows that see what is around the drone at its own height (the floor is
                                   # beyond 4 m from -5 degrees up at 0.5 m altitude)
STALE_S = 0.3                      # a reading this old (wall clock) is no reading: the eye stopped
PASS_DEAD_M = 0.02                 # a closest approach this near the drone's centre line is "straight at it"


class EyeSource:
    """Receives the simulator's compound-eye readings in a thread; latest() is the newest
    (sim time, position, distances[n_el, n_az]), or None if there is none newer than STALE_S."""

    def __init__(self, port: int = PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(('127.0.0.1', port))
        self._sock.settimeout(0.5)
        self._latest, self._lock, self._running = None, threading.Lock(), True
        self._latest_wall = 0.0
        self.readings = 0
        threading.Thread(target=self._run, daemon=True).start()

    def latest(self):
        with self._lock:
            return self._latest if time.monotonic() - self._latest_wall < STALE_S else None

    def close(self):
        self._running = False

    def _run(self):
        while self._running:
            try:
                raw = self._sock.recv(65535)
            except OSError:
                continue
            if raw[:4] == MAGIC:      # rows -30..+30
                n_az, n_el = struct.unpack_from('<HH', raw, 4)
                offset = 8
            elif raw[:4] == MAGIC2:   # rows from el_lo to el_hi (straight down to +30)
                n_az, n_el, el_lo, el_hi = struct.unpack_from('<HHhh', raw, 4)
                offset = 12
                if ELEVATIONS.get(n_el) != (el_lo, el_hi):
                    continue  # a layout this module doesn't know
            else:
                continue
            t, x, y, z = struct.unpack_from('<dddd', raw, offset)
            d = np.frombuffer(raw, np.float16, offset=offset + 32, count=n_az * n_el).astype(np.float32).reshape(n_el, n_az)
            with self._lock:
                self._latest, self._latest_wall = (t, (x, y, z), d), time.monotonic()
                self.readings += 1


def elevations(n_el: int) -> np.ndarray:
    """Each row's elevation in degrees, lowest first, for an eye of n_el rows (the layouts in ELEVATIONS)."""
    lo, hi = ELEVATIONS[n_el]
    return np.linspace(lo, hi, n_el)


BELOW_DEG = -60.0  # rows at least this steep look at what is under the drone...
BELOW_RADIUS_M = 0.6  # ...where they hit within this of it, horizontally. Further out is beside it, not under it:
                      # 0.9 m from an 8 m wall, the steep rays met the wall 1.6 m below the drone, and it never
                      # came down from cruise height


def clear_below(reading) -> float:
    """How far the drone could descend before reaching what is under it (m): the shortest vertical drop
    the steepest rows see within BELOW_RADIUS_M of it (the floor, or the top of a person or a cart). inf
    if the eye has no such rows or sees nothing within range there (high up, the floor is out of range)."""
    d = reading[2]
    el = elevations(d.shape[0])
    steep = el <= BELOW_DEG
    if not steep.any():
        return math.inf
    rows = d[steep].astype(np.float32)
    a = np.radians(-el[steep])[:, None]
    hit = (rows > SELF_M) & (rows < MAX_RANGE_M) & (rows * np.cos(a) <= BELOW_RADIUS_M)
    drop = np.where(hit, rows * np.sin(a), np.inf)
    return float(drop.min())


def points_near(reading, x: float, y: float, radius: float, z_lo: float, z_hi: float) -> np.ndarray:
    """Where the eye's rays hit something within `radius` (horizontally) of (x, y), between heights z_lo
    and z_hi: [(x, y, z)] world. For a landing spot: what stands on it (a person, a parked cart)."""
    _, pos, d = reading
    d = np.asarray(d, dtype=np.float32)
    n_el, n_az = d.shape
    el = np.radians(elevations(n_el))[:, None]
    az = np.radians(np.arange(n_az) * (360.0 / n_az))[None, :]
    flat = d * np.cos(el)
    px, py, pz = pos[0] + flat * np.cos(az), pos[1] + flat * np.sin(az), pos[2] + d * np.sin(el)
    sel = ((d > SELF_M) & (d < MAX_RANGE_M) & (np.hypot(px - x, py - y) <= radius) & (pz >= z_lo) & (pz <= z_hi))
    return np.column_stack((px[sel], py[sel], pz[sel]))


def ring(reading, yaw_deg: float) -> tuple:
    """What is around the drone at its own height: (azimuth relative to the heading in degrees, + = left;
    the nearest distance in that direction over the RING_ELEVATION_DEG rows, inf = nothing within range
    or unknown). One entry per ommatidium column, all the way round."""
    d = reading[2]
    n_el, n_az = d.shape
    el = elevations(n_el)
    rows = d[(el >= RING_ELEVATION_DEG[0]) & (el <= RING_ELEVATION_DEG[1])]
    near = np.where((rows > SELF_M) & (rows < MAX_RANGE_M), rows, np.inf).min(axis=0)
    az = (np.arange(n_az) * (360.0 / n_az) - yaw_deg + 180.0) % 360.0 - 180.0
    return az, near


CLEARANCE_HALF_DEG = 100.0  # the supervisor's resume clearance looks this far either side of the heading


def clearance(az_deg: np.ndarray, near: np.ndarray) -> float:
    """The nearest thing ahead and to the sides (within CLEARANCE_HALF_DEG of the heading): what the
    supervisor's resume gate needs clear, as it needed the Multi-ranger's front, left and right rays clear,
    now with nothing in between unseen. What is behind doesn't block going on."""
    front = np.abs(az_deg) <= CLEARANCE_HALF_DEG
    return float(near[front].min()) if front.any() else math.inf


def corridors(az_deg: np.ndarray, near: np.ndarray, half_width_m: float) -> dict:
    """How far the drone could fly straight front / back / left / right before something within
    half_width_m of its path is in the way (inf: nothing within range). The eye's answer to the
    Multi-ranger's four rays: a corridor as wide as the drone plus margin, not a line."""
    a = np.radians(az_deg)
    out = {}
    for name, heading in (('front', 0.0), ('back', np.pi), ('left', np.pi / 2), ('right', -np.pi / 2)):
        along, across = near * np.cos(a - heading), near * np.sin(a - heading)
        hit = (along > 0.0) & (np.abs(across) < half_width_m)
        out[name] = float(along[hit].min()) if hit.any() else math.inf
    return out


def clear_out_ring(az_deg: np.ndarray, near: np.ndarray, clearance_m: float, speed: float) -> tuple:
    """Body-frame velocity (forward, left, 0) away from everything closer than clearance_m + 0.1 m all
    round (avoidance.clear_out, from the eye). Pinned between things on opposite sides, it goes toward
    the most open direction instead."""
    close = near < clearance_m + 0.1
    if not close.any():
        return (0.0, 0.0, 0.0)
    a = np.radians(az_deg)
    f, l = -float(np.cos(a[close]).sum()), -float(np.sin(a[close]).sum())
    if math.hypot(f, l) < 0.5:  # pushed from both sides: slide out the open way
        k = int(np.argmax(np.where(np.isfinite(near), near, 1e9)))
        f, l = math.cos(a[k]), math.sin(a[k])
    n = math.hypot(f, l)
    return (speed * f / n, speed * l / n, 0.0)


class SeenObject(NamedTuple):
    """One object around the drone in one reading (CollisionCourse.objects)."""
    cols: np.ndarray         # the eye's world-frame columns that see it
    weight: float            # 0..1: how much it is on a collision course with the drone
    velocity: tuple          # (vx, vy) m/s, world; (0, 0) when slower than STATIC_MPS
    moved: bool              # ever seen moving faster than STATIC_MPS: a person or a vehicle, not a wall
    points: np.ndarray       # [(x, y)] world: where the eye's rays hit it
    miss: tuple = (0.0, 0.0)  # world (x, y) from the drone to the object's centre where it passes closest: on
                              # the object's own path if it moves, on the drone's if it stands still
    kind: str = 'static'      # 'static' (never seen moving), 'person', 'vehicle' or 'unknown' (CollisionCourse)
    width: float = 0.0        # how wide it looks, m: the median spread of its points over the last readings
    still_s: float = math.inf  # how long since it was last seen moving (inf: never)


class _Track:
    def __init__(self):
        self.hist = deque(maxlen=36)  # (t, cx, cy): ~1.8 s (HEADING_S)
        self.moved = False
        self.spreads = deque(maxlen=5)  # the largest distance between two of its points, per reading
        self.steady = 0                 # readings in a row it has moved fast, the same way
        self.last_v = None              # its velocity at the last of those
        self.last_moving_t = None       # when it was last seen moving (after it counted as moved)


class CollisionCourse:
    """Which of the things around the drone are on a collision course with it: LPLC2's selectivity.

    A fly's LPLC2 neurons respond to an image that expands outward on all sides of a point in their
    receptive field, and not to one that also slides across it. Both edges of an approaching object's image
    move outward exactly when the line of sight stays inside the object: when drone and object will
    collide. Something the drone flies past grows too, but slides; LPLC2 ignores it, and so does this.

    Each reading, the ring at flying height (ring()) is cut into objects: runs of neighbouring directions
    at similar distances. Each object is tracked by the centroid of its points (world frame) to get its
    velocity; slower than STATIC_MPS counts as standing still (people walk ~1 m/s, the cart drives 1.5;
    centroid jitter, as directions join and leave an object, reaches ~0.35 m/s). So does parallax: as the
    drone flies past a crate, the face coming into view drags the centroid of what is seen along with the
    drone, by up to its own speed; motion within PARALLAX_DEG of the drone's own, no faster than it plus
    PARALLAX_MPS, is taken for that. An object counts as having moved (SeenObject.moved) once it has moved
    fast and the same way (within MOVE_TURN_DEG) for MOVE_CONFIRM readings in a row: a crate's centroid
    jitters, a person keeps walking. With the drone's own velocity (from the eye's positions), every point of
    the object gets its closest approach on the current relative motion; the object counts fully if one
    comes within MISS_FULL_M of the drone, and not at all beyond MISS_ZERO_M. update() returns that weight
    per world-frame column (azimuth 0 = +x, as the eye's).

    Each object also gets a kind: never seen moving, 'static'; moving and wider than VEHICLE_M, a 'vehicle'
    (it can be flown over, and is too wide and fast to get out of the way of in time); moving and narrower,
    a 'person' (to be gone round, never over); moving and wider than VEHICLE_MAX_M, 'unknown' (too big to
    be one vehicle: something merged). The width is the median over the last few readings, so a
    person passing close to a bench, merged with it for a reading or two, stays a person. Size is the cue
    that works all round and at any heading; speed would not (people jog, carts crawl), and height doesn't
    separate them (a car ~1.5 m, a person ~1.7 m)."""

    JUMP_REL = 0.15        # a depth step larger than this fraction of the distance (+ 0.2 m) splits objects
    MATCH_M = 0.6          # an object's centroid moved at most this far between readings
    WINDOW_S = 0.5         # velocity from the centroid's displacement over this long...
    HEADING_S = 1.5        # ...its direction over this long, while that agrees (a person walks straight on; over
                           # 0.5 s the heading was ~10 degrees off, which put a person's line 0.37 m wrong at 2 m
                           # and the drone escaped across their path: ft_headon)
    MIN_SPAN_S = 0.15      # ...and at least this long (4 readings)
    STATIC_MPS = 0.5
    PARALLAX_DEG = 45.0    # an object seen moving within this of the drone's own direction of flight...
    PARALLAX_MPS = 0.3     # ...and no faster than the drone plus this: what passing it looks like (fm_blocked:
                           # crates "walking" beside a drone that escaped at 0.8 m/s)
    MOVE_CONFIRM = 3       # readings in a row (0.15 s) of fast motion before an object counts as moved...
    MOVE_TURN_DEG = 45.0   # ...each within this of the one before
    MISS_FULL_M = 0.15     # the drone's 0.07 m radius plus margin
    MISS_ZERO_M = 0.3      # the planner's clearance: a planned pass beside a mapped obstacle never counts
    HORIZON_S = 4.0        # closest approaches further off than this don't count yet
    STOP_MARGIN_M = 0.15   # a static thing this far past where the route stops still counts (braking overshoot)
    VEHICLE_M = 0.75       # a mover wider than this is a vehicle (the cart 1.2 x 2.2 m, a car ~1.8 x 4.5 m), a
                           # narrower one a person (0.36 m here; ~0.6 m with a bag, or a bicycle head-on).
                           # The 5-degree columns put the cart at >= 0.84 m from 3 m in
    VEHICLE_MAX_M = 3.0    # wider than any one vehicle looks (the cart reads <= 2.3 m): 'unknown'. In testing
                           # these were walls whose visible stretch slides along with the passing drone, or a
                           # person merged with a bench (4.9-7.6 m). A real car seen corner-on reads ~5 m: then
                           # this needs the camera (it is the cue the eye can't give)

    def __init__(self):
        self._tracks = []                 # [_Track]
        self._drone = deque(maxlen=12)    # (t, x, y)
        self.objects = []                 # the last reading's objects (SeenObject)

    def update(self, reading, stop_after_m: float = math.inf) -> np.ndarray:
        """stop_after_m: the drone will stop this far ahead (an efference copy of its route): something
        standing still further along its path than that is no threat."""
        t, pos, _ = reading
        az, near = ring(reading, 0.0)
        n = len(near)
        a = np.radians(az)
        ux, uy = np.cos(a), np.sin(a)
        self._drone.append((t, pos[0], pos[1]))
        vdx, vdy = self._velocity(self._drone)
        weight = np.zeros(n)
        tracks, self.objects = [], []
        for cols in self._split(near):
            px, py = pos[0] + near[cols] * ux[cols], pos[1] + near[cols] * uy[cols]
            cx, cy = float(px.mean()), float(py.mean())
            track = self._match(cx, cy)
            track.hist.append((t, cx, cy))
            tracks.append(track)
            dpx, dpy = px[:, None] - px[None, :], py[:, None] - py[None, :]
            track.spreads.append(float(np.sqrt((dpx * dpx + dpy * dpy).max())))
            width = float(np.median(track.spreads))
            vox, voy = self._velocity(track.hist, robust=True)
            speed = math.hypot(vox, voy)
            hx, hy = self._velocity(track.hist, robust=True, window=self.HEADING_S)
            if speed > 0.0 and math.hypot(hx, hy) > 0.7 * speed and hx * vox + hy * voy > 0.0:
                k = speed / math.hypot(hx, hy)  # the long window's direction, the short one's speed (not round a turn)
                vox, voy = hx * k, hy * k
            if speed < self.STATIC_MPS or self._parallax(vox, voy, vdx, vdy):
                vox = voy = 0.0
                track.steady, track.last_v = 0, None
            else:
                lv = track.last_v
                turn_ok = lv is None or (vox * lv[0] + voy * lv[1]) >= (
                    math.cos(math.radians(self.MOVE_TURN_DEG)) * math.hypot(vox, voy) * math.hypot(*lv))
                track.steady = track.steady + 1 if turn_ok else 1
                track.last_v = (vox, voy)
                if track.steady >= self.MOVE_CONFIRM:
                    track.moved = True
                    track.last_moving_t = t
            kind = ('static' if not track.moved else 'person' if width < self.VEHICLE_M
                    else 'vehicle' if width <= self.VEHICLE_MAX_M else 'unknown')
            rvx, rvy = vox - vdx, voy - vdy  # the object's motion relative to the drone
            v2 = rvx * rvx + rvy * rvy
            w, miss_vec = 0.0, (0.0, 0.0)
            if v2 > 0.05 ** 2:
                rx, ry = px - pos[0], py - pos[1]
                t_cpa = -(rx * rvx + ry * rvy) / v2
                mx, my = rx + rvx * t_cpa, ry + rvy * t_cpa
                miss = np.hypot(mx, my)
                ahead = (t_cpa > 0.0) & (t_cpa < self.HORIZON_S)
                if vox == voy == 0.0 and math.isfinite(stop_after_m):  # it stands still; the drone will stop
                    ahead &= t_cpa * math.sqrt(v2) <= stop_after_m + self.STOP_MARGIN_M
                if ahead.any():
                    m = float(miss[ahead].min())
                    w = min(max((self.MISS_ZERO_M - m) / (self.MISS_ZERO_M - self.MISS_FULL_M), 0.0), 1.0)
                    # Which side it passes on: its centre's closest approach (a person straddles the path), on
                    # its own path if it moves (the drone is about to change course; where the mover goes is
                    # what to get out of), on the drone's if it stands still.
                    rcx, rcy = cx - pos[0], cy - pos[1]
                    lx, ly = (vox, voy) if (vox or voy) else (rvx, rvy)
                    tc = -(rcx * lx + rcy * ly) / (lx * lx + ly * ly)
                    miss_vec = (rcx + lx * tc, rcy + ly * tc)
            weight[cols] = w
            still = math.inf if track.last_moving_t is None else t - track.last_moving_t
            self.objects.append(SeenObject(cols, w, (vox, voy), track.moved, np.column_stack((px, py)), miss_vec,
                                           kind, width, still))
        self._tracks = tracks
        return weight

    def _parallax(self, vox: float, voy: float, vdx: float, vdy: float) -> bool:
        """Whether an object's apparent motion is what flying past a standing thing looks like."""
        sd, so = math.hypot(vdx, vdy), math.hypot(vox, voy)
        if sd < 0.15 or so > sd + self.PARALLAX_MPS:
            return False
        return vox * vdx + voy * vdy >= math.cos(math.radians(self.PARALLAX_DEG)) * so * sd

    def _split(self, near: np.ndarray) -> list:
        """Runs of neighbouring columns (round the ring) without a depth step between them."""
        n = len(near)
        fin = np.isfinite(near)
        if not fin.any():
            return []
        nxt = np.roll(near, -1)
        with np.errstate(invalid='ignore'):
            step = np.abs(nxt - near) > self.JUMP_REL * np.minimum(near, nxt) + 0.2
        brk = ~fin | ~np.roll(fin, -1) | step  # a break between column k and k + 1
        if not brk.any():
            return [np.arange(n)]
        start = (int(np.flatnonzero(brk)[0]) + 1) % n
        out, cur = [], []
        for i in range(n):
            k = (start + i) % n
            if fin[k]:
                cur.append(k)
            if brk[k] and cur:
                out.append(np.array(cur))
                cur = []
        if cur:
            out.append(np.array(cur))
        return out

    def _match(self, cx: float, cy: float) -> '_Track':
        """The track of the previous reading's object nearest to this centroid, or a new one."""
        best, best_d = None, self.MATCH_M
        for i, track in enumerate(self._tracks):
            d = math.hypot(track.hist[-1][1] - cx, track.hist[-1][2] - cy)
            if d < best_d:
                best, best_d = i, d
        if best is None:
            return _Track()
        return self._tracks.pop(best)  # one object per track

    def _velocity(self, hist, robust: bool = False, window: float | None = None) -> tuple:
        """Displacement over the last WINDOW_S (or `window`) / time. robust (an object's centroid): without
        the largest reading-to-reading step. A face of a crate coming into view makes the centroid jump once,
        which over the window looks like half a second of walking; a person moves a bit at every step."""
        if len(hist) < 2:
            return 0.0, 0.0
        t1, x1, y1 = hist[-1]
        old = [h for h in hist if t1 - h[0] <= (window or self.WINDOW_S)]
        t0, x0, y0 = old[0]
        if t1 - t0 < self.MIN_SPAN_S:
            return 0.0, 0.0
        if not robust or len(old) < 4:
            return (x1 - x0) / (t1 - t0), (y1 - y0) / (t1 - t0)
        steps = [(b[0] - a[0], b[1] - a[1], b[2] - a[2]) for a, b in zip(old, old[1:])]
        k = max(range(len(steps)), key=lambda i: steps[i][1] ** 2 + steps[i][2] ** 2)
        span = t1 - t0 - steps[k][0]
        return (x1 - x0 - steps[k][1]) / span, (y1 - y0 - steps[k][2]) / span


class Looming:
    """Feed eye readings in order (with the drone's heading); drive() is each eye's looming drive, 0..1.
    collision_selective: only what is on a collision course looms (CollisionCourse); else everything that
    gets closer does."""

    def __init__(self, collision_selective: bool = True):
        self._prev = self._prev_loom = None
        self.left = self.right = 0.0
        self.field_rate = None   # looming rate per 30-degree field relative to the heading, 1/s (dashboard)
        self.peak = (0.0, 0.0)   # each eye's most looming field, 1/s
        self.front_back = (0.0, 0.0)  # the most looming field ahead of / behind the drone (either side), 1/s
        self.pass_side = 0            # the most looming object will pass the drone on its left (+1) or right (-1)
        self.threat_velocity = (0.0, 0.0)  # its velocity (world; (0, 0): standing still)
        self.threat_miss = (0.0, 0.0)      # drone -> its centre where its path passes closest (world)
        self.threat_points = np.zeros((0, 2))  # where the eye's rays hit it, [(x, y)] world: how big it is
        self.threat_kind = ''                  # what it is: SeenObject.kind ('' when nothing looms)
        self.objects = CollisionCourse() if collision_selective else None

    def update(self, reading, yaw_deg: float, stop_after_m: float = math.inf):
        t, _, d = reading
        weight = self.objects.update(reading, stop_after_m) if self.objects else None
        if self._prev is None or t <= self._prev[0]:
            self._prev = (t, d)
            return
        t0, d0 = self._prev
        self._prev = (t, d)
        valid = (d < MAX_RANGE_M) & (d0 < MAX_RANGE_M) & (d > SELF_M) & (d0 > SELF_M) & (np.abs(d - d0) < JUMP_M)
        loom = np.where(valid, np.maximum(d0 - d, 0.0) / (t - t0) / np.maximum(d, SELF_M), 0.0)
        # Only what looms in two readings running counts: an edge flickering in and out of an ommatidium doesn't.
        steady = loom if self._prev_loom is None or self._prev_loom.shape != loom.shape else np.minimum(loom, self._prev_loom)
        self._prev_loom = loom
        n_el = d.shape[0]
        rows = elevations(n_el) >= MIN_ELEVATION_DEG
        column = steady[rows].max(axis=0)  # per azimuth: the most looming elevation (floor rows left out)
        if weight is not None:
            column = column * weight  # only what is on a collision course looms (LPLC2's selectivity)
            self._threat(column, yaw_deg)
        n_az = len(column)
        rel = (np.arange(n_az) * (360.0 / n_az) - yaw_deg + 180.0) % 360.0 - 180.0  # + = left of the heading
        centers = np.arange(-180.0 + FIELD_DEG / 2, 180.0, FIELD_DEG)
        rates = np.array([np.percentile(column[np.abs((rel - c + 180.0) % 360.0 - 180.0) <= FIELD_DEG / 2], 90)
                          for c in centers])
        self.field_rate = (centers, rates)
        left = rates[centers >= -OVERLAP_DEG].max()   # fields from the frontal overlap round the left side
        right = rates[centers <= OVERLAP_DEG].max()
        self.peak = (float(left), float(right))
        ahead = np.abs(centers) < 90.0
        self.front_back = (float(rates[ahead].max()), float(rates[~ahead].max()))
        self.left, self.right = min(float(left) / LOOM_REF, 1.0), min(float(right) / LOOM_REF, 1.0)

    def drive(self) -> tuple:
        return self.left, self.right

    PERSON_RECENT_S = 2.0  # nearest_person(): people seen moving this recently (people in our scenes walk on; a
                           # crate's centroid can still jump like a step now and then, and then stays still)

    def nearest_person(self, position) -> float:
        """Horizontal distance from position to the nearest point of anything the eye takes for a person
        (SeenObject.kind) and saw moving within PERSON_RECENT_S, inf if none: wherever the threat is, the
        drone doesn't climb next to people."""
        best = math.inf
        if self.objects:
            for obj in self.objects.objects:
                if obj.kind == 'person' and obj.still_s <= self.PERSON_RECENT_S and len(obj.points):
                    d = np.hypot(obj.points[:, 0] - position[0], obj.points[:, 1] - position[1])
                    best = min(best, float(d.min()))
        return best

    def _threat(self, column: np.ndarray, yaw_deg: float):
        """The most looming object: which side of the drone it will pass on (pass_side: +1 left, -1 right,
        0 straight at it, within PASS_DEAD_M, or nothing looming), its velocity, where its centre passes and
        where it is and what it is (threat_velocity, threat_miss, threat_points, threat_kind: see SeenObject)."""
        best, score = None, 0.0
        for obj in self.objects.objects:
            s = float(column[obj.cols].max())
            if s > score:
                best, score = obj, s
        if best is None:
            self.pass_side, self.threat_velocity, self.threat_miss = 0, (0.0, 0.0), (0.0, 0.0)
            self.threat_points, self.threat_kind = np.zeros((0, 2)), ''
            return
        yaw = math.radians(yaw_deg)
        lateral = -math.sin(yaw) * best.miss[0] + math.cos(yaw) * best.miss[1]  # + = left of the heading
        self.pass_side = 1 if lateral > PASS_DEAD_M else -1 if lateral < -PASS_DEAD_M else 0
        self.threat_velocity, self.threat_miss, self.threat_points = best.velocity, best.miss, best.points
        self.threat_kind = best.kind
