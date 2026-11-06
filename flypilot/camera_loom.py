"""Looming from the AI-deck camera: what grows in the image, around itself.

The ideal compound eye (eye.py) computes looming from the simulator's distances. This computes it from the
camera's images. The quantity is the same, 1 / time-to-contact in 1/s: an object whose distance shrinks at
d'/d grows in the image at k = -d'/d. So is the selectivity: LPLC2 responds to an image expanding around a
point inside its receptive field, not to one that slides or turns across it (Klapoetke et al. 2017, Nature
551:237); what the drone will hit expands around itself, what it will pass expands around a point beside it.

How (LPLC2's function, not its mechanism): the image is cut into regions bounded by edges (the simulator's
surfaces are plain, so an object is a region of its own: a person, a crate, a tree's crown), and each region
is tracked from frame to frame (by overlap). Its growth is a straight-line fit of ln(size) over the last
FIT_S (its area, or its width where the image's edge cuts it off): k is the slope, and the fit must be good
(FIT_TOL): an approach grows smoothly, regions merging and splitting jump. The point it grows around is
found the same way: its centre's drift (less the drone's own turning, yaw: an efference copy) / k; a region
counts only if that point is inside it (CENTRE_FRACTION of its radius). Every 5-degree unit of the view
inside a looming region gets its k: loom_points() gives [(azimuth, elevation, k)] in the body frame, the
brain's retinotopic input (flybrain/brain_server.py 'loom'), as eye.Looming does for the whole sphere.

Motion-detector versions came first (dense optic flow standing in for T4/T5, then four-arm LPLC2 units, or
a shift + turn + expansion fit per field). On the recorded flights they could not tell looming from noise at
this resolution: a person's outline 2 m away moves ~0.5 pixel per frame, under the image's shift from the
drone's own motion, and the plain surfaces carry no motion inside them (flybrain/README.md).

The camera sees ~86 x 70 degrees ahead. What it can't see (sides, behind) is the range sensors'
(policies.FlyBrainCamPolicy).
"""
import math
from collections import deque

import cv2
import numpy as np

from camera_kind import CAM_FOVY_DEG

GRID_DEG = 5.0          # one unit every this many degrees (the compound eye's spacing)
FIT_S = 0.3             # growth fitted over this long (~6 frames at 20 fps)...
FIT_MIN = 4             # ...from at least this many frames
FIT_TOL = 0.04          # the fit's RMS residual in ln(size) at most this (an approach grows smoothly)
CANNY = (20, 50)        # edge thresholds (grey levels)
MIN_AREA_PX = 25        # smaller regions (half-resolution pixels) are texture, not objects
MAX_AREA_FRACTION = 0.3  # larger ones are floor, walls, sky
MIN_OVERLAP = 0.5       # a region and its self a frame earlier share at least this much of the smaller of the two
MAX_K = 6.0             # 1/s: faster growth is two regions merging or splitting, not an approach
CENTRE_FRACTION = 1.0   # it must grow around a point within this much of its radius from its centre
NOISE = 0.4             # 1/s: slower growth doesn't count. The flight keeps mapped obstacles > 2.5 s away
                        # (avoid_demo.MAP_TTC_S), so their looming stays below 1 / 2.5 s
HEIGHT_PX = 122         # works at half resolution (as recorded; live frames are halved)


class _Track:
    def __init__(self):
        self.hist = deque(maxlen=12)  # (t, ln size, centre x less the turn, centre y, radius px, size kind)


class CameraLooming:
    """Feed camera frames with their times (and the drone's heading) in order; loom_points() is the newest
    looming per unit."""

    def __init__(self):
        self._prev = None                # the last frame's (labels, track per region)
        self._yaw, self._turn_px = None, 0.0  # the drone's turning so far, as an image shift (efference copy)
        self.loom = None                 # (units el x az) looming, 1/s
        self.az = self.el = None         # each unit's direction, deg (body frame: + left, + up)
        self._geom = None

    def _geometry(self, h: int, w: int):
        f = (h / 2.0) / math.tan(math.radians(CAM_FOVY_DEG / 2.0))
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        half_az = math.degrees(math.atan(cx / f))
        half_el = math.degrees(math.atan(cy / f))
        az = np.arange(-math.floor(half_az / GRID_DEG), math.floor(half_az / GRID_DEG) + 1) * GRID_DEG
        el = np.arange(-math.floor(half_el / GRID_DEG), math.floor(half_el / GRID_DEG) + 1) * GRID_DEG
        A, E = np.meshgrid(az, el)
        a, e = np.radians(A), np.radians(E)
        px = np.clip(np.rint(cx - f * np.tan(a)), 0, w - 1).astype(int)
        py = np.clip(np.rint(cy - f * np.tan(e) / np.cos(a)), 0, h - 1).astype(int)
        self.az, self.el = A, E
        self._geom = (h, w, f, px, py)

    def update(self, frame: np.ndarray, t: float, yaw_deg: float = 0.0):
        """A new camera frame (grey, any resolution: halved to HEIGHT_PX rows if bigger), taken at time t (s)
        with the drone heading yaw_deg."""
        img = frame
        while img.shape[0] >= 2 * HEIGHT_PX:
            img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
        if self._geom is None or self._geom[:2] != img.shape[:2]:
            self._geometry(*img.shape[:2])
            self._prev = None
        h, w, f, px, py = self._geom
        if self._yaw is not None:  # the view turned left: the image moved right
            self._turn_px += f * math.radians((yaw_deg - self._yaw + 180.0) % 360.0 - 180.0)
        self._yaw = yaw_deg
        edges = cv2.dilate(cv2.Canny(cv2.GaussianBlur(img, (3, 3), 0), *CANNY), np.ones((2, 2), np.uint8))
        n, labels, stats, cents = cv2.connectedComponentsWithStats((edges == 0).astype(np.uint8), connectivity=4)
        area = stats[:, cv2.CC_STAT_AREA].astype(float)
        tracks = [None] * n
        if self._prev is not None:  # each region's track: the region a frame ago it overlaps most, one-to-one
            labels0, tracks0 = self._prev
            pair = labels.ravel().astype(np.int64) * len(tracks0) + labels0.ravel()
            counts = np.bincount(pair, minlength=n * len(tracks0)).reshape(n, len(tracks0))
            counts[:, 0] = counts[0, :] = 0  # region 0 is the edges
            best = counts.argmax(axis=1)
            shared = counts[np.arange(n), best]
            area0 = np.bincount(labels0.ravel(), minlength=len(tracks0)).astype(float)
            taken = {}
            for i in np.argsort(-shared):
                j = best[i]
                if shared[i] >= MIN_OVERLAP * min(area[i], area0[j]) and j not in taken and tracks0[j] is not None:
                    taken[j] = i
                    tracks[i] = tracks0[j]
        loom_region = np.zeros(n)
        for i in range(1, n):
            if area[i] < MIN_AREA_PX or area[i] > MAX_AREA_FRACTION * h * w:
                continue
            tr = tracks[i] = tracks[i] or _Track()
            x, y, bw, bh = stats[i, :4]
            if not (x <= 0 or y <= 0 or x + bw >= w or y + bh >= h):
                kind, size = 'area', 0.5 * math.log(area[i])
            elif x > 0 and x + bw < w:  # cut off top or bottom only: its width
                kind, size = 'width', math.log(bw)
            else:
                tr.hist.clear()
                continue
            if tr.hist and tr.hist[-1][5] != kind:
                tr.hist.clear()
            tr.hist.append((t, size, cents[i][0] - self._turn_px, cents[i][1], math.sqrt(area[i] / math.pi), kind))
            k = self._growth(tr)
            if k is not None:
                loom_region[i] = k
        self._prev = (labels, tracks)
        cell = int(round(f * math.tan(math.radians(GRID_DEG)))) | 1  # each unit: the most looming in its cell
        kimg = cv2.dilate(loom_region[labels].astype(np.float32), np.ones((cell, cell), np.uint8))
        self.loom = kimg[py, px].astype(float)

    @staticmethod
    def _growth(tr: '_Track') -> float | None:
        """A track's looming (1/s), or None: no clean exponential growth over FIT_S, or not around itself."""
        t_now = tr.hist[-1][0]
        h = np.array([e[:5] for e in tr.hist if t_now - e[0] <= FIT_S + 1e-6])
        if len(h) < FIT_MIN or h[-1, 0] - h[0, 0] < 0.6 * FIT_S:
            return None
        a = np.column_stack([h[:, 0] - h[:, 0].mean(), np.ones(len(h))])
        (k, _), *_ = np.linalg.lstsq(a, h[:, 1], rcond=None)
        resid = h[:, 1] - a @ np.linalg.lstsq(a, h[:, 1], rcond=None)[0]
        if not (NOISE <= k <= MAX_K) or math.sqrt(np.mean(resid ** 2)) > FIT_TOL:
            return None
        vx, vy = (np.linalg.lstsq(a, h[:, c], rcond=None)[0][0] for c in (2, 3))
        return float(k) if math.hypot(vx, vy) / k <= CENTRE_FRACTION * h[-1, 4] else None

    def loom_points(self) -> np.ndarray:
        """[(azimuth, + left; elevation; looming 1/s)] of every unit that sees looming, deg (body frame)."""
        if self.loom is None:
            return np.zeros((0, 3))
        k = self.loom > 0.0
        return np.column_stack([self.az[k], self.el[k], self.loom[k]])


APPROACH_MPS = 0.05     # the drone approaches what it sees at least this fast for its own motion to explain it
STOP_MARGIN_M = 0.15    # a static thing this far past where the route stops still counts (eye.CollisionCourse's)
FLOOR_M = 0.15          # lower than this, it is the floor


def gate(pts: np.ndarray, position, yaw_deg: float, velocity_body, stop_m: float, on_map=None) -> np.ndarray:
    """The looming that is the drone's own approach to something the flight already deals with, taken out
    (an efference copy, as eye.CollisionCourse does with distances; the camera has none). If what looms stood
    still, the drone's own approach along its direction would make it loom at k = approach / distance: so it
    would be at distance approach / k. A point is dropped when that puts it beyond where the route stops (the
    drone will stop short of it), on the planner's map (on_map: the planner keeps the path clear of it, and
    the flight keeps it > 2.5 s away), or on the floor. Something coming at the drone looms faster than the
    drone's approach explains, so it is placed nearer than it is, in the open: it stays."""
    if not len(pts):
        return pts
    a, e = np.radians(pts[:, 0]), np.radians(pts[:, 1])
    ux, uy, uz = np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)
    approach = velocity_body[0] * ux + velocity_body[1] * uy + velocity_body[2] * uz
    d = approach / pts[:, 2]
    yaw = math.radians(yaw_deg)
    wx = position[0] + d * (math.cos(yaw) * ux - math.sin(yaw) * uy)
    wy = position[1] + d * (math.sin(yaw) * ux + math.cos(yaw) * uy)
    wz = position[2] + d * uz
    own = (approach >= APPROACH_MPS) & ((d * ux > stop_m + STOP_MARGIN_M) | (wz < FLOOR_M))
    if on_map is not None:
        own |= (approach >= APPROACH_MPS) & np.array([on_map(x, y) for x, y in zip(wx, wy)], bool)
    return pts[~own]
