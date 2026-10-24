"""What a threat is, from the AI-deck camera: a small CNN looks at a vertical strip of the image at the
threat's bearing and says background, person or vehicle.

The compound eye knows where a threat is and how wide it looks (eye.CollisionCourse: its size is the
all-round cue); the camera can tell what it is when it is in front. A real Crazyflie has only the
camera for this, and a network this size (~6 k weights, ~0.8 M multiply-adds) is what the AI-deck's
GAP8 runs. vision_train.py trains it on frames from recorded flights, labelled automatically from where
the simulator put each person and the cart (label_strips()); the weights are models/camera_kind.npz.
Here it runs in numpy, ~1 ms per strip, so the flight venv needs no PyTorch.

The camera (CrazySim's fpv_cam, like the real AI-deck's): straight ahead, CAM_AHEAD_M in front of the
drone's centre, CAM_FOVY_DEG vertical field of view, 324 x 244 grey. The network sees the image at half
resolution (162 x 122, as flights record it), so a strip is STRIP_W columns, ~21 degrees.
"""
import math
from pathlib import Path

import numpy as np

CAM_FOVY_DEG = 70.0
CAM_AHEAD_M = 0.03
STRIP_W = 32                       # half-resolution columns
KINDS = ('background', 'person', 'vehicle')
MODEL = Path(__file__).with_name('models') / 'camera_kind.npz'


def focal(height_px: int) -> float:
    """Focal length in pixels for an image this tall (square pixels)."""
    return (height_px / 2.0) / math.tan(math.radians(CAM_FOVY_DEG / 2.0))


def half_fov_deg(width_px: int, height_px: int) -> float:
    """Half the horizontal field of view (~43 degrees)."""
    return math.degrees(math.atan((width_px / 2.0) / focal(height_px)))


def column(bearing_rad: float, width_px: int, height_px: int) -> float:
    """Image column of a bearing relative to the heading (+ = left of it)."""
    return width_px / 2.0 - focal(height_px) * math.tan(bearing_rad)


def half_res(frame: np.ndarray) -> np.ndarray:
    """A full-resolution frame as flights record it and the network sees it."""
    return frame[::2, ::2] if frame.shape[0] > 200 else frame


def strip(img: np.ndarray, u: float) -> np.ndarray:
    """The STRIP_W columns of a half-resolution image centred on column u (edges repeated past the border)."""
    c0 = int(round(u)) - STRIP_W // 2
    cols = np.clip(np.arange(c0, c0 + STRIP_W), 0, img.shape[1] - 1)
    return img[:, cols]


def normalise(strips: np.ndarray) -> np.ndarray:
    """(N, H, W) uint8 -> float32, each strip to zero mean and unit spread (lighting differs)."""
    x = strips.astype(np.float32)
    m = x.mean(axis=(1, 2), keepdims=True)
    s = x.std(axis=(1, 2), keepdims=True) + 8.0
    return (x - m) / s


class StripNet:
    """The network's forward pass in numpy: conv layers (ReLU), global average pool, linear, softmax.
    Weights as vision_train.py exports them: conv{i}_w (out, in, k, k), conv{i}_b, conv{i}_stride,
    fc_w (classes, features), fc_b."""

    def __init__(self, path: Path = MODEL):
        w = np.load(path)
        self.convs = []
        i = 0
        while f'conv{i}_w' in w:
            self.convs.append((w[f'conv{i}_w'].astype(np.float32), w[f'conv{i}_b'].astype(np.float32),
                               int(w[f'conv{i}_stride'])))
            i += 1
        self.fc_w, self.fc_b = w['fc_w'].astype(np.float32), w['fc_b'].astype(np.float32)

    @staticmethod
    def _conv(x: np.ndarray, k: np.ndarray, b: np.ndarray, stride: int) -> np.ndarray:
        n, c, h, w = x.shape
        o, _, kh, kw = k.shape
        p = kh // 2
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
        ho, wo = (h + 2 * p - kh) // stride + 1, (w + 2 * p - kw) // stride + 1
        cols = np.empty((n, c, kh, kw, ho, wo), np.float32)
        for i in range(kh):
            for j in range(kw):
                cols[:, :, i, j] = xp[:, :, i:i + stride * ho:stride, j:j + stride * wo:stride]
        y = np.tensordot(cols, k, axes=([1, 2, 3], [1, 2, 3]))  # (n, ho, wo, o)
        return np.maximum(y + b, 0.0).transpose(0, 3, 1, 2)

    def probs(self, strips: np.ndarray) -> np.ndarray:
        """(N, H, STRIP_W) uint8 strips -> (N, len(KINDS)) probabilities."""
        x = normalise(strips)[:, None]
        for k, b, stride in self.convs:
            x = self._conv(x, k, b, stride)
        z = x.mean(axis=(2, 3)) @ self.fc_w.T + self.fc_b
        z = np.exp(z - z.max(axis=1, keepdims=True))
        return z / z.sum(axis=1, keepdims=True)


class CameraKind:
    """For a policy: what the threat at a world point is, from the newest camera frame. None when there is
    no trained model, no frame, the point is outside the view, or the network isn't sure (MIN_P)."""

    MIN_P = 0.8
    EDGE_DEG = 4.0       # a threat nearer the image's edge than this is half out of it

    def __init__(self, path: Path = MODEL):
        self.net = StripNet(path) if Path(path).is_file() else None
        self.last = None  # (kind, p, bearing_deg) of the last classification, for the dashboard and records

    def classify(self, frame: np.ndarray | None, position, yaw_deg: float, target_xy) -> tuple | None:
        self.last = None
        if self.net is None or frame is None:
            return None
        img = half_res(frame)
        h, w = img.shape
        yaw = math.radians(yaw_deg)
        cx, cy = position[0] + CAM_AHEAD_M * math.cos(yaw), position[1] + CAM_AHEAD_M * math.sin(yaw)
        dx, dy = target_xy[0] - cx, target_xy[1] - cy
        fwd, left = dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)
        bearing = math.atan2(left, fwd)
        if fwd <= 0.0 or abs(math.degrees(bearing)) > half_fov_deg(w, h) - self.EDGE_DEG:
            return None
        p = self.net.probs(strip(img, column(bearing, w, h))[None])[0]
        k = int(np.argmax(p))
        self.last = (KINDS[k], float(p[k]), math.degrees(bearing))
        if p[k] < self.MIN_P:
            return None
        return KINDS[k], float(p[k])


def _ray_part(ox: float, oy: float, ux: float, uy: float, part, x: float, y: float, yaw: float) -> float:
    """Distance along the ray (o + t u) to one movers.Part of a mover at (x, y) turned yaw (rad); inf if missed."""
    c, s = math.cos(yaw), math.sin(yaw)
    px, py = x + part.dx * c - part.dy * s, y + part.dx * s + part.dy * c
    if part.round:
        fx, fy = ox - px, oy - py
        b = fx * ux + fy * uy
        disc = b * b - (fx * fx + fy * fy - part.half[0] ** 2)
        if disc < 0.0:
            return math.inf
        t = -b - math.sqrt(disc)
        return t if t > 0.0 else math.inf
    lx, ly = (ox - px) * c + (oy - py) * s, -(ox - px) * s + (oy - py) * c  # ray in the part's frame
    vx, vy = ux * c + uy * s, -ux * s + uy * c
    t0, t1 = -math.inf, math.inf
    for o, v, h in ((lx, vx, part.half[0]), (ly, vy, part.half[1])):
        if abs(v) < 1e-9:
            if abs(o) > h:
                return math.inf
            continue
        a, b = (-h - o) / v, (h - o) / v
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
    return t0 if t1 >= t0 > 0.0 else math.inf


def label_strips(shapes: dict, movers: dict, headings: dict, position, yaw_deg: float, eye_near,
                 width_px: int, height_px: int, max_m: float = 5.0, step_px: int = 4) -> list:
    """Automatic labels for the strips of one frame: [(column, kind)] with kind 'person' / 'vehicle' where the
    strip's centre ray hits that mover first (no nearer thing in the eye's ring along that bearing) within
    max_m, 'background' where no mover is within half a strip of the centre ray, and nothing for strips in
    between (half on a mover) or on a mover further than max_m. shapes: movers.load_movers(); movers:
    {name: (x, y)} for those a walk has moved (the others stand where the scene parks them: still people);
    headings: {name: heading_deg} (a cart's long axis is along it); eye_near(az_world_deg) -> the eye's
    nearest distance that way (inf if unknown)."""
    movers = {**{n: s.xy for n, s in shapes.items()}, **movers}
    yaw = math.radians(yaw_deg)
    ox, oy = position[0] + CAM_AHEAD_M * math.cos(yaw), position[1] + CAM_AHEAD_M * math.sin(yaw)
    f = focal(height_px)
    hit = {}  # column -> (distance to the first mover along it, its kind, world azimuth deg); every 2 columns
    for u in range(0, width_px + 1, 2):
        a = yaw + math.atan((width_px / 2.0 - u) / f)
        ux, uy = math.cos(a), math.sin(a)
        best = (math.inf, None)
        for name, (x, y) in movers.items():
            s = shapes.get(name)
            if s is None:
                continue
            myaw = math.radians(headings.get(name, 90.0) - 90.0) if s.kind == 'cart' else 0.0
            d = min(_ray_part(ox, oy, ux, uy, p, x, y, myaw) for p in s.parts)
            if d < best[0]:
                best = (d, s.kind)
        hit[u] = (*best, math.degrees(a))
    out = []
    for u in range(STRIP_W // 2, width_px - STRIP_W // 2 + 1, step_px):
        d, kind, az = hit[u - u % 2]
        if kind is not None:
            if d <= max_m and eye_near(az) >= d - 0.35:
                out.append((u, 'vehicle' if kind == 'cart' else 'person'))
            continue  # on a mover: hidden, or too far to label either way
        if all(hit[c][1] is None for c in range(u - u % 2 - STRIP_W // 2, u - u % 2 + STRIP_W // 2 + 1, 2)
               if c in hit):
            out.append((u, 'background'))
    return out
