"""Avoidance dashboard: which system is deciding, what it senses, and why it switches modes.

avoid_demo.py streams one snapshot per control tick (20 Hz) here over local UDP, so the GUI can never
slow down or break a flight. By default the panel is drawn inside the MuJoCo window, along its right
edge (via shared memory; the patched simulator overlays it next to the drone camera inset).
launch.cmd avoid starts it for you; launch.cmd deliver also opens the Stops window (stops_window.py).
By hand:
    bash py.sh dashboard.py            panel inside the MuJoCo window
    bash py.sh dashboard.py --window   panel in its own window (q/Esc quits, s saves a screenshot)
    bash py.sh dashboard.py --stops    also the Stops window, to add and remove delivery stops
"""
import argparse
import json
import math
import os
import signal
import socket
import struct
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from planner import SCENE, geofence, load_markings, load_obstacles, room

PORT = 5300
OVERLAY_PATH = os.environ.get('CRAZYSIM_OVERLAY', '/dev/shm/crazysim_overlay.rgb')  # read by crazysim.py
OVERLAY_MAGIC = b'CSO1'
W, H = 560, 860
HISTORY_S = 15.0
RAYS = {'front': (1.0, 0.0), 'back': (-1.0, 0.0), 'left': (0.0, 1.0), 'right': (0.0, -1.0)}

# BGR
BG, PANEL, GRID = (28, 28, 28), (44, 44, 44), (62, 62, 62)
WHITE, GRAY, DIM = (235, 235, 235), (160, 160, 160), (95, 95, 95)
GREEN, RED, ORANGE, YELLOW, CYAN = (90, 200, 90), (70, 70, 235), (40, 160, 255), (60, 220, 230), (220, 190, 80)
BLUE = (255, 140, 64)
ORDER_COLORS = {'queued': ORANGE, 'active': YELLOW, 'delivered': GREEN, 'skipped': RED}
MAGENTA = (210, 80, 220)  # movers: people, carts


# --------------------------------------------------------------------------- sender (flight side)

def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None  # JSON has no inf: None means "nothing in range"
    return x


class DashboardFeed:
    """Sends snapshots to the dashboard. Fire-and-forget: works the same with no dashboard running."""

    def __init__(self, port: int = PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = ('127.0.0.1', port)

    def send(self, msg: dict):
        try:
            self._sock.sendto(json.dumps(_jsonable(msg)).encode(), self._addr)
        except OSError:
            pass


# --------------------------------------------------------------------------- drawing helpers

def text(img, s, org, scale=0.45, color=WHITE, thick=1):
    cv2.putText(img, s, (int(org[0]), int(org[1])), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def dashed_hline(img, x1, x2, y, color, dash=7):
    for x in range(int(x1), int(x2), dash * 2):
        cv2.line(img, (x, int(y)), (min(x + dash, int(x2)), int(y)), color, 1)


def threat_color(t):
    t = 0.0 if t is None else max(0.0, min(1.0, t))
    return (40, int(210 * (1 - t)) + 40, int(235 * t) + 20)


def panel(img, rect, title):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL, -1)
    text(img, title, (x + 8, y + 17), 0.45, GRAY)


def draw_markings(img, markings, P, dim: float = 0.55):
    """Floor markings (walkway, road) as dimmed filled boxes, drawn under everything else."""
    for pos, size, rgb, _name in markings:
        color = tuple(int(255 * dim * c) for c in reversed(rgb))  # rgb 0..1 -> dimmed BGR
        cv2.rectangle(img, P(pos[0] - size[0], pos[1] + size[1]), P(pos[0] + size[0], pos[1] - size[1]), color, -1)


def draw_movers(img, movers, P, scale: float, labels: bool = True):
    """Movers from the flight's snapshot ([name, kind, x, y, yaw, half_x, half_y, round]) on a top view."""
    for name, _kind, x, y, yaw, hx, hy, is_round in movers or []:
        if is_round:
            cv2.circle(img, P(x, y), max(3, int(round(hx * scale))), MAGENTA, -1, cv2.LINE_AA)
        else:
            c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            corners = [P(x + c * a - s * b, y + s * a + c * b) for a, b in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy))]
            cv2.fillPoly(img, [np.array(corners, np.int32)], MAGENTA, cv2.LINE_AA)
        if labels:
            px, py = P(x, y)
            text(img, name, (px - 24, py - int(hy * scale) - 6), 0.36, MAGENTA)


def fmt_m(d):
    return '> 4 m' if d is None or not math.isfinite(d) else f'{d:.2f} m'


# --------------------------------------------------------------------------- the panel

class Dashboard:
    def __init__(self, scene: Path):
        self.obstacles = load_obstacles(scene)
        self.markings = load_markings(scene)
        self.room, self.fence = room(scene), geofence(scene)
        self.msg = None
        self.last_rx = 0.0
        self.history = deque()           # (t, signal, ttc_signal, avoiding)
        self.trail = deque(maxlen=3000)  # (x, y, avoiding)

    def update(self, msg: dict):
        if self.msg is not None and msg['t'] < self.msg['t'] - 1.0:  # a new flight started
            self.history.clear()
            self.trail.clear()
        self.msg, self.last_rx = msg, time.monotonic()
        avoiding = msg['mode'] == 'AVOID'
        self.history.append((msg['t'], msg['signal'], msg['info'].get('ttc_signal'), avoiding))
        while self.history and self.history[0][0] < msg['t'] - HISTORY_S:
            self.history.popleft()
        self.trail.append((msg['state']['x'], msg['state']['y'], avoiding))

    def render(self) -> np.ndarray:
        img = np.full((H, W, 3), BG, np.uint8)
        if self.msg is None:
            text(img, 'AVOIDANCE DASHBOARD', (12, 30), 0.7, WHITE, 2)
            text(img, f'waiting for a flight on udp://127.0.0.1:{PORT} ...', (12, 60), 0.45, GRAY)
            text(img, 'start one with:  launch.cmd avoid   or   launch.cmd deliver', (12, 84), 0.45, DIM)
            return img
        m = self.msg
        self._header(img, m)
        self._pipeline(img, m, (6, 82, W - 12, 140))
        self._mode(img, m, (6, 228, W - 12, 150))
        self._plot(img, m, (6, 384, W - 12, 160))
        self._map(img, m, (6, 550, 304, 304))
        self._ranges(img, m, (316, 550, W - 322, 304))
        return img

    def _header(self, img, m):
        text(img, 'AVOIDANCE TEST', (12, 26), 0.7, WHITE, 2)
        route = 'live stops' if m['route_name'] == 'live' else f'route "{m["route_name"]}"'
        book = m.get('orders') or {}
        text(img, f'{route}   scene: {m.get("scene", "-")}   map: {book.get("map", "-")}', (12, 50), 0.48, WHITE)
        speed = math.hypot(m['state']['vx'], m['state']['vy'])
        text(img, f'{m["phase"]}   t={m["t"]:.1f} s   {speed:.2f} m/s', (12, 72), 0.42, GRAY)
        cost = m.get('decision_cost') or {}
        if cost.get('count'):
            text(img, f'decision {cost["mean_us"]:.0f} us (p99 {cost["p99_us"]:.0f})', (W - 205, 50), 0.42, CYAN)
        stale = time.monotonic() - self.last_rx
        if stale > 2.0:
            text(img, f'no data for {stale:.0f} s', (W - 150, 26), 0.45, ORANGE)

    @staticmethod
    def _brain_status(m) -> str:
        info = m['info']
        if info.get('brain_ok') is False:
            return 'NOT ANSWERING'
        gl, gr = info.get('gf_left'), info.get('gf_right')
        return 'giant fibers L/R' if gl is None else f'giant fiber L {gl:.0f} R {gr:.0f} Hz'

    def _pipeline(self, img, m, rect):
        x, y, w, h = rect
        panel(img, rect, 'What is making the avoidance decision')
        inputs = m['policy']['inputs']
        fps = m.get('camera_fps')
        fps_s = f' ({fps:.0f} fps)' if fps else ''

        def box(bx, by, bw, bh, label, sub, color, active):
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), color if active else DIM, 2 if active else 1)
            text(img, label, (bx + 6, by + 13), 0.4, WHITE if active else GRAY)
            text(img, sub, (bx + 6, by + bh - 5), 0.34, color if active else DIM)

        info = m['info']
        notes = m['policy'].get('notes', {})
        if not m['camera_streaming']:
            camera_sub = 'no stream'
        elif 'camera' not in inputs:
            camera_sub = 'not used' + fps_s
        else:  # what the camera made of the threat, or what it is there for
            camera_sub = info.get('camera_kind') or notes.get('camera', 'used') + fps_s
        if '360 eye' not in inputs:
            eye_sub = 'not used'
        elif 'loom_left' in info:
            eye_sub = f'looming L {info["loom_left"]:.1f} R {info["loom_right"]:.1f} /s'
        else:
            eye_sub = f'used, nearest {fmt_m(m.get("clearance"))}'
        ranger_sub = notes.get('multiranger', 'used by policy') if 'multiranger' in inputs else 'not used'
        rows = [('Multi-ranger ToF', ranger_sub, 'multiranger' in inputs),
                ('AI-deck camera', camera_sub, 'camera' in inputs),
                ('360 compound eye', eye_sub, '360 eye' in inputs),
                ('Fly brain', self._brain_status(m) if 'fly brain' in inputs else 'not used', 'fly brain' in inputs)]
        mid = y + 80
        for i, (label, sub, used) in enumerate(rows):
            by = y + 22 + i * 29
            box(x + 8, by, 160, 27, label, sub, GREEN, used)
            cv2.arrowedLine(img, (x + 168, by + 13), (x + 183, mid), GREEN if used else DIM, 1, tipLength=0.15)
        if 'map' in inputs:  # the planner's map, into the policy from above: which way round is shorter
            chose = info.get('side_from') == 'map'
            box(x + 185, y + 22, 140, 28, 'Route map', 'chose the side' if chose else notes.get('map', 'used'),
                YELLOW if chose else GREEN, True)
            cv2.arrowedLine(img, (x + 255, y + 50), (x + 255, mid - 23), GREEN, 1, tipLength=0.4)
        mode_col = RED if m['mode'] == 'AVOID' else GREEN
        box(x + 185, mid - 22, 140, 44, 'Policy', m['policy']['name'], ORANGE, True)
        cv2.arrowedLine(img, (x + 325, mid), (x + 340, mid), WHITE, 1, tipLength=0.3)
        box(x + 342, mid - 22, 100, 44, 'Supervisor', m['mode'], mode_col, True)
        cv2.arrowedLine(img, (x + 442, mid), (x + 456, mid), WHITE, 1, tipLength=0.3)
        box(x + 458, mid - 22, w - 466, 44, 'Firmware', 'PID control', CYAN, True)
        text(img, 'the resume gate also reads the clearance ahead and aside (eye)' if m.get('eye_ring') is not None
             else 'the resume gate also reads Multi-ranger clearance', (x + 185, mid + 42), 0.36, DIM)

    def _mode(self, img, m, rect):
        x, y, w, h = rect
        panel(img, rect, 'Supervisor state machine')
        th, gate = m['thresholds'], m['gate']
        avoiding = m['mode'] == 'AVOID'
        cv2.rectangle(img, (x + 10, y + 28), (x + 160, y + 88), RED if avoiding else GREEN, -1)
        text(img, m['mode'], (x + 20, y + 70), 0.95 if avoiding else 0.85, (20, 20, 20), 2)
        text(img, f'for {m["time_in_mode"]:.1f} s', (x + 12, y + 110), 0.45, WHITE)
        text(img, f'avoid episodes: {m["episodes"]}', (x + 12, y + 132), 0.45, WHITE)

        sig = m['signal']
        sig_s = '-' if sig is None else f'{sig:.2f}'
        tx = x + 175
        if not avoiding:
            text(img, f'Switches to AVOID when signal >= {th["enter"]:.2f}', (tx, y + 42), 0.46, WHITE)
            text(img, f'avoid signal now: {sig_s}', (tx, y + 66), 0.46, YELLOW if sig and sig > th['exit'] else GRAY)
            text(img, f'(leaves AVOID below {th["exit"]:.2f}: that gap', (tx, y + 90), 0.4, GRAY)
            text(img, ' is the hysteresis that stops chatter)', (tx, y + 108), 0.4, GRAY)
        else:
            text(img, f'Back to route when ALL hold {th["dwell_s"]:.2f} s:', (tx, y + 42), 0.46, WHITE)
            for i, (ok, s) in enumerate(((gate['signal_ok'], f'signal {sig_s} < exit {th["exit"]:.2f}'),
                                         (gate['clear_ok'], f'clearance {fmt_m(m["clearance"])} > {th["clearance_m"]:.2f} m'))):
                text(img, f'{"[x]" if ok else "[ ]"} {s}', (tx, y + 64 + 21 * i), 0.46, GREEN if ok else RED)
            held = gate['held_s']
            text(img, f'held {held:.2f} / {th["dwell_s"]:.2f} s', (tx, y + 108), 0.42, GRAY)
            cv2.rectangle(img, (tx + 140, y + 100), (tx + 350, y + 108), DIM, 1)
            cv2.rectangle(img, (tx + 140, y + 100), (tx + 140 + int(210 * min(1.0, held / th['dwell_s'])), y + 108), GREEN, -1)
        text(img, f'policy maneuver: {m["info"].get("maneuver", "-")}', (tx, y + 128), 0.42, ORANGE)
        obj = m['info'].get('object_mps') or 0.0
        if obj > 0.0:
            text(img, f'moving obstacle: {obj:.1f} m/s toward the drone ({m["info"].get("object_ray")} ray)',
                 (tx, y + 145), 0.4, MAGENTA)

    def _plot(self, img, m, rect):
        x, y, w, h = rect
        panel(img, rect, f'Avoid signal, last {HISTORY_S:.0f} s')
        px0, px1, py0, py1 = x + 38, x + w - 10, y + 26, y + h - 24
        cv2.rectangle(img, (px0, py0), (px1, py1), GRID, 1)
        t_now = m['t']

        def X(t):
            return int(px1 - (t_now - t) / HISTORY_S * (px1 - px0))

        def Y(v):
            return int(py1 - max(0.0, min(1.0, v)) * (py1 - py0))

        hist = list(self.history)
        for (t1, *_), (t2, _, _, avoiding) in zip(hist, hist[1:]):
            if avoiding:
                cv2.rectangle(img, (X(t1), py0 + 1), (X(t2), py1 - 1), (55, 45, 90), -1)
        th = m['thresholds']
        dashed_hline(img, px0, px1, Y(th['enter']), RED)
        dashed_hline(img, px0, px1, Y(th['exit']), GREEN)
        text(img, f'enter {th["enter"]:.2f}', (px1 - 72, Y(th['enter']) - 4), 0.37, RED)
        text(img, f'exit {th["exit"]:.2f}', (px1 - 72, Y(th['exit']) - 4), 0.37, GREEN)
        for v in (0.0, 0.5, 1.0):
            text(img, f'{v:.1f}', (x + 8, Y(v) + 4), 0.37, GRAY)
        for idx, col, thick in ((2, ORANGE, 1), (1, WHITE, 2)):
            pts = [(X(e[0]), Y(e[idx])) for e in hist if e[idx] is not None]
            if len(pts) > 1:
                cv2.polylines(img, [np.array(pts, np.int32)], False, col, thick, cv2.LINE_AA)
        text(img, 'white: avoid signal   orange: raw time-to-collision   red band: AVOID', (px0, y + h - 7), 0.37, GRAY)

    def _map(self, img, m, rect):
        x, y, w, h = rect
        book = m.get('orders')
        if book:
            todo = sum(o[3] in ('queued', 'active') for o in book['orders'])
            done = sum(o[3] == 'delivered' for o in book['orders'])
            panel(img, rect, f'Top view: {todo} to go, {done} delivered')
        else:
            panel(img, rect, 'Top view (1 m grid)')
        rx0, rx1, ry0, ry1 = self.room
        S = min((w - 8) / (rx1 - rx0 + 0.5), (h - 12) / (ry1 - ry0 + 0.5))
        cx, cy = x + w / 2, y + 12 + (h - 12) / 2
        mx, my = (rx0 + rx1) / 2, (ry0 + ry1) / 2

        def P(wx, wy):
            return int(round(cx + (wx - mx) * S)), int(round(cy - (wy - my) * S))

        for k in range(round(rx0), round(rx1) + 1):
            cv2.line(img, P(k, round(ry0)), P(k, round(ry1)), GRID, 1)
        for k in range(round(ry0), round(ry1) + 1):
            cv2.line(img, P(round(rx0), k), P(round(rx1), k), GRID, 1)
        st = m['state']
        alt = st['z'] if st['z'] > 0.2 else 0.5
        draw_markings(img, self.markings, P)
        for kind, pos, size, _name in self.obstacles:
            half_h = size[2] if kind == 'box' else size[1]
            at_alt = pos[2] - half_h - 0.05 <= alt <= pos[2] + half_h + 0.05
            color, fill = ((125, 150, 125), -1) if at_alt else ((100, 100, 100), 1)  # outline: above the drone
            if kind == 'box':
                cv2.rectangle(img, P(pos[0] - size[0], pos[1] + size[1]), P(pos[0] + size[0], pos[1] - size[1]), color, fill)
            else:
                cv2.circle(img, P(pos[0], pos[1]), max(2, int(size[0] * S)), color, fill)

        for sx, sy in m.get('seen') or []:  # obstacles the avoidance met, now on the planner's map
            cv2.circle(img, P(sx, sy), 2, ORANGE, -1)
        if book:
            self._stops(img, book, P)
        draw_movers(img, m.get('movers'), P, S, labels=False)
        trail = list(self.trail)
        for (x1, y1, _), (x2, y2, avoiding) in zip(trail, trail[1:]):
            cv2.line(img, P(x1, y1), P(x2, y2), RED if avoiding else GREEN, 2)

        px, py, yaw = st['x'], st['y'], math.radians(st['yaw'])
        for az_w, d in m.get('eye_ring') or []:  # what the compound eye sees all round
            a = math.radians(az_w)
            cv2.circle(img, P(px + d * math.cos(a), py + d * math.sin(a)), 1, CYAN, -1)
        for ray, (f, l) in RAYS.items():
            ux, uy = f * math.cos(yaw) - l * math.sin(yaw), f * math.sin(yaw) + l * math.cos(yaw)
            d = m['ranges'].get(ray)
            col = threat_color(m['info'].get(f'threat_{ray}')) if d is not None else DIM
            end = P(px + ux * (d if d is not None else 4.0), py + uy * (d if d is not None else 4.0))
            inside, a, b = cv2.clipLine((x, y + 22, w, h - 22), P(px, py), end)
            if inside:
                cv2.line(img, a, b, col, 1, cv2.LINE_AA)
            if d is not None:
                cv2.circle(img, end, 3, col, -1)
        cv2.circle(img, P(px, py), int(m['thresholds']['clearance_m'] * S), DIM, 1, cv2.LINE_AA)
        cv2.circle(img, P(px, py), 5, WHITE, -1)
        cv2.arrowedLine(img, P(px, py), P(px + 0.4 * math.cos(yaw), py + 0.4 * math.sin(yaw)), WHITE, 2, tipLength=0.35)
        v = m['v_cmd']
        if math.hypot(v[0], v[1]) > 0.02:
            cv2.arrowedLine(img, P(px, py), P(px + v[0] * 2, py + v[1] * 2), YELLOW, 2, tipLength=0.3)
        if book:
            cv2.rectangle(img, (x + 2, y + h - 37), (x + w - 2, y + h - 2), PANEL, -1)  # off the south wall
            text(img, book['event'], (x + 8, y + h - 24), 0.36, WHITE)
            text(img, 'add/remove stops: Stops window or double-click', (x + 8, y + h - 8), 0.34, GRAY)
        else:
            text(img, 'trail: green route, red avoiding', (x + 8, y + h - 8), 0.36, GRAY)

    @staticmethod
    def _stops(img, book, P):
        """Live delivery stops: home (H), the planned route still to fly, and each stop coloured by state."""
        hx, hy = book['home']
        hp = P(hx, hy)
        cv2.rectangle(img, (hp[0] - 7, hp[1] - 7), (hp[0] + 7, hp[1] + 7), BLUE, 2 if book['return_home'] else 1)
        text(img, 'H', (hp[0] - 4, hp[1] + 4), 0.33, BLUE)
        for leg in book['route']:
            for a, b in zip(leg, leg[1:]):
                cv2.line(img, P(*a), P(*b), ORANGE, 1, cv2.LINE_AA)
        for n, x, y, state in book['orders']:
            c = ORDER_COLORS[state]
            cv2.circle(img, P(x, y), 5, c, -1 if state == 'delivered' else 2, cv2.LINE_AA)
            text(img, str(n), P(x + 0.12, y + 0.12), 0.4, c)

    def _ranges(self, img, m, rect):
        if m.get('eye_ring') is not None:
            return self._surroundings(img, m, rect)
        x, y, w, h = rect
        panel(img, rect, 'Multi-ranger (body frame)')
        bx0, bx1 = x + 52, x + w - 58
        scale = (bx1 - bx0) / 4.0
        clear_x = bx0 + int(m['thresholds']['clearance_m'] * scale)
        for i, ray in enumerate(('front', 'left', 'right', 'back', 'up')):
            by = y + 36 + i * 46
            d = m['ranges'].get(ray)
            text(img, ray, (x + 8, by + 13), 0.45, WHITE)
            cv2.rectangle(img, (bx0, by), (bx1, by + 18), GRID, 1)
            if d is None:  # nothing within 4 m
                cv2.rectangle(img, (bx0, by), (bx1, by + 18), DIM, -1)
            else:
                col = threat_color(m['info'].get(f'threat_{ray}')) if ray != 'up' else CYAN
                cv2.rectangle(img, (bx0, by), (bx0 + int(min(d, 4.0) * scale), by + 18), col, -1)
            text(img, fmt_m(d), (bx1 + 5, by + 13), 0.4, WHITE)
        cv2.line(img, (clear_x, y + 30), (clear_x, y + 36 + 5 * 46 - 22), GRAY, 1)
        text(img, f'| = clearance radius {m["thresholds"]["clearance_m"]:.1f} m', (x + 8, y + h - 30), 0.37, GRAY)
        text(img, 'bar colour: green safe -> red threat', (x + 8, y + h - 10), 0.37, GRAY)

    @staticmethod
    def _surroundings(img, m, rect):
        """What the compound eye sees all round at flying height, as a radar: the drone in the middle,
        heading up, 1 m rings out to 4 m. The Multi-ranger's four rays are the thin gray lines."""
        x, y, w, h = rect
        panel(img, rect, 'All round (compound eye), heading up')
        cx, cy = x + w // 2, y + 22 + (h - 62) // 2
        S = (min(w, h - 62) / 2 - 6) / 4.0
        clearance = m['thresholds']['clearance_m']
        for r in (1, 2, 3, 4):
            cv2.circle(img, (cx, cy), int(r * S), GRID, 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), int(clearance * S), DIM, 1, cv2.LINE_AA)
        yaw = m['state']['yaw']

        def P(az_body_deg, d):  # + azimuth = left of the heading; the heading points up
            a = math.radians(az_body_deg)
            return int(round(cx - math.sin(a) * d * S)), int(round(cy - math.cos(a) * d * S))

        for ray, az in (('front', 0.0), ('left', 90.0), ('back', 180.0), ('right', -90.0)):
            d = m['ranges'].get(ray)
            cv2.line(img, (cx, cy), P(az, min(d, 4.0) if d is not None else 4.0), DIM, 1, cv2.LINE_AA)
        for az_w, d in m['eye_ring']:
            col = RED if d < clearance else ORANGE if d < 1.0 else GREEN
            cv2.circle(img, P(az_w - yaw, d), 2, col, -1, cv2.LINE_AA)
        info = m['info']
        if info.get('worst_az') is not None and info.get('ttc_signal', 0.0) >= 0.25:
            cv2.arrowedLine(img, P(info['worst_az'], 3.6), P(info['worst_az'], 1.2), threat_color(info['ttc_signal']),
                            2, tipLength=0.25)
        cv2.circle(img, (cx, cy), 4, WHITE, -1)
        cv2.arrowedLine(img, (cx, cy), (cx, cy - 16), WHITE, 1, tipLength=0.35)
        near = min((d for _, d in m['eye_ring']), default=None)
        text(img, f'nearest {fmt_m(near)} all round; | ring = clearance {clearance:.1f} m', (x + 8, y + h - 28), 0.36, GRAY)
        text(img, 'dots: what the eye sees; gray lines: Multi-ranger', (x + 8, y + h - 10), 0.36, GRAY)


# --------------------------------------------------------------------------- outputs

class OverlayWriter:
    """Hands the panel to the simulator's viewer through a shared-memory file, replaced atomically."""

    def __init__(self, path: str = OVERLAY_PATH):
        self.path, self._tmp = path, path + '.tmp'

    def write(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with open(self._tmp, 'wb') as f:
            f.write(struct.pack('<4sHH', OVERLAY_MAGIC, rgb.shape[1], rgb.shape[0]) + rgb.tobytes())
        os.replace(self._tmp, self.path)

    def remove(self):
        for p in (self.path, self._tmp):
            try:
                os.remove(p)
            except OSError:
                pass


def _exit(*_):
    raise SystemExit(0)


def receive(sock):
    while True:
        try:
            pkt = sock.recv(65535)
        except BlockingIOError:
            return
        yield json.loads(pkt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--window', action='store_true', help='show the panel in its own window instead of the MuJoCo window')
    ap.add_argument('--port', type=int, default=PORT)
    ap.add_argument('--scene', type=Path, default=SCENE, help='MuJoCo scene drawn on the top view')
    ap.add_argument('--stops', action='store_true', help='also open the Stops window (for deliver flights)')
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(('127.0.0.1', args.port))
    except OSError as e:
        raise SystemExit(f'cannot listen on udp port {args.port} ({e}); is another dashboard running?')
    sock.setblocking(False)
    signal.signal(signal.SIGTERM, _exit)  # launch.py stops us with SIGTERM: still remove the overlay file

    dash = Dashboard(args.scene)
    overlay = None if args.window else OverlayWriter()
    snap_dir = Path(__file__).resolve().parent / 'snapshots'
    if args.window:
        cv2.namedWindow('Avoidance dashboard', cv2.WINDOW_AUTOSIZE)
    else:
        print(f'drawing the panel into the MuJoCo window (via {OVERLAY_PATH})', flush=True)
    stops = None
    if args.stops:
        from routes import ALTITUDE_M
        from stops_window import StopsWindow  # imported here: it imports this module's drawing helpers
        stops = StopsWindow(dash, ALTITUDE_M)
    last_write = 0.0
    try:
        while True:
            got = False
            for msg in receive(sock):
                dash.update(msg)
                got = True
            if args.window:
                img = dash.render()
                cv2.imshow('Avoidance dashboard', img)
                key = cv2.waitKey(50) & 0xFF
                if key in (ord('q'), 27) or cv2.getWindowProperty('Avoidance dashboard', cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord('s'):
                    snap_dir.mkdir(exist_ok=True)
                    path = snap_dir / time.strftime('dashboard_%Y%m%d_%H%M%S.png')
                    cv2.imwrite(str(path), img)
                    print(f'saved {path}')
            else:
                now = time.monotonic()
                if got or now - last_write > 1.0:  # refresh at least every second, so it never looks stale
                    overlay.write(dash.render())
                    last_write = now
            if stops and not stops.closed:
                if got:
                    stops.show()
                if not args.window:
                    cv2.waitKey(30)  # runs the Stops window's mouse handling
            elif not args.window:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if overlay:
            overlay.remove()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
