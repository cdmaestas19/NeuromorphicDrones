"""The Stops window: add and remove delivery stops with the mouse, and see the planned route.

Opened by the dashboard process during deliver flights (launch.cmd deliver runs dashboard.py --stops).
It shows the flight's state from the dashboard feed, and sends every request to the flight's OrderBook
on UDP ORDERS_PORT, the same way double-clicks in the 3D view arrive. The flight checks every request,
so the window only displays and asks.

    left-click the map           add a stop there (or, on a stop / on H, the same as its button)
    right-click a stop           remove it
    X next to a stop             remove it (a stop not done yet is cancelled)
    buttons                      go home after the last stop, cancel open stops, clear finished ones,
                                 and what the path planner knows (planner.py)
"""
import json
import math
import socket

import cv2
import numpy as np

from dashboard import (BG, BLUE, DIM, GRAY, GREEN, GRID, MAGENTA, ORANGE, ORDER_COLORS, PANEL, RED, WHITE, YELLOW,
                       draw_markings, draw_movers, text)
from flight import GEOFENCE
from orders import KEEP_OUT_M, ORDERS_PORT, SELECT_RADIUS_M, obstacle_gap
from planner import MAP_HELP, MAPS, Planner, on_map

NAME = 'Delivery stops'
W, H = 960, 620
MAP = (10, 10, 600, 600)  # x, y, w, h of the map
SPAN_M = 6.1              # metres across the map (the room is 6 x 6 m)
PANEL_X = 625
ROWS = 9                  # stop rows shown in the list
ZONE = (52, 46, 46)       # BGR shade of the zones the planner keeps out of


class StopsWindow:
    def __init__(self, dash, altitude: float):
        self.dash, self.altitude = dash, altitude
        self.obstacles = [o for o in dash.obstacles if _spans(o, altitude)]
        self.S = MAP[2] / SPAN_M
        self.cx, self.cy = MAP[0] + MAP[2] / 2, MAP[1] + MAP[3] / 2
        self._zones = {}
        self._buttons = []   # (x0, y0, x1, y1, request) rebuilt by every render
        self._hover = None   # world (x, y) under the mouse, when over the map
        self.closed = False
        self._tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        cv2.namedWindow(NAME, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(NAME, self._on_mouse)
        self.show()

    # -- input ------------------------------------------------------------------------------------

    def _on_mouse(self, event, px, py, _flags, _param):
        inside = MAP[0] <= px < MAP[0] + MAP[2] and MAP[1] <= py < MAP[1] + MAP[3]
        if event == cv2.EVENT_MOUSEMOVE:
            self._hover = self._world(px, py) if inside else None
            self.show()
        elif event == cv2.EVENT_LBUTTONDOWN:
            for x0, y0, x1, y1, request in self._buttons:
                if x0 <= px <= x1 and y0 <= py <= y1:
                    return self._send(request)
            if inside:
                wx, wy = self._world(px, py)
                self._send({'cmd': 'pick', 'x': wx, 'y': wy, 'z': 0.0})  # the flight decides: add, remove or home
        elif event == cv2.EVENT_RBUTTONDOWN and inside:
            n = self._stop_at(*self._world(px, py))
            if n is not None:
                self._send({'cmd': 'remove', 'n': n})

    def _send(self, request: dict):
        if self._book() is None:
            return  # no deliver flight is listening yet
        try:
            self._tx.sendto(json.dumps(request).encode(), ('127.0.0.1', ORDERS_PORT))
        except OSError:
            pass

    # -- output -----------------------------------------------------------------------------------

    def show(self):
        if self.closed:
            return
        if cv2.getWindowProperty(NAME, cv2.WND_PROP_VISIBLE) < 1:
            self.closed = True  # the user closed it; don't bring it back
            return
        cv2.imshow(NAME, self.render())

    def render(self) -> np.ndarray:
        img = np.full((H, W, 3), BG, np.uint8)
        self._buttons = []
        book, m = self._book(), self.dash.msg
        self._draw_map(img, book, m)
        self._draw_panel(img, book, m)
        return img

    def _draw_map(self, img, book, m):
        x, y, w, h = MAP
        map_name = book['map'] if book else 'all'
        img[y:y + h, x:x + w] = self._zone_image(map_name)
        P = self._px
        for k in range(-3, 4):
            cv2.line(img, P(k, -3), P(k, 3), GRID, 1)
            cv2.line(img, P(-3, k), P(3, k), GRID, 1)
        for kind, pos, size, name in self.obstacles:
            mapped = on_map(name, map_name)
            color = (125, 150, 125) if mapped else ORANGE
            if kind == 'box':
                a, b = P(pos[0] - size[0], pos[1] + size[1]), P(pos[0] + size[0], pos[1] - size[1])
                cv2.rectangle(img, a, b, color, -1 if mapped else 2)
            else:
                cv2.circle(img, P(pos[0], pos[1]), max(2, int(size[0] * self.S)), color, -1 if mapped else 2)
            if not mapped:
                text(img, 'not on map', (P(pos[0], pos[1])[0] - 30, P(pos[0], pos[1])[1] + 4), 0.33, ORANGE)
        if not book:
            return
        legs = book['route']
        for k, leg in enumerate(legs):
            pts = np.array([P(*p) for p in leg], np.int32)
            current = k == 0 and m['phase'].startswith(('flying', 'turning'))
            cv2.polylines(img, [pts], False, YELLOW if current else ORANGE, 2 if current else 1, cv2.LINE_AA)
        trail = list(self.dash.trail)
        for (x1, y1, _), (x2, y2, avoiding) in zip(trail, trail[1:]):
            cv2.line(img, P(x1, y1), P(x2, y2), RED if avoiding else (70, 120, 70), 1)
        hp = P(*book['home'])
        cv2.rectangle(img, (hp[0] - 11, hp[1] - 11), (hp[0] + 11, hp[1] + 11), BLUE, 3 if book['return_home'] else 1)
        text(img, 'H', (hp[0] - 6, hp[1] + 6), 0.5, BLUE, 2)
        for n, sx, sy, state in book['orders']:
            c, p = ORDER_COLORS[state], P(sx, sy)
            if state == 'skipped':
                cv2.line(img, (p[0] - 8, p[1] - 8), (p[0] + 8, p[1] + 8), c, 2)
                cv2.line(img, (p[0] - 8, p[1] + 8), (p[0] + 8, p[1] - 8), c, 2)
            else:
                cv2.circle(img, p, 9, c, -1 if state == 'delivered' else 2, cv2.LINE_AA)
            text(img, str(n), (p[0] + 11, p[1] - 9), 0.5, c, 1)
        draw_movers(img, m.get('movers'), P, self.S)
        st = m['state']
        dp, yaw = P(st['x'], st['y']), math.radians(st['yaw'])
        cv2.circle(img, dp, 6, WHITE, -1)
        cv2.arrowedLine(img, dp, P(st['x'] + 0.35 * math.cos(yaw), st['y'] + 0.35 * math.sin(yaw)), WHITE, 2, tipLength=0.35)
        if self._hover is not None:
            hx, hy = self._hover
            ok, _ = self._hover_check(book, hx, hy)
            hp = P(hx, hy)
            cv2.circle(img, hp, int(KEEP_OUT_M * self.S), GREEN if ok else RED, 1, cv2.LINE_AA)
            text(img, f'({hx:.2f}, {hy:.2f})', (hp[0] + 14, hp[1] + 30), 0.4, GREEN if ok else RED)

    def _draw_panel(self, img, book, m):
        x = PANEL_X
        cv2.rectangle(img, (x - 5, 10), (W - 10, H - 10), PANEL, -1)
        text(img, 'DELIVERY STOPS', (x + 5, 38), 0.7, WHITE, 2)
        if not book:
            text(img, 'waiting for a deliver flight...', (x + 5, 66), 0.45, GRAY)
            text(img, '(launch.cmd deliver starts one)', (x + 5, 88), 0.42, DIM)
            return
        for i, line in enumerate(('Left-click the map: add a stop',
                                  'Click a stop, or its X below: remove it',
                                  'Click H: go home after the last stop',
                                  'In the 3D view, double-clicks work the same')):
            text(img, line, (x + 5, 64 + 19 * i), 0.42, WHITE if i < 3 else GRAY)

        y = 150
        self._button(img, x + 5, y, 150, 'Go home: ' + ('ON' if book['return_home'] else 'off'), {'cmd': 'home'},
                     BLUE if book['return_home'] else None)
        self._button(img, x + 165, y, 150, 'Cancel open stops', {'cmd': 'cancel_all'})
        self._button(img, x + 5, y + 38, 150, 'Clear finished', {'cmd': 'clear_finished'})

        y = 245
        text(img, 'Path planner knows:', (x + 5, y), 0.45, WHITE)
        for i, name in enumerate(MAPS):
            self._button(img, x + 5 + 105 * i, y + 10, 98, name, {'cmd': 'map', 'name': name},
                         GREEN if book['map'] == name else None)
        text(img, MAP_HELP[book['map']], (x + 5, y + 62), 0.38, GRAY)
        text(img, 'shaded = where the planner keeps out', (x + 5, y + 80), 0.38, GRAY)
        if m.get('movers'):
            text(img, 'magenta = people / cart: never on the map', (x + 5, y + 97), 0.38, MAGENTA)

        y = 362
        text(img, m['phase'][:46], (x + 5, y), 0.42, YELLOW)
        text(img, book['event'][:48], (x + 5, y + 20), 0.42, WHITE)

        y = 405
        stops = book['orders']
        todo = sum(s[3] in ('queued', 'active') for s in stops)
        done = sum(s[3] == 'delivered' for s in stops)
        text(img, f'Stops: {todo} to go, {done} delivered', (x + 5, y), 0.45, WHITE)
        shown = stops[-ROWS:]
        if len(stops) > ROWS:
            text(img, f'(+{len(stops) - ROWS} earlier)', (x + 225, y), 0.38, DIM)
        for i, (n, sx, sy, state) in enumerate(shown):
            ry = y + 12 + 21 * i
            text(img, f'{n:>2}', (x + 8, ry + 14), 0.45, ORDER_COLORS[state])
            text(img, f'({sx:5.2f}, {sy:5.2f})', (x + 40, ry + 14), 0.42, WHITE)
            text(img, state, (x + 170, ry + 14), 0.42, ORDER_COLORS[state])
            self._button(img, W - 48, ry, 26, 'X', {'cmd': 'remove', 'n': n}, RED, height=18)
        if not stops:
            text(img, 'none yet: click the map', (x + 8, y + 26), 0.42, DIM)

    def _button(self, img, x, y, w, label, request, accent=None, height=30):
        cv2.rectangle(img, (x, y), (x + w, y + height), accent or (70, 70, 70), -1)
        cv2.rectangle(img, (x, y), (x + w, y + height), GRAY, 1)
        scale = 0.42 if height >= 30 else 0.4
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        text(img, label, (x + (w - tw) // 2, y + (height + th) // 2), scale, (20, 20, 20) if accent else WHITE)
        self._buttons.append((x, y, x + w, y + height, request))

    # -- helpers ----------------------------------------------------------------------------------

    def _book(self):
        m = self.dash.msg
        return m.get('orders') if m else None

    def _hover_check(self, book, x, y) -> tuple[bool, str]:
        """The same keep-out rule the flight applies (it has the final say)."""
        if self._stop_at(x, y) is not None or math.dist((x, y), book['home']) < SELECT_RADIUS_M:
            return True, ''
        (x0, x1), (y0, y1) = GEOFENCE['x'], GEOFENCE['y']
        if not (x0 + 0.1 <= x <= x1 - 0.1 and y0 + 0.1 <= y <= y1 - 0.1):
            return False, 'outside'
        gap, name = obstacle_gap(x, y, self.altitude, self.obstacles)
        return gap >= KEEP_OUT_M, name

    def _stop_at(self, x, y):
        book = self._book()
        if not book:
            return None
        near = [(math.dist((x, y), (sx, sy)), n) for n, sx, sy, _ in book['orders']]
        near = [d_n for d_n in near if d_n[0] < SELECT_RADIUS_M]
        return min(near)[1] if near else None

    def _zone_image(self, map_name: str) -> np.ndarray:
        """The map background: dark floor with the scene's markings, shaded where the planner keeps out.
        Cached per map."""
        if map_name not in self._zones:
            x, y, w, h = MAP
            planner = Planner(self.dash.obstacles, GEOFENCE, self.altitude, map_name)
            us, vs = np.meshgrid(np.arange(w) + x, np.arange(h) + y)
            wx, wy = (us - self.cx) / self.S, (self.cy - vs) / self.S
            i = np.rint((wx - planner.x0) / 0.05).astype(int)
            j = np.rint((wy - planner.y0) / 0.05).astype(int)
            inside = (i >= 0) & (i < planner.nx) & (j >= 0) & (j < planner.ny)
            blocked = np.ones((h, w), bool)
            blocked[inside] = planner.blocked[i[inside], j[inside]]
            zone = np.empty((h, w, 3), np.uint8)
            zone[:] = BG
            draw_markings(zone, self.dash.markings, lambda px, py: (self._px(px, py)[0] - x, self._px(px, py)[1] - y),
                          dim=0.45)
            zone[blocked] = ZONE
            self._zones[map_name] = zone
        return self._zones[map_name]

    def _px(self, wx: float, wy: float) -> tuple:
        return int(round(self.cx + wx * self.S)), int(round(self.cy - wy * self.S))

    def _world(self, px: int, py: int) -> tuple:
        return (px - self.cx) / self.S, (self.cy - py) / self.S


def _spans(obstacle, z: float) -> bool:
    kind, pos, size, _ = obstacle
    half_h = size[2] if kind == 'box' else size[1]
    return pos[2] - half_h <= z <= pos[2] + half_h


if __name__ == '__main__':
    raise SystemExit('run it through the dashboard: bash py.sh dashboard.py --stops')
