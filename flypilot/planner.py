"""Path planning for delivery stops: a route around the obstacles that are on the drone's map.

The map is an occupancy grid of the room at flight altitude, built from the scene file's obstacles and
grown by CLEARANCE_M, so a planned path never passes closer than that to a mapped obstacle. A* finds
the shortest grid path; line-of-sight shortcuts then turn it into a few straight legs.

What is on the map is a choice, because the planner and the avoidance layer split the work:
    all        every obstacle in the scene: the planner routes around everything, avoidance is the backup
    building   walls, pillars and partitions but not the crates. Crates are clutter nobody mapped, so the
               avoidance layer has to handle them: the realistic test for a reactive (fly-brain) policy.
    none       an empty room: straight lines, and every obstacle is left to the avoidance layer

Where the map comes from: here, CrazySim's scene file (load_obstacles). For a real flight it would be a
floor plan, building footprints (e.g. OpenStreetMap), or a map the drone builds with its own sensors;
the planner only needs the obstacle list.

Planning runs when a leg starts and after an avoidance episode, never per control tick (one plan takes
a few ms here). On a real Crazyflie it would run on the ground station, which already sends the setpoints.

The map also helps the avoidance choose a side. With each leg's plan comes a cost-to-go field: how far
the stop is from everywhere, around everything on the map (cost_field, ~10 ms). route_cost() then says
how long the way to the stop is if the drone first escapes to some point: a policy compares going round
a threat on the left with going round it on the right, and takes the shorter (policies.py).
"""
import heapq
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SCENE = (Path(os.environ.get('CRAZYSIM_DIR', Path.home() / 'CrazySim')) / 'crazyflie-firmware' /
         'tools/crazyflie-simulation/simulator_files/mujoco/scene_obstacles.xml')  # run_sim.sh's default scene
SCENES = {  # name -> scene file; launch.py starts the sim on the one a route asks for
    'obstacles': SCENE,                                                   # CrazySim's room of static obstacles
    'street': Path(__file__).resolve().parent / 'scenes' / 'street.xml',  # walkway, road, people, a cart
}

CELL_M = 0.05
CLEARANCE_M = 0.3   # planned paths keep this far from mapped obstacles (stops need orders.KEEP_OUT_M = 0.35)
EDGE_M = 0.1        # and this far inside the geofence
MAPS = ('all', 'building', 'none')
MAP_HELP = {'all': 'every obstacle is mapped', 'building': 'crates, new poles not mapped: avoidance handles them',
            'none': 'nothing mapped: straight lines'}

_SQ2 = math.sqrt(2.0)
_MOVES = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
          (1, 1, _SQ2), (1, -1, _SQ2), (-1, 1, _SQ2), (-1, -1, _SQ2))
FIELD_STEP = 2      # the cost-to-go field uses every other grid cell (10 cm)
_SQ5 = math.sqrt(5.0)
_FIELD_MOVES = _MOVES + tuple((a * i, b * j, _SQ5) for i, j in ((1, 2), (2, 1)) for a in (1, -1) for b in (1, -1))
DRONE_M = 0.1       # route_cost: an escape must keep this far from a mapped obstacle's surface


def load_obstacles(path: Path = SCENE) -> list:
    """The static boxes and cylinders of a MuJoCo scene file: (kind, pos, size, name). Only geoms placed
    directly in worldbody count; bodies (the movers) and visual-only geoms (contype = conaffinity = 0,
    such as floor markings) are left out."""
    try:
        world = ET.parse(path).getroot().find('worldbody')
    except (OSError, ET.ParseError):
        return []
    out = []
    for g in world.findall('geom') if world is not None else []:
        if g.get('contype') == '0' and g.get('conaffinity') == '0':
            continue
        if g.get('type') in ('box', 'cylinder'):
            pos = [float(v) for v in g.get('pos', '0 0 0').split()]
            size = [float(v) for v in g.get('size', '0 0 0').split()]
            out.append((g.get('type'), pos, size, g.get('name', '')))
    return out


def load_markings(path: Path = SCENE) -> list:
    """The scene's visual-only boxes (floor markings such as a walkway or a road), for the top views:
    (pos, size, rgb 0..1, name). Their colour is the geom's rgba or its material's."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return []
    colors = {m.get('name'): [float(v) for v in m.get('rgba', '0.5 0.5 0.5 1').split()[:3]]
              for m in root.iter('material')}
    world = root.find('worldbody')
    out = []
    for g in world.findall('geom') if world is not None else []:
        if g.get('type') == 'box' and g.get('contype') == '0' and g.get('conaffinity') == '0':
            rgb = ([float(v) for v in g.get('rgba').split()[:3]] if g.get('rgba')
                   else colors.get(g.get('material'), [0.5, 0.5, 0.5]))
            out.append(([float(v) for v in g.get('pos', '0 0 0').split()],
                        [float(v) for v in g.get('size', '0 0 0').split()], rgb, g.get('name', '')))
    return out


ROOM_WALLS = ('wall_west', 'wall_east', 'wall_south', 'wall_north')
FENCE_INSET_M = 0.35  # the geofence stays this far inside the room's walls
FENCE_Z = (-0.2, 1.4)  # without walls to go by
CEILING_BELOW_TOP_M = 0.5  # the geofence's ceiling: this far below the top of the walls...
CEILING_MAX_M = 6.0        # ...and no higher than this (cruising at 4.5 m above people and cars, with room to
                           # climb out of trouble; CrazySim's room, 3 m walls, stays at 2.5 m)


def room(path: Path = SCENE) -> tuple:
    """(x0, x1, y0, y1): the floor inside the scene's walls, the box geoms named ROOM_WALLS. A 6 x 6 m room
    if the scene has no such walls."""
    walls = {name: (pos, size) for kind, pos, size, name in load_obstacles(path) if kind == 'box' and name in ROOM_WALLS}
    if len(walls) < len(ROOM_WALLS):
        return (-2.95, 2.95, -2.95, 2.95)
    inner = lambda name, axis, sign: walls[name][0][axis] + sign * walls[name][1][axis]  # noqa: E731
    return inner('wall_west', 0, 1), inner('wall_east', 0, -1), inner('wall_south', 1, 1), inner('wall_north', 1, -1)


def geofence(path: Path = SCENE) -> dict:
    """The flight geofence for a scene: FENCE_INSET_M inside its walls, and up to CEILING_BELOW_TOP_M below
    their top (at most CEILING_MAX_M): +-2.6 m and 2.5 m in CrazySim's 6 x 6 m room, whose walls are 3 m
    tall. (flight.GEOFENCE, for flights without a scene, keeps the old 1.4 m ceiling.)"""
    x0, x1, y0, y1 = room(path)
    m = FENCE_INSET_M
    tops = [pos[2] + size[2] for kind, pos, size, name in load_obstacles(path) if kind == 'box' and name in ROOM_WALLS]
    ceiling = min(min(tops) - CEILING_BELOW_TOP_M, CEILING_MAX_M) if len(tops) == len(ROOM_WALLS) else FENCE_Z[1]
    return {'x': (round(x0 + m, 3), round(x1 - m, 3)), 'y': (round(y0 + m, 3), round(y1 - m, 3)),
            'z': (FENCE_Z[0], round(ceiling, 3))}


def on_map(name: str, map_name: str) -> bool:
    """Whether the obstacle called `name` is on the planner's map `map_name`: 'building' leaves out the
    clutter (crate*) and what went up after the map was made (new_*)."""
    if map_name == 'none':
        return False
    return map_name == 'all' or not name.startswith(('crate', 'new_'))


class Planner:
    """Occupancy grid + A*. `obstacles` as from load_obstacles: (kind, pos, size, name)."""

    def __init__(self, obstacles, fence: dict, altitude: float, map_name: str = 'all'):
        self.altitude = altitude
        self.obstacles = [o for o in obstacles if _spans(o, altitude)]
        (self.x0, x1), (self.y0, y1) = fence['x'], fence['y']
        self.nx = int(round((x1 - self.x0) / CELL_M)) + 1
        self.ny = int(round((y1 - self.y0) / CELL_M)) + 1
        xs = self.x0 + CELL_M * np.arange(self.nx)
        ys = self.y0 + CELL_M * np.arange(self.ny)
        self._X, self._Y = np.meshgrid(xs, ys, indexing='ij')
        m = EDGE_M - 1e-6  # a stop exactly EDGE_M inside the fence is allowed (orders.FENCE_MARGIN_M)
        self._edge = ((self._X < self.x0 + m) | (self._X > x1 - m) | (self._Y < self.y0 + m) | (self._Y > y1 - m))
        self.seen = np.zeros(self._edge.shape, bool)  # cells near obstacles the drone saw that aren't on the map
        self.seen_points = []                          # [(x, y)]: what was added, for the top views
        self._field = self._field_goal = None          # cost_field()
        self.set_map(map_name)

    # -- the map ------------------------------------------------------------------------------------

    def set_map(self, map_name: str):
        if map_name not in MAPS:
            raise ValueError(f'unknown map {map_name!r}; choose from {MAPS}')
        self.map_name = map_name
        self.mapped = [o for o in self.obstacles if on_map(o[3], map_name)]
        blocked = self._edge.copy()
        for kind, pos, size, _ in self.mapped:
            if kind == 'box':
                d = np.hypot(np.maximum(np.abs(self._X - pos[0]) - size[0], 0.0),
                             np.maximum(np.abs(self._Y - pos[1]) - size[1], 0.0))
            else:
                d = np.maximum(np.hypot(self._X - pos[0], self._Y - pos[1]) - size[0], 0.0)
            blocked |= d < CLEARANCE_M
        self._mapped_blocked = blocked
        self._update_blocked()

    def add_seen(self, points) -> int:
        """Put an obstacle the drone saw (its surface points, world x, y) on the map: from now on paths keep
        CLEARANCE_M from those points, as from a mapped obstacle. Returns the number of grid cells newly
        blocked (0: already known)."""
        before = int(self.seen.sum())
        for x, y in points:
            i0, i1 = self._cell(x - CLEARANCE_M, y - CLEARANCE_M), self._cell(x + CLEARANCE_M, y + CLEARANCE_M)
            a0, a1 = max(i0[0], 0), min(i1[0] + 1, self.nx)
            b0, b1 = max(i0[1], 0), min(i1[1] + 1, self.ny)
            if a0 >= a1 or b0 >= b1:
                continue
            near = np.hypot(self._X[a0:a1, b0:b1] - x, self._Y[a0:a1, b0:b1] - y) < CLEARANCE_M
            self.seen[a0:a1, b0:b1] |= near
            self.seen_points.append((float(x), float(y)))
        added = int(self.seen.sum()) - before
        if added:
            self._update_blocked()
        return added

    def _update_blocked(self):
        self.blocked = self._mapped_blocked | self.seen
        self._flat = self.blocked.ravel().tolist()  # plain list: much faster to index in the A* loop
        self._reach_from = self._reach = None
        self._field = self._field_goal = None  # the map changed: the next leg's plan computes it again

    # -- which way round: the cost-to-go field ------------------------------------------------------

    def cost_field(self, goal: tuple):
        """How far `goal` is from every cell (FIELD_STEP x CELL_M grid, 8 directions plus knight moves:
        within a few % of straight-line lengths), around what is on the map; route_cost() reads it. Kept
        until the goal or the map changes. ~10 ms on the street scene: computed with each leg's plan, never
        while deciding."""
        if self._field is not None and self._field_goal == goal:
            return
        k = FIELD_STEP
        blocked = self.blocked[::k, ::k]
        nx, ny = blocked.shape
        f = np.full((nx, ny), np.inf)
        gi, gj = (min(max(int(round(v)), 0), n - 1) for v, n in
                  (((goal[0] - self.x0) / (CELL_M * k), nx), ((goal[1] - self.y0) / (CELL_M * k), ny)))
        f[gi, gj] = 0.0
        free = ~blocked
        free[gi, gj] = True
        r = 2  # the knight moves reach two cells
        while True:
            q = np.pad(f, r, constant_values=np.inf)
            nxt = f
            for di, dj, step in _FIELD_MOVES:
                nxt = np.minimum(nxt, q[r + di: r + di + nx, r + dj: r + dj + ny] + step)
            nxt = np.where(free, nxt, np.inf)
            if np.array_equal(nxt, f):
                break
            f = nxt
        self._field, self._field_goal = f * (CELL_M * k), goal

    @property
    def field_goal(self) -> tuple | None:
        """The goal the cost-to-go field is for (None: none, or the map changed since)."""
        return self._field_goal

    def route_cost(self, x0: float, y0: float, x1: float, y1: float) -> float:
        """The length of the way to the field's goal (cost_field) that starts with a straight escape from
        (x0, y0) to (x1, y1): the escape, plus the shortest path on from where it ends. inf if the escape
        runs into a mapped obstacle (within DRONE_M) or ends where the planner won't fly, or within DRONE_M
        of it (within CLEARANCE_M of an obstacle, or by the fence: the end must have room for the drone, not
        just its centre; wallside: an end 0.01 m inside the flyable area, beside the crate, was "free");
        nan without a field."""
        if self._field is None:
            return math.nan
        d = math.hypot(x1 - x0, y1 - y0)
        if d > 1e-6 and self.distance_ahead(x0, y0, (x1 - x0) / d, (y1 - y0) / d, d + DRONE_M) < d + DRONE_M:
            return math.inf
        return d + max(self._field_at(x1 + dx, y1 + dy) for dx, dy in
                       ((0.0, 0.0), (DRONE_M, 0.0), (-DRONE_M, 0.0), (0.0, DRONE_M), (0.0, -DRONE_M)))

    def _field_at(self, x: float, y: float) -> float:
        i, j = (int(round((v - o) / (CELL_M * FIELD_STEP))) for v, o in ((x, self.x0), (y, self.y0)))
        nx, ny = self._field.shape
        return float(self._field[i, j]) if 0 <= i < nx and 0 <= j < ny else math.inf

    def distance_ahead(self, x: float, y: float, ux: float, uy: float, max_d: float) -> float:
        """Distance from (x, y) along the unit direction (ux, uy) to the first mapped obstacle surface
        (exact, not the grown grid), or max_d if there is none closer. The same thing a range-sensor ray
        would measure, if only the mapped obstacles existed."""
        best = max_d
        for kind, pos, size, _ in self.mapped:
            if kind == 'box':
                t0, t1 = -math.inf, math.inf
                for o, u, c, h in ((x, ux, pos[0], size[0]), (y, uy, pos[1], size[1])):
                    if abs(u) < 1e-9:
                        if abs(o - c) > h:
                            t0 = math.inf  # parallel to this slab and outside it: no hit
                        continue
                    a, b = (c - h - o) / u, (c + h - o) / u
                    t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
                if t0 <= t1 and t1 >= 0.0:
                    best = min(best, max(t0, 0.0))
            else:
                dx, dy, rad = x - pos[0], y - pos[1], size[0]
                b = dx * ux + dy * uy
                disc = b * b - (dx * dx + dy * dy - rad * rad)
                if disc >= 0.0:
                    t = -b - math.sqrt(disc)
                    if t >= 0.0 or -b + math.sqrt(disc) >= 0.0:
                        best = min(best, max(t, 0.0))
        return best

    def free(self, x: float, y: float) -> bool:
        i, j = self._cell(x, y)
        return self._inside(i, j) and not self._flat[i * self.ny + j]

    def reachable(self, frm: tuple, to: tuple) -> bool:
        """Whether a path exists from `frm` to `to` on the current map (flood fill, cached per start cell)."""
        start = self._escape(self._cell(*frm))
        if start is None:
            return False
        if self._reach_from != start:
            self._reach, self._reach_from = self._flood(start), start
        i, j = self._cell(*to)
        return self._inside(i, j) and self._reach[i * self.ny + j]

    # -- planning -----------------------------------------------------------------------------------

    def plan(self, start: tuple, goal: tuple) -> list | None:
        """Waypoints [(x, y), ...] from start to goal, ending exactly at goal; None if there is no way.
        A start inside a grown obstacle (after a dodge) first heads to the nearest free cell."""
        g = self._cell(*goal)
        if not self._inside(*g) or self._flat[g[0] * self.ny + g[1]]:
            return None
        s0 = self._cell(*start)
        s = self._escape(s0)
        if s is None:
            return None
        cells = self._astar(s, g)
        if cells is None:
            return None
        pts = [self._center(c) for c in self._shortcut(cells)]
        pts[-1] = (float(goal[0]), float(goal[1]))
        if s == s0:
            pts[0] = (float(start[0]), float(start[1]))
        else:
            pts.insert(0, (float(start[0]), float(start[1])))
        return pts

    def _astar(self, s: tuple, g: tuple) -> list | None:
        ny, flat = self.ny, self._flat
        n = self.nx * ny
        start, goal = s[0] * ny + s[1], g[0] * ny + g[1]
        gx, gy = g
        cost = [math.inf] * n
        came = [-1] * n
        cost[start] = 0.0
        heap = [(0.0, start)]
        while heap:
            _, cur = heapq.heappop(heap)
            if cur == goal:
                break
            ci, cj = divmod(cur, ny)
            c0 = cost[cur]
            for di, dj, step in _MOVES:
                i, j = ci + di, cj + dj
                if not (0 <= i < self.nx and 0 <= j < ny):
                    continue
                nb = i * ny + j
                if flat[nb]:
                    continue
                if di and dj and (flat[ci * ny + j] or flat[i * ny + cj]):
                    continue  # no squeezing diagonally between two blocked cells
                c = c0 + step
                if c < cost[nb]:
                    cost[nb], came[nb] = c, cur
                    dx, dy = abs(i - gx), abs(j - gy)
                    heapq.heappush(heap, (c + max(dx, dy) + (_SQ2 - 1.0) * min(dx, dy), nb))
        if came[goal] < 0 and goal != start:
            return None
        cells, cur = [], goal
        while cur != -1:
            cells.append(divmod(cur, ny))
            cur = came[cur]
        return cells[::-1]

    def _shortcut(self, cells: list) -> list:
        """Keep only the turning points: from each kept cell, go as far along the path as is visible."""
        out, anchor, k = [cells[0]], 0, 1
        while k < len(cells) - 1:
            if self._visible(cells[anchor], cells[k + 1]):
                k += 1
            else:
                out.append(cells[k])
                anchor, k = k, k + 1
        out.append(cells[-1])
        return out

    def _visible(self, a: tuple, b: tuple) -> bool:
        steps = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
        ny, flat = self.ny, self._flat
        for k in range(steps + 1):
            t = k / steps
            i, j = int(round(a[0] + (b[0] - a[0]) * t)), int(round(a[1] + (b[1] - a[1]) * t))
            if flat[i * ny + j]:
                return False
        return True

    def _escape(self, c: tuple) -> tuple | None:
        """The nearest free cell to c (c itself if free), by breadth-first search."""
        if self._inside(*c) and not self._flat[c[0] * self.ny + c[1]]:
            return c
        i0, j0 = min(max(c[0], 0), self.nx - 1), min(max(c[1], 0), self.ny - 1)
        seen, frontier = {(i0, j0)}, [(i0, j0)]
        while frontier:
            nxt = []
            for ci, cj in frontier:
                for di, dj, _ in _MOVES:
                    i, j = ci + di, cj + dj
                    if (i, j) in seen or not self._inside(i, j):
                        continue
                    if not self._flat[i * self.ny + j]:
                        return i, j
                    seen.add((i, j))
                    nxt.append((i, j))
            frontier = nxt
        return None

    def _flood(self, s: tuple) -> list:
        ny, flat = self.ny, self._flat
        reach = [False] * (self.nx * ny)
        reach[s[0] * ny + s[1]] = True
        stack = [s]
        while stack:
            ci, cj = stack.pop()
            for di, dj, _ in _MOVES[:4]:
                i, j = ci + di, cj + dj
                if 0 <= i < self.nx and 0 <= j < ny and not reach[i * ny + j] and not flat[i * ny + j]:
                    reach[i * ny + j] = True
                    stack.append((i, j))
        return reach

    def _cell(self, x: float, y: float) -> tuple:
        return int(round((x - self.x0) / CELL_M)), int(round((y - self.y0) / CELL_M))

    def _center(self, c: tuple) -> tuple:
        return round(self.x0 + c[0] * CELL_M, 3), round(self.y0 + c[1] * CELL_M, 3)

    def _inside(self, i: int, j: int) -> bool:
        return 0 <= i < self.nx and 0 <= j < self.ny


def path_length(path: list) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))


def _spans(obstacle, z: float) -> bool:
    kind, pos, size, _ = obstacle
    half_h = size[2] if kind == 'box' else size[1]
    return pos[2] - half_h <= z <= pos[2] + half_h
