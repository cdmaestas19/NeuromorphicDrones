"""Test scenarios (metres; the drone takes off at 0, 0).

Every flight plans its path on a map (planner.py), so a scenario is its stops plus that map, the scene
it runs in, and what moves (movers.py). Anything the test wants the avoidance layer to deal with is left
off the map: unmapped crates, and every mover (people and carts are never on the map). After the last
stop the drone flies home and lands.

The moving-obstacle scenarios run in the "street" scene (scenes/street.xml): a walkway along y = 1.3
(1.0..1.6) and a road along x = 1.6 (1.2..2.0). Their walks start when the drone reaches a trigger point,
so each encounter happens in the same place on every run.
"""
from typing import NamedTuple

from movers import Walk

ALTITUDE_M = 0.5


class Route(NamedTuple):
    stops: list            # (x, y) in metres, visited in order
    map: str               # what the planner knows: planner.MAPS
    about: str
    scene: str = 'obstacles'  # planner.SCENES
    movers: tuple = ()     # movers.Walk scripts
    dwell_s: float = 1.0   # hover at each stop (the drop-off)


ROUTES = {
    'square': Route([(1.0, 0.0), (1.0, -0.8), (0.0, -0.8)], 'all',
                    'open floor, every leg >= 0.45 m from any obstacle and everything mapped: the avoidance must '
                    'stay quiet (false-alarm test; also fly_waypoints.py\'s baseline)'),
    'blocked': Route([(2.5, 0.0), (2.5, -0.9), (0.0, -0.9)], 'building',
                     'the first leg runs straight at crate3 (x 1.55-2.05, y -0.25..0.25, 1 m tall), which is not '
                     'on the map: the avoidance has to dodge it'),

    # ---- moving obstacles (street scene)
    'headon': Route([(0.8, 1.3), (-2.3, 1.3)], 'all',
                    'the drone flies west along the walkway while a person walks east toward it: a head-on '
                    'approach on the front ray (closing 0.4 + 0.8 m/s)', scene='street',
                    movers=(Walk('person1', [(-2.6, 1.3), (2.6, 1.3)], 0.8, start_near=((0.8, 1.3), 0.15),
                                 start_after_s=1.0),)),
    'hover': Route([(-1.0, 0.0)], 'all',
                   'the drone hovers at a stop facing west; a person walks up behind it along its back ray. '
                   'Only the range rate shows this threat: the drone itself is not moving', scene='street',
                   movers=(Walk('person2', [(2.4, 0.0), (-2.4, 0.0)], 0.8, start_near=((-1.0, 0.0), 0.15)),),
                   dwell_s=8.0),
    'crossing': Route([(0.0, 2.3)], 'all',
                      'the drone crosses the walkway northward while a person walks along it, timed to meet at '
                      '(0, 1.3). The person comes from the side, between the rays: hard for 4 thin rays',
                      scene='street',
                      movers=(Walk('person1', [(-2.6, 1.3), (2.6, 1.3)], 0.8, start_near=((0.0, 1.3), 1.25)),)),
    'cart': Route([(2.5, 0.0)], 'all',
                  'the drone crosses the road eastward while the cart drives down it, timed to meet at (1.6, 0)',
                  scene='street',
                  movers=(Walk('cart1', [(1.6, 2.5), (1.6, -2.5)], 0.8, start_near=((1.6, 0.0), 1.3)),)),
}

# What moves during a live flight (launch.cmd deliver --scene NAME): back and forth all the time.
LIVE_MOVERS = {
    'obstacles': (),
    'street': (Walk('person1', [(-2.6, 1.3), (2.6, 1.3)], 0.7, loop=True, pause_s=2.0),
               Walk('cart1', [(1.6, -2.5), (1.6, 2.5)], 0.8, loop=True, pause_s=2.0)),
}
