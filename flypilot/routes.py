"""Test scenarios (metres; the drone takes off at 0, 0).

Every flight plans its path on a map (planner.py), so a scenario is its stops plus that map, the scene
it runs in, and what moves (movers.py). Anything the test wants the avoidance layer to deal with is left
off the map: unmapped crates, and every mover (people and carts are never on the map). After the last
stop the drone flies home and lands.

The moving-obstacle scenarios run in the "street" scene (scenes/street.xml), a 12 x 12 m courtyard: a
walkway along y = 3 (2..4) and a road along x = 3 (1.5..4.5). People walk at 1.0 m/s, the cart drives at
1.5 m/s. Their walks start when the drone reaches a trigger point, so the first encounter happens in the
same place on every run; after that the people and the cart keep going back and forth along their paths,
so later encounters (e.g. someone overtaking the drone from behind) differ from run to run.
"""
from typing import NamedTuple

from movers import Walk

ALTITUDE_M = 0.5
DELIVERY_CRUISE_M = 4.5  # deliveries in the street fly between stops this high: over people (1.7 m) and the cart
                         # (1.5 m), below the buildings; what is that tall is a streetlight, a tree's canopy, a
                         # building. They come down at each stop (avoid_demo: carefully)


class Route(NamedTuple):
    stops: list            # (x, y) in metres, visited in order
    map: str               # what the planner knows: planner.MAPS
    about: str
    scene: str = 'obstacles'  # planner.SCENES
    movers: tuple = ()     # movers.Walk scripts
    dwell_s: float = 1.0   # hover at each stop (the drop-off)
    cruise_m: float | None = None  # fly between stops this high, descending to ALTITUDE_M at each stop (a real
                                   # delivery drone cruises above people and cars); None: everything at ALTITUDE_M


ROUTES = {
    'square': Route([(1.0, 0.0), (1.0, -0.8), (0.0, -0.8)], 'all',
                    'open floor, every leg >= 0.45 m from any obstacle and everything mapped: the avoidance must '
                    'stay quiet (false-alarm test; also fly_waypoints.py\'s baseline)'),
    'blocked': Route([(2.5, 0.0), (2.5, -0.9), (0.0, -0.9)], 'building',
                     'the first leg runs straight at crate3 (x 1.55-2.05, y -0.25..0.25, 1 m tall), which is not '
                     'on the map: the avoidance has to dodge it'),

    # ---- moving obstacles (street scene)
    'headon': Route([(1.0, 3.0), (-3.0, 3.0)], 'all',
                    'the drone flies west along the walkway while a person walks east toward it: a head-on '
                    'approach on the front ray (closing 0.4 + 1.0 m/s)', scene='street',
                    movers=(Walk('person1', [(-5.5, 3.0), (5.5, 3.0)], 1.0, start_near=((1.0, 3.0), 0.15),
                                 start_after_s=1.0),)),
    'hover': Route([(-2.0, -0.6)], 'all',
                   'the drone hovers at a stop facing west; a person walks up behind it along its back ray. '
                   'Only the range rate shows this threat: the drone itself is not moving. (Their line passes '
                   '0.6 m beside the home pad: one through it would walk into the landing drone.)', scene='street',
                   movers=(Walk('person2', [(1.2, -0.6), (-5.5, -0.6)], 1.0, start_near=((-2.0, -0.6), 0.15)),),
                   dwell_s=8.0),
    'crossing': Route([(0.0, 5.2)], 'all',
                      'the drone crosses the walkway northward while a person walks along it, timed to meet at '
                      '(0, 3). The person comes from the side, between the rays: hard for 4 thin rays',
                      scene='street',
                      movers=(Walk('person1', [(-5.5, 3.0), (5.5, 3.0)], 1.0, start_near=((0.0, 3.0), 2.1)),)),
    'cart': Route([(5.2, 0.0)], 'all',
                  'the drone crosses the road eastward while the cart drives down it, timed to meet at (3, 0)',
                  scene='street',
                  movers=(Walk('cart1', [(3.0, 4.6), (3.0, -4.6)], 1.5, start_near=((3.0, 0.0), 1.5)),)),
    'lane': Route([(5.2, 0.0)], 'all',
                  'the cart drives up the road from the far end as the drone enters its 1.2 m lane (x 2.37), timed '
                  'to reach it mid-lane (x 3.4, 2.3 s later): the giant fibers alarm at ~1.3 s to impact, with the '
                  'drone near the middle of the lane, ~0.6 m from either side. Out of the lane, or up over the cart',
                  scene='street',
                  movers=(Walk('cart1', [(3.0, -4.6), (3.0, 4.6)], 1.5, start_near=((3.0, 0.0), 0.63)),)),
    'wallside': Route([(2.0, -5.0), (-2.0, -5.0)], 'building',
                      'the drone flies west along the south wall, 0.5 m inside the geofence, and a 1 m crate that '
                      'is not on the map stands on its path (0, -5). Round it on the wall side there is no room: '
                      'the map knows the wall (Sensors.route_cost), the eye and the brain see only the crate',
                      scene='street'),
    'dropoff': Route([(-1.0, 3.0)], 'all',
                     'cruising at 4.5 m, above people and the cart, the drone comes down to drop off on the walkway; '
                     'a person walks along it into the drop-off point while the drone is down there',
                     scene='street', cruise_m=DELIVERY_CRUISE_M, dwell_s=8.0,
                     movers=(Walk('person1', [(-5.5, 3.0), (5.5, 3.0)], 1.0, start_near=((-1.0, 3.0), 0.15),
                                  start_after_s=4.0),)),
    'poles': Route([(5.0, -3.0)], 'building',
                   'cruising at 4.5 m across the courtyard to a stop beyond the road: a 6 m pole that went up after '
                   'the map was made (new_pole1, at (1, -0.6)) stands on the path at cruise height. The avoidance '
                   'has to see a pole 0.24 m thick in time, high up where people and the cart are no concern',
                   scene='street', cruise_m=DELIVERY_CRUISE_M),
    'landing': Route([(3.0, -4.6)], 'all',
                     'cruising at 4.5 m to a stop where the cart is parked (no map knows about a parked cart): the '
                     'drone has to see that the drop-off point is taken, wait above it, then come down beside it',
                     scene='street', cruise_m=DELIVERY_CRUISE_M),
}

# How high live flights (launch.cmd deliver) cruise between stops; ALTITUDE_M where not listed (CrazySim's
# room: 3 m walls).
LIVE_CRUISE_M = {'street': DELIVERY_CRUISE_M}

# What moves during a live flight (launch.cmd deliver --scene NAME): back and forth all the time.
LIVE_MOVERS = {
    'obstacles': (),
    'street': (Walk('person1', [(-5.5, 3.0), (1.2, 3.0)], 1.0, pause_s=2.0),  # turns back at the kerb
               Walk('person2', [(-5.4, 0.6), (0.6, 0.6)], 0.9, pause_s=3.0),
               Walk('person3', [(-4.0, -5.3), (-1.5, -0.6)], 0.8, pause_s=2.5),
               Walk('cart1', [(3.0, -4.6), (3.0, 4.6)], 1.5, pause_s=3.0)),
}
