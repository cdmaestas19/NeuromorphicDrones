# flypilot: Crazyflie flight stack in CrazySim

Fly Brain Pilot's flight stack: a conventional flight stack, running in simulation, that the fly-brain
avoidance layer plugs into. It runs the real Crazyflie firmware in
[CrazySim](https://github.com/gtfactslab/CrazySim) on the MuJoCo physics backend, and provides:

- **One-command startup from Windows.** The 3D scene, the drone's camera view and an avoidance
  dashboard all appear in one window.
- **A baseline waypoint flight**, with safety stops: it lands on Ctrl-C and on any error.
- **An obstacle-avoidance demo** driven by the Multi-ranger range sensors. A time-to-collision rule
  feeds an explicit NOMINAL/AVOID state machine with hysteresis and a minimum dwell time.
- **Moving obstacles: people and a cart** in a street scene (a walkway and a road), in repeatable
  encounter scenarios. The avoidance measures how fast things approach the drone, and every run
  reports the closest approach and any contact.
- **Path planning on a map, for every flight.** The planner routes around every obstacle on its map.
  Whatever isn't on the map is left to the avoidance layer.
- **Delivery stops chosen during the flight.** Add stops in the **Stops window** (or by double-clicking
  the floor in the 3D view), the way a customer's order would arrive.
- **One swappable decision interface** (`AvoidancePolicy`), where the avoidance policies plug in.
- **The fly brain as a policy** (`--policy flybrain`): a 360-degree compound eye feeds the escape
  circuit of the fruit-fly connectome, a spiking network running in its own process. It is described
  in [../flybrain/README.md](../flybrain/README.md).
- **Decision-cost measurement** and a record-and-replay benchmark, because the end target is a small
  drone with little computing power.

The fly circuit only ever decides *when* and *which way* to dodge; the firmware's controller always
keeps the drone stable.

The step-by-step checklist for a new PC, with expected output, is in [run.txt](run.txt).

---

## Quick start (Windows)

You need Windows 11 (or Windows 10 21H2 or newer), WSL with Ubuntu 24.04, and a GPU with an
up-to-date driver. Any GPU works; it doesn't have to be a powerful one. Details are in run.txt section 1.

```
flypilot\setup.cmd             one time, ~5-10 min; safe to re-run after git pull
flypilot\launch.cmd waypoints  baseline flight
flypilot\launch.cmd avoid      avoidance demo
flypilot\launch.cmd deliver    you pick the stops, in the Stops window
flypilot\launch.cmd avoid --route headon     a person walks at the drone (all the tests: next section)
flypilot\launch.cmd avoid --route hover --policy flybrain    the same tests, decided by the fly brain
```

Run these from a normal Windows terminal in the repo folder. The `.cmd` files run everything inside
WSL for you.

## Run a test yourself

```
flypilot\launch.cmd avoid --route headon --policy flybrain
```

- **Choose the scenario** with `--route`:

  | Route | What happens |
  |---|---|
  | `hover` | a person walks up behind the hovering drone |
  | `headon` | a person walks at the drone |
  | `crossing` | a person crosses its path |
  | `cart` | the cart crosses its path |
  | `lane` | the cart reaches it mid-lane |
  | `dropoff` | cruise at 4.5 m; a person walks into the drop-off point |
  | `poles` | an unmapped pole at cruise height |
  | `landing` | the cart is parked on the drop-off point |
  | `wallside` | an unmapped crate by the wall |
  | `square` | open floor: the false-alarm test |
  | `blocked` | an unmapped crate on the path |

- **Compare with the geometric rule:** leave out `--policy flybrain`.
- **Where the windows open:** on a screen other than your main one, if you have two. `--screen main`
  keeps them on the main screen; `--headless` opens no windows at all (run tests while you use the PC
  for something else).
- **What you'll see:** the 3D view, with the dashboard on its right side showing what is making each
  decision. The terminal logs each `AVOID episode`, and at the end the closest approach to each person
  or cart, and any contacts.
- **To stop:** Ctrl-C lands the drone and shuts everything down; closing the 3D window also shuts
  everything down.
- **Your own deliveries:** `flypilot\launch.cmd deliver --scene street --policy flybrain`. Click the
  map in the Stops window to add stops; it cruises at 4.5 m between them, with people and the cart
  moving the whole time.

Don't start a flight while another is running: a new launch shuts down the simulator that is up.

## Run order, and what you should see

| # | Command | What appears | Pass if |
|---|---|---|---|
| 1 | `flypilot\setup.cmd` | no windows; build output in the terminal | it ends with `Setup finished` |
| 2 | `flypilot\launch.cmd` | the **"MuJoCo : Drone scene with obstacles"** window: a 6 x 6 m room with gray walls, brown pillars and green crates, and the drone on the floor in the middle. The drone's gray camera view is inset top-left, and `RTF: x.xx` is shown bottom-left | the terminal prints `Graphics: D3D12 (<your GPU>)`; RTF reads 0.97-1.03 |
| 3 | `flypilot\launch.cmd waypoints` | the same window; the drone takes off to 0.5 m, flies a 1.0 x 0.8 m square (turning to face each leg), and lands at the start | the terminal ends with `exit code 0`, with no `SAFETY STOP` |
| 4 | `flypilot\launch.cmd avoid` | the same window, plus the **avoidance dashboard** along its right edge, and the separate **"Delivery stops"** window. The route's three stops appear as markers. The drone flies straight at a crate (it isn't on this route's map), sidesteps it, passes it, visits the other stops, and lands back home | exactly one `AVOID episode` line, `3 delivered, 0 skipped with 1 avoidance episode(s)`, `exit code 0` |
| 5 | `flypilot\launch.cmd avoid --route square` | as in 4, on the clear route with everything mapped | `0 avoidance episode(s)` (no false alarms) |
| 6 | `flypilot\launch.cmd deliver` | as in 4, with no stops yet; the drone hovers over a blue **H**. Click the Stops window's map to add stops | each stop turns green (`stop N delivered`); **Go home** brings it back to land (`back home`, `exit code 0`) |
| 7 | `flypilot\launch.cmd avoid --route headon` (then `hover`) | the **"MuJoCo : Street scene with people and a cart"** window. A person (blue) walks along the walkway at the drone; in `hover`, one walks up behind the hovering drone | an `AVOID episode` line well before contact, `contacts: none` |
| 8 | `flypilot\launch.cmd avoid --route crossing` (then `cart`) | a person crossing the drone's path, or the cart driving across it | these record a contact with today's geometric policy (see [Moving obstacles](#moving-obstacles-people-and-a-cart)); pass = the flight ends with a `closest approach` line |
| 9 | `flypilot\launch.cmd --camera-window` | also a separate **"AI-deck camera"** window | it reads `324x244  ~20 fps` |

**To stop:** close the MuJoCo window or press Ctrl-C. Ctrl-C during a flight lands the drone first;
a second Ctrl-C cuts the motors. `flypilot\stop_sim.cmd` cleans up anything left running.

**RTF** is how fast the simulation runs compared to real time. The firmware runs on the real clock, so
an RTF well below 1.00 breaks its timing and flights fail.

### Stops and the Stops window

Every `avoid` and `deliver` flight is a list of stops. A real delivery drone doesn't know them before
take-off: a customer tells it where to go. You play the customer in the **"Delivery stops"** window,
which opens next to the 3D window:

| In the Stops window | Does |
|---|---|
| left-click the map | adds a stop there: an orange numbered marker. The drone flies to the stops in order, hovers 1 s at each one (the drop-off), and marks it green |
| click a stop, right-click it, or press its **X** in the list | removes it. A stop that isn't done yet is cancelled, even while the drone is on its way there |
| **Go home** (or click H) | returns home and lands once the queued stops are done; press again to undo |
| **Cancel open stops** / **Clear finished** | empties the queue / tidies away delivered and skipped stops |
| **Path planner knows: all / building / none** | which obstacles are on the planner's map (see below) |

- **Live checks.** A green ring follows the mouse over the map, and turns red where a stop would be
  refused.
- **Same actions in the 3D view.** Double-clicks there work the same, including on a stop's marker ball.
- **Between stops.** With nothing queued, the drone hovers and waits. Ctrl-C lands it where it is, as in every flight.
- **Refused stops.** A stop on an obstacle, outside the flight area, within 0.35 m of any obstacle or
  wall, or with no way to reach it, is refused. The reason appears in the Stops window and in the terminal.
- **Unreachable in practice.** If a stop can't be reached in time, the drone skips it (it turns red)
  and goes on to the next.
- **Markers are only for you.** They exist only in the 3D view, not in the simulated world, so the
  drone's camera and range sensors never see them.

### Path planning, and what it leaves to the avoidance

Each leg follows a path planned on a map (`planner.py`):
- a 5 cm occupancy grid of the room at flight altitude
- the map's obstacles grown by 0.3 m
- A* search, then straightened into a few legs

The drone flies the legs without stopping, slowing down before sharp turns. A plan takes a few
milliseconds and runs once per leg, never in the 20 Hz control loop.

**Whatever is not on the map is the avoidance layer's job.** When the avoidance dodges something, the
drone re-plans from wherever the dodge left it. Three maps:

| Map | The planner knows | Used by |
|---|---|---|
| `all` | every obstacle in the scene | `deliver` (default), `square`, `waypoints` |
| `building` | walls, pillars and partitions, but not the crates. The crates are clutter nobody mapped | `blocked`: the avoidance must handle crate3. This is the realistic test for a reactive policy |
| `none` | nothing: straight lines | limit tests of the avoidance on its own |

New test scenarios belong in `routes.py`, as a list of stops plus a map. Leave any obstacle the test
should exercise the avoidance on off the map. `waypoints` has no avoidance, so it always plans on
`all`.

For a real deployment the map would come from a floor plan, from building footprints (for example
OpenStreetMap), or from the drone's own sensors. The planner only needs a list of obstacles.

**Scripted runs.** For repeatable tests, give the stops up front:
`launch.cmd deliver --at 2.5,0 --at 2.5,-0.9 --at 0,-0.9 --return`. Here `--return` sends the drone
home after the last stop instead of waiting for more. To choose the map, add `--map building`. You can
still add stops in the window.

### Moving obstacles: people and a cart

These tests run in the **street scene** (`scenes/street.xml`), a 12 x 12 m courtyard. It has a 2 m
walkway across it (y = 2 to 4), a 3 m road down it (x = 1.5 to 4.5) with a zebra crossing, and trees,
benches, a lamp post, a kiosk and crates. Three people (1.7 m tall, walking 1 m/s) and a golf-cart-sized
cart (1.2 x 2.2 x 1.5 m, 1.5 m/s) are scripted in `routes.py` (`movers.py`), and they go back and forth
for the whole flight. The simulator moves them; the range sensors, the camera and the compound eye see
them like anything else. Movers are never on the planner's map: dealing with them is the avoidance
layer's job. `launch.cmd` starts the simulator on the scene that the route asks for.

Each first encounter starts when the drone reaches a trigger point, so it happens at the same place on
every run. Later ones (someone turning round and overtaking the drone, the cart coming back) vary. At
the end of a run the terminal prints the closest approach to each mover (drone edge to mover edge) and
any contacts. This is the "minimum distance to people" measurement from the project plan. Height
counts, part by part: a drone above the cart's front body (1.05 m) is not touching it, though its
cabin (1.49 m) behind is higher. The same goes for a person's shoulders (1.49 m) and head (1.67 m).

| Route | What happens |
|---|---|
| `headon` | a person walks along the walkway at the drone, then turns round and comes back |
| `hover` | a person walks up behind the hovering drone, then back again as it flies home |
| `crossing` | a person walks along the walkway while the drone crosses it |
| `cart` | the cart drives down the road while the drone crosses it, and comes back up |
| `lane` | the cart drives up the road as the drone enters its lane, timed to reach it mid-lane. The alarm comes with the drone ~0.6 m from either side of the lane: it has to get out faster than it cruises, or go up over the cart |
| `dropoff` | the drone cruises at 4.5 m, above people and the cart, and comes down carefully to drop off on the walkway; a person walks into the drop-off point |
| `wallside` | no movers: along the south wall, an unmapped 1 m crate on the path with no room on the wall side (the map's veto, see below) |

**Results** (2026-10-24, all routes on the same code, headless; the fly brain flew each twice). The
closest approach to the person or cart is drone edge to mover edge; static routes say whether every
stop was delivered.

| Route | Geometric rule (4 rays) | Fly brain (360-degree eye), run 1 / run 2 |
|---|---|---|
| `square` (false alarms) | 1 false alarm | 0 / 0 |
| `blocked` | delivered | delivered / delivered |
| `hover` | 0.04 m | 0.16 m / 0.21 m |
| `headon` | **contact** | 0.20 m / 0.21 m |
| `crossing` | **contact** | 0.42 m / 0.44 m |
| `cart` | **contact, crash** | 0.29 m / 0.18 m |
| `lane` | **contact, crash** | 0.34 m / 0.31 m |
| `dropoff` | **contact, crash** | 0.12 m / 0.23 m |
| `wallside` | delivered | delivered / delivered |
| `poles` | delivered | delivered / delivered |
| `landing` | delivered beside the parked cart | delivered beside it / beside it |

The geometric rule's contacts are threats it saw late, coming in between or along its four rays. The
fly brain also has the 360-degree eye, so this compares sensor sets as much as deciders; see
[../flybrain/README.md](../flybrain/README.md#results) for what the brain itself decided.

`launch.cmd deliver --scene street` gives live stops in the street, with people walking and the cart
driving back and forth the whole time. Live flights there cruise high and come down carefully at each
stop (next section; `--cruise M` changes the height, `--cruise 0` flies everything at 0.5 m).

### Cruising high, and coming down carefully

A delivery drone flies high and only comes down at its stops. In the street, deliveries (`deliver`,
`dropoff`, `poles`, `landing`) cruise at **4.5 m** (`routes.DELIVERY_CRUISE_M`): over people (1.7 m) and
the cart (1.5 m). Up there the hazards are the tall things: the courtyard's walls are 8 m buildings,
the streetlight is 6 m with an arm at the top, the trees' canopies reach from 3.3 to 5.7 m, there is a
7 m building block, and a 6 m pole (`new_pole1`) that went up after the map was made. The planner
routes round what is on its map at cruise height; the avoidance handles the rest. The geofence's
ceiling is 0.5 m below the buildings' tops, at most 6 m. Climbs and descents go at up to 1 m/s, slowing
to 0.4 m/s below 1.5 m, and climbs brake before the ceiling.

**Coming down at a stop** (`avoid_demo._come_down`):
1. Down to 2.5 m, above people's heads, and look at the drop-off point with the eye.
2. The point is **taken** if anything within 0.7 m of it stands 0.25 m or more tall and isn't part of
   an obstacle in the scene file: a person, a parked cart, something new.
3. While it's taken, the drone waits above it. Once it has been clear for 1 s, the drone comes down.
4. Still taken after 6 s (something parked there): the drone comes down at the nearest clear spot round
   it instead (0.9-2.4 m away, a spot a stop could be at). After 40 s with no clear spot it skips the stop.
5. On the way down, the descent pauses while the spot is taken. The avoidance stays on throughout, and
   the final landing at home watches its spot the same way.

| Route | What happens |
|---|---|
| `poles` | cruise at 4.5 m across the courtyard; the unmapped 6 m pole stands on the path at cruise height |
| `landing` | the stop is where the cart is parked: wait above, then come down beside it |
| `dropoff` | cruise at 4.5 m, come down on the walkway; a person walks into the drop-off point |

### Round or over, and which way round

Two rules shape every escape, for both the geometric and the fly-brain policy:

- **A person: round them, never over.** A person is tall, and in most places flying over people is
  restricted. Nor does the drone climb with anyone within 1.5 m, whatever it is escaping from
  (`policies.PERSON_CLEAR_M`). In 24 recorded person flights, no climb would have been allowed
  closer than 1.9 m to the walking person. **A vehicle: up over it**, while getting out of its path. A car or the cart is too wide and
  too fast to get out of the way of in time: in the `lane` test the alarm comes about 1.3 s before the
  cart arrives, and its lane is 1.2 m wide. Static things: as before (the fly brain climbs, as a fly's
  escape takeoff does; the geometric policy goes round).
- **Which way round: the shorter way on to the stop.** With each leg's plan, the planner works out how
  far the stop is from everywhere on its map (`planner.cost_field`, ~10 ms). At an alarm, the policy
  compares going round the threat on the left with going round it on the right (`Sensors.route_cost`).
  The threat itself is never on the map (the planner would have gone round it), so the policy measures
  from a point just past the threat on each side.
  - **Geometric policy:** the shorter side (by 0.3 m or more), unless its ray on that side sees
    something within 0.4 m.
  - **Fly brain:** it keeps its own side when its steering neurons are clearly lateralized, unless the
    other way is 1 m or more shorter. When they aren't, the map chooses, then the eye.
  - **Moving threats:** the side is still the one that gets out of the mover's path soonest. The map only
    breaks a near-tie, or vetoes a way that runs into something it knows.
  - **What is logged:** each episode records what chose the side (`side_from`: brain, eye, map, motion);
    the fly brain's end-of-flight summary counts them.

**How the drone knows what a threat is.** Two senses, each strongest where the other is weak:

- **The compound eye's size of it** (`eye.CollisionCourse`). A thing that moves and is wider than
  0.75 m is a vehicle; a narrower one, a person; one wider than 3 m is something merged (a person with a
  wall) and decides nothing. This works all round. The cart reads 0.8-2.3 m wide, a person 0.2-0.6 m.
  An episode counts as a vehicle only if the eye says so twice as often as it says person. A person
  walking past a bench can read as one vehicle-sized thing for a while; the cart's first readings, a
  sliver of it, look like a person but are few.
- **The camera** (`camera_kind.py`). A small CNN (~6 k weights, the size the AI-deck's GAP8 runs) looks
  at a strip of the image at the threat's bearing and says background, person or vehicle. When it sees
  the threat and is sure (p >= 0.8), its verdict wins: it can tell what a thing is, the eye only how big.
  It only sees ahead (~86 degrees).
  - **Training:** `flypilot\camera_train.cmd` trains it on the recorded flights in `recordings\`
    (`avoid_demo.py --record`). Every strip is labelled automatically from where the simulator put
    each person and the cart (`camera_kind.label_strips`); whole flights are held out for the test. The
    first run sets up its own Python environment with PyTorch (~200 MB, CPU only). The weights are
    `models/camera_kind.npz`; the flight runs them in numpy (~1 ms).
  - **Limitation:** the sim's people and cart are plain shapes. A network trained on them won't
    recognise real ones; for real flights it needs real (or heavily randomised) images.

**How well the camera network does** (trained on 105 recorded street flights, 75,278 labelled strips;
tested on 26 other flights, 15,180 strips):

| | background | person | vehicle |
|---|---|---|---|
| **true background** | 9,050 | 489 | 584 |
| **true person** | 412 | 3,516 | 164 |
| **true vehicle** | 86 | 15 | 864 |

- **Accuracy:** 88.5% of strips.
- **When sure** (p >= 0.8, 78% of strips): right 94.8% of the time.
- **"Person" (sure):** right 95.5%.
- **"Vehicle" (sure):** right only 74.2%. Its usual mistake is background (a box, the road) taken
  for the cart.
- **Per avoidance episode, on those held-out flights:** with the person ahead (`headon`) the camera
  agreed with the eye in every episode where it saw enough of the threat. From behind or the side
  (`hover`, `cart`, `lane`) it mostly never saw the threat, so the eye decided.
- **A stray reading:** once, a single sure "vehicle" reading would have taken the drone up beside a
  person. So the camera's verdict counts only after 3 sure readings in an episode
  (`_ThreatKinds.CAMERA_MIN`).

The `wallside` route tests the map's veto: a crate that is not on the map stands on the drone's path
along the south wall, with no room to pass it on the wall side.

### Other commands

| Command | Does |
|---|---|
| `flypilot\launch.cmd deliver --at X,Y [--at X,Y ...] [--return] [--map all\|building\|none] [--scene street]` | stops given at launch (metres, the drone starts at 0,0), as described above. `--at` and `--map` also work with `avoid` |
| `flypilot\launch.cmd avoid --route headon\|hover\|crossing\|cart\|lane\|dropoff\|wallside` | the street tests above |
| `flypilot\launch.cmd avoid\|deliver ... --policy flybrain` | any flight, with the fly brain deciding instead of the geometric policy |
| `flypilot\launch.cmd avoid\|deliver ... --cruise 3` | fly between stops at 3 m (street default 4.5 m) and come down to 0.5 m at each stop |
| `flypilot\launch.cmd ... --screen main` | where the windows open: `auto` (default: a screen other than the main one, if there is one), `main`, `1`, `2`, ..., `off` (`screens.py`) |
| `flypilot\launch.cmd ... --headless` | no windows at all: for test runs while the PC is used for something else (a game). The flight and its log are the same |
| `flypilot\brain_reproduce.cmd`, `flypilot\brain_validate.cmd` | check the fly-brain model against published results, and its cut-out escape circuit against the whole brain (see [../flybrain/README.md](../flybrain/README.md)) |
| `flypilot\launch.cmd avoid --record recordings\blocked.jsonl` | also saves every decision's inputs and outputs |
| `flypilot\bench.cmd recordings\blocked.jsonl [--policy NAME]` | replays a recording offline (no sim needed), checks the decisions still match the recording, and prints the cost per decision |
| `flypilot\run_sim.cmd`, `flypilot\camera.cmd` | the sim, or the separate camera window, on their own (one terminal each) |
| `bash flypilot/py.sh <script.py>` | inside WSL or on native Linux: runs any script in the project's Python environment |

---

## How it fits together

```
Windows terminal --> launch.cmd --> WSL: launch.py
                                      |-- run_sim.sh --> CrazySim sitl_camera.sh
                                      |                    cf2 (Crazyflie firmware, SITL)
                                      |                      <-> crazysim.py (MuJoCo physics, 3D window, camera render)
                                      |                      <-> crazysim_cpx.py (AI-deck WiFi bridge emulator)
                                      |-- dashboard.py --(shared memory)--> side panel in the 3D window
                                      |     '-- Stops window (stops_window.py) --requests--> avoid_demo.py
                                      |-- ../flybrain/brain_server.py (only for --policy flybrain)
                                      '-- fly_waypoints.py / avoid_demo.py --cflib UDP--> firmware
                                            avoid_demo.py <--double-clicks / markers--> crazysim.py
                                            avoid_demo.py --mover positions--> crazysim.py (people, cart)
                                            avoid_demo.py <--compound eye-- crazysim.py
                                            avoid_demo.py <--looming / spike counts--> brain_server.py
```

| Port / path | Between | Carries |
|---|---|---|
| UDP 19850 | flight scripts (cflib) <-> crazysim.py | CRTP: commands and telemetry (`udp://127.0.0.1:19850`) |
| UDP 19950 | crazysim.py <-> firmware | sensor data and motor commands (internal) |
| UDP 5200 | crazysim.py -> camera bridge | raw camera frames (internal: do not listen here) |
| TCP 5050 | camera bridge -> one client | AI-deck CPX stream, the same protocol as the real deck (port 5000 on hardware) |
| UDP 5300 | avoid_demo.py -> dashboard.py | one JSON snapshot every 0.1 s |
| `/dev/shm/crazysim_overlay.rgb` | dashboard.py -> crazysim.py | the side-panel image |
| UDP 5310 | Stops window and crazysim.py -> avoid_demo.py (`orders.py`) | requests (add, remove, go home, map) and double-clicked points, as JSON |
| UDP 5311 | avoid_demo.py -> crazysim.py | markers to draw in the 3D view, as JSON (refreshed every second; dropped after 3 s without a refresh) |
| UDP 5312 | avoid_demo.py (`movers.py`) -> crazysim.py | each mover's position and velocity at 20 Hz; the sim keeps them moving smoothly between messages, and stops them if messages stop |
| UDP 5320 | avoid_demo.py (`FlyBrainPolicy`) <-> brain_server.py | each tick: the two eyes' looming drive in, the descending neurons' spike counts back (JSON) |
| UDP 5330 | crazysim.py -> avoid_demo.py (`eye.py`) | the compound eye: 1800 distances (72 azimuths x 25 elevations, 5 degrees apart, from straight down to 30 degrees up) at 20 Hz |

### The avoidance loop (20 Hz)

```
sensors + camera frame --> POLICY --> avoid signal (0..1) + dodge velocity
avoid signal + Multi-ranger clearance --> SUPERVISOR --> NOMINAL or AVOID
NOMINAL: fly the planned path    AVOID: fly the policy's dodge, then re-plan    the firmware PID keeps it stable
```

The avoidance runs on every tick the drone is flying a leg or hovering at a stop.

- **Supervisor (`avoidance.py`).** It enters AVOID when the signal is at least 0.50. It returns to the
  route only after **both** of these have held without interruption for 0.4 s: the signal is below 0.25,
  and the clearance is above 0.5 m. It never returns sooner than 0.4 s after entering AVOID. A single
  threshold would make the drone chatter in and out of AVOID near an obstacle: in a test with a noisy
  signal hovering at 0.5, a single threshold flipped 97 times in 10 s, and this supervisor switched once.
- **Geometric policy (`policies.py`): time-to-collision, chosen over a potential field.** For each
  horizontal range sensor, TTC = distance / closing speed, and signal = 1 s / TTC of the worst one.
  - It yields exactly the one "how soon do I hit something" number that the thresholds need.
  - It is what a fly's looming circuit estimates from vision, so this baseline and the fly-brain
    policy produce comparable signals.
  - A potential field gives a direction rather than a threat level, and stalls in front of an
    obstacle dead ahead on the route.
  - **Closing speed has two parts.** The first is the drone's own motion, counted only while the path
    still commands it forward, so braking into a planned stop isn't a threat. The second is the
    obstacle's own motion, from how fast the range shrinks.
  - **That second part is a straight-line fit** over the last 0.25 s of new readings. It is ignored
    while the drone turns, while it moves across the sensor's direction, and when the readings jump
    between surfaces. On recorded static flights it adds no false alarms.
  - **The dodge gets out of the line of approach.** For something ahead: sidestep toward the freer
    side, keep going a little, then pass it; if it is still ahead, sidestep again. For something
    moving ahead, sidestep faster and further, then let it go by. For something from the side or
    behind: move across its line of approach.
- **Flight-side safety around the policy (`avoid_demo.py`).** These rules apply to every policy:
  - **Speed near mapped obstacles.** Near obstacles on the map, the flight slows down so that its time
    to collision with them stays above 2.5 s. The avoidance then never sees a collision course with
    something the planner already routes around.
  - **Clear-out.** When a dodge ends too close to something for the supervisor to resume, the drone
    backs out (`avoidance.clear_out`).
  - **Geofence.** The setpoint never goes within 0.1 m of the geofence. The geofence is 0.35 m inside
    the scene's walls and 0.5 m below their top, at most 6 m (`planner.geofence`): +-2.6 m and 2.5 m
    in the 6 x 6 m room, +-5.6 m and 6 m in the street. Climbs brake before the ceiling (an escape climb
    that stopped dead overshot it once).
  - **No dodging into things.** A dodge slows down rather than fly into what the compound eye sees in
    its path.
  - **Down only when clear below.** After a dodge climbed, or when descending to a stop, the drone goes
    down only while the eye's downward-looking rows see room under it (within 0.6 m of it: a tall wall
    beside the drone isn't under it).
  - **Remember what was dodged.** After each episode, unmapped static things the eye saw nearby go on
    the planner's map (orange dots), so the re-planned path goes round them.
  - **Resume clearance from the eye.** The supervisor's 0.5 m resume clearance is checked on everything
    ahead and to the sides (within 100 degrees of the heading), not only on three rays.
  - **Land with the avoidance on.** The drone keeps deciding (and dodging) down to 0.3 m above the
    landing spot; only the last 0.3 m is the pilot's plain touchdown.
- **Fly-brain policy (`policies.py` `FlyBrainPolicy`, `eye.py`).** The compound eye's looming, per eye,
  drives the looming detectors of the connectome's escape circuit. The giant fibers' firing rate is the
  avoid signal, and the side whose steering neurons fire more is the side to escape from. The escape
  is a 1 s burst at 0.4 m/s sideways, plus forward or back away from where the eye sees the looming,
  then a hold while the threat passes. From a moving threat it goes straight out of its path, as fast
  as that takes (0.4-0.8 m/s). Up over a vehicle, never over a person; the map can choose the side (see
  [Round or over](#round-or-over-and-which-way-round)). See
  [../flybrain/README.md](../flybrain/README.md).
- **Flight stack (`flight.py`).** cflib connection (bounded in time, with retries), telemetry at 20 Hz,
  and a `Pilot` that streams position setpoints and moves them smoothly toward each goal. Every tick
  it runs safety checks: link lost, stale telemetry, geofence, and the firmware reporting tumbled or
  crashed.

### Files

| File | Purpose |
|---|---|
| `launch.py` / `launch.cmd` | one-command startup and shutdown of everything |
| `flight.py` | cflib link, telemetry, `Pilot`, safety stops, `fly_mission()` |
| `fly_waypoints.py` | the baseline flight (planned on the full map, no avoidance) |
| `avoid_demo.py` | stops on planned paths plus the avoidance layer; `--route` (`live` = no stops yet), `--at`, `--return`, `--map`, `--policy`, `--record` |
| `orders.py` | `OrderBook`: the stop queue. It checks requests, interprets clicks, previews the planned route, and sends the markers |
| `planner.py` | the map (`load_obstacles` from the scene file; `all` / `building` / `none`), the occupancy grid and A* |
| `stops_window.py` | the Stops window (run by `dashboard.py --stops`) |
| `movers.py` | people and carts: scripted walks (`Walk`), streaming them to the sim, closest-approach and contact counts |
| `scenes/street.xml` | the street scene: a 12 x 12 m courtyard with a walkway, a road, three people and a cart |
| `avoidance.py` | `Sensors`, `AvoidCommand`, the `AvoidancePolicy` interface, the supervisor, `clear_out`, `CostMeter` |
| `policies.py` | `GeometricTTCPolicy`, `FlyBrainPolicy`, and the `POLICIES` registry |
| `eye.py` | the compound eye: `EyeSource` receives it from the sim, `Looming` turns it into each eye's looming drive; `CollisionCourse` tracks what is around and how big it is |
| `camera_kind.py` | person / vehicle from the camera: `CameraKind` (used by the policies), the network's numpy forward pass, `label_strips` (automatic labels) |
| `vision_train.py`, `camera_train.cmd` / `.sh`, `vision-requirements.txt` | training that network on recorded flights (its own venv); weights in `models/camera_kind.npz` |
| `routes.py` | test scenarios (stops, map, scene, movers): `square`, `blocked`, `headon`, `hover`, `crossing`, `cart`, `lane`, `wallside`, `dropoff`; what moves in live flights |
| `frames.py` | `FrameSource`: AI-deck camera frames over CPX/TCP; `get_frame()` waits for a new frame, `latest()` returns the newest |
| `camera_view.py` | the separate camera window |
| `dashboard.py` | the avoidance dashboard (drawn into the 3D window; `--window` for its own window; `--stops` adds the Stops window) |
| `bench.py` | offline record-and-replay benchmark |
| `setup.*`, `system_deps.sh`, `env.sh`, `py.sh`, `run_sim.*`, `stop_sim.*` | setup and plumbing |
| `patches/` | our fixes to CrazySim, applied by setup (see below) |
| `run.txt` | test checklist for lab mates, with expected output |

---

## Adding a policy

A policy sees the sensors and the newest camera frame each tick. It returns an avoid signal and a
dodge. Nothing else in the stack needs to change. `FlyBrainPolicy` in `policies.py` is a worked example
of a policy with its own sense (the compound eye) and its own process (the brain):

```python
# policies.py
class LoomingPolicy(AvoidancePolicy):
    name = 'Looming detector'
    inputs = ('camera',)                  # the dashboard shows which sensors are in use

    def step(self, sensors, frame):       # frame: 244 x 324 uint8 grayscale, or None
        signal = ...                      # 0..1, how strongly to avoid
        dodge = (0.0, 0.3, 0.0)           # body-frame velocity while in AVOID: forward, left, up [m/s]
        return AvoidCommand(signal, dodge, {'maneuver': 'dodge left'})

POLICIES['looming'] = LoomingPolicy
```

Then run `flypilot\launch.cmd avoid --policy looming`. The supervisor's thresholds, dwell and resume
gate apply unchanged, which keeps the comparison between policies fair. `reset()` is called before
each flight and after every avoidance episode.

`Sensors` contains:
- `t`, `position`, `yaw_deg`
- `velocity_world`, `velocity_body`
- `intended_velocity_body` (what the planned path is asking for)
- `ranges` (front/back/left/right/up in metres; `inf` means nothing within 4 m)
- `mode` (the supervisor's mode on the previous tick)

## Decision cost

The end target is a small drone, so each decision is timed. Measured on this PC, in Python:

| | Per decision |
|---|---|
| geometric rule alone (the four rays; replayed offline, `bench.cmd`) | ~3.6 us |
| geometric policy as it flies now: the rule plus the eye's object tracking (what a threat is) | ~0.57 ms replayed |
| the camera network, per look at a threat (numpy) | ~0.2 ms |
| during a live flight, geometric (printed at the end of each `avoid` run, and on the dashboard) | ~1.5 ms mean |
| during a live flight, fly brain (eye, camera, the round trip to the brain; not the brain's own ~5-30 ms, in its own process) | ~1.6 ms mean |
| budget: one 20 Hz control tick | 50 000 us |

By operation count, the geometric rule plus the supervisor is under 10 divisions and ~200 simple
operations per decision. Its state is the maneuver plus the last few range readings, about 300 bytes.
On the Crazyflie's STM32F405 that should take microseconds. This is an estimate, not a measurement on
hardware. What makes the policies cost milliseconds now is telling what a threat is: tracking the
eye's objects in Python, and the camera network (on a real Crazyflie that network would run on the
AI-deck's GAP8). Path planning (a few ms) and the map's cost-to-go field (~15 ms) run once per leg,
never per decision.

The heavy compute will come later, from camera preprocessing (cost grows with the number of pixels)
and from the spiking network itself (cost grows with neurons and synapses). Guidelines for new
decision code:
- write plain scalar code
- do no per-tick allocation
- downsample camera frames early
- check every change with `bench.cmd`, which must still report MATCH

Recordings include camera frames (`X.frames.npz`, half resolution), but `bench.cmd` doesn't replay them
yet, so camera-based parts get `frame=None` there.

Reaction time is currently limited by sensor and loop rates, not by compute. The range sensors update
at 10 Hz, and telemetry and the control loop run at 20 Hz, so it can take up to ~150 ms from an
obstacle appearing to a reaction.

---

## What we changed in CrazySim

CrazySim is pinned to commit `3ec8b55`. `setup.cmd` checks it out into `~/CrazySim` inside WSL and
applies the patches in `patches/`:

- **`crazyflie-firmware/sitl-64bit-tick-stabilizer.patch`.** A one-line firmware fix. The simulated
  firmware has 64-bit timer ticks, but the stabilizer kept its wake-up time in a 32-bit variable. With
  GCC 13 (Ubuntu 24.04) the stabilizer never sleeps, the firmware freezes at "Wait for sensor
  calibration", and cflib cannot connect.
- **`crazyflie-simulation/crazysim-wsl-realtime.patch`.** Keeps the simulator at real time under WSL,
  and adds the overlays:
  - the 3D window's text overlay and the camera render no longer block the physics loop
  - the physics loop catches up after slow steps
  - floor reflections are off (also more realistic for the camera)
  - the 3D window shows the camera inset and any side panel written to shared memory
  - double-clicks on the scenery are forwarded to UDP 5310 (with the camera position, so a click on a
    marker ball can select it), and markers sent to UDP 5311 are drawn in the 3D view only
  - the scene's mocap bodies (people, a cart) move as UDP 5312 commands them
  - each horizontal range sensor casts a level fan of three rays (-10, 0 and +10 degrees) and reports
    the nearest hit. The real VL53L1x sees a cone of about 27 degrees, and a single thin ray missed
    box corners until 0.07 m away
  - a compound eye: every 5 degrees all the way round and from straight down to 30 degrees up, the
    distance to the first surface, 20 times a second, to UDP 5330. Rays that
    graze a surface at more than ~75 degrees off its normal report "unknown": a ray skimming a crate top
    at flying height moves its hit point ~10 cm per 1 cm of bob, which reads as looming

Both bugs are worth reporting upstream.

Two facts about CrazySim that differ from older notes:
- It no longer ships its own cflib: PyPI `cflib==0.1.34` has the UDP driver.
- Camera frames are read from TCP 5050 (the CPX stream), not from UDP 5200.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `no GPU acceleration` warning, RTF well below 1.00, flights drift or tumble | Update the GPU driver, then `wsl --update` and `wsl --shutdown`. Remote Desktop sessions and VMs usually have no GPU for WSL. |
| No windows appear | `wsl --update` (WSLg; Windows 10 needs 21H2+). |
| `X11: Failed to open display :0` / `could not initialize GLFW` in `flypilot\logs\sim.log` | WSL's graphics service (WSLg) didn't start with WSL. `wsl --shutdown`, then launch again. |
| `the fly brain is not answering on udp://127.0.0.1:5320` | Check `flypilot\logs\brain.log`. `no brain environment`: re-run `setup.cmd`. |
| The fly brain never reacts to the people in the street | Re-run `setup.cmd` (the compound eye is part of the CrazySim patch; `flypilot\logs\sim.log` should list `compound eye: 1800 ommatidia`). A scene's moving bodies each need an `<inertial>`: see the comment in `scenes/street.xml`. |
| The fly brain reacts late (its end-of-flight summary counts many "answers later than one tick") | The brain needs ~12 ms of one core per 50 ms tick; games and other heavy programs push it to 40-45 ms. Close them for fly-brain runs. |
| `not flying: no simulator answering` / `could not connect` | Use `launch.cmd`, which waits for the sim; otherwise `stop_sim.cmd` and retry. Check `flypilot\logs\sim.log`. |
| The camera window stays on `waiting for tcp://127.0.0.1:5050` | The camera stream serves one client; don't run `camera.cmd` during `launch.cmd avoid`. |
| No dashboard in the 3D window during `avoid` | Check `flypilot\logs\dashboard.log`; `cannot listen on udp port 5300` means an old dashboard is still running, so run `stop_sim.cmd`. |
| No "Delivery stops" window | It opens with `avoid` and `deliver`, sometimes behind other windows or on another monitor: check the taskbar. Errors are in `flypilot\logs\dashboard.log`. |
| Clicks in the Stops window do nothing | It only accepts clicks once the flight is running (its panel then shows the instructions). The reason for a refused stop is shown under the buttons. |
| Double-clicking in the 3D view does nothing | Re-run `setup.cmd`: the picking is part of the CrazySim patch. Double-click (left button) the floor itself; a click on the drone selects the drone instead. |
| The people and the cart stand still in a street test | Re-run `setup.cmd` (moving them is part of the patch); `flypilot\logs\sim.log` should list `movers ['cart1', 'person1', 'person2', 'person3']`. |
| `cannot listen for orders on udp port 5310` | Another `avoid`/`deliver` flight is still running; close it, or run `stop_sim.cmd`. |
| The 3D window is frozen, or something is still running | `flypilot\stop_sim.cmd` |
| `QFontDatabase: Cannot find font directory` | Harmless (opencv-python ships Qt without fonts). |
| The wrong WSL distro is used | `set FLYPILOT_DISTRO=Ubuntu-24.04` (cmd) or `$env:FLYPILOT_DISTRO="Ubuntu-24.04"` (PowerShell) |

Logs: `flypilot\logs\` (sim, dashboard, camera, brain). The firmware console is
`~/CrazySim/crazyflie-firmware/sitl_make/build/0/out.log` inside WSL. More detail is in run.txt
section 5.

## Known limitations

- **Four sensor directions.** Each range sensor sees about 20 degrees around its axis (a three-ray
  fan), so anything approaching diagonally is seen late or not at all. The `crossing` and `cart`
  tests show it. The geometric dodge assumes a fairly small obstacle: around long unmapped ones (try
  the `none` map against a partition) it can take many tries, or hit the 20 s episode limit, which
  lands the drone.
- **Scripted movers.** The people and the cart follow their paths and don't react to the drone.
- **The fly brain's eye is ideal.** Its looming comes from the simulator's geometry, not from camera
  images, and it counts things the drone flies past closely as looming, which a fly's looming
  detectors would mostly ignore. That costs false alarms (see [../flybrain/README.md](../flybrain/README.md)).
  For a fair comparison, the other avoidance arms need the same 360-degree eye.
- **GPU required.** Without a GPU (software rendering) the sim falls about 10% behind real time and
  flights fail. A 1-core CPU flies but has no margin; 2 or more cores is clean.
- **Brief RTF dips.** On some runs, especially back-to-back ones, the firmware briefly reports ~94-96%
  of real time. Flights still pass; the cause is not yet found.
- **The planner's map only grows.** It starts from the scene file; after each avoidance episode the
  static things the compound eye saw nearby are added (orange dots), but nothing is ever removed, so
  something that was moved away stays on the map until the flight ends. Stops are visited in the
  order given, not reordered into a shorter tour. The marker numbers show through obstacles in the 3D view.
- **One camera client.** The camera bridge serves one client at a time. When a client connects, the
  bridge briefly takes over the cflib link; this is handled by connection ordering and retries.
- **Limited testing.** Tested on Windows 11 with WSL Ubuntu 24.04 (RTX 4080), and teammates report it
  runs on their PCs. Native Linux and Ubuntu 22.04 are untested.
