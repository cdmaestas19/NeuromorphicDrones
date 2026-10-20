# flypilot: Crazyflie flight stack in CrazySim

Milestone M3 of Fly Brain Pilot: a conventional flight stack, running in simulation, that the fly-brain
avoidance layer will attach to later. It runs the real Crazyflie firmware in
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
- **One swappable decision interface** (`AvoidancePolicy`), where the looming-detector and
  connectome policies will plug in.
- **Decision-cost measurement** and a record-and-replay benchmark, because the end target is a small
  drone with little computing power.

There is no spiking-neural-network code here yet. The fly circuit only ever decides *when* and *which
way* to dodge; the firmware's controller always keeps the drone stable.

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
flypilot\launch.cmd avoid --route headon     a person walks at the drone (also hover, crossing, cart)
```

Run these from a normal Windows terminal in the repo folder. The `.cmd` files run everything inside
WSL for you.

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

These tests run in the **street scene** (`scenes/street.xml`): the same 6 x 6 m room with a walkway
across it (y = 1.0 to 1.6), a road down it (x = 1.2 to 2.0), and a few static obstacles. Two people
and a cart are scripted in `routes.py` (`movers.py`). The simulator moves them, and the range sensors
and the camera see them like anything else. The cart stands in for a car at this room's scale: it is
0.5 m wide, 0.8 m long and 0.6 m tall. Movers are never on the planner's map: dealing with them is the
avoidance layer's job. `launch.cmd` starts the simulator on the scene that the route asks for.

Each encounter starts when the drone reaches a trigger point, so it happens at the same place on every
run. At the end of a run the terminal prints the closest approach to each mover (drone edge to mover
edge) and any contacts. This is the "minimum distance to people" measurement from the project plan.

Results with today's geometric policy:

| Route | What happens | Spotted at | Closest | Contact |
|---|---|---|---|---|
| `headon` | a person walks along the walkway at the drone (front sensor) | 1.4-1.5 m | 0.12-0.14 m | none |
| `hover` | a person walks up behind the hovering drone (back sensor) | 1.6-2.2 m | 0.15 m | none |
| `crossing` | a person crosses the drone's path from the side | 0.1-0.2 m | touch | yes |
| `cart` | the cart drives down the road across the drone's path | 0.2 m | touch | yes |

- **`hover` needs the range rate.** The drone isn't moving, so only how fast a range reading shrinks
  shows the person coming.
- **`crossing` and `cart` come in between the four sensor directions.** Nothing is seen until it is
  almost on top of the drone. A camera-based looming detector, and the fly brain, should do better
  here: that is the comparison these tests are for.

`launch.cmd deliver --scene street` gives live stops in the street, with the person walking the
walkway and the cart driving the road back and forth the whole time.

### Other commands

| Command | Does |
|---|---|
| `flypilot\launch.cmd deliver --at X,Y [--at X,Y ...] [--return] [--map all\|building\|none] [--scene street]` | stops given at launch (metres, the drone starts at 0,0), as described above. `--at` and `--map` also work with `avoid` |
| `flypilot\launch.cmd avoid --route headon\|hover\|crossing\|cart` | the moving-obstacle tests above |
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
                                      '-- fly_waypoints.py / avoid_demo.py --cflib UDP--> firmware
                                            avoid_demo.py <--double-clicks / markers--> crazysim.py
                                            avoid_demo.py --mover positions--> crazysim.py (people, cart)
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
  - **Geofence.** The setpoint never goes within 0.1 m of the geofence.
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
| `scenes/street.xml` | the street scene: walkway, road, two people and a cart |
| `avoidance.py` | `Sensors`, `AvoidCommand`, the `AvoidancePolicy` interface, the supervisor, `clear_out`, `CostMeter` |
| `policies.py` | `GeometricTTCPolicy`, and the `POLICIES` registry |
| `routes.py` | test scenarios (stops, map, scene, movers): `square`, `blocked`, `headon`, `hover`, `crossing`, `cart`; what moves in live flights |
| `frames.py` | `FrameSource`: AI-deck camera frames over CPX/TCP; `get_frame()` waits for a new frame, `latest()` returns the newest |
| `camera_view.py` | the separate camera window |
| `dashboard.py` | the avoidance dashboard (drawn into the 3D window; `--window` for its own window; `--stops` adds the Stops window) |
| `bench.py` | offline record-and-replay benchmark |
| `setup.*`, `system_deps.sh`, `env.sh`, `py.sh`, `run_sim.*`, `stop_sim.*` | setup and plumbing |
| `patches/` | our fixes to CrazySim, applied by setup (see below) |
| `run.txt` | test checklist for lab mates, with expected output |

---

## Adding a policy (this is where the fly brain goes)

A policy sees the sensors and the newest camera frame each tick. It returns an avoid signal and a
dodge. Nothing else in the stack needs to change:

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

The end target is a small drone, so each decision is timed. Measured on the geometric policy:

| | Per decision |
|---|---|
| replayed offline (`bench.cmd`) | ~3.6 us (~2.2 us before the range-rate estimate) |
| during a live flight (printed at the end of each `avoid` run, and shown on the dashboard) | ~20-25 us (includes Python waiting on the radio and camera threads) |
| budget: one 20 Hz control tick | 50 000 us |

By operation count, the geometric policy plus the supervisor is under 10 divisions and ~200 simple
operations per decision. Its state is the maneuver plus the last few range readings, about 300 bytes.
On the Crazyflie's STM32F405 that should take microseconds. This is an estimate, not a measurement on
hardware. Path planning (a few ms) runs once per leg, never per decision.

The heavy compute will come later, from camera preprocessing (cost grows with the number of pixels)
and from the spiking network itself (cost grows with neurons and synapses). Guidelines for new
decision code:
- write plain scalar code
- do no per-tick allocation
- downsample camera frames early
- check every change with `bench.cmd`, which must still report MATCH

Recordings do not include camera frames yet, so camera-based policies get `frame=None` in `bench.cmd`.

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
| `not flying: no simulator answering` / `could not connect` | Use `launch.cmd`, which waits for the sim; otherwise `stop_sim.cmd` and retry. Check `flypilot\logs\sim.log`. |
| The camera window stays on `waiting for tcp://127.0.0.1:5050` | The camera stream serves one client; don't run `camera.cmd` during `launch.cmd avoid`. |
| No dashboard in the 3D window during `avoid` | Check `flypilot\logs\dashboard.log`; `cannot listen on udp port 5300` means an old dashboard is still running, so run `stop_sim.cmd`. |
| No "Delivery stops" window | It opens with `avoid` and `deliver`, sometimes behind other windows or on another monitor: check the taskbar. Errors are in `flypilot\logs\dashboard.log`. |
| Clicks in the Stops window do nothing | It only accepts clicks once the flight is running (its panel then shows the instructions). The reason for a refused stop is shown under the buttons. |
| Double-clicking in the 3D view does nothing | Re-run `setup.cmd`: the picking is part of the CrazySim patch. Double-click (left button) the floor itself; a click on the drone selects the drone instead. |
| The people and the cart stand still in a street test | Re-run `setup.cmd` (moving them is part of the patch); `flypilot\logs\sim.log` should list `movers ['cart1', 'person1', 'person2']`. |
| `cannot listen for orders on udp port 5310` | Another `avoid`/`deliver` flight is still running; close it, or run `stop_sim.cmd`. |
| The 3D window is frozen, or something is still running | `flypilot\stop_sim.cmd` |
| `QFontDatabase: Cannot find font directory` | Harmless (opencv-python ships Qt without fonts). |
| The wrong WSL distro is used | `set FLYPILOT_DISTRO=Ubuntu-24.04` (cmd) or `$env:FLYPILOT_DISTRO="Ubuntu-24.04"` (PowerShell) |

Logs: `flypilot\logs\` (sim, dashboard, camera). The firmware console is
`~/CrazySim/crazyflie-firmware/sitl_make/build/0/out.log` inside WSL. More detail is in run.txt
section 5.

## Known limitations

- **Four sensor directions.** Each range sensor sees about 20 degrees around its axis (a three-ray
  fan), so anything approaching diagonally is seen late or not at all. The `crossing` and `cart`
  tests show it. The geometric dodge assumes a fairly small obstacle: around long unmapped ones (try
  the `none` map against a partition) it can take many tries, or hit the 20 s episode limit, which
  lands the drone.
- **Scripted movers.** The people and the cart follow their paths and don't react to the drone.
- **GPU required.** Without a GPU (software rendering) the sim falls about 10% behind real time and
  flights fail. A 1-core CPU flies but has no margin; 2 or more cores is clean.
- **Brief RTF dips.** On some runs, especially back-to-back ones, the firmware briefly reports ~94-96%
  of real time. Flights still pass; the cause is not yet found.
- **The planner's map is fixed.** It comes from the scene file, and the drone doesn't add what its
  range sensors see. Stops are visited in the order given, not reordered into a shorter tour. The
  marker numbers show through obstacles in the 3D view.
- **One camera client.** The camera bridge serves one client at a time. When a client connects, the
  bridge briefly takes over the cflib link; this is handled by connection ordering and retries.
- **Limited testing.** Tested on Windows 11 with WSL Ubuntu 24.04 (RTX 4080), and teammates report it
  runs on their PCs. Native Linux and Ubuntu 22.04 are untested.
