# Fly Brain Pilot — project context for Claude Code

## What this project is
CS 491L research project, UNM. Team: Adam Hart, Kevin Puebla, Christian Maestas.
Mentors: Dr. Wenbin Wan (robotic control), Felix Wang (Sandia, neuromorphic),
Dr. Melanie Moses (research process).

Goal: run spiking circuits taken from the Drosophila connectome — the looming
detectors (LC4, LPLC2, LC6) and the giant-fiber escape pathway (DNp01) — as an
obstacle-avoidance layer on a Crazyflie drone. A conventional flight controller
keeps the drone stable; the fly circuit only decides WHEN and WHICH WAY to dodge.

Research question: does real biological wiring give any advantage in reaction
speed, efficiency, or reliability over conventional approaches? Comparison arms:
fly connectome / CNN / shuffled connectome / abstract looming model / geometric
rule. A negative result is an acceptable result.

## Environment
- Windows 11 host, WSL2 Ubuntu 24.04 (distro name `Ubuntu`). VS Code runs on
  the Windows side; the user runs everything through the `.cmd` wrappers in
  `flypilot\`, which call into WSL. Teammates' PCs may differ.
- Flight stack Python: a pip venv at `~/.venvs/flybrain` (Python 3.12, from
  `flypilot/requirements.txt`, created by `flypilot/setup.sh`). Run Python in
  it with `flypilot/py.sh <script>`. No conda.
- GPUs: RTX 4080 on this PC, RTX 4070 Ti available elsewhere. UNM CARC
  available for batch work.
- The sim REQUIRES a GPU WSL can see. Without one it falls ~10% behind real
  time and flights fail, because the firmware runs on the real clock.
- numpy: the flight venv runs numpy 2.x (`numpy>=1.24,<3`; 2.5.4 installed).
  Brian2 and several connectome tools may assume the 1.x API, so give the SNN
  its own environment rather than downgrading the flight venv.
- Brian2 forks joblib workers; forking after PyTorch is loaded OOM-kills them
  and shows up as a confusing TerminatedWorkerError. Run the SNN in its own
  process.
- SNN Python: its own venv `~/.venvs/flybrain-snn` (Brian2, from
  `flybrain/requirements.txt`, created by `flybrain/setup_brain.sh`, which
  `flypilot/setup.sh` runs). Run Python in it with `flybrain/py.sh <script>`.
  Model + FlyWire data: `~/Drosophila_brain_model` (sazulai fork of Shiu et al.
  2024, pinned in `flypilot/env.sh`); cell-type labels: `~/flywire_annotations`.
- Whole-brain simulations take 3 GB (FlyWire 630) to 4.3 GB (783) each; running
  one per core (the model repo's `n_proc=-1`) exhausts memory. WSL gets half the
  host RAM by default.

## What already exists — do not rebuild it
`flypilot/` holds a working baseline flight stack (see flypilot/README.md):
- CrazySim pinned at commit 3ec8b55, MuJoCo backend, real Crazyflie firmware in
  software-in-the-loop. One command (flypilot\launch.cmd) brings up the 3D
  scene, the simulated AI-deck camera, and an avoidance dashboard.
- Flight stack over cflib/UDP: connect, take off, stream position setpoints at
  20 Hz, land. Safety stops on Ctrl-C, error, link loss, stale telemetry,
  geofence exit, firmware crash report.
- Sensors available to a policy each 20 Hz tick: position, yaw, velocity (world
  and body), route intended velocity, Multi-ranger front/back/left/right/up at
  10 Hz with 4 m max (each horizontal sensor a 3-ray level fan, +-10 deg, as
  patched), and the newest camera frame at 324x244 grayscale, 20 fps, over the
  same CPX protocol as a real AI-deck.
- Policy interface: `AvoidancePolicy.step(sensors, frame) -> AvoidCommand(
  signal 0..1, dodge velocity)`. Policies are selected by name; the flight code
  never depends on a specific one. ALL new avoidance work goes behind this
  interface.
- Supervisor state machine NOMINAL/AVOID. Enters AVOID at signal >= 0.50.
  Returns to route only after signal < 0.25 AND clearance > 0.5 m have held for
  0.4 s, and never sooner than 0.4 s after entering AVOID. (A single threshold
  flipped 97 times in 10 s on a noisy signal; the supervisor switched once.)
- First policy: geometric time-to-collision on the range sensors. TTC was chosen
  over a potential field because TTC is the quantity a fly's looming circuit
  estimates, so vision-based policies will produce comparable signals. Closing
  speed = own motion (intent-gated) + the obstacle's motion from the range rate
  (straight-line fit over 0.25 s, ignored while turning / moving across the
  ray / on jumps); recorded static flights replay with identical decisions.
- Flight-side safety around any policy (`avoid_demo.py`): speed capped so the
  TTC to mapped obstacles stays > 2.5 s (along the rays and the path corridor),
  a dodge slows rather than fly into what the eye sees (`_dodge_safe`), the
  clear-out backs a finished dodge out of the resume clearance (from the eye
  when there is one), setpoint kept 0.1 m inside the geofence (0.4 m below the
  ceiling), descents only while the eye sees room below (`eye.clear_below`:
  steep rays' hits within 0.6 m horizontally; walls beside it aren't below),
  landing with the avoidance on down to 0.3 m (`_land`). The setpoint leash
  (`flight.step_velocity`) is per axis: a lagging climb mustn't freeze a
  sideways escape. Contacts count height per part (`movers.Shape.parts`, one
  per geom: over the cart's front body isn't touching it, its higher cabin
  behind would be). The geofence is per scene:
  0.35 m inside the walls named wall_*, ceiling 0.5 m below their top, at most
  6 m (`planner.geofence`). Emergency landings (`flight.land`) count touchdown
  when the drone stops descending (it may sit on a crate), timeout by height.
- The compound eye is a sensor of the flight, not of one policy: `avoid_demo`
  owns the `EyeSource` and passes the reading in `Sensors.eye` to every policy.
  The resume gate's clearance comes from it (`eye.clearance`: nearest thing
  within +-100 deg of the heading). After each episode, static things the eye
  sees within 1.2 m that aren't mapped go on the planner's map
  (`planner.add_seen`; orange dots on the top views): the map now grows.
- Cruise height: `--cruise M` (routes' `cruise_m`; street deliveries cruise at
  4.5 m, `routes.DELIVERY_CRUISE_M`) flies between stops at M; vertical 1 m/s,
  0.4 m/s below 1.5 m; climbs brake before the ceiling (street: 8 m buildings,
  ceiling 6 m = `planner.CEILING_MAX_M`; CrazySim's room 2.5 m). Careful landing
  (`avoid_demo._come_down`): down to 2.5 m, the drop-off point is "taken" if the
  eye sees anything >= 0.25 m tall within 0.7 m that isn't a scene obstacle
  (`eye.points_near`, `orders.obstacle_gap`); wait, after 6 s come down at the
  nearest clear spot on rings 0.9-2.4 m; the descent (and the final landing)
  pauses while the spot is taken. `Sensors.route_stop_m` is an efference copy
  of where the route will stop.
- Windows: `launch.cmd ... --screen auto|main|N|off` (`screens.py`: XRandR
  monitors via ctypes, Windows' main screen via powershell.exe, windows moved
  with xdotool once as they appear; default auto = a non-main screen);
  `--headless` opens none (CrazySim's sitl_camera.sh drops `--vis` when
  CRAZYSIM_HEADLESS=1, our patch; no dashboard). Test suites run headless.
- Moving obstacles: `scenes/street.xml`, a 12 x 12 m courtyard (walkway y 2..4,
  road x 1.5..4.5, static trees/benches/kiosk/crates) with three people and a
  golf-cart-sized cart as mocap bodies, moved over UDP 5312 by `movers.py`
  scripts (people 1.0 m/s, cart 1.5 m/s; they go back and forth for the whole
  flight). Routes `headon`, `hover`, `crossing`, `cart`, `lane` (the cart
  reaches the drone mid-lane), `dropoff` (cruise at 4.5 m, a person walks into
  the drop-off point), `wallside` (unmapped crate3 by the south wall: the map's
  veto), `poles` (unmapped 6 m new_pole1 at cruise height), `landing` (the stop
  is where the cart is parked), and `deliver --scene street`. The 'building'
  map leaves out crate* and new_*. Each run logs the closest approach and contacts. Mocap bodies need an
  explicit `<inertial>`: without one (inertiafromgeom="false") a rotated mover is
  invisible to mj_multiRay, i.e. to the compound eye.
- The fly brain (`flybrain/`, see flybrain/README.md), policy `flybrain`:
  - Compound eye in the sim: 72 x 25 distance rays (5 deg, -90..+30 elevation,
    packet 'CSE2'), world-aligned, 20 Hz on UDP 5330; `flypilot/eye.py` turns it
    into looming (-d'/d) per eye (two halves, 15 deg frontal overlap) -> drive
    0..1 (LOOM_REF 5/s). Only collision courses loom (`eye.CollisionCourse`:
    objects = runs of similar depth, tracked; centroid velocity without its
    largest reading-to-reading step (a crate face coming into view is one jump);
    static below 0.5 m/s or when it moves like parallax of the drone's own
    motion; "moved" needs 3 readings of steady motion; counts if a point passes
    within 0.15 m, not beyond 0.3 m): LPLC2's selectivity.
    Ideal/geometric, not image-based.
  - `brain_server.py`: the escape subcircuit (paths <= 3 synapses from LC4/LPLC2
    to DNp01 + 35 other DNs, connections >= 5 synapses: 3,118 neurons) of the
    Shiu et al. LIF model, one long Brian2 run stepped every 50 ms over UDP 5320;
    integrates at 0.2 ms in flight (same GF response as 0.1 ms, half the
    compute: ~5 ms per tick; several times that on a busy PC). Non-blocking both
    ways: the flight uses the newest reply, the server skips to the newest step.
  - `FlyBrainPolicy`: signal = max giant-fiber rate / 100 Hz; left eye drive x
    1.35 (EYE_GAIN: the connectome's left pathway is less sensitive); side =
    away from the stronger steering DNs (DNp02/04/06/11), or, when they are
    weakly lateralized, away from the side the threat will pass (eye); forward/
    back from where the looming is, or for a moving threat straight out of its
    path (eye's tracked velocity), to the side of its line the drone will be on
    after 0.5 s of its own motion (mean over the 0.4 s settle window), at the
    speed that clears it before it arrives (0.4..0.8 m/s, `_escape_speed`);
    up at 1 m/s except beside a person; which way round: the planner's map
    (`Sensors.route_cost` from `planner.cost_field`, computed with each leg's
    plan) when the brain is unsure (or the brain's way is >= 1 m longer, or
    blocked on the map); `side_from` logged per episode; 1 s bursts back to back while
    alarmed, then hold; no braking on AVOID entry (`brake_on_avoid`).
  - Checks: `flypilot\brain_reproduce.cmd` (sugar -> MN9 vs the paper; looming
    -> GF in the whole brain) and `flypilot\brain_validate.cmd` (subcircuit vs
    whole brain: the 4-hop cut matches all 8 GF rates; the flown 3-hop cut is
    close and its GF curves match the 4-hop's).
  - Finding: the right eye's pathway drives the giant fibers at much weaker
    looming than the left's (in the data, not the cut); left as is.
- Every flight is stops on planned paths. `planner.py` (5 cm occupancy grid,
  0.3 m clearance, A*, re-plan after every dodge) plans on a map:
  `all` / `building` (no crates) / `none`. Whatever is not on the map is the
  avoidance layer's job. Test scenarios in `routes.py` are stops plus a map
  (`blocked` uses `building`, so crate3 is left to the avoidance).
- Stops arrive live (`flypilot\launch.cmd deliver`, `orders.py`): from the
  "Delivery stops" window (`stops_window.py`, run by `dashboard.py --stops`),
  from double-clicks in the 3D view, or from `--at X,Y`. Requests go to UDP
  5310; the patched sim draws markers sent to UDP 5311, viewer-only, so
  sensors never see them.
- Record-and-replay benchmark: replays a recorded flight through any policy
  offline and asserts the decisions are identical. `--record X.jsonl` also
  saves X.frames.npz (half-resolution camera frames) and X.eye.npz (compound-eye
  readings, replayed by bench.py), and each decision line carries the policy's
  info, the movers' positions and route_stop_m.
- Policies: `geometric` (default, four Multi-ranger rays decide when; the eye
  and camera only say what a threat is, the map which side), `geometric360`
  (EXPERIMENTAL 360-degree version: false movers on oblique surfaces, its
  maneuver re-triggers in clutter), `flybrain`.
- Decision timing (Python): the geometric rule alone ~3.6 us replayed; with the
  eye's object tracking ~0.57 ms; camera network ~0.2 ms per look; live ~1.5 ms
  (geometric) and ~1.6 ms (fly brain, without the brain's own 5-30 ms in its
  process), against a 50 ms per-tick budget.

Two local patches to CrazySim: a firmware bug where a 32-bit variable was used
for 64-bit timer ticks (froze the simulated firmware), and a sim patch: real
time under WSL, viewer overlays/picking/markers, movers, range-sensor fan,
compound eye (straight down to +30 deg, with a grazing-ray filter). The first
two fixes should go upstream.

## Known limitations we are deliberately working on
- Nothing tested on hardware.
- The fly brain's eye is ideal: its collision-course selectivity is LPLC2's
  function computed geometrically, not its mechanism. Next steps: a motion-based
  (T4/T5 -> LPLC2-like) front end, and retinotopic input to LC4/LPLC2 so the
  brain itself sees front vs back (today the eye supplies front/back and the
  side for near-frontal threats).
- The other comparison arms need the same 360-degree eye for a fair comparison;
  the geometric policy still uses the four range sensors (geometric360 isn't
  ready).
- Car vs person (user: both policies; eye now, camera next): the eye's size
  (`eye.CollisionCourse`: moving and > 0.75 m wide = vehicle, < = person,
  > 3 m = unknown, decides nothing) and a camera CNN (`camera_kind.py`, trained
  by `camera_train.cmd` / `vision_train.py` in its own venv
  `~/.venvs/flybrain-vision`, auto-labelled from recorded flights; weights
  `models/camera_kind.npz`, run in numpy). Rule (`policies._ThreatKinds`):
  person -> round, never over; vehicle -> up over it (eye readings counted:
  vehicle only at 2:1 over person; camera counts after 3 sure readings); and
  no climb with a recently-moving person within 1.5 m (`PERSON_CLEAR_M`,
  `eye.Looming.nearest_person`). The camera network is
  trained on the sim's plain shapes: it needs real images for hardware.
- Reaction time is bounded by sensor/loop rates (10 Hz ranges, 20 Hz control =
  up to ~150 ms), not by compute.
- The planner's map is the scene file plus what the avoidance met (static only;
  nothing ever expires from it). Stops are flown in the order given.
- Movers follow scripts; they don't react to the drone.
- Intermittent firmware "stabilizer loop rate is off" warnings (sim RTF dips),
  some runs 20-40, most 0; flights pass. Cause not found.

## Conventions
- Validate scripts before delivering: `bash -n` for shell, `python -m py_compile`
  for Python.
- Prefer calling CrazySim's own entrypoints over reimplementing them.
- Anything the user starts or stops is a Windows `.cmd` in `flypilot\`; keep
  WSL details inside the scripts.
- Every flight and every new test uses the path planner. A new scenario goes
  in `routes.py` as stops plus a map; any obstacle the test should exercise
  the avoidance on is left off the map.
- Inspect the actual repo before writing code that depends on its paths; if
  reality contradicts this document, trust the repo and tell me what changed.
- MaleCNS data is CC-BY 4.0: attribution required in every figure and report.
  Same for the FlyWire connectome (Dorkenwald et al. 2024) and annotations
  (Schlegel et al. 2024) used by flybrain/; the model is Shiu et al. 2024 (MIT).
