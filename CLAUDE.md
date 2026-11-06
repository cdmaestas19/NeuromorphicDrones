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
  TTC to mapped obstacles stays > 2.5 s (along the rays and the path corridor;
  `planner.distance_ahead(z=)`: the mapped obstacles at the drone's actual
  height, any height: a drone left high by an escape flew into a 1.5 m beam
  the 0.5 m plan didn't have),
  a dodge slows rather than fly into what the eye sees (`_dodge_safe`), the
  clear-out backs a finished dodge out of the resume clearance (from the eye
  when there is one), setpoint kept 0.1 m inside the geofence (0.4 m below the
  ceiling), descents only while the eye sees room below (`eye.clear_below`:
  steep rays' hits within 0.6 m horizontally; walls beside it aren't below;
  hits on what the map has count only within 0.2 m, `planner.mapped_at`, not
  where a mover was lately: a crate 0.54 m beside home held a landing 30 s, and
  in the obstacles room an escape's climb to 2 m was never undone),
  landing with the avoidance on down to 0.3 m (`_land`), turns to face a leg
  with the avoidance on (`_turn_to`; `pilot.turn_to` turned blind: the cart hit
  a drone turning in its lane, ac1_cart), a stall watchdog (`STALL_S` 2 s
  without a decision: faulthandler dumps every thread's stack to the log;
  the firmware cuts the motors ~2 s after setpoints stop). The setpoint leash
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
  within +-100 deg of the heading). The map grows from what the avoidance
  meets (`planner.add_seen`; orange dots on the top views), as whole objects:
  `eye.footprint` completes the side the eye sees (circle fit, if it bulges
  toward the eye mid-chord (a face + one corner point bulges at its end) and
  spans >= 70% of the front such a circle shows -> the whole
  disc for crowns/poles/people standing; two faces of a box <= 2 m -> mirrored
  across their chord; one flat face -> 0.3-0.6 m deep, straight back along its
  normal, not along the view rays: a slanted view overshot the far side). An
  object's points on scene obstacles already mapped (`planner.mapped_at`, at
  the height each ring point was hit, `SeenObject.z`) are dropped first: a
  wall + the cart's side made one "corner" mirrored across the courtyard, and
  a beam seen from 1.15-1.5 m went on the 0.5 m map (both: no path home).
  `add_seen` takes a point if its 5 cm cell has none yet (not: if it is outside
  the clearance round the others: a crate's far face 0.1 m beyond the guess
  was never added, the path home passed it at 0.2 m, 9 dodges).
  During an episode, static things on a collision course go on the map
  (`_map_threats`); after it, static things within 2.5 m; while flying on,
  more of what is already on the map as it comes into view (`_grow_map`,
  every 0.5 s); each growth bumps `orders.map_version` -> re-plan. The eye's
  tracker ignores the motion of a not-yet-moved object while its width
  changes (`EMERGE_M`: a crown rising into view during the climb had been
  tagged a moving vehicle, never mapped, 9-10 dodges at tree1). Only before
  it ever moved, and at 0.5 m: applied to people, or at 0.25 m (ray spacing
  makes a walker 0 -> 0.3 m wide), it hid one walking up behind a hovering
  drone (contacts in headon, hover). Nothing that moves now, nor anything
  within 0.5 m of where a mover was in the last 20 s
  (`CollisionCourse.near_movers`: a paused cart re-found as a new track),
  goes on the map. Climbs brake below mapped obstacles overhead within 0.35 m
  sideways (`planner.room_above`; an escape slid under the obstacles room's
  1.5 m beam, hit it, fell and bounced through the geofence).
- Cruise height: `--cruise M` (routes' `cruise_m`; street deliveries cruise at
  4.5 m, `routes.DELIVERY_CRUISE_M`) flies between stops at M; vertical 1 m/s,
  0.4 m/s below 1.5 m; climbs brake before the ceiling (street: 8 m buildings,
  ceiling 6 m = `planner.CEILING_MAX_M`; CrazySim's room 2.5 m). Careful landing
  (`avoid_demo._come_down`): down to 2.5 m, the drop-off point is "taken" if the
  eye sees anything >= 0.25 m tall within 0.7 m that isn't a scene obstacle
  (`eye.points_near`, `orders.obstacle_gap`); wait, after 6 s come down at the
  nearest clear spot on rings 0.9-2.4 m; the descent (and the final landing)
  pauses while the spot is taken. `Sensors.route_stop_m` is an efference copy
  of how far the route goes on straight ahead: to the stop or to the first
  turn > 30 deg (`avoid_demo._straight_ahead`; ending only at the stop, a wall
  past a corner waypoint loomed and the fly brain looped for 10 min).
- Windows: `launch.cmd ... --screen auto|main|N|off` (`screens.py`: XRandR
  monitors via ctypes, Windows' main screen via powershell.exe, windows moved
  with xdotool once as they appear; default auto = a non-main screen);
  `--headless` opens none (CrazySim's sitl_camera.sh drops `--vis` when
  CRAZYSIM_HEADLESS=1, our patch; no dashboard). Test suites run headless.
  `--chase` (CRAZYSIM_CHASE=1: the viewer follows drone 0 from 1 m behind,
  22 deg above, heading smoothed) and `--video FILE` (CRAZYSIM_VIDEO). Live
  offscreen recording got ~10 fps (WSL2's GL translation serializes all
  contexts; readback ~40 ms), so the sim only logs per 30 fps slot (qpos,
  mocap, prop quats, mat_rgba, viewer cam/opt, markers, RTF; FPV + dashboard
  panel as ffmpeg side streams in FILE.parts/) and on exit (SIGTERM handled)
  spawns `crazysim.py --render-video DIR`, which draws every frame with
  MuJoCo's own calls (`_FrameDrawer`: text overlay BEFORE drawPixels, else
  images come out white) to H.264 (imageio-ffmpeg); launch.py waits with
  progress; stop_sim.sh spares the renderer. Propellers spin visually at
  1/40 real speed (`PROP_VISUAL_SPEED`, massless bodies' body_quat).
  `flypilot\showcase.cmd` = showcase, flybrain-rf, chase, video to `videos/`.
- Moving obstacles: `scenes/street.xml`, a 12 x 12 m courtyard (walkway y 2..4,
  road x 1.5..4.5, static trees/benches/kiosk/crates) with three people and a
  golf-cart-sized cart as mocap bodies, moved over UDP 5312 by `movers.py`
  scripts (people 1.0 m/s, cart 1.5 m/s; they go back and forth for the whole
  flight). Routes `headon`, `hover`, `crossing`, `cart`, `lane` (the cart
  reaches the drone mid-lane), `dropoff` (cruise at 4.5 m, a person walks into
  the drop-off point), `wallside` (unmapped crate3 by the south wall: the map's
  veto), `poles` (unmapped 6 m new_pole1 at cruise height), `landing` (the stop
  is where the cart is parked), `showcase` (map 'notrees': both tree crowns
  at cruise height, then it lands at the last stop, `Route.home=False`,
  beside the walkway as a person passes), and `deliver --scene street`. The
  'building' map leaves out crate* and new_*; 'notrees' tree* too. Each run logs the closest approach and contacts. Mocap bodies need an
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
    motion, or when >= half its points lie on the scene's mapped obstacles
    (`Sensors.mapped` = `planner.mapped_at(seen=False)`; walls/beam/partition
    slid along as their visible stretch changed and counted as movers: 11
    dodges of a wall; not what the drone mapped itself, which may be a person
    who stood still; bench.py replays without it);
    "moved" needs 3 readings of steady motion; counts if a point passes
    within 0.15 m, not beyond 0.3 m): LPLC2's selectivity.
    Ideal/geometric, not image-based.
  - `brain_server.py --circuit active|h3|h4` (default `active`; `launch.cmd
    --brain-circuit`): a circuit of the Shiu et al. LIF model, stepped every
    50 ms over UDP 5320; integrates at 0.2 ms in flight (same GF response as
    0.1 ms). `active` (2026-10-26, mentor feedback: "run the whole brain on many
    tests, keep what spikes"): `flybrain/prune.py` ran the whole brain on 120
    looming tests x 2 seeds (spheres from every 30 deg, 2 sizes/speeds; 12
    recorded flights, both input forms): 1,654 neurons spiked -> 1,656 with
    all inputs/outputs, 131k connections (`flybrain/active_783.txt` in git,
    `circuit.active_circuit()`). Lossless on training tests (a neuron acts only
    by spiking); on 44 held-out tests x 2 seeds identical to the whole brain in
    82/88 runs, 0 alarms missed/extra; the old 3-hop cut `h3` (paths <= 3
    synapses LC4/LPLC2 -> DNp01 + 35 DNs, >= 5 synapses, 3,118 neurons: 2,181
    of them never fired, 717 active neurons outside it) 4/88, 6 missed + 6
    extra; `h4` (18,326) 4/88, 2 + 1. Flown: all routes clean, 1.0-1.8 ms per
    tick. Readout per circuit (`models/brain_readout_<circuit>.npz`; the reset
    reply names the circuit, `FlyBrainRFPolicy` loads its readout). Engine `fast` (default):
    `fastlif.c` via ctypes, the same network spike for spike (`check_fast.py`;
    Brian2 drops input to refractory "(unless refractory)" variables, the C
    does too; tiny values flushed to 0, subnormals cost 10x), ~2 ms per tick in
    flight vs Brian2's ~17 ms, 170 MB vs 2.5 GB; the cut circuit is cached
    (`circuit.escape_circuit`, `flybrain/cache/`, built by setup_brain.sh).
    `--engine brian2` is the reference; `--hops 4` now fits (~30 ms). Non-blocking both
    ways: the flight uses the newest reply, the server skips to the newest step.
    The whole brain runs in the same C kernel (`circuit.whole_circuit`, mentor
    feedback "fast CPU, not GPU"): matches the model repo's Brian2 runs (99-100%
    of firing neurons, r 0.999, all 8 GF rates; `prune.py check`); from 10k
    neurons `lif_run_sparse` steps only neurons not at rest (bit-identical,
    3.1x): ~0.9 s per simulated s on one core, ~1 GB (Brian2: 4.3 GB), 240
    whole-brain runs in 5 min; peaks ~95 ms per 50 ms tick, so not in flight
    yet. Brian2GeNN (other feedback) not used: GeNN is batch-oriented and WSL
    has no CUDA toolkit; the CPU made it unnecessary.
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
  maneuver re-triggers in clutter), `flybrain`, `looming` (comparison arm:
  the abstract looming model, `LoomingModelPolicy` = FlyBrainPolicy with the
  decider swapped, `_connect` / `_think`: alarm linear in the summed looming
  drive, 50 Hz at 0.13 = the brain's head-on alarm point; same eye, escapes,
  delay and smoothing), `flybrain-rf` (`FlyBrainRFPolicy`: the eye sends
  looming per ommatidium, `eye.Looming.loom_points`; the server drives each
  LC4/LPLC2 by its own receptive field, `brain_server.Inputs` with
  `RF_STRETCH` (no blind spot behind), `RF_PROFILE` (equal warning from every
  side, per eye) and `RF_GAIN`; a trained linear readout of all 37 outputs ->
  bearing, `flybrain/train_readout.py` -> `models/brain_readout_<circuit>.npz`; side and
  front/back of a standing threat's escape from it; movers still the eye's),
  `flybrain-cam` (EXPERIMENTAL `FlyBrainCamPolicy`: no eye; AI-deck camera ->
  `camera_loom.py` (edge-bounded regions tracked, ln-size fit = looming, grows
  around itself; `gate()`: the drone's own approach to what is beyond the stop,
  on the map or the floor dropped) -> the same brain and readout; the
  geometric rule on the Multi-ranger beside/behind, its front ray a safety net
  at half weight).
- Decision timing (Python): the geometric rule alone ~3.6 us replayed; with the
  eye's object tracking ~0.57 ms; camera network ~0.2 ms per look; live ~1.5 ms
  (geometric) and ~1.6 ms (fly brain, plus the brain's own ~2 ms in its
  process), against a 50 ms per-tick budget. A fly-brain flight's CPU (one
  desktop core = 100%): sim ~66%, simulated firmware ~50%, brain ~4%, flight ~5%.

Two local patches to CrazySim: a firmware bug where a 32-bit variable was used
for 64-bit timer ticks (froze the simulated firmware), and a sim patch: real
time under WSL, viewer overlays/picking/markers, movers, range-sensor fan,
compound eye (straight down to +30 deg, with a grazing-ray filter). The first
two fixes should go upstream.

## Known limitations we are deliberately working on
- Nothing tested on hardware.
- The fly brain's eye is ideal: its collision-course selectivity is LPLC2's
  function computed geometrically, not its mechanism. Next steps: a camera
  front end that works (see camera bullet below). Retinotopic input: done
  (`flybrain-rf`); a mover's heading is still the eye's (next: from the
  readout's bearing over time).
- Retinotopy (2026-10-25): `flybrain/retinotopy.py` derives each LC4/LPLC2
  receptive field from the connectome (Mi1 cell-body sheet -> harmonic
  extension over 33 columnar types -> each detector's inputs; oriented by
  FlyWire y/z, signed by VS1..8 order, HS order as check).
  `flybrain/directions.py` (open loop): per-neuron fields let a linear readout
  of the DNs get direction to 3 deg, but a shuffled map nearly matches it:
  own patches matter, the specific map barely. Flown as `flybrain-rf`: no
  contacts on 12 routes, closest approaches as good or better than
  `flybrain`'s; readout 6 deg median, L/R 100%. The fly's raw map touched
  people from behind/side (blind spot, frontal bias): hence RF_STRETCH and
  RF_PROFILE. User's slide goal: camera -> SNN -> small trained readout ->
  controller, range sensors -> controller; the slide says MaleCNS, the code
  uses FlyWire (the user should fix the slide).
- Camera looming is the weak link: dense-flow and four-arm LPLC2 versions
  failed at half resolution (a person 2 m off moves ~0.5 px/frame, under the
  drone's own image motion; plain sim surfaces carry no motion inside them).
  The tracked-region version catches ~10 of 24 eye alarms in view, ~3 extra
  alarms per flight. Flown: static routes fine, contacts on 6 of 8 mover
  routes: the real Crazyflie's sensors (camera ahead + 4 thin rays) don't see
  a cart from the side or a person overtaking at an angle (cb_headon), as the
  geometric rule on the same rays didn't. Coverage, not the brain, limits it.
- The other comparison arms need the same 360-degree eye for a fair comparison;
  the geometric policy still uses the four range sensors (geometric360 isn't
  ready). Done: the abstract looming model (`--policy looming`). Next: the
  shuffled connectome (the C engine makes it cheap), then a CNN arm.
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
