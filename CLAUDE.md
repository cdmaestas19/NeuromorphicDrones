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
  TTC to mapped obstacles stays > 2.5 s, `avoidance.clear_out` backs a finished
  dodge out of the resume clearance, setpoint kept 0.1 m inside the geofence.
- Moving obstacles: `scenes/street.xml` (walkway, road, two people, a cart as
  mocap bodies), moved over UDP 5312 by `movers.py` scripts; routes `headon`,
  `hover`, `crossing`, `cart`, and `deliver --scene street`. Each run logs the
  closest approach and contacts. The geometric policy avoids headon and hover
  and is touched in crossing and cart (they come between its four sensor
  directions): the comparison case for camera/fly-brain policies.
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
  offline and asserts the decisions are identical.
- Decision timing: ~3.6 us replayed offline, ~20-25 us live in Python, against
  a 50 ms per-tick budget.

Two local patches to CrazySim: a firmware bug where a 32-bit variable was used
for 64-bit timer ticks (froze the simulated firmware), and a sim patch: real
time under WSL, viewer overlays/picking/markers, movers, range-sensor fan. The
first two fixes should go upstream.

## Known limitations we are deliberately working on
- No spiking-network code exists yet. Nothing tested on hardware.
- Recordings do not include camera frames, so the replay benchmark cannot
  evaluate camera-based policies.
- Reaction time is bounded by sensor/loop rates (10 Hz ranges, 20 Hz control =
  up to ~150 ms), not by compute.
- The planner's map is fixed (from the scene file); range-sensor hits are not
  added to it. Stops are flown in the order given.
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
