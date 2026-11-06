# flybrain: the fruit fly's escape circuit, flying the drone

This folder runs the escape circuit of the adult fruit-fly connectome as a spiking network, and lets it
decide when and which way the drone dodges. The flight stack is in [../flypilot](../flypilot/README.md);
the brain plugs into it as one avoidance policy (`--policy flybrain`) next to the geometric baseline.

```
compound eye (sim) --> looming per eye --> LC4 + LPLC2 looming detectors (Poisson input, per eye)
     360 degrees           eye.py                      |
                                escape circuit of FlyWire 783 (1,656 LIF neurons, pruned by activity)
                                                       |
                         giant fibers DNp01 (when)  +  steering DNs DNp02/04/06/11 (which side)
                                                       |
                         FlyBrainPolicy (flypilot/policies.py) --> supervisor --> dodge
```

- **The model** is the whole-brain leaky integrate-and-fire model of Shiu et al. 2024 (Nature
  634:210), unchanged: the same equations and parameters (`model.py` of
  [Drosophila_brain_model](https://github.com/sazulai/Drosophila_brain_model)). Each connection's
  weight is its signed synapse count from the FlyWire connectome.
- **The circuit** is the part of the whole brain that fires when it sees looming: the whole brain was
  run on 240 looming tests and every neuron that spiked was kept (check 3 below; 1,656 neurons). Until
  2026-10-26 it was cut out by wiring instead: every neuron on a path of at most 3 synapses from the
  looming detectors (LC4, LPLC2) to the escape outputs, through connections of at least 5 synapses
  (3,118 neurons; `brain_server.py --circuit h3`). There are 37 outputs: the two giant fibers (DNp01, the fly's escape command) and the 35
  other descending neurons that get at least 20 synapses straight from the looming detectors.
- **The eye** is an ideal one: the simulator measures the distance to the scenery every 5 degrees all
  the way round, from straight down to 30 degrees up. Looming is how fast that distance shrinks
  relative to itself (1 / time to contact), counted only for things on a collision course with the
  drone, as the fly's LPLC2 neurons respond only to an image that expands without sliding. Each eye
  covers its half of the world, with 15 degrees of overlap in front, as in flies.
- **The brain runs in its own process** (`brain_server.py`), in step with the flight's 20 Hz loop,
  with its own Python environment. Brian2 and PyTorch don't mix in one process, and the flight
  stack's numpy is 2.x.

There are three versions of the brain's input and readout, each a policy of its own:

| Policy | Input to LC4 / LPLC2 | Readout | Senses |
|---|---|---|---|
| `flybrain` | two channels: every neuron of an eye at that eye's looming | giant fibers (when), steering DNs (side) | the 360-degree eye |
| `flybrain-rf` | each neuron its own receptive field, from the connectome (`retinotopy.py`) | giant fibers (when), a **trained readout** of all 37 outputs (where the threat is: side and front/back) | the 360-degree eye |
| `flybrain-cam` (experimental) | the same receptive fields, on looming from the **AI-deck camera** (`flypilot/camera_loom.py`) | the same | camera ahead; Multi-ranger beside, behind, and as a safety net ahead |

```
camera (or 360 eye) --> looming per direction --> each LC4/LPLC2 its own receptive field --> escape circuit
                                                                                               |
                         controller <-- escape away from it <-- trained readout: where <-- 37 outputs
                                                            giant fibers: when <------------'
```

## Running it

`flypilot\setup.cmd` sets everything up, the brain included. It fetches the model and its connectome
data (~200 MB) and the FlyWire cell-type labels, and creates the brain's Python environment
(`~/.venvs/flybrain-snn`, Brian2). Then:

```
flypilot\launch.cmd avoid --route hover --policy flybrain     a person walks up behind the hovering drone
flypilot\launch.cmd avoid --route headon --policy flybrain    (also crossing, cart, dropoff, square, blocked)
flypilot\launch.cmd deliver --scene street --policy flybrain  live stops among people and a cart; it cruises
                                                              at 4.5 m and comes down carefully at each stop
flypilot\launch.cmd avoid --route showcase --policy flybrain-rf   receptive fields and the trained readout
flypilot\launch.cmd avoid --route blocked --policy flybrain-cam   on the camera (experimental)
flypilot\brain_reproduce.cmd                                  check the model against published results
flypilot\brain_validate.cmd                                   check the cut-out circuit against the whole brain
```

`launch.cmd` starts the brain server for any of the three (its log is `flypilot\logs\brain.log`).
After changing the receptive fields or the brain's input, retrain the readout:
`bash py.sh train_readout.py` (~5 min; `--calibrate` and `--equalize` redo the gains, see below).
The dashboard's pipeline shows the compound eye's looming per eye and the giant fibers' rates. At the
end of each flight the terminal prints what the eyes and the giant fibers did, the number of escape
bursts, and the brain's compute time per tick.

## Checks before the drone relies on it

**1. The model reproduces its paper on this PC** (`flypilot\brain_reproduce.cmd`).

| Check | Result here | Reference | |
|---|---|---|---|
| sugar-sensing neurons -> MN9 (FlyWire 630), 10 trials | 84.3 +- 3.4 Hz | paper: 93.3 +- 3.2 Hz; the model repo's re-run with current software: 83.5 +- 4.7 Hz | within 10% of both: PASS |
| LC4, left eye (54 neurons at 150 Hz) -> giant fibers (FlyWire 783) | left 113.6 Hz, right 48.7 Hz | the same side's giant fiber fires most | PASS |
| LC4, right eye (50 neurons) | left 29.9 Hz, right 119.5 Hz | | PASS |
| LPLC2, left eye (108 neurons) | left 132.6 Hz, right 106.5 Hz | | PASS |
| LPLC2, right eye (102 neurons) | left 100.3 Hz, right 159.4 Hz | | PASS |

The looming detectors' strongest downstream targets include the steering descending neurons the
policy reads (DNp02, DNp04, DNp11), each on its own side.

**2. The cut-out circuit behaves like the whole brain** (`flypilot\brain_validate.cmd`). For each looming experiment
it runs the circuit and compares every output neuron's rate with the whole brain's. A rate matches
when the difference is within 10%, or within 3 standard errors of the trial-to-trial noise, whichever
is larger.

| Circuit (FlyWire 783) | Neurons | Giant-fiber rates matched (of 8) | Speed |
|---|---|---|---|
| 2 synapses deep, connections >= 5 synapses | 566 | no: misses inhibition; the opposite side's giant fiber fires up to 2.5x too much | |
| 3 deep, >= 5 (flown until 2026-10-26) | 3,118 | close; one opposite-side rate off | **Brian2: ~5 ms per 50 ms tick idle, 17 ms on average in flight. C engine: 1.6-4 ms, 2.0 ms on average in flight** |
| 4 deep, >= 10 | 4,854 | close; one opposite-side rate off | |
| 4 deep, >= 5 | 18,326 | all 8 | Brian2: 33-69 ms per tick next to the simulator, too slow to fly. C engine: 9 ms quiet, ~30 ms with looming |

Until the pruned circuit (check 3), the drone flew the 3-deep circuit. It was the largest one that kept
up with real time next to the simulator in Brian2, and its giant fibers respond to looming practically as the fully validated 4-deep
circuit's do (table below), so the cut costs little where it matters. With the C engine the 4-deep
circuit fits in the 50 ms tick too (`brain_server.py --hops 4`), with less room to spare on a laptop.

**3. Pruned by activity: run the whole brain, keep what fires** (`flypilot\brain_prune.cmd`, `prune.py`;
suggested in review). In this model a neuron acts on others only when it spikes: no background
activity, nothing below threshold reaches another neuron. So a neuron that never spikes during a test can
be removed without changing that test at all; the only question is whether the tests cover what the
drone meets. `prune.py` runs the whole brain (in C, below) on 120 training tests, 2 seeds each: spheres
approaching from every 30 degrees, two elevations, two sizes and speeds, and 12 recorded flights replayed
through the eye, with both kinds of input. **1,654 neurons spiked; with every input and output, the
pruned circuit has 1,656 neurons and 131k connections** (`active_783.txt`, in git; setup builds the
circuit from it). Of the flown 3-hop cut's 3,118 neurons, 2,181 never fired, and 717 neurons outside it
did (among them the descending neurons DNge119 and DNge053). On 6 training tests the pruned circuit's
every neuron spikes exactly as in the whole brain, as it must. Then 44 **held-out tests** it was never
chosen on (spheres from the directions in between, larger and faster ones than any it trained on,
12 other flights), every circuit against the whole brain on the same input spikes:

| Circuit | Neurons | Held-out runs identical to the whole brain | Alarms missed / extra | Outputs off (90th percentile) |
|---|---|---|---|---|
| **pruned by activity** | **1,656** | **82 of 88** | **0 / 0** | **0%** |
| 3 deep (flown until now) | 3,118 | 4 of 88 | 6 / 6 | 16-41% |
| 4 deep | 18,326 | 4 of 88 | 2 / 1 | 12-21% |

It is half the size of the 3-deep cut and more faithful than a cut eleven times its size. Spike for
spike it runs the same in C as in Brian2 (`check_fast.py --circuit active`), at 2.1 ms per 50 ms tick
against the 3-deep cut's 3.8 (1.0-1.8 ms in flight). Its giant fibers alarm as early from every side as
the 3-deep cut's with the same input gains (within 0.1 s), and its trained readout is as good (7 degrees
median bearing error, 98% left/right, on the same 80 flights). `brain_server.py --circuit active|h3|h4`
picks the circuit; each has its own readout (`flypilot/models/brain_readout_<circuit>.npz`), and the
policy loads the one the server names.

**The C engine** (`fastlif.c`, the flight's default; `brain_server.py --engine brian2` runs Brian2).
Brian2 steps the network from Python, and at 250 steps per 50 ms tick its per-step overhead, not the
arithmetic, is most of the cost. `fastlif.c` is the same model in ~60 lines of C: the same equations
(solved exactly per step), parameters (read from the model repo's `model.py`), order of events within a
step, and the same rule that input reaching a refractory neuron is dropped (Brian2 adds that guard
itself; without it the network spiked ~5% more). `check_fast.py` checks it: given the same input
spikes for 1 s, every one of the 3,118 neurons spikes exactly as often as in Brian2 (18,170 spikes in
both), and the giant fibers' looming response agrees at every drive level. In flight the brain's
compute fell from 17 ms to 2 ms per tick, its CPU from 40% to 4% of a core, and its memory from 2.5 GB
to 170 MB: the server loads the cut circuit from `cache/` instead of the whole connectome
(`circuit.escape_circuit`; setup builds it). The kernel needs ~4 MB of arrays, and a few million simple
operations per tick: within reach of the drone's own AI-deck (GAP8: 8 MB of RAM, 8 cores, fixed-point
arithmetic), which would be a port, not a recompile.
Its giant-fiber response to looming is practically the same as the fully validated 4-deep circuit's
(table below), so the cut costs little where it matters.

**The whole brain on a CPU core** (suggested in review: a fast CPU rather than a GPU). The same kernel
runs all 138,639 neurons and 15.1M connections (`circuit.whole_circuit`). Against the model repo's own
Brian2 runs of the four looming experiments (10 trials each), 99.3-100% of the neurons that fire match in
rate (correlation 0.999) and all 8 giant-fiber rates match (`prune.py check`). For a big network the
kernel steps only the neurons not at rest (`lif_run_sparse`: a neuron at rest is a fixed point of the
update; spikes are taken in index order, so the result is bit for bit the dense loop's): 3.1x faster on
the whole brain, ~0.9 s per simulated second on one core during a looming test (a GPU has little to do
when ~1% of the neurons are busy). Peaks reach ~95 ms per 50 ms tick, so the whole brain could not yet fly
next to the simulator; it can run hundreds of tests: 240 whole-brain runs took 5 minutes, ~1 GB each where
a Brian2 run takes 4.3 GB. Brian2GeNN (Brian2 on a GPU through GeNN) was the other suggestion; GeNN is
built for single long runs, not a step every 50 ms, and it would need the CUDA toolkit installed in WSL.
With the CPU numbers above it isn't needed.

In flight the brain integrates in 0.2 ms steps instead of the model's 0.1 ms (`brain_server.py
--dt-ms`). The model's equations are linear and integrated exactly, so the step mostly sets how finely
spike times are resolved. Measured over the table below, the giant fibers' rates at 0.2 ms match those
at 0.1 ms within trial-to-trial noise, at half the compute (5.4 instead of 10.9 ms per 50 ms tick).
The checks (`brain_validate.cmd`) use the model's 0.1 ms.

## How the brain decides

Each 50 ms tick:
1. **Eye.** `eye.Looming` turns two successive compound-eye readings into looming per ommatidium.
   `eye.CollisionCourse` keeps only what is on a collision course. It groups the directions into objects
   (neighbouring directions at similar distances) and tracks each one's motion; people and carts are
   fast enough to measure, and anything slower than 0.5 m/s counts as standing still, as does anything
   mostly on the scene's map (a wall's visible stretch slides as the drone passes, which looked like
   motion). An object counts
   if, on the current relative motion, any of its points will pass within 0.15 m of the drone, and not
   at all beyond 0.3 m (the planner's clearance). An approaching person counts, and so does a crate the
   drone flies at, but a crate it flies past does not. Walls and anything else on the scene's map never
   count: the planner keeps its paths clear of them (a wall beside a planned path once alarmed the brain,
   and the escape went into a tree). Each eye's drive is its most looming 30-degree
   field (90th percentile) / 5 per second, capped at 1.
2. **Brain.** The drive (the left eye's x 1.35, see below) sets the Poisson rate of that eye's LC4 and
   LPLC2 neurons (drive x 150 Hz, the model's activation rate). The brain simulates 50 ms and returns
   every output neuron's spike count. The answer arrives one tick late, about a fly's own
   looming-to-giant-fiber delay. The flight never waits for it: on a busy PC the brain answers a tick
   or two later (the end-of-flight summary counts those ticks) instead of stalling the control loop.
3. **When.** Avoid signal = the faster giant fiber's rate / 100 Hz, smoothed over ~150 ms. The
   supervisor enters AVOID at 0.5 (50 Hz), the same threshold as for every policy.
4. **Which way.** The escape goes away from the side whose steering descending neurons (DNp02, DNp04,
   DNp06, DNp11) fire more. These are strongly lateralized; the giant fibers much less so. When they
   barely favour a side (the threat is nearly straight ahead or behind, in both eyes' view), the
   planner's map chooses: the shorter way on to the stop, round the threat on that side. If the map
   can't tell (within 0.3 m), the eye does: away from the side the threat will pass on. For a moving
   threat that means its own path, for a static one the drone's. The map also overrules a clear brain
   choice that would mean a way on 1 m or more longer, and any side that runs into something mapped.
   Each episode logs what chose its side (brain, eye, map), and the end-of-flight summary counts them,
   so the brain's share of the decisions stays measurable.
5. **Escape.** Like a fly's escape takeoff, a burst up and away: for 1 s, up at 1 m/s (when the up
   range sensor sees room) and away at 0.4 m/s. Up, that is, unless the threat is a person: round
   them, never over (flying over people is restricted in most places). Over a vehicle and over
   static things it climbs. What the threat is comes from the camera when it sees it (a small CNN,
   `flypilot/camera_kind.py`), else from the eye's size of it (wider than 0.75 m and moving: a
   vehicle).
   - From a static threat: sideways to the chosen side, and forward or back away from where the eye
     sees the looming. Unless the threat is ahead, it doesn't brake below the speed it had.
   - From a moving one (a person, the cart): straight out of its path, across its line of travel, away
     from that line. Backing away along a walker's path only lets them follow the drone. "Away" means
     the side of the line the drone will be on once it could turn round: its offset plus 0.5 s of its
     own motion across the line. Reversing in the cart's 1.2 m lane costs about 0.9 s, about as much
     time as there is. The speed is what it takes to clear the mover's far edge (+0.15 m) before its
     front arrives: 0.4 m/s for a person walking at the drone, up to 0.8 m/s (in practice ~0.7, the
     setpoint leash) for the cart.

   While the giant fibers stay at alarm level, bursts follow one another. The side is chosen once per
   episode, because flipping sides between bursts leaves the drone in the way. In the first 0.4 s, a
   clear reading from the eye may still correct it, judged on the mean of the readings so far. A
   mover's line at the drone is only as good as its heading estimate: 10 degrees off on the cart 2.5 m
   away moves it 0.4 m, and one such reading once turned the drone round in the cart's lane. A
   near-tie goes to the map's shorter way on, then to the way the drone is already moving. Once the
   giant fibers calm down, the drone holds and lets the threat pass. An episode climbs at most 1.4 m.
6. **Efference copy.** The flight tells the policy how far ahead its route will stop. The eye doesn't
   count a static thing beyond that point as a collision course: flying at a crate that the drone
   will stop short of is not a threat. Flies similarly decouple escape from the looming they expect,
   for example when landing.

The flight adds the same safety rules around every policy, the fly brain included:
- **No dodging into things.** A dodge slows down rather than fly into what the eye sees in its path,
  stopping 0.3 m short of it (the planner's clearance).
- **Down only when clear below.** After a climb, the drone returns to its route altitude (up to 1 m/s,
  0.4 m/s near the ground), and only while the eye's downward-looking rows see room below it: never
  onto a person or a cart. What is on the map counts only right under it (a crate beside it doesn't).
- **Remember what was dodged, all of it.** What an avoidance episode is about goes on the planner's map
  (orange dots on the top views), so the re-planned path goes round it instead of straight back into it.
  The eye sees only the near side, so the rest is filled in (`eye.footprint`): a round thing (a tree's
  crown) becomes the whole circle that side fits, a flat face is taken to go straight back as deep as it is wide.
  Going round it, more of it goes on the map as it comes into view, and the path is planned again. Only
  things never seen moving are added (a person or a cart moves on), and nothing the map already has.
- **Land with the avoidance on.** The avoidance keeps deciding down to 0.3 m above the landing spot,
  so someone walking onto it is dodged. Only the last 0.3 m is the pilot's plain touchdown.

What the brain does with looming, measured with `brain_server.py` (giant-fiber rate in Hz, left / right,
3 s per value; the 4-deep circuit gives nearly the same numbers):

| Eye drive | Left eye only | Right eye only | Both eyes |
|---|---|---|---|
| 0.02 | 0 / 0 | 2 / 7 | 1 / 6 |
| 0.04 | 2 / 0 | 12 / 20 | 16 / 29 |
| 0.06 | 8 / 6 | 16 / 30 | 25 / 45 |
| 0.08 | 15 / 14 | 21 / 36 | 32 / 55 |
| 0.10 | 27 / 26 | 25 / 48 | 45 / 71 |
| 0.15 | 56 / 53 | 38 / 65 | 71 / 93 |
| 0.20 | 77 / 73 | 53 / 88 | 88 / 112 |
| 0.30 | 101 / 89 | 74 / 119 | 112 / 132 |
| 0.50 | 129 / 100 | 88 / 150 | 135 / 152 |
| 1.00 | 167 / 104 | 98 / 184 | 171 / 193 |

- **The right eye's pathway is more sensitive to weak looming.** At drive 0.04 it already drives the
  giant fibers to 12-20 Hz, while the left eye's barely moves them. With strong looming each eye drives
  its own side's giant fiber hardest, as expected. The asymmetry is in the connectome data, not in our
  cut: the 4-deep circuit shows it too. Whether it is the fly's or the reconstruction's is open. The
  brain is left as it is; the drone calibrates its interface instead. The left eye's drive is scaled by
  1.35 (`FlyBrainPolicy.EYE_GAIN`), so both eyes bring the giant fibers to the alarm level at the same
  looming. Uncalibrated, a person coming up from behind on the left was flagged only 0.66 m away.
- **Looming on both eyes adds up.** Something approaching head-on, seen in the frontal overlap, raises
  the alarm (50 Hz) at about 3 s to contact (drive ~0.065). Either eye alone does so at ~2 s (the
  left eye calibrated, see above). The geometric policy alarms at 2 s. Before the eye counted only
  collision courses, and before the efference copy, everything the drone flew past or toward a stop
  loomed too. The gain then had to be 40% lower, which put the alarm at 1.9 s (both eyes) and 1.2 s
  (right) or 0.9 s (left).

## Results

The final comparison (2026-10-24): every test route in `flypilot/routes.py`, all on the same code,
flown headless. The fly brain flew each route twice, the geometric rule once. The numbers are the
closest the drone came to the person or the cart (drone edge to mover edge, height counted per part:
`flypilot/movers.py`). Static routes report whether every stop was delivered.

| Route | What it tests | Fly brain (run 1 / run 2) | Geometric rule |
|---|---|---|---|
| `square` | false alarms (open floor) | 0 episodes / 0 | 1 false alarm |
| `blocked` | an unmapped crate on the path | delivered / delivered | delivered |
| `hover` | a person walks up behind the hovering drone | 0.16 m / 0.21 m | 0.04 m |
| `headon` | a person walks at the drone | 0.20 m / 0.21 m | **contact** |
| `crossing` | a person crosses its path from the side | 0.42 m / 0.44 m | **contact** |
| `cart` | the cart crosses its path | 0.29 m / 0.18 m | **contact, crash** |
| `lane` | the cart reaches the drone mid-lane | 0.34 m / 0.31 m | **contact, crash** |
| `dropoff` | cruising at 4.5 m, a person walks into the drop-off point | 0.12 m / 0.23 m | **contact, crash** |
| `wallside` | an unmapped crate with no room on the wall side | delivered / delivered | delivered |
| `poles` | an unmapped 6 m pole at cruise height | delivered / delivered | delivered |
| `landing` | the cart parked on the drop-off point | delivered beside it / beside it | delivered beside it |

- **Contacts:** none in the fly brain's 22 flights; 5 of 11 routes for the geometric rule.
- **Who chose the side** of the fly brain's 55 avoidance episodes: the eye's collision geometry 33
  times, the brain's steering neurons 10, the planner's map 9, the drone's own motion (a tie) 3. The
  giant fibers decided *when* in every episode.
- **The brain's cost:** 18 ms per 50 ms tick on average, over 22,882 ticks, in Brian2 (the C engine
  came after this comparison: ~2 ms). It answered more than a tick late about 30 times per flight (the
  PC was in use).

Read this with care:

- **Small samples.** Two runs per route for the fly brain and one for the rule. Earlier suites on
  slightly different code had a fly-brain contact in `headon`, which led to the longer heading window
  for movers (`eye.CollisionCourse.HEADING_S`).
- **Not the same senses.** The rule decides on the four Multi-ranger rays, what a real Crazyflie
  carries; the fly brain on the 360-degree eye. Much of the gap is the eye's: the rule's contacts are
  threats it saw late, between or along its rays. Giving the other arms the same eye is the next step
  for a fair comparison (see Limitations).
- **The direction is mostly not the brain's.** The fly circuit's clear contribution here is the
  timing of the escape. The side came from the brain in 18% of the episodes.

### Against a looming model without the connectome

The abstract looming model arm (`--policy looming`, `flypilot/policies.py` `LoomingModelPolicy`) is the
fly-brain policy with the brain taken out: the same eye, collision-course selectivity, escapes, map,
person / vehicle rules, one-tick delay and smoothing. Its alarm is a threshold on the brain's own input,
the two eyes' summed looming drive. Calibrated to the brain's head-on alarm point (summed drive 0.13),
flown on every route twice (2026-10-24), closest approach to the person or cart:

| Route | Fly brain | Looming model (0.13) |
|---|---|---|
| `hover` | 0.16 / 0.21 m | 0.19 / 0.21 m |
| `headon` | 0.20 / 0.21 m | 0.33 / 0.26 m |
| `crossing` | 0.42 / 0.44 m | 0.46 / 0.34 m |
| `cart` | 0.29 / 0.18 m | 0.30 / 0.11 m |
| `lane` | 0.34 / 0.31 m | 0.10 / 0.11 m |
| `dropoff` | 0.12 / 0.23 m | 0.22 / 0.21 m |

Neither had a contact or a false alarm (`square`), and both delivered every static route. The brain did
better in `lane`: it raised the alarm with the cart 2.3-2.6 m away, the model at 1.9-2.0 m. That is the
connectome's sensitive right-eye pathway (the cart comes from the drone's right). But replaying the
brain's flights through the model at other thresholds shows the same can be had without the wiring:

| Model threshold | Alarm vs the brain's (median) | Brain episodes missed (of ~55) | Alarms the brain didn't raise | On `square` |
|---|---|---|---|---|
| 0.13 | 0.20 s later | 14 | 16 | 0 |
| 0.10 | the same | 1 | 12 | 0 |
| 0.08 | 0.10 s earlier | 0 | 10 | 0 |

So far, a plain looming threshold tuned to be as sensitive does what the connectome does here. The
brain's edge in `lane` is its sensitivity, which a threshold can match, not a computation the threshold
lacks. (Open loop, on the brain's flights: to be confirmed by flying the model at 0.10, and by more
runs per route.) A negative result for the research question, and a useful one.

### Where is the threat? Each looming detector with its own receptive field (open loop)

In flight every LC4/LPLC2 neuron of an eye gets the same input, so the brain can't know where a threat
is. `retinotopy.py` gives each of the 314 its own receptive field, from the wiring alone:
1. **Columns.** The eye's ~780 columns are placed by their Mi1 cells' positions, flattened.
2. **Columnar neurons.** Their positions are spread over the synapses of 33 columnar cell types, ~24,500
   neurons per eye.
3. **Receptive fields.** Each LC4/LPLC2 field is centred where its own inputs come from.

The map is oriented by the cells' physical axes and by the VS cells' known front-to-back order:
- The front-to-back sign, found independently for each eye, comes out the same in both.
- HSN/HSE/HSS land dorsal/equatorial/ventral in the left eye. In the right eye, HSN lands just below HSE.
- Each eye's fields tile its own side, from 0 to ~155 degrees (`results/retinotopy.png`).

`directions.py` then sends looming from 24 directions (a 0.25 m sphere at 1.5 m/s) through the escape
circuit and decodes the descending neurons. The decoder is a linear readout, cross-validated by trial
(2026-10-25, 8 trials per direction):

| Input | Left/right | Front/back | Azimuth error | Unseen direction | Up/down |
|---|---|---|---|---|---|
| two channels (as flown) | 89% | 66% | 29 deg | 45 deg | 100% |
| each neuron its own field | 100% | 100% | 3 deg | 7 deg | 100% |
| fields shuffled among each eye's neurons | 100% | 100% | 5 deg | 10 deg | 90% |

- **With its own receptive fields, the circuit's descending neurons carry where the threat is.** A small
  trained readout recovers it to a few degrees, including directions it was never trained on. Today's
  two channels can't give front/back, which is why the eye supplies it in flight.
- **But a shuffled map does almost as well.** What matters is that each neuron sees its own patch, not
  much which patch. The connectome's own map is only slightly better (3 vs 5 deg; up/down 100% vs 90%;
  one shuffle, not yet tested for significance).
- **Timing.** The input gain was calibrated so a head-on threat alarms the giant fibers about as early
  as with two channels (1.57 vs 1.65 s before contact, gain 128). The brain then prefers the front:
  - Lateral and rear threats alarm later (0.45-1.35 s).
  - Threats 20 degrees below alarm earlier than in flight (1.0-1.55 s vs 0.60 s). The flight's eye leaves
    out rows below -10 degrees.
  - The left eye is less sensitive here too.

### Flying it: receptive fields and a trained readout (`--policy flybrain-rf`)

The eye sends its looming per ommatidium (`eye.Looming.loom_points`). The brain server drives each
LC4/LPLC2 neuron with the looming inside its own receptive field (`brain_server.Inputs`). A small trained
readout turns the 37 output neurons' rates into where the threat is (`train_readout.py`:
linear, ridge regression, `flypilot/models/brain_readout_<circuit>.npz`). The escape for a standing threat goes away
from it, as a fly's takeoff goes away from the looming: sideways to the far side, and forward or back. The
giant fibers still decide when.

**Two changes to the fly's eye for a drone.** The first flight of this version (2026-10-25) touched a person
twice: in `hover` (someone walks up from behind) and in `crossing` (from the side). The giant fibers
alarmed late, and both reasons are biology:
- **No blind spot behind.** The fly's fields reach ~155 degrees round each side and leave ~50 degrees blind
  behind. `RF_STRETCH` spreads them over all 180 degrees.
- **The same warning from every side.** The circuit answers the front first: a sphere from the side alarmed
  the giant fibers 0.4-0.8 s before contact, head-on 1.1 s. `RF_PROFILE` boosts each eye's input by
  azimuth so that a threat from any side alarms as early as head-on. `train_readout.py --equalize` finds
  the boost, up to x9.4 on the left at 90 degrees and x2.9 on the right, which is the left/right asymmetry
  again.

The head-on gain `RF_GAIN` was then set on 80 recorded flights against the two-channel brain
(`--calibrate`):
- **Timing:** alarms 0.15 s earlier (median).
- **Agreement:** 8% of its alarms are missed, the same as two runs of the two-channel brain differ from each
  other.
- **False alarms:** none on `square`.

**The readout**, on flights it was not trained on, at alarm level (4,818 ticks):

| | Trained readout | The two-channel brain's steering neurons |
|---|---|---|
| Bearing error | median 6 deg, 90% within 19 deg | - |
| Threat on the left or right | 100% | 75% |
| Threat ahead or behind | 97% | cannot tell |

**Flown on every route** (2026-10-25, after the changes; closest approach to the person or cart):

| Route | `flybrain` | `flybrain-rf` |
|---|---|---|
| `showcase` | 0.36 m | 0.36 m |
| `hover` | 0.25 m | 0.17 m |
| `headon` | 0.21 m | 0.31 m |
| `crossing` | 0.47 m | 0.51 m |
| `cart` | 0.22 m | 0.29 m |
| `lane` | 0.29 m | 0.35 m |
| `dropoff` | 0.15 m | 0.23 m |

Both delivered every stop of all 12 routes, with no contact and no false alarm on `square`.

**Which way: what the brain now decides.**
- **Front or back:** the readout decides it for every standing threat. Before, only the eye could.
- **The side:** the counts barely moved: brain 11, eye 22, map 5 (two channels: 10, 22, 4, plus 1 by
  motion). Most episodes are people and the cart. Getting out of a mover's path needs where it is heading,
  and one look at the looming doesn't carry that, so the eye's tracker still supplies it.
- **Next:** the readout's bearing changing over time does carry where a mover is heading.

### On the camera (`--policy flybrain-cam`, experimental)

This is the whole pipeline on what a real Crazyflie carries: AI-deck camera -> looming -> the same
receptive fields, circuit and trained readout. The Multi-ranger covers beside and behind with the
geometric time-to-collision rule, and is a safety net ahead at half weight. It never reads the 360-degree
eye.

**Looming from the camera images is the weak link** (`flypilot/camera_loom.py`).
- **What failed:** motion-detector versions, with dense optic flow standing in for T4/T5. Neither four-arm
  LPLC2 units nor a shift + turn + expansion fit per field could tell looming from noise at the recorded
  half resolution (162 x 122). A person 2 m off grows by ~0.5 pixel per frame, under the image's shift from
  the drone's own motion, and the simulator's plain surfaces carry no motion inside them.
- **What works best so far:** objects as image regions bounded by edges, tracked from frame to frame. The
  looming is the slope of a fit of ln(size) over 0.3 s, and it counts only if the region grows around
  itself.
- **The drone's own approach is taken out** (`gate`), as the eye does with distances. Something standing
  still would be at its approach speed / looming. Looming is dropped when that places it beyond the stop,
  on the map, or on the floor.

Replayed on 15 recorded flights, the camera-driven brain:
- raised 10 of the 24 alarms the eye-driven brain raised for a threat in the camera's view, at the same time;
- raised 48 alarms the eye-driven brain didn't (~3 per flight), one in each `square` flight.

**Flown on every route** (2026-10-25):
- **Standing obstacles:** `blocked`, `wallside`, `poles` and `landing` delivered with no contact. The brain
  chose most sides: 18 of 24 episodes there.
- **`square`:** 2 false alarms.
- **People and the cart:** contacts in 6 of the 8 routes with movers. In `cart` and `lane` nothing alarmed
  before the cart hit the drone from the side.

**The sensors, not the brain.** The camera sees ~86 degrees ahead. The four range rays are each a 20-degree
fan, so a person or cart coming from the side or overtaking at an angle is in the gaps until it is too
late. In `cb_headon` a person overtook from behind at an angle, and the rays first saw them 0.22 m off. The
geometric rule on the same rays did no better (contacts on 5 of 11 routes, Results above). With the
360-degree eye, the same brain had none. On a real Crazyflie's sensors, coverage limits the avoidance
before the brain does.

Next, for a real drone:
- wider sight: a second camera, or more range sensors;
- full-resolution frames: the camera gives 324 x 244; the recordings keep half;
- or a sensor made for looming: an event camera.

## Limitations, and what would be next

- **The eye is ideal.** Looming, and whether something is on a collision course, are computed from the
  simulator's distances and from tracking objects in them, not from images. That is the function of
  the fly's LPLC2 neurons, not their mechanism. A motion-based front end (T4/T5-like motion detectors
  feeding LPLC2-like units, on the camera or on this eye) is the next research step. Looming computed
  from camera images was tried first; its noise kept the giant fibers firing.
- **Part of the direction is the eye's, the map's and the camera's.** The brain decides when to escape,
  and the side in clear cases. The eye's collision geometry, the planner's map (which way round is
  shorter) and what the threat is (round a person, over a vehicle) supply the rest; `side_from` in
  each decision record says which. It picks the side for threats nearly straight
  ahead or behind, where the brain's left/right readout is too weak to rely on. It gives the
  forward/back part of the escape. For a moving threat it gives the whole direction (out of its path),
  since the brain's two input channels can't tell where a threat is heading.
- **One input channel per eye (`flybrain`).** All of an eye's LC4/LPLC2 neurons get the same rate, so
  that brain can't tell where in the eye the looming is; the forward/back part of its escape is the eye's.
  `flybrain-rf` drives each by its own receptive field, and its trained readout gives side and front/back
  (Results). Where a moving threat is heading is still the eye's in both.
- **The receptive fields are the fly's, changed twice for a drone** (`RF_STRETCH`: no blind spot behind;
  `RF_PROFILE`: the same warning from every side). Without them it touched people coming from behind and
  from the side. The map itself is derived, not measured: column positions from Mi1 cell bodies, oriented
  by known receptive fields (VS, HS), stretched over an approximate field of view.
- **Fairness of the comparison.** The other arms (CNN, shuffled connectome, abstract looming model,
  geometric rule) should get the same 360-degree eye. The geometric policy still uses the four range
  sensors.
- **Speed.** With the C engine the brain takes ~2 ms per 50 ms tick in flight (Brian2 took 17 ms on
  average, and more on a busy PC, when it answered late: the end-of-flight summary counts late ticks).
  The results above were flown with Brian2, before the C engine. The 4-deep circuit, which matches the
  whole brain on every giant-fiber rate, now fits the tick too (~30 ms with looming on this PC).
  Running on the drone itself would mean porting `fastlif.c` to the AI-deck (fixed point); neuromorphic
  hardware is the other route.
- **The connectome is a female brain (FlyWire).** Driving it with the drone's looming is an
  analogy: the drone is not a fly, and the descending neurons' targets (legs, wings) are replaced by
  velocity commands.

## Memory: why a laptop or a chat sandbox runs out

The whole brain (~139,000 neurons, ~15 million connections) takes **3 GB (FlyWire 630) to 4.3 GB
(FlyWire 783) of memory per simulation**, plus a one-time C++ compile. The model repo's examples run
one simulation per CPU core in parallel (`n_proc=-1`). On a 16-core PC that asks for ~70 GB.
WSL gives Linux only half of the PC's memory by default, and a chat assistant's sandbox much less.
That is what makes it crash with "out of resources".

- `brain_reproduce.cmd` and `brain_validate.cmd` never run more simulations at once than free memory allows
  (`reproduce.py` `parallel_trials`), so they work on a 16 GB PC, just more slowly.
- The drone itself only runs the 1,656-neuron circuit: well under 1 GB. The whole brain in the C engine
  (`prune.py`) needs ~1 GB per simulation.
- To give WSL more memory, add this to `C:\Users\<you>\.wslconfig` and run `wsl --shutdown`:
  ```
  [wsl2]
  memory=24GB
  ```

## Files

| File | Purpose |
|---|---|
| `connectome.py` | the FlyWire connectome and its cell-type labels: pick neurons by name (`LC4`, `DNp01`) |
| `circuit.py` | `extract()` cuts the escape circuit out of the whole brain; `escape_circuit()` caches it; `Subcircuit` runs it in Brian2 with the model's own equations |
| `reproduce.py` (`flypilot\brain_reproduce.cmd`) | check 1: the model against the paper's result, and looming -> giant fiber in the whole brain |
| `validate.py` (`flypilot\brain_validate.cmd`) | check 2: the cut-out circuit against the whole brain, and its speed |
| `brain_server.py` | the brain for the drone: the circuit, trading looming for spike counts every 50 ms over UDP 5320 (`--engine fast`, the default, or `brian2`) |
| `fastlif.c`, `fastlif.py` | the C engine: the same network ~10x cheaper in flight, and the whole brain on one core (`lif_run_sparse`); built with gcc by setup (`build/`, not in git) |
| `prune.py` (`flypilot\brain_prune.cmd`) | activity-based pruning: the whole brain on many looming tests, the neurons that fire (`active_783.txt`), held-out tests against the whole brain and the cuts; `check`: the whole brain in C against Brian2 |
| `active_783.txt` | the pruned circuit's neurons (FlyWire ids); `circuit.active_circuit()` builds the circuit from it |
| `check_fast.py` | the C engine against Brian2: spike for spike on the same input, the giant fibers' looming response, compute per tick |
| `retinotopy.py` | each LC4/LPLC2 neuron's receptive field, from the connectome (cached in `cache/`; `bash py.sh retinotopy.py` prints the checks and draws `results/retinotopy.png`) |
| `directions.py` | open loop: does the brain know where a threat is? Two channels vs receptive fields vs shuffled fields |
| `train_readout.py` | the trained readout (`../flypilot/models/brain_readout_<circuit>.npz`, `--circuit`), from recorded flights replayed offline; `--calibrate` / `--equalize` set the receptive-field input's gains (`brain_server.RF_GAIN`, `RF_PROFILE`) |
| `../flypilot/camera_loom.py` | looming from the AI-deck camera, for `flybrain-cam` |
| `cache/` | the cut circuit (`circuit.escape_circuit`), so the server doesn't load the whole connectome (not in git) |
| `setup_brain.sh`, `requirements.txt`, `py.sh` | setup (run by `flypilot/setup.sh`) and the brain's own Python environment |
| `results/` | whole-brain results of `brain_reproduce.cmd`, the receptive-field map, `directions.py`'s data (not in git) |
| `../flypilot/eye.py` | the compound eye and looming |
| `../flypilot/policies.py` `FlyBrainPolicy` | eye -> brain -> avoid signal and escape |

## Credits and licences

Data, all **CC-BY 4.0**; cite them in every figure and report that uses results from this code:
- **Connectome:** FlyWire, Dorkenwald et al. 2024, *Neuronal wiring diagram of an adult brain*,
  Nature 634:124. Versions 783 (used here) and 630, as packaged with the model.
- **Cell types:** Schlegel et al. 2024, *Whole-brain annotation and multi-connectome cell typing of
  Drosophila*, Nature 634:139
  ([flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations)).

Model: Shiu et al. 2024, *A Drosophila computational brain model reveals sensorimotor processing*,
Nature 634:210. Code: [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)
(MIT licence), via the fork [sazulai/Drosophila_brain_model](https://github.com/sazulai/Drosophila_brain_model).

Looming circuit background: von Reyn et al. 2014 (Nat Neurosci 17:962) and Ache et al. 2019 (Curr
Biol 29:1073) on LC4 / LPLC2 -> giant fiber; Card & Dickinson 2008 (Curr Biol 18:1300) on escape
direction.
