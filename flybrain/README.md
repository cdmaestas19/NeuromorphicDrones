# flybrain: the fruit fly's escape circuit, flying the drone

This folder runs the escape circuit of the adult fruit-fly connectome as a spiking network, and lets it
decide when and which way the drone dodges. The flight stack is in [../flypilot](../flypilot/README.md);
the brain plugs into it as one avoidance policy (`--policy flybrain`) next to the geometric baseline.

```
compound eye (sim) --> looming per eye --> LC4 + LPLC2 looming detectors (Poisson input, per eye)
     360 degrees           eye.py                      |
                                         escape subcircuit of FlyWire 783 (3,118 LIF neurons)
                                                       |
                         giant fibers DNp01 (when)  +  steering DNs DNp02/04/06/11 (which side)
                                                       |
                         FlyBrainPolicy (flypilot/policies.py) --> supervisor --> dodge
```

- **The model** is the whole-brain leaky integrate-and-fire model of Shiu et al. 2024 (Nature
  634:210), unchanged: the same equations and parameters (`model.py` of
  [Drosophila_brain_model](https://github.com/sazulai/Drosophila_brain_model)). Each connection's
  weight is its signed synapse count from the FlyWire connectome.
- **The circuit** is cut out of the whole brain. It keeps every neuron on a path of at most 3 synapses
  from the looming detectors (LC4, LPLC2) to the escape outputs, through connections of at least 5
  synapses. There are 37 outputs: the two giant fibers (DNp01, the fly's escape command) and the 35
  other descending neurons that get at least 20 synapses straight from the looming detectors.
- **The eye** is an ideal one: the simulator measures the distance to the scenery every 5 degrees all
  the way round, from straight down to 30 degrees up. Looming is how fast that distance shrinks
  relative to itself (1 / time to contact), counted only for things on a collision course with the
  drone, as the fly's LPLC2 neurons respond only to an image that expands without sliding. Each eye
  covers its half of the world, with 15 degrees of overlap in front, as in flies.
- **The brain runs in its own process** (`brain_server.py`), in step with the flight's 20 Hz loop,
  with its own Python environment. Brian2 and PyTorch don't mix in one process, and the flight
  stack's numpy is 2.x.

## Running it

`flypilot\setup.cmd` sets everything up, the brain included. It fetches the model and its connectome
data (~200 MB) and the FlyWire cell-type labels, and creates the brain's Python environment
(`~/.venvs/flybrain-snn`, Brian2). Then:

```
flypilot\launch.cmd avoid --route hover --policy flybrain     a person walks up behind the hovering drone
flypilot\launch.cmd avoid --route headon --policy flybrain    (also crossing, cart, dropoff, square, blocked)
flypilot\launch.cmd deliver --scene street --policy flybrain  live stops among people and a cart; it cruises
                                                              at 4.5 m and comes down carefully at each stop
flypilot\brain_reproduce.cmd                                  check the model against published results
flypilot\brain_validate.cmd                                   check the cut-out circuit against the whole brain
```

`launch.cmd` starts the brain server for `--policy flybrain` (its log is `flypilot\logs\brain.log`).
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
| **3 deep, >= 5 (flown)** | **3,118** | close; one opposite-side rate off | **~11 ms per 50 ms tick; ~5 ms at the flight's 0.2 ms step** |
| 4 deep, >= 10 | 4,854 | close; one opposite-side rate off | |
| 4 deep, >= 5 | 18,326 | all 8 | 0.44 s per simulated second alone; 33-69 ms per 50 ms tick next to the simulator, too slow to fly |

The drone flies the 3-deep circuit, the largest one that keeps up with real time next to the simulator.
Its giant-fiber response to looming is practically the same as the fully validated 4-deep circuit's
(table below), so the cut costs little where it matters.

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
   fast enough to measure, and anything slower than 0.5 m/s counts as standing still. An object counts
   if, on the current relative motion, any of its points will pass within 0.15 m of the drone, and not
   at all beyond 0.3 m (the planner's clearance). An approaching person counts, and so does a wall the
   drone flies at, but a crate it flies past does not. Each eye's drive is its most looming 30-degree
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
- **No dodging into things.** A dodge slows down rather than fly into what the eye sees in its path.
- **Down only when clear below.** After a climb, the drone returns to its route altitude (up to 1 m/s,
  0.4 m/s near the ground), and only while the eye's downward-looking rows see room below it: never
  onto a person or a cart.
- **Remember what was dodged.** After each avoidance episode, the static things the eye sees within
  1.2 m that aren't on the planner's map go on it (shown as orange dots on the top views), so the
  re-planned path goes round an unmapped crate instead of straight back into it. Only things never
  seen moving are added: a person or a cart moves on.
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
- **The brain's cost:** 18 ms per 50 ms tick on average, over 22,882 ticks. It answered more than a
  tick late about 30 times per flight (the PC was in use).

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
- **One input channel per eye.** All of an eye's LC4/LPLC2 neurons get the same rate, so the brain
  can't tell where in that eye the looming is. That is why the forward/back part of the escape comes
  from the eye, not from the brain. Driving each looming detector by its own receptive field
  (retinotopy, from the optic-lobe column maps) would give the brain the full direction.
- **Fairness of the comparison.** The other arms (CNN, shuffled connectome, abstract looming model,
  geometric rule) should get the same 360-degree eye. The geometric policy still uses the four range
  sensors.
- **Speed.** The 3-deep circuit takes ~5 ms per 50 ms tick (0.2 ms steps) on one free core of a
  desktop CPU. With a game or other heavy programs running it can take several times that, and the
  brain then answers a tick or more late (the end-of-flight summary counts those ticks); late answers
  were behind most remaining contacts in testing, so close heavy programs for fly-brain runs. The 4-deep circuit
  that matches the whole brain on every giant-fiber rate is too slow to fly with in Brian2 here;
  Brian2's standalone C++ mode, GeNN (GPU), or neuromorphic hardware would be the routes.
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
- The drone itself only runs the 3,118-neuron circuit: well under 1 GB.
- To give WSL more memory, add this to `C:\Users\<you>\.wslconfig` and run `wsl --shutdown`:
  ```
  [wsl2]
  memory=24GB
  ```

## Files

| File | Purpose |
|---|---|
| `connectome.py` | the FlyWire connectome and its cell-type labels: pick neurons by name (`LC4`, `DNp01`) |
| `circuit.py` | `extract()` cuts the escape circuit out of the whole brain; `Subcircuit` runs it with the model's own equations |
| `reproduce.py` (`flypilot\brain_reproduce.cmd`) | check 1: the model against the paper's result, and looming -> giant fiber in the whole brain |
| `validate.py` (`flypilot\brain_validate.cmd`) | check 2: the cut-out circuit against the whole brain, and its speed |
| `brain_server.py` | the brain for the drone: the circuit in one long Brian2 run, trading looming for spike counts every 50 ms over UDP 5320 |
| `setup_brain.sh`, `requirements.txt`, `py.sh` | setup (run by `flypilot/setup.sh`) and the brain's own Python environment |
| `results/` | whole-brain results of `brain_reproduce.cmd` (not in git) |
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
