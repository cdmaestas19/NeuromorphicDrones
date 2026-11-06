"""The fly brain as a server for the drone: the escape subcircuit, stepped every 50 ms tick.

    bash py.sh brain_server.py [--circuit active|h3|h4] [--engine fast|brian2]
    (launch.cmd starts it for --policy flybrain / flybrain-rf)

Engines: fast (the default) runs the network in C (fastlif.py: the same model, spike for spike, at ~1.4 ms
per tick: check_fast.py); brian2 runs it in Brian2, one long run paced by a network operation (5-18 ms per
tick, more on a busy PC). The circuit (circuit.flight_circuit) comes from cache/, so the server doesn't load
the whole connectome (~2.5 GB) after the first start. Every TICK_MS of simulated time it trades data with the
drone's policy (flypilot/policies.py FlyBrainPolicy) over UDP:
  receives {"type": "step", "tick": k, "drive": [left, right]}: each eye's looming drive 0..1
           (flypilot/eye.py); that eye's LC4 and LPLC2 neurons then get Poisson input at drive x 150 Hz,
           the model's activation rate (two channels: the brain can't tell where in an eye the looming is);
        or {"type": "step", "tick": k, "loom": [[az, el, rate], ...]}: the looming per ommatidium (eye.py
           Looming.loom_points: body frame, degrees, 1/s); each LC4 / LPLC2 neuron is driven by what is in
           its own receptive field (retinotopy.py): see Inputs.
  replies  {"tick": k - 1, "gf": [left, right], "dn": {...}, "lc": [left, right], "out": [...],
           "compute_ms": ...}: spike counts over the TICK_MS just simulated, which saw the previous tick's
           input; "out" is every output neuron's, in the order the reset reply names them. So the brain
           answers one tick (50 ms) late, about the delay of the fly's own looming-to-giant-fiber response.
  {"type": "reset"} puts every neuron back at rest (a new flight); the reply names the outputs and the circuit
  (the readout trained on that circuit's outputs goes with it).
Brain time stands still while no step arrives, so it stays in step with the drone whatever the wall clock does.
If it falls behind (a busy PC), it skips to the newest step queued; the drone never waits for it either
(FlyBrainPolicy uses the newest reply there is), so a slow brain answers late instead of stalling the flight.

The circuits are circuit.py's, checked against the whole brain by validate.py (the cuts) and prune.py (all
three, on held-out looming tests). Model and data: Shiu et al.
2024 (Nature 634:210); FlyWire (Dorkenwald et al. 2024, Nature 634:124); cell types: Schlegel et al. 2024
(Nature 634:139). CC-BY 4.0.
"""
import argparse
import json
import socket
import sys
import time

import numpy as np

from circuit import CIRCUITS, flight_circuit

PORT = 5320          # flight -> brain; the reply goes back to the sender
FLIGHT_CIRCUIT = 'active'  # pruned by activity (prune.py; circuit.CIRCUITS): 1,656 neurons, on held-out looming
                     # tests identical to the whole brain in 82 of 88 runs, no alarm missed or extra; 1.0-1.8 ms per
                     # tick in flight. Before it, the 3-hop cut (h3, 3,118 neurons): 4 of 88, 6 alarms missed and
                     # 6 extra. The 4-hop cut (18,326 neurons) took 33-69 ms per tick in flight in Brian2.
FLIGHT_DT_MS = 0.2   # integration step: half the compute of the model's 0.1 ms, with the giant fibers' response
                     # to looming the same within trial noise (flybrain/README.md); a busy PC needed it
TICK_MS = 50.0       # the flight's control tick (20 Hz)
MAX_RATE_HZ = 150.0  # Poisson rate for drive 1.0 (the model's default activation rate)
LOOM_REF = 5.0       # flypilot/eye.py: the looming rate (1/s) of drive 1.0
RF_GAIN = (43.2, 32.0)  # receptive-field input, left and right eye, head-on (x RF_PROFILE elsewhere): a neuron's
                     # drive is the looming in its field (a small share of a whole eye's), times this. Set on 80
                     # recorded flights (train_readout.py --calibrate) against the two-channel brain: alarms
                     # 0.15 s earlier (median), 8% of its alarms missed (two runs of the two-channel brain itself
                     # differ by ~8%: 314 vs 338 alarms), none on square. The left x1.35, as the two channels'
                     # EYE_GAIN (the connectome's left pathway is less sensitive)


# Two changes to the fly's eye for a drone (the fly's own map: retinotopy.py):
RF_STRETCH = 180.0 / 155.0  # receptive-field azimuths x this: the fly's fields reach ~155 degrees round each side,
                            # leaving ~50 degrees blind behind; a drone can't be (a person walking up from behind
                            # alarmed the giant fibers only at ~0.4 m: rb_hover)
RF_PROFILE = (((0.0, 1.0), (45.0, 2.48), (90.0, 9.41), (135.0, 3.67), (180.0, 1.18)),   # left eye's gain by
              ((0.0, 1.0), (45.0, 2.18), (90.0, 2.92), (135.0, 3.79), (180.0, 1.18)))   # |azimuth|, right eye's:
                            # the circuit answers the front first (from the side the giant fibers alarmed 0.4-0.8 s
                            # before contact where head-on gave 1.1 s); a drone needs the same warning from every
                            # side. Each makes a sphere from there alarm as early as head-on (train_readout.py
                            # --equalize: 0.25 m at 1.5 m/s, 1.05-1.23 s vs 1.10 s)


class Inputs:
    """What a step message becomes: the Poisson rate of every input neuron (the circuit's LC4 / LPLC2, in its
    order). 'loom' (the looming per ommatidium): each neuron at the looming inside its own receptive field
    (retinotopy.RFPool, azimuths x RF_STRETCH) / LOOM_REF x RF_GAIN x RF_PROFILE, capped at MAX_RATE_HZ. Else
    'drive' (two channels): every neuron of an eye at that eye's drive x MAX_RATE_HZ."""

    def __init__(self, c: dict, gain: tuple = RF_GAIN, profile: tuple = RF_PROFILE):
        from retinotopy import RFPool, receptive_fields
        self.left = c['input_left']
        rf = receptive_fields('783')
        pos = {int(f): i for i, f in enumerate(rf['ids'])}
        order = np.array([pos[int(f)] for f in c['ids'][c['inputs']]])
        self.az = np.clip(rf['az'][order] * RF_STRETCH, -180.0, 180.0)
        self.pool = RFPool({'az': self.az, 'el': rf['el'][order], 'size': rf['size'][order]})
        self.set_gain(gain, profile)

    def set_gain(self, gain: tuple, profile: tuple = RF_PROFILE):
        """gain: (left eye, right eye); profile: (left eye's, right eye's) ((|azimuth|, factor), ...)."""
        (at_l, g_l), (at_r, g_r) = (zip(*p) for p in profile)
        a = np.abs(self.az)
        self.gain = np.where(self.left, gain[0] * np.interp(a, at_l, g_l), gain[1] * np.interp(a, at_r, g_r))

    def rates(self, msg: dict) -> np.ndarray:
        if 'loom' in msg:
            pts = np.asarray(msg['loom'], float).reshape(-1, 3)
            drive = self.pool.drive(pts[:, 0], pts[:, 1], pts[:, 2]) / LOOM_REF
            return np.minimum(self.gain * drive, 1.0) * MAX_RATE_HZ
        left, right = (min(max(float(d), 0.0), 1.0) for d in msg['drive'])
        return np.where(self.left, left, right) * MAX_RATE_HZ


def outputs(c: dict) -> tuple[np.ndarray, list]:
    """The output neurons (positions in the circuit) and their names: both giant fibers, then the other
    descending neurons."""
    return (np.concatenate([c['gf'], c['dn_pos']]).astype(np.int64),
            ['DNp01 (left)', 'DNp01 (right)'] + c['dn_names'].tolist())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=PORT)
    ap.add_argument('--circuit', choices=CIRCUITS, default=FLIGHT_CIRCUIT,
                    help=f'which circuit (circuit.py; default {FLIGHT_CIRCUIT}): active = pruned by activity '
                         '(prune.py), h3 / h4 = cut 3 / 4 synapses deep')
    ap.add_argument('--hops', type=int, choices=(3, 4), help='the same as --circuit h3 / h4')
    ap.add_argument('--dt-ms', type=float, default=FLIGHT_DT_MS,
                    help=f'integration step (default {FLIGHT_DT_MS}; the model\'s is 0.1 ms)')
    ap.add_argument('--engine', choices=('fast', 'brian2'), default='fast',
                    help='fast: the network in C (fastlif.py, the default); brian2: in Brian2 (the reference)')
    args = ap.parse_args()

    t0 = time.time()
    name = f'h{args.hops}' if args.hops else args.circuit
    c = flight_circuit(name)
    n = len(c['ids'])
    eye_l, idx_in = c['input_left'], c['inputs']
    gf = [int(i) for i in c['gf']]
    dn = dict(zip(c['dn_names'].tolist(), (int(i) for i in c['dn_pos'])))
    lc_l, lc_r = idx_in[eye_l], idx_in[~eye_l]
    inputs = Inputs(c)
    out_pos, out_names = outputs(c)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', args.port))
    sock.settimeout(1.0)
    state = {'tick': -1, 'peer': None, 't_wall': None}

    def reply(counts: np.ndarray):
        """The TICK_MS just simulated, to the drone: the spike counts it reads (see the module docstring)."""
        if state['peer'] is None or state['tick'] < 0:
            return
        msg = {'tick': state['tick'], 'gf': [int(counts[gf[0]]), int(counts[gf[1]])],
               'lc': [int(counts[lc_l].sum()), int(counts[lc_r].sum())],
               'dn': {k: int(counts[i]) for k, i in dn.items() if counts[i]},
               'out': counts[out_pos].tolist(),
               'compute_ms': None if state['t_wall'] is None else 1000.0 * (time.perf_counter() - state['t_wall'])}
        sock.sendto(json.dumps(msg).encode(), state['peer'])

    def next_step(reset) -> np.ndarray:
        """Wait for the drone's next step (brain time stands still meanwhile); handle resets. Returns the
        Poisson rate per input neuron, from that step's input (Inputs)."""
        step = None
        while step is None:
            try:
                raw, peer = sock.recvfrom(65535)
            except socket.timeout:
                continue
            sock.setblocking(False)  # then everything else already queued: behind, the newest step wins
            try:
                while True:
                    try:
                        msg = json.loads(raw)
                    except ValueError:
                        msg = {}
                    if msg.get('type') == 'reset':
                        reset()
                        state['tick'], step = -1, None
                        sock.sendto(json.dumps({'reset': True, 'outputs': out_names, 'circuit': name}).encode(), peer)
                    elif msg.get('type') == 'step':
                        step = (msg, peer)
                    try:
                        raw, peer = sock.recvfrom(65535)
                    except BlockingIOError:
                        break
            finally:
                sock.settimeout(1.0)
        msg, peer = step
        state['tick'], state['peer'], state['t_wall'] = int(msg['tick']), peer, time.perf_counter()
        return inputs.rates(msg)

    def ready(n_connections: int):
        print(f'fly brain ready in {time.time() - t0:.0f} s: escape circuit "{name}" of {n} neurons, {n_connections} '
              f'connections; {len(lc_l)} + {len(lc_r)} looming-detector inputs (left + right eye), each with its '
              f'own receptive field; {len(out_pos)} outputs; '
              f'step {args.dt_ms:g} ms, {args.engine} engine; listening on udp://127.0.0.1:{args.port}', flush=True)

    try:
        if args.engine == 'fast':
            from fastlif import FastCircuit
            brain = FastCircuit(n, c['pre'], c['post'], c['count'], idx_in, dt_ms=args.dt_ms)
            ready(brain.n_connections)
            while True:
                brain.set_rates(next_step(brain.reset))
                reply(brain.run(TICK_MS))
        else:
            serve_brian2(c, args.dt_ms, next_step, reply, ready)
    except KeyboardInterrupt:
        pass
    return 0


def serve_brian2(c: dict, dt_ms: float, next_step, reply, ready):
    """The Brian2 engine: one long run, a network operation at each tick boundary trading with the drone."""
    from brian2 import Hz, ms, mV, network_operation
    from circuit import Subcircuit
    n = len(c['ids'])
    sub = Subcircuit(None, list(range(n)), c['inputs'], list(c['gf']), dt_ms=dt_ms,
                     arrays=(c['pre'], c['post'], c['count']))
    v_rest = sub.neu.v[0]
    last = {'count': np.zeros(n, np.int64)}

    def reset():
        sub.neu.v, sub.neu.g = v_rest, 0 * mV

    @network_operation(dt=TICK_MS * ms, when='start')
    def exchange():
        now = np.asarray(sub.mon.count, np.int64)
        counts, last['count'] = now - last['count'], now.copy()
        reply(counts)
        sub.poisson.rates = next_step(reset) * Hz

    sub.net.add(exchange)
    ready(sub.n_connections)
    sub.net.run(1e9 * ms)  # "forever": the network operation paces it


if __name__ == '__main__':
    sys.exit(main())
