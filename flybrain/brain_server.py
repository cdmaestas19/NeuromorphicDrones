"""The fly brain as a server for the drone: the escape subcircuit, stepped every 50 ms tick.

    bash py.sh brain_server.py [--engine fast|brian2]    (launch.cmd starts it for --policy flybrain)

Engines: fast (the default) runs the network in C (fastlif.py: the same model, spike for spike, at ~1.4 ms
per tick: check_fast.py); brian2 runs it in Brian2, one long run paced by a network operation (5-18 ms per
tick, more on a busy PC). The circuit comes from circuit.escape_circuit()'s cache, so the server doesn't load
the whole connectome (~2.5 GB) after the first start. Every TICK_MS of simulated time it trades data with the
drone's policy (flypilot/policies.py FlyBrainPolicy) over UDP:
  receives {"type": "step", "tick": k, "drive": [left, right]}: each eye's looming drive 0..1
           (flypilot/eye.py); that eye's LC4 and LPLC2 neurons then get Poisson input at drive x 150 Hz,
           the model's activation rate.
  replies  {"tick": k - 1, "gf": [left, right], "dn": {...}, "lc": [left, right], "compute_ms": ...}:
           spike counts over the TICK_MS just simulated, which saw the previous tick's drive. So the brain
           answers one tick (50 ms) late, about the delay of the fly's own looming-to-giant-fiber response.
  {"type": "reset"} puts every neuron back at rest (a new flight).
Brain time stands still while no step arrives, so it stays in step with the drone whatever the wall clock does.
If it falls behind (a busy PC), it skips to the newest step queued; the drone never waits for it either
(FlyBrainPolicy uses the newest reply there is), so a slow brain answers late instead of stalling the flight.

The subcircuit is circuit.py's, validated against the whole brain by validate.py. Model and data: Shiu et al.
2024 (Nature 634:210); FlyWire (Dorkenwald et al. 2024, Nature 634:124); cell types: Schlegel et al. 2024
(Nature 634:139). CC-BY 4.0.
"""
import argparse
import json
import socket
import sys
import time

import numpy as np

from circuit import HOPS, MIN_SYNAPSES, escape_circuit

PORT = 5320          # flight -> brain; the reply goes back to the sender
FLIGHT_HOPS = 3      # 3,118 neurons: ~10 ms per 50 ms tick alone, ~2-3x that next to the simulator. The 4-hop
                     # cut (18,326 neurons) matches the whole brain on every giant-fiber rate but took 33-69 ms
                     # per tick in flight: slower than real time.
FLIGHT_DT_MS = 0.2   # integration step: half the compute of the model's 0.1 ms, with the giant fibers' response
                     # to looming the same within trial noise (flybrain/README.md); a busy PC needed it
TICK_MS = 50.0       # the flight's control tick (20 Hz)
MAX_RATE_HZ = 150.0  # Poisson rate for drive 1.0 (the model's default activation rate)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=PORT)
    ap.add_argument('--hops', type=int, default=FLIGHT_HOPS,
                    help=f'subcircuit size (circuit.py): {FLIGHT_HOPS} keeps up with the drone on a busy PC; '
                         f'{HOPS} matches the whole brain best (validate.py) but needs ~2x the compute')
    ap.add_argument('--dt-ms', type=float, default=FLIGHT_DT_MS,
                    help=f'integration step (default {FLIGHT_DT_MS}; the model\'s is 0.1 ms)')
    ap.add_argument('--engine', choices=('fast', 'brian2'), default='fast',
                    help='fast: the network in C (fastlif.py, the default); brian2: in Brian2 (the reference)')
    args = ap.parse_args()

    t0 = time.time()
    c = escape_circuit('783', args.hops, MIN_SYNAPSES)
    n = len(c['ids'])
    eye_l, idx_in = c['input_left'], c['inputs']
    gf = [int(i) for i in c['gf']]
    dn = dict(zip(c['dn_names'].tolist(), (int(i) for i in c['dn_pos'])))
    lc_l, lc_r = idx_in[eye_l], idx_in[~eye_l]

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
               'compute_ms': None if state['t_wall'] is None else 1000.0 * (time.perf_counter() - state['t_wall'])}
        sock.sendto(json.dumps(msg).encode(), state['peer'])

    def next_step(reset) -> np.ndarray:
        """Wait for the drone's next step (brain time stands still meanwhile); handle resets. Returns the
        Poisson rate per input neuron, from that step's drive."""
        step = None
        while step is None:
            try:
                raw, peer = sock.recvfrom(4096)
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
                        sock.sendto(json.dumps({'reset': True}).encode(), peer)
                    elif msg.get('type') == 'step':
                        step = (msg, peer)
                    try:
                        raw, peer = sock.recvfrom(4096)
                    except BlockingIOError:
                        break
            finally:
                sock.settimeout(1.0)
        msg, peer = step
        left, right = (min(max(float(d), 0.0), 1.0) for d in msg['drive'])
        state['tick'], state['peer'], state['t_wall'] = int(msg['tick']), peer, time.perf_counter()
        return np.where(eye_l, left, right) * MAX_RATE_HZ

    def ready(n_connections: int):
        print(f'fly brain ready in {time.time() - t0:.0f} s: escape subcircuit of {n} neurons, {n_connections} '
              f'connections; {len(lc_l)} + {len(lc_r)} looming-detector inputs (left + right eye); '
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
