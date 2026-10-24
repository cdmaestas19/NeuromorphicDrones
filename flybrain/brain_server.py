"""The fly brain as a server for the drone: the escape subcircuit, running continuously in Brian2.

    bash py.sh brain_server.py            (launch.cmd starts it for --policy flybrain)

One long Brian2 run, so there is no per-call overhead. Every TICK_MS of simulated time a network operation
trades data with the drone's policy (flypilot/policies.py FlyBrainPolicy) over UDP:
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
from brian2 import Hz, ms, mV, network_operation

from circuit import HOPS, MIN_SYNAPSES, Subcircuit, extract
from connectome import GIANT_FIBER, Connectome

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
    args = ap.parse_args()

    t0 = time.time()
    conn = Connectome('783')
    ins, outs, ids = extract(conn, args.hops, MIN_SYNAPSES)
    sub = Subcircuit(conn, ids, ins, outs, dt_ms=args.dt_ms)
    side_of = {f: conn.labels.side.get(f) for f in ins}
    eye_l = np.array([side_of[f] == 'left' for f in ins])
    eye_r = np.array([side_of[f] == 'right' for f in ins])
    pos = {f: i for i, f in enumerate(ids)}
    gf = [pos[conn.cells(GIANT_FIBER, s)[0]] for s in ('left', 'right')]
    dn = {conn.name(f): pos[f] for f in outs if not conn.name(f).startswith(GIANT_FIBER)}
    lc_l = np.array([pos[f] for f in ins if side_of[f] == 'left'])
    lc_r = np.array([pos[f] for f in ins if side_of[f] == 'right'])
    v_rest = sub.neu.v[0]
    del conn  # the full connectivity table is no longer needed

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', args.port))
    sock.settimeout(1.0)
    state = {'tick': -1, 'peer': None, 'last': np.zeros(len(ids), np.int64), 't_wall': None}

    @network_operation(dt=TICK_MS * ms, when='start')
    def exchange():
        """At each tick boundary: report the TICK_MS just simulated, then wait for the drone's next tick."""
        now = np.asarray(sub.mon.count, np.int64)
        counts, state['last'] = now - state['last'], now.copy()
        if state['peer'] is not None and state['tick'] >= 0:
            reply = {'tick': state['tick'], 'gf': [int(counts[gf[0]]), int(counts[gf[1]])],
                     'lc': [int(counts[lc_l].sum()), int(counts[lc_r].sum())],
                     'dn': {k: int(counts[i]) for k, i in dn.items() if counts[i]},
                     'compute_ms': None if state['t_wall'] is None else 1000.0 * (time.perf_counter() - state['t_wall'])}
            sock.sendto(json.dumps(reply).encode(), state['peer'])
        step = None
        while step is None:  # brain time stands still until the drone's next step
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
                        sub.neu.v, sub.neu.g = v_rest, 0 * mV
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
        sub.poisson.rates = (eye_l * left + eye_r * right) * MAX_RATE_HZ * Hz
        state['tick'], state['peer'], state['t_wall'] = int(msg['tick']), peer, time.perf_counter()

    sub.net.add(exchange)
    print(f'fly brain ready in {time.time() - t0:.0f} s: escape subcircuit of {len(ids)} neurons, '
          f'{sub.n_connections} connections; {len(lc_l)} + {len(lc_r)} looming-detector inputs (left + right eye); '
          f'step {args.dt_ms:g} ms; listening on udp://127.0.0.1:{args.port}', flush=True)
    try:
        sub.net.run(1e9 * ms)  # "forever": the network operation paces it
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
