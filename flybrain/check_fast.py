"""Does fastlif (the C engine the flight uses) run the same network as Brian2 (circuit.Subcircuit)?

    bash py.sh check_fast.py [--hops 3] [--dt-ms 0.2] [--trials 6]

1. Spike for spike: both engines get the same input spikes (a random draw, given to Brian2 through a
   SpikeGeneratorGroup) for 1 s; every neuron's spike count must match. Only floating-point rounding can
   tell them apart (the exact solution of the linear equations, computed in different order).
2. The giant fibers' response to looming, with each engine's own Poisson input: mean rate over trials at a
   few drive levels, both eyes; they must agree within trial-to-trial noise (3 standard errors, or 10%).
3. Compute: wall time per 50 ms tick, each engine.
"""
import argparse
import sys
import time

import numpy as np
from brian2 import Hz, SpikeGeneratorGroup, ms

from brain_server import FLIGHT_DT_MS, FLIGHT_HOPS, MAX_RATE_HZ, TICK_MS
from circuit import MIN_SYNAPSES, Subcircuit, escape_circuit
from fastlif import FastCircuit, _steps


def brian(c: dict, dt_ms: float, source=None) -> Subcircuit:
    n = len(c['ids'])
    return Subcircuit(None, list(range(n)), c['inputs'], list(c['gf']), dt_ms=dt_ms,
                      arrays=(c['pre'], c['post'], c['count']), input_source=source)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hops', type=int, default=FLIGHT_HOPS)
    ap.add_argument('--dt-ms', type=float, default=FLIGHT_DT_MS)
    ap.add_argument('--trials', type=int, default=6)
    args = ap.parse_args()
    c = escape_circuit('783', args.hops, MIN_SYNAPSES)
    n, n_in, dt = len(c['ids']), len(c['inputs']), args.dt_ms
    print(f'circuit: {n} neurons, {len(c["pre"])} connections, {n_in} looming-detector inputs; step {dt:g} ms')
    ok = True

    # 1. the same input spikes
    rng = np.random.default_rng(1)
    steps = _steps(1000.0, dt)
    rates = np.where(c['input_left'], 0.3, 0.12) * MAX_RATE_HZ  # a looming left of the drone
    spikes = (rng.random((steps, n_in)) < rates * dt / 1000.0).astype(np.uint8)
    s_idx, i_idx = np.nonzero(spikes)
    gen = SpikeGeneratorGroup(n_in, i_idx, s_idx * dt * ms)
    b = brian(c, dt, source=gen)
    t0 = time.perf_counter()
    cb = b.run(1000.0)
    tb = time.perf_counter() - t0
    f = FastCircuit(n, c['pre'], c['post'], c['count'], c['inputs'], dt_ms=dt)
    t0 = time.perf_counter()
    cf = f.run(1000.0, in_spikes=spikes)
    tf = time.perf_counter() - t0
    diff = np.flatnonzero(cb != cf)
    print(f'\n1. same input spikes, 1 s: {int(cb.sum())} spikes in Brian2, {int(cf.sum())} in C; '
          f'{len(diff)} of {n} neurons differ' + (f' (largest difference {int(np.abs(cb - cf).max())} spikes)' if len(diff) else ''))
    print(f'   giant fibers L/R: Brian2 {cb[c["gf"]].tolist()}, C {cf[c["gf"]].tolist()}')
    if len(diff) > max(3, n // 1000):
        ok = False

    # 2. the giant fibers' response, each engine with its own Poisson input
    print(f'\n2. giant-fiber rate (Hz, left / right) with Poisson input, both eyes, {args.trials} x 1 s:')
    print('   drive     Brian2               C')
    b = brian(c, dt)
    for drive in (0.05, 0.1, 0.2, 0.5):
        r = np.full(n_in, drive * MAX_RATE_HZ)
        res = {}
        for name, eng in (('brian2', b), ('c', f)):
            gf = []
            for _ in range(args.trials):
                eng.reset()
                eng.set_rates(r)
                gf.append(eng.run(1000.0)[c['gf']])
            res[name] = np.array(gf, float)
        mb, mf = res['brian2'].mean(0), res['c'].mean(0)
        se = np.sqrt(res['brian2'].var(0, ddof=1) / args.trials + res['c'].var(0, ddof=1) / args.trials)
        agree = np.all(np.abs(mb - mf) <= np.maximum(3 * se, 0.1 * np.maximum(mb, 1.0)))
        ok &= bool(agree)
        print(f'   {drive:4.2f}   {mb[0]:6.1f} / {mb[1]:6.1f}     {mf[0]:6.1f} / {mf[1]:6.1f}   '
              f'{"agree" if agree else "DIFFER"}')

    # 3. compute per tick
    print('\n3. compute per 50 ms tick (drive 0.2, both eyes):')
    for name, eng in (('Brian2', b), ('C', f)):
        eng.reset()
        eng.set_rates(np.full(n_in, 0.2 * MAX_RATE_HZ))
        eng.run(TICK_MS)
        t0 = time.perf_counter()
        for _ in range(40):
            eng.run(TICK_MS)
        print(f'   {name:7s} {1000 * (time.perf_counter() - t0) / 40:6.2f} ms')
    print(f'   (test 1 took {tb:.2f} s in Brian2 and {tf * 1000:.1f} ms in C for 1 s of brain time)')

    print('\n' + ('PASS: the C engine runs the same network' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
