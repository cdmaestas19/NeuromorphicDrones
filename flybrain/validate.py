"""Is the escape subcircuit faithful to the whole brain, and fast enough to fly with?

    flypilot\\brain_validate.cmd [--hops 4] [--min-synapses 5] [--trials 10]

Results on FlyWire 783 (10 trials each, 2026-10-21): hops 2 / >= 5 synapses (566 neurons) misses inhibition,
the contralateral giant fiber fires up to 2.5x too much; hops 3 / >= 5 (3,118 neurons) and hops 4 / >= 10
(4,854) are close, one contralateral rate off; hops 4 / >= 5 (18,326 neurons, 2.4M connections) matches all
8 giant-fiber rates and runs at 0.44 s per simulated second: the default.

Run `flypilot\\brain_reproduce.cmd escape` first: it leaves the whole-brain results in results/. For the same experiments
(one eye's LC4 or LPLC2 at 150 Hz), this runs the subcircuit (circuit.py) for the same number of trials
and compares every output neuron's firing rate with the whole brain's. A rate matches when the difference
is within 10%, or within 3 standard errors of the difference (trial-to-trial noise), whichever is larger.
Pass: both giant fibers match in every experiment. It also measures speed (wall time per simulated second;
the drone needs well under 1).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from circuit import HOPS, MIN_SYNAPSES, Subcircuit, extract
from connectome import GIANT_FIBER, Connectome
from reproduce import RESULTS, rates

EXPERIMENTS = [('LC4', 'left'), ('LPLC2', 'left'), ('LC4', 'right'), ('LPLC2', 'right')]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hops', type=int, default=HOPS, help='longest input->output path kept, in synaptic steps')
    ap.add_argument('--min-synapses', type=int, default=MIN_SYNAPSES, help='weakest connection followed when cutting')
    ap.add_argument('--trials', type=int, default=10)
    args = ap.parse_args()

    conn = Connectome('783')
    t0 = time.perf_counter()
    ins, outs, ids = extract(conn, args.hops, args.min_synapses)
    sub = Subcircuit(conn, ids, ins, outs)
    print(f'subcircuit (hops <= {args.hops}, connections >= {args.min_synapses} synapses): {len(ids)} neurons, '
          f'{sub.n_connections} connections ({len(ins)} input, {len(outs)} output); built in {time.perf_counter() - t0:.1f} s')
    print('outputs:', ', '.join(conn.name(f) for f in outs))

    gf = set(conn.cells(GIANT_FIBER))
    all_ok, walls = True, []
    for cell_type, side in EXPERIMENTS:
        whole_path = RESULTS / f'{cell_type}_{side}.parquet'
        if not whole_path.is_file():
            print(f'no whole-brain result {whole_path.name}: run flypilot\\brain_reproduce.cmd escape first')
            return 1
        whole = rates(whole_path, args.trials, 1.0)
        stim = set(conn.cells(cell_type, side))
        r_in = [150.0 if f in stim else 0.0 for f in ins]
        counts = np.zeros((args.trials, len(ids)))
        for k in range(args.trials):
            sub.reset()
            sub.set_rates(r_in)
            t = time.perf_counter()
            counts[k] = sub.run(1000.0)
            walls.append(time.perf_counter() - t)
        mean, sd = counts.mean(axis=0), counts.std(axis=0, ddof=1)
        print(f'\n{cell_type} {side} at 150 Hz, {args.trials} trials: output rates in Hz, '
              f'whole brain vs subcircuit (mean +- sd)')
        for f in outs:
            w, w_sd = whole.rate.get(f, 0.0), whole['std'].get(f, 0.0)
            s, s_sd = mean[ids.index(f)], sd[ids.index(f)]
            if w < 1 and s < 1:
                continue
            se = ((w_sd ** 2 + s_sd ** 2) / args.trials) ** 0.5
            ok = abs(s - w) <= max(0.10 * w, 3.0 * se, 1.0)
            if f in gf:
                all_ok &= ok
            print(f'  {conn.name(f):22s} {w:6.1f} +- {w_sd:4.1f}   {s:6.1f} +- {s_sd:4.1f}  '
                  f'{"ok" if ok else "DIFFERENT"}{"  <- giant fiber" if f in gf else ""}')

    speed = np.mean(walls[1:]) if len(walls) > 1 else walls[0]  # the first trial includes compiling
    print(f'\nspeed: {speed:.2f} s wall per simulated second '
          f'({"fast enough for the drone" if speed < 0.6 else "too slow for the drone"})')
    print('\nGIANT FIBERS MATCH THE WHOLE BRAIN' if all_ok else '\nGIANT FIBERS DIFFER FROM THE WHOLE BRAIN')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
