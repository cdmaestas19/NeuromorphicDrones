"""Activity-based pruning: run the whole brain on many looming tests, keep the neurons that spike.

    bash py.sh prune.py check      the whole brain in C (fastlif) against the model repo's Brian2 runs
    bash py.sh prune.py run        the training tests through the whole brain (results/prune_train.npz)
    bash py.sh prune.py build      the neurons that spiked -> active_783.txt (in git; circuit.active_circuit())
    bash py.sh prune.py evaluate   held-out tests: the active circuit and the 3- / 4-hop cuts vs the whole brain
    (flypilot\\brain_prune.cmd runs all four)

A mentor's suggestion. In this model (Shiu et al. 2024) a neuron acts on others only when it spikes: no
background activity, nothing below threshold reaches another neuron. So a neuron that never spikes during
a test can be removed without changing that test's result at all; the question is only whether the tests
covered what the drone will meet. Training tests: spheres approaching from every 30 degrees of azimuth, two
elevations, two sizes and speeds (train_readout.SYNTH), and recorded flights replayed through the eye, both
the receptive-field input (flybrain-rf) and the two-channel one (flybrain). Held-out tests, never used to
choose neurons: spheres from the directions in between, at an intermediate size and speed ("between"), larger
and faster ones than any training sphere ("beyond"), and other recorded flights.

Every circuit runs in the C engine (fastlif: a fast CPU, not a GPU, as another mentor suggested: the whole
brain is ~139k neurons, too sparse in activity for a GPU to pay off, and the C engine needs ~0.3 GB where a
Brian2 run needs 4.3 GB, so many run at once). All circuits share the same input neurons in the same order,
so the same seed gives them the same input spikes: the comparisons are paired, test by test.

Model and data: Shiu et al. 2024 (Nature 634:210); FlyWire (Dorkenwald et al. 2024, Nature 634:124); cell
types: Schlegel et al. 2024 (Nature 634:139). CC-BY 4.0.
"""
import argparse
import multiprocessing as mp
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

from brain_server import FLIGHT_DT_MS, RF_GAIN, TICK_MS, Inputs, outputs
from circuit import ACTIVE_IDS, MIN_SYNAPSES, active_circuit, escape_circuit, flight_circuit as circuit, whole_circuit
from fastlif import FastCircuit
from train_readout import QUIET_TICKS, SYNTH, alarm_onsets, flights, replay_eye, smooth, synthetic

RESULTS = Path(__file__).resolve().parent / 'results'
TEST_BETWEEN = [(float(az), 5.0, 0.35, 1.5) for az in range(-165, 180, 30)]  # between the training directions
TEST_BEYOND = [(float(az), 0.0, 0.8, 3.0) for az in range(-180, 180, 45)]    # larger and faster than any trained
FLIGHTS = 24             # recorded flights (most recent): even ones train, odd ones are held out
SEEDS = 2                # runs per test (Poisson input)
MIN_SPIKES = 1           # a neuron is active if it spiked this often in one training run (1: lossless on them)
EXPERIMENTS = [('LC4', 'left'), ('LPLC2', 'left'), ('LC4', 'right'), ('LPLC2', 'right')]  # validate.py's


# ---- circuits (circuit.flight_circuit: 'whole', 'active', 'h3', 'h4') -----------------------------------------

_W = {}  # per worker process


def _init(name: str, dt_ms: float):
    os.nice(10)  # batch work: the desktop (and a game) first
    c = circuit(name)
    _W['c'], _W['inputs'], _W['out'] = c, Inputs(c), outputs(c)[0]
    _W['brain'] = FastCircuit(len(c['ids']), c['pre'], c['post'], c['count'], c['inputs'], dt_ms=dt_ms)


def run(job: tuple) -> tuple:
    """(messages, form 'loom' / 'drive', seed) -> (spikes per neuron over the test, output spikes per tick).
    As train_readout.run: quiet stretches skipped and the brain reset after them."""
    msgs, form, seed = job
    brain, inputs, out = _W['brain'], _W['inputs'], _W['out']
    inputs.set_gain(RF_GAIN)
    brain.rng[0] = seed
    total = np.zeros(brain.n, np.int32)
    ticks = np.zeros((len(msgs), len(out)), np.int32)
    quiet, at_rest = QUIET_TICKS, True
    brain.reset()
    for k, m in enumerate(msgs):
        rates = inputs.rates({'loom': m['loom']} if form == 'loom' else {'drive': m['drive']})
        quiet = 0 if rates.any() else quiet + 1
        if quiet >= QUIET_TICKS:
            at_rest = True
            continue
        if at_rest:
            brain.reset()
            at_rest = False
        brain.set_rates(rates)
        counts = brain.run(TICK_MS)
        total += counts.astype(np.int32)
        ticks[k] = counts[out]
    return total, ticks


def _pool(name: str, procs: int, dt_ms: float = FLIGHT_DT_MS):
    return mp.get_context('fork').Pool(procs, initializer=_init, initargs=(name, dt_ms))


def procs_for(name: str, requested: int) -> int:
    """Workers: cores less two (the desktop), and ~0.6 GB each for the whole brain."""
    avail_gb = 0.0
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                avail_gb = int(line.split()[1]) / 1e6
    per = 0.6 if name == 'whole' else 0.1
    return max(1, min(requested, int((avail_gb - 2.0) // per), (os.cpu_count() or 2) - 2))


# ---- tests ------------------------------------------------------------------------------------------------

def tests(which: str) -> list:
    """[(label, messages, form)]: 'train' or 'test' (held out)."""
    out = []
    if which == 'train':
        out += [(f'sphere az {a:+.0f} el {e:.0f} r {r} v {v}', m, 'loom') for (a, e, r, v), m in zip(SYNTH, synthetic(SYNTH))]
    else:
        out += [(f'between az {a:+.0f}', m, 'loom') for (a, *_), m in zip(TEST_BETWEEN, synthetic(TEST_BETWEEN))]
        out += [(f'beyond az {a:+.0f}', m, 'loom') for (a, *_), m in zip(TEST_BEYOND, synthetic(TEST_BEYOND))]
    for i, rec in enumerate(flights(FLIGHTS)):
        if i % 2 == (0 if which == 'train' else 1):
            msgs = replay_eye(rec)
            out += [(f'flight {rec.stem} ({form})', msgs, form) for form in ('loom', 'drive')]
    return out


def run_tests(name: str, which: str, procs: int, keep_totals: bool) -> dict:
    t0 = time.time()
    ts = tests(which)
    jobs = [(m, form, 1000 + 10 * i + s) for i, (_, m, form) in enumerate(ts) for s in range(SEEDS)]
    labels = [label for label, _, _ in ts for _ in range(SEEDS)]
    n_procs = procs_for(name, procs)
    print(f'{name}: {len(ts)} {which} tests x {SEEDS} seeds, {n_procs} at a time...', flush=True)
    res = []
    with _pool(name, n_procs) as pool:
        for r in pool.imap(run, jobs, chunksize=1):
            res.append(r)
            if len(res) % max(1, len(jobs) // 10) == 0:
                print(f'  {len(res)} / {len(jobs)} ({time.time() - t0:.0f} s)', flush=True)
    print(f'  done in {time.time() - t0:.0f} s', flush=True)
    out = {'labels': np.array(labels), 'seeds': np.array([j[2] for j in jobs]),
           'ticks': np.concatenate([r[1] for r in res]), 'lengths': np.array([len(r[1]) for r in res])}
    if keep_totals:
        out['totals'] = np.stack([np.minimum(r[0], np.iinfo(np.int16).max).astype(np.int16) for r in res])
    return out


def split_ticks(d: dict) -> list:
    return np.split(d['ticks'], np.cumsum(d['lengths'])[:-1])


# ---- check: the whole brain in C against Brian2 -------------------------------------------------------------

def check(trials: int, procs: int) -> int:
    """validate.py's experiments (one eye's LC4 or LPLC2 at 150 Hz, 1 s) on the whole brain in C at the model's
    0.1 ms step, against the model repo's Brian2 runs (reproduce.py escape, results/): every neuron's rate."""
    from reproduce import RESULTS as BRIAN_RESULTS, rates
    from connectome import Connectome
    t0 = time.time()
    c = whole_circuit('783')
    print(f'whole brain: {len(c["ids"])} neurons, {len(c["pre"])} connections ({time.time() - t0:.0f} s to load)')
    conn = Connectome('783')
    pos_in = {int(f): k for k, f in enumerate(c['ids'][c['inputs']])}
    ok = True
    with _pool('whole', procs_for('whole', procs), dt_ms=0.1) as pool:
        for cell_type, side in EXPERIMENTS:
            stim = conn.cells(cell_type, side)
            r_in = np.zeros(len(c['inputs']))
            r_in[[pos_in[f] for f in stim]] = 150.0
            res = pool.map(_check_trial, [(r_in, 7000 + k) for k in range(trials)])
            wall = float(np.median([r[2] for r in res]))
            counts = np.stack([r[0] for r in res]).astype(float)
            mean, sd = counts.mean(0), counts.std(0, ddof=1)
            ref = rates(BRIAN_RESULTS / f'{cell_type}_{side}.parquet', trials, 1.0)
            b_mean = np.array([ref.rate.get(int(f), 0.0) for f in c['ids']])
            b_sd = np.array([ref['std'].get(int(f), 0.0) for f in c['ids']])
            active = (b_mean >= 1.0) | (mean >= 1.0)
            se = np.sqrt((sd ** 2 + b_sd ** 2) / trials)
            match = np.abs(mean - b_mean) <= np.maximum.reduce([0.1 * b_mean, 3.0 * se, np.ones_like(se)])
            gf = c['gf']
            gf_ok = bool(match[gf].all())
            ok &= gf_ok
            corr = np.corrcoef(mean[active], b_mean[active])[0, 1]
            print(f'{cell_type} {side}: {active.sum()} neurons fire >= 1 Hz in either; {100 * match[active].mean():.1f}% '
                  f'of them match Brian2 (10% / 3 SE / 1 Hz); rate correlation {corr:.4f}; giant fibers L/R '
                  f'C {mean[gf[0]]:.1f} / {mean[gf[1]]:.1f} Hz, Brian2 {b_mean[gf[0]]:.1f} / {b_mean[gf[1]]:.1f} Hz '
                  f'{"ok" if gf_ok else "DIFFERENT"}; {wall:.1f} s per simulated second per core', flush=True)
    rss = max(r[1] for r in res)
    print(f'memory per worker: {rss / 1e9:.2f} GB peak (the model repo\'s Brian2 run: 4.3 GB). Speed with '
          f'{min(trials, procs_for("whole", procs))} whole brains at once on this PC (they share its memory bandwidth)')
    print('\nTHE WHOLE BRAIN IN C MATCHES BRIAN2' if ok else '\nTHE WHOLE BRAIN IN C DIFFERS FROM BRIAN2')
    return 0 if ok else 1


def _check_trial(job: tuple) -> tuple:
    r_in, seed = job
    brain = _W['brain']
    brain.reset()
    brain.rng[0] = seed
    brain.set_rates(r_in)
    t = time.perf_counter()
    counts = brain.run(1000.0)
    return counts, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, time.perf_counter() - t


# ---- build: the active circuit --------------------------------------------------------------------------------

def build(min_spikes: int) -> int:
    from connectome import Connectome
    d = np.load(RESULTS / 'prune_train.npz')
    whole = whole_circuit('783')
    totals = d['totals']
    active = (totals >= min_spikes).any(axis=0)
    keep = active.copy()
    keep[whole['inputs']] = True
    out_pos = outputs(whole)[0]
    keep[out_pos] = True
    ids = [int(f) for f in whole['ids'][keep]]
    ACTIVE_IDS.write_text(
        '# The fly brain pruned by activity (flybrain/prune.py): FlyWire 783 ids of the neurons that spiked while\n'
        f'# the whole brain (Shiu et al. 2024 model) ran {len(totals)} looming tests (>= {min_spikes} spike in a run),\n'
        '# plus every LC4 / LPLC2 input and every output. circuit.active_circuit() builds the circuit from it.\n'
        '# Data: FlyWire (Dorkenwald et al. 2024), cell types Schlegel et al. 2024; CC-BY 4.0.\n'
        + ''.join(f'{f}\n' for f in ids))
    c = active_circuit()
    conn = Connectome('783')
    h3, h4 = (set(int(f) for f in escape_circuit('783', h, MIN_SYNAPSES)['ids']) for h in (3, 4))
    act = set(int(f) for f in whole['ids'][active])
    print(f'{len(totals)} training runs; {active.sum()} neurons spiked (>= {min_spikes} in a run); with every input '
          f'and output: {len(ids)} neurons, {len(c["pre"])} connections -> {ACTIVE_IDS.name}')
    for name, cut in (('3-hop cut (flown)', h3), ('4-hop cut', h4)):
        print(f'  vs the {name} ({len(cut)} neurons): {len(act & cut)} active in it, {len(cut - act)} of it never '
              f'spiked, {len(act - cut)} active outside it')
    lab = conn.labels
    outside = sorted(act - h3, key=lambda f: -int(totals[:, np.searchsorted(whole['ids'], f)].sum()))
    print(f'  most active neurons outside the 3-hop cut (spikes over all training runs):')
    for f in outside[:15]:
        row = lab.loc[f] if f in lab.index else None
        kind = f'{row.cell_type} ({row.side}), {row.super_class}' if row is not None else '?'
        print(f'    {f}: {kind}: {int(totals[:, np.searchsorted(whole["ids"], f)].sum())}')
    return 0


# ---- evaluate: held-out tests -----------------------------------------------------------------------------------

def compare(ref: dict, other: dict) -> dict:
    """Paired, test by test: alarms (the policy's: smoothed giant fiber >= 50 Hz), their timing, the outputs."""
    hit = miss = extra = same = 0
    lag, err = [], []
    for a, b in zip(split_ticks(ref), split_ticks(other)):
        same += int(np.array_equal(a, b))
        oa, ob = alarm_onsets(a), alarm_onsets(b)
        for i in oa:
            near = ob[np.abs(ob - i) <= 20]
            if len(near):
                hit += 1
                lag.append((near[np.argmin(np.abs(near - i))] - i) * TICK_MS)
            else:
                miss += 1
        extra += sum(1 for j in ob if not np.any(np.abs(oa - j) <= 20))
        ra, rb = smooth(a.astype(float)), smooth(b.astype(float))
        busy = ra.sum(1) >= 30.0
        if busy.any():
            err.append(np.abs(rb[busy] - ra[busy]).sum(1) / np.maximum(ra[busy].sum(1), 1.0))
    return {'hit': hit, 'miss': miss, 'extra': extra, 'lag': np.array(lag), 'same': same,
            'out_err': np.concatenate(err) if err else np.zeros(0)}


def evaluate(procs: int, reuse: bool = False) -> int:
    t0 = time.time()
    saved = RESULTS / 'prune_test.npz'
    if reuse and saved.is_file():
        with np.load(saved) as d:
            res = {n: {k: d[f'{n}_{k}'] for k in ('labels', 'seeds', 'ticks', 'lengths')}
                   for n in ('whole', 'active', 'h3', 'h4')}
    else:
        res = {name: run_tests(name, 'test', procs, keep_totals=False) for name in ('whole', 'active', 'h3', 'h4')}
    groups = {'between': 'between', 'beyond': 'beyond', 'flights (loom)': '(loom)', 'flights (drive)': '(drive)'}
    sizes = {name: len(circuit(name)['ids']) for name in res}
    print(f'\nheld-out tests, each circuit vs the whole brain (paired: same input spikes; {SEEDS} seeds each):')
    for group, key in groups.items():
        sel = np.array([key in label for label in res['whole']['labels']])
        if not sel.any():
            continue
        print(f'\n  {group}: {sel.sum()} runs')
        for name in ('active', 'h3', 'h4'):
            pick = lambda d: {**d, 'ticks': np.concatenate([t for t, s in zip(split_ticks(d), sel) if s]),  # noqa: E731
                              'lengths': d['lengths'][sel]}
            r = compare(pick(res['whole']), pick(res[name]))
            lag = (f'{np.mean(np.abs(r["lag"])):.0f} ms off on average, at most {np.abs(r["lag"]).max():.0f}'
                   if len(r['lag']) else '-')
            e = r['out_err']
            print(f'    {name:6s} ({sizes[name]:6d} neurons): {r["same"]:2d} of {sel.sum()} runs identical; alarms '
                  f'{r["hit"]} as the whole brain, {r["miss"]} missed, {r["extra"]} extra; timing {lag}; outputs off by '
                  f'{100 * np.median(e):.0f}% (median), {100 * np.percentile(e, 90):.0f}% (90th percentile)')
    print(f'\n({time.time() - t0:.0f} s)')
    if not reuse:
        np.savez_compressed(saved, **{f'{n}_{k}': v for n, d in res.items() for k, v in d.items()})
    return 0


def lossless(procs: int, n_tests: int = 6) -> int:
    """The active circuit on training tests: every neuron it kept must spike exactly as in the whole brain."""
    d = np.load(RESULTS / 'prune_train.npz')
    whole, act = whole_circuit('783'), circuit('active')
    ts = tests('train')[:n_tests]
    jobs = [(m, form, 1000 + 10 * i) for i, (_, m, form) in enumerate(ts)]
    with _pool('active', procs_for('active', procs)) as pool:
        res = pool.map(run, jobs)
    where = np.searchsorted(whole['ids'], act['ids'])
    same = all(np.array_equal(r[0], d['totals'][SEEDS * i][where]) for i, r in enumerate(res))
    print(f'active circuit on {n_tests} training tests: every kept neuron\'s spikes '
          f'{"IDENTICAL to the whole brain" if same else "DIFFER from the whole brain"}')
    return 0 if same else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('steps', nargs='*', choices=['check', 'run', 'build', 'evaluate'],
                    help='default: all four, in order')
    ap.add_argument('--procs', type=int, default=12, help='at most this many processes (memory permitting)')
    ap.add_argument('--trials', type=int, default=10, help='(check) trials per experiment')
    ap.add_argument('--min-spikes', type=int, default=MIN_SPIKES, help='(build) spikes in one run to count as active')
    ap.add_argument('--reuse', action='store_true', help='(evaluate) analyse the saved held-out runs again')
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    rc = 0
    for step in args.steps or ['check', 'run', 'build', 'evaluate']:
        if step == 'check':
            rc |= check(args.trials, args.procs)
        elif step == 'run':
            d = run_tests('whole', 'train', args.procs, keep_totals=True)
            np.savez_compressed(RESULTS / 'prune_train.npz', **d)
        elif step == 'build':
            rc |= build(args.min_spikes)
            rc |= lossless(args.procs)
        elif step == 'evaluate':
            rc |= evaluate(args.procs, args.reuse)
    return rc


if __name__ == '__main__':
    sys.exit(main())
