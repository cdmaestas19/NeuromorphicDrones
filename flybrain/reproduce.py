"""Check the brain model on this PC against published results, before the drone relies on it.

    flypilot\\brain_reproduce.cmd                 both checks, 10 trials each
    flypilot\\brain_reproduce.cmd sugar           1. the model paper's own result
    flypilot\\brain_reproduce.cmd escape          2. the circuit the drone will use
    flypilot\\brain_reproduce.cmd --trials 30     more trials (the paper used 30)

1. sugar (Shiu et al. 2024, Nature 634:210, Fig. 1): activating the 21 sugar-sensing gustatory neurons
   (Poisson input, 150 Hz) drives MN9, the motor neuron that extends the proboscis. The model repo ships
   the authors' own 30-trial result (FlyWire 630). Pass: our MN9 rate within 10% of theirs.
2. escape: activating one eye's looming detectors (LC4, or LPLC2) should drive the giant fiber DNp01, the
   command neuron for the escape takeoff, since both synapse directly onto it (von Reyn et al. 2014,
   Nat Neurosci 17:962; Ache et al. 2019, Curr Biol 29:1073). Pass: the giant fiber fires. FlyWire 783,
   the version the cell-type labels are for.

Every trial builds the whole brain (~139k neurons, ~15M connections): 3 GB (FlyWire 630) to 4.3 GB (783) of
memory, ~5 s per simulated second on one core, plus a one-time C++ compile (~30 s). Trials run in parallel,
but never more at once than free memory allows (running one per core is what makes a 16 GB PC run out).
--reuse analyses results already in results/ without simulating them again.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

from connectome import BRAIN_MODEL_DIR, GIANT_FIBER, Connectome

sys.path.insert(0, str(BRAIN_MODEL_DIR))
import model  # noqa: E402  (the brain model repo's Brian2 model: model.py)

RESULTS = Path(__file__).resolve().parent / 'results'
GB_PER_TRIAL = {'630': 3.5, '783': 5.0}  # measured peaks ~3.0 and ~4.3 GB, plus headroom

# Shiu et al. 2024, example.ipynb in the model repo: the sugar-sensing gustatory receptor neurons and MN9.
SUGAR_GRNS = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
    720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
    720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
MN9 = 720575940660219265


def parallel_trials(requested: int, version: str) -> int:
    """How many trials to run at once: limited by free memory and by cores (one left for the desktop)."""
    avail_gb = 0.0
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                avail_gb = int(line.split()[1]) / 1e6
    return max(1, min(requested, int(avail_gb // GB_PER_TRIAL[version]), (os.cpu_count() or 2) - 1))


def rates(path: Path, n_run: int, t_run_s: float) -> pd.DataFrame:
    """Mean and std firing rate (Hz) per FlyWire id over the trials of one run_exp result file."""
    df = pd.read_parquet(path)
    counts = (df.groupby(['flywire_id', 'trial']).size().unstack(fill_value=0)
              .reindex(columns=range(n_run), fill_value=0))
    return pd.DataFrame({'rate': counts.mean(axis=1) / t_run_s, 'std': counts.std(axis=1) / t_run_s})


def run(name: str, excite: list, conn: Connectome, trials: int, opts) -> Path:
    path = RESULTS / f'{name}.parquet'
    if opts.reuse and path.is_file():
        print(f'(reusing {path.name})')
        return path
    procs = parallel_trials(opts.parallel, conn.version)
    print(f'{name}: {trials} trials, {procs} at a time (limited by free memory and cores)', flush=True)
    params = dict(model.default_params, n_run=trials)
    model.run_exp(exp_name=name, neu_exc=excite, path_res=RESULTS, path_comp=conn.path_comp,
                  path_con=conn.path_con, params=params, n_proc=procs, force_overwrite=True)
    return path


def check_sugar(trials: int, opts) -> bool:
    print('\n=== 1. sugar-sensing neurons -> MN9 (Shiu et al. 2024; FlyWire 630)')
    conn = Connectome('630')
    ours = rates(run('sugarR', SUGAR_GRNS, conn, trials, opts), trials, 1.0)
    r, sd = ours.rate.get(MN9, 0.0), ours['std'].get(MN9, 0.0)
    print(f'MN9 here: {r:.1f} +- {sd:.1f} Hz over {trials} trials. Stored results in the model repo:')
    ok = False
    # The paper's original run, and the fork author's re-run with current software (Brian2 2.x, numpy 2):
    # those two differ by ~10% themselves (93 vs 84 Hz), a software-version effect.
    for label, sub in (('paper (original software)', 'example'), ('re-run, current software', 'example_cpu_fresh')):
        ref = rates(BRAIN_MODEL_DIR / 'results' / sub / 'sugarR.parquet', 30, 1.0)
        r_ref, sd_ref = ref.rate.get(MN9, 0.0), ref['std'].get(MN9, 0.0)
        close = r_ref > 0 and abs(r - r_ref) / r_ref < 0.10
        ok |= close
        print(f'  {label:27s} {r_ref:5.1f} +- {sd_ref:.1f} Hz over 30 trials: {"within 10%" if close else "more than 10% off"}')
    print('-> PASS' if ok else '-> FAIL')
    return ok


def check_escape(trials: int, opts) -> bool:
    print('\n=== 2. looming detectors -> giant fiber DNp01 (FlyWire 783)')
    conn = Connectome('783')
    gf = {side: conn.cells(GIANT_FIBER, side)[0] for side in ('left', 'right')}
    ok = True
    for cell_type in ('LC4', 'LPLC2'):
        for side in ('left', 'right'):
            ids = conn.cells(cell_type, side)
            r = rates(run(f'{cell_type}_{side}', ids, conn, trials, opts), trials, 1.0)
            gf_rates = {s: r.rate.get(f, 0.0) for s, f in gf.items()}
            fired = gf_rates[side] > 0
            ok &= fired
            others = r.drop(index=ids, errors='ignore').sort_values('rate', ascending=False).head(6)
            top = ', '.join(f'{conn.name(f)} {v:.0f}' for f, v in others.rate.items())
            print(f'{cell_type} {side:5s} ({len(ids)} neurons at 150 Hz): giant fiber left {gf_rates["left"]:5.1f} Hz, '
                  f'right {gf_rates["right"]:5.1f} Hz -> {"PASS" if fired else "FAIL"}')
            print(f'    most active downstream (Hz): {top}')
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('checks', nargs='*', choices=['sugar', 'escape'], help='which checks (default: both)')
    ap.add_argument('--trials', type=int, default=10, help='trials per experiment (the paper used 30)')
    ap.add_argument('--parallel', type=int, default=8, help='at most this many trials at once (memory permitting)')
    ap.add_argument('--reuse', action='store_true', help='analyse result files already in results/ instead of rerunning')
    args = ap.parse_args()
    args.checks = args.checks or ['sugar', 'escape']
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    results = []
    if 'sugar' in args.checks:
        results.append(check_sugar(args.trials, args))
    if 'escape' in args.checks:
        results.append(check_escape(args.trials, args))
    print(f'\n{"ALL PASSED" if all(results) else "SOME CHECKS FAILED"} in {time.time() - t0:.0f} s')
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
