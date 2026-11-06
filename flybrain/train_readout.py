"""The small trained readout: where the threat is, from the brain's descending neurons.

With each looming detector driven from its own receptive field (brain_server.Inputs 'loom'), the descending
neurons carry where a threat is (directions.py). The flight's policy (flypilot/policies.py FlyBrainRFPolicy)
reads it with this readout: a linear map (ridge regression) from the output neurons' smoothed spike rates to
the threat's bearing (cos, sin; body frame). It is trained on recorded flights replayed offline through the
same eye code (flypilot/eye.py Looming) and the same brain input (brain_server.Inputs) and engine as in
flight, plus synthetic looming from all round (spheres as in directions.py), so every direction is covered.
The label is where the looming is: the looming-weighted mean direction of what the brain was given,
smoothed as the rates are. It is checked on flights it was not trained on, against the two-channel brain's
own side rule (away from the stronger steering neurons).

    bash py.sh train_readout.py               writes ../flypilot/models/brain_readout_<circuit>.npz
    bash py.sh train_readout.py --circuit h3  for another circuit than the flight's (brain_server.FLIGHT_CIRCUIT)
    bash py.sh train_readout.py --equalize    the gain by azimuth (brain_server.RF_PROFILE) that makes a threat
                                              from any side alarm the giant fibers as early as one head-on
    bash py.sh train_readout.py --calibrate   then: when the giant fibers alarm with two channels and with
                                              receptive fields at several gains (sets brain_server.RF_GAIN)
    (the readout last: it learns the brain's answers to the input as set)

Model and data: Shiu et al. 2024; FlyWire (Dorkenwald et al. 2024; Schlegel et al. 2024; Matsliah et al.
2024), CC-BY 4.0.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

from brain_server import FLIGHT_CIRCUIT, FLIGHT_DT_MS, RF_GAIN, TICK_MS, Inputs, outputs
from circuit import CIRCUITS, flight_circuit
from fastlif import FastCircuit

HERE = Path(__file__).resolve().parent
FLYPILOT = HERE.parent / 'flypilot'
sys.path.insert(0, str(FLYPILOT))
from eye import Looming, elevations  # noqa: E402

RECORDINGS = FLYPILOT / 'recordings'
EYE_GAIN = (1.35, 1.0)        # policies.FlyBrainPolicy: two channels, left and right
SMOOTH = 0.35                 # policies.FlyBrainPolicy: the newest tick's weight in the smoothed rates
ALARM_HZ = 50.0               # the supervisor's enter threshold x GF_FULL_HZ
QUIET_TICKS = 40              # this long (2 s) without input, the network is at rest: skipped (and reset)
MIN_OUT_HZ = 30.0             # training ticks: the outputs together fire at least this much...
MIN_LOOM = 0.2                # ...and the smoothed looming (1/s, summed) is at least this
LAMBDAS = (1.0, 10.0, 100.0, 1000.0)
STEER_DNS = ('DNp02', 'DNp04', 'DNp06', 'DNp11')  # policies.FlyBrainPolicy's side rule
SYNTH = [(float(az), el, r, v) for az in range(-180, 180, 30) for el in (0.0, 15.0)
         for r in (0.2, 0.5) for v in (1.0, 2.0)]   # direction, sphere radius m, closing speed m/s
SYNTH_START_M = 4.0


# ---- inputs: recorded flights replayed through the eye, and synthetic looming ----------------------------

def flights(limit: int) -> list[Path]:
    """Recorded flights with the full 25-row eye, most recent first."""
    out = []
    for p in sorted(RECORDINGS.glob('*.eye.npz'), key=lambda p: p.stat().st_mtime, reverse=True):
        rec = p.with_name(p.name.replace('.eye.npz', '.jsonl'))
        if rec.is_file():
            with np.load(p) as e:
                if e['dist'].ndim == 3 and e['dist'].shape[1] == 25:
                    out.append(rec)
        if len(out) >= limit:
            break
    return out


def replay_eye(rec: Path) -> list[dict]:
    """Per decision of the flight: what its fly-brain policy would send (both forms), as the flight decided
    (each decision on the newest eye reading by then, as flypilot/bench.py)."""
    rows = [json.loads(line) for line in rec.read_text().splitlines() if line.strip()]
    e = np.load(rec.with_name(rec.name.replace('.jsonl', '.eye.npz')))
    readings = [(float(st), tuple(float(v) for v in p), d.astype(np.float32))
                for st, p, d in zip(e['sim_t'], e['pos'], e['dist'])]
    looming, k, last, out = Looming(), -1, None, []
    for r in rows:
        while k + 1 < len(readings) and e['t'][k + 1] <= r['t']:
            k += 1
        if k >= 0 and k != last:
            last = k
            stop = r.get('route_stop_m')
            looming.update(readings[k], r['yaw_deg'], math.inf if stop is None else stop)
        drive = [min(d * g, 1.0) for d, g in zip(looming.drive(), EYE_GAIN)]
        out.append({'drive': drive, 'loom': looming.loom_points()})
    return out


def synthetic(stimuli: list = SYNTH) -> list[list[dict]]:
    """Spheres approaching (stimuli: (azimuth, elevation, radius m, speed m/s)), on the drone's eye grid, as
    the eye gives them: only rows from eye.MIN_ELEVATION_DEG up. Each a list of messages, one per tick."""
    from eye import MIN_ELEVATION_DEG
    el_grid = elevations(25)
    az_grid = np.arange(-180.0, 180.0, 5.0)
    az, el = np.meshgrid(az_grid, el_grid)
    keep = el >= MIN_ELEVATION_DEG
    az, el = az[keep], el[keep]
    u = np.stack([np.cos(np.radians(el)) * np.cos(np.radians(az)), np.cos(np.radians(el)) * np.sin(np.radians(az)),
                  np.sin(np.radians(el))], axis=1)
    out = []
    for a0, e0, radius, speed in stimuli:
        d0 = np.array([math.cos(math.radians(e0)) * math.cos(math.radians(a0)),
                       math.cos(math.radians(e0)) * math.sin(math.radians(a0)), math.sin(math.radians(e0))])
        psi = np.degrees(np.arccos(np.clip(u @ d0, -1.0, 1.0)))
        msgs, t = [], 0.0
        while SYNTH_START_M - speed * t > radius + 0.05:
            d = SYNTH_START_M - speed * t
            seen = psi <= math.degrees(math.asin(radius / d))
            rate = speed / (d - radius)
            pts = np.column_stack([az[seen], el[seen], np.full(seen.sum(), rate)])
            msgs.append({'drive': [0.0, 0.0], 'loom': pts})
            t += TICK_MS / 1000.0
        out.append(msgs + [{'drive': [0.0, 0.0], 'loom': np.zeros((0, 3))}] * 10)
    return out


# ---- the brain, offline -------------------------------------------------------------------------------------

_W = {}  # per worker process: the circuit, its inputs and outputs


def readout_path(circuit: str) -> Path:
    """The readout trained on this circuit's outputs (policies.FlyBrainRFPolicy loads the one for the brain
    server's circuit)."""
    return FLYPILOT / 'models' / f'brain_readout_{circuit}.npz'


def _init(seed_base: int, circuit: str = FLIGHT_CIRCUIT):
    c = flight_circuit(circuit)
    _W['brain'] = FastCircuit(len(c['ids']), c['pre'], c['post'], c['count'], c['inputs'], dt_ms=FLIGHT_DT_MS,
                              seed=seed_base + os.getpid())
    _W['inputs'] = Inputs(c)
    _W['out'] = outputs(c)[0]


def run(job: tuple) -> np.ndarray:
    """(messages, form 'loom' or 'drive', gain[, profile]) -> output spike counts per tick (ticks x outputs)."""
    msgs, form, gain = job[:3]
    brain, inputs, out = _W['brain'], _W['inputs'], _W['out']
    inputs.set_gain(gain, *job[3:])
    counts = np.zeros((len(msgs), len(out)), np.int32)
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
        counts[k] = brain.run(TICK_MS)[out]
    return counts


def smooth(x: np.ndarray) -> np.ndarray:
    """The policy's smoothing, per tick (rows), of rates in Hz (counts / tick)."""
    out, s = np.zeros(x.shape, float), np.zeros(x.shape[1:], float)
    for k in range(len(x)):
        s = s + SMOOTH * (x[k] / (TICK_MS / 1000.0) - s)
        out[k] = s
    return out


def alarm_onsets(counts: np.ndarray) -> np.ndarray:
    gf = smooth(counts[:, :2].astype(float)).max(axis=1) >= ALARM_HZ
    return np.flatnonzero(gf & ~np.r_[False, gf[:-1]])


def labels(msgs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per tick: the looming-weighted mean direction of the input (body frame unit vector, x forward, y left;
    smoothed as the rates are) and the smoothed total looming."""
    v, total = np.zeros((len(msgs), 2)), np.zeros(len(msgs))
    sv, st = np.zeros(2), 0.0
    for k, m in enumerate(msgs):
        pts = m['loom']
        now = np.zeros(2)
        if len(pts):
            a = np.radians(pts[:, 0])
            now = np.array([(pts[:, 2] * np.cos(a)).sum(), (pts[:, 2] * np.sin(a)).sum()])
        sv = sv + SMOOTH * (now - sv)
        st = st + SMOOTH * ((pts[:, 2].sum() if len(pts) else 0.0) - st)
        v[k], total[k] = sv, st
    return v, total


def features(rates_hz: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(rates_hz, 0.0))


def fit(x: np.ndarray, y: np.ndarray, lam: float) -> tuple:
    mu, sd = x.mean(0), x.std(0) + 1e-6
    a = np.column_stack([(x - mu) / sd, np.ones(len(x))])
    reg = lam * np.eye(a.shape[1])
    reg[-1, -1] = 0.0
    return mu, sd, np.linalg.solve(a.T @ a + reg, a.T @ y)


def predict(model: tuple, x: np.ndarray) -> np.ndarray:
    mu, sd, w = model
    return np.column_stack([(x - mu) / sd, np.ones(len(x))]) @ w


def angle_err(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.abs((np.degrees(np.arctan2(p[:, 1], p[:, 0]) - np.arctan2(y[:, 1], y[:, 0])) + 180.0) % 360.0 - 180.0)


EQUALIZE_AZ = (0.0, 45.0, 90.0, 135.0, 180.0)
EQUALIZE_TRIALS = 8


def equalize(procs: int, circuit: str) -> int:
    """For each azimuth (each side), the gain multiplier at which a sphere (directions.py's: 0.25 m at
    1.5 m/s) alarms the giant fibers as long before contact as one head-on at multiplier 1 (median over
    trials, bisection in log gain); printed as brain_server.RF_PROFILE."""
    flat = (tuple((a, 1.0) for a in EQUALIZE_AZ),) * 2
    pool = mp.get_context('fork').Pool(procs, initializer=_init, initargs=(777, circuit))
    contact = (SYNTH_START_M - 0.25) / 1.5

    def ttc(az: float, m: float) -> float:
        msgs = synthetic([(az, 0.0, 0.25, 1.5)])[0]
        runs = pool.map(run, [(msgs, 'loom', (RF_GAIN[0] * m, RF_GAIN[1] * m), flat)] * EQUALIZE_TRIALS)
        t = [contact - (o[0] + 1) * TICK_MS / 1000.0 if len(o := alarm_onsets(c)) else 0.0 for c in runs]
        return float(np.median(t))

    target = ttc(0.0, 1.0)
    print(f'head-on at RF_GAIN {RF_GAIN}: the giant fibers alarm {target:.2f} s before contact')
    prof = {}
    for side in (1.0, -1.0):
        for az in EQUALIZE_AZ[1:]:
            lo, hi = 0.0, math.log(64.0)  # log multiplier
            for _ in range(7):
                mid = 0.5 * (lo + hi)
                if ttc(side * az, math.exp(mid)) >= target:
                    hi = mid
                else:
                    lo = mid
            prof[(side, az)] = math.exp(hi)
            print(f'  {"left" if side > 0 else "right"} {az:3.0f} deg: x{math.exp(hi):.2f} '
                  f'(alarm {ttc(side * az, math.exp(hi)):.2f} s; x1: {ttc(side * az, 1.0):.2f} s)', flush=True)
    side = lambda k: ((0.0, 1.0),) + tuple((az, round(prof[(k, az)], 2)) for az in EQUALIZE_AZ[1:])  # noqa: E731
    print(f'RF_PROFILE = ({side(1.0)},\n              {side(-1.0)})  (left eye, right eye)')
    return 0


# ---- main -----------------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--calibrate', action='store_true', help='compare giant-fiber alarms across input gains')
    ap.add_argument('--gains', type=float, nargs='+', default=[16.0, 24.0, 32.0, 48.0, 64.0],
                    help='(--calibrate) the right eye\'s gains to try; the left eye\'s is x1.35')
    ap.add_argument('--flights', type=int, default=80, help='recorded flights to use (most recent first)')
    ap.add_argument('--procs', type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument('--circuit', choices=CIRCUITS, default=FLIGHT_CIRCUIT,
                    help=f'the brain circuit (brain_server.py --circuit; default {FLIGHT_CIRCUIT})')
    ap.add_argument('--equalize', action='store_true',
                    help='the gain profile by azimuth (brain_server.RF_PROFILE) that alarms the giant fibers as '
                         'early for a threat from any side as for one head-on')
    args = ap.parse_args()
    t0 = time.time()
    if args.equalize:
        return equalize(args.procs, args.circuit)
    recs = flights(args.flights)
    print(f'{len(recs)} recorded flights; replaying their eyes...', flush=True)
    msgs = [replay_eye(r) for r in recs]
    print(f'  {sum(map(len, msgs))} decisions ({time.time() - t0:.0f} s)', flush=True)
    names = outputs(flight_circuit(args.circuit))[1]
    pool = mp.get_context('fork').Pool(args.procs, initializer=_init, initargs=(12345, args.circuit))

    if args.calibrate:
        two = pool.map(run, [(m, 'drive', RF_GAIN) for m in msgs])
        ref = [alarm_onsets(c) for c in two]
        print(f'two channels: {sum(map(len, ref))} alarms; on the square route(s): '
              f'{sum(len(o) for o, r in zip(ref, recs) if "square" in r.name)}')
        for g in args.gains:
            gain = (g * EYE_GAIN[0], g * EYE_GAIN[1])
            rf = pool.map(run, [(m, 'loom', gain) for m in msgs])
            lead, missed, extra, sq = [], 0, 0, 0
            for o_ref, c, r in zip(ref, rf, recs):
                o = alarm_onsets(c)
                sq += len(o) if 'square' in r.name else 0
                for i in o_ref:
                    near = o[np.abs(o - i) <= 20]
                    if len(near):
                        lead.append((i - near[0]) * TICK_MS / 1000.0)
                    else:
                        missed += 1
                extra += sum(1 for j in o if not np.any(np.abs(o_ref - j) <= 20))
            print(f'receptive fields, gain {g:4.0f} (left x{EYE_GAIN[0]}): {sum(len(alarm_onsets(c)) for c in rf)} '
                  f'alarms; vs two channels: median {np.median(lead) if lead else math.nan:+.2f} s earlier, '
                  f'{missed} missed, {extra} extra; on square: {sq} ({time.time() - t0:.0f} s)', flush=True)
        return 0

    print(f'running the brain (receptive fields, gain {RF_GAIN}) on the flights and on {len(SYNTH)} synthetic '
          f'approaches...', flush=True)
    synth = synthetic()
    rf = pool.map(run, [(m, 'loom', RF_GAIN) for m in msgs + synth])
    two = pool.map(run, [(m, 'drive', RF_GAIN) for m in msgs])
    xs, ys, groups, alarm = [], [], [], []
    for g, (m, c) in enumerate(zip(msgs + synth, rf)):
        rates = smooth(c.astype(float))
        v, total = labels(m)
        use = (rates.sum(1) >= MIN_OUT_HZ) & (total >= MIN_LOOM) & (np.hypot(v[:, 0], v[:, 1]) > 1e-9)
        xs.append(features(rates[use]))
        ys.append(v[use] / np.hypot(v[use, 0], v[use, 1])[:, None])
        groups.append(np.full(use.sum(), g))
        alarm.append(rates[use, :2].max(1) >= ALARM_HZ)
    x, y, groups, alarm = np.vstack(xs), np.vstack(ys), np.concatenate(groups), np.concatenate(alarm)
    real = groups < len(msgs)
    print(f'  {len(x)} training ticks ({real.sum()} from flights, {(~real).sum()} synthetic; {alarm[real].sum()} '
          f'flight ticks at alarm level) ({time.time() - t0:.0f} s)', flush=True)

    # Cross-validation by flight: 5 folds of the recorded flights; synthetic data always in training.
    fold = np.where(real, groups % 5, -1)
    best = None
    for lam in LAMBDAS:
        err = []
        for f in range(5):
            tr, te = fold != f, (fold == f) & alarm
            if te.any():
                err.append(angle_err(predict(fit(x[tr], y[tr], lam), x[te]), y[te]))
        e = float(np.median(np.concatenate(err)))
        print(f'  ridge {lam:6.0f}: held-out flights, at alarm: median bearing error {e:.0f} deg')
        if best is None or e < best[1]:
            best = (lam, e)
    lam = best[0]
    pred = np.zeros_like(y)
    for f in range(5):
        tr, te = fold != f, fold == f
        if te.any():
            pred[te] = predict(fit(x[tr], y[tr], lam), x[te])
    te = real & alarm
    side = te & (np.abs(y[:, 1]) >= math.sin(math.radians(15)))
    fb = te & (np.abs(y[:, 0]) >= math.sin(math.radians(15)))
    print(f'\nreadout (ridge {lam:g}) on flights it was not trained on, at alarm level ({te.sum()} ticks):')
    print(f'  bearing error: median {np.median(angle_err(pred[te], y[te])):.0f} deg, 90% within '
          f'{np.percentile(angle_err(pred[te], y[te]), 90):.0f} deg')
    print(f'  threat left or right (>= 15 deg off the axis): {100 * np.mean(np.sign(pred[side, 1]) == np.sign(y[side, 1])):.0f}%'
          f' of {side.sum()}')
    print(f'  threat ahead or behind (>= 15 deg off the side axis): '
          f'{100 * np.mean(np.sign(pred[fb, 0]) == np.sign(y[fb, 0])):.0f}% of {fb.sum()}')

    # The two-channel brain's own side rule on the same flights (same label, its own alarm ticks).
    steer = [[i for i, n in enumerate(names) if n.split(' ')[0] in STEER_DNS and side_ in n]
             for side_ in ('(left)', '(right)')]
    hits = tot = 0
    for m, c in zip(msgs, two):
        rates = smooth(c.astype(float))
        v, total = labels(m)
        at = (rates[:, :2].max(1) >= ALARM_HZ) & (total >= MIN_LOOM) & (np.abs(v[:, 1]) >= math.sin(math.radians(15))
                                                                        * np.hypot(v[:, 0], v[:, 1]))
        sl, sr = rates[at][:, steer[0]].sum(1), rates[at][:, steer[1]].sum(1)
        # the threat is on the side of the stronger steering neurons (the policy escapes the other way)
        hits += int(np.sum(np.where(sl >= sr, 1.0, -1.0) == np.sign(v[at, 1])))
        tot += int(at.sum())
    print(f'  for comparison, the two-channel brain\'s steering neurons: threat left or right {100 * hits / max(tot, 1):.0f}%'
          f' of {tot}')

    model = fit(x, y, lam)
    path = readout_path(args.circuit)
    path.parent.mkdir(exist_ok=True)
    np.savez(path, names=np.array(names), circuit=args.circuit, mu=model[0], sd=model[1], w=model[2], smooth=SMOOTH,
             gain=np.array(RF_GAIN), ridge=lam, ticks=len(x), flights=len(msgs),
             median_err_deg=float(np.median(angle_err(pred[te], y[te]))), trained=time.strftime('%Y-%m-%d'))
    print(f'\nreadout: {path} ({len(names)} outputs -> bearing; {time.time() - t0:.0f} s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
