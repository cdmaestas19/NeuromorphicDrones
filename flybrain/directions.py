"""Does the brain know where a threat is? Looming from 24 directions through the escape subcircuit, read out
from its descending neurons.

The flight drives the brain through two channels: every LC4 / LPLC2 neuron of an eye at that eye's looming
drive (flypilot/eye.py), so the brain can't know where on that side a threat is. retinotopy.py gives each of
those neurons its own receptive field from the connectome. This compares, for the same looming stimuli:
  two channels   as the flight does today (eye.py's 30-degree fields, rows from -10 degrees up)
  retinotopic    each neuron driven by the looming inside its own receptive field
  shuffled       the same receptive fields dealt out at random among each eye's neurons of a type: the same
                 input statistics with the connectome's map broken (the control: does *its* wiring carry
                 direction, or would any map do?)
Stimulus: a sphere (RADIUS_M) approaching at SPEED_MPS from START_M, on the drone's eye grid (5 degrees);
looming = speed / distance where the sphere covers a direction. Readout: each output neuron's spikes (both
giant fibers and the other descending neurons) over WINDOW_TTC before contact, decoded by a linear readout
(ridge regression) trained on the other trials (cross-validated by trial): left/right, front/back, azimuth.
The retinotopic input's gain is calibrated first so a head-on threat alarms the giant fibers as early as with
two channels.

    bash py.sh directions.py [--trials 8]

Model and data: Shiu et al. 2024; FlyWire (Dorkenwald et al. 2024; Schlegel et al. 2024; Matsliah et al.
2024), CC-BY 4.0.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

from brain_server import FLIGHT_DT_MS, MAX_RATE_HZ, TICK_MS
from circuit import MIN_SYNAPSES, escape_circuit
from fastlif import FastCircuit
from retinotopy import receptive_fields, rf_weights

EYE_AZ = np.arange(-180.0, 180.0, 5.0)        # the drone's eye (crazysim.py CompoundEye), body frame
EYE_EL = np.linspace(-90.0, 30.0, 25)
LOOM_REF = 5.0                                # eye.py: looming rate (1/s) for full drive
FIELD_DEG, OVERLAP_DEG, MIN_ELEVATION_DEG = 30.0, 15.0, -10.0  # eye.py's two channels
EYE_GAIN = (1.35, 1.0)                        # policies.FlyBrainPolicy (left, right)
GF_ALARM_HZ, SMOOTH = 50.0, 0.35              # policies.FlyBrainPolicy: AVOID at 50 Hz, smoothing
RADIUS_M, SPEED_MPS, START_M = 0.25, 1.5, 4.0
DIRECTIONS = [(float(az), el) for el in (0.0, -20.0) for az in range(-150, 181, 30)]
WINDOW_TTC = (1.25, 0.25)                     # readout window: seconds before contact
GAINS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
RESULTS = Path(__file__).resolve().parent / 'results'


def angle_to(az: float, el: float) -> np.ndarray:
    """Angle (deg) from (az, el) to every direction of the eye grid (rows, columns)."""
    a, e = np.radians(EYE_AZ)[None, :], np.radians(EYE_EL)[:, None]
    a0, e0 = math.radians(az), math.radians(el)
    c = np.sin(e) * math.sin(e0) + np.cos(e) * math.cos(e0) * np.cos(a - a0)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def looming(direction: tuple, t: float, psi: np.ndarray) -> np.ndarray:
    """Looming rate (1/s) per direction of the eye grid at time t."""
    d = START_M - SPEED_MPS * t
    alpha = math.degrees(math.asin(min(RADIUS_M / d, 1.0)))
    return np.where(psi <= alpha, SPEED_MPS / max(d - RADIUS_M, 0.05), 0.0)


def two_channel(loom: np.ndarray, left: np.ndarray) -> np.ndarray:
    """eye.py's drive: per azimuth the most looming row from MIN_ELEVATION_DEG up; 30-degree fields (90th
    percentile); each eye its most looming field; every neuron of an eye at its eye's drive x EYE_GAIN."""
    column = loom[EYE_EL >= MIN_ELEVATION_DEG].max(axis=0)
    centers = np.arange(-180.0 + FIELD_DEG / 2, 180.0, FIELD_DEG)
    rates = np.array([np.percentile(column[np.abs((EYE_AZ - c + 180.0) % 360.0 - 180.0) <= FIELD_DEG / 2], 90)
                      for c in centers])
    drive = (rates[centers >= -OVERLAP_DEG].max() / LOOM_REF, rates[centers <= OVERLAP_DEG].max() / LOOM_REF)
    d = (min(drive[0] * EYE_GAIN[0], 1.0), min(drive[1] * EYE_GAIN[1], 1.0))
    return np.where(left, d[0], d[1]) * MAX_RATE_HZ


class Bench:
    def __init__(self, seed: int = 1):
        c = escape_circuit('783', 3, MIN_SYNAPSES)  # the 3-hop cut: the README's results
        self.brain = FastCircuit(len(c['ids']), c['pre'], c['post'], c['count'], c['inputs'], dt_ms=FLIGHT_DT_MS,
                                 seed=seed)
        self.left = c['input_left']
        self.out = np.concatenate([c['gf'], c['dn_pos']])
        self.out_names = ['GF (left)', 'GF (right)'] + c['dn_names'].tolist()
        rf = receptive_fields('783')
        pos = {int(f): i for i, f in enumerate(rf['ids'])}
        order = np.array([pos[int(f)] for f in c['ids'][c['inputs']]])
        self.rf = {k: rf[k][order] for k in ('az', 'el', 'size', 'type', 'side')}
        self.weights = {'retinotopic': rf_weights(self.rf, EYE_AZ, EYE_EL)}
        rng = np.random.default_rng(seed)
        shuffled = {k: v.copy() for k, v in self.rf.items()}
        for side in ('left', 'right'):
            for t in ('LC4', 'LPLC2'):
                m = np.flatnonzero((self.rf['side'] == side) & (self.rf['type'] == t))
                p = rng.permutation(m)
                for k in ('az', 'el', 'size'):
                    shuffled[k][m] = self.rf[k][p]
        self.weights['shuffled'] = rf_weights(shuffled, EYE_AZ, EYE_EL)

    def rates(self, scheme: str, loom: np.ndarray, gain: float) -> np.ndarray:
        if scheme == 'two channels':
            return two_channel(loom, self.left)
        drive = np.einsum('nij,ij->n', self.weights[scheme], loom) / LOOM_REF
        return np.minimum(gain * drive, 1.0) * MAX_RATE_HZ

    def trial(self, scheme: str, direction: tuple, gain: float = 1.0) -> tuple[np.ndarray, float]:
        """Output spike counts per tick (ticks x outputs) and the time to contact when the giant fibers
        alarmed (nan: never)."""
        psi = angle_to(*direction)
        self.brain.reset()
        n = int((START_M - RADIUS_M) / SPEED_MPS / (TICK_MS / 1000.0))
        counts = np.zeros((n, len(self.out)), np.int64)
        gf_hz, alarm_ttc = np.zeros(2), math.nan
        for k in range(n):
            t = k * TICK_MS / 1000.0
            self.brain.set_rates(self.rates(scheme, looming(direction, t, psi), gain))
            counts[k] = self.brain.run(TICK_MS)[self.out]
            gf_hz += SMOOTH * (counts[k, :2] * (1000.0 / TICK_MS) - gf_hz)
            if math.isnan(alarm_ttc) and gf_hz.max() >= GF_ALARM_HZ:
                alarm_ttc = (START_M - RADIUS_M) / SPEED_MPS - (t + TICK_MS / 1000.0)
        return counts, alarm_ttc


def ttc_of_tick(n: int) -> np.ndarray:
    contact = (START_M - RADIUS_M) / SPEED_MPS
    return contact - (np.arange(n) + 1) * TICK_MS / 1000.0


def cv_ridge(x: np.ndarray, y: np.ndarray, groups: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Predictions for each sample from a ridge regression fitted without its group."""
    pred = np.zeros_like(y, dtype=float)
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-9
        a = np.column_stack([(x[tr] - mu) / sd, np.ones(tr.sum())])
        w = np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ y[tr])
        pred[te] = np.column_stack([(x[te] - mu) / sd, np.ones(te.sum())]) @ w
    return pred


def decode(x: np.ndarray, az: np.ndarray, el: np.ndarray, groups: np.ndarray, direction: np.ndarray) -> dict:
    """Cross-validated by trial; 'unseen direction': the azimuth of each direction from a readout trained
    without any trial of that direction (interpolating needs a map that changes smoothly with direction)."""
    out = {}
    cs = np.column_stack([np.cos(np.radians(az)), np.sin(np.radians(az))])
    p = cv_ridge(x, cs, direction)
    out['unseen direction'] = float(np.median(
        np.abs((np.degrees(np.arctan2(p[:, 1], p[:, 0])) - az + 180.0) % 360.0 - 180.0)))
    lat = (np.abs(az) > 1) & (np.abs(az) < 179)
    p = cv_ridge(x[lat], np.sign(az[lat])[:, None], groups[lat])[:, 0]
    out['left/right'] = float(np.mean(np.sign(p) == np.sign(az[lat])))
    fb = np.abs(np.abs(az) - 90.0) > 1
    yfb = np.where(np.abs(az[fb]) < 90.0, 1.0, -1.0)
    p = cv_ridge(x[fb], yfb[:, None], groups[fb])[:, 0]
    out['front/back'] = float(np.mean(np.sign(p) == yfb))
    p = cv_ridge(x, np.column_stack([np.cos(np.radians(az)), np.sin(np.radians(az))]), groups)
    err = np.abs((np.degrees(np.arctan2(p[:, 1], p[:, 0])) - az + 180.0) % 360.0 - 180.0)
    out['azimuth error'] = float(np.median(err))
    yel = np.where(el > -10.0, 1.0, -1.0)
    p = cv_ridge(x, yel[:, None], groups)[:, 0]
    out['up/down'] = float(np.mean(np.sign(p) == yel))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trials', type=int, default=8, help='trials per direction and input scheme')
    args = ap.parse_args()
    t0 = time.time()
    b = Bench()
    print(f'{len(b.left)} looming-detector inputs, {len(b.out)} outputs; {len(DIRECTIONS)} directions x '
          f'{args.trials} trials; sphere r {RADIUS_M} m at {SPEED_MPS} m/s from {START_M} m')

    head_on = (0.0, 0.0)
    ref = np.nanmedian([b.trial('two channels', head_on)[1] for _ in range(6)])
    print(f'\nhead-on, two channels: giant fibers alarm {ref:.2f} s before contact')
    best, best_gap = GAINS[0], math.inf
    for g in GAINS:
        ttc = np.nanmedian([b.trial('retinotopic', head_on, g)[1] for _ in range(6)])
        print(f'  retinotopic, gain {g:4.0f}: {ttc:.2f} s' if not math.isnan(ttc) else f'  gain {g:4.0f}: no alarm')
        if not math.isnan(ttc) and abs(ttc - ref) < best_gap:
            best, best_gap = g, abs(ttc - ref)
    print(f'retinotopic input gain: {best:g}')

    results, alarms, saved = {}, {}, {}
    for scheme in ('two channels', 'retinotopic', 'shuffled'):
        xs, az, el, groups, dirs, ttcs = [], [], [], [], [], []
        for i, d in enumerate(DIRECTIONS):
            for k in range(args.trials):
                counts, ttc = b.trial(scheme, d, best)
                win = (ttc_of_tick(len(counts)) <= WINDOW_TTC[0]) & (ttc_of_tick(len(counts)) >= WINDOW_TTC[1])
                xs.append(counts[win].sum(0))
                az.append(d[0]), el.append(d[1]), groups.append(k), dirs.append(i), ttcs.append(ttc)
        x, az, el, ttcs = np.array(xs, float), np.array(az), np.array(el), np.array(ttcs)
        groups, dirs = np.array(groups), np.array(dirs)
        results[scheme] = decode(x, az, el, groups, dirs) | {'active outputs': int((x.sum(0) > 0).sum())}
        alarms[scheme] = (az, el, ttcs)
        saved |= {f'{scheme}/{k}': v for k, v in (('x', x), ('az', az), ('el', el), ('trial', groups), ('ttc', ttcs))}
        print(f'{scheme}: done ({time.time() - t0:.0f} s)', flush=True)
    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(RESULTS / 'directions.npz', gain=best, outputs=np.array(b.out_names), **saved)

    print('\nGiant-fiber alarm, seconds before contact (median over trials; - = never)')
    print('direction (az, el)  ' + ''.join(f'{s:>14s}' for s in alarms))
    for d in DIRECTIONS:
        row = []
        for s, (az, el, ttcs) in alarms.items():
            m = (az == d[0]) & (el == d[1])
            v = np.nanmedian(ttcs[m]) if np.isfinite(ttcs[m]).any() else math.nan
            row.append(f'{v:14.2f}' if not math.isnan(v) else f'{"-":>14s}')
        print(f'{d[0]:+5.0f} {d[1]:+4.0f}           ' + ''.join(row))

    print(f'\nWhere is it? Linear readout of the descending neurons ({WINDOW_TTC[0]}-{WINDOW_TTC[1]} s before '
          f'contact), cross-validated by trial (chance: 50%, 50%, ~90 deg, 50%)')
    keys = ('left/right', 'front/back', 'azimuth error', 'unseen direction', 'up/down', 'active outputs')
    print(f'{"":14s}' + ''.join(f'{k:>18s}' for k in keys))

    def cell(k, v):
        return (f'{v:14.0f} deg' if k in ('azimuth error', 'unseen direction') else f'{v:18d}' if k == 'active outputs'
                else f'{100 * v:17.0f}%')
    for s, r in results.items():
        print(f'{s:14s}' + ''.join(cell(k, r[k]) for k in keys))
    return 0


if __name__ == '__main__':
    sys.exit(main())
