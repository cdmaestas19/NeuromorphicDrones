"""Replay a recorded flight through an avoidance policy and the supervisor, offline, and report what
each decision costs and whether the decisions match the recording exactly.

Use it to make the decision path faster without changing what it decides, and to try a new policy
on real flight data without flying:
    launch.cmd avoid --record recordings/blocked.jsonl          (record once)
    bash py.sh bench.py recordings/blocked.jsonl [--policy geometric] [--repeat 20]
The replay is open loop: the drone's motion is what was recorded, so a policy that decides
differently is reported as a mismatch rather than flown. The compound eye is replayed from the
recording's .eye.npz when there is one.
The fly brain needs its brain server running and is not deterministic (Poisson input), so it doesn't
replay to a MATCH.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from avoidance import AvoidanceSupervisor, Mode, make_sensors
from eye import clearance as eye_clearance
from eye import ring
from policies import POLICIES


def load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for r in rows:  # into the same types the flight passes (tuples, inf), outside the timed region
        r['ranges'] = {k: math.inf if v is None else v for k, v in r['ranges'].items()}
        for key in ('position', 'velocity_world', 'intended_velocity_world'):
            r[key] = tuple(r[key])
        stop = r.get('route_stop_m', 'absent')  # absent in recordings made before it existed
        r['route_stop_m'] = math.inf if stop is None or stop == 'absent' else stop
    eye_path = path.with_suffix('.eye.npz')
    if eye_path.is_file():  # each row gets the eye reading the flight decided on (the newest by then)
        e = np.load(eye_path)
        readings = [(float(st), tuple(float(v) for v in p), d.astype(np.float32))
                    for st, p, d in zip(e['sim_t'], e['pos'], e['dist'])]
        k = -1
        for r in rows:
            while k + 1 < len(readings) and e['t'][k + 1] <= r['t']:
                k += 1
            r['eye'] = readings[k] if k >= 0 else None
    return rows


def replay(rows: list[dict], policy_name: str):
    """One pass. Returns the decisions and the per-part cost in ns: (build, policy, supervisor)."""
    policy, sup = POLICIES[policy_name](), AvoidanceSupervisor()
    decisions, costs = [], []
    clock = time.perf_counter_ns
    for r in rows:
        t0 = clock()
        s = make_sensors(r['t'], r['position'], r['yaw_deg'], r['velocity_world'], r['intended_velocity_world'],
                         r['ranges'], sup.mode, r.get('eye'), r['route_stop_m'])
        t1 = clock()
        cmd = policy.step(s, None)
        t2 = clock()
        clearance = (eye_clearance(*ring(s.eye, s.yaw_deg)) if s.eye is not None  # as avoid_demo.py
                     else min(s.ranges['front'], s.ranges['left'], s.ranges['right']))
        changed = sup.update(s.t, cmd.signal, clearance)
        t3 = clock()
        if changed and sup.mode is Mode.NOMINAL:
            policy.reset()  # as avoid_demo.py does after each episode
        decisions.append((cmd.signal, sup.mode.value))
        costs.append((t1 - t0, t2 - t1, t3 - t2))
    return decisions, costs


def stats(values_us: list[float]) -> str:
    s = sorted(values_us)
    return (f'mean {sum(s) / len(s):6.2f}  median {s[len(s) // 2]:6.2f}  '
            f'p99 {s[min(len(s) - 1, int(0.99 * len(s)))]:6.2f}  max {s[-1]:7.2f} us')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('recording', type=Path)
    ap.add_argument('--policy', choices=sorted(POLICIES), default='geometric')
    ap.add_argument('--repeat', type=int, default=20, help='replay passes to time (default 20)')
    args = ap.parse_args()

    rows = load(args.recording)
    has_eye = any(r.get('eye') is not None for r in rows)
    if '360 eye' in POLICIES[args.policy].inputs and not has_eye:
        print(f'note: {args.recording} has no compound-eye recording; policy "{args.policy}" decides on the rays')
    episodes = sum(1 for a, b in zip(rows, rows[1:]) if a['mode_after'] == 'NOMINAL' and b['mode_after'] == 'AVOID')
    episodes += rows[0]['mode_after'] == 'AVOID'
    print(f'{args.recording}: {len(rows)} decisions over {rows[-1]["t"] - rows[0]["t"]:.1f} s of flight, '
          f'{episodes} avoidance episode(s) recorded')

    decisions, _ = replay(rows, args.policy)  # warm-up pass, also the correctness check
    mode_match = sum(d[1] == r['mode_after'] for d, r in zip(decisions, rows))
    max_diff = max(abs(d[0] - r['signal']) for d, r in zip(decisions, rows))
    print(f'decisions vs recording: {mode_match}/{len(rows)} identical modes, max signal difference {max_diff:.2e}'
          f'{"  -> MATCH" if mode_match == len(rows) and max_diff < 1e-9 else "  -> DIFFERENT"}')

    parts = ([], [], [])
    for _ in range(args.repeat):
        _, costs = replay(rows, args.policy)
        for c in costs:
            for i in range(3):
                parts[i].append(c[i] / 1000.0)
    total = [a + b + c for a, b, c in zip(*parts)]
    print(f'decision cost for policy "{args.policy}" ({args.repeat} passes, this PC, Python):')
    print(f'  total        {stats(total)}')
    for name, values in zip(('build inputs', 'policy', 'supervisor'), parts):
        print(f'  {name:12s} {stats(values)}')
    print(f'  the 20 Hz control loop allows 50000 us per tick; this uses {100 * sum(total) / len(total) / 50000:.3f}%')


if __name__ == '__main__':
    main()
