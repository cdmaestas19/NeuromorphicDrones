"""Train the camera's person / vehicle classifier (camera_kind.py) on recorded flights.

    flypilot\\camera_train.cmd [--test PREFIX,...] [--epochs 15] [--rebuild]

Every street-scene recording with camera frames (avoid_demo.py --record) is a source. Each frame's strips
are labelled automatically from where the simulator put the people and the cart (camera_kind.label_strips:
person, vehicle, background, and nothing where it is ambiguous), so no frame is labelled by hand. Whole
flights are held out for the test: every fifth recording by default, or those whose names start with one of
--test. The test prints a confusion matrix over strips from flights the network never saw, and how often it
is sure (camera_kind.CameraKind.MIN_P) and right. The weights go to models/camera_kind.npz, which
camera_kind.StripNet runs in the flight with numpy.

Runs in its own venv (PyTorch; CPU is plenty for ~6 k weights): camera_train.sh creates it.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

import camera_kind as ck
import movers
from eye import ring

HERE = Path(__file__).resolve().parent
RECORDINGS = HERE / 'recordings'
CACHE = HERE / '.cache' / 'camera_strips.npz'
SCENE = HERE / 'scenes' / 'street.xml'
PAD = 4                    # strips are cut this much wider on each side, for shifting during training
FRAME_STEP = 2             # every other frame (neighbours are nearly the same picture)
PER_FRAME = {'person': 3, 'vehicle': 3, 'background': 2}  # strips per frame and kind, at most


def _headings(rows: list) -> list:
    """Per row: {mover: heading_deg} from its recorded motion (the last one known)."""
    out, last, known = [], {}, {}
    for r in rows:
        for name, xy in r.get('movers', {}).items():
            if name in last and math.dist(last[name], xy) > 1e-4:
                known[name] = math.degrees(math.atan2(xy[1] - last[name][1], xy[0] - last[name][0]))
            last[name] = xy
        out.append(dict(known))
    return out


def _flight(path: Path, shapes: dict, rng: np.random.Generator):
    """(strips, labels) of one recording, or None if it isn't a street flight with frames and eye."""
    frames, eye = path.with_suffix('.frames.npz'), path.with_suffix('.eye.npz')
    if not (frames.is_file() and eye.is_file()):
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not any(r.get('movers') for r in rows):
        return None  # not the street scene: the labels would put its people where they aren't
    F, E = np.load(frames), np.load(eye)
    row_t = np.array([r['t'] for r in rows])
    heads = _headings(rows)
    xs, ys = [], []
    for k in range(0, len(F['t']), FRAME_STEP):
        img = F['frames'][k]
        h, w = img.shape
        i = int(np.argmin(np.abs(row_t - F['t'][k])))
        e = int(np.argmin(np.abs(E['t'] - F['t'][k])))
        if abs(row_t[i] - F['t'][k]) > 0.1 or abs(E['t'][e] - F['t'][k]) > 0.15:
            continue
        az, near = ring((E['sim_t'][e], tuple(E['pos'][e]), E['dist'][e].astype(np.float32)), 0.0)
        n = len(near)
        eye_near = lambda a: float(near[int(round((a % 360.0) / (360.0 / n))) % n])  # noqa: E731
        r = rows[i]
        labels = ck.label_strips(shapes, r.get('movers', {}), heads[i], r['position'], r['yaw_deg'], eye_near, w, h)
        wide = np.pad(img, ((0, 0), (ck.STRIP_W // 2 + PAD, ck.STRIP_W // 2 + PAD)), mode='edge')
        for kind, cap in PER_FRAME.items():
            us = [u for u, kd in labels if kd == kind]
            for u in (rng.choice(us, size=min(cap, len(us)), replace=False) if us else []):
                xs.append(wide[:, u: u + ck.STRIP_W + 2 * PAD])
                ys.append(ck.KINDS.index(kind))
    if not xs:
        return None
    return np.stack(xs), np.array(ys, np.int8)


def build(rebuild: bool) -> dict:
    paths = sorted(RECORDINGS.glob('*.jsonl'))
    key = hashlib.sha1('\n'.join(f'{p.name}:{p.stat().st_mtime_ns}' for p in paths).encode()).hexdigest()
    if CACHE.is_file() and not rebuild:
        d = np.load(CACHE, allow_pickle=False)
        if str(d['key']) == key:
            print(f'dataset: {CACHE} (cached)')
            return {k: d[k] for k in d.files}
    shapes = movers.load_movers(SCENE)
    rng = np.random.default_rng(0)
    X, Y, G, names = [], [], [], []
    for p in paths:
        got = _flight(p, shapes, rng)
        if got is None:
            continue
        X.append(got[0])
        Y.append(got[1])
        G.append(np.full(len(got[1]), len(names), np.int16))
        names.append(p.stem)
        print(f'  {p.stem:24s} {len(got[1]):5d} strips  ' +
              '  '.join(f'{k} {int((got[1] == j).sum())}' for j, k in enumerate(ck.KINDS)), flush=True)
    d = {'X': np.concatenate(X), 'y': np.concatenate(Y), 'flight': np.concatenate(G), 'names': np.array(names),
         'key': np.array(key)}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **d)
    return d


class Net(nn.Module):
    """camera_kind.StripNet's layers: 3 strided convs (ReLU), global average pool, linear."""

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(1, 8, 5, 2, 2), nn.Conv2d(8, 16, 3, 2, 1), nn.Conv2d(16, 32, 3, 2, 1)])
        self.fc = nn.Linear(32, len(ck.KINDS))

    def forward(self, x):
        for c in self.convs:
            x = torch.relu(c(x))
        return self.fc(x.mean(dim=(2, 3)))


def _batch(X: np.ndarray, idx: np.ndarray, rng: np.random.Generator, train: bool) -> torch.Tensor:
    """Strips cut to STRIP_W (shifted by up to PAD, mirrored, noisy when training), normalised."""
    off = rng.integers(0, 2 * PAD + 1, len(idx)) if train else np.full(len(idx), PAD)
    s = np.stack([X[i][:, o: o + ck.STRIP_W] for i, o in zip(idx, off)])
    if train:
        flip = rng.random(len(idx)) < 0.5
        s[flip] = s[flip][:, :, ::-1]
        s = np.clip(s.astype(np.float32) + rng.normal(0, 3.0, s.shape), 0, 255).astype(np.uint8)
    return torch.from_numpy(ck.normalise(s)[:, None])


def export(net: Net, path: Path):
    w = {}
    for i, c in enumerate(net.convs):
        w[f'conv{i}_w'] = c.weight.detach().numpy()
        w[f'conv{i}_b'] = c.bias.detach().numpy()
        w[f'conv{i}_stride'] = np.array(c.stride[0])
    w['fc_w'], w['fc_b'] = net.fc.weight.detach().numpy(), net.fc.bias.detach().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **w)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--test', default='', help='hold out recordings whose names start with these (comma-separated); '
                                               'default: every fifth recording')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--rebuild', action='store_true', help='label the recordings again even if cached')
    ap.add_argument('--out', type=Path, default=ck.MODEL)
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    d = build(args.rebuild)
    X, y, flight, names = d['X'], d['y'].astype(np.int64), d['flight'], [str(n) for n in d['names']]
    prefixes = [p for p in args.test.split(',') if p]
    held = np.array([any(n.startswith(p) for p in prefixes) if prefixes else k % 5 == 4 for k, n in enumerate(names)])
    test = held[flight]
    tr, te = np.flatnonzero(~test), np.flatnonzero(test)
    count = lambda idx: ', '.join(f'{k} {int((y[idx] == j).sum())}' for j, k in enumerate(ck.KINDS))  # noqa: E731
    print(f'{len(names)} flights, {len(y)} strips. train: {len(tr)} ({count(tr)}); '
          f'test: {len(te)} from {int(held.sum())} flights ({count(te)})')
    if not len(tr) or not len(te):
        print('need both training and test flights', file=sys.stderr)
        return 1

    net = Net()
    freq = np.bincount(y[tr], minlength=len(ck.KINDS)).astype(np.float32)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(freq.sum() / np.maximum(freq, 1.0) / len(freq)))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    for epoch in range(args.epochs):
        net.train()
        order = rng.permutation(tr)
        total = 0.0
        for b in range(0, len(order), 256):
            idx = order[b: b + 256]
            opt.zero_grad()
            loss = loss_fn(net(_batch(X, idx, rng, True)), torch.from_numpy(y[idx]))
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        print(f'epoch {epoch + 1:2d}: loss {total / len(order):.3f}', flush=True)

    net.eval()
    with torch.no_grad():
        p = torch.cat([torch.softmax(net(_batch(X, te[b: b + 1024], rng, False)), 1)
                       for b in range(0, len(te), 1024)]).numpy()
    pred, truth = p.argmax(1), y[te]
    print('\ntest flights: ' + ', '.join(n for n, h in zip(names, held) if h))
    print('confusion (rows: true, columns: predicted)\n' + ' ' * 12 + ''.join(f'{k:>12s}' for k in ck.KINDS))
    for j, k in enumerate(ck.KINDS):
        print(f'{k:>12s}' + ''.join(f'{int(((truth == j) & (pred == i)).sum()):12d}' for i in range(len(ck.KINDS))))
    sure = p.max(1) >= ck.CameraKind.MIN_P
    print(f'accuracy {np.mean(pred == truth):.3f}; sure (p >= {ck.CameraKind.MIN_P}) on {sure.mean():.1%} of strips, '
          f'right on {np.mean(pred[sure] == truth[sure]):.3f} of those')
    for j, k in enumerate(ck.KINDS[1:], 1):
        said = sure & (pred == j)
        print(f'  said "{k}" (sure): {int(said.sum())} times, right {np.mean(truth[said] == j) if said.any() else float("nan"):.3f}; '
              f'of the true {k}s, found (sure) {np.mean(said[truth == j]) if (truth == j).any() else float("nan"):.3f}')

    export(net, args.out)
    check = ck.StripNet(args.out).probs(np.stack([X[i][:, PAD: PAD + ck.STRIP_W] for i in te[:64]]))
    diff = float(np.abs(check - p[:64]).max())
    print(f'\nweights -> {args.out} (numpy forward pass matches PyTorch to {diff:.1e})')
    return 0 if diff < 1e-4 else 1


if __name__ == '__main__':
    sys.exit(main())
