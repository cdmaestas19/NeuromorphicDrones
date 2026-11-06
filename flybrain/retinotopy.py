"""Where in the eye each looming detector looks: receptive fields of LC4 and LPLC2 from the connectome.

The brain used to get one input channel per eye (every LC4 / LPLC2 of an eye at the same rate), so it could
not know where a threat was. Here each of those neurons gets its own receptive field, derived from the wiring
alone (no physiology fitted), so the drone's eye can drive each neuron from its own patch of the view:

1. Columns. The eye is a lattice of ~800 columns; Mi1 has one cell per column. Mi1 cell bodies sit over their
   columns in the medulla cortex, a curved sheet: _sheet() flattens their positions to 2-D (Isomap: distances
   along the sheet, then classical MDS).
2. Every columnar neuron. Optic-lobe types with >= COLUMNAR_MIN cells per side tile the eye (33 types,
   ~25k neurons per side: L1-5, Mi, Tm, T2-T5...). Their positions are the harmonic extension of Mi1's over
   their synapses: each neuron sits at the synapse-weighted mean of its partners' positions, Mi1 held fixed.
3. Receptive fields. An LC4 / LPLC2 neuron's centre is the synapse-weighted mean of its columnar inputs'
   positions (T4/T5 for LPLC2; T2, Tm2-4... for LC4), its size their weighted spread.
4. Orientation. "Up" on the sheet is the way the Mi1 cell bodies get more dorsal (FlyWire y points ventral);
   "back" is their front-back axis (FlyWire z), signed by cells whose receptive fields are known from
   physiology: the vertical-system cells VS1..VS8 tile the field from frontal to posterior (Krapp &
   Hengstenberg 1996 in the blowfly; Joesch et al. 2008 in Drosophila), and their T4/T5 inputs place them on
   the sheet. The horizontal-system cells HSN / HSE / HSS (dorsal / equatorial / ventral: Schnell et al.
   2010) are the check. The sheet is then stretched over the eye's approximate field (EYE_AZ, EYE_EL): the
   order and neighbourhoods come from the wiring, the scale doesn't.

Angles are in the body frame: azimuth from straight ahead, positive to the left; elevation positive up.

    bash py.sh retinotopy.py      checks and a map (results/retinotopy.png); the receptive fields are cached

Data: FlyWire (Dorkenwald et al. 2024, Nature 634:124) as packaged by Shiu et al. 2024 (Nature 634:210); cell
types and positions: Schlegel et al. 2024 (Nature 634:139); optic-lobe types: Matsliah et al. 2024 (Nature
634:166). CC-BY 4.0.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from connectome import ANNOTATIONS, BRAIN_MODEL_DIR, VERSIONS, Connectome

COLUMNAR_MIN = 500          # cells per side for a type to count as columnar (the eye has ~800 columns)
ISOMAP_K = 10               # neighbours per Mi1 cell body when flattening the sheet
VOXEL_UM = np.array([0.004, 0.004, 0.040])  # FlyWire voxel (4 x 4 x 40 nm) in um
EYE_AZ = (-15.0, 160.0)     # approximate field of one eye, deg: azimuth from the front (frontal binocular
EYE_EL = (-70.0, 80.0)      # overlap ~15 deg) to the rear edge; elevation ventral to dorsal
TARGETS = ('LPLC2', 'LC4')
ANCHORS_UP = ('HSN', 'HSE', 'HSS')                      # dorsal -> ventral
ANCHORS_BACK = tuple(f'VS{i}' for i in range(1, 9))     # frontal -> posterior
CACHE_DIR = Path(__file__).resolve().parent / 'cache'
RESULTS = Path(__file__).resolve().parent / 'results'
RF_VERSION = 1  # bump when the method or what the cache holds changes


def _sheet(xyz: np.ndarray, k: int = ISOMAP_K) -> np.ndarray:
    """2-D coordinates (um) of points on a curved sheet: Isomap (geodesic distances on a k-nearest-neighbour
    graph, then classical MDS)."""
    n = len(xyz)
    d, j = cKDTree(xyz).query(xyz, k + 1)
    g = sp.csr_matrix((d[:, 1:].ravel(), (np.repeat(np.arange(n), k), j[:, 1:].ravel())), shape=(n, n))
    g = g.maximum(g.T)
    if connected_components(g, directed=False)[0] > 1:
        return _sheet(xyz, 2 * k)
    geo = shortest_path(g, directed=False)
    c = np.eye(n) - 1.0 / n
    b = -0.5 * c @ (geo ** 2) @ c
    w, v = np.linalg.eigh(b)
    return v[:, -2:][:, ::-1] * np.sqrt(w[-2:][::-1])


def _harmonic(nodes: np.ndarray, fixed: np.ndarray, x_fixed: np.ndarray, pre, post, count) -> dict:
    """Positions of `nodes` (model indices) that minimise the synapse-weighted squared distance between
    partners, with `fixed` held at x_fixed: each free node at the weighted mean of its partners."""
    loc = {g: i for i, g in enumerate(nodes)}
    pos = pd.Series(np.arange(len(nodes)), index=nodes)
    a, b = pos.reindex(pre).to_numpy(), pos.reindex(post).to_numpy()
    keep = ~(np.isnan(a) | np.isnan(b)) & (a != b)
    a, b, w = a[keep].astype(int), b[keep].astype(int), count[keep].astype(float)
    n = len(nodes)
    wm = sp.csr_matrix((w, (a, b)), shape=(n, n))
    wm = (wm + wm.T).tocsr()
    is_fixed = np.zeros(n, bool)
    fx = np.array([loc[g] for g in fixed])
    is_fixed[fx] = True
    _, comp = connected_components(wm, directed=False)
    reach = np.isin(comp, np.unique(comp[fx]))  # free nodes cut off from every Mi1 get no position
    free = np.flatnonzero(~is_fixed & reach)
    lap = (sp.diags(np.asarray(wm.sum(1)).ravel()) - wm).tocsr()
    x = np.full((n, 2), np.nan)
    x[fx] = x_fixed
    rhs = -lap[free][:, fx] @ x_fixed
    lff = lap[free][:, free].tocsc()
    x[free] = np.column_stack([spsolve(lff, rhs[:, i]) for i in range(2)])
    return {g: x[i] for g, i in loc.items() if not np.isnan(x[i, 0])}


def _pooled(targets: np.ndarray, where: dict, pre, post, count) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre, spread (weighted RMS distance) and synapses counted, of each target's inputs that have a
    position."""
    sel = np.isin(post, targets) & np.isin(pre, np.fromiter(where.keys(), np.int64))
    p, q, w = pre[sel], post[sel], count[sel].astype(float)
    xy = np.array([where[i] for i in p]) if len(p) else np.zeros((0, 2))
    centre, spread, total = np.full((len(targets), 2), np.nan), np.full(len(targets), np.nan), np.zeros(len(targets))
    for k, t in enumerate(targets):
        m = q == t
        if m.any():
            c = np.average(xy[m], axis=0, weights=w[m])
            centre[k] = c
            spread[k] = np.sqrt(np.average(((xy[m] - c) ** 2).sum(1), weights=w[m]))
            total[k] = w[m].sum()
    return centre, spread, total


def _side(side: str, ann: pd.DataFrame, conn: Connectome, pre, post, count) -> dict:
    """Receptive fields of one eye's TARGETS, plus the checks report() prints."""
    index = conn.index
    here = ann[(ann.side == side) & ann.index.isin(index.keys())]
    opt = here[here.super_class == 'optic']
    sizes = opt.groupby('cell_type').size()
    columnar = sorted(sizes.index[sizes >= COLUMNAR_MIN])
    mi1 = here[(here.cell_type == 'Mi1') & here[['soma_x', 'soma_y', 'soma_z']].notna().all(1)]
    xyz = mi1[['soma_x', 'soma_y', 'soma_z']].to_numpy(float) * VOXEL_UM
    flat = _sheet(xyz)
    fixed = np.array([index[f] for f in mi1.index])
    nodes = np.array([index[f] for f in opt.index[opt.cell_type.isin(columnar)]])
    where = _harmonic(nodes, fixed, flat, pre, post, count)

    def cells(t):
        return np.array([index[f] for f in here.index[here.cell_type == t]])

    def anchor(names):  # each anchor cell's centre from its T4/T5 inputs only (their own subtype's layer)
        t45 = {g: where[g] for t in ('T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d') for g in cells(t)
               if g in where}
        return np.array([_pooled(cells(n), t45, pre, post, count)[0].mean(0) for n in names])

    hs, vs = anchor(ANCHORS_UP), anchor(ANCHORS_BACK)
    fit = np.column_stack([flat - flat.mean(0), np.ones(len(flat))])
    up = np.linalg.lstsq(fit, -xyz[:, 1], rcond=None)[0][:2]
    up /= np.linalg.norm(up)
    along_z = np.linalg.lstsq(fit, xyz[:, 2], rcond=None)[0][:2]
    back = along_z - (along_z @ up) * up
    back /= np.linalg.norm(back)
    vs_order = spearmanr(vs @ back, np.arange(len(vs)))[0]
    if vs_order < 0:
        back, vs_order = -back, -vs_order
    axes = np.column_stack([back, up])

    def angles(xy):
        p = xy @ axes
        lo, hi = np.percentile(flat @ axes, [1, 99], axis=0)
        frac = (p - lo) / (hi - lo)
        return (EYE_AZ[0] + frac[:, 0] * (EYE_AZ[1] - EYE_AZ[0]), EYE_EL[0] + frac[:, 1] * (EYE_EL[1] - EYE_EL[0]),
                np.mean([(EYE_AZ[1] - EYE_AZ[0]) / (hi[0] - lo[0]), (EYE_EL[1] - EYE_EL[0]) / (hi[1] - lo[1])]))

    out = {'ids': [], 'type': [], 'az': [], 'el': [], 'size': [], 'synapses': []}
    for t in TARGETS:
        g = cells(t)
        centre, spread, total = _pooled(g, where, pre, post, count)
        az, el, deg = angles(centre)
        out['ids'] += [int(conn.ids[i]) for i in g]
        out['type'] += [t] * len(g)
        out['az'] += list(az)
        out['el'] += list(el)
        out['size'] += list(spread * deg)
        out['synapses'] += list(total)
    mi1_az, mi1_el, _ = angles(flat)
    hs_el, vs_az = angles(hs)[1], angles(vs)[0]
    checks = {'columnar_types': len(columnar), 'placed': len(where), 'columnar_cells': len(nodes),
              'mi1': len(mi1), 'hs_el': hs_el, 'vs_az': vs_az, 'vs_order': float(vs_order),
              'frontal_z': bool(along_z @ back < 0),
              'r_up': float(np.corrcoef(mi1_el, -xyz[:, 1])[0, 1]),
              'r_back': float(abs(np.corrcoef(mi1_az, xyz[:, 2])[0, 1]))}
    return {k: np.array(v) for k, v in out.items()} | {'checks': checks}


COLUMN_DEG = 5.0  # each column's own acceptance (~ the eye's 5-degree spacing), added to the inputs' spread


def _unit(az_deg, el_deg) -> np.ndarray:
    a, e = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], axis=-1)


class RFPool:
    """Each neuron's receptive field as a weighting of directions (body frame, degrees): a Gaussian of its
    centre and size (RMS spread = sqrt(2) sigma, plus COLUMN_DEG), times solid angle (cos elevation), so that
    over the whole sphere on a 5-degree grid (the drone's eye's) it sums to 1. On a grid that sees only part
    of the sphere (the drone's eye has no sky above 30 degrees) a field loses what lies outside."""

    def __init__(self, rf: dict):
        self.centre = _unit(rf['az'], rf['el'])                              # (neurons, 3)
        self.sigma = np.radians(np.sqrt(np.asarray(rf['size']) ** 2 / 2.0 + COLUMN_DEG ** 2))
        az, el = np.meshgrid(np.arange(-180.0, 180.0, 5.0), np.arange(-87.5, 90.0, 5.0))
        self.norm = self._gauss(az.ravel(), el.ravel()).sum(axis=1)

    def _gauss(self, az_deg, el_deg) -> np.ndarray:
        """(neurons, directions): field value x solid-angle weight, unnormalised."""
        psi = np.arccos(np.clip(self.centre @ _unit(az_deg, el_deg).T, -1.0, 1.0))
        return np.exp(-0.5 * (psi / self.sigma[:, None]) ** 2) * np.cos(np.radians(el_deg))[None, :]

    def drive(self, az_deg, el_deg, values) -> np.ndarray:
        """Per neuron: the field-weighted sum of `values` seen in the directions (az_deg, el_deg)."""
        if len(values) == 0:
            return np.zeros(len(self.sigma))
        return self._gauss(np.asarray(az_deg, float), np.asarray(el_deg, float)) @ np.asarray(values, float) / self.norm

    def weights(self, az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
        """(neurons, rows, columns) on a grid: az_deg per column, el_deg per row."""
        az, el = np.meshgrid(np.asarray(az_deg, float), np.asarray(el_deg, float))
        return (self._gauss(az.ravel(), el.ravel()) / self.norm[:, None]).reshape(len(self.sigma), *az.shape)


def rf_weights(rf: dict, az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
    """(neurons, rows, columns): each receptive field's weight on a grid of directions (RFPool)."""
    return RFPool(rf).weights(az_deg, el_deg)


def receptive_fields(version: str = '783') -> dict:
    """Per LC4 / LPLC2 neuron (both eyes), from cache/ or computed (~1 min, the whole connectome in memory):
    ids (FlyWire), type, side, az, el (deg, body frame: + left, + up), size (deg, RMS spread of its inputs),
    synapses (from columnar inputs)."""
    source = BRAIN_MODEL_DIR / VERSIONS[version][1]
    stamp = f'{RF_VERSION}:{source.stat().st_size}:{int(source.stat().st_mtime)}:{ANNOTATIONS.stat().st_size}'
    path = CACHE_DIR / f'retinotopy_{version}.npz'
    if path.is_file():
        with np.load(path, allow_pickle=False) as d:
            if str(d['stamp']) == stamp:
                return {k: d[k] for k in d.files if k != 'stamp'}
    out = _compute(version)
    keep = {k: v for k, v in out.items() if k != 'checks'}
    CACHE_DIR.mkdir(exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp, stamp=np.array(stamp), **keep)
    os.replace(tmp, path)
    return keep


def _compute(version: str) -> dict:
    conn = Connectome(version)
    ann = pd.read_csv(ANNOTATIONS, sep='\t', low_memory=False,
                      usecols=['root_id', 'cell_type', 'side', 'super_class', 'soma_x', 'soma_y', 'soma_z'])
    ann = ann.set_index('root_id')
    con = conn.connectivity
    pre, post = con.Presynaptic_Index.to_numpy(), con.Postsynaptic_Index.to_numpy()
    count = con.Connectivity.to_numpy()
    parts, checks = [], {}
    for side, sign in (('left', 1.0), ('right', -1.0)):
        s = _side(side, ann, conn, pre, post, count)
        checks[side] = s.pop('checks')
        s['az'] = sign * s['az']  # the right eye looks to the right: negative azimuth in the body frame
        s['side'] = np.array([side] * len(s['ids']))
        parts.append(s)
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out['checks'] = checks
    return out


def report() -> int:
    """The checks, and a map of the receptive fields (results/retinotopy.png)."""
    out = _compute('783')
    for side, c in out.pop('checks').items():
        print(f'{side} eye: {c["columnar_types"]} columnar types, {c["placed"]} of {c["columnar_cells"]} columnar '
              f'neurons placed from {c["mi1"]} Mi1 columns')
        print(f'  the flattened sheet vs the cell bodies\' positions: elevation ~ dorsal (FlyWire -y) r = '
              f'{c["r_up"]:.2f}, azimuth ~ front-back (FlyWire z) |r| = {c["r_back"]:.2f}; frontal field at '
              f'{"larger" if c["frontal_z"] else "smaller"} z')
        print(f'  VS1..VS8 azimuth (frontal -> posterior expected; sets the sign): '
              f'{" ".join(f"{a:.0f}" for a in c["vs_az"])} deg, rank correlation {c["vs_order"]:+.2f}')
        print(f'  check: HSN / HSE / HSS elevation (dorsal / equatorial / ventral expected): '
              f'{" / ".join(f"{e:+.0f}" for e in c["hs_el"])} deg')
    keep = {k: v for k, v in out.items() if k != 'checks'}
    for side in ('left', 'right'):
        for t in TARGETS:
            m = (keep['side'] == side) & (keep['type'] == t)
            az, el, size = keep['az'][m], keep['el'][m], keep['size'][m]
            print(f'{t:5s} {side:5s}: {m.sum():3d} neurons; centres az {az.min():+.0f}..{az.max():+.0f}, '
                  f'el {el.min():+.0f}..{el.max():+.0f} deg; input spread (RMS) median {np.median(size):.0f} deg; '
                  f'{np.median(keep["synapses"][m]):.0f} columnar synapses each (median)')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('(no matplotlib: no map)')
        return 0
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for t, mk in zip(TARGETS, ('o', '^')):
        for side, colour in (('left', '#1f77b4'), ('right', '#d62728')):
            m = (keep['side'] == side) & (keep['type'] == t)
            ax.scatter(keep['az'][m], keep['el'][m], s=18, marker=mk, c=colour, alpha=0.7, label=f'{t} {side} eye')
    ax.axvline(0, color='0.6', lw=0.8)
    ax.set_xlim(180, -180)
    ax.set_xlabel('azimuth (deg, + left; 0 = straight ahead)')
    ax.set_ylabel('elevation (deg)')
    ax.set_title('Receptive-field centres of the looming detectors, derived from the connectome')
    ax.legend(fontsize=8, ncol=2, loc='lower center')
    fig.text(0.01, 0.01, 'Data: FlyWire (Dorkenwald et al. 2024; Schlegel et al. 2024; Matsliah et al. 2024), '
             'CC-BY 4.0', fontsize=7, color='0.4')
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / 'retinotopy.png', dpi=130)
    print(f'map: {RESULTS / "retinotopy.png"}')
    return 0


if __name__ == '__main__':
    sys.exit(report())
