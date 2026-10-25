"""The escape subcircuit in C (fastlif.c): the same model as circuit.Subcircuit's Brian2 network, ~20x cheaper.

Brian2 steps its network from Python: at the flight's 0.2 ms step that is 250 rounds of per-object overhead
every 50 ms tick, which dominates for a network of 3,118 neurons. fastlif.c runs a whole tick per call. The
kernel is plain C over a few MB of arrays, the kind of code a small onboard processor (the AI-deck's GAP8)
could run. check_fast.py compares it with Brian2, spike for spike on the same input spikes, and on the giant
fibers' response to looming.

FastCircuit has Subcircuit's interface: set_rates(), run(duration_ms) -> spike counts, reset().
"""
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import numpy.ctypeslib as npc
import ctypes

HERE = Path(__file__).resolve().parent
SRC = HERE / 'fastlif.c'
LIB = HERE / 'build' / 'libfastlif.so'

# The model's parameters in mV and ms: model_params() reads them from the model repo's model.py; these are
# its values, for use without Brian2 installed.
DEFAULTS = {'v_0': -52.0, 'v_rst': -52.0, 'v_th': -45.0, 't_mbr': 20.0, 'tau': 5.0, 't_rfc': 2.2, 't_dly': 1.8,
            'w_syn': 0.275, 'f_poi': 250.0}


def model_params() -> dict:
    """The whole-brain model's parameters (model.default_params), as plain numbers in mV and ms."""
    try:
        import connectome  # noqa: F401  (puts the model repo on sys.path)
        import model
        from brian2 import ms, mV
    except ImportError:
        return dict(DEFAULTS)
    p = model.default_params
    out = {k: float(p[k] / mV) for k in ('v_0', 'v_rst', 'v_th', 'w_syn')}
    out.update({k: float(p[k] / ms) for k in ('t_mbr', 'tau', 't_rfc', 't_dly')})
    out['f_poi'] = float(p['f_poi'])
    return out


def build() -> Path:
    """Compile fastlif.c into build/libfastlif.so when it is missing or older than the source."""
    if LIB.is_file() and LIB.stat().st_mtime >= SRC.stat().st_mtime:
        return LIB
    LIB.parent.mkdir(exist_ok=True)
    tmp = LIB.with_suffix(f'.{os.getpid()}.tmp')
    subprocess.run(['gcc', '-O2', '-shared', '-fPIC', '-o', str(tmp), str(SRC)], check=True)
    os.replace(tmp, LIB)  # atomic: two processes building at once can't leave half a library
    return LIB


def _kernel():
    lib = ctypes.CDLL(str(build()))
    f = lib.lif_run
    d1, i32, i64 = npc.ndpointer(np.float64, flags='C'), npc.ndpointer(np.int32, flags='C'), npc.ndpointer(np.int64, flags='C')
    u8 = ctypes.c_void_p
    f.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64,
                  d1, d1, i64, i32,
                  ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                  i32, i32, d1,
                  d1, ctypes.c_int, ctypes.c_int,
                  ctypes.c_int, i32, d1, ctypes.c_double, u8,
                  npc.ndpointer(np.uint64, flags='C'), i32, i32]
    f.restype = ctypes.c_long
    return f


def _steps(ms_: float, dt_ms: float) -> int:
    """Brian2's timestep(): how many steps a duration takes (rounded down, with its 1e-3 dt tolerance)."""
    return int((ms_ + 1e-3 * dt_ms) / dt_ms)


class FastCircuit:
    """n neurons; connections pre -> post with signed synapse counts (weight = count x w_syn); Poisson input
    to the neurons idx_in (an f_poi x w_syn kick, no refractory period, as model.poi). dt_ms: the step.
    params: model_params()'s (the default)."""

    def __init__(self, n: int, pre, post, count, idx_in, dt_ms: float = 0.1, seed: int | None = None,
                 params: dict | None = None):
        self._run = _kernel()
        self.p = p = params or model_params()
        self.n, self.dt_ms = int(n), float(dt_ms)
        order = np.argsort(np.asarray(pre), kind='stable')
        pre, post = np.asarray(pre)[order], np.asarray(post)[order]
        self.row_ptr = np.zeros(self.n + 1, np.int32)
        np.add.at(self.row_ptr, pre + 1, 1)
        self.row_ptr = np.cumsum(self.row_ptr).astype(np.int32)
        self.post = post.astype(np.int32)
        self.w = (np.asarray(count, np.float64)[order] * p['w_syn']).astype(np.float64)
        self.n_connections = len(self.post)
        self.idx_in = np.asarray(idx_in, np.int32)
        self.rfc = np.full(self.n, _steps(p['t_rfc'], dt_ms), np.int32)
        self.rfc[self.idx_in] = 0  # as model.poi: no refractory period for Poisson targets
        self.delay = _steps(p['t_dly'], dt_ms)
        self.D = self.delay + 1
        self.ev, self.eg = math.exp(-dt_ms / p['t_mbr']), math.exp(-dt_ms / p['tau'])
        self.cvg = p['tau'] / (p['tau'] - p['t_mbr']) * (self.eg - self.ev)  # exact solution's g -> v coupling
        self.kick = p['w_syn'] * p['f_poi']
        self.p_in = np.zeros(len(self.idx_in), np.float64)
        self.rng = np.array([seed or int.from_bytes(os.urandom(8), 'little') or 1], np.uint64)
        self._spk = np.zeros(self.n, np.int32)
        self.reset()

    def set_rates(self, rates_hz) -> None:
        """Poisson rate per input neuron (in the order of idx_in), in Hz."""
        self.p_in[:] = np.asarray(rates_hz, np.float64) * (self.dt_ms / 1000.0)

    def reset(self) -> None:
        """Every neuron at rest, nothing in flight (a new trial)."""
        self.v = np.full(self.n, self.p['v_0'], np.float64)
        self.g = np.zeros(self.n, np.float64)
        self.last = np.full(self.n, -(1 << 40), np.int64)
        self.ring = np.zeros(self.D * self.n, np.float64)
        self.t = 0

    def run(self, duration_ms: float, in_spikes: np.ndarray | None = None) -> np.ndarray:
        """Advance the network; spike counts of every neuron during this run. in_spikes (steps x inputs, 0/1):
        given input spikes instead of the Poisson draws (check_fast.py: the same input as Brian2's)."""
        steps = _steps(duration_ms, self.dt_ms)
        counts = np.zeros(self.n, np.int32)
        spikes = None
        if in_spikes is not None:
            spikes = np.ascontiguousarray(in_spikes[:steps], np.uint8)
        self._run(self.n, steps, self.t, self.v, self.g, self.last, self.rfc,
                  self.ev, self.eg, self.cvg, self.p['v_0'], self.p['v_rst'], self.p['v_th'],
                  self.row_ptr, self.post, self.w, self.ring, self.D, self.delay,
                  len(self.idx_in), self.idx_in, self.p_in, self.kick,
                  None if spikes is None else spikes.ctypes.data, self.rng, counts, self._spk)
        self.t += steps
        return counts.astype(np.int64)
