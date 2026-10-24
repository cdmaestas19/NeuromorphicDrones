"""The escape subcircuit: the part of the connectome between the looming detectors and escape.

The whole brain runs ~4-5x slower than real time on one core, too slow to fly with. Avoidance only needs
the pathway from the optic lobes' looming detectors (LC4, LPLC2: where the camera's input goes in) to the
descending neurons that trigger escape (the giant fibers DNp01, plus every descending neuron the looming
detectors synapse onto directly: the outputs). extract() cuts that pathway out: every neuron on a path of at
most `hops` synaptic steps from an input neuron to an output neuron, through connections of at least
`min_synapses` synapses, plus every connection among those neurons.

Subcircuit runs it with the whole-brain model's own equations and parameters (model.py of the model repo,
not a copy), with Poisson input whose rate can change between run() calls (the drone's camera drives it).
validate.py checks it against the whole brain.
"""
import numpy as np
import scipy.sparse as sp
from brian2 import Hz, Network, NeuronGroup, PoissonGroup, SpikeMonitor, Synapses, defaultclock, ms, mV

from connectome import GIANT_FIBER, Connectome
import model  # noqa: E402  (the model repo's Brian2 model; connectome put it on sys.path)

INPUT_TYPES = ('LC4', 'LPLC2')
MIN_OUTPUT_SYNAPSES = 20  # a descending neuron is an output if the looming detectors give it this many synapses


def outputs(conn: Connectome, min_synapses: int = MIN_OUTPUT_SYNAPSES) -> list:
    """The giant fibers plus every descending neuron that receives >= min_synapses synapses directly from the
    looming detectors (both sides), strongest first."""
    lab, con = conn.labels, conn.connectivity
    pre = {conn.index[f] for t in INPUT_TYPES for f in conn.cells(t)}
    dn = {conn.index[f] for f in lab.index[lab.super_class == 'descending'] if f in conn.index}
    e = con[con.Presynaptic_Index.isin(pre) & con.Postsynaptic_Index.isin(dn)]
    per_dn = e.groupby('Postsynaptic_Index').Connectivity.sum().sort_values(ascending=False)
    ids = [int(conn.ids[i]) for i, n in per_dn.items() if n >= min_synapses]
    for gf in conn.cells(GIANT_FIBER):
        if gf not in ids:
            ids.insert(0, gf)
    return ids


HOPS, MIN_SYNAPSES = 4, 5  # validated against the whole brain (validate.py): ~18k neurons, ~2.4M connections


def extract(conn: Connectome, hops: int = HOPS, min_synapses: int = MIN_SYNAPSES) -> tuple[list, list, list]:
    """(input ids, output ids, all ids of the subcircuit): neurons on input->output paths of <= hops steps."""
    con = conn.connectivity
    n = len(conn.ids)
    strong = con[con.Connectivity >= min_synapses]
    adj = sp.csr_matrix((np.ones(len(strong), bool), (strong.Presynaptic_Index, strong.Postsynaptic_Index)), shape=(n, n))
    ins = [f for t in INPUT_TYPES for f in conn.cells(t)]
    outs = outputs(conn)

    def distances(start: list, matrix) -> np.ndarray:
        d = np.full(n, np.iinfo(np.int32).max, np.int64)
        frontier = np.zeros(n, bool)
        frontier[[conn.index[f] for f in start]] = True
        d[frontier] = 0
        for k in range(1, hops + 1):
            frontier = (matrix.T @ frontier) & (d > k)  # one synaptic step further
            d[frontier] = k
        return d

    d_in = distances(ins, adj)          # steps downstream of an input
    d_out = distances(outs, adj.T.tocsr())  # steps upstream of an output
    keep = np.flatnonzero(d_in + d_out <= hops)
    ids = sorted(set(int(conn.ids[i]) for i in keep) | set(ins) | set(outs))
    return ins, outs, ids


class Subcircuit:
    """A Brian2 network of just these neurons and every connection among them, with the whole-brain model's
    equations and parameters. Input neurons get Poisson input (as model.poi does: one input per neuron, a
    w_syn x f_poi kick to v, no refractory period) at rates set_rates() can change between run() calls."""

    def __init__(self, conn: Connectome, ids: list, inputs: list, record: list, params: dict | None = None,
                 dt_ms: float = 0.1):
        """dt_ms: the integration step. 0.1 ms is the model's (Brian2's default); its equations are linear and
        integrated exactly, so a coarser step mostly coarsens spike timing (brain_server.py measures it)."""
        p = dict(model.default_params if params is None else params)
        self.ids, self.inputs, self.record = list(ids), list(inputs), list(record)
        local = {f: i for i, f in enumerate(self.ids)}
        glob = np.array([conn.index[f] for f in self.ids])
        con = conn.connectivity
        inside = np.zeros(len(conn.ids), np.int64) - 1
        inside[glob] = np.arange(len(glob))
        e = con[(inside[con.Presynaptic_Index.to_numpy()] >= 0) & (inside[con.Postsynaptic_Index.to_numpy()] >= 0)]
        self.n_connections = len(e)

        defaultclock.dt = dt_ms * ms
        self.neu = NeuronGroup(len(self.ids), model=p['eqs'], method='linear', threshold=p['eq_th'],
                               reset=p['eq_rst'], refractory='rfc', namespace=p)
        self.neu.v, self.neu.g, self.neu.rfc = p['v_0'], 0 * mV, p['t_rfc']
        self.syn = Synapses(self.neu, self.neu, 'w : volt', on_pre='g += w', delay=p['t_dly'])
        self.syn.connect(i=inside[e.Presynaptic_Index.to_numpy()], j=inside[e.Postsynaptic_Index.to_numpy()])
        self.syn.w = e['Excitatory x Connectivity'].to_numpy() * p['w_syn']

        idx_in = np.array([local[f] for f in self.inputs])
        self.neu.rfc[idx_in] = 0 * ms  # as model.poi: no refractory period for Poisson targets
        self.poisson = PoissonGroup(len(self.inputs), rates=0 * Hz)
        self.kick = Synapses(self.poisson, self.neu, on_pre=f'v += {float(p["w_syn"] * p["f_poi"] / mV)}*mV')
        self.kick.connect(i=np.arange(len(self.inputs)), j=idx_in)

        self._rec = np.array([local[f] for f in self.record])
        self.mon = SpikeMonitor(self.neu, record=False)  # per-neuron counts only: no memory growth in flight
        self.net = Network(self.neu, self.syn, self.poisson, self.kick, self.mon)
        self._last = np.zeros(len(self.ids), np.int64)
        self.net.store('initial')

    def set_rates(self, rates_hz) -> None:
        """Poisson rate per input neuron (in the order of `inputs`), in Hz."""
        self.poisson.rates = np.asarray(rates_hz, float) * Hz

    def reset(self) -> None:
        """Back to rest (a new trial)."""
        self.net.restore('initial')
        self._last = np.asarray(self.mon.count, np.int64).copy()

    def run(self, duration_ms: float) -> np.ndarray:
        """Advance the network; spike counts of every neuron (in the order of `ids`) during this run."""
        self.net.run(duration_ms * ms)
        now = np.asarray(self.mon.count, np.int64)
        counts, self._last = now - self._last, now.copy()
        return counts

    def recorded(self, counts: np.ndarray) -> np.ndarray:
        """The `record` neurons' entries of run()'s counts."""
        return counts[self._rec]
