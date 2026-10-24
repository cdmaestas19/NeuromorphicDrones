"""The fly connectome and its cell-type labels, so neurons can be picked by name ("LC4", "DNp01").

Data, all CC-BY 4.0 (cite them in anything that uses results from this code):
- neurons and connectivity: FlyWire (Dorkenwald et al. 2024, Nature 634:124), as packaged for the LIF
  model of Shiu et al. 2024 (Nature 634:210) in the Drosophila_brain_model repo. Version 783 by default;
  630 is the version the paper used.
- cell types: FlyWire annotations (Schlegel et al. 2024, Nature 634:139), flyconnectome/flywire_annotations.
  They are for version 783.
"""
import os
import sys
from pathlib import Path

import pandas as pd

BRAIN_MODEL_DIR = Path(os.environ.get('BRAIN_MODEL_DIR', Path.home() / 'Drosophila_brain_model'))
sys.path.insert(0, str(BRAIN_MODEL_DIR))  # so `import model` finds the model repo's Brian2 model
ANNOTATIONS = (Path(os.environ.get('ANNOTATIONS_DIR', Path.home() / 'flywire_annotations')) /
               'supplemental_files' / 'Supplemental_file1_neuron_annotations.tsv')
VERSIONS = {  # version -> (neuron list, connectivity), files in BRAIN_MODEL_DIR
    '783': ('Completeness_783.csv', 'Connectivity_783.parquet'),
    '630': ('2023_03_23_completeness_630_final.csv', '2023_03_23_connectivity_630_final.parquet'),
}
GIANT_FIBER = 'DNp01'


class Connectome:
    """Neuron list, connectivity and labels of one FlyWire version. The model's neuron index is the row
    of the neuron list: Connectivity's Presynaptic_Index / Postsynaptic_Index refer to it."""

    def __init__(self, version: str = '783'):
        self.version = version
        comp, con = VERSIONS[version]
        self.path_comp, self.path_con = BRAIN_MODEL_DIR / comp, BRAIN_MODEL_DIR / con
        self.ids = pd.read_csv(self.path_comp, index_col=0).index.to_numpy()
        self.index = {fid: i for i, fid in enumerate(self.ids)}
        self._con = self._labels = None

    @property
    def connectivity(self) -> pd.DataFrame:
        """One row per connected pair: indices, synapse count, sign (+1 excitatory, -1 inhibitory) and
        their product, which the model uses as the weight in units of w_syn."""
        if self._con is None:
            self._con = pd.read_parquet(self.path_con)
        return self._con

    @property
    def labels(self) -> pd.DataFrame:
        """cell_type, side, super_class and top_nt per FlyWire root id (version 783 only)."""
        if self._labels is None:
            if self.version != '783':
                raise ValueError('the FlyWire annotations are for version 783')
            df = pd.read_csv(ANNOTATIONS, sep='\t', low_memory=False,
                             usecols=['root_id', 'cell_type', 'side', 'super_class', 'top_nt'])
            self._labels = df.set_index('root_id')
        return self._labels

    def cells(self, cell_type: str, side: str | None = None) -> list:
        """FlyWire ids of every neuron of a cell type (optionally one side: 'left' / 'right') in this model."""
        lab = self.labels
        sel = lab[(lab.cell_type == cell_type) & ((lab.side == side) if side else True)]
        return [int(f) for f in sel.index if f in self.index]

    def name(self, fid: int) -> str:
        """'LC4 (left)' for a FlyWire id, or the id itself when it has no cell type."""
        try:
            row = self.labels.loc[fid]
        except KeyError:
            return str(fid)
        return f'{row.cell_type} ({row.side})' if isinstance(row.cell_type, str) else str(fid)
