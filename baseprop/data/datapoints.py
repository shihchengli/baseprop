from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit.Chem import AllChem as Chem

from baseprop.utils.utils import make_mol


@dataclass
class MoleculeDatapoint:
    mol: Chem.Mol
    """the molecule associated with this datapoint"""
    y: np.ndarray | None = None
    """the targets for the molecule with unknown targets indicated by `nan`s"""
    x_d: np.ndarray | None = None
    """A vector of length ``d_f`` containing additional features (e.g., Morgan fingerprint) that
    will be concatenated to the global representation *after* aggregation"""
    name: str | None = None
    """A string identifier for the datapoint."""

    @classmethod
    def from_smi(
        cls, smi: str, *args, keep_h: bool = False, add_h: bool = False, **kwargs
    ) -> MoleculeDatapoint:
        mol = make_mol(smi, keep_h, add_h)

        kwargs["name"] = smi if "name" not in kwargs else kwargs["name"]

        return cls(mol, *args, **kwargs)
