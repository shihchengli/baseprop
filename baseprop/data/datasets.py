from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from baseprop.data.datapoints import MoleculeDatapoint
from baseprop.data.molgraph import MolGraph
from baseprop.featurizers.atom import MultiHotAtomFeaturizer
from baseprop.featurizers.bond import MultiHotBondFeaturizer


class Datum(NamedTuple):
    """a singular training data point"""

    mg: MolGraph
    x_d: np.ndarray | None
    y: np.ndarray | None


@dataclass
class MoleculeDataset(Dataset):
    data: list[MoleculeDatapoint]
    atom_featurizer: MultiHotAtomFeaturizer
    bond_featurizer: MultiHotBondFeaturizer

    def __post_init__(self):
        if self.data is None:
            raise ValueError("Data cannot be None!")

        self.reset()

    @property
    def smiles(self) -> list[str]:
        """the SMILES strings associated with the dataset"""
        return [Chem.MolToSmiles(d.mol) for d in self.data]

    @property
    def mols(self) -> list[Chem.Mol]:
        """the molecules associated with the dataset"""
        return [d.mol for d in self.data]

    @cached_property
    def _Y(self) -> np.ndarray:
        """the raw targets of the dataset"""
        return np.array([d.y for d in self.data], float)

    @property
    def Y(self) -> np.ndarray:
        """the (scaled) targets of the dataset"""
        return self.__Y

    @Y.setter
    def Y(self, Y: ArrayLike):
        self._validate_attribute(Y, "targets")

        self.__Y = np.array(Y, float)

    @cached_property
    def _X_d(self) -> np.ndarray:
        """the raw extra descriptors of the dataset"""
        return np.array([d.x_d for d in self.data])

    @property
    def X_d(self) -> np.ndarray:
        """the (scaled) extra descriptors of the dataset"""
        return self.__X_d

    @property
    def names(self) -> list[str]:
        return [d.name for d in self.data]

    @X_d.setter
    def X_d(self, X_d: ArrayLike):
        self._validate_attribute(X_d, "extra descriptors")

        self.__X_d = np.array(X_d)

    def _validate_attribute(self, X: np.ndarray, label: str):
        if not len(self.data) == len(X):
            raise ValueError(
                f"number of molecules ({len(self.data)}) and {label} ({len(X)}) "
                "must have same length!"
            )

    def reset(self):
        """Reset the atom and bond features; atom and extra descriptors; and targets of each
        datapoint to their initial, unnormalized values."""
        self.__Y = self._Y
        self.__X_d = self._X_d

    def normalize_targets(self, scaler: StandardScaler | None = None) -> StandardScaler:
        """Normalizes the targets of this dataset using a :obj:`StandardScaler`

        The :obj:`StandardScaler` subtracts the mean and divides by the standard deviation for
        each task independently. NOTE: This should only be used for regression datasets.

        Returns
        -------
        StandardScaler
            a scaler fit to the targets.
        """

        if scaler is None:
            scaler = StandardScaler().fit(self._Y)

        self.Y = scaler.transform(self._Y)

        return scaler

    def mol_to_molgraph(self, mol):
        """Converts RDKit Mol object to MolGraph."""
        n_bonds = mol.GetNumBonds()
        bond_fdim = len(self.bond_featurizer)

        V = np.array([self.atom_featurizer(a) for a in mol.GetAtoms()], dtype=np.single)
        E = np.empty((2 * n_bonds, bond_fdim))
        edge_index = [[], []]

        i = 0
        for bond in mol.GetBonds():
            x_e = self.bond_featurizer(bond)
            E[i : i + 2] = x_e

            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index[0].extend([u, v])
            edge_index[1].extend([v, u])

            i += 2

        edge_index = np.array(edge_index, int)

        return MolGraph(V=V, E=E, edge_index=edge_index)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Datum:
        d = self.data[idx]
        mol = d.mol
        mg = self.mol_to_molgraph(mol)

        return Datum(mg, self.X_d[idx], self.Y[idx])
