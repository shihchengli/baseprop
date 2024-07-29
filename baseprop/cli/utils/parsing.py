from os import PathLike
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from baseprop.data.datapoints import MoleculeDatapoint
from baseprop.featurizers.molecule import MoleculeFeaturizerRegistry
from baseprop.utils.utils import make_mol


def make_datapoints(
    smiss: list[list[str]] | None,
    Y: np.ndarray,
    molecule_featurizers: list[str] | None,
    keep_h: bool,
    add_h: bool,
) -> list[MoleculeDatapoint]:
    """Make the :class:`MoleculeDatapoint`s for a given dataset.

    Parameters
    ----------
    smiss : list[list[str]] | None
        a list of ``j`` lists of ``n`` SMILES strings, where ``j`` is the number of molecules per
        datapoint and ``n`` is the number of datapoints. If ``None``, the corresponding list of
        :class:`MoleculeDatapoint`\s will be empty.
    Y : np.ndarray
        the target values of shape ``n x m``, where ``m`` is the number of targets
    molecule_featurizers : list[str] | None
        a list of molecule featurizer names to generate additional molecule features to use as extra
        descriptors. If there are multiple molecules per datapoint, the featurizers will be applied
        to each molecule and concatenated. Note that a :code:`ReactionDatapoint` has two
        RDKit :class:`~rdkit.Chem.Mol` objects, reactant(s) and product(s). Each
        ``molecule_featurizer`` will be applied to both of these objects.
    keep_h : bool
    add_h : bool

    Returns
    -------
    list[MoleculeDatapoint]
        a list of ``j`` lists of ``n`` :class:`MoleculeDatapoint`\s
    .. note::
        either ``j`` or ``k`` may be 0, in which case the corresponding list will be empty.

    Raises
    ------
    ValueError
        if ``smiss`` is ``None``.
    """
    if smiss is None:
        raise ValueError("args 'smiss' was `None`!")
    else:
        N = len(smiss[0])

    molss = [[make_mol(smi, keep_h, add_h) for smi in smis] for smis in smiss]

    if molecule_featurizers is None:
        X_d = [None] * N
    else:
        molecule_featurizers = [MoleculeFeaturizerRegistry[mf]() for mf in molecule_featurizers]

        if len(smiss) > 0:
            mol_descriptors = np.hstack(
                [
                    np.vstack([np.hstack([mf(mol) for mf in molecule_featurizers]) for mol in mols])
                    for mols in molss
                ]
            )
            X_d = mol_descriptors

    mol_data = [
        [
            MoleculeDatapoint(mol=molss[mol_idx][i], name=smis[i], y=Y[i], x_d=X_d[i])
            for i in range(N)
        ]
        for mol_idx, smis in enumerate(smiss)
    ][0]

    return mol_data


def parse_csv(
    path: PathLike,
    smiles_cols: Sequence[str] | None,
    target_cols: Sequence[str] | None,
    splits_col: Sequence[str] | None,
    no_header_row: bool = False,
):
    df = pd.read_csv(path, header=None if no_header_row else "infer", index_col=False)

    if smiles_cols is not None:
        smiss = df[smiles_cols].T.values.tolist()
        input_cols = smiles_cols
    else:
        smiss = df.iloc[:, [0]].T.values.tolist()
        input_cols = [df.columns[0]]

    if target_cols is None:
        target_cols = list(
            column for column in df.columns if column not in set(input_cols + (splits_col or []))
        )

    Y = df[target_cols]
    Y = Y.to_numpy(np.single)

    return smiss, Y


def build_data_from_files(
    p_data: PathLike,
    no_header_row: bool,
    smiles_cols: Sequence[str] | None,
    target_cols: Sequence[str] | None,
    splits_col: str | None,
    **featurization_kwargs: Mapping,
) -> list[list[MoleculeDatapoint]]:
    smiss, Y = parse_csv(p_data, smiles_cols, target_cols, splits_col, no_header_row)

    mol_data = make_datapoints(smiss, Y, **featurization_kwargs)

    return mol_data


def parse_indices(idxs):
    """Parses a string of indices into a list of integers. e.g. '0,1,2-4' -> [0, 1, 2, 3, 4]"""
    if isinstance(idxs, str):
        indices = []
        for idx in idxs.split(","):
            if "-" in idx:
                start, end = map(int, idx.split("-"))
                indices.extend(range(start, end + 1))
            else:
                indices.append(int(idx))
        return indices
    return idxs
