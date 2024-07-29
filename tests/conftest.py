from pathlib import Path

import pandas as pd
import pytest

_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def data_dir():
    return _DATA_DIR


@pytest.fixture
def mol_regression_data(data_dir):
    df = pd.read_csv(data_dir / "data/freesolv.csv")
    smis = df["smiles"].to_list()
    Y = df["freesolv"].to_numpy().reshape(-1, 1)

    return smis, Y


@pytest.fixture
def mol_classification_data(data_dir):
    df = pd.read_csv(data_dir / "data/bace.csv")
    smis = df["mol"].to_list()
    Y = df["Class"].to_numpy().reshape(-1, 1)

    return smis, Y
