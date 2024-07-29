"""This tests the CLI functionality of training and predicting a regression model using a nested cross-validation strategy.
"""

import pytest

from baseprop.cli.hpopt import NO_HYPEROPT, NO_RAY
from baseprop.cli.main import main

pytestmark = pytest.mark.CLI


@pytest.fixture
def data_path(data_dir):
    return str(data_dir / "freesolv.csv")


@pytest.mark.skipif(NO_RAY or NO_HYPEROPT, reason="Ray and/or Hyperopt not installed")
def test_nestedCV_GCNN_quick(monkeypatch, data_path, tmp_path):
    args = [
        "baseprop",
        "nestedCV",
        "-i",
        data_path,
        "--smiles-columns",
        "smiles",
        "--target-columns",
        "freesolv",
        "--accelerator",
        "cpu",
        "--epochs",
        "1",
        "--num-workers",
        "0",
        "--task-type",
        "regression",
        "--metric",
        "mse",
        "rmse",
        "--show-individual-scores",
        "--save-dir",
        str(tmp_path),
        "--raytune-num-samples",
        "2",
        "--raytune-search-algorithm",
        "hyperopt",
        "--molecule-featurizers",
        "morgan_count",
        "--search-parameter-keywords",
        "all",
        "--split-type",
        "cv",
        "--num-folds",
        "3",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


@pytest.mark.skipif(NO_RAY or NO_HYPEROPT, reason="Ray and/or Hyperopt not installed")
def test_nestedCV_DNN_quick(monkeypatch, data_path, tmp_path):
    args = [
        "baseprop",
        "nestedCV",
        "-i",
        data_path,
        "--smiles-columns",
        "smiles",
        "--target-columns",
        "freesolv",
        "--accelerator",
        "cpu",
        "--epochs",
        "1",
        "--num-workers",
        "0",
        "--task-type",
        "regression",
        "--metric",
        "mse",
        "rmse",
        "--show-individual-scores",
        "--save-dir",
        str(tmp_path),
        "--raytune-num-samples",
        "2",
        "--raytune-search-algorithm",
        "hyperopt",
        "--molecule-featurizers",
        "morgan_count",
        "--search-parameter-keywords",
        "all",
        "--split-type",
        "cv",
        "--num-folds",
        "3",
        "--features-only",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()
