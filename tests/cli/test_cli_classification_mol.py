"""This tests the CLI functionality of training and predicting a classification model on a single molecule."""

import pytest

from baseprop.cli.main import main
from baseprop.models.model import LitModule

pytestmark = pytest.mark.CLI


@pytest.fixture
def data_path(data_dir):
    return str(data_dir / "bace.csv")


def test_train_predict_quick(monkeypatch, data_path, tmp_path):
    args = [
        "baseprop",
        "train",
        "-i",
        data_path,
        "--smiles-columns",
        "mol",
        "--target-columns",
        "Class",
        "--accelerator",
        "cpu",
        "--epochs",
        "1",
        "--num-workers",
        "0",
        "--task-type",
        "classification",
        "--metric",
        "prc",
        "accuracy",
        "f1",
        "roc",
        "--show-individual-scores",
        "--save-dir",
        str(tmp_path),
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    args = [
        "baseprop",
        "predict",
        "-i",
        data_path,
        "--accelerator",
        "cpu",
        "--model-path",
        str(tmp_path / "model_0/best.pt"),
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_output_structure(monkeypatch, data_path, tmp_path):
    args = [
        "baseprop",
        "train",
        "-i",
        data_path,
        "--smiles-columns",
        "mol",
        "--target-columns",
        "Class",
        "--accelerator",
        "cpu",
        "--epochs",
        "1",
        "--num-workers",
        "0",
        "--save-dir",
        str(tmp_path),
        "--save-smiles-splits",
        "--task-type",
        "classification",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "model_0" / "best.pt").exists()
    assert (tmp_path / "model_0" / "checkpoints" / "last.ckpt").exists()
    assert (tmp_path / "model_0" / "trainer_logs" / "version_0").exists()
    assert (tmp_path / "train_smiles.csv").exists()


def test_train_output_structure_cv_ensemble(monkeypatch, data_path, tmp_path):
    args = [
        "baseprop",
        "train",
        "-i",
        data_path,
        "--smiles-columns",
        "mol",
        "--target-columns",
        "Class",
        "--accelerator",
        "cpu",
        "--epochs",
        "1",
        "--num-workers",
        "0",
        "--save-dir",
        str(tmp_path),
        "--save-smiles-splits",
        "--split-type",
        "cv",
        "--num-folds",
        "3",
        "--ensemble-size",
        "2",
        "--task-type",
        "classification",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "fold_2" / "model_1" / "best.pt").exists()
    assert (tmp_path / "fold_2" / "model_1" / "checkpoints" / "last.ckpt").exists()
    assert (tmp_path / "fold_2" / "model_1" / "trainer_logs" / "version_0").exists()
    assert (tmp_path / "fold_2" / "train_smiles.csv").exists()


def test_train_outputs(monkeypatch, data_path, tmp_path):
    args = [
        "baseprop",
        "train",
        "-i",
        data_path,
        "--smiles-columns",
        "mol",
        "--target-columns",
        "Class",
        "--accelerator",
        "cpu",
        "--epochs",
        "1",
        "--num-workers",
        "0",
        "--save-dir",
        str(tmp_path),
        "--task-type",
        "classification",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    checkpoint_path = tmp_path / "model_0" / "checkpoints" / "last.ckpt"

    model = LitModule.load_from_checkpoint(checkpoint_path)
    assert model is not None
