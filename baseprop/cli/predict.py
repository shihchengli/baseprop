import logging
import sys
from argparse import ArgumentError, ArgumentParser, Namespace
from pathlib import Path
from typing import Iterator

import pandas as pd
import torch
from lightning import pytorch as pl

from baseprop import data
from baseprop.cli.common import add_common_args
from baseprop.cli.utils import Subcommand, build_data_from_files
from baseprop.data import MoleculeDataset
from baseprop.featurizers.atom import get_multi_hot_atom_featurizer
from baseprop.featurizers.bond import MultiHotBondFeaturizer
from baseprop.models import load_model

logger = logging.getLogger(__name__)


class PredictSubcommand(Subcommand):
    COMMAND = "predict"
    HELP = "use a pretrained baseprop model for prediction"

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        parser = add_common_args(parser)
        return add_predict_args(parser)

    @classmethod
    def func(cls, args: Namespace):
        args = process_predict_args(args)
        main(args)


def add_predict_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "-i",
        "--test-path",
        required=True,
        type=Path,
        help="Path to an input CSV file containing SMILES.",
    )
    parser.add_argument(
        "-o",
        "--output",
        "--preds-path",
        type=Path,
        help="Path to which predictions will be saved. If the file extension is .pkl, will be saved as a pickle file. Otherwise, will save predictions as a CSV. If multiple models are used to make predictions, the average predictions will be saved in the file, and another file ending in '_individual' with the same file extension will save the predictions for each individual model, with the column names being the target names appended with the model index (e.g., '_model_<index>').",
    )
    parser.add_argument(
        "--drop-extra-columns",
        action="store_true",
        help="Whether to drop all columns from the test data file besides the SMILES columns and the new prediction columns.",
    )
    parser.add_argument(
        "--model-paths",
        "--model-path",
        required=True,
        type=Path,
        nargs="+",
        help="Location of checkpoint(s) or model file(s) to use for prediction. It can be a path to either a single pretrained model checkpoint (.ckpt) or single pretrained model file (.pt), a directory that contains these files, or a list of path(s) and directory(s). If a directory, will recursively search and predict on all found (.pt) models.",
    )
    parser.add_argument(
        "--target-columns",
        nargs="+",
        help="Column names to save the predictions to. If not provided, the predictions will be saved to columns named 'pred_0', 'pred_1', etc.",
    )

    return parser


def process_predict_args(args: Namespace) -> Namespace:
    if args.test_path.suffix not in [".csv"]:
        raise ArgumentError(
            argument=None,
            message=f"Input data must be a CSV file. Got {args.test_path}",
        )
    if args.output is None:
        args.output = args.test_path.parent / (args.test_path.stem + "_preds.csv")
    if args.output.suffix not in [".csv", ".pkl"]:
        raise ArgumentError(
            argument=None,
            message=f"Output must be a CSV or Pickle file. Got {args.output}",
        )
    return args


def find_models(model_paths: list[Path]):
    collected_model_paths = []

    for model_path in model_paths:
        if model_path.suffix in [".ckpt", ".pt"]:
            collected_model_paths.append(model_path)
        elif model_path.is_dir():
            collected_model_paths.extend(list(model_path.rglob("*.pt")))
        else:
            raise ArgumentError(
                argument=None,
                message=f"Model path must be a .ckpt, .pt file, or a directory. Got {model_path}",
            )

    return collected_model_paths


def make_prediction_for_models(
    args: Namespace, model_paths: Iterator[Path], output_path: Path
):
    model = load_model(model_paths[0])
    format_kwargs = dict(
        no_header_row=args.no_header_row,
        smiles_cols=args.smiles_columns,
        target_cols=[],
        splits_col=[],
    )
    featurization_kwargs = dict(
        molecule_featurizers=args.molecule_featurizers,
        keep_h=args.keep_h,
        add_h=args.add_h,
    )

    test_data = build_data_from_files(
        args.test_path,
        **format_kwargs,
        **featurization_kwargs,
    )
    logger.info(f"test size: {len(test_data)}")

    atom_featurizer = get_multi_hot_atom_featurizer("V2")
    bond_featurizer = MultiHotBondFeaturizer()

    test_dset = MoleculeDataset(test_data, atom_featurizer, bond_featurizer)
    test_loader = data.build_dataloader(
        test_dset, args.batch_size, args.num_workers, shuffle=False
    )

    individual_preds = []
    for model_path in model_paths:
        logger.info(f"Predicting with model at '{model_path}'")

        model = load_model(model_path)

        logger.info(model)

        trainer = pl.Trainer(
            logger=False,
            enable_progress_bar=True,
            accelerator=args.accelerator,
            devices=args.devices,
        )

        predss = trainer.predict(model, test_loader)

        preds = torch.concat(predss, 0)
        individual_preds.append(preds)

    average_preds = torch.mean(torch.stack(individual_preds).float(), dim=0)
    if args.target_columns is not None:
        assert (
            len(args.target_columns) == model.n_tasks
        ), "Number of target columns must match the number of tasks."
        target_columns = args.target_columns
    else:
        target_columns = [f"pred_{i}" for i in range(preds.shape[1])]

    df_test = pd.read_csv(args.test_path)
    df_test[target_columns] = average_preds
    if output_path.suffix == ".pkl":
        df_test = df_test.reset_index(drop=True)
        df_test.to_pickle(output_path)
    else:
        df_test.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to '{output_path}'")

    if len(model_paths) > 1:
        individual_preds = torch.concat(individual_preds, 1)
        target_columns = [
            f"{col}_model_{i}"
            for i in range(len(model_paths))
            for col in target_columns
        ]

        df_test = pd.read_csv(args.test_path)
        df_test[target_columns] = individual_preds

        output_path = output_path.parent / Path(
            str(args.output.stem) + "_individual" + str(output_path.suffix)
        )
        if output_path.suffix == ".pkl":
            df_test = df_test.reset_index(drop=True)
            df_test.to_pickle(output_path)
        else:
            df_test.to_csv(output_path, index=False)
        logger.info(f"Individual predictions saved to '{output_path}'")
        for i, model_path in enumerate(model_paths):
            logger.info(
                f"Results from model path {model_path} are saved under the column name ending with 'model_{i}'"
            )


def main(args):
    model_paths = find_models(args.model_paths)

    make_prediction_for_models(args, model_paths, output_path=args.output)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = PredictSubcommand.add_args(parser)

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
    args = parser.parse_args()
    args = PredictSubcommand.func(args)
