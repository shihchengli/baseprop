import json
import logging
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
from configargparse import ArgumentError, ArgumentParser, Namespace
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy

from baseprop.cli.common import add_common_args
from baseprop.cli.conf import NOW
from baseprop.cli.utils import (LookupAction, Subcommand,
                                build_data_from_files, parse_indices)
from baseprop.cli.utils.args import uppercase
from baseprop.data import (MoleculeDataset, SplitType, build_dataloader,
                           make_split_indices, split_data_by_indices)
from baseprop.featurizers.atom import get_multi_hot_atom_featurizer
from baseprop.featurizers.bond import MultiHotBondFeaturizer
from baseprop.models import save_model
from baseprop.models.model import LitModule
from baseprop.nn import (GCN, LossFunctionRegistry, MetricRegistry,
                         PredictorRegistry)
from baseprop.nn.transforms import UnscaleTransform
from baseprop.nn.utils import Activation
from baseprop.utils import Factory

logger = logging.getLogger(__name__)


class TrainSubcommand(Subcommand):
    COMMAND = "train"
    HELP = "train a baseprop model"
    parser = None

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        parser = add_common_args(parser)
        parser = add_train_args(parser)
        cls.parser = parser
        return parser

    @classmethod
    def func(cls, args: Namespace):
        args = process_train_args(args)
        args.output_dir.mkdir(exist_ok=True, parents=True)
        config_path = args.output_dir / "config.toml"
        save_config(cls.parser, args, config_path)
        main(args)


def add_train_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--config-path",
        type=Path,
        is_config_file=True,
        help="Path to a configuration file. Command line arguments override values in the configuration file.",
    )
    parser.add_argument(
        "-i",
        "--data-path",
        type=Path,
        help="Path to an input CSV file containing SMILES and the associated target values.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        "--save-dir",
        type=Path,
        help="Directory where training outputs will be saved. Defaults to 'CURRENT_DIRECTORY/baseprop_training/STEM_OF_INPUT/TIME_STAMP'.",
    )

    transfer_args = parser.add_argument_group("transfer learning args")
    transfer_args.add_argument(
        "--model-frzn",
        help="Path to model checkpoint file to be loaded for overwriting and freezing weights.",
    )
    transfer_args.add_argument(
        "--frzn-ffn-layers",
        type=int,
        default=0,
        help="Overwrites weights for the first n layers of the ffn from checkpoint model (specified checkpoint_frzn), where n is specified in the input. Automatically also freezes mpnn weights.",
    )

    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=1,
        help="Number of models in ensemble for each splitting of data.",
    )

    mp_args = parser.add_argument_group("GCN")
    mp_args.add_argument(
        "--hidden_channels",
        type=int,
        default=300,
        help="hidden dimension of the GCN",
    )
    mp_args.add_argument("--depth", type=int, default=3, help="Number of GCN layers.")
    mp_args.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="dropout probability in GCN/FFN layers",
    )
    mp_args.add_argument(
        "--activation",
        type=uppercase,
        default="RELU",
        choices=list(Activation.keys()),
        help="activation function in GCN/FFN layers",
    )

    ffn_args = parser.add_argument_group("FFN args")
    ffn_args.add_argument(
        "--ffn-hidden-dim",
        type=int,
        default=300,
        help="hidden dimension in the FFN top model",
    )
    ffn_args.add_argument(
        "--ffn-num-layers",
        type=int,
        default=1,
        help="number of layers in FFN top model",
    )

    ffn_args.add_argument(
        "--features-only",
        action="store_true",
        help="Use only the additional features in an FFN, no graph network.",
    )

    extra_mpnn_args = parser.add_argument_group("extra MPNN args")
    extra_mpnn_args.add_argument(
        "--no-batch-norm",
        action="store_true",
        help="Don't use batch normalization.",
    )

    train_data_args = parser.add_argument_group("training input data args")
    train_data_args.add_argument(
        "--target-columns",
        nargs="+",
        help="Name of the columns containing target values. By default, uses all columns except the SMILES column and the :code:`ignore_columns`.",
    )

    train_args = parser.add_argument_group("training args")
    train_args.add_argument(
        "-t",
        "--task-type",
        default="regression",
        action=LookupAction(PredictorRegistry),
        help="Type of dataset. This determines the default loss function used during training. Defaults to regression.",
    )
    train_args.add_argument(
        "-l",
        "--loss-function",
        action=LookupAction(LossFunctionRegistry),
        help="Loss function to use during training. If not specified, will use the default loss function for the given task type (see documentation).",
    )

    train_args.add_argument(
        "--metrics",
        "--metric",
        nargs="+",
        action=LookupAction(MetricRegistry),
        help="evaluation metrics. If unspecified, will use the following metrics for given dataset types: regression->rmse, classification->roc, multiclass->ce ('cross entropy'), spectral->sid. If multiple metrics are provided, the 0th one will be used for early stopping and checkpointing",
    )
    train_args.add_argument(
        "--show-individual-scores",
        action="store_true",
        help="Show all scores for individual targets, not just average, at the end.",
    )

    train_args.add_argument(
        "--lr", type=float, default=1e-4, help="Initial learning rate."
    )
    train_args.add_argument(
        "--epochs", type=int, default=50, help="the number of epochs to train over"
    )
    train_args.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Number of epochs to wait for improvement before early stopping.",
    )
    train_args.add_argument(
        "--grad-clip",
        type=float,
        help="Passed directly to the lightning trainer which controls grad clipping. See the :code:`Trainer()` docstring for details.",
    )

    split_args = parser.add_argument_group("split args")
    split_args.add_argument(
        "--split",
        "--split-type",
        type=uppercase,
        default="RANDOM",
        choices=list(SplitType.keys()),
        help="Method of splitting the data into train/val/test (case insensitive).",
    )
    split_args.add_argument(
        "--split-sizes",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        help="Split proportions for train/validation/test sets.",
    )
    split_args.add_argument(
        "-k",
        "--num-folds",
        type=int,
        default=1,
        help="Number of folds when performing cross validation.",
    )
    split_args.add_argument(
        "--save-smiles-splits",
        action="store_true",
        help="Save smiles for each train/val/test splits for prediction convenience later.",
    )
    split_args.add_argument(
        "--splits-file",
        type=Path,
        help="Path to a JSON file containing pre-defined splits for the input data, formatted as a list of dictionaries with keys 'train', 'val', and 'test' and values as lists of indices or strings formatted like '0-2,4'. See documentation for more details.",
    )
    train_data_args.add_argument(
        "--splits-column",
        help="Name of the column in the input CSV file containing 'train', 'val', or 'test' for each row.",
    )
    split_args.add_argument(
        "--data-seed",
        type=int,
        default=0,
        help="Random seed to use when splitting data into train/val/test sets. When :code`num_folds > 1`, the first fold uses this seed and all subsequent folds add 1 to the seed. Also used for shuffling data in :code:`build_dataloader` when :code:`shuffle` is True.",
    )

    parser.add_argument(
        "--pytorch-seed",
        type=int,
        default=None,
        help="Seed for PyTorch randomness (e.g., random initial weights).",
    )

    return parser


def process_train_args(args: Namespace) -> Namespace:
    if args.config_path is None and args.data_path is None:
        raise ArgumentError(
            argument=None, message="Data path must be provided for training."
        )

    if args.data_path.suffix not in [".csv"]:
        raise ArgumentError(
            argument=None,
            message=f"Input data must be a CSV file. Got {args.data_path}",
        )
    if args.output_dir is None:
        args.output_dir = Path(f"baseprop_training/{args.data_path.stem}/{NOW}")

    return args


def save_config(parser: ArgumentParser, args: Namespace, config_path: Path):
    config_args = deepcopy(args)
    for key, value in vars(config_args).items():
        if isinstance(value, Path):
            setattr(config_args, key, str(value))

    parser.write_config_file(
        parsed_namespace=config_args, output_file_paths=[str(config_path)]
    )


def save_smiles_splits(args: Namespace, output_dir, train_dset, val_dset, test_dset):
    train_smis = train_dset.smiles
    df_train = pd.DataFrame(train_smis, columns=args.smiles_columns)
    df_train.to_csv(output_dir / "train_smiles.csv", index=False)
    df_train = pd.DataFrame(train_smis, columns=args.smiles_columns)
    for i, target in enumerate(args.target_columns):
        df_train[target] = train_dset._Y[:, i]
    df_train.to_csv(output_dir / "train_full.csv", index=False)

    val_smis = val_dset.smiles
    df_val = pd.DataFrame(val_smis, columns=args.smiles_columns)
    df_val.to_csv(output_dir / "val_smiles.csv", index=False)
    for i, target in enumerate(args.target_columns):
        df_val[target] = val_dset._Y[:, i]
    df_val.to_csv(output_dir / "val_full.csv", index=False)

    if test_dset is not None:
        test_smis = test_dset.smiles
        df_test = pd.DataFrame(test_smis, columns=args.smiles_columns)
        df_test.to_csv(output_dir / "test_smiles.csv", index=False)
        for i, target in enumerate(args.target_columns):
            df_test[target] = test_dset._Y[:, i]
        df_test.to_csv(output_dir / "test_full.csv", index=False)


def build_splits(args, format_kwargs, featurization_kwargs):
    """build the train/val/test splits"""
    logger.info(f"Pulling data from file: {args.data_path}")
    all_data = build_data_from_files(
        args.data_path,
        **format_kwargs,
        **featurization_kwargs,
    )

    if args.splits_column is not None:
        df = pd.read_csv(
            args.data_path,
            header=None if args.no_header_row else "infer",
            index_col=False,
        )
        grouped = df.groupby(df[args.splits_column].str.lower())
        train_indices = grouped.groups.get("train", pd.Index([])).tolist()
        val_indices = grouped.groups.get("val", pd.Index([])).tolist()
        test_indices = grouped.groups.get("test", pd.Index([])).tolist()
        train_indices, val_indices, test_indices = (
            [train_indices],
            [val_indices],
            [test_indices],
        )

    elif args.splits_file is not None:
        with open(args.splits_file, "rb") as json_file:
            split_idxss = json.load(json_file)
        train_indices = [parse_indices(d["train"]) for d in split_idxss]
        val_indices = [parse_indices(d["val"]) for d in split_idxss]
        test_indices = [parse_indices(d["test"]) for d in split_idxss]

    else:
        splitting_mols = [datapoint.mol for datapoint in all_data]
        train_indices, val_indices, test_indices = make_split_indices(
            splitting_mols, args.split, args.split_sizes, args.data_seed, args.num_folds
        )
        if not (
            SplitType.get(args.split) == SplitType.CV_NO_VAL
            or SplitType.get(args.split) == SplitType.CV
        ):
            train_indices, val_indices, test_indices = (
                [train_indices],
                [val_indices],
                [test_indices],
            )
    train_data, val_data, test_data = split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )
    for i_split in range(len(train_data)):
        sizes = [
            len(train_data[i_split]),
            len(val_data[i_split]),
            len(test_data[i_split]),
        ]
        logger.info(f"train/val/test split_{i_split} sizes: {sizes}")

    return train_data, val_data, test_data


def train_model(
    args,
    train_loader,
    val_loader,
    test_loader,
    output_dir,
    output_transform,
):
    for model_idx in range(args.ensemble_size):
        model_output_dir = output_dir / f"model_{model_idx}"
        model_output_dir.mkdir(exist_ok=True, parents=True)

        if args.pytorch_seed is None:
            seed = torch.seed()
            deterministic = False
        else:
            seed = args.pytorch_seed + model_idx
            deterministic = True

        torch.manual_seed(seed)

        # build model
        if args.loss_function is not None:
            criterion = Factory.build(
                LossFunctionRegistry[args.loss_function],
            )
        else:
            criterion = None
        if args.metrics is not None:
            metrics = [Factory.build(MetricRegistry[metric]) for metric in args.metrics]
        else:
            metrics = None

        if args.features_only:
            encoder = None
            input_dim = 0
        else:
            encoder_cls = GCN
            encoder = Factory.build(
                encoder_cls,
                n_features=len(train_loader.dataset[0].mg.V[0]),
                hidden_channels=args.hidden_channels,
                dropout=args.dropout,
                num_gcn_layers=args.depth,
                batch_norm=not args.no_batch_norm,
                activation=args.activation,
            )
            input_dim = args.hidden_channels
        predictor_cls = PredictorRegistry[args.task_type]
        predictor = Factory.build(
            predictor_cls,
            input_dim=input_dim + len(train_loader.dataset[0].x_d)
            if args.molecule_featurizers
            else input_dim,
            n_tasks=train_loader.dataset[0].y.shape[0],
            hidden_dim=args.ffn_hidden_dim,
            n_layers=args.ffn_num_layers,
            dropout=args.dropout,
            activation=args.activation,
            criterion=criterion,
            output_transform=output_transform,
        )
        model = LitModule(encoder=encoder, predictor=predictor, metrics=metrics)
        logger.info(model)

        monitor_mode = "min" if model.metrics[0].minimize else "max"
        logger.debug(
            f"Evaluation metric: '{model.metrics[0].alias}', mode: '{monitor_mode}'"
        )

        try:
            trainer_logger = TensorBoardLogger(model_output_dir, "trainer_logs")
        except ModuleNotFoundError:
            trainer_logger = CSVLogger(model_output_dir, "trainer_logs")

        checkpointing = ModelCheckpoint(
            model_output_dir / "checkpoints",
            "best-{epoch}-{val_loss:.2f}",
            "val_loss",
            mode=monitor_mode,
            save_last=True,
        )

        patience = args.patience if args.patience is not None else args.epochs
        early_stopping = EarlyStopping("val_loss", patience=patience, mode=monitor_mode)

        trainer = pl.Trainer(
            logger=trainer_logger,
            enable_progress_bar=True,
            accelerator=args.accelerator,
            devices=args.devices,
            max_epochs=args.epochs,
            callbacks=[checkpointing, early_stopping],
            gradient_clip_val=args.grad_clip,
            deterministic=deterministic,
        )
        trainer.fit(model, train_loader, val_loader)

        if test_loader is not None:
            if isinstance(trainer.strategy, DDPStrategy):
                torch.distributed.destroy_process_group()

                best_ckpt_path = trainer.checkpoint_callback.best_model_path
                trainer = pl.Trainer(
                    logger=trainer_logger,
                    enable_progress_bar=True,
                    accelerator=args.accelerator,
                    devices=1,
                )
                model = model.load_from_checkpoint(best_ckpt_path)
                predss = trainer.predict(model, dataloaders=test_loader)
            else:
                predss = trainer.predict(dataloaders=test_loader)

            preds = torch.concat(predss, 0).numpy()

            evaluate_and_save_predictions(
                preds, test_loader, model.metrics, model_output_dir, args
            )

        best_model_path = checkpointing.best_model_path
        model = model.__class__.load_from_checkpoint(best_model_path)
        p_model = model_output_dir / "best.pt"
        save_model(p_model, model)
        logger.info(f"Best model saved to '{p_model}'")


def evaluate_and_save_predictions(preds, test_loader, metrics, model_output_dir, args):
    test_dset = test_loader.dataset
    targets = test_dset.Y

    individual_scores = dict()
    for metric in metrics:
        individual_scores[metric.alias] = []
        for i, col in enumerate(args.target_columns):
            preds_slice = torch.from_numpy(preds[:, i])
            targets_slice = torch.from_numpy(targets[:, i])
            preds_loss = metric(
                preds_slice,
                targets_slice,
            )
            individual_scores[metric.alias].append(preds_loss)

    logger.info("Entire Test Set results:")
    for metric in metrics:
        avg_loss = sum(individual_scores[metric.alias]) / len(
            individual_scores[metric.alias]
        )
        logger.info(f"entire_test/{metric.alias}: {avg_loss}")

    if args.show_individual_scores:
        logger.info("Entire Test Set individual results:")
        for metric in metrics:
            for i, col in enumerate(args.target_columns):
                logger.info(
                    f"entire_test/{col}/{metric.alias}: {individual_scores[metric.alias][i]}"
                )

    names = test_loader.dataset.names
    namess = [names]
    df_preds = pd.DataFrame(
        list(zip(*namess, *preds.T)), columns=args.smiles_columns + args.target_columns
    )
    df_preds.to_csv(model_output_dir / "test_predictions.csv", index=False)


def main(args):
    format_kwargs = dict(
        no_header_row=args.no_header_row,
        smiles_cols=args.smiles_columns,
        target_cols=args.target_columns,
        splits_col=args.splits_column,
    )

    featurization_kwargs = dict(
        molecule_featurizers=args.molecule_featurizers,
        keep_h=args.keep_h,
        add_h=args.add_h,
    )

    splits = build_splits(args, format_kwargs, featurization_kwargs)

    for fold_idx, (train_data, val_data, test_data) in enumerate(zip(*splits)):
        if args.num_folds == 1:
            output_dir = args.output_dir
        else:
            output_dir = args.output_dir / f"fold_{fold_idx}"

        output_dir.mkdir(exist_ok=True, parents=True)

        atom_featurizer = get_multi_hot_atom_featurizer(
            args.multi_hot_atom_featurizer_mode
        )
        bond_featurizer = MultiHotBondFeaturizer()

        train_dset = MoleculeDataset(train_data, atom_featurizer, bond_featurizer)
        val_dset = MoleculeDataset(val_data, atom_featurizer, bond_featurizer)
        test_dset = MoleculeDataset(test_data, atom_featurizer, bond_featurizer)

        if args.save_smiles_splits:
            save_smiles_splits(args, output_dir, train_dset, val_dset, test_dset)

        if args.task_type == "regression":
            output_scaler = train_dset.normalize_targets()
            val_dset.normalize_targets(output_scaler)
            logger.info(
                f"Train data: mean = {output_scaler.mean_} | std = {output_scaler.scale_}"
            )
            output_transform = UnscaleTransform.from_standard_scaler(output_scaler)
        else:
            output_transform = None

        train_loader = build_dataloader(
            dataset=train_dset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = build_dataloader(
            dataset=val_dset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        if test_dset is not None:
            test_loader = build_dataloader(
                dataset=test_dset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
        else:
            test_loader = None

        train_model(
            args,
            train_loader,
            val_loader,
            test_loader,
            output_dir,
            output_transform,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = TrainSubcommand.add_args(parser)

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
    args = parser.parse_args()
    TrainSubcommand.func(args)
