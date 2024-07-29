from copy import deepcopy
import logging
from pathlib import Path
import sys

from configargparse import ArgumentParser, Namespace
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import numpy as np
import torch

from baseprop.cli.common import add_common_args
from baseprop.cli.train import (
    TrainSubcommand,
    add_train_args,
    build_splits,
    process_train_args,
    save_config,
    save_smiles_splits,
    train_model,
)
from baseprop.cli.utils.command import Subcommand
from baseprop.data import (
    MoleculeDataset,
    build_dataloader,
    make_split_indices,
    split_data_by_indices,
)
from baseprop.featurizers.atom import get_multi_hot_atom_featurizer
from baseprop.featurizers.bond import MultiHotBondFeaturizer
from baseprop.models.model import LitModule
from baseprop.nn import GCN, LossFunctionRegistry, MetricRegistry, PredictorRegistry
from baseprop.nn.transforms import UnscaleTransform
from baseprop.nn.utils import Activation
from baseprop.utils import Factory

NO_RAY = False
DEFAULT_SEARCH_SPACE = {
    "activation": None,
    "batch_size": None,
    "depth": None,
    "dropout": None,
    "ffn_hidden_dim": None,
    "ffn_num_layers": None,
    "hidden_channels": None,
    "lr": None,
}

try:
    import ray
    from ray import tune
    from ray.air import session
    from ray.train import CheckpointConfig, RunConfig, ScalingConfig
    from ray.train.lightning import (
        RayDDPStrategy,
        RayLightningEnvironment,
        RayTrainReportCallback,
        prepare_trainer,
    )
    from ray.train.torch import TorchTrainer
    from ray.tune.schedulers import ASHAScheduler, FIFOScheduler

    DEFAULT_SEARCH_SPACE = {
        "activation": tune.choice(categories=list(Activation.keys())),
        "batch_size": tune.choice([16, 32, 64, 128, 256]),
        "depth": tune.qrandint(lower=2, upper=6, q=1),
        "dropout": tune.choice([0.0] * 8 + list(np.arange(0.05, 0.45, 0.05))),
        "ffn_hidden_dim": tune.qrandint(lower=300, upper=2400, q=100),
        "ffn_num_layers": tune.qrandint(lower=1, upper=3, q=1),
        "hidden_channels": tune.qrandint(lower=300, upper=2400, q=100),
        "lr": tune.loguniform(lower=1e-6, upper=1e-2),
    }
except ImportError:
    NO_RAY = True

NO_HYPEROPT = False
try:
    from ray.tune.search.hyperopt import HyperOptSearch
except ImportError:
    NO_HYPEROPT = True

NO_OPTUNA = False
try:
    from ray.tune.search.optuna import OptunaSearch
except ImportError:
    NO_OPTUNA = True


logger = logging.getLogger(__name__)

SEARCH_SPACE = DEFAULT_SEARCH_SPACE

SEARCH_PARAM_KEYWORDS_MAP = {
    "basic": ["depth", "ffn_num_layers", "dropout", "ffn_hidden_dim", "hidden_channels"],
    "all": list(DEFAULT_SEARCH_SPACE.keys()),
    "lr": ["lr"],
}


class NestedCVSubcommand(Subcommand):
    COMMAND = "nestedCV"
    HELP = "perform nested cross-validation on the given task"

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        parser = add_common_args(parser)
        parser = add_train_args(parser)
        return add_hpopt_args(parser)

    @classmethod
    def func(cls, args: Namespace):
        args = process_train_args(args)
        args = process_hpopt_args(args)
        main(args)


def add_hpopt_args(parser: ArgumentParser) -> ArgumentParser:
    hpopt_args = parser.add_argument_group("Baseprop hyperparameter optimization arguments")

    hpopt_args.add_argument(
        "--search-parameter-keywords",
        type=str,
        nargs="+",
        default=["basic"],
        help=f"""The model parameters over which to search for an optimal hyperparameter configuration.
    Some options are bundles of parameters or otherwise special parameter operations.

    Special keywords:
        basic - the default set of hyperparameters for search: depth, ffn_num_layers, dropout, message_hidden_dim, and ffn_hidden_dim.
        all - include search for all inidividual keyword options

    Individual supported parameters:
        {list(DEFAULT_SEARCH_SPACE.keys())}
    """,
    )

    hpopt_args.add_argument(
        "--hpopt-save-dir",
        type=Path,
        help="Directory to save the hyperparameter optimization results",
    )

    raytune_args = parser.add_argument_group("Ray Tune arguments")

    raytune_args.add_argument(
        "--raytune-num-samples",
        type=int,
        default=10,
        help="Passed directly to Ray Tune TuneConfig to control number of trials to run",
    )

    raytune_args.add_argument(
        "--raytune-search-algorithm",
        choices=["random", "hyperopt", "optuna"],
        default="hyperopt",
        help="Passed to Ray Tune TuneConfig to control search algorithm",
    )

    raytune_args.add_argument(
        "--raytune-trial-scheduler",
        choices=["FIFO", "AsyncHyperBand"],
        default="FIFO",
        help="Passed to Ray Tune TuneConfig to control trial scheduler",
    )

    raytune_args.add_argument(
        "--raytune-num-workers",
        type=int,
        default=1,
        help="Passed directly to Ray Tune ScalingConfig to control number of workers to use",
    )

    raytune_args.add_argument(
        "--raytune-use-gpu",
        action="store_true",
        help="Passed directly to Ray Tune ScalingConfig to control whether to use GPUs",
    )

    raytune_args.add_argument(
        "--raytune-num-checkpoints-to-keep",
        type=int,
        default=1,
        help="Passed directly to Ray Tune CheckpointConfig to control number of checkpoints to keep",
    )

    raytune_args.add_argument(
        "--raytune-grace-period",
        type=int,
        default=10,
        help="Passed directly to Ray Tune ASHAScheduler to control grace period",
    )

    raytune_args.add_argument(
        "--raytune-reduction-factor",
        type=int,
        default=2,
        help="Passed directly to Ray Tune ASHAScheduler to control reduction factor",
    )

    raytune_args.add_argument(
        "--raytune-temp-dir", help="Passed directly to Ray Tune init to control temporary directory"
    )

    raytune_args.add_argument(
        "--raytune-num-cpus",
        type=int,
        help="Passed directly to Ray Tune init to control number of CPUs to use",
    )

    raytune_args.add_argument(
        "--raytune-num-gpus",
        type=int,
        help="Passed directly to Ray Tune init to control number of GPUs to use",
    )

    raytune_args.add_argument(
        "--raytune-max-concurrent-trials",
        type=int,
        help="Passed directly to Ray Tune TuneConfig to control maximum concurrent trials",
    )

    hyperopt_args = parser.add_argument_group("Hyperopt arguments")

    hyperopt_args.add_argument(
        "--hyperopt-n-initial-points",
        type=int,
        help="Passed directly to HyperOptSearch to control number of initial points to sample",
    )

    hyperopt_args.add_argument(
        "--hyperopt-random-state-seed",
        type=int,
        default=None,
        help="Passed directly to HyperOptSearch to control random state seed",
    )

    return parser


def process_hpopt_args(args: Namespace) -> Namespace:
    if args.hpopt_save_dir is None:
        args.hpopt_save_dir = Path(f"baseprop_hpopt/{args.data_path.stem}")

    args.hpopt_save_dir.mkdir(exist_ok=True, parents=True)

    search_parameters = set()

    available_search_parameters = list(SEARCH_SPACE.keys()) + list(SEARCH_PARAM_KEYWORDS_MAP.keys())

    for keyword in args.search_parameter_keywords:
        if keyword not in available_search_parameters:
            raise ValueError(
                f"Search parameter keyword: {keyword} not in available options: {available_search_parameters}."
            )

        search_parameters.update(
            SEARCH_PARAM_KEYWORDS_MAP[keyword]
            if keyword in SEARCH_PARAM_KEYWORDS_MAP
            else [keyword]
        )

    args.search_parameter_keywords = list(search_parameters)

    if not args.hyperopt_n_initial_points:
        args.hyperopt_n_initial_points = args.raytune_num_samples // 2

    return args


def build_search_space(search_parameters: list[str]) -> dict:
    return {param: SEARCH_SPACE[param] for param in search_parameters}


def update_args_with_config(args: Namespace, config: dict) -> Namespace:
    args = deepcopy(args)

    for key, value in config.items():
        assert key in args, f"Key: {key} not found in args."
        setattr(args, key, value)

    return args


def train_with_splits(config, args, data_splits, logger, output_transform):
    args = update_args_with_config(args, config)
    val_losses = []
    for fold_idx, (train_data, val_data, _) in enumerate(zip(*data_splits)):
        logger.info(f"split {fold_idx}...")
        atom_featurizer = get_multi_hot_atom_featurizer(args.multi_hot_atom_featurizer_mode)
        bond_featurizer = MultiHotBondFeaturizer()

        train_dset = MoleculeDataset(train_data, atom_featurizer, bond_featurizer)
        val_dset = MoleculeDataset(val_data, atom_featurizer, bond_featurizer)
        if "regression" in args.task_type:
            output_scaler = train_dset.normalize_targets()
            val_dset.normalize_targets(output_scaler)
            logger.info(f"Train data: mean = {output_scaler.mean_} | std = {output_scaler.scale_}")
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

        seed = args.pytorch_seed if args.pytorch_seed is not None else torch.seed()

        torch.manual_seed(seed)

        # build model
        if args.loss_function is not None:
            criterion = Factory.build(LossFunctionRegistry[args.loss_function])
        else:
            criterion = None
        if args.metrics is not None:
            metrics = [Factory.build(MetricRegistry[metric]) for metric in args.metrics]
        else:
            metrics = None

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
        predictor_cls = PredictorRegistry[args.task_type]
        predictor = Factory.build(
            predictor_cls,
            input_dim=encoder.hidden_channels + len(len(train_loader.dataset[0].x_d))
            if args.molecule_featurizers
            else encoder.hidden_channels,
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
        logger.debug(f"Evaluation metric: '{model.metrics[0].alias}', mode: '{monitor_mode}'")

        patience = args.patience if args.patience is not None else args.epochs
        early_stopping = EarlyStopping("val_loss", patience=patience, mode=monitor_mode)

        trainer = pl.Trainer(
            accelerator=args.accelerator,
            devices=args.devices,
            max_epochs=args.epochs,
            gradient_clip_val=args.grad_clip,
            strategy=RayDDPStrategy(),
            callbacks=[RayTrainReportCallback(), early_stopping],
            plugins=[RayLightningEnvironment()],
            deterministic=args.pytorch_seed is not None,
        )
        trainer = prepare_trainer(trainer)
        trainer.fit(model, train_loader, val_loader)
        val_loss = trainer.callback_metrics["val_loss"].item()
        logger.info(f"Split {fold_idx}/{len(data_splits[0])}: val_loss = {val_loss}")
        val_losses.append(val_loss)

    avg_val_loss = np.mean(val_losses)
    logger.info(f"Average val_loss over split {fold_idx}: {avg_val_loss}")
    session.report({"val_loss": avg_val_loss})


def tune_model(args, data_splits, logger, monitor_mode, output_transform):
    match args.raytune_trial_scheduler:
        case "FIFO":
            scheduler = FIFOScheduler()
        case "AsyncHyperBand":
            scheduler = ASHAScheduler(
                max_t=args.epochs,
                grace_period=min(args.raytune_grace_period, args.epochs),
                reduction_factor=args.raytune_reduction_factor,
            )
        case _:
            raise ValueError(f"Invalid trial scheduler! got: {args.raytune_trial_scheduler}.")

    resources_per_worker = {}
    if args.raytune_num_cpus and args.raytune_max_concurrent_trials:
        resources_per_worker["CPU"] = args.raytune_num_cpus / args.raytune_max_concurrent_trials
    if args.raytune_num_gpus and args.raytune_max_concurrent_trials:
        resources_per_worker["GPU"] = args.raytune_num_gpus / args.raytune_max_concurrent_trials
    if not resources_per_worker:
        resources_per_worker = None

    if args.raytune_num_gpus:
        use_gpu = True
    else:
        use_gpu = args.raytune_use_gpu

    scaling_config = ScalingConfig(
        num_workers=args.raytune_num_workers,
        use_gpu=use_gpu,
        resources_per_worker=resources_per_worker,
        trainer_resources={"CPU": 0},
    )

    checkpoint_config = CheckpointConfig(
        num_to_keep=args.raytune_num_checkpoints_to_keep,
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order=monitor_mode,
    )

    run_config = RunConfig(
        checkpoint_config=checkpoint_config,
        storage_path=args.hpopt_save_dir.absolute() / "ray_results",
    )

    ray_trainer = TorchTrainer(
        lambda config: train_with_splits(config, args, data_splits, logger, output_transform),
        scaling_config=scaling_config,
        run_config=run_config,
    )

    match args.raytune_search_algorithm:
        case "random":
            search_alg = None
        case "hyperopt":
            if NO_HYPEROPT:
                raise ImportError(
                    "HyperOptSearch requires hyperopt to be installed. Use 'pip install -U hyperopt' to install or use 'pip install -e .[hpopt]' in baseprop folder if you installed from source to install all hpopt relevant packages."
                )

            search_alg = HyperOptSearch(
                n_initial_points=args.hyperopt_n_initial_points,
                random_state_seed=args.hyperopt_random_state_seed,
            )
        case "optuna":
            if NO_OPTUNA:
                raise ImportError(
                    "OptunaSearch requires optuna to be installed. Use 'pip install -U optuna' to install or use 'pip install -e .[hpopt]' in baseprop folder if you installed from source to install all hpopt relevant packages."
                )

            search_alg = OptunaSearch()

    tune_config = tune.TuneConfig(
        metric="val_loss",
        mode=monitor_mode,
        num_samples=args.raytune_num_samples,
        scheduler=scheduler,
        search_alg=search_alg,
    )

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": build_search_space(args.search_parameter_keywords)},
        tune_config=tune_config,
    )

    return tuner.fit()


def main(args: Namespace):
    if NO_RAY:
        raise ImportError(
            "Ray Tune requires ray to be installed. If you installed baseprop from PyPI, make sure that your Python version is 3.11 and use 'pip install -U ray[tune]' to install ray. If you installed from source, use 'pip install -e .[hpopt]' in baseprop folder to install all hpopt relevant packages."
        )

    if not ray.is_initialized():
        try:
            ray.init(
                _temp_dir=args.raytune_temp_dir,
                num_cpus=args.raytune_num_cpus,
                num_gpus=args.raytune_num_gpus,
            )
        except OSError as e:
            if "AF_UNIX path length cannot exceed 107 bytes" in str(e):
                raise OSError(
                    f"Ray Tune fails due to: {e}. This can sometimes be solved by providing a temporary directory, num_cpus, and num_gpus to Ray Tune via the CLI: --raytune-temp-dir <absolute_path> --raytune-num-cpus <int> --raytune-num-gpus <int>."
                )
            else:
                raise e
    else:
        logger.info("Ray is already initialized.")

    format_kwargs = dict(
        no_header_row=args.no_header_row,
        smiles_cols=args.smiles_columns,
        target_cols=args.target_columns,
        splits_col=args.splits_column,
    )

    featurization_kwargs = dict(
        molecule_featurizers=args.molecule_featurizers, keep_h=args.keep_h, add_h=args.add_h
    )

    outer_train_data, outer_val_data, outer_test_data = build_splits(
        args, format_kwargs, featurization_kwargs
    )

    atom_featurizer = get_multi_hot_atom_featurizer(args.multi_hot_atom_featurizer_mode)
    bond_featurizer = MultiHotBondFeaturizer()

    # outer loop
    for fold_idx in range(args.num_folds):
        logger.info(f"Outer loop: {fold_idx}")
        if args.num_folds == 1:
            output_dir = args.output_dir
        else:
            output_dir = args.output_dir / f"fold_{fold_idx}"

        output_dir.mkdir(exist_ok=True, parents=True)

        outer_train_dset = MoleculeDataset(
            outer_train_data[fold_idx], atom_featurizer, bond_featurizer
        )
        outer_val_dset = MoleculeDataset(outer_val_data[fold_idx], atom_featurizer, bond_featurizer)
        outer_test_dset = MoleculeDataset(
            outer_test_data[fold_idx], atom_featurizer, bond_featurizer
        )

        if args.save_smiles_splits:
            save_smiles_splits(args, output_dir, outer_train_dset, outer_val_dset, outer_test_dset)

        if "regression" in args.task_type:
            output_scaler = outer_train_dset.normalize_targets()
            outer_val_dset.normalize_targets(output_scaler)
            logger.info(f"Train data: mean = {output_scaler.mean_} | std = {output_scaler.scale_}")
            output_transform = UnscaleTransform.from_standard_scaler(output_scaler)
        else:
            output_transform = None

        # inner loop
        splitting_mols = [datapoint.mol for datapoint in outer_train_data[fold_idx]]
        train_indices, val_indices, test_indices = make_split_indices(
            splitting_mols, "CV_NO_VAL", args.split_sizes, args.data_seed, args.num_folds
        )
        inner_train_data, _, inner_test_data = split_data_by_indices(
            outer_train_data[fold_idx], train_indices, val_indices, test_indices
        )
        inner_data_splits = (inner_train_data, inner_test_data, _)
        # reverse the val and test so only train and test sets are used
        # the outer_test_data is treated as outer_val_data

        train_loader = build_dataloader(
            dataset=outer_train_dset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = build_dataloader(
            dataset=outer_val_dset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        if outer_test_dset is not None:
            test_loader = build_dataloader(
                dataset=outer_test_dset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
        else:
            test_loader = None

        # build model
        if args.loss_function is not None:
            criterion = Factory.build(LossFunctionRegistry[args.loss_function])
        else:
            criterion = None
        if args.metrics is not None:
            metrics = [Factory.build(MetricRegistry[metric]) for metric in args.metrics]
        else:
            metrics = None

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
        predictor_cls = PredictorRegistry[args.task_type]
        predictor = Factory.build(
            predictor_cls,
            input_dim=encoder.hidden_channels + len(len(train_loader.dataset[0].x_d))
            if args.molecule_featurizers
            else encoder.hidden_channels,
            n_tasks=train_loader.dataset[0].y.shape[0],
            hidden_dim=args.ffn_hidden_dim,
            n_layers=args.ffn_num_layers,
            dropout=args.dropout,
            activation=args.activation,
            criterion=criterion,
            output_transform=output_transform,
        )
        model = LitModule(encoder=encoder, predictor=predictor, metrics=metrics)
        monitor_mode = "min" if model.metrics[0].minimize else "max"

        results = tune_model(args, inner_data_splits, logger, monitor_mode, output_transform)

        best_result = results.get_best_result()
        best_config = best_result.config["train_loop_config"]

        best_config_save_path = args.hpopt_save_dir / f"best_config_{fold_idx}.toml"
        all_progress_save_path = args.hpopt_save_dir / f"all_progress_{fold_idx}.csv"

        logger.info(
            f"Best hyperparameters for split {fold_idx} saved to: '{best_config_save_path}'"
        )

        args = update_args_with_config(args, best_config)

        parser = ArgumentParser()
        parser = TrainSubcommand.add_args(parser)
        train_args = TrainSubcommand.parser.parse_known_args(namespace=args)[0]
        save_config(TrainSubcommand.parser, train_args, best_config_save_path)

        logger.info(
            f"Hyperparameter optimization results for split {fold_idx} saved to '{all_progress_save_path}'"
        )

        result_df = results.get_dataframe()

        result_df.to_csv(all_progress_save_path, index=False)

        # Use optimized hyperparameters to train on external training set
        train_model(train_args, train_loader, val_loader, test_loader, output_dir, output_transform)

    ray.shutdown()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = NestedCVSubcommand.add_args(parser)

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
    args = parser.parse_args()
    NestedCVSubcommand.func(args)
