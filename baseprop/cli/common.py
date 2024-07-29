import logging
from argparse import ArgumentParser

from baseprop.cli.utils import LookupAction
from baseprop.cli.utils.args import uppercase
from baseprop.featurizers import AtomFeatureMode, MoleculeFeaturizerRegistry

logger = logging.getLogger(__name__)


def add_common_args(parser: ArgumentParser) -> ArgumentParser:
    data_args = parser.add_argument_group("Shared input data args")
    data_args.add_argument(
        "-s",
        "--smiles-columns",
        nargs="+",
        help="The column names in the input CSV containing SMILES strings. If unspecified, uses the the 0th column.",
    )
    data_args.add_argument(
        "--no-header-row",
        action="store_true",
        help="If specified, the first row in the input CSV will not be used as column names.",
    )

    dataloader_args = parser.add_argument_group("Dataloader args")
    dataloader_args.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=0,
        help="""Number of workers for parallel data loading (0 means sequential).
Warning: setting num_workers>0 can cause hangs on Windows and MacOS.""",
    )
    dataloader_args.add_argument(
        "-b", "--batch-size", type=int, default=64, help="Batch size."
    )

    parser.add_argument(
        "--accelerator",
        default="auto",
        help="Passed directly to the lightning Trainer().",
    )
    parser.add_argument(
        "--devices",
        default="auto",
        help="Passed directly to the lightning Trainer(). If specifying multiple devices, must be a single string of comma separated devices, e.g. '1, 2'.",
    )

    featurization_args = parser.add_argument_group("Featurization args")
    featurization_args.add_argument(
        "--multi-hot-atom-featurizer-mode",
        type=uppercase,
        default="V2",
        choices=list(AtomFeatureMode.keys()),
        help="""Choices for multi-hot atom featurization scheme. This will affect both non-reatction and reaction feturization (case insensitive):
- `V1`: Corresponds to the original configuration employed in the Chemprop V1.
- `V2`: Tailored for a broad range of molecules, this configuration encompasses all elements in the first four rows of the periodic table, along with iodine. It is the default in Chemprop V2.""",
    )
    featurization_args.add_argument(
        "--keep-h",
        action="store_true",
        help="Whether hydrogens explicitly specified in input should be kept in the mol graph.",
    )
    featurization_args.add_argument(
        "--add-h",
        action="store_true",
        help="Whether hydrogens should be added to the mol graph.",
    )
    featurization_args.add_argument(
        "--molecule-featurizers",
        "--features-generators",
        nargs="+",
        action=LookupAction(MoleculeFeaturizerRegistry),
        help="Method(s) of generating molecule features to use as extra descriptors.",
    )

    return parser
