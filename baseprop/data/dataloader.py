import warnings

from torch.utils.data import DataLoader

from baseprop.data.collate import collate_batch
from baseprop.data.datasets import MoleculeDataset


def build_dataloader(
    dataset: MoleculeDataset,
    batch_size: int = 64,
    num_workers: int = 0,
    shuffle: bool = True,
    **kwargs,
):
    """Return a :obj:`~torch.utils.data.DataLoader` for :class:`MolGraphDataset`\s

    Parameters
    ----------
    dataset : MoleculeDataset
        The dataset containing the molecules or reactions to load.
    batch_size : int, default=64
        the batch size to load.
    num_workers : int, default=0
        the number of workers used to build batches.
    shuffle : bool, default=False
        whether to shuffle the data during sampling.
    """

    if len(dataset) % batch_size == 1:
        warnings.warn(
            f"Dropping last batch of size 1 to avoid issues with batch normalization \
(dataset size = {len(dataset)}, batch_size = {batch_size})"
        )
        drop_last = True
    else:
        drop_last = False

    return DataLoader(
        dataset,
        batch_size,
        shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
        drop_last=drop_last,
        **kwargs,
    )
