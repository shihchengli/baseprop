from os import PathLike

import torch

from baseprop.models.model import LitModule


def save_model(path: PathLike, model: LitModule) -> None:
    torch.save({"hyper_parameters": model.hparams, "state_dict": model.state_dict()}, path)


def load_model(path: PathLike) -> LitModule:
    model = LitModule.load_from_file(path, map_location=lambda storage, loc: storage)

    return model
