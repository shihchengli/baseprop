from __future__ import annotations

from typing import Iterable

import torch
from lightning import pytorch as pl
from torch import Tensor, distributed, nn, optim
from torch_geometric.nn import global_mean_pool

from baseprop.data import BatchMolGraph, TrainingBatch
from baseprop.nn import LossFunction, Predictor
from baseprop.nn.metrics import Metric
from baseprop.nn.transforms import ScaleTransform


class LitModule(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module | None = None,
        predictor: Predictor | None = None,
        metrics: Iterable[Metric] | None = None,
        lr: float = 1e-4,
        X_d_transform: ScaleTransform | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["X_d_transform", "encoder", "predictor"])
        self.hparams["X_d_transform"] = X_d_transform
        self.hparams["encoder"] = encoder
        self.hparams.update(
            {
                "predictor": predictor.hparams,
            }
        )
        self.encoder = encoder
        self.predictor = predictor
        self.X_d_transform = (
            X_d_transform if X_d_transform is not None else nn.Identity()
        )
        self.metrics = metrics if metrics else [self.predictor._T_default_metric()]
        self.lr = lr

    @property
    def output_dim(self) -> int:
        return self.predictor.output_channels

    @property
    def n_tasks(self) -> int:
        return self.predictor.n_tasks

    @property
    def criterion(self) -> LossFunction:
        return self.predictor.criterion

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.7, patience=5, min_lr=self.lr / 100
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def forward(self, bmg: BatchMolGraph, X_d: Tensor | None = None) -> Tensor:
        """Generate predictions for the input molecules/reactions"""
        if self.encoder is not None:
            H = self.encoder(bmg)

            # agg
            H = global_mean_pool(H, bmg.batch)

            H = H if X_d is None else torch.cat((H, self.X_d_transform(X_d)), 1)
            return self.predictor(H)
        else:
            return self.predictor(X_d)

    def training_step(self, batch: TrainingBatch, batch_idx):
        bmg, X_d, targets = batch
        preds = self(bmg, X_d)
        l = self.criterion(preds, targets)

        self.log("train_loss", l, prog_bar=True)

        return l

    def on_validation_model_eval(self) -> None:
        self.eval()
        self.predictor.output_transform.train()

    def validation_step(self, batch: TrainingBatch, batch_idx: int = 0):
        bmg, X_d, targets = batch
        preds = self(bmg, X_d)
        l = self.criterion(preds, targets)
        self.log(
            "val_loss",
            l,
            batch_size=len(batch[0]),
            prog_bar=True,
            sync_dist=distributed.is_initialized(),
        )

        losses = self._evaluate_batch(batch)
        metric2loss = {f"val/{m.alias}": l for m, l in zip(self.metrics, losses)}

        self.log_dict(metric2loss, batch_size=len(batch[0]))

    def test_step(self, batch: TrainingBatch, batch_idx: int = 0):
        losses = self._evaluate_batch(batch)
        metric2loss = {
            f"batch_averaged_test/{m.alias}": l for m, l in zip(self.metrics, losses)
        }

        self.log_dict(metric2loss, batch_size=len(batch[0]))

    def _evaluate_batch(self, batch) -> list[Tensor]:
        bmg, X_d, targets = batch
        preds = self(bmg, X_d)

        return [metric(preds, targets) for metric in self.metrics]

    def predict_step(
        self, batch: TrainingBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Return the predictions of the input batch

        Parameters
        ----------
        batch : TrainingBatch
            the input batch

        Returns
        -------
        Tensor
            a tensor of varying shape depending on the task type:

            * regression/binary classification: ``n x (t * s)``, where ``n`` is the number of input
              molecules/reactions, ``t`` is the number of tasks, and ``s`` is the number of targets
              per task. The final dimension is flattened, so that the targets for each task are
              grouped. I.e., the first ``t`` elements are the first target for each task, the second
              ``t`` elements the second target, etc.

            * multiclass classification: ``n x t x c``, where ``c`` is the number of classes
        """
        bmg, X_d, _ = batch

        return self(bmg, X_d)

    @classmethod
    def load_submodules(cls, checkpoint_path, **kwargs):
        hparams = torch.load(checkpoint_path)["hyper_parameters"]

        kwargs |= {
            key: hparams[key].pop("cls")(**hparams[key])
            for key in ["predictor"]
            if key not in kwargs
        }
        return kwargs

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict=True,
        **kwargs,
    ) -> LitModule:
        kwargs = cls.load_submodules(checkpoint_path, **kwargs)
        return super().load_from_checkpoint(
            checkpoint_path, map_location, hparams_file, strict, **kwargs
        )

    @classmethod
    def load_from_file(cls, model_path, map_location=None, strict=True) -> LitModule:
        d = torch.load(model_path, map_location=map_location)

        try:
            hparams = d["hyper_parameters"]
            state_dict = d["state_dict"]
        except KeyError:
            raise KeyError(
                f"Could not find hyper parameters and/or state dict in {model_path}. "
            )

        for key in ["predictor"]:
            hparam_kwargs = hparams[key]
            hparam_cls = hparam_kwargs.pop("cls")
            hparams[key] = hparam_cls(**hparam_kwargs)

        model = cls(**hparams)
        model.load_state_dict(state_dict, strict=strict)

        return model
