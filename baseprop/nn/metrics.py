from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torchmetrics import functional as F
from torchmetrics.utilities.compute import auc

from baseprop.nn.loss import BCELoss, CrossEntropyLoss, LossFunction, MSELoss
from baseprop.utils.registry import ClassRegistry


class Metric(LossFunction):
    minimize: bool = True

    def forward(
        self,
        preds: Tensor,
        targets: Tensor,
    ):
        mask = torch.from_numpy(np.isfinite(targets.cpu().numpy()))
        return self._calc_unreduced_loss(preds, targets)[mask].mean()

    @abstractmethod
    def _calc_unreduced_loss(self, preds, targets) -> Tensor:
        pass


MetricRegistry = ClassRegistry[Metric]()


@dataclass
class ThresholdedMixin:
    threshold: float | None = 0.5

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}"


@MetricRegistry.register("mae")
class MAEMetric(Metric):
    def _calc_unreduced_loss(self, preds, targets) -> Tensor:
        return (preds - targets).abs()


@MetricRegistry.register("mse")
class MSEMetric(MSELoss, Metric):
    pass


@MetricRegistry.register("rmse")
class RMSEMetric(MSEMetric):
    def forward(
        self,
        preds: Tensor,
        targets: Tensor,
    ):
        mask = torch.from_numpy(np.isfinite(targets.cpu().numpy()))
        squared_errors = super()._calc_unreduced_loss(preds, targets)

        return squared_errors[mask].mean().sqrt()


@MetricRegistry.register("r2")
class R2Metric(Metric):
    minimize = False

    def forward(self, preds: Tensor, targets: Tensor):
        mask = torch.from_numpy(np.isfinite(targets.cpu().numpy()))
        return F.r2_score(preds[mask], targets[mask])


@MetricRegistry.register("roc")
class BinaryAUROCMetric(Metric):
    minimize = False

    def forward(self, preds: Tensor, targets: Tensor):
        return self._calc_unreduced_loss(preds, targets)

    def _calc_unreduced_loss(self, preds, targets) -> Tensor:
        mask = torch.from_numpy(np.isfinite(targets.cpu().numpy()))
        return F.auroc(preds[mask], targets[mask].long(), task="binary")


@MetricRegistry.register("prc")
class BinaryAUPRCMetric(Metric):
    minimize = False

    def forward(self, preds: Tensor, targets: Tensor):
        p, r, _ = F.precision_recall_curve(preds, targets.long(), task="binary")
        return auc(r, p)


@MetricRegistry.register("accuracy")
class BinaryAccuracyMetric(Metric, ThresholdedMixin):
    minimize = False

    def forward(self, preds: Tensor, targets: Tensor):
        mask = torch.from_numpy(np.isfinite(targets.cpu().numpy()))
        return F.accuracy(
            preds[mask], targets[mask].long(), threshold=self.threshold, task="binary"
        )


@MetricRegistry.register("f1")
class BinaryF1Metric(Metric, ThresholdedMixin):
    minimize = False

    def forward(self, preds: Tensor, targets: Tensor):
        mask = torch.from_numpy(np.isfinite(targets.cpu().numpy()))
        return F.f1_score(
            preds[mask], targets[mask].long(), threshold=self.threshold, task="binary"
        )


@MetricRegistry.register("bce")
class BCEMetric(BCELoss, Metric):
    pass


@MetricRegistry.register("ce")
class CrossEntropyMetric(CrossEntropyLoss, Metric):
    pass
