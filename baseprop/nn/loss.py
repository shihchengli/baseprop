from abc import abstractmethod

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from baseprop.utils.registry import ClassRegistry


class LossFunction(nn.Module):
    def forward(
        self,
        preds: Tensor,
        targets: Tensor,
    ):
        """Calculate the mean loss function value given predicted and target values

        Parameters
        ----------
        preds : Tensor
            a tensor of shape `b x (t * s)` (regression), `b x t` (binary classification), or
            `b x t x c` (multiclass classification) containing the predictions, where `b` is the
            batch size, `t` is the number of tasks to predict, `s` is the number of
            targets to predict for each task, and `c` is the number of classes.
        targets : Tensor
            a float tensor of shape `b x t` containing the target values

        Returns
        -------
        Tensor
            a scalar containing the fully reduced loss
        """
        mask = torch.from_numpy(np.isfinite(targets.cpu().numpy()))
        mask = mask.to(preds.device)
        L = self._calc_unreduced_loss(preds, targets)
        L = L * mask

        return L.sum() / mask.sum()

    @abstractmethod
    def _calc_unreduced_loss(self, preds, targets) -> Tensor:
        """Calculate a tensor of shape `b x t` containing the unreduced loss values."""


LossFunctionRegistry = ClassRegistry[LossFunction]()


@LossFunctionRegistry.register("mse")
class MSELoss(LossFunction):
    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor) -> Tensor:
        return F.mse_loss(preds, targets, reduction="none")


@LossFunctionRegistry.register("bce")
class BCELoss(LossFunction):
    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(preds, targets, reduction="none")


@LossFunctionRegistry.register("ce")
class CrossEntropyLoss(LossFunction):
    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor) -> Tensor:
        preds = preds.transpose(1, 2)
        targets = targets.long()

        return F.cross_entropy(preds, targets, reduction="none")
