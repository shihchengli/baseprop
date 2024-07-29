import torch
from numpy.typing import ArrayLike
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn


class _ScaleTransformMixin(nn.Module):
    def __init__(self, mean: ArrayLike, scale: ArrayLike, pad: int = 0):
        super().__init__()

        mean = torch.cat([torch.zeros(pad), torch.tensor(mean, dtype=torch.float)])
        scale = torch.cat([torch.ones(pad), torch.tensor(scale, dtype=torch.float)])

        if mean.shape != scale.shape:
            raise ValueError(
                f"uneven shapes for 'mean' and 'scale'! got: mean={mean.shape}, scale={scale.shape}"
            )

        self.register_buffer("mean", mean.unsqueeze(0))
        self.register_buffer("scale", scale.unsqueeze(0))

    @classmethod
    def from_standard_scaler(cls, scaler: StandardScaler, pad: int = 0):
        return cls(scaler.mean_, scaler.scale_, pad=pad)


class ScaleTransform(_ScaleTransformMixin):
    def forward(self, X: Tensor) -> Tensor:
        if self.training:
            return X

        return (X - self.mean) / self.scale


class UnscaleTransform(_ScaleTransformMixin):
    def forward(self, X: Tensor) -> Tensor:
        if self.training:
            return X

        return X * self.scale + self.mean

    def transform_variance(self, var: Tensor) -> Tensor:
        if self.training:
            return var
        return var * (self.scale**2)