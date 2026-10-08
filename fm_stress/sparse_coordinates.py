"""Joint full-rank velocity-PCA coordinates; no centering of derivatives."""
import torch
from torch import nn


class JointCoordinates(nn.Module):
    def __init__(self, base, pca=None, transform_input=False, balanced_loss=False):
        super().__init__()
        self.base = base
        self.transform_input = bool(transform_input)
        self.balanced_loss = bool(balanced_loss or transform_input)
        n = base.horizon * base.channels
        self.register_buffer('q', torch.eye(n) if pca is None else torch.tensor(pca.components.T, dtype=torch.float32))
        self.register_buffer('scales', torch.ones(n) if pca is None else torch.tensor(pca.scales, dtype=torch.float32))

    def to_coordinates(self, x):
        return ((x.flatten(1) @ self.q) / self.scales).reshape_as(x)

    def from_coordinates(self, x):
        return ((x.flatten(1) * self.scales) @ self.q.T).reshape_as(x)

    def forward(self, x, t, history):
        pred = self.base(self.to_coordinates(x) if self.transform_input else x, t, history)
        return self.from_coordinates(pred) if self.transform_input else pred

    def objective(self, prediction, velocity):
        error = prediction - velocity
        if self.balanced_loss:
            error = self.to_coordinates(error)
        return error.square().mean()

    def auxiliary_loss(self):
        fn = getattr(self.base, 'auxiliary_loss', None)
        return fn() if fn else self.q.new_zeros(())

    def activity_statistics(self):
        fn = getattr(self.base, 'activity_statistics', None)
        return fn() if fn else {}
