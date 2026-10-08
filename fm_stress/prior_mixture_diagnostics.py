"""Parameter-gradient diagnostics in raw observation coordinates.

These measure loss gradients with respect to velocity-head parameters. They
are neither input Jacobians nor proof that gradient imbalance causes failure.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .metrics import PCA


def gradient_diagnostics(model, x, t, history, velocity, pca: PCA):
    """Decompose RAW MSE gradients into PC1, PC2 and PCs3:F contributions.

    ``model(x, t, history)`` must return raw observed velocity ``[B,F]``.
    Every group uses the same denominator B*F, so its loss and gradient are
    additive contributions to raw MSE. ``gradient_norm_per_component`` instead
    divides the group-gradient norm by its component count; it is not whitening
    or a variance-normalized loss. Encoder-named parameters are excluded, even
    inside wrappers. Conditioning/time projections remain part of the head.

    No parameter .grad is read or modified. All module training flags (including
    mixed train/eval states) are restored, including on errors. This helper
    draws no random numbers and evaluates the model in eval mode. Call on a
    fixed small diagnostic batch, outside torch.inference_mode().
    """
    if velocity.ndim != 2 or velocity.shape[1] < 3 or len(velocity) == 0:
        raise ValueError("velocity must have shape [B,F], B>=1 and F>=3")
    if x.shape != velocity.shape:
        raise ValueError("x and velocity must have matching [B,F] shapes")
    horizon = velocity.shape[1]
    components = np.asarray(pca.components)
    if components.shape != (horizon, horizon) or not np.isfinite(components).all():
        raise ValueError("PCA components must be finite square [F,F]")
    if not np.allclose(components @ components.T, np.eye(horizon), atol=2e-5):
        raise ValueError("PCA components must be orthonormal rows")
    included = [(name, parameter) for name, parameter in model.named_parameters()
                if parameter.requires_grad and "encoder" not in name.lower()]
    excluded = [name for name, parameter in model.named_parameters()
                if parameter.requires_grad and "encoder" in name.lower()]
    if not included:
        raise ValueError("No trainable velocity-head parameters after encoder exclusion")
    parameters = [parameter for _, parameter in included]
    states = [(module, module.training) for module in model.modules()]

    def vector_gradient(loss, retain_graph):
        if loss.requires_grad:
            gradients = torch.autograd.grad(loss, parameters, allow_unused=True,
                                            retain_graph=retain_graph)
        else:
            gradients = [None] * len(parameters)
        vector = torch.cat([(torch.zeros_like(parameter) if gradient is None else gradient)
                            .detach().reshape(-1)
                            for parameter, gradient in zip(parameters, gradients)])
        if not torch.isfinite(vector).all().item():
            raise ValueError("Nonfinite parameter gradient in diagnostic batch")
        return vector

    try:
        model.eval()
        with torch.enable_grad():
            prediction = model(x.detach(), t.detach(), history.detach())
            if prediction.shape != velocity.shape:
                raise ValueError("Model must return raw observed velocity with shape [B,F]")
            error = prediction - velocity.detach()
            if not torch.isfinite(error).all().item():
                raise ValueError("Nonfinite prediction or velocity in diagnostic batch")
            axes = torch.as_tensor(components, dtype=error.dtype, device=error.device)
            projected = error @ axes.T  # Error is not centered by the PCA mean.
            groups = {"pc1": (slice(0, 1), 1), "pc2": (slice(1, 2), 1),
                      "residual": (slice(2, horizon), horizon - 2)}
            losses = {name: projected[:, indices].square().sum() / error.numel()
                      for name, (indices, _) in groups.items()}
            gradients = {name: vector_gradient(loss, retain_graph=True)
                         for name, loss in losses.items()}
            total_loss = error.square().mean()
            total_gradient = vector_gradient(total_loss, retain_graph=False)
            total_norm = float(torch.linalg.vector_norm(total_gradient).item())
            reconstructed = sum(gradients.values())
            reconstruction_error = float(torch.linalg.vector_norm(
                reconstructed - total_gradient).item())
            norms = {name: float(torch.linalg.vector_norm(gradient).item())
                     for name, gradient in gradients.items()}
            cosines = {}
            for first, second in (("pc1", "pc2"), ("pc1", "residual"),
                                  ("pc2", "residual")):
                denominator = norms[first] * norms[second]
                value = (float(torch.dot(gradients[first], gradients[second]).item()) /
                         denominator) if denominator > 0 else None
                cosines[f"{first}_vs_{second}"] = (
                    None if value is None else max(-1.0, min(1.0, value)))
            output = {
                "coordinate_system": "raw_observed_velocity",
                "parameter_scope": "trainable parameters excluding names containing encoder",
                "loss_denominator": "batch_size * horizon for every group",
                "batch_size": int(len(velocity)), "horizon": int(horizon),
                "head_parameter_count": sum(parameter.numel() for parameter in parameters),
                "excluded_encoder_parameter_names": excluded,
                "raw_total_mse": float(total_loss.detach().item()),
                "groups": {name: {
                    "components": count,
                    "raw_mse_contribution": float(losses[name].detach().item()),
                    "gradient_norm": norms[name],
                    "gradient_norm_per_component": norms[name] / count,
                } for name, (_, count) in groups.items()},
                "gradient_cosines": cosines,
                "raw_total_gradient_norm": total_norm,
                "summed_group_gradient_norm": float(torch.linalg.vector_norm(
                    reconstructed).item()),
                "gradient_reconstruction_absolute_error": reconstruction_error,
                "gradient_reconstruction_relative_error": (
                    reconstruction_error / total_norm if total_norm > 0 else None),
                "loss_reconstruction_absolute_error": abs(
                    sum(float(loss.detach().item()) for loss in losses.values())
                    - float(total_loss.detach().item())),
            }
            if not all(math.isfinite(value) for value in norms.values()):
                raise ValueError("Nonfinite gradient norms in diagnostic batch")
            return output
    finally:
        for module, training in states:
            module.training = training
