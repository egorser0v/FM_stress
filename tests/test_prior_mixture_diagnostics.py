import numpy as np
import pytest
import torch
from torch import nn

from fm_stress.metrics import PCA
from fm_stress.prior_mixture_diagnostics import gradient_diagnostics


class Head(nn.Module):
    def __init__(self, horizon=4):
        super().__init__()
        self.encoder = nn.Linear(horizon, horizon, bias=False)
        self.output = nn.Linear(horizon, horizon, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.unused = nn.Parameter(torch.ones(2))

    def forward(self, x, t, history):
        return self.output(self.dropout(x)) + self.encoder(history)


def arguments(horizon=4):
    x = torch.arange(1, 2 * horizon + 1, dtype=torch.float64).reshape(2, horizon) / 10
    return (x, torch.tensor([0.2, 0.7], dtype=torch.float64), x.flip(1), x * 0.5,
            PCA(np.ones(horizon) * 100, np.eye(horizon), np.ones(horizon), 1e-6))


def test_gradient_group_additivity_matches_analytic_linear_result():
    model = Head().double()
    args = arguments()
    result = gradient_diagnostics(model, *args)
    x, _, history, target, _ = args
    error = (model.output(x) + model.encoder(history) - target).detach()
    analytic = 2 * error.T @ x / error.numel()
    assert result["raw_total_gradient_norm"] == pytest.approx(analytic.norm().item())
    assert result["groups"]["pc1"]["gradient_norm"] == pytest.approx(analytic[0].norm().item())
    assert result["groups"]["residual"]["gradient_norm"] == pytest.approx(analytic[2:].norm().item())
    assert result["groups"]["residual"]["gradient_norm_per_component"] == pytest.approx(
        analytic[2:].norm().item() / 2)
    assert result["head_parameter_count"] == 18
    assert result["excluded_encoder_parameter_names"] == ["encoder.weight"]
    assert result["gradient_reconstruction_absolute_error"] < 1e-12
    assert result["loss_reconstruction_absolute_error"] < 1e-12
    assert sum(g["raw_mse_contribution"] for g in result["groups"].values()) == pytest.approx(
        error.square().mean().item())


def test_preserves_mixed_module_states_existing_gradients_and_rng():
    model = Head().double()
    model.train()
    model.encoder.eval()
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 3)
    states = [module.training for module in model.modules()]
    gradients = [(parameter.grad, parameter.grad.clone()) for parameter in model.parameters()]
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        gradient_diagnostics(model, *arguments())
    assert states == [module.training for module in model.modules()]
    assert torch.equal(rng, torch.get_rng_state())
    for parameter, (original, expected) in zip(model.parameters(), gradients):
        assert parameter.grad is original
        assert torch.equal(parameter.grad, expected)


def test_rotated_axes_and_wrapped_head_remain_additive():
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = Head().double()

        def forward(self, *args):
            return self.base(*args)

    model = Wrapper()
    args = list(arguments())
    axes, _ = np.linalg.qr(np.arange(16).reshape(4, 4) + np.eye(4))
    args[-1].components = axes
    result = gradient_diagnostics(model, *args)
    assert result["gradient_reconstruction_absolute_error"] < 1e-12
    assert result["excluded_encoder_parameter_names"] == ["base.encoder.weight"]


def test_zero_gradients_have_null_cosines():
    model = Head().double()
    args = list(arguments())
    args[0] = torch.zeros_like(args[0])
    result = gradient_diagnostics(model, *args)
    assert result["raw_total_gradient_norm"] == 0
    assert all(value is None for value in result["gradient_cosines"].values())
    assert result["gradient_reconstruction_relative_error"] is None


def test_bad_output_restores_module_states():
    model = Head().double()
    model.encoder.eval()
    args = list(arguments())
    args[3][0, 0] = float("nan")
    states = [module.training for module in model.modules()]
    with pytest.raises(ValueError, match="Nonfinite"):
        gradient_diagnostics(model, *args)
    assert states == [module.training for module in model.modules()]


def test_nonorthogonal_pca_is_rejected():
    args = list(arguments())
    args[-1].components *= 2
    with pytest.raises(ValueError, match="orthonormal"):
        gradient_diagnostics(Head().double(), *args)


@pytest.mark.parametrize("kind", ["mlp", "s4"])
def test_original_models_return_finite_additive_diagnostics(kind):
    from fm_stress.models import AdaLNMLP, S4Velocity
    model = (AdaLNMLP if kind == "mlp" else S4Velocity)(
        horizon=4, lookback=4, width=8, encoder_dim=4, time_dim=4)
    args = list(arguments())
    args[:4] = [value.float() for value in args[:4]]
    result = gradient_diagnostics(model, *args)
    assert np.isfinite(result["raw_total_gradient_norm"])
    assert result["gradient_reconstruction_absolute_error"] < 1e-5
    assert all(parameter.grad is None for parameter in model.parameters())
