import numpy as np
import pytest
import torch

from fm_stress.metrics import PCA
from fm_stress.prior_mixture import MixtureHead, objective
from scripts.analyze_prior_training_gradients import actual_training_diagnostics


class LinearHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4)
        self.output = torch.nn.Linear(4, 4)

    def forward(self, x, t, history):
        return self.output(x) + self.encoder(history)


@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("whiten_input", [False, True])
def test_actual_objective_gradient_matches_direct_autograd(balanced, whiten_input):
    q, _ = np.linalg.qr(np.arange(16).reshape(4, 4) + np.eye(4))
    pca = PCA(np.zeros(4), q, np.array([4., 1., .1, .01]), 1e-6)
    model = MixtureHead(LinearHead(), pca, whiten_input, balanced)
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
    t = torch.tensor([.1, .3, .8])
    h, u = x.flip(1), x / 2
    before = torch.get_rng_state().clone()
    result = actual_training_diagnostics(model, x, t, h, u, pca)
    assert torch.equal(before, torch.get_rng_state())
    assert all(parameter.grad is None for parameter in model.parameters())
    loss = objective(model, model(x, t, h), u)
    params = [p for name, p in model.named_parameters() if "encoder" not in name]
    gradients = torch.autograd.grad(loss, params)
    norm = torch.cat([g.flatten() for g in gradients]).norm().item()
    assert result["objective_mse"] == pytest.approx(loss.item(), rel=2e-5)
    assert result["objective_total_gradient_norm"] == pytest.approx(norm, rel=2e-5)
    assert sum(g["objective_mse_contribution"] for g in result["groups"].values()) == pytest.approx(
        loss.item(), rel=2e-5)
    assert "raw_total_mse" not in result
    assert ("PCA" in result["coordinate_system"]) == balanced
