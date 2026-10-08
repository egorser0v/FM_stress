"""Independent invariants for joint time/channel velocity coordinates."""
import numpy as np
import pytest
import torch
from torch import nn

from fm_stress.metrics import fit_pca
from fm_stress.sparse_coordinates import JointCoordinates


class EchoField(nn.Module):
    def __init__(self, horizon=7, channels=3):
        super().__init__()
        self.horizon, self.channels = horizon, channels
        self.seen = None

    def forward(self, x, t, history):
        self.seen = (x, t, history)
        return x


def make_coordinates(transform=True, balanced=False, extreme=False):
    rng = np.random.default_rng(919)
    q, _ = np.linalg.qr(rng.normal(size=(21, 21)))
    scales = np.geomspace(3., 1e-5 if extreme else .15, 21)
    # Nonzero train mean makes accidental derivative centering detectable.
    train = (rng.normal(size=(800, 21)) * scales) @ q + np.arange(21) * 2.
    pca = fit_pca(train)
    return JointCoordinates(EchoField(), pca, transform, balanced), pca


@pytest.mark.parametrize('extreme', [False, True])
def test_joint_round_trip_retains_every_coordinate(extreme):
    model, pca = make_coordinates(extreme=extreme)
    torch.manual_seed(2)
    x = torch.randn(5, 7, 3)
    z = model.to_coordinates(x)
    assert z.shape == x.shape
    assert model.q.shape == (21, 21)
    torch.testing.assert_close(model.from_coordinates(z), x, atol=2e-6, rtol=2e-6)
    assert torch.isfinite(z).all()
    if extreme:
        assert np.any(pca.eigenvalues < pca.floor)


def test_interpolation_and_velocity_derivative_use_same_linear_map():
    model, _ = make_coordinates()
    torch.manual_seed(3)
    eps, y = torch.randn(6, 7, 3), torch.randn(6, 7, 3)
    t = torch.linspace(0., 1., 6).reshape(-1, 1, 1)
    z = model.to_coordinates((1-t)*eps + t*y)
    torch.testing.assert_close(z, (1-t)*model.to_coordinates(eps)+t*model.to_coordinates(y), atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(model.to_coordinates(y-eps), model.to_coordinates(y)-model.to_coordinates(eps), atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(model.to_coordinates(torch.zeros_like(y)), torch.zeros_like(y), atol=0, rtol=0)


def test_forward_restores_velocity_and_leaves_history_untouched():
    model, _ = make_coordinates()
    x, h, t = torch.randn(4, 7, 3), torch.randn(4, 11, 3), torch.rand(4)
    out = model(x, t, h)
    torch.testing.assert_close(out, x, atol=2e-6, rtol=2e-6)
    assert model.base.seen[1] is t
    assert model.base.seen[2] is h
    torch.testing.assert_close(model.base.seen[0], model.to_coordinates(x))


@pytest.mark.parametrize('transform,balanced', [(False, False), (False, True), (True, False)])
def test_objective_matches_independent_pca_formula(transform, balanced):
    model, pca = make_coordinates(transform, balanced)
    prediction, truth = torch.randn(6, 7, 3), torch.randn(6, 7, 3)
    error = (prediction-truth).numpy().reshape(6, -1).astype(np.float64)
    if transform or balanced:
        error = error @ pca.components.T / pca.scales
    assert float(model.objective(prediction, truth)) == pytest.approx(float(np.square(error).mean()), rel=2e-6)


def test_whitened_and_raw_euler_updates_are_equivalent_for_same_field():
    model, _ = make_coordinates()
    x = torch.randn(3, 7, 3)
    history, t = torch.randn(3, 11, 3), torch.zeros(3)
    raw = x.clone()
    z = model.to_coordinates(x)
    # The EchoField in whitened coordinates induces v(x)=x in raw coordinates.
    for _ in range(16):
        raw = raw + model(raw, t, history)/16
        z = z + model.base(z, t, history)/16
    torch.testing.assert_close(raw, model.from_coordinates(z), atol=8e-6, rtol=8e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS unavailable')
def test_whitening_mps_cpu_parity_with_floored_spectrum():
    model, _ = make_coordinates(extreme=True)
    torch.manual_seed(5)
    x = torch.randn(4, 7, 3)
    expected = model.to_coordinates(x)
    actual = model.to('mps').to_coordinates(x.to('mps')).cpu()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-5)
