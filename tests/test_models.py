"""Numerical checks for the real S4 adaptation and model interfaces."""
import math

import pytest
import torch

from fm_stress.models import (S4NPLRKernel, S4TemporalLayer, _hippo_nplr,
                              create_model)


def complex_reference(kernel, length):
    """Independent dense complex bilinear SSM, all conjugate states retained."""
    n = kernel.state_size // 2
    w = torch.complex(-kernel.log_a_real.exp(), kernel.a_imag)
    p = torch.complex(kernel.p[:, :n], kernel.p[:, n:])
    b = torch.complex(kernel.b[:, :n], kernel.b[:, n:])
    c = torch.complex(kernel.c[..., :n], -kernel.c[..., n:]) / 2
    w, p, b, c = [torch.cat((v, v.conj()), dim=-1) for v in (w, p, b, c)]
    a = torch.diag_embed(w) - p.unsqueeze(-1) * p.conj().unsqueeze(-2)
    eye = torch.eye(2 * n, dtype=a.dtype, device=a.device)
    dt = kernel.log_dt.exp()[:, None, None]
    ab = torch.linalg.solve(eye - dt * a / 2, eye + dt * a / 2)
    bb = torch.linalg.solve(eye - dt * a / 2, dt * b.unsqueeze(-1))
    outputs = []
    state = bb
    for _ in range(length):
        outputs.append(torch.einsum("dhn,hnk->dhk", c, state).squeeze(-1).real)
        state = ab @ state
    return torch.stack(outputs, dim=-1)


def test_real_s4_kernel_and_gradients_match_complex_bilinear():
    torch.manual_seed(9)
    kernel = S4NPLRKernel(3, state_size=8).double()
    real = kernel(16)
    reference = complex_reference(kernel, 16)
    torch.testing.assert_close(real, reference, atol=1e-10, rtol=1e-9)
    weights = torch.randn_like(real)
    real_grad = torch.autograd.grad((real * weights).sum(), tuple(kernel.parameters()))
    complex_grad = torch.autograd.grad((reference * weights).sum(), tuple(kernel.parameters()))
    for a, b in zip(real_grad, complex_grad):
        torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-8)


def test_initial_s4_is_similar_to_hippo_legs():
    n = 8
    w, p, b = _hippo_nplr(n)
    fullw = torch.cat((w, w.conj()))
    fullp = torch.cat((p, p.conj()))
    matrix = torch.diag(fullw) - fullp[:, None] * fullp.conj()[None]
    # LegS eigenvalues are -1, -2, ..., -N. Eigenvalues of highly nonnormal
    # matrices are ill-conditioned in float32; compare trace invariants instead.
    torch.testing.assert_close(matrix.trace().real, -torch.arange(1, n + 1).float().sum())
    fullb = torch.cat((b, b.conj()))
    torch.testing.assert_close(fullb.abs().square().sum(), torch.tensor(float(n * n)))


def test_bidirectional_dense_convolution_matches_official_fft_convention():
    torch.manual_seed(19)
    layer = S4TemporalLayer(3, state_size=8).double()
    x = torch.randn(2, 16, 3, dtype=torch.float64)
    actual = layer(x)
    kernel = layer.kernel(16)
    k = torch.nn.functional.pad(kernel[0], (0, 16)) + torch.nn.functional.pad(kernel[1].flip(-1), (16, 0))
    y = torch.fft.irfft(torch.fft.rfft(x.transpose(1, 2), n=32) * torch.fft.rfft(k, n=32), n=32)[..., :16]
    reference = layer.mix(torch.nn.functional.gelu(y.transpose(1, 2) + x * layer.skip))
    torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("kind", ["mlp", "s4"])
def test_model_shapes_endpoints_and_learning(kind):
    torch.manual_seed(23)
    model = create_model(kind, width=16, state_size=8)
    x, history = torch.randn(4, 16), torch.randn(4, 32)
    t = torch.tensor([0., 1., 0.2, 0.8])
    output = model(x, t, history)
    assert output.shape == x.shape
    torch.testing.assert_close(output, model(x, t[:, None], history))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        loss = (model(x, t, history) - x).square().mean()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        opt.step()
    assert all(p.grad is not None for p in model.parameters())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not exposed")
def test_mps_matches_cpu_kernel_and_backward():
    torch.manual_seed(33)
    cpu = S4NPLRKernel(4, state_size=16)
    mps = S4NPLRKernel(4, state_size=16).to("mps")
    mps.load_state_dict(cpu.state_dict())
    a, b = cpu(16), mps(16)
    torch.testing.assert_close(a, b.cpu(), atol=3e-5, rtol=3e-4)
    a.square().sum().backward()
    b.square().sum().backward()
    for p, q in zip(cpu.parameters(), mps.parameters()):
        torch.testing.assert_close(p.grad, q.grad.cpu(), atol=1e-4, rtol=2e-3)
