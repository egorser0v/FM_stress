"""Sparse FM integration, control fidelity, causal conditioning and random streams."""
import copy
import pytest
import torch
import torch.nn.functional as F

from fm_stress.sparse_models import (
    create_sparse_model, topk_mask, HardConcreteGate,
    HistoryConvolutionalFM,
)

HEADS = ('history_dense_ae', 'history_topk_ae', 'history_conv_dense',
         'history_conv_sparse', 'dictionary_dense', 'dictionary_topk', 'unet_l0')


def make(head, channels=3, **kwargs):
    return create_sparse_model(head, lookback=7, horizon=9, channels=channels,
                               width=8, depth=3, time_dim=8, latent_dim=12, topk=3, **kwargs)


@pytest.mark.parametrize('head', HEADS)
@pytest.mark.parametrize('channels', [1, 3, 7, 8])
def test_all_heads_joint_shape_finite_and_learn(head, channels):
    torch.manual_seed(101)
    model = make(head, channels)
    x, history = torch.randn(3, 9, channels), torch.randn(3, 7, channels)
    time = torch.tensor([.1, .5, .9])
    target = x * .3 + history.mean(1)[:, None]
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    for _ in range(4):
        optimizer.zero_grad()
        prediction = model(x, time, history)
        assert prediction.shape == x.shape
        loss = F.mse_loss(prediction, target) + model.auxiliary_loss()
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    model.eval()
    output = model(x, time, history)
    assert output.abs().sum() > 0
    torch.testing.assert_close(output, model(x, time[:, None], history))
    assert all(torch.isfinite(torch.tensor(v)) for v in model.activity_statistics().values())


def test_exact_topk_support_and_selected_gradient():
    values = torch.tensor([[1., -4., 3., 2.], [0., 0., 0., 0.]], requires_grad=True)
    out = topk_mask(values, 2, magnitude=True)
    torch.testing.assert_close(out[0], torch.tensor([0., -4., 3., 0.]))
    assert ((out != 0).sum(-1) <= 2).all()
    out.sum().backward()
    assert values.grad[0].tolist() == [0., 1., 1., 0.]
    with pytest.raises(ValueError):
        topk_mask(values, 5)


@pytest.mark.parametrize('dense,sparse', [('history_dense_ae', 'history_topk_ae'),
                                         ('dictionary_dense', 'dictionary_topk')])
def test_topk_control_has_identical_initial_parameters(dense, sparse):
    torch.manual_seed(21)
    a = make(dense)
    torch.manual_seed(21)
    b = make(sparse)
    assert a.state_dict().keys() == b.state_dict().keys()
    for name, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[name], atol=0, rtol=0)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())


@pytest.mark.parametrize('head', ['history_topk_ae', 'dictionary_topk'])
def test_topk_integrated_codes_obey_budget(head):
    torch.manual_seed(31)
    model = make(head)
    history = torch.randn(5, 7, 3)
    if head == 'history_topk_ae':
        code = model.encode_history(history)
    else:
        # Inspect nontrivial learned-readout regime, not all-zero initialization.
        torch.nn.init.normal_(model.core.output_projection.weight, std=.1)
        code = model.coefficients(torch.randn(5, 9, 3), torch.rand(5), history)
    assert ((code != 0).sum(-1) <= 3).all()
    assert code.count_nonzero() > 0


@pytest.mark.parametrize('head', ['history_dense_ae', 'history_topk_ae',
                                   'history_conv_dense', 'history_conv_sparse'])
def test_history_reconstruction_cannot_see_interpolant_or_time(head):
    model = make(head).eval()
    history = torch.randn(3, 7, 3)
    model(torch.randn(3, 9, 3), torch.rand(3), history)
    first = model.auxiliary_loss().detach().clone()
    model(torch.randn(3, 9, 3) * 100, torch.ones(3), history)
    torch.testing.assert_close(first, model.auxiliary_loss(), atol=0, rtol=0)
    # AE is actually part of the prediction graph once zero initialization opens.
    with torch.no_grad():
        model.core.output_projection.weight.normal_(std=.1)
        model.core.final_modulation[-1].weight.normal_(std=.1)
    h = history.clone().requires_grad_(True)
    model(torch.randn(3, 9, 3), torch.rand(3), h).square().mean().backward()
    assert h.grad.abs().sum() > 0


def test_convolution_sparse_control_threshold_and_reconstruction_gradients():
    model = make('history_conv_sparse')
    x, t, h = torch.randn(4, 9, 3), torch.rand(4), torch.randn(4, 7, 3)
    model(x, t, h)
    model.auxiliary_loss().backward()
    assert model.threshold_logits.grad.abs().sum() > 0
    assert model.dictionary.grad.abs().sum() > 0
    assert model.step_logits.grad.abs().sum() > 0
    code, reconstruction = model.encode_history(h)
    assert reconstruction.shape == h.shape
    assert 0 < (code != 0).float().mean() < 1


def test_conv_dense_equals_sparse_when_threshold_tends_to_zero():
    torch.manual_seed(8)
    dense = make('history_conv_dense')
    sparse = make('history_conv_sparse')
    sparse.load_state_dict(dense.state_dict(), strict=False)
    with torch.no_grad():
        sparse.threshold_logits.fill_(-100)
    h = torch.randn(3, 7, 3)
    da, dr = dense.encode_history(h)
    sa, sr = sparse.encode_history(h)
    torch.testing.assert_close(da, sa)
    torch.testing.assert_close(dr, sr)


def test_conv_tied_operator_adjoint_and_nonexpansive_bound():
    model = make('history_conv_sparse')
    weight = model.normalized_dictionary()
    history, code = torch.randn(2, 3, 7), torch.randn(2, 12, 7)
    analysis = F.conv1d(history, weight, padding=2)
    synthesis = F.conv_transpose1d(code, weight, padding=2)
    torch.testing.assert_close((analysis * code).sum(), (history * synthesis).sum())
    # Construct the finite linear operator, independently checking the bound.
    eye = torch.eye(21).reshape(21, 3, 7)
    operator = F.conv1d(eye, weight, padding=2).flatten(1).T
    assert torch.linalg.matrix_norm(operator, ord=2) <= 1 + 1e-6


def test_hard_concrete_expected_l0_derivative_and_empirical_probability():
    gate = HardConcreteGate(20000, seed=93)
    with torch.no_grad():
        gate.log_alpha.fill_(-.7)
    probability = gate.expected_nonzero().mean()
    observed = (gate(torch.ones(1, 20000, 1)) != 0).float().mean()
    assert abs(float(probability.detach() - observed)) < .015
    probability.backward()
    assert torch.all(gate.log_alpha.grad > 0)


def test_gate_rng_is_private_checkpointed_and_eval_deterministic():
    gate = HardConcreteGate(32, seed=44)
    x = torch.ones(3, 32, 5)
    global_state = torch.random.get_rng_state().clone()
    gate(x)
    assert torch.equal(global_state, torch.random.get_rng_state())
    state = copy.deepcopy(gate.state_dict())
    continuation = gate(x)
    other = HardConcreteGate(32, seed=99)
    other.load_state_dict(state)
    torch.testing.assert_close(continuation, other(x), atol=0, rtol=0)
    gate.eval()
    before = gate.get_extra_state()['rng_state'].clone()
    torch.testing.assert_close(gate(x), gate(x), atol=0, rtol=0)
    assert torch.equal(before, gate.get_extra_state()['rng_state'])


def test_l0_unet_recovers_original_when_gates_forced_open():
    from fm_stress.extended_models import TemporalUNet
    torch.manual_seed(77)
    ordinary = TemporalUNet(7, 9, 3, width=8, depth=3, time_dim=8)
    torch.manual_seed(77)
    gated = make('unet_l0').eval()
    ordinary.eval()
    with torch.no_grad():
        ordinary.output_projection.weight.normal_(std=.1)
        gated.output_projection.weight.copy_(ordinary.output_projection.weight)
        for gate in gated.gates():
            gate.log_alpha.fill_(100)
    x, h, t = torch.randn(2, 9, 3), torch.randn(2, 7, 3), torch.rand(2)
    torch.testing.assert_close(ordinary(x, t, h), gated(x, t, h), atol=0, rtol=0)


def test_invalid_shapes_and_options_rejected():
    model = make('history_topk_ae')
    with pytest.raises(ValueError, match='x must'):
        model(torch.randn(3, 9), torch.rand(3), torch.randn(3, 7, 3))
    with pytest.raises(ValueError, match='topk'):
        create_sparse_model('history_topk_ae', 7, 9, 3, topk=100, latent_dim=12)
    with pytest.raises(ValueError, match='odd'):
        HistoryConvolutionalFM(7, 9, 3, conv_kernel=4)


def test_dictionary_initialization_uses_more_than_one_topk_support():
    torch.manual_seed(311)
    model = make('dictionary_topk')
    code = model.coefficients(torch.randn(64, 9, 3), torch.rand(64), torch.randn(64, 7, 3))
    assert ((code != 0).sum(-1) <= model.topk).all()
    assert (code != 0).any(0).sum() > model.topk
    prediction = model(torch.randn(64, 9, 3), torch.rand(64), torch.randn(64, 7, 3))
    prediction.square().sum().backward()
    assert (model.core.output_projection.weight.grad.abs().sum(1) > 0).sum() > model.topk
