"""Routing controls must differ only by active expert selection."""
import pytest
import torch

from fm_stress.sparse_moe import TinyMoE


def inputs():
    return torch.randn(8, 7, 3), torch.rand(8), torch.randn(8, 11, 3)


def activate_experts(model):
    for expert in model.experts:
        torch.nn.init.normal_(expert[-1].weight, std=.1)


def test_top_all_matches_dense_in_outputs_aux_and_gradients():
    torch.manual_seed(2)
    dense = TinyMoE(11, 7, 3, width=12, experts=4)
    activate_experts(dense)
    routed = TinyMoE(11, 7, 3, width=12, experts=4, topk=4)
    routed.load_state_dict(dense.state_dict())
    args = inputs()
    a, b = dense(*args), routed(*args)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(dense.auxiliary_loss(), routed.auxiliary_loss())
    (a.square().mean()+dense.auxiliary_loss()).backward()
    (b.square().mean()+routed.auxiliary_loss()).backward()
    for p, q in zip(dense.parameters(), routed.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)


def test_top_two_is_deterministic_sparse_and_trainable():
    torch.manual_seed(3)
    model = TinyMoE(11, 7, 3, width=12, experts=4, topk=2)
    activate_experts(model)
    args = inputs()
    pred = model(*args)
    torch.testing.assert_close(pred, model(*args), atol=0, rtol=0)
    assert pred.shape == args[0].shape
    assert float(model.activity_statistics()['routing_fraction']) == .5
    assert float(model.auxiliary_loss().detach()) >= -1e-7
    (pred.square().mean()+model.auxiliary_loss()).backward()
    assert model.router.weight.grad.abs().sum() > 0
    assert model.history.weight.grad.abs().sum() > 0
    assert model.state.weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS unavailable')
def test_mps_top_two_matches_cpu():
    torch.manual_seed(7)
    model = TinyMoE(11, 7, 3, width=12, experts=4, topk=2)
    activate_experts(model)
    args = inputs()
    expected = model(*args)
    actual = model.to('mps')(*(a.to('mps') for a in args)).cpu()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
