"""Joint-channel capacity, sparse functional masks and U-Net temporal geometry."""
import io
import pytest
import torch
from fm_stress.extended_models import (create_extended_model, MaskedLinear,parameter_counts,optimizer_groups)

HEADS=('mlp','sparse_mlp','unet','s4')

@pytest.mark.parametrize('head',HEADS)
def test_joint_shape_condition_shapes_zero_initialization_and_gradients(head):
    torch.manual_seed(44)
    model=create_extended_model(head,7,15,3,width=8,depth=3,state_size=8,density=.5,time_dim=8)
    x,h,t=torch.randn(3,15,3),torch.randn(3,7,3),torch.tensor([0.,.4,1.])
    y=model(x,t,h)
    assert y.shape==x.shape
    torch.testing.assert_close(y,torch.zeros_like(x),atol=0,rtol=0)
    torch.testing.assert_close(y,model(x,t[:,None],h),atol=0,rtol=0)
    opt=torch.optim.AdamW(optimizer_groups(model,1e-3,1e-4))
    target=x.roll(1,dims=-1)+h.mean(1)[:,None]
    for _ in range(3):
        opt.zero_grad();loss=(model(x,t,h)-target).square().mean();loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
    assert torch.isfinite(model(x,t,h)).all()
    assert model(x,t,h).abs().sum()>0


@pytest.mark.parametrize('head',HEADS)
def test_other_channels_influence_output_and_history_conditioning(head):
    torch.manual_seed(57)
    model=create_extended_model(head,7,9,3,width=12,state_size=8,density=.5,time_dim=8)
    # Open deliberately zero-initialized output/conditioning maps to examine
    # architecture connectivity, not the trivial all-zero initialization.
    with torch.no_grad():
        for module in model.modules():
            weight=getattr(module,'weight',None)
            if isinstance(weight,torch.nn.Parameter) and weight.ndim>=2 and not weight.count_nonzero():
                torch.nn.init.normal_(weight,std=.1)
    x=torch.randn(2,9,3,requires_grad=True);h=torch.randn(2,7,3,requires_grad=True)
    y=model(x,torch.tensor([.2,.8]),h)[:,:,0].square().sum()
    gx,gh=torch.autograd.grad(y,(x,h))
    assert gx[:,:,1].abs().sum()>1e-9, 'Head behaves like independent per-channel models'
    assert gh[:,:,1].abs().sum()>1e-9, 'History from other channels is inaccessible'


def test_sparse_mask_is_reproducible_persistent_and_functionally_enforced():
    def make():return create_extended_model('sparse_mlp',4,5,2,width=8,density=.5,time_dim=8)
    torch.manual_seed(9);a=make();torch.manual_seed(9);b=make();torch.manual_seed(10);c=make()
    ma=[m for m in a.modules() if isinstance(m,MaskedLinear)]
    mb=[m for m in b.modules() if isinstance(m,MaskedLinear)]
    mc=[m for m in c.modules() if isinstance(m,MaskedLinear)]
    assert len(ma)>3
    assert all(torch.equal(x.mask,y.mask) for x,y in zip(ma,mb))
    assert any(not torch.equal(x.mask,y.mask) for x,y in zip(ma,mc))
    layer=ma[0];x=torch.randn(3,layer.in_features)
    expected=torch.nn.functional.linear(x,layer.weight*layer.mask,layer.bias)
    torch.testing.assert_close(layer(x),expected)
    layer(x).square().sum().backward()
    assert torch.count_nonzero(layer.weight.grad[layer.mask==0])==0
    before=layer(x).detach().clone()
    with torch.no_grad():layer.weight[layer.mask==0]=1e8
    torch.testing.assert_close(before,layer(x),atol=0,rtol=0)
    state=io.BytesIO();torch.save(a.state_dict(),state);state.seek(0)
    c.load_state_dict(torch.load(state,weights_only=True))
    assert all(torch.equal(x.mask,y.mask) for x,y in zip(ma,mc))
    counts=parameter_counts(a)
    expected_active=sum(p.numel() for p in a.parameters())-sum(int((m.mask==0).sum()) for m in ma)
    assert counts['active']==expected_active<counts['allocated']


@pytest.mark.parametrize('horizon',[1,5,16,23])
def test_unet_skip_alignment_for_arbitrary_temporal_lengths(horizon):
    model=create_extended_model('unet',5,horizon,2,width=4,depth=3,time_dim=8)
    x=torch.randn(1,horizon,2);h=torch.randn(1,5,2)
    out=model(x,torch.tensor([.5]),h)
    assert out.shape==x.shape
    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_s4_optimizer_preserves_ssm_no_decay_and_allocation_count():
    model=create_extended_model('s4',7,9,3,width=8,state_size=8,time_dim=8)
    groups=optimizer_groups(model,.001,.03)
    actual={id(p):group['weight_decay'] for group in groups for p in group['params']}
    special=[p for p in model.parameters() if hasattr(p,'_optim')]
    assert special and all(actual[id(p)]==0 for p in special)
    counts=parameter_counts(model)
    assert counts['active']==counts['allocated']==sum(p.numel() for p in model.parameters())


def test_shape_validation_rejects_implicit_channel_batching():
    model=create_extended_model('mlp',5,7,2,width=8,time_dim=8)
    with pytest.raises(ValueError,match='x must'):
        model(torch.zeros(4,7),torch.zeros(4),torch.zeros(4,5,2))
