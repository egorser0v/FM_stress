import numpy as np
import pytest
import torch
from fm_stress.extended_data import WindowDataset
from fm_stress.followup_normalization import transform, prior_numpy, training_prior, objective, identities


def sample():
    rng=np.random.default_rng(7)
    return WindowDataset(rng.normal(size=(5,48,1)),rng.normal(size=(5,24,1)),np.arange(5))


@pytest.mark.parametrize('mode',['global','center','scale','history','history_rawloss'])
def test_transform_inverse_and_future_leakage(mode):
    ds=sample();nd,mu,sd=transform(ds,mode)
    np.testing.assert_allclose(nd.target*sd+mu,ds.target,atol=1e-14)
    perturbed=WindowDataset(ds.context,ds.target+900,ds.starts)
    other,mu2,sd2=transform(perturbed,mode)
    np.testing.assert_array_equal(mu,mu2);np.testing.assert_array_equal(sd,sd2)
    np.testing.assert_array_equal(nd.context,other.context)


@pytest.mark.parametrize('mode',['center','history','history_rawloss'])
def test_level_equivariance(mode):
    ds=sample();a,mu,sd=transform(ds,mode)
    b,mu2,sd2=transform(WindowDataset(ds.context+3,ds.target+3,ds.starts),mode)
    np.testing.assert_allclose(a.context,b.context,atol=1e-14)
    np.testing.assert_allclose(sd,sd2);np.testing.assert_allclose(mu2-mu,3)


def test_source_projection_and_replacement():
    f=24;cov={'gp':.8*np.ones((f,f))+.2*np.eye(f),'white':np.eye(f)}
    gp=prior_numpy(cov,10000,'gp',98,f);zero=prior_numpy(cov,10000,'gp_zero_level',98,f)
    white=prior_numpy(cov,10000,'gp_white_level',98,f)
    np.testing.assert_allclose(zero,gp-gp.mean(1,keepdims=True),atol=1e-14)
    np.testing.assert_allclose(white-white.mean(1,keepdims=True),zero,atol=1e-14)
    assert abs(white.mean(1).var()-1/f)<.002
    z=np.random.default_rng(98).standard_normal((10000,f))
    level=np.random.default_rng(40098).standard_normal((10000,1,1))
    got=training_prior(torch.tensor(z),torch.tensor(np.linalg.cholesky(cov['gp'])),'gp_white_level',f,torch.tensor(level))
    np.testing.assert_allclose(got.numpy(),white,atol=1e-14)


def test_loss_units():
    x=torch.randn(4,24,1);u=torch.randn_like(x);sd=torch.rand(4,1,1)+.1
    assert torch.allclose(objective(x,u,sd,'history_rawloss'),((sd*x-sd*u)**2).mean())
    assert torch.allclose(objective(x,u,sd,'history'),((x-u)**2).mean())


def test_frozen_matrix():
    import json
    from pathlib import Path
    c=json.loads((Path(__file__).parents[1]/'configs/followup_normalization.json').read_text())
    assert len(identities(c))==len(set(identities(c)))==84
