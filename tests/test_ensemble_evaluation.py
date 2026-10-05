import numpy as np
import pytest

from fm_stress.ensemble_evaluation import (fair_crps, fair_energy_score,
    interval_metrics, context_scores, shared_inputs, aggregate)
from fm_stress.data import GPConfig


def explicit_scores(x, y):
    h, s, f = x.shape
    crps, energy = np.zeros(h), np.zeros(h)
    for i in range(h):
        first_c = sum(np.abs(x[i,j]-y[i]).mean() for j in range(s))/s
        pair_c = sum(np.abs(x[i,j]-x[i,k]).mean() for j in range(s) for k in range(j+1,s))/(s*(s-1))
        first_e = sum(np.linalg.norm(x[i,j]-y[i]) for j in range(s))/s
        pair_e = sum(np.linalg.norm(x[i,j]-x[i,k]) for j in range(s) for k in range(j+1,s))/(s*(s-1))
        crps[i], energy[i] = first_c-pair_c, first_e-pair_e
    return crps, energy


def test_crps_known_value_and_sorted_parity():
    x = np.array([[[0.], [2.]]])
    # E|X-0| = 1, half offdiagonal E|X-X'| = 1.
    np.testing.assert_allclose(fair_crps(x, np.array([[0.]])), [0.])
    x = np.random.default_rng(41).normal(size=(4,7,3))
    y = np.random.default_rng(42).normal(size=(4,3))
    crps, _ = explicit_scores(x,y)
    np.testing.assert_allclose(fair_crps(x,y), crps, atol=1e-14)
    np.testing.assert_allclose(fair_crps(x[:,::-1],y), crps, atol=1e-14)


def test_energy_explicit_loops_translation_rotation_and_scale():
    rng=np.random.default_rng(4)
    x,y=rng.normal(size=(3,5,4)),rng.normal(size=(3,4))
    _, expected=explicit_scores(x,y)
    np.testing.assert_allclose(fair_energy_score(x,y),expected,atol=1e-14)
    q,_=np.linalg.qr(rng.normal(size=(4,4)))
    np.testing.assert_allclose(fair_energy_score(x@q+3,y@q+3),expected,atol=1e-14)
    np.testing.assert_allclose(fair_energy_score(2*x,2*y),2*expected,atol=1e-14)


def test_interval_definition_linear_quantiles_and_inclusive_boundaries():
    x=np.array([[[0.,0.],[10.,10.]]])
    # Type-7 central80 interval [1,9], including endpoints.
    coverage,width=interval_metrics(x,np.array([[1.,9.]]),.8)
    np.testing.assert_allclose(coverage,[1.])
    np.testing.assert_allclose(width,[8.])
    coverage,_=interval_metrics(x,np.array([[.99,9.01]]),.8)
    np.testing.assert_allclose(coverage,[0.])
    coverage,width=interval_metrics(x,np.array([[.5,9.5]]),.9)
    np.testing.assert_allclose(coverage,[1.])
    np.testing.assert_allclose(width,[9.])


def test_context_weighting_and_shared_input_grouping():
    gp=GPConfig(lookback=4,horizon=3,lengthscale=2)
    cfg=dict(n_histories=3,ensemble_size=5,data_seed=87041,source_seed=88041)
    ds,eps=shared_inputs(gp,cfg)
    histories=np.repeat(ds.context,cfg['ensemble_size'],axis=0)
    assert eps.shape==(3,5,3)
    for h in range(3):
        np.testing.assert_array_equal(histories[5*h:5*h+5],np.repeat(ds.context[h:h+1],5,axis=0))
    ds2,eps2=shared_inputs(gp,cfg)
    np.testing.assert_array_equal(ds.target,ds2.target)
    np.testing.assert_array_equal(eps,eps2)
    scores=context_scores(eps,ds.target)
    assert all(v.shape==(3,) for v in scores.values())
    np.testing.assert_allclose(scores['generated_roughness'],np.abs(np.diff(eps,axis=2)).mean(axis=(1,2)))


def test_aggregate_bootstraps_histories_after_seed_average():
    cfg=dict(heads=['mlp','s4','whitened_mlp'],seeds=[0,1],bootstrap_seed=5,bootstrap_repetitions=100)
    records=[]
    for head,shift in [('mlp',0),('s4',2),('whitened_mlp',-1)]:
        for seed in cfg['seeds']:
            vals=np.arange(5,dtype=float)+seed+shift
            records.append(dict(head=head,seed=seed,means={'fair_crps':float(vals.mean())},per_context={'fair_crps':vals.tolist()}))
    summary=aggregate(records,cfg,'test')
    assert summary['complete']
    assert summary['contrasts'][0]['pairs']==5
    assert summary['contrasts'][0]['mean_difference']==2
    assert summary['contrasts'][0]['ci_low']==2
    assert summary['contrasts'][0]['ci_high']==2
    assert not aggregate(records[:-1],cfg,'test')['complete']
    assert not aggregate(records[:-1],cfg,'test')['contrasts']


def test_nonfinite_or_single_member_is_rejected():
    with pytest.raises(ValueError):
        fair_crps(np.zeros((2,1,3)),np.zeros((2,3)))
    with pytest.raises(ValueError):
        fair_energy_score(np.full((2,3,3),np.nan),np.zeros((2,3)))
