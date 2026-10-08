import copy

import numpy as np
import pytest

from fm_stress.extended_data import build_data
from fm_stress.extended_metrics import forecast_scores, means
from fm_stress.sparse_baselines import fit_sparse_baselines, select_elastic_net


def small_bundle():
    return build_data(dict(kind='synthetic',lookback=4,horizon=3,channels=2,
        lengthscale=2,source_lengthscale=2,channel_correlation=.4,seed=21,
        n_train=160,n_val=30,n_test=23))


def test_selection_uses_prespecified_grid_and_independent_validation_errors():
    bundle=small_bundle()
    train,val=bundle.splits['train'],bundle.splits['val']
    x,y=train.context.reshape(160,-1),train.target.reshape(160,-1)
    vx,vy=val.context.reshape(30,-1),val.target.reshape(30,-1)
    model,candidates,selected=select_elastic_net(x,y,vx,vy)
    assert len(candidates)==12
    assert {(r['alpha'],r['l1_ratio']) for r in candidates}=={
        (a,r) for a in [.001,.01,.1,1.] for r in [.5,.9,1.]}
    assert candidates[selected]['converged']
    assert candidates[selected]['validation_mse']==pytest.approx(np.mean((model.predict(vx)-vy)**2))
    assert candidates[selected]['validation_mse']==min(r['validation_mse'] for r in candidates if r['converged'])
    assert any(r['active_input_features']<x.shape[1] for r in candidates)


def test_test_perturbation_cannot_change_fit_and_bootstrap_reconstructs_vectors(tmp_path):
    bundle=small_bundle()
    cfg={'n_ensemble_contexts':7,'ensemble_size':9,'evaluation_seed':100}
    record=fit_sparse_baselines(bundle,cfg,tmp_path/'first')
    changed=copy.deepcopy(bundle)
    changed.splits['test'].context+=20
    changed.splits['test'].target-=30
    second=fit_sparse_baselines(changed,cfg,tmp_path/'changed')
    a=np.load(tmp_path/'first'/'sparse_baseline_samples.npz')
    b=np.load(tmp_path/'changed'/'sparse_baseline_samples.npz')
    for key in ('coefficients','intercept','active_input_mask','residual_draw_indices','training_residuals'):
        np.testing.assert_array_equal(a[key],b[key])
    assert record['selected_alpha']==second['selected_alpha']
    assert record['selected_l1_ratio']==second['selected_l1_ratio']
    np.testing.assert_array_equal(a['indices'],np.linspace(0,22,7,dtype=int))
    draws=np.random.default_rng(115).integers(160,size=(7,9))
    np.testing.assert_array_equal(a['residual_draw_indices'],draws)
    train=bundle.splits['train']
    residual=train.target.reshape(160,-1)-(
        train.context.reshape(160,-1)@a['coefficients'].T+a['intercept'])
    np.testing.assert_allclose(a['training_residuals'],residual)
    expected=a['multitask_elastic_net'][:,None]+residual[draws].reshape(7,9,3,2)
    np.testing.assert_allclose(a['multitask_elastic_net_residual_bootstrap'],expected)
    scores=means(forecast_scores(expected,bundle.splits['test'].target[a['indices']]))
    for key,value in scores.items():
        assert record['records']['multitask_elastic_net_residual_bootstrap']['means'][key]==pytest.approx(value)
    assert record['active_input_features']==np.any(a['coefficients']!=0,axis=0).sum()


def test_nonconverged_candidates_are_recorded_and_excluded():
    rng=np.random.default_rng(98)
    x=rng.normal(size=(60,12));y=.1*(x@rng.normal(size=(12,4))+rng.normal(size=(60,4))*.1)
    _,candidates,selected=select_elastic_net(x,y,x,y,max_iter=1,tol=1e-14)
    assert any(not r['converged'] for r in candidates)
    assert all(r['warnings'] for r in candidates if not r['converged'])
    assert candidates[selected]['converged']
    with pytest.raises(RuntimeError,match='No Elastic Net candidate converged'):
        select_elastic_net(x,10*y,x,10*y,max_iter=1,tol=1e-14)


def test_resume_verifies_config_and_sample_integrity(tmp_path):
    cfg={'n_ensemble_contexts':5,'ensemble_size':8,'evaluation_seed':10}
    bundle=small_bundle()
    first=fit_sparse_baselines(bundle,cfg,tmp_path)
    assert fit_sparse_baselines(bundle,cfg,tmp_path)==first
    with pytest.raises(ValueError,match='prespecified'):
        fit_sparse_baselines(bundle,{**cfg,'elastic_alphas':[.1]},tmp_path)
    with pytest.raises(RuntimeError,match='provenance'):
        fit_sparse_baselines(bundle,{**cfg,'evaluation_seed':11},tmp_path)
    with (tmp_path/'sparse_baseline_samples.npz').open('ab') as f:
        f.write(b'changed')
    with pytest.raises(RuntimeError,match='changed'):
        fit_sparse_baselines(bundle,cfg,tmp_path)
