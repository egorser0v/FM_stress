import numpy as np
import pytest
from sklearn.linear_model import MultiTaskElasticNet
from fm_stress.sparse_baselines_qr import select_elastic_net, original_objective_certificate


@pytest.mark.parametrize('n,p',[(1000,30),(20,30)])
def test_qr_preserves_original_multitask_objective_and_forecasts(n,p):
    rng=np.random.default_rng(101)
    x=rng.normal(size=(n,p))+rng.normal(size=p)
    y=x@rng.normal(size=(p,5))+.2*rng.normal(size=(n,5))+rng.normal(size=5)
    vx=rng.normal(size=(70,p));vy=vx@rng.normal(size=(p,5))
    compressed,candidates,selected=select_elastic_net(x,y,vx,vy,tol=1e-9,max_iter=50000)
    alpha,ratio=candidates[selected]['alpha'],candidates[selected]['l1_ratio']
    direct=MultiTaskElasticNet(alpha=alpha,l1_ratio=ratio,tol=1e-9,max_iter=50000).fit(x,y)
    np.testing.assert_allclose(compressed.coef_,direct.coef_,atol=2e-6,rtol=2e-6)
    np.testing.assert_allclose(compressed.predict(vx),direct.predict(vx),atol=1e-5,rtol=1e-5)
    cert=original_objective_certificate(x,y,compressed,1e-9)
    assert cert['dual_certificate_passed']
    assert cert['intercept_gradient_max_abs']<1e-10
    assert cert['duality_gap']<cert['dual_gap_tolerance']*1.1+1e-10


def test_original_certificate_rejects_incorrect_coefficients():
    rng=np.random.default_rng(2)
    x=rng.normal(size=(60,10));y=x@rng.normal(size=(10,3))
    model=MultiTaskElasticNet(alpha=.01,l1_ratio=.9,tol=1e-9,max_iter=50000).fit(x,y)
    assert original_objective_certificate(x,y,model,1e-9)['dual_certificate_passed']
    model.coef_[0,0]+=1
    cert=original_objective_certificate(x,y,model,1e-9)
    assert not cert['dual_certificate_passed']
    assert cert['maximum_feature_kkt_residual']>.1
