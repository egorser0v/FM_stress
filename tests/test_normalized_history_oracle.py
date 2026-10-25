import numpy as np
from fm_stress.data import GPConfig, kernel_covariance, source_covariance, sample_dataset, conditional_oracle_velocity
from fm_stress.normalized_history_oracle import NormalizedHistoryOracle


def setup(source='gp'):
    g=GPConfig()
    data=sample_dataset(g, 24, 981231)
    oracle=NormalizedHistoryOracle(kernel_covariance(g), g.lookback, source_covariance(g,source))
    return g,data,oracle


def test_posterior_mean_and_covariance_by_raw_history_integration():
    g,data,o=setup()
    h=data.context[:1]
    p=o.posterior(h)
    rng=np.random.default_rng(71)
    n=50000
    s=np.sqrt(rng.gamma(o.shape,2/p.rate_twice[0],n))
    b=h[0]@o.p1
    m=-s*b/o.c+rng.normal(size=n)/np.sqrt(o.c)
    raw_h=m[:,None]+s[:,None]*h
    joint=kernel_covariance(g)
    R=joint[g.lookback:,g.lookback:]-o.A@joint[:g.lookback,g.lookback:]
    raw_y=raw_h@o.A.T+rng.normal(size=(n,g.horizon))@np.linalg.cholesky((R+R.T)/2).T
    y=(raw_y-m[:,None])/s[:,None]
    mean,cov=o.target_moments(h)
    np.testing.assert_allclose(y.mean(0),mean[0],atol=.025)
    np.testing.assert_allclose(np.cov(y,rowvar=False),cov[0],rtol=.05,atol=.003)


def test_small_scale_branch_is_exact_gaussian():
    g,d,o=setup('white')
    h=d.context[:8]*.005
    mean,cov=o.target_moments(h)
    np.testing.assert_allclose(cov,np.broadcast_to(o.V,cov.shape))
    x=np.random.default_rng(42).normal(size=(8,g.horizon))
    t=np.linspace(.1,.9,8)
    got=o.flow(x,t,h)
    expected=conditional_oracle_velocity(x,t,mean,cov,o.C)
    np.testing.assert_allclose(got['velocity'],expected,rtol=1e-8,atol=1e-8)
    assert got['small_scale_branch'].all()


def test_flow_endpoints_and_noise():
    for source in ['gp','white']:
        g,d,o=setup(source)
        h=d.context[:8]
        x=d.target[:8]
        mean,cov=o.target_moments(h)
        at0=o.flow(x,0,h,order=128,noise_covariance=True)
        at1=o.flow(x,1,h,order=128,noise_covariance=True)
        np.testing.assert_allclose(at0['velocity'],mean-x,atol=3e-8)
        np.testing.assert_allclose(at1['velocity'],x,atol=3e-8)
        np.testing.assert_allclose(at0['jacobian'],np.broadcast_to(-np.eye(g.horizon),(8,g.horizon,g.horizon)),atol=3e-8)
        np.testing.assert_allclose(at1['jacobian'],np.broadcast_to(np.eye(g.horizon),(8,g.horizon,g.horizon)),atol=3e-8)
        np.testing.assert_allclose(at0['noise_covariance'],cov,rtol=1e-7,atol=3e-8)
        np.testing.assert_allclose(at1['noise_covariance'],np.broadcast_to(o.C,cov.shape),atol=3e-8)


def test_quadrature_converges_and_jacobian_finite_differences():
    for source in ['gp','white']:
        g,d,o=setup(source)
        h,y=d.context[:6],d.target[:6]
        eps=np.random.default_rng(55).normal(size=y.shape)@np.linalg.cholesky(o.C).T
        t=np.array([.1,.3,.5,.7,.9,.97])
        x=(1-t[:,None])*eps+t[:,None]*y
        a=o.flow(x,t,h,order=64)
        b=o.flow(x,t,h,order=128)
        np.testing.assert_allclose(a['velocity'],b['velocity'],rtol=3e-5,atol=3e-6)
        np.testing.assert_allclose(a['jacobian'],b['jacobian'],rtol=2e-4,atol=2e-4)
        numerical=[]
        for i in range(g.horizon):
            dx=np.zeros_like(x); dx[:,i]=1e-5
            plus=o.flow(x+dx,t,h,order=128,jacobian=False)['velocity']
            minus=o.flow(x-dx,t,h,order=128,jacobian=False)['velocity']
            numerical.append((plus-minus)/(2e-5))
        numerical=np.stack(numerical,axis=2)
        np.testing.assert_allclose(b['jacobian'],numerical,rtol=3e-4,atol=3e-4)


def test_target_sampler_moments():
    _,d,o=setup()
    mean,cov=o.target_moments(d.context[:2])
    y=o.sample_target(d.context[:2],60000,82424)
    np.testing.assert_allclose(y.mean(1),mean,atol=.025)
    for i in range(2):
        np.testing.assert_allclose(np.cov(y[i],rowvar=False),cov[i],atol=.003,rtol=.06)


def test_truncated_scale_quadrature_branch_and_moments():
    # An artificially large threshold exercises the otherwise negligible tail.
    L,F=8,3
    C=np.eye(F)
    o=NormalizedHistoryOracle(np.eye(L+F),L,C,threshold=.6)
    h=np.arange(L,dtype=float)[None];h-=h.mean(1,keepdims=True);h/=h.std(1,keepdims=True)
    mean,cov=o.target_moments(h)
    x=np.zeros((1,F))
    got=o.flow(x,0,h,order=256,noise_covariance=True)
    assert got['truncated_quantile_rule'].all()
    assert got['threshold_tail_probability'][0]>.01
    np.testing.assert_allclose(got['noise_covariance'],cov,rtol=2e-5,atol=1e-5)
