import csv
from datetime import datetime,timedelta
import json

import numpy as np
import pytest

from fm_stress.extended_data import (build_data,sample_source,make_windows,
    estimate_channel_correlation,equicorrelation,chronological_bounds)


def synthetic(**kwargs):
    cfg=dict(kind='synthetic',lookback=6,horizon=4,channels=3,
             channel_correlation=.6,lengthscale=3.,seed=97041,
             n_train=3000,n_val=100,n_test=100,source_lengthscale=3.)
    cfg.update(kwargs)
    return cfg


def write_csv(path,x):
    with path.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['date','a','b'])
        for i,row in enumerate(x):
            w.writerow([(datetime(2020,1,1)+timedelta(hours=i)).isoformat(' '),*row])


def test_synthetic_separable_gp_shape_correlation_and_reconstruction():
    bundle=build_data(synthetic())
    ds=bundle.splits['train']
    assert ds.context.shape==(3000,6,3)
    assert ds.target.shape==(3000,4,3)
    full=np.concatenate([ds.context,ds.target],axis=1)
    np.testing.assert_allclose(full.mean(axis=(0,1)),0,atol=1e-13)
    np.testing.assert_allclose(full.std(axis=(0,1)),1,atol=1e-13)
    raw=full*np.asarray(bundle.metadata['scaler_scale'])+np.asarray(bundle.metadata['scaler_mean'])
    # Across independent windows, cross-time/cross-channel covariance factors.
    cross=np.cov(raw[:,0,0],raw[:,2,1])[0,1]
    assert abs(cross-.6*np.exp(-.5*(2/3)**2))<.065
    channel=np.corrcoef(raw[:,0,:],rowvar=False)
    np.testing.assert_allclose(channel,equicorrelation(3,.6),atol=.065)
    other=build_data(synthetic())
    np.testing.assert_array_equal(ds.context,other.splits['train'].context)
    np.testing.assert_array_equal(ds.target,other.splits['train'].target)
    json.dumps(bundle.metadata,allow_nan=False)


def test_synthetic_splits_and_scaler_do_not_depend_on_validation_size():
    a=build_data(synthetic(n_train=40,n_val=30,n_test=30))
    b=build_data(synthetic(n_train=40,n_val=50,n_test=50))
    np.testing.assert_array_equal(a.splits['train'].target,b.splits['train'].target)
    np.testing.assert_array_equal(a.source_covariances['gp'],b.source_covariances['gp'])
    assert a.metadata['split_seeds']=={'train':97041,'val':97042,'test':97043}
    assert not np.array_equal(a.splits['val'].context,a.splits['test'].context)
    assert a.metadata['scaler_mean']==b.metadata['scaler_mean']


def test_temporally_white_and_gp_share_channel_covariance_and_unit_scale():
    bundle=build_data(synthetic(n_train=500,n_val=20,n_test=20))
    c=np.asarray(bundle.metadata['source_channel_correlation'])
    white,gp=bundle.source_covariances['white'],bundle.source_covariances['gp']
    np.testing.assert_allclose(white[:3,:3],c)
    np.testing.assert_allclose(gp[:3,:3],c)
    np.testing.assert_allclose(white[:3,3:6],0)
    np.testing.assert_allclose(gp[:3,3:6],np.exp(-.5/9)/(1+1e-6)*c)
    np.testing.assert_allclose(np.diag(white),1)
    np.testing.assert_allclose(np.diag(gp),1)
    assert np.linalg.eigvalsh(gp).min()>0
    x=sample_source(bundle,30000,'white',123)
    assert x.shape==(30000,4,3)
    np.testing.assert_allclose(np.cov(x[:,0,:],rowvar=False),c,atol=.025)
    assert abs(np.corrcoef(x[:,0,0],x[:,1,0])[0,1])<.02
    y=sample_source(bundle,50,'gp',124)
    np.testing.assert_array_equal(y,sample_source(bundle,50,'gp',124))
    assert not np.array_equal(y,sample_source(bundle,50,'gp',125))


def test_univariate_and_constant_channel_source_safe():
    bundle=build_data(synthetic(channels=1,n_train=20,n_val=10,n_test=10))
    assert bundle.splits['train'].target.shape==(20,4,1)
    np.testing.assert_allclose(bundle.source_covariances['white'],np.eye(4))
    corr=estimate_channel_correlation(np.ones((3,4,3)))
    np.testing.assert_allclose(corr,np.eye(3))


def test_real_splits_confine_all_windows_and_train_only_standardization(tmp_path):
    x=np.stack([np.arange(120,dtype=float),np.sin(np.arange(120)/5)],axis=1)
    path=tmp_path/'data.csv';write_csv(path,x)
    cfg=dict(kind='real',path=str(path),format='csv',columns=['a','b'],lookback=8,horizon=4,
        split_ends=[60,90,110],strides={'train':3,'val':4,'test':4},max_windows={'train':7})
    bundle=build_data(cfg)
    np.testing.assert_allclose(bundle.metadata['scaler_mean'],x[:60].mean(axis=0))
    np.testing.assert_allclose(bundle.metadata['scaler_scale'],x[:60].std(axis=0))
    for split,(start,end) in {'train':(0,60),'val':(60,90),'test':(90,110)}.items():
        ds=bundle.splits[split]
        assert np.all(ds.starts>=start)
        assert np.all(ds.starts+8+4<=end)
        mean=np.array(bundle.metadata['scaler_mean']);scale=np.array(bundle.metadata['scaler_scale'])
        for j,t in enumerate(ds.starts):
            np.testing.assert_allclose(ds.context[j]*scale+mean,x[t:t+8],atol=1e-14)
            np.testing.assert_allclose(ds.target[j]*scale+mean,x[t+8:t+12],atol=1e-14)
    assert len(bundle.splits['train'])==7
    assert bundle.metadata['omitted_tail_rows']==10
    # Changing every held-out value leaves scaler and source fit unchanged.
    x[60:]+=1e6;write_csv(path,x)
    changed=build_data(cfg)
    assert bundle.metadata['scaler_mean']==changed.metadata['scaler_mean']
    assert bundle.metadata['scaler_scale']==changed.metadata['scaler_scale']
    np.testing.assert_array_equal(bundle.source_covariances['gp'],changed.source_covariances['gp'])
    np.testing.assert_array_equal(bundle.splits['train'].context,changed.splits['train'].context)
    assert bundle.metadata['file']['file_sha256']!=changed.metadata['file']['file_sha256']


def test_exchange_numeric_text_and_fraction_boundaries(tmp_path):
    x=np.random.default_rng(1).normal(size=(100,8))
    path=tmp_path/'exchange_rate.txt';np.savetxt(path,x,delimiter=',')
    bundle=build_data(dict(kind='real',path=str(path),format='text',lookback=3,horizon=2,
        split_fractions=[.7,.1,.2],strides={'train':1,'val':2,'test':2}))
    assert bundle.splits['train'].context.shape[-1]==8
    assert bundle.metadata['split_row_bounds']=={'train':[0,70],'val':[70,80],'test':[80,100]}
    assert bundle.source_covariances['gp'].shape==(16,16)
    assert bundle.metadata['file']['timestamp_range'] is None


def test_invalid_covariance_boundaries_windows_and_nonchronological_data(tmp_path):
    with pytest.raises(ValueError):equicorrelation(3,-.6)
    with pytest.raises(ValueError):chronological_bounds(100,{'split_ends':[60,50,100]})
    with pytest.raises(ValueError):make_windows(np.zeros((10,2)),0,5,4,3)
    path=tmp_path/'bad.csv'
    path.write_text('date,a,b\n2020-01-02,1,2\n2020-01-01,2,3\n')
    with pytest.raises(ValueError,match='strictly increasing'):
        build_data(dict(kind='real',path=str(path),lookback=1,horizon=2))
