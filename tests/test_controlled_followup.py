"""Independent loss algebra and pilot/final isolation for prospective follow-up."""
import copy
import json
from unittest.mock import patch
import numpy as np
import pytest
import torch
from fm_stress import controlled_followup as cf


def tiny():
    cfg=json.load(open('configs/controlled_followup.json'))
    cfg.update(n_train=64,n_val=32,n_test=32,n_paths=4,pilot_steps=2,steps=2,eval_every=1,batch_size=8,euler_steps=2)
    cfg['model'].update(width=8,state_size=4)
    return cfg


def test_loss_and_gradient_match_independent_quadratic_form():
    rng=np.random.default_rng(10);q,_=np.linalg.qr(rng.normal(size=(4,4)));sc=np.array([.2,1,3,7])
    a=rng.normal(size=(3,4));e=torch.tensor(a,requires_grad=True)
    got=cf.objective(e,'balanced',torch.tensor(q),torch.tensor(sc))
    metric=q@np.diag(1/sc**2)@q.T
    assert np.isclose(got.item(),sum(row@metric@row for row in a)/a.size)
    got.backward();np.testing.assert_allclose(e.grad.numpy(),2*a@metric/a.size,atol=1e-12)
    assert np.isclose(cf.objective(e,'raw',torch.tensor(q),torch.tensor(sc)).item(),np.mean(a*a))


def test_pilots_never_access_test_and_final_requires_all_pilots(tmp_path):
    cfg=tiny();original=cf.sample_dataset
    def guarded(gp,n,seed):
        assert seed != cfg['data_seed']+2, 'Pilot accessed test dataset'
        return original(gp,n,seed)
    with patch.object(cf,'sample_dataset',guarded):
        gp,data,pca=cf.prepare(cfg,tmp_path)
        assert set(data)=={'train','val'}
        for head in cf.HEADS:
            model=cf.make_model(cfg,head,20,torch.device('cpu'))
            x=torch.randn(3,32)
            assert isinstance(model.encoder,torch.nn.Identity)
            torch.testing.assert_close(model.encoder(x),x,atol=0,rtol=0)
        with pytest.raises(ValueError,match='All validation-only pilots'):
            cf.train_run(cfg,gp,data,pca,tmp_path,'mlp','raw',10,.001,2,'final',torch.device('cpu'))
        first_batches=[]
        original_factory=cf.make_model
        def capture_model(*args):
            model=original_factory(*args)
            seen=[]
            def capture(module, inputs):
                if module.training and not seen:
                    first_batches.append(tuple(a.detach().clone() for a in inputs))
                    seen.append(True)
            model.register_forward_pre_hook(capture)
            return model
        with patch.object(cf,'make_model',capture_model):
            for head in cf.HEADS:
                for loss in cf.LOSSES:
                    for lr in cfg['pilot_learning_rates']:
                        r=cf.train_run(cfg,gp,data,pca,tmp_path,head,loss,20,lr,2,'pilot',torch.device('cpu'))
                        assert 'metrics' not in r
        assert len(first_batches)==8
        for batch in first_batches[1:]:
            for a,b in zip(first_batches[0],batch):
                torch.testing.assert_close(a,b,atol=0,rtol=0)
        selection=cf.choose_rates(cfg,tmp_path)
        assert selection['all_pilots_complete']
        assert not (tmp_path/'pilots'/'samples').exists()
    # Small real final verifies restore, raw temporal output, metrics and samples.
    lr=selection['selected']['s4_balanced']['learning_rate']
    r=cf.train_run(cfg,gp,data,pca,tmp_path,'s4','balanced',10,lr,2,'final',torch.device('cpu'))
    assert np.isfinite(r['metrics']['residual_mse'])
    assert (tmp_path/'samples'/(r['run']+'.npz')).exists()
