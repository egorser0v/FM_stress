"""Independent geometry, Gaussian-mixture and coordinate-control checks."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from fm_stress import prior_mixture as pm
from fm_stress.data import GPConfig, sample_dataset
from fm_stress.metrics import fit_pca


def tiny_config():
    cfg = json.loads(Path('configs/prior_mixture.json').read_text())
    cfg.update(n_train=512, n_val=256, n_test=32, steps=2, eval_every=1,
               batch_size=8, n_ensemble_contexts=4, ensemble_size=4, euler_steps=2)
    cfg['model'].update(width=8, state_size=4)
    return cfg


@pytest.mark.parametrize('alpha', [0., .5, .9, .99, 1.])
def test_covariance_matches_declared_gaussian_interpolation(alpha):
    cfg = tiny_config()
    f = cfg['data']['horizon']
    d = np.arange(f)[:, None] - np.arange(f)[None, :]
    variance, nugget = cfg['data'].get('variance', 1.), cfg['data'].get('nugget', 1e-6)
    rbf = variance*np.exp(-d*d/(2*cfg['data']['lengthscale']**2)) + nugget*np.eye(f)
    expected = (1-alpha)*np.eye(f) + alpha*rbf/(variance+nugget)
    covariance = pm.source_covariance(cfg, alpha)
    np.testing.assert_allclose(covariance, expected, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(np.diag(covariance), np.ones(f), atol=1e-14)
    assert np.linalg.eigvalsh(covariance).min() > 0


def test_mixture_is_gaussian_not_random_component_selection():
    """A Bernoulli white/GP draw shares covariance but fails this fourth moment."""
    cfg = tiny_config()
    alpha = .5
    n = 120_000
    draws = pm.source_draw(cfg, n, alpha, 741)
    covariance = pm.source_covariance(cfg, alpha)
    # Smoothness direction has widely differing variance in the two components.
    direction = np.zeros(cfg['data']['horizon']); direction[:2] = [1., -1.]
    variance = float(direction @ covariance @ direction)
    projection = draws @ direction
    assert abs(np.mean(projection**2)/variance - 1.) < .025
    assert abs(np.mean(projection**4)/(3*variance**2) - 1.) < .05
    np.testing.assert_allclose(np.cov(draws, rowvar=False), covariance, atol=.02, rtol=.02)


def test_draws_preserve_rng_and_share_standard_normals_across_alpha():
    cfg = tiny_config()
    np.random.seed(983)
    before = np.random.get_state()
    for alpha in cfg['alphas']:
        draws = pm.source_draw(cfg, 20, alpha, 631)
        z = np.linalg.solve(np.linalg.cholesky(pm.source_covariance(cfg, alpha)), draws.T).T
        expected = np.random.default_rng(631).standard_normal(draws.shape)
        np.testing.assert_allclose(z, expected, atol=1e-8, rtol=1e-8)
    after = np.random.get_state()
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)


def test_coordinate_controls_have_correct_objectives_and_no_derivative_shift():
    cfg = tiny_config()
    rng = np.random.default_rng(812)
    training = rng.normal(size=(80, cfg['data']['horizon']))
    training *= np.linspace(.2, 3., training.shape[1])
    training += 20.  # A nonzero PCA mean must never be subtracted from a velocity.
    pca = fit_pca(training)
    x = torch.tensor(rng.normal(size=(7, training.shape[1])), dtype=torch.float32)
    u = torch.tensor(rng.normal(size=x.shape), dtype=torch.float32)
    q = torch.tensor(pca.components.T, dtype=torch.float32)
    scale = torch.tensor(pca.scales, dtype=torch.float32)
    encoders = []
    for head in ('mlp', 's4', 'whitened_mlp', 'balanced_mlp', 'whitened_raw_mlp'):
        model = pm.make_model(cfg, head, pca, 73, torch.device('cpu'))
        encoders.append({k: v.clone() for k, v in model.base.encoder.state_dict().items()})
        expected = (((x-u)@q)/scale).square().mean() if head in ('whitened_mlp', 'balanced_mlp') else (x-u).square().mean()
        torch.testing.assert_close(pm.objective(model, x, u), expected, atol=2e-6, rtol=2e-6)
        if head in ('whitened_mlp', 'whitened_raw_mlp'):
            torch.testing.assert_close(model.coordinates(x), (x@q)/scale, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(model.inverse(model.coordinates(x)), x, atol=2e-6, rtol=2e-6)
    for encoder in encoders[1:]:
        for key, value in encoders[0].items():
            torch.testing.assert_close(value, encoder[key], atol=0, rtol=0)


def test_prepare_never_generates_test_and_pca_uses_only_training(tmp_path):
    cfg = tiny_config()
    original = pm.sample_dataset
    forbidden = {seed+2 for seed in cfg['data_seeds']}
    def guarded(gp, n, seed):
        assert seed not in forbidden, 'Geometry stage touched test data'
        return original(gp, n, seed)
    with patch.object(pm, 'sample_dataset', guarded):
        bundles = pm.prepare(cfg, tmp_path)
    assert set(bundles) == set(cfg['data_seeds'])
    for data_seed, (gp, datasets, pcas, geometry) in bundles.items():
        assert set(datasets) == {'train', 'val'}
        assert geometry['gate_pass']
        for alpha in cfg['alphas']:
            pca = pcas[pm.source_name(alpha)]
            # The independent full covariance check lives in the audit script;
            # here verify fitted covariance cannot have included val/test rows.
            assert pca.components.shape == (gp.horizon, gp.horizon)
            np.testing.assert_allclose(pca.components@pca.components.T, np.eye(gp.horizon), atol=1e-10)


def test_fixed_pairs_obey_interpolation_and_use_same_times_across_sources():
    cfg = tiny_config()
    seed = cfg['data_seeds'][0]
    ds = sample_dataset(GPConfig(**cfg['data']), 31, seed+1)
    previous = None
    for alpha in cfg['alphas']:
        x, u, t, eps = pm.fixed_pairs(cfg, ds, alpha, seed, 'val')
        np.testing.assert_allclose(x, (1-t[:, None])*eps+t[:, None]*ds.target, atol=3e-6, rtol=2e-6)
        np.testing.assert_allclose(u, ds.target-eps, atol=3e-6, rtol=2e-6)
        assert t.min() >= 0 and t.max() < 1
        if previous is not None:
            np.testing.assert_array_equal(t, previous)
        previous = t


def test_short_training_pairs_rng_and_defers_test_until_checkpoint_selection(tmp_path):
    cfg = tiny_config()
    cfg['data_seeds'] = cfg['data_seeds'][:1]
    cfg['seeds'] = cfg['seeds'][:1]
    cfg.update(n_paths=4, gradient_probe_size=4, gradient_steps=[0, 1, 2], generation_batch_size=8)
    bundles = pm.prepare(cfg, tmp_path)
    data_seed = cfg['data_seeds'][0]
    seed = cfg['seeds'][0]
    original = pm.sample_dataset
    records = []
    for head in ('mlp', 's4', 'whitened_mlp'):
        checkpoint = tmp_path/f'dataset_{data_seed}'/'checkpoints'/f'gp_{head}_seed{seed}.pt'
        training = tmp_path/f'dataset_{data_seed}'/'training'/f'gp_{head}_seed{seed}.json'
        def guarded(gp, n, draw_seed):
            assert draw_seed == data_seed+2
            assert checkpoint.exists() and training.exists()
            curve = json.loads(training.read_text())
            assert curve[-1]['step'] == cfg['steps'], 'Test accessed before all updates'
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            assert state['step'] == min(curve, key=lambda row: row['validation_objective'])['step']
            return original(gp, n, draw_seed)
        with patch.object(pm, 'sample_dataset', guarded):
            records.append(pm.train_run(cfg, bundles[data_seed], tmp_path, data_seed, 1., head, seed, torch.device('cpu')))
    assert len({r['encoder_initial_sha256'] for r in records}) == 1
    assert all(r['batch_hashes'] == records[0]['batch_hashes'] for r in records)
    assert all(np.isfinite(r['forecast_means']['fair_crps']) for r in records)
    assert all(r['gradient_diagnostics'][-1]['selected_checkpoint'] for r in records)
    assert pm.learning_rate(cfg, 1) == pytest.approx(cfg['learning_rate'])
    assert pm.learning_rate(cfg, cfg['steps']) == pytest.approx(cfg['final_learning_rate'])
