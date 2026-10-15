"""Kernels, oracle, independence, widths and a CPU smoke run for the kernel scan."""
import copy
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from scipy.special import gamma, kv

from fm_stress import kernel_scan as ks
from fm_stress import data as main_data
from fm_stress.models import create_model

NU = {'matern52': 2.5, 'matern32': 1.5, 'ou': 0.5}


def scan_config():
    return json.loads(Path('configs/kernel_scan.json').read_text())


def tiny_config():
    cfg = scan_config()
    cfg.update(kernels=['ou'], sources=['gp'], replicates=cfg['replicates'][:1],
               n_train=512, n_val=128, n_test=64, n_paths=16, n_ensemble_contexts=4,
               ensemble_size=4, steps=4, eval_every=2, batch_size=16, euler_steps=4,
               generation_batch_size=32)
    cfg['geometries'] = [g for g in cfg['geometries'] if g['id'] in ('F16_l8', 'F16_l2')]
    cfg['solver_check'] = {'replicate_index': 0, 'n_contexts': 2, 'euler_steps': 8}
    return cfg


def independent_matern(nu, lag, lengthscale, variance):
    """General Matern formula through the modified Bessel function K_nu."""
    r = np.abs(np.asarray(lag, dtype=float)) / lengthscale
    out = np.full(r.shape, variance)
    positive = r > 0
    z = math.sqrt(2 * nu) * r[positive]
    out[positive] = variance * 2 ** (1 - nu) / gamma(nu) * z ** nu * kv(nu, z)
    return out


@pytest.mark.parametrize('kernel', ks.KERNELS)
@pytest.mark.parametrize('size,lengthscale', [(48, 8.0), (96, 8.0), (192, 8.0), (48, 2.0)])
def test_kernel_matrices_are_valid_and_match_closed_forms(kernel, size, lengthscale):
    variance, nugget = 1.3, 1e-6
    k = ks.kernel_matrix(kernel, size, lengthscale, variance, nugget)
    lag = np.subtract.outer(np.arange(size), np.arange(size)).astype(float)
    if kernel == 'rbf':
        expected = variance * np.exp(-lag ** 2 / (2 * lengthscale ** 2))
    else:
        expected = independent_matern(NU[kernel], lag, lengthscale, variance)
    np.testing.assert_allclose(k, expected + nugget * np.eye(size), rtol=1e-12, atol=1e-13)
    np.testing.assert_array_equal(k, k.T)
    np.testing.assert_allclose(np.diag(k), variance + nugget, rtol=0, atol=1e-15)
    # Stationary on the unit grid: every diagonal of the Toeplitz matrix is constant.
    for offset in (1, 5, size - 1):
        diagonal = np.diagonal(k, offset)
        np.testing.assert_allclose(diagonal, diagonal[0], rtol=1e-12)
    assert np.linalg.eigvalsh(k).min() > 0
    np.linalg.cholesky(k)


def test_rbf_matrix_is_bit_identical_to_main_and_roughness_order_holds():
    for size, lengthscale in ((48, 8.0), (192, 8.0), (48, 2.0)):
        np.testing.assert_array_equal(ks.kernel_matrix('rbf', size, lengthscale),
                                      main_data.rbf_covariance(size, lengthscale))
    # Same lengthscale, rougher kernel -> lower lag-one correlation.
    lag_one = [ks.kernel_function(k, 1.0, 8.0) for k in ks.KERNELS]
    assert lag_one == sorted(lag_one, reverse=True)
    assert all(ks.kernel_function(k, 0.0, 3.0, 2.0) == 2.0 for k in ks.KERNELS)
    with pytest.raises(ValueError):
        ks.kernel_function('matern72', 1.0, 8.0)


@pytest.mark.parametrize('lookback,horizon', [(32, 16), (64, 32)])
def test_rbf_sampling_reproduces_main_data_module(lookback, horizon):
    geometry = ks.ScanGeometry('rbf', lookback, horizon, 8.0)
    ours = ks.sample_windows(geometry, 300, 4242)
    theirs = main_data.sample_dataset(main_data.GPConfig(lookback=lookback, horizon=horizon), 300, 4242)
    for name in ('context', 'target', 'raw_context', 'raw_target', 'location', 'scale'):
        np.testing.assert_array_equal(getattr(ours, name), getattr(theirs, name))
    for source in ks.SOURCES:
        np.testing.assert_array_equal(ks.source_covariance(geometry, source),
                                      main_data.source_covariance(main_data.GPConfig(lookback=lookback, horizon=horizon), source))


@pytest.mark.parametrize('source', ks.SOURCES)
def test_oracle_reproduces_data_conditional_target_for_rbf(source):
    config = main_data.GPConfig()
    geometry = ks.ScanGeometry('rbf', config.lookback, config.horizon, config.lengthscale)
    dataset = main_data.sample_dataset(config, 200, 31)
    means, residual = ks.conditional_moments(geometry, dataset)
    reference_means, reference_covs = main_data.conditional_target(config, dataset)
    np.testing.assert_allclose(means, reference_means, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(residual[None] / dataset.scale[:, :, None] ** 2, reference_covs,
                               rtol=1e-10, atol=1e-14)
    rng = np.random.default_rng(5)
    time = rng.uniform(0, 1, len(dataset))
    eps = ks.source_draw(geometry, len(dataset), source, 6)
    x = (1 - time[:, None]) * eps + time[:, None] * dataset.target
    source_cov = ks.source_covariance(geometry, source)
    np.testing.assert_allclose(
        ks.oracle_velocity(x, time, means, residual, dataset.scale, source_cov, chunk=64),
        main_data.conditional_oracle_velocity(x, time, reference_means, reference_covs, source_cov),
        rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(
        ks.oracle_noise_covariance(time, residual, dataset.scale, source_cov, chunk=64),
        main_data.conditional_oracle_noise(time, reference_covs, source_cov).mean(axis=0),
        rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        ks.sample_conditional(means, residual, dataset.scale, 3, 77),
        main_data.sample_conditional_target(config, dataset, samples=3, seed=77), rtol=1e-6, atol=1e-6)


def test_ou_conditional_matches_markov_closed_form():
    """OU without nugget is Markov: the future depends on the last raw value only."""
    lookback, horizon, lengthscale = 12, 5, 3.0
    geometry = ks.ScanGeometry('ou', lookback, horizon, lengthscale, nugget=0.0)
    raw = np.random.default_rng(3).standard_normal((7, lookback + horizon)) @ np.linalg.cholesky(geometry.covariance()).T
    dataset = main_data.GPDataset(raw[:, :lookback], raw[:, lookback:], raw[:, :lookback], raw[:, lookback:],
                                  np.zeros((7, 1)), np.ones((7, 1)))
    means, residual = ks.conditional_moments(geometry, dataset)
    steps = np.arange(1, horizon + 1)
    decay = np.exp(-steps / lengthscale)
    np.testing.assert_allclose(means, raw[:, lookback - 1:lookback] * decay, rtol=1e-9, atol=1e-12)
    expected = np.exp(-np.abs(np.subtract.outer(steps, steps)) / lengthscale) - np.outer(decay, decay)
    np.testing.assert_allclose(residual, expected, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize('source', ks.SOURCES)
def test_oracle_velocity_is_the_monte_carlo_regression_for_a_rough_kernel(source):
    """E[y-eps|x_t] and its residual covariance agree with a simulated least-squares fit."""
    geometry = ks.ScanGeometry('matern32', 10, 5, 2.0)
    rng = np.random.default_rng(11)
    mean = rng.normal(size=geometry.horizon)
    residual = ks.kernel_matrix('ou', geometry.horizon, 3.0) * 0.4
    scale = np.array([[1.7]])
    t = 0.37
    n = 400_000
    y = mean + rng.standard_normal((n, geometry.horizon)) @ np.linalg.cholesky(residual / scale[0, 0] ** 2).T
    eps = ks.source_draw(geometry, n, source, 12)
    x, u = (1 - t) * eps + t * y, y - eps
    design = np.column_stack([x, np.ones(n)])
    coef, *_ = np.linalg.lstsq(design, u, rcond=None)
    probe = x[:200]
    oracle = ks.oracle_velocity(probe, np.full(200, t), np.repeat(mean[None], 200, 0), residual,
                                np.repeat(scale, 200, 0), ks.source_covariance(geometry, source))
    np.testing.assert_allclose(np.column_stack([probe, np.ones(200)]) @ coef, oracle, atol=0.02)
    noise = ks.oracle_noise_covariance(np.array([t]), residual, scale, ks.source_covariance(geometry, source))
    np.testing.assert_allclose(np.cov((u - design @ coef).T), noise, atol=0.01)


def test_sources_are_independent_of_targets_and_paired_across_heads():
    geometry = ks.ScanGeometry('matern32', 32, 16, 8.0)
    before = np.random.get_state()
    a, b = ks.sample_windows(geometry, 4000, 1), ks.sample_windows(geometry, 4000, 2)
    pairs_a = ks.fixed_pairs(geometry, a, 'gp', 900, 'test')
    pairs_b = ks.fixed_pairs(geometry, b, 'gp', 900, 'test')
    # Source draws and times come from their own seeds, never from the targets.
    np.testing.assert_array_equal(pairs_a[3], pairs_b[3])
    np.testing.assert_array_equal(pairs_a[2], pairs_b[2])
    white = ks.fixed_pairs(geometry, a, 'white', 900, 'test')
    np.testing.assert_array_equal(white[2], pairs_a[2])
    np.testing.assert_allclose(white[3] @ np.linalg.cholesky(ks.source_covariance(geometry, 'gp')).T,
                               pairs_a[3], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pairs_a[1], a.target - pairs_a[3], rtol=0, atol=0)
    for state_a, state_b in zip(before, np.random.get_state()):
        np.testing.assert_array_equal(state_a, state_b)
    # Empirical cross-correlation between training sources and targets/histories is null.
    n = 20000
    windows = ks.sample_windows(geometry, n, 7)
    eps = ks.source_draw(geometry, n, 'gp', 7 + ks.PCA_SOURCE_OFFSET)
    joint = np.corrcoef(np.column_stack([eps, windows.target, windows.context]).T)[:16, 16:]
    assert np.abs(joint).max() < 5 / math.sqrt(n)
    np.testing.assert_allclose(np.cov(eps.T), ks.source_covariance(geometry, 'gp'), atol=0.04)
    np.testing.assert_allclose(np.cov(ks.source_draw(geometry, n, 'white', 8).T), np.eye(16), atol=0.04)


def test_nugget_floor_defines_residual_groups():
    cfg = scan_config()
    rbf = ks.make_geometry(cfg, 'rbf', 'F16_l8')
    train = ks.sample_windows(rbf, 32768, 307041)
    floor = ks.nugget_floor(rbf, train.scale)
    np.testing.assert_allclose(floor['target_floor'], rbf.nugget * np.mean(train.scale ** -2.0))
    target = ks.fit_pca(train.target)
    signal = ks.signal_count(target.eigenvalues, floor['target_floor'], cfg['signal_floor_multiple'])
    assert signal == 7  # PCs 8..16 are the normalized nugget at RBF l=8, F=16
    groups = ks.pc_groups(16, signal)
    assert groups['residual_signal'] == list(range(2, 7)) and groups['residual_tail'] == list(range(7, 16))
    velocity = ks.fit_pca(train.target - ks.source_draw(rbf, len(train), 'gp', 1))
    tail = velocity.eigenvalues[groups['residual_tail']] / floor['velocity_floor']['gp']
    assert np.all((tail > 0.9) & (tail < 1.2)) and abs(tail.mean() - 1) < 0.05
    ou = ks.make_geometry(cfg, 'ou', 'F16_l8')
    ou_train = ks.sample_windows(ou, 8192, 307041)
    ou_floor = ks.nugget_floor(ou, ou_train.scale)['target_floor']
    assert ks.signal_count(ks.fit_pca(ou_train.target).eigenvalues, ou_floor, 2.0) == 16
    assert ks.pc_groups(16, 16)['residual_tail'] == []
    assert ks.group_means(np.arange(16.), ks.pc_groups(16, 1))['residual_signal'] is None


def test_widths_match_parameters_and_long_horizons_are_supported():
    cfg = scan_config()
    main = json.loads(Path('configs/main.json').read_text())['model']
    assert cfg['widths']['16'] == {'mlp': main['width'], 's4': main['width']}
    assert {w['mlp'] for w in cfg['widths'].values()} == {main['width']}
    for horizon in (16, 32, 64):
        lookback = 2 * horizon
        mlp = ks.parameter_count(cfg, horizon, lookback, 'mlp')
        s4 = ks.parameter_count(cfg, horizon, lookback, 's4')
        assert abs(mlp - s4) / mlp < 0.005, (horizon, mlp, s4)
        torch.manual_seed(0)
        for head in ('mlp', 's4'):
            model = create_model(head, horizon=horizon, lookback=lookback,
                                 width=ks.width_for(cfg, horizon, head), **cfg['model'])
            x, t, h = torch.randn(5, horizon), torch.rand(5), torch.randn(5, lookback)
            out = model(x, t, h)
            assert out.shape == (5, horizon) and torch.isfinite(out).all()
            out.square().sum().backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if head == 's4':
                kernel = model.blocks[0].temporal.kernel(horizon)
                assert kernel.shape == (2, ks.width_for(cfg, horizon, 's4'), horizon)
                assert torch.isfinite(kernel).all()


def test_select_filters_and_rejects_unknown_values():
    cfg = scan_config()
    assert len(ks.select(cfg)) == 4 * 5 * 2 * 3 * 3
    assert {row[3] for row in ks.select(cfg, ratio='2')} == {'F16_l8'}
    assert {row[3] for row in ks.select(cfg, ratio='8')} == {'F64_l8', 'F16_l2'}
    rows = ks.select(cfg, kernel='ou,rbf', head='s4', source='gp', data_seed='407041')
    assert len(rows) == 2 * 5 and all(r[4] in ('ou', 'rbf') and r[6] == 's4' and r[1] == 407041 for r in rows)
    for kwargs in ({'kernel': 'matern12'}, {'ratio': '3'}, {'head': 'unet'}):
        with pytest.raises(ValueError):
            ks.select(cfg, **kwargs)


def test_task1_branch_labels():
    ok = {'mean_within_patch_std': .3, 'roughness': .05}
    assert ks.task1_branch(.97, .84, ok) == 'contrast_exists'
    assert ks.task1_branch(.79, .60, ok) == 'both_full_rank'
    assert ks.task1_branch(.97, .95, ok) == 'both_collapsed'
    assert ks.task1_branch(.97, .84, {'mean_within_patch_std': .01, 'roughness': .05}) == 'collapsed_targets'


def test_manifest_and_geometry_refuse_changes_and_prepare_never_touches_test(tmp_path):
    cfg = tiny_config()
    fingerprint = ks.prepare_root(cfg, tmp_path)
    assert ks.prepare_root(cfg, tmp_path) == fingerprint
    changed = copy.deepcopy(cfg)
    changed['steps'] += 1
    with pytest.raises(ValueError, match='Changed provenance'):
        ks.prepare_root(changed, tmp_path)
    real_hash = ks.file_hash
    with patch.object(ks, 'file_hash', lambda path: 'edited' if str(path).endswith('models.py') else real_hash(path)):
        with pytest.raises(ValueError, match='Changed provenance'):
            ks.prepare_root(cfg, tmp_path)
    data_seed = cfg['replicates'][0]['data_seed']
    original = ks.sample_windows

    def guarded(geometry, n, seed):
        assert seed != data_seed + ks.SPLIT_OFFSETS['test'], 'Geometry stage generated test windows'
        return original(geometry, n, seed)
    with patch.object(ks, 'sample_windows', guarded):
        bundle = ks.prepare_group(cfg, tmp_path, 'ou', 'F16_l2', data_seed, fingerprint)
    assert set(bundle['data']) == {'train', 'val'} and bundle['record']['gate_enforced'] is False
    path = ks.group_dir(tmp_path, 'ou', 'F16_l2', data_seed) / 'geometry.json'
    record = json.loads(path.read_text())
    record['signal_pcs'] += 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='Geometry differs'):
        ks.prepare_group(cfg, tmp_path, 'ou', 'F16_l2', data_seed, fingerprint)


def test_cpu_smoke_run_pairs_heads_defers_test_and_skips_completed(tmp_path):
    cfg = tiny_config()
    fingerprint = ks.prepare_root(cfg, tmp_path)
    data_seed, seed = cfg['replicates'][0]['data_seed'], cfg['replicates'][0]['seed']
    bundle = ks.prepare_group(cfg, tmp_path, 'ou', 'F16_l2', data_seed, fingerprint)
    root = ks.group_dir(tmp_path, 'ou', 'F16_l2', data_seed)
    original = ks.sample_windows
    records = []
    for head in cfg['heads']:
        def guarded(geometry, n, draw_seed, head=head):
            if draw_seed == data_seed + ks.SPLIT_OFFSETS['test']:
                curve = json.loads((root / 'training' / f'gp_{head}.json').read_text())
                assert curve[-1]['step'] == cfg['steps'], 'Test accessed before the last update'
                state = torch.load(root / 'checkpoints' / f'gp_{head}.pt', map_location='cpu', weights_only=False)
                assert state['step'] == min(curve, key=lambda row: row['validation_objective'])['step']
            return original(geometry, n, draw_seed)
        with patch.object(ks, 'sample_windows', guarded):
            records.append(ks.train_run(cfg, bundle, tmp_path, 'ou', 'F16_l2', data_seed, seed, 'gp', head,
                                        torch.device('cpu'), fingerprint, 0))
    assert len({r['encoder_initial_sha256'] for r in records}) == 1
    assert all(r['batch_hashes'] == records[0]['batch_hashes'] for r in records)
    assert all(r['test_input_hashes'] == records[0]['test_input_hashes'] for r in records)
    oracle = json.loads((root / 'oracle_gp.json').read_text())
    for r in records:
        m = r['metrics']
        assert len(m['pc_mse']) == 16 and np.isfinite(r['forecast_means']['fair_crps'])
        assert r['group_mse']['residual'] == pytest.approx(m['residual_mse'])
        assert r['oracle_gap']['fair_crps_excess'] == pytest.approx(
            r['forecast_means']['fair_crps'] - oracle['forecast_means']['fair_crps'])
        assert r['solver_sensitivity']['euler_steps'] == 8
        with np.load(root / 'samples' / f"gp_{r['head']}.npz") as samples:
            assert samples['prediction'].shape == (cfg['n_test'], 16)
            assert samples['ensemble'].shape == (cfg['n_ensemble_contexts'], cfg['ensemble_size'], 16)
    assert records[1]['width'] == 56 and records[1]['parameters'] == ks.parameter_count(cfg, 16, 32, 's4')
    # Oracle velocity error cannot be meaningfully below the irreducible noise.
    assert oracle['velocity']['velocity_mse'] > 0
    # Completed runs are skipped; a tampered checkpoint is refused.
    again = ks.train_run(cfg, bundle, tmp_path, 'ou', 'F16_l2', data_seed, seed, 'gp', 'mlp',
                         torch.device('cpu'), fingerprint, 0)
    assert again == records[0]
    (root / 'checkpoints' / 'gp_mlp.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='incompatible'):
        ks.train_run(cfg, bundle, tmp_path, 'ou', 'F16_l2', data_seed, seed, 'gp', 'mlp',
                     torch.device('cpu'), fingerprint, 0)


def test_mps_request_without_mps_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    cfg_path = tmp_path / 'cfg.json'
    cfg_path.write_text(json.dumps(tiny_config()))
    monkeypatch.setattr('sys.argv', ['kernel_scan', '--config', str(cfg_path), '--output', str(tmp_path / 'out')])
    with pytest.raises(RuntimeError, match='no silent CPU fallback'):
        ks.main()
    assert not (tmp_path / 'out').exists()


def test_cuda_request_without_cuda_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    cfg_path = tmp_path / 'cfg.json'
    cfg_path.write_text(json.dumps(tiny_config()))
    monkeypatch.setattr('sys.argv', ['kernel_scan', '--config', str(cfg_path), '--output', str(tmp_path / 'out'),
                                     '--device', 'cuda'])
    with pytest.raises(RuntimeError, match='no silent CPU fallback'):
        ks.main()
    assert not (tmp_path / 'out').exists()


def test_refuses_mixing_devices_in_one_output(monkeypatch, tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'environment_mps.json').write_text('{}')
    cfg_path = tmp_path / 'cfg.json'
    cfg_path.write_text(json.dumps(tiny_config()))
    monkeypatch.setattr('sys.argv', ['kernel_scan', '--config', str(cfg_path), '--output', str(out), '--device', 'cpu'])
    with pytest.raises(RuntimeError, match='another device'):
        ks.main()
