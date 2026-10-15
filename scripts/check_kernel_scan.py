#!/usr/bin/env python3
"""Independent NumPy audit of the kernel scan (docs/kernel_scan_protocol.md §9).

No fm_stress data, PCA or metric routine is used. Kernels, window sampling,
Sundial normalization, source draws, velocity pairs, PCA, per-PC errors,
roughness, the literal rule, fair CRPS / energy, coverage and the analytic
conditional-GP oracle are re-implemented below from their written definitions.

Checkpoint replay necessarily reuses the model architecture (fm_stress.models and
the CoordinateModel wrapper). It verifies that saved predictions come from the
saved checkpoints; it does not re-run training.

The runs were produced on a different machine (CUDA, Linux BLAS), so rebuilt
arrays agree numerically rather than bitwise. Split hashes are therefore
reported for information, and agreement is judged with tolerances fixed below.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Fixed before the audit was first run.
RTOL_INVARIANT = 1e-6    # quantities invariant to the PCA rotation inside degenerate subspaces
RTOL_GROUP = 1e-4        # signal/tail group means (boundary between near-floor PCs)
RTOL_FORECAST = 1e-6     # CRPS, energy, coverage, width from saved ensembles
REPLAY_REL_RMS = 1e-4    # CPU replay of a CUDA checkpoint, relative RMS over predictions
REPLAY_N = 1024          # replayed test pairs per run

SPLITS = {'train': 0, 'val': 1, 'test': 2}
PAIRS = {'val': 800, 'test': 900}
PCA_SOURCE, VAL_SOURCE, ENSEMBLE, ORACLE_PATHS, ORACLE_ENSEMBLE = 100, 110, 1000, 1100, 1200


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# ------------------------------------------------------------ data model

def covariance(kernel, n, lengthscale, variance, nugget):
    d = np.abs(np.subtract.outer(np.arange(n, dtype=float), np.arange(n, dtype=float))) / lengthscale
    if kernel == 'rbf':
        k = np.exp(-0.5 * d * d)
    elif kernel == 'matern52':
        k = (1 + math.sqrt(5) * d + 5 * d * d / 3) * np.exp(-math.sqrt(5) * d)
    elif kernel == 'matern32':
        k = (1 + math.sqrt(3) * d) * np.exp(-math.sqrt(3) * d)
    elif kernel == 'ou':
        k = np.exp(-d)
    else:
        raise ValueError(kernel)
    return variance * k + nugget * np.eye(n)


class Windows:
    def __init__(self, raw, lookback, threshold):
        self.raw_context, self.raw_target = raw[:, :lookback], raw[:, lookback:]
        self.location = self.raw_context.mean(axis=1, keepdims=True)
        scale = self.raw_context.std(axis=1, keepdims=True)
        self.scale = np.where(scale > threshold, scale, 1.0)
        self.context = (self.raw_context - self.location) / self.scale
        self.target = (self.raw_target - self.location) / self.scale


def windows(spec, n, seed):
    k = covariance(spec['kernel'], spec['lookback'] + spec['horizon'], spec['lengthscale'],
                   spec['variance'], spec['nugget'])
    raw = np.random.default_rng(seed).standard_normal((n, len(k))) @ np.linalg.cholesky(k).T
    return Windows(raw, spec['lookback'], spec['threshold'])


def source_cov(spec, source):
    if source == 'white':
        return np.eye(spec['horizon'])
    return covariance(spec['kernel'], spec['horizon'], spec['lengthscale'], spec['variance'], spec['nugget'])


def source_draw(spec, n, source, seed):
    return np.random.default_rng(seed).standard_normal((n, spec['horizon'])) @ np.linalg.cholesky(source_cov(spec, source)).T


def pairs(spec, target, source, data_seed, split):
    eps = source_draw(spec, len(target), source, data_seed + PAIRS[split])
    t = np.random.default_rng(data_seed + PAIRS[split] + 1).uniform(0, 1, len(target))
    return (1 - t[:, None]) * eps + t[:, None] * target, target - eps, t, eps


def principal_axes(x):
    """Rows are axes, sorted by decreasing training variance; unbiased covariance."""
    c = x - x.mean(axis=0)
    values, vectors = np.linalg.eigh(c.T @ c / (len(x) - 1))
    order = np.argsort(values)[::-1]
    return np.maximum(values[order], 0), vectors[:, order].T


def group_means(values, groups):
    return {k: (float(np.mean(values[v])) if v else None) for k, v in groups.items()}


# -------------------------------------------------------------- scores

def fair_crps(x, y):
    """x [H,S,F], y [H,F] -> per-context mean over F of E|X-y| - (1/2)E|X-X'| (off-diagonal)."""
    s = x.shape[1]
    first = np.abs(x - y[:, None]).mean(axis=1)
    pair = np.abs(x[:, :, None, :] - x[:, None, :, :]).sum(axis=(1, 2)) / (2 * s * (s - 1))
    return (first - pair).mean(axis=1)


def fair_energy(x, y):
    s = x.shape[1]
    first = np.linalg.norm(x - y[:, None], axis=2).mean(axis=1)
    pair = np.linalg.norm(x[:, :, None, :] - x[:, None, :, :], axis=3).sum(axis=(1, 2)) / (2 * s * (s - 1))
    return first - pair


def interval(x, y, level):
    lo, hi = np.quantile(x, [(1 - level) / 2, (1 + level) / 2], axis=1)
    return ((y >= lo) & (y <= hi)).mean(axis=1), (hi - lo).mean(axis=1)


def forecast_scores(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = {'fair_crps': fair_crps(x, y), 'fair_energy_score': fair_energy(x, y),
           'mean_forecast_mse': np.square(x.mean(axis=1) - y).mean(axis=1)}
    for level in (.8, .9):
        out[f'coverage_{int(level * 100)}'], out[f'width_{int(level * 100)}'] = interval(x, y, level)
    return out


def roughness(p):
    return float(np.mean(np.abs(np.diff(np.asarray(p, float), axis=-1))))


def literal_rule(pc1, residual, generated, target, tol=.2):
    v = (1 - tol) * pc1 <= residual <= (1 + tol) * pc1 and pc1 > 0
    r = (1 - tol) * target <= generated <= (1 + tol) * target and target > 0
    return bool(v), bool(r), bool(v and r)


# -------------------------------------------------------------- oracle

def oracle(spec, test, x, t, source, chunk=512):
    """Conditional mean velocity and its MSE inputs, from raw-history Gaussian conditioning."""
    l = spec['lookback']
    k = covariance(spec['kernel'], l + spec['horizon'], spec['lengthscale'], spec['variance'], spec['nugget'])
    gain = np.linalg.solve(k[:l, :l], k[:l, l:])
    means = (test.raw_context @ gain - test.location) / test.scale
    residual = k[l:, l:] - k[l:, :l] @ gain
    residual = (residual + residual.T) / 2
    cs = source_cov(spec, source)
    v = np.empty_like(x)
    for a in range(0, len(x), chunk):
        sl = slice(a, a + chunk)
        tt = t[sl, None, None]
        cy = residual[None] / test.scale[sl, :, None] ** 2
        sx = tt ** 2 * cy + (1 - tt) ** 2 * cs
        ux = tt * cy - (1 - tt) * cs
        v[sl] = means[sl] + np.einsum('nij,nj->ni', ux, np.linalg.solve(sx, (x[sl] - t[sl, None] * means[sl])[..., None])[..., 0])
    return v, means, residual


def conditional_draws(means, residual, scale, samples, seed):
    z = np.random.default_rng(seed).standard_normal((len(means), samples, means.shape[1]))
    return means[:, None, :] + (z @ np.linalg.cholesky(residual).T) / scale.reshape(-1, 1, 1)


# --------------------------------------------------------------- checks

class Ledger:
    def __init__(self):
        self.failures, self.max_dev, self.checks = [], {}, 0

    def close(self, label, key, got, want, rtol, atol=1e-12):
        self.checks += 1
        got, want = np.asarray(got, float), np.asarray(want, float)
        dev = float(np.max(np.abs(got - want) / (atol + np.abs(want)))) if got.size else 0.0
        self.max_dev[key] = max(self.max_dev.get(key, 0.0), dev)
        if got.shape != want.shape or not np.allclose(got, want, rtol=rtol, atol=atol):
            self.failures.append(f'{label}: {key} deviates (max relative {dev:.3g})')

    def true(self, label, condition, message):
        self.checks += 1
        if not condition:
            self.failures.append(f'{label}: {message}')


def replay(record, checkpoint, x, t, h):
    import torch
    from fm_stress.models import create_model
    from fm_stress.experiment import CoordinateModel
    g = record['geometry']
    cfg = read(ROOT / 'configs/kernel_scan.json')
    base = create_model('s4' if record['head'] == 's4' else 'mlp', horizon=g['horizon'], lookback=g['lookback'],
                        width=record['width'], **cfg['model'])
    model = CoordinateModel(base)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['state_dict'])
    model.eval()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with torch.no_grad():
        f = lambda a: torch.as_tensor(a, dtype=torch.float32)
        out = model(f(x), f(t), f(h)).numpy()
    return out, params, state


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', default='results/kernel-scan')
    p.add_argument('--config', default='configs/kernel_scan.json')
    p.add_argument('--no-replay', action='store_true')
    a = p.parse_args()
    out, cfg = ROOT / a.results, read(ROOT / a.config)
    led = Ledger()

    # Provenance of the whole directory.
    manifest = read(out / 'manifest.json')
    led.true('root', read(out / 'config.json') == cfg, 'saved config differs from configs/kernel_scan.json')
    led.true('root', manifest['config_sha256'] == stable_hash(cfg), 'manifest config hash mismatch')
    led.true('root', manifest['fingerprint'] == stable_hash({k: manifest[k] for k in ('config_sha256', 'source_hashes')}),
             'manifest fingerprint mismatch')
    for name, digest in manifest['source_hashes'].items():
        led.true('root', sha(ROOT / name) == digest, f'{name} changed since the scan')
    devices = sorted(p.name for p in out.glob('environment_*.json'))
    led.true('root', devices == ['environment_cuda.json'], f'unexpected device records {devices}')
    fingerprint = manifest['fingerprint']

    specs = {g['id']: g for g in cfg['geometries']}
    expected = {(k, g, r['data_seed'], s, h) for r in cfg['replicates'] for g in specs
                for k in cfg['kernels'] for s in cfg['sources'] for h in cfg['heads']}
    observed = {(f.parts[-5], f.parts[-4], int(f.parts[-3][4:]), *f.stem.split('_', 1))
                for f in out.glob('*/*/draw*/runs/*.json')}
    led.true('root', observed == expected, f'missing {len(expected - observed)}, extra {len(observed - expected)} runs')

    bitwise = {'split': 0, 'split_total': 0, 'test': 0, 'test_total': 0}
    replays, last_step, verified = [], 0, 0
    for rep_index, rep in enumerate(cfg['replicates']):
        d = rep['data_seed']
        for gid, g in specs.items():
            for kernel in cfg['kernels']:
                spec = {'kernel': kernel, 'lookback': g['lookback'], 'horizon': g['horizon'],
                        'lengthscale': float(g['lengthscale']), 'variance': cfg['variance'],
                        'nugget': cfg['nugget'], 'threshold': cfg['normalization_threshold']}
                F = spec['horizon']
                folder = out / kernel / gid / f'draw{d}'
                label = f'{kernel}/{gid}/draw{d}'
                geo = read(folder / 'geometry.json')
                led.true(label, geo['fingerprint'] == fingerprint and geo['test_inspected'] is False,
                         'geometry fingerprint or test flag')
                train, val = windows(spec, cfg['n_train'], d + SPLITS['train']), windows(spec, cfg['n_val'], d + SPLITS['val'])
                for split, ds in (('train', train), ('val', val)):
                    for key in ('raw_context', 'raw_target'):
                        bitwise['split_total'] += 1
                        bitwise['split'] += array_sha(getattr(ds, key)) == geo['split_hashes'][split][key]
                # Nugget floor, signal count and groups.
                tvals, tbase = principal_axes(train.target)
                led.close(label, 'target_eigenvalues', tvals, geo['target_pca']['eigenvalues'], RTOL_INVARIANT, 1e-10)
                floor = spec['nugget'] * float(np.mean(1 / train.scale[:, 0] ** 2))
                led.close(label, 'target_floor', floor, geo['nugget_floor']['target_floor'], RTOL_INVARIANT)
                m = int(np.sum(tvals >= cfg['signal_floor_multiple'] * floor))
                led.true(label, m == geo['signal_pcs'], f'signal count {m} vs {geo["signal_pcs"]}')
                mm = min(max(m, 2), F)
                groups = {'pc1': [0], 'pc2': [1], 'residual': list(range(2, F)),
                          'residual_signal': list(range(2, mm)), 'residual_tail': list(range(mm, F))}
                led.true(label, {k: list(v) for k, v in geo['pc_groups'].items()} == groups, 'PC groups differ')
                axes = {}
                for source in cfg['sources']:
                    eps = source_draw(spec, cfg['n_train'], source, d + PCA_SOURCE)
                    vals, vecs = principal_axes(train.target - eps)
                    axes[source] = (vals, vecs)
                    led.close(label, 'velocity_eigenvalues', vals, geo['sources'][source]['pca']['eigenvalues'],
                              RTOL_INVARIANT, 1e-10)
                    veps = source_draw(spec, cfg['n_val'], source, d + VAL_SOURCE)
                    pv = ((val.target - veps) @ vecs.T).var(axis=0, ddof=1)
                    led.close(label, 'validation_top2', pv[:2].sum() / pv.sum(),
                              geo['sources'][source]['validation_velocity']['top2_energy'], RTOL_INVARIANT)
                test = windows(spec, cfg['n_test'], d + SPLITS['test'])
                n, nc, ns = cfg['n_paths'], cfg['n_ensemble_contexts'], cfg['ensemble_size']
                for source in cfg['sources']:
                    vals, vecs = axes[source]
                    x, u, t, eps = pairs(spec, test.target, source, d, 'test')
                    # Oracle reference shared by the heads of this (group, source).
                    oref = read(folder / f'oracle_{source}.json')
                    led.true(label, oref['fingerprint'] == fingerprint, 'oracle fingerprint')
                    ov, means, residual = oracle(spec, test, x, t, source)
                    opc = np.mean(((ov - u) @ vecs.T) ** 2, axis=0)
                    og = group_means(opc, groups)
                    for key in ('pc1', 'pc2', 'residual'):
                        led.close(label, 'oracle_' + key, og[key], oref['velocity_groups'][key], RTOL_INVARIANT)
                    oens = conditional_draws(means[:nc], residual, test.scale[:nc], ns, d + ORACLE_ENSEMBLE)
                    ofs = forecast_scores(oens, test.target[:nc])
                    for key in ('fair_crps', 'fair_energy_score'):
                        led.close(label, 'oracle_' + key, ofs[key].mean(), oref['forecast_means'][key], RTOL_FORECAST)
                    for head in cfg['heads']:
                        run = f'{source}_{head}'
                        rl = f'{label}/{run}'
                        rec = read(folder / 'runs' / f'{run}.json')
                        ckpt, smp = folder / 'checkpoints' / f'{run}.pt', folder / 'samples' / f'{run}.npz'
                        led.true(rl, rec['status'] == 'complete' and rec['fingerprint'] == fingerprint
                                 and rec['config_sha256'] == manifest['config_sha256'], 'status/fingerprint')
                        led.true(rl, rec['checkpoint_sha256'] == sha(ckpt) and rec['samples_sha256'] == sha(smp),
                                 'checkpoint or samples file hash')
                        led.true(rl, rec['device'] == 'cuda' and rec['steps'] == cfg['steps'], 'device/steps')
                        led.true(rl, rec['seed'] == rep['seed'] and rec['width'] == int(cfg['widths'][str(F)]['s4' if head == 's4' else 'mlp']),
                                 'seed/width')
                        bitwise['test_total'] += 1
                        bitwise['test'] += array_sha(test.raw_target) == rec['test_input_hashes']['raw_target']
                        # Validation-only checkpoint selection.
                        curve = rec['curve']
                        steps = [c['step'] for c in curve]
                        led.true(rl, steps == list(range(cfg['eval_every'], cfg['steps'] + 1, cfg['eval_every'])),
                                 'evaluation schedule')
                        objective = [c['validation_objective'] for c in curve]
                        first_best = steps[int(np.argmin(objective))]
                        led.true(rl, rec['best_step'] == first_best and math.isclose(rec['best_validation_objective'], min(objective)),
                                 'selected checkpoint is not the first validation minimum')
                        last_step += rec['best_step'] == cfg['steps']
                        z = np.load(smp)
                        pred = z['prediction'].astype(float)
                        led.close(rl, 'path_target', z['path_target'], test.target[:n], 1e-9, 1e-9)
                        led.close(rl, 'path_history', z['path_history'], test.context[:n], 1e-9, 1e-9)
                        led.close(rl, 'ensemble_target', z['ensemble_target'], test.target[:nc], 1e-9, 1e-9)
                        # Velocity metrics in the cell's training-velocity basis.
                        pc = np.mean(((pred - u) @ vecs.T) ** 2, axis=0)
                        met = rec['metrics']
                        led.close(rl, 'velocity_mse', np.mean((pred - u) ** 2), met['velocity_mse'], RTOL_INVARIANT)
                        gm = group_means(pc, groups)
                        for key in ('pc1', 'pc2', 'residual'):
                            led.close(rl, key + '_mse', gm[key], met[key + '_mse'], RTOL_INVARIANT)
                        for key in ('residual_signal', 'residual_tail'):
                            if gm[key] is not None:
                                led.close(rl, key, gm[key], rec['group_mse'][key], RTOL_GROUP)
                        tpc = np.mean(((pred - u) @ tbase.T) ** 2, axis=0)
                        led.close(rl, 'fixed_target_pc1', tpc[0], rec['fixed_target_basis_metrics']['pc1_mse'], RTOL_INVARIANT)
                        # Paths and the literal rule.
                        gr, tr = roughness(z['generated']), roughness(test.target[:n])
                        led.close(rl, 'generated_roughness', gr, met['generated_roughness'], RTOL_INVARIANT)
                        led.close(rl, 'target_roughness', tr, met['target_roughness'], RTOL_INVARIANT)
                        rule = literal_rule(gm['pc1'], gm['residual'], gr, tr)
                        s = met['success']
                        led.true(rl, rule == (s['velocity_rule_pass'], s['roughness_rule_pass'], s['works']), 'literal rule')
                        # Forecast scores from the saved ensemble.
                        fs = forecast_scores(z['ensemble'], test.target[:nc])
                        for key, values in fs.items():
                            led.close(rl, key, values, rec['per_context'][key], RTOL_FORECAST, 1e-10)
                            led.close(rl, key + '_mean', values.mean(), rec['forecast_means'][key], RTOL_FORECAST, 1e-10)
                        # Oracle gaps quoted by the run.
                        gap = rec['oracle_gap']
                        led.close(rl, 'residual_mse_excess', gap['residual_mse_excess'], met['residual_mse'] - og['residual'],
                                  RTOL_GROUP, 1e-9)
                        led.close(rl, 'fair_crps_ratio', gap['fair_crps_ratio'], fs['fair_crps'].mean() / ofs['fair_crps'].mean(),
                                  RTOL_FORECAST)
                        # CPU replay of the saved checkpoint.
                        if not a.no_replay:
                            got, params, state = replay(rec, ckpt, x[:REPLAY_N], t[:REPLAY_N], test.context[:REPLAY_N])
                            want = pred[:REPLAY_N]
                            rel = float(np.sqrt(np.mean((got - want) ** 2) / np.mean(want ** 2)))
                            replays.append(rel)
                            led.true(rl, rel <= REPLAY_REL_RMS, f'CPU replay relative RMS {rel:.3g}')
                            led.true(rl, params == rec['parameters'], f'parameter count {params} vs {rec["parameters"]}')
                            led.true(rl, state['step'] == rec['best_step'] and state['fingerprint'] == fingerprint,
                                     'checkpoint step/fingerprint')
                        verified += 1
                print(f'{label}: {verified} runs checked, {len(led.failures)} failures so far', flush=True)

    result = {'complete': not led.failures and verified == len(expected), 'fingerprint': fingerprint,
              'verified_runs': verified, 'expected_runs': len(expected), 'checks': led.checks,
              'failures': led.failures[:200], 'n_failures': len(led.failures),
              'tolerances': {'invariant': RTOL_INVARIANT, 'group': RTOL_GROUP, 'forecast': RTOL_FORECAST,
                             'replay_relative_rms': REPLAY_REL_RMS, 'replay_pairs': REPLAY_N},
              'max_relative_deviation': led.max_dev,
              'replay_relative_rms': {'max': max(replays), 'median': float(np.median(replays))} if replays else None,
              'selected_at_last_update': last_step,
              'bitwise_identical_rebuilds': bitwise,
              'note': 'Runs were produced on CUDA/Linux; rebuilt arrays agree numerically, not bitwise.'}
    path = out / 'independent_verification.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('complete', 'verified_runs', 'checks', 'n_failures',
                                             'replay_relative_rms', 'selected_at_last_update')}, indent=1))
    if not result['complete']:
        print('\n'.join(result['failures'][:20]))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
