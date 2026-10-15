"""Pre-declared kernel x window/lengthscale scan of the AdaLN-MLP / S4 head gap.

Supplementary study; protocol in docs/kernel_scan_protocol.md. It reuses the
main study's normalization, heads, training and metrics, and never edits them:
their SHA-256 hashes are frozen into <output>/manifest.json, and a resumed run
refuses any change.

Kernels use the Rasmussen-Williams parametrization on a unit grid, so Matern-nu
tends to the RBF with the same lengthscale as nu -> infinity. As in
configs/main.json the matched GP source is the raw kernel (with nugget) on the
horizon grid. The privileged conditional-GP oracle is re-implemented here
because data.py is RBF-only; tests check it against data.py for the RBF.

Example (sequential MPS, the default device; --device cuda on an NVIDIA host):
  python -m fm_stress.kernel_scan --prepare-only
  python -m fm_stress.kernel_scan --geometry F16_l8,F32_l8,F64_l8
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import hashlib
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .data import GPDataset, sundial_normalize
from .ensemble_evaluation import context_scores
from .experiment import (CoordinateModel, config_hash, generate, load_config,
                         predict, save_json, tensor)
from .metrics import evaluate, fit_pca, spectrum, success_rule, velocity_metrics
from .models import count_parameters, create_model, optimizer_groups

SOURCE_FILES = ('fm_stress/kernel_scan.py', 'fm_stress/models.py', 'fm_stress/data.py',
                'fm_stress/metrics.py', 'fm_stress/experiment.py',
                'fm_stress/ensemble_evaluation.py', 'fm_stress/robustness.py')
KERNELS = ('rbf', 'matern52', 'matern32', 'ou')
SOURCES = ('white', 'gp')
HEADS = ('mlp', 's4', 'whitened_mlp')
# Seed offsets from each replicate's data_seed (main/prior_mixture convention).
SPLIT_OFFSETS = {'train': 0, 'val': 1, 'test': 2}
PAIR_OFFSETS = {'val': 800, 'test': 900}
PCA_SOURCE_OFFSET, VAL_SOURCE_OFFSET = 100, 110
ENSEMBLE_OFFSET, ORACLE_PATH_OFFSET, ORACLE_ENSEMBLE_OFFSET = 1000, 1100, 1200


# ---------------------------------------------------------------- kernels

def kernel_function(kernel, lag, lengthscale, variance=1.0):
    """Stationary kernel at signed lags (closed forms, unit grid spacing)."""
    if lengthscale <= 0 or variance <= 0:
        raise ValueError('Invalid kernel parameters')
    lag = np.asarray(lag, dtype=np.float64)
    if kernel == 'rbf':
        # Same expression as data.rbf_covariance, so RBF matrices are bit-identical.
        return variance * np.exp(-0.5 * (lag / lengthscale) ** 2)
    r = np.abs(lag) / lengthscale
    if kernel == 'matern52':
        s = math.sqrt(5.0) * r
        return variance * (1.0 + s + s * s / 3.0) * np.exp(-s)
    if kernel == 'matern32':
        s = math.sqrt(3.0) * r
        return variance * (1.0 + s) * np.exp(-s)
    if kernel == 'ou':
        return variance * np.exp(-r)
    raise ValueError(f'Unknown kernel: {kernel}')


def kernel_matrix(kernel, size, lengthscale, variance=1.0, nugget=1e-6):
    if size < 1 or nugget < 0:
        raise ValueError('Invalid covariance size or nugget')
    grid = np.arange(size, dtype=np.float64)
    return kernel_function(kernel, grid[:, None] - grid, lengthscale, variance) + nugget * np.eye(size)


@dataclass(frozen=True)
class ScanGeometry:
    kernel: str
    lookback: int
    horizon: int
    lengthscale: float
    variance: float = 1.0
    nugget: float = 1e-6
    normalization_threshold: float = 1e-2

    def __post_init__(self):
        if self.kernel not in KERNELS:
            raise ValueError(f'Unknown kernel: {self.kernel}')
        if self.lookback < 2 or self.horizon < 3:
            raise ValueError('lookback >= 2 and horizon >= 3 are required')
        if self.lengthscale <= 0 or self.variance <= 0 or self.nugget < 0:
            raise ValueError('Invalid kernel parameters')

    @property
    def ratio(self):
        return self.horizon / self.lengthscale

    def covariance(self, size=None):
        return kernel_matrix(self.kernel, self.lookback + self.horizon if size is None else size,
                             self.lengthscale, self.variance, self.nugget)

    def to_dict(self):
        return {**asdict(self), 'ratio': self.ratio}


def geometry_specs(cfg):
    return {g['id']: g for g in cfg['geometries']}


def make_geometry(cfg, kernel, geometry_id):
    g = geometry_specs(cfg)[geometry_id]
    return ScanGeometry(kernel, g['lookback'], g['horizon'], float(g['lengthscale']),
                        cfg['variance'], cfg['nugget'], cfg['normalization_threshold'])


# ------------------------------------------------------- data and sources

def sample_windows(geometry: ScanGeometry, n, seed):
    """Joint lookback+horizon draws, then history-only Sundial normalization."""
    if n <= 0:
        raise ValueError('n must be positive')
    chol = np.linalg.cholesky(geometry.covariance())
    windows = np.random.default_rng(seed).standard_normal((n, geometry.lookback + geometry.horizon)) @ chol.T
    raw_context, raw_target = windows[:, :geometry.lookback], windows[:, geometry.lookback:]
    context, target, location, scale = sundial_normalize(raw_context, raw_target,
                                                         geometry.normalization_threshold)
    return GPDataset(context, target, raw_context, raw_target, location, scale)


def source_covariance(geometry: ScanGeometry, source):
    if source == 'white':
        return np.eye(geometry.horizon)
    if source == 'gp':
        return geometry.covariance(geometry.horizon)
    raise ValueError("source must be 'white' or 'gp'")


def source_draw(geometry: ScanGeometry, n, source, seed):
    """Normalized-coordinate source from its own seed: independent of every target."""
    z = np.random.default_rng(seed).standard_normal((n, geometry.horizon))
    return z @ np.linalg.cholesky(source_covariance(geometry, source)).T


def fixed_pairs(geometry, dataset, source, data_seed, split):
    """Shared times, source normals and windows across heads (white/GP share normals)."""
    offset = PAIR_OFFSETS[split]
    eps = source_draw(geometry, len(dataset), source, data_seed + offset)
    times = np.random.default_rng(data_seed + offset + 1).uniform(0, 1, len(dataset))
    return (1 - times[:, None]) * eps + times[:, None] * dataset.target, dataset.target - eps, times, eps


# --------------------------------------------- privileged analytic oracle

def conditional_moments(geometry: ScanGeometry, dataset):
    """Normalized-future Gaussian given the FULL RAW history (privileged).

    Window n has mean means[n] and covariance residual / scale[n]**2. Networks
    see only the normalized history, so this is an informational floor.
    """
    k, l = geometry.covariance(), geometry.lookback
    a = np.linalg.solve(k[:l, :l], k[:l, l:])
    means = (dataset.raw_context @ a - dataset.location) / dataset.scale
    residual = k[l:, l:] - k[l:, :l] @ a
    return means, (residual + residual.T) / 2


def _blocks(time, residual, scale, source_cov):
    t = np.asarray(time, dtype=np.float64).reshape(-1, 1, 1)
    target_cov = residual[None] / np.asarray(scale, dtype=np.float64).reshape(-1, 1, 1) ** 2
    sx = t ** 2 * target_cov + (1 - t) ** 2 * source_cov
    ux = t * target_cov - (1 - t) * source_cov
    return target_cov, sx, ux


def oracle_velocity(x, time, means, residual, scale, source_cov, chunk=1024):
    """E[y - eps | x_t, raw history] for independent Gaussian source and target."""
    x, time = np.asarray(x, dtype=np.float64), np.asarray(time, dtype=np.float64).reshape(-1)
    if np.any((time < 0) | (time > 1)):
        raise ValueError('time must be in [0,1]')
    out = np.empty_like(x)
    for start in range(0, len(x), chunk):
        part = slice(start, start + chunk)
        _, sx, ux = _blocks(time[part], residual, scale[part], source_cov)
        centered = x[part] - time[part, None] * means[part]
        solved = np.linalg.solve(sx, centered[..., None])[..., 0]
        out[part] = means[part] + np.einsum('nij,nj->ni', ux, solved)
    return out


def oracle_noise_covariance(time, residual, scale, source_cov, chunk=1024):
    """Mean over pairs of Cov[y - eps | x_t, raw history] (irreducible velocity noise)."""
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    total = np.zeros_like(source_cov, dtype=np.float64)
    for start in range(0, len(time), chunk):
        part = slice(start, start + chunk)
        target_cov, sx, ux = _blocks(time[part], residual, scale[part], source_cov)
        cov = target_cov + source_cov - ux @ np.linalg.solve(sx, ux.swapaxes(-1, -2))
        total += ((cov + cov.swapaxes(-1, -2)) / 2).sum(axis=0)
    return total / len(time)


def sample_conditional(means, residual, scale, samples, seed):
    """Exact normalized future draws [N,S,F] given raw history."""
    z = np.random.default_rng(seed).standard_normal((len(means), samples, means.shape[1]))
    chol = np.linalg.cholesky(residual)
    return means[:, None, :] + np.einsum('nsj,ij->nsi', z, chol) / np.asarray(scale).reshape(-1, 1, 1)


# ---------------------------------------- nugget floor and residual groups

def nugget_floor(geometry: ScanGeometry, scale):
    """Isotropic eigenvalue floors implied by the nugget after normalization.

    The raw nugget is white and independent of the history statistics, so it
    adds nugget*E[1/scale^2] to every normalized-target eigenvalue; the source
    adds 1 (white) or its own nugget (GP).
    """
    inverse = float(np.mean(1.0 / np.square(np.asarray(scale, dtype=np.float64))))
    target = geometry.nugget * inverse
    return {'mean_inverse_scale_squared': inverse, 'target_floor': target,
            'velocity_floor': {'white': 1.0 + target, 'gp': target + geometry.nugget}}


def signal_count(target_eigenvalues, target_floor, multiple):
    """Leading training-target PCs whose variance is >= multiple x the nugget floor."""
    return int(np.sum(np.asarray(target_eigenvalues) >= multiple * target_floor))


def pc_groups(horizon, signal):
    """Velocity-PC index groups. PCs 1 and 2 always stand alone."""
    m = min(max(int(signal), 2), horizon)
    return {'pc1': [0], 'pc2': [1], 'residual': list(range(2, horizon)),
            'residual_signal': list(range(2, m)), 'residual_tail': list(range(m, horizon))}


def group_means(values, groups):
    values = np.asarray(values, dtype=np.float64)
    return {name: (float(values[index].mean()) if index else None) for name, index in groups.items()}


def energy_count(eigenvalues, fraction):
    energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    return int(np.searchsorted(energy, fraction) + 1)


def task1_branch(gp_top2, white_top2, target_spectrum):
    """The assignment's Task-1 reading of a cell (reported; never a gate here)."""
    if target_spectrum['mean_within_patch_std'] <= .02 or target_spectrum['roughness'] <= .002:
        return 'collapsed_targets'
    if gp_top2 > .9 and white_top2 < .9:
        return 'contrast_exists'
    if gp_top2 <= .9 and white_top2 <= .9:
        return 'both_full_rank'
    if gp_top2 > .9 and white_top2 >= .9:
        return 'both_collapsed'
    return 'reversed_contrast'


# --------------------------------------------------- provenance helpers

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def manifest(cfg):
    root = Path(__file__).resolve().parents[1]
    m = {'config_sha256': config_hash(cfg),
         'source_hashes': {name: file_hash(root / name) for name in SOURCE_FILES}}
    m['fingerprint'] = config_hash(m)
    return m


def prepare_root(cfg, out):
    """Freeze config + numerical source hashes; refuse any change on resume."""
    out = Path(out)
    m = manifest(cfg)
    for name, record in (('config.json', cfg), ('manifest.json', m)):
        path = out / name
        if path.exists() and load_config(path) != record:
            raise ValueError('Changed provenance: use fresh output directory (' + str(path) + ')')
        save_json(path, record)
    return m['fingerprint']


def group_dir(out, kernel, geometry_id, data_seed):
    return Path(out) / kernel / geometry_id / f'draw{data_seed}'


def width_for(cfg, horizon, head):
    return int(cfg['widths'][str(horizon)]['s4' if head == 's4' else 'mlp'])


def parameter_count(cfg, horizon, lookback, head):
    model = create_model('s4' if head == 's4' else 'mlp', horizon=horizon, lookback=lookback,
                         width=width_for(cfg, horizon, head), **cfg['model'])
    return count_parameters(model)


# ------------------------------------------- geometry (no test data, no models)

def prepare_group(cfg, out, kernel, geometry_id, data_seed, fingerprint):
    """Train/val data, PCA bases, spectra and floors for one (kernel, geometry, draw).

    Recomputed deterministically on every invocation and compared with the
    frozen geometry.json. Test windows are never generated here.
    """
    geometry = make_geometry(cfg, kernel, geometry_id)
    data = {split: sample_windows(geometry, cfg['n_' + split], data_seed + SPLIT_OFFSETS[split])
            for split in ('train', 'val')}
    target_pca = fit_pca(data['train'].target)
    floor = nugget_floor(geometry, data['train'].scale)
    signal = signal_count(target_pca.eigenvalues, floor['target_floor'], cfg['signal_floor_multiple'])
    groups = pc_groups(geometry.horizon, signal)
    raw = np.sort(np.linalg.eigvalsh(source_covariance(geometry, 'gp')))[::-1]
    validation_target = spectrum(data['val'].target, target_pca)
    record = {'kernel': kernel, 'geometry_id': geometry_id, 'geometry': geometry.to_dict(),
              'data_seed': data_seed, 'config_sha256': config_hash(cfg), 'fingerprint': fingerprint,
              'test_inspected': False,
              'raw_kernel_horizon_spectrum': {
                  'eigenvalues': raw.tolist(), 'top2_share': float(raw[:2].sum() / raw.sum()),
                  'components_above_1e-3_of_first': int(np.sum(raw > 1e-3 * raw[0]))},
              'nugget_floor': floor, 'signal_floor_multiple': cfg['signal_floor_multiple'],
              'signal_pcs': signal, 'pc_groups': groups,
              'target_pca': target_pca.to_dict(), 'validation_target': validation_target,
              'split_hashes': {split: {'raw_context': array_hash(ds.raw_context),
                                       'raw_target': array_hash(ds.raw_target)}
                               for split, ds in data.items()},
              'sources': {}}
    pcas = {}
    for source in SOURCES:
        eps = source_draw(geometry, cfg['n_train'], source, data_seed + PCA_SOURCE_OFFSET)
        pca = fit_pca(data['train'].target - eps)
        pcas[source] = pca
        validation_eps = source_draw(geometry, cfg['n_val'], source, data_seed + VAL_SOURCE_OFFSET)
        velocity_floor = floor['velocity_floor'][source]
        tail = groups['residual_tail']
        record['sources'][source] = {
            'pca': pca.to_dict(),
            'validation_velocity': spectrum(data['val'].target - validation_eps, pca),
            'pcs_for_90pct_energy': energy_count(pca.eigenvalues, .90),
            'pcs_for_99pct_energy': energy_count(pca.eigenvalues, .99),
            'velocity_floor': velocity_floor,
            'eigenvalue_to_floor': (pca.eigenvalues / velocity_floor).tolist(),
            'tail_mean_eigenvalue_to_floor': (float(pca.eigenvalues[tail].mean() / velocity_floor)
                                              if tail else None),
            'floored_directions': int(np.sum(pca.eigenvalues < pca.floor))}
    gp_top2 = record['sources']['gp']['validation_velocity']['top2_energy']
    white_top2 = record['sources']['white']['validation_velocity']['top2_energy']
    record['task1_branch'] = task1_branch(gp_top2, white_top2, validation_target)
    record['reference_gate_pass'] = bool(gp_top2 > .9 and validation_target['mean_within_patch_std'] > .02
                                         and validation_target['roughness'] > .002)
    record['gate_enforced'] = False
    path = group_dir(out, kernel, geometry_id, data_seed) / 'geometry.json'
    if path.exists() and load_config(path) != record:
        raise ValueError('Geometry differs from frozen record: ' + str(path))
    save_json(path, record)
    return {'geometry': geometry, 'data': data, 'pcas': pcas, 'target_pca': target_pca,
            'groups': groups, 'record': record}


# --------------------------------------------------------------- training

def make_model(cfg, geometry, head, pca, seed, device):
    """Main's initialization: seed -> head, encoder re-initialized identically across heads."""
    if head not in HEADS:
        raise ValueError(head)
    torch.manual_seed(seed)
    base = create_model('s4' if head == 's4' else 'mlp', horizon=geometry.horizon,
                        lookback=geometry.lookback, width=width_for(cfg, geometry.horizon, head),
                        **cfg['model'])
    torch.manual_seed(10000 + seed)
    for module in base.encoder.modules():
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
    return CoordinateModel(base, pca if head == 'whitened_mlp' else None).to(device)


def oracle_reference(cfg, bundle, test, source, data_seed, pairs, fingerprint, path):
    """Analytic reference shared by all heads of one (group, source); cached on disk."""
    tx, tu, tt, _ = pairs
    inputs = {'raw_context': array_hash(test.raw_context), 'raw_target': array_hash(test.raw_target),
              'time': array_hash(tt), 'velocity': array_hash(tu)}
    if path.exists():
        record = load_config(path)
        if record['fingerprint'] != fingerprint or record['inputs'] != inputs:
            raise ValueError('Oracle reference does not match frozen inputs: ' + str(path))
        return record
    geometry, pca, groups = bundle['geometry'], bundle['pcas'][source], bundle['groups']
    means, residual = conditional_moments(geometry, test)
    source_cov = source_covariance(geometry, source)
    prediction = oracle_velocity(tx, tt, means, residual, test.scale, source_cov, cfg['oracle_chunk'])
    velocity = velocity_metrics(prediction, tu, pca)
    noise = oracle_noise_covariance(tt, residual, test.scale, source_cov, cfg['oracle_chunk'])
    pc_noise = np.diag(pca.components @ noise @ pca.components.T)
    n = min(cfg['n_paths'], len(test))
    paths = sample_conditional(means[:n], residual, test.scale[:n], 1,
                               data_seed + ORACLE_PATH_OFFSET)[:, 0]
    nc = min(cfg['n_ensemble_contexts'], len(test))
    ensemble = sample_conditional(means[:nc], residual, test.scale[:nc], cfg['ensemble_size'],
                                  data_seed + ORACLE_ENSEMBLE_OFFSET)
    scores = context_scores(ensemble, test.target[:nc])
    record = {'information_set': 'privileged raw history; networks see normalized history only',
              'fingerprint': fingerprint, 'source': source, 'inputs': inputs,
              'residual_cov_min_eigenvalue': float(np.linalg.eigvalsh(residual).min()),
              'velocity': velocity, 'velocity_groups': group_means(velocity['pc_mse'], groups),
              'irreducible_noise': {'pc_noise': pc_noise.tolist(), 'groups': group_means(pc_noise, groups)},
              'exact_paths': {'generated_roughness': float(np.mean(np.abs(np.diff(paths, axis=1)))),
                              'target_roughness': float(np.mean(np.abs(np.diff(test.target[:n], axis=1))))},
              'forecast_means': {k: float(v.mean()) for k, v in scores.items()},
              'per_context': {k: v.tolist() for k, v in scores.items()}}
    save_json(path, record)
    return record


def oracle_gap(groups_model, forecast, oracle):
    gap = {}
    for name, value in groups_model.items():
        reference = oracle['velocity_groups'][name]
        if value is not None and reference is not None:
            gap[name + '_mse_excess'] = value - reference
            gap[name + '_mse_ratio'] = value / reference if reference > 0 else None
    for name in ('fair_crps', 'fair_energy_score'):
        gap[name + '_excess'] = forecast[name] - oracle['forecast_means'][name]
        gap[name + '_ratio'] = forecast[name] / oracle['forecast_means'][name]
    return gap


def train_run(cfg, bundle, out, kernel, geometry_id, data_seed, seed, source, head, device,
              fingerprint, replicate_index):
    geometry, data, pca, groups = bundle['geometry'], bundle['data'], bundle['pcas'][source], bundle['groups']
    root = group_dir(out, kernel, geometry_id, data_seed)
    run = f'{source}_{head}'
    label = f'{kernel}/{geometry_id}/draw{data_seed}/{run}'
    record_path = root / 'runs' / (run + '.json')
    checkpoint = root / 'checkpoints' / (run + '.pt')
    sample_path = root / 'samples' / (run + '.npz')
    if record_path.exists():
        old = load_config(record_path)
        if (old['fingerprint'] != fingerprint or old['checkpoint_sha256'] != file_hash(checkpoint)
                or old['samples_sha256'] != file_hash(sample_path)):
            raise ValueError('Cannot reuse incompatible result ' + label)
        print('SKIP', label, flush=True)
        return old
    started = time.perf_counter()
    model = make_model(cfg, geometry, head, pca, seed, device)
    parameters = count_parameters(model)
    encoder_hash = array_hash(np.concatenate([p.detach().cpu().numpy().ravel()
                                              for p in model.base.encoder.parameters()]))
    optimizer = torch.optim.AdamW(optimizer_groups(model, cfg['learning_rate'], cfg['weight_decay']))
    y, h = tensor(data['train'].target, device), tensor(data['train'].context, device)
    chol = tensor(np.linalg.cholesky(source_covariance(geometry, source)), device)
    vx, vu, vt, _ = fixed_pairs(geometry, data['val'], source, data_seed, 'val')
    vx, vu, vt, vh = [tensor(a, device) for a in (vx, vu, vt, data['val'].context)]
    # Training randomness reset after initialization: batches, normals and times paired across heads.
    torch.manual_seed(20000 + seed)
    if device.type == 'mps':
        torch.mps.manual_seed(20000 + seed)
    elif device.type == 'cuda':
        torch.cuda.manual_seed(20000 + seed)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best, best_step, curve, batch_hashes = float('inf'), 0, [], {}
    print('START', label, 'params', parameters, 'device', device, 'steps', cfg['steps'], flush=True)
    for step in range(1, cfg['steps'] + 1):
        model.train()
        ix = torch.randint(len(y), (cfg['batch_size'],), device=device)
        z = torch.randn(len(ix), geometry.horizon, device=device)
        t = torch.rand(len(ix), device=device)
        if step in (1, cfg['steps']):
            batch_hashes[str(step)] = {k: array_hash(a.detach().cpu().numpy())
                                       for k, a in (('indices', ix), ('normals', z), ('time', t))}
        eps, yy, hh = z @ chol.T, y[ix], h[ix]
        x, u = (1 - t[:, None]) * eps + t[:, None] * yy, yy - eps
        optimizer.zero_grad(set_to_none=True)
        loss = (model.forward_coordinates(x, t, hh) - model.to_coordinates(u)).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad'])
        optimizer.step()
        if step % cfg['eval_every'] == 0 or step == cfg['steps']:
            model.eval()
            with torch.no_grad():
                vp = predict(model, vx, vt, vh)
                validation = model.to_coordinates(vp - vu).square().mean().item()
                raw = (vp - vu).square().mean().item()
            if not np.isfinite(validation):
                raise FloatingPointError('Non-finite validation loss: ' + label)
            curve.append({'step': step, 'train_loss': loss.item(), 'validation_objective': validation,
                          'validation_raw_mse': raw, 'seconds': time.perf_counter() - started})
            if validation < best:
                best, best_step = validation, step
                torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            'config': cfg, 'kernel': kernel, 'geometry_id': geometry_id,
                            'data_seed': data_seed, 'source': source, 'head': head, 'seed': seed,
                            'step': step, 'pca': pca.to_dict(), 'fingerprint': fingerprint}, checkpoint)
            save_json(root / 'training' / (run + '.json'), curve)
            print('PROGRESS', label, step, 'val', round(validation, 6), flush=True)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)['state_dict'])
    model.eval()
    # First access to test windows: after the last update and checkpoint selection.
    test = sample_windows(geometry, cfg['n_test'], data_seed + SPLIT_OFFSETS['test'])
    pairs = fixed_pairs(geometry, test, source, data_seed, 'test')
    tx, tu, tt, teps = pairs
    prediction = predict(model, tensor(tx, device), tensor(tt, device), tensor(test.context, device)).cpu().numpy()
    n = min(cfg['n_paths'], len(test))
    generated = generate(model, tensor(teps[:n], device), tensor(test.context[:n], device),
                         cfg['euler_steps'], cfg['generation_batch_size'])
    nc, ns = min(cfg['n_ensemble_contexts'], len(test)), cfg['ensemble_size']
    ensemble_eps = source_draw(geometry, nc * ns, source, data_seed + ENSEMBLE_OFFSET).reshape(nc, ns, -1)
    ensemble_history = tensor(np.repeat(test.context[:nc], ns, axis=0), device)
    ensemble = generate(model, tensor(ensemble_eps.reshape(-1, geometry.horizon), device), ensemble_history,
                        cfg['euler_steps'], cfg['generation_batch_size']).reshape(nc, ns, -1)
    scores = context_scores(ensemble, test.target[:nc])
    forecast = {k: float(v.mean()) for k, v in scores.items()}
    metrics = evaluate(prediction, tu, generated, test.target[:n], pca)
    groups_model = group_means(metrics['pc_mse'], groups)
    if groups_model['residual_signal'] is not None:
        signal_rule = success_rule(metrics['pc1_mse'], groups_model['residual_signal'],
                                   metrics['generated_roughness'], metrics['target_roughness'])
    else:
        signal_rule = None
    solver, extra = {}, {}
    check = cfg['solver_check']
    if replicate_index == check['replicate_index']:
        ni = min(check['n_contexts'], nc)
        fine = generate(model, tensor(ensemble_eps[:ni].reshape(-1, geometry.horizon), device),
                        ensemble_history[:ni * ns], check['euler_steps'],
                        cfg['generation_batch_size']).reshape(ni, ns, -1)
        fine_scores, base_scores = context_scores(fine, test.target[:ni]), context_scores(ensemble[:ni], test.target[:ni])
        solver = {'n_contexts': ni, 'euler_steps': check['euler_steps'],
                  'endpoint_rmse': float(np.sqrt(np.mean((fine.astype(float) - ensemble[:ni]) ** 2))),
                  'score_changes': {k: float(fine_scores[k].mean() - base_scores[k].mean()) for k in base_scores}}
        extra['ensemble_fine_solver'] = fine
    oracle = oracle_reference(cfg, bundle, test, source, data_seed, pairs, fingerprint,
                              root / f'oracle_{source}.json')
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sample_path, prediction=prediction, generated=generated, ensemble=ensemble,
                        path_target=test.target[:n], path_history=test.context[:n],
                        ensemble_target=test.target[:nc], **extra)
    record = {'run': run, 'kernel': kernel, 'geometry_id': geometry_id, 'geometry': geometry.to_dict(),
              'data_seed': data_seed, 'seed': seed, 'source': source, 'head': head,
              'width': width_for(cfg, geometry.horizon, head), 'parameters': parameters,
              'config_sha256': config_hash(cfg), 'fingerprint': fingerprint, 'status': 'complete',
              'steps': cfg['steps'], 'device': str(device), 'torch': torch.__version__,
              'best_step': best_step, 'best_validation_objective': best,
              'encoder_initial_sha256': encoder_hash, 'batch_hashes': batch_hashes, 'curve': curve,
              'test_input_hashes': {'raw_context': array_hash(test.raw_context),
                                    'raw_target': array_hash(test.raw_target), 'time': array_hash(tt),
                                    'source': array_hash(teps), 'ensemble_source': array_hash(ensemble_eps)},
              'pc_groups': groups, 'signal_pcs': bundle['record']['signal_pcs'],
              'metrics': metrics, 'group_mse': groups_model, 'signal_group_rule_diagnostic': signal_rule,
              'fixed_target_basis_metrics': velocity_metrics(prediction, tu, bundle['target_pca']),
              'forecast_means': forecast, 'per_context': {k: v.tolist() for k, v in scores.items()},
              'oracle_gap': oracle_gap(groups_model, forecast, oracle),
              'solver_sensitivity': solver, 'test_spectrum': spectrum(tu, pca),
              'checkpoint_sha256': file_hash(checkpoint), 'samples_sha256': file_hash(sample_path),
              'seconds': time.perf_counter() - started}
    save_json(record_path, record)
    print('DONE', label, 'seconds', round(record['seconds'], 1), 'residual', round(metrics['residual_mse'], 6),
          'CRPS', round(forecast['fair_crps'], 6), flush=True)
    return record


# --------------------------------------------------------------- CLI

def identities(cfg):
    """(replicate index, data seed, seed, geometry, kernel, source, head) in run order."""
    return [(i, r['data_seed'], r['seed'], g['id'], k, s, h)
            for i, r in enumerate(cfg['replicates']) for g in cfg['geometries']
            for k in cfg['kernels'] for s in cfg['sources'] for h in cfg['heads']]


def _parse(values, allowed, kind, cast=str):
    if values is None:
        return None
    chosen = [cast(v) for v in values.split(',')]
    unknown = [v for v in chosen if v not in allowed]
    if unknown:
        raise ValueError(f'Unknown {kind}: {unknown}; allowed {sorted(allowed)}')
    return set(chosen)


def select(cfg, kernel=None, ratio=None, geometry=None, head=None, source=None, data_seed=None):
    specs = geometry_specs(cfg)
    ratios = {g: specs[g]['horizon'] / specs[g]['lengthscale'] for g in specs}
    keep = {'kernel': _parse(kernel, set(cfg['kernels']), 'kernel'),
            'ratio': _parse(ratio, set(ratios.values()), 'ratio', float),
            'geometry': _parse(geometry, set(specs), 'geometry'),
            'head': _parse(head, set(cfg['heads']), 'head'),
            'source': _parse(source, set(cfg['sources']), 'source'),
            'data_seed': _parse(data_seed, {r['data_seed'] for r in cfg['replicates']}, 'data seed', int)}
    rows = []
    for row in identities(cfg):
        _, d, _, g, k, s, h = row
        values = {'kernel': k, 'ratio': ratios[g], 'geometry': g, 'head': h, 'source': s, 'data_seed': d}
        if all(keep[name] is None or values[name] in keep[name] for name in keep):
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/kernel_scan.json')
    p.add_argument('--output', default='results/kernel-scan')
    p.add_argument('--device', choices=['mps', 'cuda', 'cpu'], default='mps')
    p.add_argument('--prepare-only', action='store_true', help='geometry records only; no training, no test data')
    p.add_argument('--list', action='store_true', help='print selected runs and their status, then exit')
    for name in ('kernel', 'ratio', 'geometry', 'head', 'source', 'data-seed'):
        p.add_argument('--' + name, help='comma-separated filter')
    a = p.parse_args()
    cfg = load_config(a.config)
    rows = select(cfg, a.kernel, a.ratio, a.geometry, a.head, a.source, a.data_seed)
    if a.list:
        done = 0
        for _, d, _, g, k, s, h in rows:
            exists = (group_dir(a.output, k, g, d) / 'runs' / f'{s}_{h}.json').exists()
            done += exists
            print('done   ' if exists else 'pending', k, g, f'draw{d}', s, h)
        print(f'{done}/{len(rows)} selected runs complete')
        return
    device = torch.device(a.device)
    if not a.prepare_only and a.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable to this process; no silent CPU fallback (pass --device cpu explicitly)')
    if not a.prepare_only and a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable to this process; no silent CPU fallback (pass --device cpu explicitly)')
    if not a.prepare_only:
        # GPU floating point differs across backends: one results directory holds one device.
        others = sorted(p.name for p in Path(a.output).glob('environment_*.json')
                        if p.name != f'environment_{a.device}.json')
        if others:
            raise RuntimeError(f'{a.output} already holds runs from another device ({others}); use a fresh output directory')
    torch.set_num_threads(4)
    fingerprint = prepare_root(cfg, a.output)
    if not a.prepare_only:
        save_json(Path(a.output) / f'environment_{a.device}.json',
                  {'device': str(device), 'torch': torch.__version__, 'numpy': np.__version__,
                   'platform': platform.platform(), 'mps_available': torch.backends.mps.is_available(),
                   'cuda_available': torch.cuda.is_available(),
                   'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    print('SELECTED', len(rows), 'runs', flush=True)
    bundle, key = None, None
    for index, d, seed, g, k, s, h in rows:
        if (d, g, k) != key:
            bundle, key = prepare_group(cfg, a.output, k, g, d, fingerprint), (d, g, k)
            r = bundle['record']
            print('GEOMETRY', k, g, f'draw{d}', 'gp_top2', round(r['sources']['gp']['validation_velocity']['top2_energy'], 4),
                  'white_top2', round(r['sources']['white']['validation_velocity']['top2_energy'], 4),
                  'signal_pcs', r['signal_pcs'], r['task1_branch'], flush=True)
        if not a.prepare_only:
            train_run(cfg, bundle, a.output, k, g, d, seed, s, h, device, fingerprint, index)


if __name__ == '__main__':
    main()
