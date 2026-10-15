"""E5: ETTh1-OT with the GP prior's level component removed (docs/etth1_prior_level_protocol.md).

The training and evaluation loop mirrors extended_experiment.train_cell step for
step, including the RNG order. The only additions are the `gp_zero_level` prior
(GP draws minus their mean over the horizon) and a local re-run of the white and
GP reference cells on one device. Numerical source hashes are frozen into
<output>/manifest.json, and a resumed run refuses any change.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .experiment import config_hash, generate, load_config, predict, save_json, tensor
from .extended_data import build_data, sample_source
from .extended_experiment import evaluation_indices
from .extended_metrics import forecast_scores, means
from .extended_models import create_extended_model, parameter_counts
from .metrics import fit_pca, velocity_metrics
from .models import optimizer_groups

SOURCE_FILES = ('fm_stress/etth1_prior_level.py', 'fm_stress/extended_data.py', 'fm_stress/extended_models.py',
                'fm_stress/extended_metrics.py', 'fm_stress/extended_experiment.py', 'fm_stress/models.py',
                'fm_stress/experiment.py', 'fm_stress/metrics.py')
PRIORS = ('white', 'gp', 'gp_zero_level', 'gp_white_level')
# Separate stream for the replacement level of gp_white_level, so the main RNG order is unchanged.
LEVEL_SEED_OFFSET = 40000


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base_family(prior):
    if prior not in PRIORS:
        raise ValueError(f'Unknown prior: {prior}')
    return 'white' if prior == 'white' else 'gp'


def prior_draw(bundle, n, prior, seed):
    """[N,F,C] prior draws.

    gp_zero_level removes each curve's mean over the horizon. gp_white_level replaces
    it by an independent N(0, 1/F) level, the level variance of the white prior,
    so the GP shape is kept while the level is as small as the white prior's.
    """
    e = sample_source(bundle, n, base_family(prior), seed)
    if prior in ('gp_zero_level', 'gp_white_level'):
        e = e - e.mean(axis=1, keepdims=True)
    if prior == 'gp_white_level':
        f, c = e.shape[1], e.shape[2]
        e = e + np.random.default_rng(seed + LEVEL_SEED_OFFSET).standard_normal((n, 1, c)) / np.sqrt(f)
    return e


def pairs(bundle, split, prior, seed):
    ds = bundle.splits[split]
    e = prior_draw(bundle, len(ds), prior, seed).astype(np.float32)
    t = np.random.default_rng(seed + 1).random(len(ds)).astype(np.float32)
    y = ds.target.astype(np.float32)
    return (1 - t[:, None, None]) * e + t[:, None, None] * y, y - e, t, e


def manifest(cfg):
    root = Path(__file__).resolve().parents[1]
    m = {'config_sha256': config_hash(cfg),
         'source_hashes': {name: file_hash(root / name) for name in SOURCE_FILES},
         'dataset_sha256': file_hash(root / cfg['dataset']['path'])}
    m['fingerprint'] = config_hash(m)
    return m


def prepare_root(cfg, out):
    out = Path(out)
    m = manifest(cfg)
    for name, record in (('config.json', cfg), ('manifest.json', m)):
        path = out / name
        if path.exists() and load_config(path) != record:
            raise ValueError('Changed provenance: use fresh output directory (' + str(path) + ')')
        save_json(path, record)
    return m['fingerprint']


def train_run(cfg, bundle, out, head, prior, seed, fingerprint, device):
    name = f'{prior}_{head}_seed{seed}'
    out = Path(out)
    record_path, checkpoint, sample_path = (out / 'runs' / f'{name}.json', out / 'checkpoints' / f'{name}.pt',
                                            out / 'samples' / f'{name}.npz')
    if record_path.exists():
        old = load_config(record_path)
        if (old['fingerprint'] != fingerprint or old['checkpoint_sha256'] != file_hash(checkpoint)
                or old['samples_sha256'] != file_hash(sample_path)):
            raise ValueError('Cannot reuse incompatible result ' + name)
        print('SKIP', name, flush=True)
        return old
    started = time.perf_counter()
    options = {**cfg['model'], 'head': head}
    torch.manual_seed(seed)
    model = create_extended_model(**options).to(device)
    optimizer = torch.optim.AdamW(optimizer_groups(model, cfg['learning_rate'], cfg['weight_decay']))
    train, val, test = (bundle.splits[s] for s in ('train', 'val', 'test'))
    y, h = tensor(train.target, device), tensor(train.context, device)
    chol = tensor(np.linalg.cholesky(bundle.source_covariances[base_family(prior)]), device)
    vx, vu, vt, _ = pairs(bundle, 'val', prior, cfg['evaluation_seed'])
    vx, vu, vt, vh = [tensor(a, device) for a in (vx, vu, vt, val.context)]
    # Same RNG order as extended_experiment: minibatches, normals and times paired across heads.
    torch.manual_seed(seed + 30000)
    if device.type == 'mps':
        torch.mps.manual_seed(seed + 30000)
    elif device.type == 'cuda':
        torch.cuda.manual_seed(seed + 30000)
    level_generator = torch.Generator().manual_seed(seed + LEVEL_SEED_OFFSET)
    curve, best, best_step = [], float('inf'), 0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print('START', name, parameter_counts(model)['effective_trainable'], 'device', device, flush=True)
    for step in range(1, cfg['steps'] + 1):
        model.train()
        ix = torch.randint(len(y), (cfg['batch_size'],), device=device)
        yy, hh = y[ix], h[ix]
        eps = (torch.randn(len(ix), chol.shape[0], device=device) @ chol.T).reshape_as(yy)
        if prior in ('gp_zero_level', 'gp_white_level'):
            eps = eps - eps.mean(dim=1, keepdim=True)
        if prior == 'gp_white_level':
            level = torch.randn(len(ix), 1, yy.shape[2], generator=level_generator) / yy.shape[1] ** .5
            eps = eps + level.to(device)
        t = torch.rand(len(ix), device=device)
        xx = (1 - t[:, None, None]) * eps + t[:, None, None] * yy
        optimizer.zero_grad(set_to_none=True)
        loss = (model(xx, t, hh) - (yy - eps)).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad'])
        optimizer.step()
        if step % cfg['eval_every'] == 0 or step == cfg['steps']:
            model.eval()
            with torch.no_grad():
                validation = float((predict(model, vx, vt, vh) - vu).square().mean().cpu())
            if not np.isfinite(validation):
                raise FloatingPointError(name)
            curve.append({'step': step, 'train_loss': float(loss.detach().cpu()), 'validation_mse': validation,
                          'seconds': time.perf_counter() - started})
            if validation < best:
                best, best_step = validation, step
                torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            'options': options, 'step': step, 'seed': seed, 'prior': prior,
                            'fingerprint': fingerprint}, checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)['state_dict'])
    model.eval()
    # First access to test windows: after the last update and checkpoint selection.
    tx, tu, tt, _ = pairs(bundle, 'test', prior, cfg['evaluation_seed'] + 100)
    with torch.no_grad():
        prediction = predict(model, tensor(tx, device), tensor(tt, device), tensor(test.context, device)).cpu().numpy()
    target_pca = fit_pca(train.target.reshape(len(train), -1))
    vm = velocity_metrics(prediction.reshape(len(test), -1), tu.reshape(len(test), -1), target_pca)
    ix = evaluation_indices(test, cfg['n_ensemble_contexts'])
    n, s, f, c = len(ix), cfg['ensemble_size'], test.target.shape[1], test.target.shape[2]
    eps = prior_draw(bundle, n * s, prior, cfg['evaluation_seed'] + 200)
    hh = np.repeat(test.context[ix], s, axis=0)
    gen = generate(model, tensor(eps, device), tensor(hh, device), cfg['euler_steps'],
                   cfg['generation_batch_size']).reshape(n, s, f, c)
    scores = forecast_scores(gen, test.target[ix])
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sample_path, prediction=prediction, generated=gen, indices=ix)
    record = {'name': name, 'status': 'complete', 'prior': prior, 'head': head, 'seed': seed, 'device': str(device),
              'torch': torch.__version__, 'model_options': options, 'parameters': parameter_counts(model),
              'steps': cfg['steps'], 'best_step': best_step, 'best_validation_mse': best, 'curve': curve,
              'test_velocity_mse': float(np.mean((prediction - tu) ** 2)),
              'velocity_metrics_target_basis': vm, 'forecast_means': means(scores),
              'per_context': {k: v.tolist() for k, v in scores.items()},
              'config_sha256': config_hash(cfg), 'fingerprint': fingerprint,
              'checkpoint_sha256': file_hash(checkpoint), 'samples_sha256': file_hash(sample_path),
              'seconds': time.perf_counter() - started}
    save_json(record_path, record)
    print('DONE', name, 'best_step', best_step, 'val', round(best, 5), 'CRPS', round(record['forecast_means']['fair_crps'], 5),
          'seconds', round(record['seconds'], 1), flush=True)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/etth1_prior_level.json')
    p.add_argument('--output', default='results/etth1-prior-level')
    p.add_argument('--device', choices=['mps', 'cuda', 'cpu'], default='mps')
    p.add_argument('--head', choices=['mlp', 's4'])
    p.add_argument('--prior', choices=PRIORS)
    p.add_argument('--seed', type=int)
    p.add_argument('--smoke-steps', type=int, help='development only: override the number of updates')
    a = p.parse_args()
    cfg = load_config(a.config)
    if a.smoke_steps:
        cfg = {**cfg, 'steps': a.smoke_steps, 'smoke': True}
    device = torch.device(a.device)
    if a.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable to this process; no silent CPU fallback (pass --device cpu explicitly)')
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable to this process; no silent CPU fallback (pass --device cpu explicitly)')
    out = Path(a.output)
    others = sorted(q.name for q in out.glob('environment_*.json') if q.name != f'environment_{a.device}.json')
    if others:
        raise RuntimeError(f'{out} already holds runs from another device ({others}); use a fresh output directory')
    torch.set_num_threads(4)
    bundle = build_data(cfg['dataset'])
    sizes = bundle.metadata['split_sizes']
    if sizes != {'train': 2143, 'val': 118, 'test': 118}:
        raise ValueError(f'Unexpected ETTh1 split sizes {sizes}')
    fingerprint = prepare_root(cfg, out)
    save_json(out / f'environment_{a.device}.json', {'device': str(device), 'torch': torch.__version__,
                                                     'numpy': np.__version__, 'platform': platform.platform()})
    for seed in cfg['seeds']:
        if a.seed is not None and seed != a.seed:
            continue
        for prior in cfg['priors']:
            if a.prior and prior != a.prior:
                continue
            for head in cfg['heads']:
                if a.head and head != a.head:
                    continue
                train_run(cfg, bundle, out, head, prior, seed, fingerprint, device)


if __name__ == '__main__':
    main()
