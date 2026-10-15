"""E4: ETTh1-OT under global versus history (Sundial) normalization (docs/etth1_normalization_protocol.md).

The global cells repeat the extension / E5 setup. The history cells normalize every
window by its own history mean and standard deviation (threshold 0.01 -> 1, as in
data.sundial_normalize). The model trains and generates in those coordinates, and
forecasts are mapped back to train-standardized units, mu_h + s_h * x. All
forecasts are therefore scored on the same 64 test contexts in the same units.
The training loop, RNG order and evaluation draws are those of
etth1_prior_level.train_run.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .etth1_prior_level import base_family, file_hash, pairs, prior_draw
from .experiment import config_hash, generate, load_config, predict, save_json, tensor
from .extended_data import ExtendedData, WindowDataset, build_data
from .extended_experiment import evaluation_indices
from .extended_metrics import forecast_scores, means
from .extended_models import create_extended_model, parameter_counts
from .models import optimizer_groups

SOURCE_FILES = ('fm_stress/etth1_normalization.py', 'fm_stress/etth1_prior_level.py', 'fm_stress/extended_data.py',
                'fm_stress/extended_models.py', 'fm_stress/extended_metrics.py', 'fm_stress/extended_experiment.py',
                'fm_stress/models.py', 'fm_stress/experiment.py', 'fm_stress/metrics.py')
NORMALIZATIONS = ('global', 'history')


def history_normalize(ds, threshold=1e-2):
    """Per-window history mean/std normalization; returns the dataset and its [N,1,C] statistics."""
    mu = ds.context.mean(axis=1, keepdims=True)
    sd = ds.context.std(axis=1, keepdims=True)
    sd = np.where(sd > threshold, sd, 1.0)
    return WindowDataset((ds.context - mu) / sd, (ds.target - mu) / sd, ds.starts), mu, sd


def model_bundle(bundle, normalization):
    """The data the model sees, plus per-split (mu, sd) mapping model units back to train-standardized units."""
    if normalization == 'global':
        return bundle, {s: (np.zeros((len(ds), 1, ds.target.shape[2])), np.ones((len(ds), 1, ds.target.shape[2])))
                        for s, ds in bundle.splits.items()}
    if normalization != 'history':
        raise ValueError(normalization)
    splits, stats = {}, {}
    for s, ds in bundle.splits.items():
        splits[s], mu, sd = history_normalize(ds)
        stats[s] = (mu, sd)
    return ExtendedData(splits, bundle.source_covariances, bundle.metadata), stats


def manifest(cfg):
    root = Path(__file__).resolve().parents[1]
    m = {'config_sha256': config_hash(cfg), 'source_hashes': {n: file_hash(root / n) for n in SOURCE_FILES},
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


def train_run(cfg, bundle, out, normalization, head, prior, seed, fingerprint, device):
    name = f'{normalization}_{prior}_{head}_seed{seed}'
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
    mb, stats = model_bundle(bundle, normalization)
    options = {**cfg['model'], 'head': head}
    torch.manual_seed(seed)
    model = create_extended_model(**options).to(device)
    optimizer = torch.optim.AdamW(optimizer_groups(model, cfg['learning_rate'], cfg['weight_decay']))
    train, val, test = (mb.splits[s] for s in ('train', 'val', 'test'))
    y, h = tensor(train.target, device), tensor(train.context, device)
    chol = tensor(np.linalg.cholesky(mb.source_covariances[base_family(prior)]), device)
    vx, vu, vt, _ = pairs(mb, 'val', prior, cfg['evaluation_seed'])
    vx, vu, vt, vh = [tensor(a, device) for a in (vx, vu, vt, val.context)]
    torch.manual_seed(seed + 30000)
    if device.type == 'mps':
        torch.mps.manual_seed(seed + 30000)
    elif device.type == 'cuda':
        torch.cuda.manual_seed(seed + 30000)
    curve, best, best_step = [], float('inf'), 0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print('START', name, parameter_counts(model)['effective_trainable'], 'device', device, flush=True)
    for step in range(1, cfg['steps'] + 1):
        model.train()
        ix = torch.randint(len(y), (cfg['batch_size'],), device=device)
        yy, hh = y[ix], h[ix]
        eps = (torch.randn(len(ix), chol.shape[0], device=device) @ chol.T).reshape_as(yy)
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
                            'normalization': normalization, 'fingerprint': fingerprint}, checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)['state_dict'])
    model.eval()
    # First access to test windows: after the last update and checkpoint selection.
    tx, tu, tt, _ = pairs(mb, 'test', prior, cfg['evaluation_seed'] + 100)
    with torch.no_grad():
        prediction = predict(model, tensor(tx, device), tensor(tt, device), tensor(test.context, device)).cpu().numpy()
    ix = evaluation_indices(test, cfg['n_ensemble_contexts'])
    n, s, f, c = len(ix), cfg['ensemble_size'], test.target.shape[1], test.target.shape[2]
    eps = prior_draw(mb, n * s, prior, cfg['evaluation_seed'] + 200)
    hh = np.repeat(test.context[ix], s, axis=0)
    gen_model = generate(model, tensor(eps, device), tensor(hh, device), cfg['euler_steps'],
                         cfg['generation_batch_size']).reshape(n, s, f, c)
    mu, sd = stats['test']
    gen = mu[ix][:, None] + sd[ix][:, None] * gen_model          # back to train-standardized units
    scores = forecast_scores(gen, bundle.splits['test'].target[ix])
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sample_path, generated=gen, generated_model_units=gen_model, indices=ix, prediction=prediction)
    record = {'name': name, 'status': 'complete', 'normalization': normalization, 'prior': prior, 'head': head,
              'seed': seed, 'device': str(device), 'torch': torch.__version__, 'model_options': options,
              'parameters': parameter_counts(model), 'steps': cfg['steps'], 'best_step': best_step,
              'best_validation_mse': best, 'curve': curve,
              'test_velocity_mse_model_units': float(np.mean((prediction - tu) ** 2)),
              'forecast_means': means(scores), 'per_context': {k: v.tolist() for k, v in scores.items()},
              'config_sha256': config_hash(cfg), 'fingerprint': fingerprint,
              'checkpoint_sha256': file_hash(checkpoint), 'samples_sha256': file_hash(sample_path),
              'seconds': time.perf_counter() - started}
    save_json(record_path, record)
    print('DONE', name, 'best_step', best_step, 'CRPS', round(record['forecast_means']['fair_crps'], 5),
          'seconds', round(record['seconds'], 1), flush=True)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/etth1_normalization.json')
    p.add_argument('--output', default='results/etth1-normalization')
    p.add_argument('--device', choices=['mps', 'cuda', 'cpu'], default='cuda')
    p.add_argument('--normalization', choices=NORMALIZATIONS)
    p.add_argument('--head', choices=['mlp', 's4'])
    p.add_argument('--prior', choices=['white', 'gp'])
    p.add_argument('--seed', type=int)
    a = p.parse_args()
    cfg = load_config(a.config)
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
    if bundle.metadata['split_sizes'] != {'train': 2143, 'val': 118, 'test': 118}:
        raise ValueError(f"Unexpected ETTh1 split sizes {bundle.metadata['split_sizes']}")
    fingerprint = prepare_root(cfg, out)
    save_json(out / f'environment_{a.device}.json', {'device': str(device), 'torch': torch.__version__,
                                                     'numpy': np.__version__, 'platform': platform.platform()})
    for seed in cfg['seeds']:
        if a.seed is not None and seed != a.seed:
            continue
        for normalization in cfg['normalizations']:
            if a.normalization and normalization != a.normalization:
                continue
            for prior in cfg['priors']:
                if a.prior and prior != a.prior:
                    continue
                for head in cfg['heads']:
                    if a.head and head != a.head:
                        continue
                    train_run(cfg, bundle, out, normalization, head, prior, seed, fingerprint, device)


if __name__ == '__main__':
    main()
