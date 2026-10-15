#!/usr/bin/env python3
"""Read-only analysis of E4 (docs/etth1_normalization_protocol.md).

Checks the 24 runs, recomputes fair CRPS from the saved train-standardized forecasts
with an independent NumPy implementation, applies the pre-declared rules, and checks
that history-normalized forecasts follow a history shift exactly. As a supplementary
baseline added after the verdict, it refits the extension's ridge + residual bootstrap
under each normalization on the same contexts and units.
Writes <results>/summary.json.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fm_stress.etth1_normalization import model_bundle  # noqa: E402
from fm_stress.etth1_prior_level import prior_draw  # noqa: E402
from fm_stress.experiment import generate  # noqa: E402
from fm_stress.extended_data import WindowDataset, build_data  # noqa: E402
from fm_stress.extended_experiment import evaluation_indices  # noqa: E402
from fm_stress.extended_models import create_extended_model  # noqa: E402


def fair_crps(x, y):
    s = x.shape[1]
    first = np.abs(x - y[:, None]).mean(axis=1)
    pair = np.abs(x[:, :, None, :] - x[:, None, :, :]).sum(axis=(1, 2)) / (2 * s * (s - 1))
    return (first - pair).mean(axis=1)


def ridge_crps(bundle, norm, ix, target, extension_cfg):
    """Ridge + residual bootstrap baseline of the extension, refitted on the given normalization's model data."""
    mb, stats = model_bundle(bundle, norm)
    tr, va, te = (mb.splits[s] for s in ('train', 'val', 'test'))
    design = lambda ds, rows=slice(None): np.column_stack((ds.context[rows].reshape(len(ds.context[rows]), -1),
                                                           np.ones(len(ds.context[rows]))))
    X, VX, TX = design(tr), design(va), design(te, ix)
    Y, VY = tr.target.reshape(len(tr.target), -1), va.target.reshape(len(va.target), -1)
    P = np.eye(X.shape[1])
    P[-1, -1] = 0                                                      # intercept is not penalized
    fits = [(float(np.mean((VX @ w - VY) ** 2)), a, w) for a in extension_cfg['ridge_alphas']
            for w in [np.linalg.solve(X.T @ X + a * P, X.T @ Y)]]
    _, alpha, w = min(fits, key=lambda r: r[:2])                       # validation selects alpha
    residuals = Y - X @ w
    draw = np.random.default_rng(extension_cfg['evaluation_seed'] + 15).integers(len(Y), size=(len(ix), 32))
    forecasts = (TX @ w)[:, None] + residuals[draw]                    # model units
    mu, sd = stats['test']
    gen = mu[ix][:, None] + sd[ix][:, None] * forecasts.reshape((len(ix), 32) + te.target.shape[1:])
    return float(fair_crps(gen.reshape(len(ix), 32, -1), target).mean()), float(alpha)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', default='results/etth1-normalization')
    out = ROOT / p.parse_args().results
    cfg = json.loads((out / 'config.json').read_text())
    fingerprint = json.loads((out / 'manifest.json').read_text())['fingerprint']
    bundle = build_data(cfg['dataset'])
    test = bundle.splits['test']
    ix = evaluation_indices(test, cfg['n_ensemble_contexts'])
    target = test.target[ix].reshape(len(ix), -1)
    runs, problems = {}, []
    for norm in cfg['normalizations']:
        for prior in cfg['priors']:
            for head in cfg['heads']:
                for seed in cfg['seeds']:
                    name = f'{norm}_{prior}_{head}_seed{seed}'
                    rec = json.loads((out / 'runs' / f'{name}.json').read_text())
                    if rec['status'] != 'complete' or rec['fingerprint'] != fingerprint:
                        problems.append(name + ': status/fingerprint')
                    gen = np.load(out / 'samples' / f'{name}.npz')['generated'].reshape(len(ix), cfg['ensemble_size'], -1)
                    if not np.allclose(fair_crps(gen, target), rec['per_context']['fair_crps'], rtol=1e-6, atol=1e-10):
                        problems.append(name + ': CRPS recomputation differs')
                    runs[norm, prior, head, seed] = rec

    def mean(norm, prior, head, key='fair_crps'):
        return float(np.mean([runs[norm, prior, head, s]['forecast_means'][key] for s in cfg['seeds']]))

    table = {f'{n}|{p}|{h}': {'crps_mean': mean(n, p, h),
                              'crps_per_seed': [runs[n, p, h, s]['forecast_means']['fair_crps'] for s in cfg['seeds']],
                              'coverage_90': mean(n, p, h, 'coverage_90'), 'width_90': mean(n, p, h, 'width_90')}
             for n in cfg['normalizations'] for p in cfg['priors'] for h in cfg['heads']}
    inter = {n: (mean(n, 'gp', 's4') - mean(n, 'gp', 'mlp')) - (mean(n, 'white', 's4') - mean(n, 'white', 'mlp'))
             for n in cfg['normalizations']}
    reference = mean('global', 'gp', 'mlp') >= 1.10 * mean('global', 'white', 'mlp') and \
        mean('global', 'gp', 's4') < mean('global', 'gp', 'mlp')
    mlp_ratio_history = mean('history', 'gp', 'mlp') / mean('history', 'white', 'mlp')
    if not reference:
        verdict = 'uninterpretable (reference pattern not reproduced)'
    elif abs(inter['history']) <= .5 * abs(inter['global']) and abs(mlp_ratio_history - 1) <= .05:
        verdict = 'C2 supported'
    elif abs(inter['history']) >= .8 * abs(inter['global']) and np.sign(inter['history']) == np.sign(inter['global']):
        verdict = 'C2 refuted'
    else:
        verdict = 'inconclusive'

    # Exact level-following of history-normalized forecasts (seed 31 models, shift c = 0.5).
    mb, stats = model_bundle(bundle, 'history')
    follow = {}
    n, s = len(ix), cfg['ensemble_size']
    for prior in cfg['priors']:
        eps = torch.as_tensor(prior_draw(mb, n * s, prior, cfg['evaluation_seed'] + 200), dtype=torch.float32)
        for head in cfg['heads']:
            state = torch.load(out / 'checkpoints' / f'history_{prior}_{head}_seed31.pt', map_location='cpu',
                               weights_only=False)
            model = create_extended_model(**state['options'])
            model.load_state_dict(state['state_dict'])
            model.eval()
            levels = []
            for c in (0.0, 0.5):
                shifted = WindowDataset(test.context + c, test.target + c, test.starts)
                from fm_stress.etth1_normalization import history_normalize
                nds, mu, sd = history_normalize(shifted)
                hist = torch.as_tensor(np.repeat(nds.context[ix], s, axis=0), dtype=torch.float32)
                with torch.no_grad():
                    g = generate(model, eps, hist, cfg['euler_steps'], cfg['generation_batch_size']).reshape(n, s, -1, 1)
                levels.append((mu[ix][:, None] + sd[ix][:, None] * g).mean())
            follow[f'{prior}|{head}'] = float((levels[1] - levels[0]) / 0.5)

    # Supplementary baseline (added after E4's verdict): ridge under each normalization, same contexts and units.
    extension_cfg = json.loads((ROOT / 'configs/extension.json').read_text())
    same_dataset = extension_cfg['datasets']['etth1_uni'] == cfg['dataset']
    ridge = {}
    for norm in cfg['normalizations']:
        crps, alpha = ridge_crps(bundle, norm, ix, target, extension_cfg)
        ridge[norm] = {'fair_crps': crps, 'alpha': alpha}
    if same_dataset:                                                   # the extension's saved ridge is ETTh1-OT only
        saved = json.loads((ROOT / 'results/extension/etth1_uni/baselines.json').read_text())
        saved_global = saved['records']['ridge_residual_bootstrap']['means']['fair_crps']
        if not np.isclose(ridge['global']['fair_crps'], saved_global, rtol=1e-6):
            problems.append('ridge: global refit does not reproduce the saved extension baseline')
        ridge['saved_extension_global'] = saved_global

    summary = {'complete': not problems and len(runs) == 24, 'problems': problems, 'table': table, 'ridge': ridge,
               'interaction': inter, 'reference_reproduced': bool(reference),
               'mlp_gp_over_white_history': mlp_ratio_history, 'verdict': verdict,
               'history_level_following_seed31': follow,
               'best_step_at_last_update': int(sum(r['best_step'] == cfg['steps'] for r in runs.values()))}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('complete', 'problems', 'reference_reproduced', 'interaction',
                                              'mlp_gp_over_white_history', 'verdict', 'history_level_following_seed31',
                                              'ridge')},
                     indent=1))
    for key, v in table.items():
        print('%-22s CRPS %.4f %s  cov90 %.2f  width %.3f' % (key, v['crps_mean'], [round(x, 4) for x in v['crps_per_seed']],
                                                          v['coverage_90'], v['width_90']))


if __name__ == '__main__':
    main()
