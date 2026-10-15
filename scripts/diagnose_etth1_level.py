#!/usr/bin/env python3
"""Exploratory diagnostics E0-E2 for the ETTh1-OT level hypothesis
(docs/etth1_level_hypothesis.tex, section 6). No training; saved runs only.

Reading rules, fixed before the first run:
  E0 (level shift). Supports (C3) if at least 20% of the 64 evaluated test contexts
     have a history level outside the central 95% (2.5-97.5 percentiles) of the
     training history levels.
  E1 (attribution). Supports H if the per-context CRPS excess of MLP+GP over S4+GP
     (seed means) has a positive Spearman correlation with the history level's
     distance outside that training range, and the mean excess in out-of-range
     contexts is larger than in in-range contexts. No p-values are computed.
  E2 (probe). Shift every evaluated test window (history and target) by c in
     train-standardized units, regenerate with the original source draws and
     solver, and record mean CRPS. Supports (C4) if, at the largest |c|, the CRPS
     growth ratio CRPS(c)/CRPS(0) of MLP+GP exceeds both S4+GP and MLP+white in
     each of the three seeds.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fm_stress.extended_data import build_data, sample_source  # noqa: E402
from fm_stress.extended_experiment import evaluation_indices  # noqa: E402
from fm_stress.extended_metrics import forecast_scores  # noqa: E402
from fm_stress.extended_models import create_extended_model  # noqa: E402
from fm_stress.experiment import generate  # noqa: E402

RUNS = ROOT / 'results/extension/etth1_uni'
SHIFTS = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
SEEDS = (31, 32, 33)
CELLS = [('mlp', 'white'), ('mlp', 'gp'), ('s4', 'white'), ('s4', 'gp')]


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    cfg = json.loads((ROOT / 'configs/extension.json').read_text())
    bundle = build_data(cfg['datasets']['etth1_uni'])
    meta = bundle.metadata
    assert meta['split_sizes'] == json.loads((RUNS / 'data_metadata.json').read_text())['split_sizes']
    train, val, test = (bundle.splits[s] for s in ('train', 'val', 'test'))
    a, b = meta['scaler_mean'][0], meta['scaler_scale'][0]
    level = {s: bundle.splits[s].context[..., 0].mean(axis=1) for s in ('train', 'val', 'test')}
    future = {s: bundle.splits[s].target[..., 0].mean(axis=1) for s in ('train', 'val', 'test')}
    lo, hi = np.percentile(level['train'], [2.5, 97.5])
    ix = evaluation_indices(test, cfg['n_ensemble_contexts'])
    ev = level['test'][ix]
    outside = (ev < lo) | (ev > hi)
    distance = np.maximum(0, np.maximum(ev - hi, lo - ev))
    out = {'E0': {
        'raw_units': {'scaler_mean_degC': a, 'scaler_scale_degC': b},
        'history_level_quantiles': {s: np.percentile(v, [0, 2.5, 25, 50, 75, 97.5, 100]).round(3).tolist()
                                    for s, v in level.items()},
        'future_level_quantiles': {s: np.percentile(v, [0, 2.5, 50, 97.5, 100]).round(3).tolist()
                                   for s, v in future.items()},
        'train_central_95': [float(lo), float(hi)],
        'evaluated_test_contexts': int(len(ix)),
        'fraction_outside_train_central_95': float(outside.mean()),
        'fraction_outside_train_minmax': float(((ev < level['train'].min()) | (ev > level['train'].max())).mean()),
        'supports_C3': bool(outside.mean() >= .2)}}

    # E1: per-context attribution with the saved CRPS arrays.
    crps = {(h, s): np.mean([json.loads((RUNS / 'runs' / f'{s}_{h}_seed{d}.json').read_text())['per_context']['fair_crps']
                             for d in SEEDS], axis=0) for h, s in CELLS}
    excess = crps['mlp', 'gp'] - crps['s4', 'gp']
    excess_src = crps['mlp', 'gp'] - crps['mlp', 'white']
    out['E1'] = {
        'spearman_excess_vs_distance': spearman(excess, distance) if distance.any() else None,
        'spearman_excess_vs_level': spearman(excess, ev),
        'spearman_mlp_gp_minus_white_vs_level': spearman(excess_src, ev),
        'mean_excess_outside': float(excess[outside].mean()) if outside.any() else None,
        'mean_excess_inside': float(excess[~outside].mean()),
        'n_outside': int(outside.sum())}
    out['E1']['supports_H'] = bool(outside.any() and out['E1']['spearman_excess_vs_distance'] is not None
                                   and out['E1']['spearman_excess_vs_distance'] > 0
                                   and out['E1']['mean_excess_outside'] > out['E1']['mean_excess_inside'])

    # E2: level-shift probe of the saved checkpoints (CPU).
    torch.set_num_threads(4)
    n, s_ = len(ix), cfg['ensemble_size']
    probe = {}
    for h, s in CELLS:
        eps = sample_source(bundle, n * s_, s, cfg['evaluation_seed'] + 200)
        for d in SEEDS:
            rec = json.loads((RUNS / 'runs' / f'{s}_{h}_seed{d}.json').read_text())
            model = create_extended_model(**rec['model_options'])
            model.load_state_dict(torch.load(RUNS / 'checkpoints' / f'{s}_{h}_seed{d}.pt', map_location='cpu',
                                             weights_only=False)['state_dict'])
            model.eval()
            row = {}
            for c in SHIFTS:
                hist = np.repeat(test.context[ix] + c, s_, axis=0)
                with torch.no_grad():
                    gen = generate(model, torch.as_tensor(eps, dtype=torch.float32),
                                   torch.as_tensor(hist, dtype=torch.float32), cfg['euler_steps'],
                                   cfg['generation_batch_size']).reshape(n, s_, *test.target.shape[1:])
                sc = forecast_scores(gen, test.target[ix] + c)
                bias = float((gen.mean(axis=(1, 2, 3)) - (test.target[ix] + c).mean(axis=(1, 2))).mean())
                row[str(c)] = {'crps': float(sc['fair_crps'].mean()), 'level_bias': bias}
            row['saved_crps_c0'] = rec['forecast_means']['fair_crps']
            probe[f'{h}_{s}_seed{d}'] = row
            print(h, s, d, {k: round(v['crps'], 4) for k, v in row.items() if k != 'saved_crps_c0'},
                  'saved', round(row['saved_crps_c0'], 4), flush=True)
    growth = {}
    for c in (min(SHIFTS), max(SHIFTS)):
        growth[str(c)] = {f'{h}_{s}': [probe[f'{h}_{s}_seed{d}'][str(c)]['crps'] / probe[f'{h}_{s}_seed{d}']['0.0']['crps']
                                       for d in SEEDS] for h, s in CELLS}
    supports = all(all(g['mlp_gp'][i] > max(g['s4_gp'][i], g['mlp_white'][i]) for i in range(3)) for g in growth.values())
    out['E2'] = {'shifts': SHIFTS, 'runs': probe, 'crps_growth_at_extreme_shifts': growth, 'supports_C4': bool(supports)}
    path = ROOT / 'results/diagnostics/etth1_level_diagnostics.json'
    path.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'E0': out['E0'], 'E1': out['E1'],
                      'E2_growth': {k: {kk: [round(x, 3) for x in vv] for kk, vv in v.items()} for k, v in growth.items()},
                      'E2_supports_C4': supports}, indent=1))


if __name__ == '__main__':
    main()
