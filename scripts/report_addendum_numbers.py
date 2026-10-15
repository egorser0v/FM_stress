"""Recompute every number quoted in docs/final_report_addendum.md from saved arrays.

Read-only with respect to existing results: it uses NumPy only, imports no
fm_stress code, and writes one new file, results/diagnostics/report_addendum.json.
"""
from pathlib import Path
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/diagnostics/report_addendum.json'
T4 = 2.7764451051977987  # Student t quantile 0.975, 4 degrees of freedom


def read(path):
    return json.loads(Path(path).read_text())


def runs(directory):
    return [read(p) for p in sorted((ROOT / directory / 'runs').glob('*.json'))]


def t_interval(d):
    d = np.asarray(d, float)
    half = T4 * d.std(ddof=1) / math.sqrt(len(d))
    return [d.mean() - half, d.mean() + half]


def percentile_bootstrap(d, seed=0, repetitions=10000):
    d = np.asarray(d, float)
    means = d[np.random.default_rng(seed).integers(0, len(d), (repetitions, len(d)))].mean(1)
    return np.percentile(means, [2.5, 97.5]).tolist()


def rbf(size, lengthscale, nugget):
    i = np.arange(size, dtype=float)
    return np.exp(-0.5 * ((i[:, None] - i) / lengthscale) ** 2) + nugget * np.eye(size)


def kernel(size, lengthscale, kind):
    d = np.abs(np.arange(size)[:, None] - np.arange(size)[None, :]) / lengthscale
    k = {'RBF': np.exp(-d ** 2 / 2),
         'Matern52': (1 + math.sqrt(5) * d + 5 * d ** 2 / 3) * np.exp(-math.sqrt(5) * d),
         'Matern32': (1 + math.sqrt(3) * d) * np.exp(-math.sqrt(3) * d),
         'OU': np.exp(-d)}[kind]
    return k + 1e-6 * np.eye(size)


def per_cell(records):
    cells = {}
    for r in records:
        cells.setdefault(f"{r['source']}_{r['head']}", []).append(r)
    return cells


def main():
    out = {}
    main_runs = per_cell(runs('results/main'))
    assert sum(len(v) for v in main_runs.values()) == 25

    # 1. The report's Task 2/3 table.
    table = {}
    for name, cell in sorted(main_runs.items()):
        get = lambda key: np.array([r['metrics'][key] for r in cell])
        ratio = np.array([r['metrics']['success']['roughness_ratio'] for r in cell])
        table[name] = {'pc1_mse': [get('pc1_mse').mean(), get('pc1_mse').std(ddof=1)],
                       'residual_mse': [get('residual_mse').mean(), get('residual_mse').std(ddof=1)],
                       'roughness_ratio': [ratio.mean(), ratio.std(ddof=1)],
                       'roughness_band_passes': int(((ratio >= .8) & (ratio <= 1.2)).sum()),
                       'literal_passes': int(sum(bool(r['metrics']['success']['works']) for r in cell)),
                       'selected_at_last_step': int(sum(r['best_step'] == r['curve'][-1]['step'] for r in cell))}
    out['task2_table'] = table

    # 2. What the residual group (PCs 3..16) is made of.
    composition = {}
    for name in ('gp_mlp', 'gp_s4', 'gp_whitened_mlp'):
        raw = np.mean([r['metrics']['pc_mse'] for r in main_runs[name]], 0)
        norm = np.mean([r['metrics']['pc_normalized_mse'] for r in main_runs[name]], 0)
        reported = np.mean([r['metrics']['residual_normalized_mse'] for r in main_runs[name]])
        assert np.isclose(norm[2:].mean(), reported, rtol=1e-9)
        composition[name] = {'raw_pc3_to_pc6': raw[2:6].tolist(), 'raw_pc7_to_pc16_mean': raw[6:].mean(),
                             'pc3_share_of_raw_residual': raw[2] / raw[2:].sum(),
                             'normalized_residual': reported,
                             'pc7_to_pc16_share_of_normalized_residual': norm[6:].sum() / norm[2:].sum()}
    out['residual_group_composition_main'] = composition
    follow = per_cell(runs('results/controlled_followup'))
    follow_out = {}
    for r in runs('results/controlled_followup'):
        key = f"{r['head']}_{r['loss']}"
        follow_out.setdefault(key, []).append((r['metrics']['residual_normalized_mse'],
                                               np.asarray(r['metrics']['pc_normalized_mse'])))
    out['residual_group_composition_controlled_followup'] = {
        k: {'normalized_residual': float(np.mean([a for a, _ in v])),
            'pc7_to_pc16_share': float(np.mean([p[6:].sum() / p[2:].sum() for _, p in v]))}
        for k, v in sorted(follow_out.items())}
    del follow

    # 3. Eigenvalue tails versus the normalized nugget.
    geometry = read(ROOT / 'results/main/geometry.json')
    eig = {s: np.asarray(geometry['pca'][s]['eigenvalues']) for s in ('target', 'white', 'gp')}
    cfg = read(ROOT / 'configs/main.json')['data']
    window = cfg['lookback'] + cfg['horizon']
    draws = np.random.default_rng(20261012).standard_normal((200000, window)) @ np.linalg.cholesky(
        rbf(window, cfg['lengthscale'], cfg['nugget'])).T
    scale = draws[:, :cfg['lookback']].std(axis=1)
    scale = np.where(scale > cfg.get('normalization_threshold', 1e-2), scale, 1.0)
    inflation = float(np.mean(1 / scale ** 2))
    source_eig = np.sort(np.linalg.eigvalsh(rbf(cfg['horizon'], cfg['lengthscale'], cfg['nugget'])))[::-1]
    out['spectra'] = {'target': eig['target'].tolist(), 'gp_velocity': eig['gp'].tolist(),
                      'white_velocity': eig['white'].tolist(), 'matched_source_kernel': source_eig.tolist(),
                      'mean_inverse_history_variance': inflation,
                      'normalized_nugget_level': cfg['nugget'] * inflation,
                      'target_to_source_variance_ratio_pc1_to_pc4': (eig['target'][:4] / source_eig[:4]).tolist()}

    # 4. Roughness of exactly truncated true targets (seed-0 test windows).
    pca = geometry['pca']['target']
    q, mean = np.asarray(pca['components']), np.asarray(pca['mean'])
    target = np.load(ROOT / 'results/main/samples/gp_mlp_seed0.npz')['target']
    rough = lambda x: np.abs(np.diff(x, axis=1)).mean()
    truncation = {}
    for k in (1, 2, 3, 4):
        z = (target - mean) @ q[:k].T @ q[:k] + mean
        truncation[k] = {'roughness_ratio': rough(z) / rough(target),
                         'variance_lost': ((target - z) ** 2).sum() / ((target - mean) ** 2).sum()}
    out['rank_truncation_of_true_targets'] = truncation

    # 5. Paired GP-source head contrasts and interval resolution.
    by_seed = {name: {r['seed']: r['metrics'] for r in cell} for name, cell in main_runs.items()}
    seeds = sorted(by_seed['gp_mlp'])
    s4_minus_mlp = {}
    for key in ('pc1_mse', 'residual_mse'):
        d = [by_seed['gp_s4'][s][key] - by_seed['gp_mlp'][s][key] for s in seeds]
        s4_minus_mlp[key] = {'per_seed': d, 'mean': float(np.mean(d)), 't95': t_interval(d)}
    distance = lambda m: abs(m['success']['roughness_ratio'] - 1)
    s4_minus_mlp['roughness_distance_from_target'] = {
        'mlp': [distance(by_seed['gp_mlp'][s]) for s in seeds], 's4': [distance(by_seed['gp_s4'][s]) for s in seeds]}
    out['gp_head_contrast_s4_minus_mlp'] = s4_minus_mlp
    d = [by_seed['gp_whitened_mlp'][s]['residual_mse'] - by_seed['gp_mlp'][s]['residual_mse'] for s in seeds]
    out['whitening_residual_interval'] = {'per_seed': d, 'mean': float(np.mean(d)), 't95': t_interval(d),
                                          'percentile_bootstrap95': percentile_bootstrap(d),
                                          'distinct_bootstrap_resamples': math.comb(2 * len(d) - 1, len(d))}

    # 6. Raw-kernel geometry outside the tested point (no normalization, no training).
    scan = []
    for kind in ('RBF', 'Matern52', 'Matern32', 'OU'):
        for ratio in (1, 2, 4, 8):
            e = np.sort(np.linalg.eigvalsh(kernel(16, 16 / ratio, kind)))[::-1]
            scan.append({'kernel': kind, 'F_over_lengthscale': ratio, 'top2_share': e[:2].sum() / e.sum(),
                         'components_above_1e-3_of_first': int((e > 1e-3 * e[0]).sum())})
    out['raw_kernel_geometry_F16'] = scan

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float) + '\n')
    print(json.dumps(out, indent=1, default=lambda v: round(float(v), 6))[:6000])


if __name__ == '__main__':
    main()
