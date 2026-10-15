"""Read-only analysis of the kernel scan, implementing docs/kernel_scan_protocol.md §5.

It refuses to run unless the independent audit (scripts/check_kernel_scan.py) has
passed for the same fingerprint. Nothing here selects, re-weights or excludes a
run. Outputs go to output/kernel-scan/.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AXES = {'horizon': ('F16_l8', 'F32_l8', 'F64_l8'), 'lengthscale': ('F16_l8', 'F16_l4', 'F16_l2')}
GAUSS_MARKOV_CELLS = [(k, g) for k in ('matern52', 'matern32', 'ou') for g in ('F32_l8', 'F64_l8')]
ENDPOINTS = {'E1': 'residual_mse', 'E2': 'fair_crps'}
VELOCITY = ('residual_mse', 'pc1_mse', 'residual_signal', 'residual_tail', 'roughness_distance')
FORECAST = ('fair_crps', 'fair_energy_score')
KERNEL_LABEL = {'rbf': 'RBF', 'matern52': 'Matérn-5/2', 'matern32': 'Matérn-3/2', 'ou': 'OU'}


def read(path):
    return json.loads(Path(path).read_text())


def metric(rec, name):
    if name in ('residual_mse', 'pc1_mse', 'velocity_mse'):
        return rec['metrics'][name]
    if name in ('residual_signal', 'residual_tail'):
        return rec['group_mse'][name]
    if name == 'roughness_distance':
        return abs(rec['metrics']['success']['roughness_ratio'] - 1)
    if name == 'crps_to_oracle':
        return rec['oracle_gap']['fair_crps_ratio']
    return rec['forecast_means'][name]


def material(name):
    return .8 if name in VELOCITY else .95


def load(results, cfg):
    runs = {}
    for r in cfg['replicates']:
        for g in cfg['geometries']:
            for k in cfg['kernels']:
                folder = results / k / g['id'] / f"draw{r['data_seed']}"
                for s in cfg['sources']:
                    for h in cfg['heads']:
                        runs[k, g['id'], r['data_seed'], s, h] = read(folder / 'runs' / f'{s}_{h}.json')
    return runs


def ratios(runs, cfg, name, num, den, kernel, geometry, source):
    out = []
    for r in cfg['replicates']:
        a = metric(runs[kernel, geometry, r['data_seed'], source, num], name)
        b = metric(runs[kernel, geometry, r['data_seed'], source, den], name)
        out.append(None if a is None or b is None or b == 0 else a / b)
    return out


def classify(values, threshold):
    """§5.1: label of numerator vs denominator. 'num' means the numerator head has the advantage."""
    if any(v is None for v in values):
        return {'label': 'not defined', 'geometric_mean': None, 'direction_consistent': False}
    gm = float(np.exp(np.mean(np.log(values))))
    below, above = all(v < 1 for v in values), all(v > 1 for v in values)
    label = 'num' if below and gm <= threshold else 'den' if above and gm >= 1 / threshold else 'match'
    return {'label': label, 'geometric_mean': gm, 'direction_consistent': below or above, 'per_replicate': values}


def slope(values):
    """OLS slope of log ratio on log2(F/l) = 1, 2, 3."""
    y = np.log(values)
    return float(np.sum((np.arange(3) - 1) * (y - y.mean())) / 2)


def sign_label(values):
    return 's4' if all(v < 0 for v in values) else 'mlp' if all(v > 0 for v in values) else 'inconsistent'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', default='results/kernel-scan')
    p.add_argument('--config', default='configs/kernel_scan.json')
    p.add_argument('--output', default='output/kernel-scan')
    a = p.parse_args()
    results, out, cfg = ROOT / a.results, ROOT / a.output, read(ROOT / a.config)
    audit, manifest = read(results / 'independent_verification.json'), read(results / 'manifest.json')
    if not (audit['complete'] and audit['fingerprint'] == manifest['fingerprint'] and audit['verified_runs'] == 360):
        raise RuntimeError('Independent audit has not passed for this result directory')
    runs = load(results, cfg)
    kernels, geometries = cfg['kernels'], [g['id'] for g in cfg['geometries']]
    ratio_of = {g['id']: g['horizon'] / g['lengthscale'] for g in cfg['geometries']}
    out.mkdir(parents=True, exist_ok=True)

    # §5.1 cell contrasts for every metric, source and head pair.
    pairs = {'s4/mlp': ('s4', 'mlp'), 'whitened/mlp': ('whitened_mlp', 'mlp'), 'whitened/s4': ('whitened_mlp', 's4')}
    cells = {}
    for k in kernels:
        for g in geometries:
            for s in cfg['sources']:
                for pair, (num, den) in pairs.items():
                    for name in VELOCITY + FORECAST:
                        cells[k, g, s, pair, name] = classify(ratios(runs, cfg, name, num, den, k, g, s), material(name))

    # §5.2 trends and kernel contrast, GP source, S4/MLP.
    trends, contrasts = {}, {}
    for e, name in ENDPOINTS.items():
        for k in kernels:
            for axis, ids in AXES.items():
                per = [slope([cells[k, g, 'gp', 's4/mlp', name]['per_replicate'][i] for g in ids]) for i in range(3)]
                trends[e, k, axis] = {'slopes': per, 'favours': sign_label(per)}
        per = []
        for i in range(3):
            d = [math.log(cells['ou', g, 'gp', 's4/mlp', name]['per_replicate'][i])
                 - math.log(cells['rbf', g, 'gp', 's4/mlp', name]['per_replicate'][i]) for g in geometries]
            per.append(float(np.mean(d)))
        order = sorted(kernels, key=lambda k: np.mean([math.log(cells[k, g, 'gp', 's4/mlp', name]['geometric_mean'])
                                                       for g in geometries]))
        contrasts[e] = {'ou_minus_rbf_mean_log_ratio': per, 'favours': sign_label(per),
                        'kernels_from_most_s4_favourable': order}

    # §5.3 verdict on H_temporal per endpoint.
    verdicts = {}
    for e, name in ENDPOINTS.items():
        horizon_s4 = [k for k in kernels if trends[e, k, 'horizon']['favours'] == 's4']
        horizon_mlp = [k for k in kernels if trends[e, k, 'horizon']['favours'] == 'mlp']
        length_s4 = [k for k in kernels if trends[e, k, 'lengthscale']['favours'] == 's4']
        s4_cells = [(k, g) for k in kernels for g in geometries if cells[k, g, 'gp', 's4/mlp', name]['label'] == 'num']
        cond = {'a_horizon_trend_s4_in_3_of_4': len(horizon_s4) >= 3,
                'b_ou_minus_rbf_favours_s4': contrasts[e]['favours'] == 's4',
                'c_s4_advantage_at_ratio_ge_4': any(ratio_of[g] >= 4 for _, g in s4_cells)}
        reversed_ = len(horizon_mlp) >= 3 and contrasts[e]['favours'] == 'mlp'
        if all(cond.values()):
            verdict = 'Supported'
        elif reversed_:
            verdict = 'Reversed'
        elif any(cond.values()):
            verdict = 'Partial'
        elif not s4_cells:
            verdict = 'Refuted'
        else:
            verdict = 'Unclassified (S4 advantage only at F/l = 2)'
        attribution = {k: ('F/l effect' if k in horizon_s4 and k in length_s4 else
                           'window length / dimension' if k in horizon_s4 else
                           'per-step roughness / resolution' if k in length_s4 else 'none') for k in kernels}
        verdicts[e] = {'verdict': verdict, 'conditions': cond, 'reversed_condition': reversed_,
                       'horizon_trend_s4': horizon_s4, 'horizon_trend_mlp': horizon_mlp,
                       'lengthscale_trend_s4': length_s4, 's4_advantage_cells_gp': s4_cells,
                       'axis_attribution': attribution}

    # §5.4 labels per kernel x geometry.
    labels = {}
    for k in kernels:
        for g in geometries:
            branches = {read(results / k / g / f"draw{r['data_seed']}" / 'geometry.json')['task1_branch']
                        for r in cfg['replicates']}
            works = {(h, s): [bool(runs[k, g, r['data_seed'], s, h]['metrics']['success']['works'])
                              for r in cfg['replicates']] for h in cfg['heads'] for s in cfg['sources']}
            if not any(any(v) for v in works.values()):
                literal = ['Empty task']
            else:
                literal = []
                if sum(w != g_ for w, g_ in zip(works['mlp', 'white'], works['mlp', 'gp'])) >= 2:
                    literal.append('The source is the knob')
                if sum(s4 and not m for s4, m in zip(works['s4', 'gp'], works['mlp', 'gp'])) >= 2:
                    literal.append('Head, not prior')
                if sum(works['s4', 'white']) >= 2:
                    literal.append('S4 was doing the GP\'s job')
                literal = literal or ['Some cell works; no Task-2 pattern']
            three = {n: cells[k, g, 'gp', 's4/mlp', n]['label'] for n in ('pc1_mse', 'residual_mse', 'roughness_distance')}
            head = []
            if three['pc1_mse'] == 'match' and 'num' in (three['residual_mse'], three['roughness_distance']):
                head.append('Head, not prior')
            white_s4 = 'num' in (cells[k, g, 'white', 's4/mlp', 'residual_mse']['label'],
                                 cells[k, g, 'white', 's4/mlp', 'roughness_distance']['label'])
            gp_s4 = 'num' in (three['residual_mse'], three['roughness_distance'])
            crps_gain = float(np.exp(np.mean([math.log(metric(runs[k, g, r['data_seed'], 'gp', 's4'], 'fair_crps')
                                                       / metric(runs[k, g, r['data_seed'], 'white', 's4'], 'fair_crps'))
                                              for r in cfg['replicates']])))
            if white_s4 and gp_s4 and crps_gain > .95:
                head.append('S4 was doing the GP\'s job')
            if all(v in ('match', 'den') for v in three.values()):
                head.append('Prior was overstated')
            head = head or ['None']
            task3 = {}
            for e, name in ENDPOINTS.items():
                s4_adv = cells[k, g, 'gp', 's4/mlp', name]['label'] == 'num'
                if s4_adv:
                    ws4 = cells[k, g, 'gp', 'whitened/s4', name]
                    closes = ws4['geometric_mean'] <= (1.25 if name in VELOCITY else 1 / .95)
                    task3[e] = ('Whitening closes the gap' if closes else
                                'Third head does not close the gap, S4 still wins' if ws4['label'] == 'den' else
                                'S4 advantage over MLP; whitened MLP neither closes the gap nor loses materially')
                else:
                    wm = cells[k, g, 'gp', 'whitened/mlp', name]['label'] == 'num'
                    task3[e] = 'Whitening branch' if wm else 'Third head and S4 tie raw MLP'
            labels[k, g] = {'task1': sorted(branches), 'literal_task2': literal, 'head_comparison': head,
                            'head_comparison_three_numbers': three, 's4_gp_over_white_crps': crps_gain,
                            'task3': task3, 'assignment_literal_scope': branches == {'contrast_exists'}}

    # Recommendation for the latent Gauss-Markov programme (six cells, GP source).
    def advantage(k, g, pair, name):
        return cells[k, g, 'gp', pair, name]['label'] == 'num'
    still = {e: [c for c in GAUSS_MARKOV_CELLS if labels[c]['task3'][e].startswith('Third head does not close')]
             for e in ENDPOINTS}
    other = {'E1': 'E2', 'E2': 'E1'}
    switch = any(len([c for c in still[e] if cells[c[0], c[1], 'gp', 's4/mlp', ENDPOINTS[other[e]]]['label'] != 'den']) >= 4
                 for e in ENDPOINTS)
    s4_adv_cells = {e: [c for c in GAUSS_MARKOV_CELLS if advantage(*c, 's4/mlp', ENDPOINTS[e])] for e in ENDPOINTS}
    closes = any(len([c for c in s4_adv_cells[e] if labels[c]['task3'][e] == 'Whitening closes the gap']) >= 4
                 for e in ENDPOINTS)
    whitened_best = any(len([c for c in GAUSS_MARKOV_CELLS if advantage(*c, 'whitened/mlp', ENDPOINTS[e])
                             and cells[c[0], c[1], 'gp', 'whitened/s4', ENDPOINTS[e]]['label'] == 'num']) >= 4
                        for e in ENDPOINTS)
    none_adv = all(len([c for c in GAUSS_MARKOV_CELLS if not advantage(*c, 's4/mlp', ENDPOINTS[e])
                        and not advantage(*c, 'whitened/mlp', ENDPOINTS[e])]) >= 4 for e in ENDPOINTS)
    flags = {'switch_to_s4': switch, 'flow_match_after_whitening': closes or whitened_best, 'keep_timeflow_head': none_adv}
    recommendation = next((n for n, v in (('Switch to an S4/SSM-like head', switch),
                                          ('Flow-match after whitening', closes or whitened_best),
                                          ('Keep TimeFlow\'s head', none_adv)) if v), 'No change to the recommendation')

    # Exploratory, added after the primary verdicts were computed: the protocol lists oracle
    # gaps as secondary endpoints but defines no rule for comparing them between heads.
    # Ratio of the excess over the privileged oracle (S4 excess / MLP excess), GP source.
    excess = {}
    for k in kernels:
        for g in geometries:
            for e, name in ENDPOINTS.items():
                per, closeness = [], {}
                for r in cfg['replicates']:
                    d = r['data_seed']
                    oracle = read(results / k / g / f'draw{d}' / 'oracle_gp.json')
                    ref = oracle['velocity_groups']['residual'] if e == 'E1' else oracle['forecast_means']['fair_crps']
                    gaps = {h: metric(runs[k, g, d, 'gp', h], name) - ref for h in ('mlp', 's4')}
                    per.append(gaps['s4'] / gaps['mlp'])
                    for h in ('mlp', 's4'):
                        closeness.setdefault(h, []).append(metric(runs[k, g, d, 'gp', h], name) / ref)
                excess[k, g, e] = {'s4_over_mlp_excess': per,
                                   'direction': sign_label([math.log(v) for v in per]),
                                   'model_over_oracle_mean': {h: float(np.mean(v)) for h, v in closeness.items()}}

    # Convergence and oracle summaries.
    last = {}
    for (k, g, d, s, h), rec in runs.items():
        key = f'{h}@{g}'
        last.setdefault(key, [0, 0])
        last[key][0] += rec['best_step'] == cfg['steps']
        last[key][1] += 1

    summary = {'status': 'complete', 'fingerprint': manifest['fingerprint'], 'audit_checks': audit['checks'],
               'runs': len(runs), 'interpretation_notes': [
                   'Velocity-MSE thresholds (0.8) also apply to roughness distance |R-1|; CRPS thresholds (0.95) to the energy score.',
                   '"Refuted" requires no GP-source S4 advantage; an S4 advantage only at F/l = 2 is reported as Unclassified.',
                   'Recommendation flags are listed; the first true one in protocol table order is reported.'],
               'verdicts': verdicts, 'contrasts': contrasts,
               'trends': {f'{e}|{k}|{axis}': v for (e, k, axis), v in trends.items()},
               'labels': {f'{k}|{g}': v for (k, g), v in labels.items()},
               'recommendation': {'result': recommendation, 'flags': flags,
                                  'cells': [f'{k}|{g}' for k, g in GAUSS_MARKOV_CELLS]},
               'selected_at_last_update': {k: f'{v[0]}/{v[1]}' for k, v in sorted(last.items())},
               'exploratory_excess_over_oracle': {
                   'note': 'Post-hoc and exploratory; not a protocol endpoint. S4 excess / MLP excess over the privileged '
                           'oracle (residual MSE for E1, fair CRPS for E2), GP source, per data draw.',
                   'cells': {f'{k}|{g}|{e}': v for (k, g, e), v in excess.items()}}}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with open(out / 'exploratory_oracle_excess.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['kernel', 'geometry', 'F_over_l', 'endpoint', 'direction', 'r1', 'r2', 'r3',
                    'mlp_over_oracle', 's4_over_oracle'])
        for (k, g, e), v in excess.items():
            w.writerow([k, g, ratio_of[g], e, v['direction'], *v['s4_over_mlp_excess'],
                        v['model_over_oracle_mean']['mlp'], v['model_over_oracle_mean']['s4']])

    with open(out / 'cell_contrasts.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['kernel', 'geometry', 'F_over_l', 'source', 'pair', 'metric', 'label', 'geometric_mean',
                    'direction_consistent', 'r1', 'r2', 'r3'])
        for (k, g, s, pair, name), c in cells.items():
            label = {'num': pair.split('/')[0] + ' advantage', 'den': pair.split('/')[1] + ' advantage'}.get(c['label'], c['label'])
            w.writerow([k, g, ratio_of[g], s, pair, name, label, c['geometric_mean'], c['direction_consistent'],
                        *(c.get('per_replicate') or [None] * 3)])
    with open(out / 'runs.csv', 'w', newline='') as f:
        w = csv.writer(f)
        cols = ['pc1_mse', 'residual_mse', 'residual_signal', 'residual_tail', 'roughness_distance',
                'fair_crps', 'fair_energy_score', 'coverage_90', 'crps_to_oracle']
        w.writerow(['kernel', 'geometry', 'data_seed', 'source', 'head', 'best_step', 'works', *cols])
        for (k, g, d, s, h), rec in runs.items():
            w.writerow([k, g, d, s, h, rec['best_step'], rec['metrics']['success']['works'], *[metric(rec, c) for c in cols]])

    figure(cells, kernels, out / 's4_over_mlp_gp.png')
    excess_figure(excess, kernels, out / 'exploratory_oracle_excess.png')
    print(json.dumps({'verdicts': {e: v['verdict'] for e, v in verdicts.items()}, 'recommendation': recommendation},
                     indent=1))


def panels(series, kernels, path, title, band, ylabel, ticks):
    """2x2 small multiples: rows = endpoints, columns = axes; series[(e, k, axis)] -> (means[3], draws[3x3])."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter
    # Validated default categorical slots 1-4 (dataviz reference palette, light surface).
    colors = dict(zip(kernels, ('#2a78d6', '#eb6834', '#1baf7a', '#eda100')))
    ink, muted, grid = '#1f1f1e', '#6b6a63', '#e4e3dc'
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), sharex=True)
    for row, e in enumerate(ENDPOINTS):
        for col, axis in enumerate(AXES):
            ax = axes[row, col]
            lo, hi = band[e]
            ax.axhspan(lo, hi, color='#f1f0ea', zorder=0)
            ax.axhline(1, color=muted, lw=1, zorder=1)
            for k in kernels:
                means, draws = series[e, k, axis]
                x = np.arange(1, 4)
                for i in range(draws.shape[1]):
                    ax.plot(x, draws[:, i], color=colors[k], lw=0, marker='o', ms=3.5, alpha=.4, zorder=2)
                ax.plot(x, means, color=colors[k], lw=2, marker='o', ms=8, mec='white', mew=2, zorder=3,
                        label=KERNEL_LABEL[k])
            ax.set_yscale('log')
            ax.set_yticks(ticks[e])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xlim(.8, 3.2)
            ax.set_xticks([1, 2, 3], ['2', '4', '8'])
            ax.grid(axis='y', color=grid, lw=.8)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=muted, labelsize=9)
            ax.set_title(f"{'Residual-PC velocity MSE (E1)' if e == 'E1' else 'Fair CRPS (E2)'} · "
                         f"{'horizon axis (F = 16/32/64, ℓ = 8)' if axis == 'horizon' else 'lengthscale axis (ℓ = 8/4/2, F = 16)'}",
                         fontsize=9.5, color=ink, loc='left')
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9, color=muted)
            if row == 1:
                ax.set_xlabel('F / ℓ', fontsize=9, color=muted)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper right', ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(.99, .935))
    fig.suptitle(title, fontsize=11, color=ink, x=.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, .9))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure(cells, kernels, path):
    series = {}
    for e, name in ENDPOINTS.items():
        for k in kernels:
            for axis, ids in AXES.items():
                series[e, k, axis] = ([cells[k, g, 'gp', 's4/mlp', name]['geometric_mean'] for g in ids],
                                      np.array([cells[k, g, 'gp', 's4/mlp', name]['per_replicate'] for g in ids]))
    band = {e: (material(n), 1 / material(n)) for e, n in ENDPOINTS.items()}
    panels(series, kernels, path,
           'Protocol endpoints, GP source: S4 / MLP ratio\n'
           'lines = geometric mean of 3 data draws; dots = draws; shaded = "match" zone (no material difference)',
           band, 'S4 / MLP  (below 1 = S4 better)', {'E1': [.8, .9, 1, 1.1, 1.2], 'E2': [.95, .975, 1, 1.025, 1.05]})


def excess_figure(excess, kernels, path):
    series = {}
    for e in ENDPOINTS:
        for k in kernels:
            for axis, ids in AXES.items():
                draws = np.array([excess[k, g, e]['s4_over_mlp_excess'] for g in ids])
                series[e, k, axis] = (np.exp(np.log(draws).mean(axis=1)), draws)
    panels(series, kernels, path,
           'Exploratory (post hoc), GP source: S4 excess / MLP excess over the privileged oracle (below 1 = S4 better)\n'
           'only the reducible part of each error is compared; not a protocol endpoint',
           {e: (1, 1) for e in ENDPOINTS}, 'S4 excess / MLP excess',
           {'E1': [.5, .7, 1, 1.4, 2], 'E2': [.7, .85, 1, 1.15]})


if __name__ == '__main__':
    main()
