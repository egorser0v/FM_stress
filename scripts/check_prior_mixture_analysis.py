#!/usr/bin/env python3
"""Audit prior-mixture summary arithmetic without importing its analyzer.

Reconstructs each row from audited run records and saved arrays, then computes
all dataset-specific means, seed spreads and declared paired contrast equations.
Two dataset draws remain two statistical groups; training seeds are paired
within each group. This checker makes no population significance claim.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from check_prior_mixture import (load, sha, stable_hash, close, independent_velocity,
                                 independent_scores, independent_success)

METRICS = ('pc1_mse', 'pc2_mse', 'residual_mse', 'residual_normalized_mse',
           'roughness_ratio', 'fair_crps', 'target_pca_tail_ratio',
           'fair_energy_score', 'fixed_target_residual_mse', 'fixed_target_pc1_mse', 'coverage_90')


def check_structure(actual, expected, label):
    """Recursive numeric comparison with strict field and sequence coverage."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(label+' keys differ')
        for key in expected:
            check_structure(actual[key], expected[key], label+'/'+str(key))
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise AssertionError(label+' list length differs')
        for i, (left, right) in enumerate(zip(actual, expected)):
            check_structure(left, right, label+'/'+str(i))
    elif expected is None or isinstance(expected, (str, bool)):
        if actual != expected:
            raise AssertionError(label+' value differs')
    else:
        close(actual, expected, label, atol=1e-10, rtol=2e-9)


def scalar_summary(values):
    values = [float(x) for x in values]
    n = len(values)
    mean = sum(values)/n
    # Explicit sum-of-squares formula, independent of analyzer's np.std.
    sd = (sum((x-mean)**2 for x in values)/(n-1))**.5 if n > 1 else None
    return dict(mean=mean, seed_sd=sd, min=min(values), max=max(values), n=n)


def expected_row(record, arrays, geometry):
    pca = geometry['sources'][record['source']]['pca']
    base = independent_velocity(arrays['prediction'], arrays['velocity'], pca)
    fixed = independent_velocity(arrays['prediction'], arrays['velocity'], geometry['target_pca'])
    generated, target = arrays['generated'].astype(float), arrays['target'][:len(arrays['generated'])]
    generated_roughness = float(np.abs(np.diff(generated, axis=1)).mean())
    target_roughness = float(np.abs(np.diff(target, axis=1)).mean())
    scores = independent_scores(arrays['ensemble'], arrays['ensemble_target'])
    tpca = geometry['target_pca']
    q, mean = np.asarray(tpca['components']), np.asarray(tpca['mean'])
    generated_pc = (generated-mean)@q.T
    target_pc = (target-mean)@q.T
    tail_ratio = float(np.square(generated_pc[:, 6:]).mean()/np.square(target_pc[:, 6:]).mean())
    row = {key: record[key] for key in ('run', 'data_seed', 'alpha', 'source', 'head', 'seed',
                                      'steps', 'best_step', 'parameters', 'seconds')}
    row.update({key: value for key, value in base.items() if np.isscalar(value)})
    row.update(generated_roughness=generated_roughness, target_roughness=target_roughness)
    row.update({key: float(value.mean()) for key, value in scores.items()
                if key not in ('generated_roughness', 'target_roughness')})
    row.update({'fixed_target_'+key: value for key, value in fixed.items() if np.isscalar(value)})
    rule = independent_success(base['pc1_mse'], base['residual_mse'], generated_roughness, target_roughness)
    row.update(roughness_ratio=rule['roughness_ratio'], literal_success=rule['works'], target_pca_tail_ratio=tail_ratio)
    return row


def build_contrast_specs(cfg):
    """Literal planned equations: coefficient, source covariance weight, head."""
    specs = {}
    def add(kind, alpha, first, second, terms):
        for metric in METRICS:
            specs[(kind, float(alpha), first, second, metric)] = terms
    for alpha in cfg['alphas']:
        for head in ('s4', 'whitened_mlp'):
            add('head_difference', alpha, head, 'mlp', [(1, alpha, head), (-1, alpha, 'mlp')])
            if alpha != 0:
                add('head_by_source_interaction_vs_white', alpha, head, 'mlp',
                    [(1, alpha, head), (-1, alpha, 'mlp'), (-1, 0., head), (1, 0., 'mlp')])
        if alpha != 0:
            for head in ('mlp', 's4', 'whitened_mlp'):
                add('source_difference_vs_white', alpha, head, head, [(1, alpha, head), (-1, 0., head)])
    for first, second in [('balanced_mlp', 'mlp'), ('whitened_mlp', 'whitened_raw_mlp'),
                          ('whitened_raw_mlp', 'mlp'), ('whitened_mlp', 'balanced_mlp')]:
        add('gp_coordinate_loss_control', 1., first, second, [(1, 1., first), (-1, 1., second)])
    add('gp_coordinate_by_loss_interaction', 1., 'whitened_mlp', 'mlp',
        [(1, 1., 'whitened_mlp'), (-1, 1., 'whitened_raw_mlp'), (-1, 1., 'balanced_mlp'), (1, 1., 'mlp')])
    return specs


def verify(results, analysis):
    root, analysis = Path(results), Path(analysis)
    cfg, manifest, audit = [load(root/name) for name in ('config.json', 'manifest.json', 'independent_verification.json')]
    path = analysis/'summary.json'
    summary = load(path)
    if not audit['complete'] or not summary['complete']:
        raise AssertionError('Complete execution audit and analysis required')
    if stable_hash(cfg) != manifest['config_sha256'] or audit['config_sha256'] != stable_hash(cfg) or summary['config_sha256'] != stable_hash(cfg):
        raise AssertionError('Config fingerprint differs')
    if summary['manifest_fingerprint'] != manifest['fingerprint'] or audit['manifest_fingerprint'] != manifest['fingerprint']:
        raise AssertionError('Manifest fingerprint differs')
    if stable_hash({k: v for k, v in manifest.items() if k != 'fingerprint'}) != manifest['fingerprint']:
        raise AssertionError('Manifest self-fingerprint differs')
    if summary.get('config') != cfg:
        raise AssertionError('Analysis embedded config differs')
    if summary['data_seeds'] != cfg['data_seeds'] or summary['seeds'] != cfg['seeds']:
        raise AssertionError('Analysis statistical grouping differs')
    expected_ids = {(d, float(a), h, s) for d in cfg['data_seeds'] for a in cfg['alphas']
                    for h in cfg['heads']+(cfg.get('gp_control_heads', []) if a == 1 else []) for s in cfg['seeds']}
    if summary['verified_runs'] != len(expected_ids) or audit['verified_runs'] != len(expected_ids) or audit['expected_runs'] != len(expected_ids):
        raise AssertionError('Required run count differs')
    paths = sorted(root.glob('dataset_*/runs/*.json'))
    record_hashes = {str(p.relative_to(root)): sha(p) for p in paths}
    if summary['record_sha256'] != record_hashes or audit['record_sha256'] != record_hashes:
        raise AssertionError('Analysis/audit/current record hashes differ')
    expected_geometry_paths = {f'dataset_{d}/geometry.json' for d in cfg['data_seeds']}
    if set(audit['geometry_sha256']) != expected_geometry_paths:
        raise AssertionError('Audit geometry coverage differs')
    for relative, digest in audit['geometry_sha256'].items():
        if sha(root/relative) != digest:
            raise AssertionError('Audited geometry changed '+relative)
    geometry = {d: load(root/f'dataset_{d}/geometry.json') for d in cfg['data_seeds']}
    if 'geometry' in summary:
        geometry_rows = []
        for data_seed in sorted(cfg['data_seeds']):
            geo = geometry[data_seed]
            for source in sorted(geo['sources'].values(), key=lambda v: v['alpha']):
                cov = np.asarray(source['covariance'])
                adjacent_variance = np.diag(cov)[:-1]+np.diag(cov)[1:]-2*np.diag(cov, 1)
                geometry_rows.append(dict(data_seed=data_seed, alpha=source['alpha'],
                    velocity_top2_energy=source['validation_velocity']['top2_energy'],
                    floored_directions=source['floored_directions'],
                    theoretical_source_roughness=float(np.sqrt(2*adjacent_variance/np.pi).mean()),
                    validation_source_roughness=source['validation_source']['roughness'],
                    validation_target_roughness=geo['validation_target']['roughness']))
        check_structure(summary['geometry'], geometry_rows, 'embedded geometry')
    independent_rows, records = {}, {}
    for record_path in paths:
        record = load(record_path)
        identity = (record['data_seed'], float(record['alpha']), record['head'], record['seed'])
        if identity not in expected_ids or identity in records:
            raise AssertionError('Unexpected or duplicate source run')
        dest = record_path.parent.parent
        samples, checkpoint = dest/'samples'/f"{record['run']}.npz", dest/'checkpoints'/f"{record['run']}.pt"
        if sha(samples) != record['samples_sha256'] or sha(checkpoint) != record['checkpoint_sha256']:
            raise AssertionError('Audited result artifacts changed')
        with np.load(samples) as arrays:
            independent_rows[identity] = expected_row(record, arrays, geometry[record['data_seed']])
        records[identity] = record
    if set(records) != expected_ids:
        raise AssertionError('Missing source runs')
    if len(summary['rows']) != len(expected_ids):
        raise AssertionError('Analysis row count differs')
    row_ids = set()
    for row in summary['rows']:
        identity = (row['data_seed'], float(row['alpha']), row['head'], row['seed'])
        if identity in row_ids or identity not in independent_rows:
            raise AssertionError('Unexpected/duplicate summary row')
        row_ids.add(identity)
        check_structure(row, independent_rows[identity], 'row '+str(identity))
    expected_cells = {(a, h) for _, a, h, _ in expected_ids}
    cell_keys, cell_checks = set(), 0
    for cell in summary['cells']:
        key = (float(cell['alpha']), cell['head'])
        if key not in expected_cells or key in cell_keys:
            raise AssertionError('Unexpected/duplicate analysis cell')
        cell_keys.add(key)
        per_draw = []
        for data_seed in sorted(cfg['data_seeds']):
            rows = [independent_rows[(data_seed, key[0], key[1], seed)] for seed in sorted(cfg['seeds'])]
            per_draw.append(dict(data_seed=data_seed, seeds=sorted(cfg['seeds']),
                                 metrics={metric: scalar_summary([row[metric] for row in rows]) for metric in METRICS},
                                 literal_pass_count=sum(row['literal_success'] for row in rows)))
        aggregate = {}
        for metric in METRICS:
            draw_means = [row['metrics'][metric]['mean'] for row in per_draw]
            aggregate[metric] = dict(mean=sum(draw_means)/len(draw_means), draw_means=draw_means,
                                     draw_min=min(draw_means), draw_max=max(draw_means))
        expected = dict(alpha=key[0], head=key[1], runs=len(cfg['data_seeds'])*len(cfg['seeds']),
                        data_draws=len(cfg['data_seeds']), by_draw=per_draw, metrics=aggregate)
        check_structure(cell, expected, 'cell '+str(key))
        cell_checks += len(METRICS)*len(cfg['data_seeds'])
    if cell_keys != expected_cells:
        raise AssertionError('Missing analysis cells')
    specs = build_contrast_specs(cfg)
    observed_specs, seed_checks = set(), 0
    for contrast in summary['contrasts']:
        key = tuple(contrast[k] for k in ('kind', 'alpha', 'first', 'second', 'metric'))
        if key not in specs or key in observed_specs:
            raise AssertionError('Unexpected/duplicate contrast '+str(key))
        observed_specs.add(key)
        per_draw = []
        for data_seed in sorted(cfg['data_seeds']):
            differences = []
            for seed in sorted(cfg['seeds']):
                effect = sum(coef*independent_rows[(data_seed, float(alpha), head, seed)][key[-1]]
                             for coef, alpha, head in specs[key])
                differences.append(dict(seed=seed, difference=effect))
                seed_checks += 1
            per_draw.append(dict(data_seed=data_seed, paired_seed_effects=differences,
                                 **scalar_summary([r['difference'] for r in differences])))
        expected = dict(zip(('kind', 'alpha', 'first', 'second', 'metric'), key))
        expected.update(by_draw=per_draw, equal_draw_mean_difference=sum(r['mean'] for r in per_draw)/len(per_draw))
        check_structure(contrast, expected, 'contrast '+str(key))
    if observed_specs != set(specs):
        raise AssertionError('Missing planned contrasts')
    expected_solver = [{k: r[k] for k in ('data_seed', 'alpha', 'head', 'seed', 'solver_sensitivity')}
                       for r in records.values() if r['solver_sensitivity']]
    check_structure(summary['solver_sensitivity'], expected_solver, 'solver records')
    return dict(status='passed', summary_sha256=sha(path), analysis_summary=str(path),
                manifest_fingerprint=manifest['fingerprint'], config_sha256=stable_hash(cfg),
                rows_verified=len(independent_rows), cells_verified=len(expected_cells),
                dataset_metric_summaries_verified=cell_checks, contrasts_verified=len(specs),
                individual_seed_contrasts_verified=seed_checks,
                record_sha256=record_hashes, geometry_sha256=audit['geometry_sha256'],
                scope='Independent saved-array rows, within-dataset seed mean/SD, equal-weight dataset means, all paired head/source and coordinate/loss equations; no analyzer helpers imported.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results/prior-mixture')
    parser.add_argument('--analysis', default='output/prior-mixture')
    parser.add_argument('--output', default='results/diagnostics/prior_mixture_analysis_verification.json')
    args = parser.parse_args()
    report = verify(args.results, args.analysis)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('record_sha256', 'geometry_sha256')}, indent=2))


if __name__ == '__main__':
    main()
