#!/usr/bin/env python3
"""Re-score 39 archived forecast ensembles; no model execution or extraction.

Writes canonical per-run/group results, paired contrasts, context-level scores,
input hashes and CSV tables under results/trajectory-metrics by default.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fm_stress.trajectory_metrics import (paired_context_bootstrap, roughness,
                                         summarize_scores, trajectory_scores)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(data):
    return (json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def read_npz(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def check_existing(scores, record, mean_key):
    """Reproduction check against original fair scores and R1, not a new gate."""
    checked = []
    for key, original in [('fair_crps', 'fair_crps'), ('fair_energy_score', 'fair_energy_score'),
                          ('R1_generated', 'generated_roughness'), ('R1_target', 'target_roughness')]:
        if original not in record.get(mean_key, {}):
            continue
        old = np.asarray(record['per_context'][original], dtype=float)
        if old.shape != scores[key].shape or not np.allclose(scores[key], old, rtol=1e-6, atol=1e-10):
            raise ValueError(f'Archived per-context {original} is not reproduced')
        if not np.isclose(scores[key].mean(), record[mean_key][original], rtol=1e-6, atol=1e-10):
            raise ValueError(f'Archived mean {original} is not reproduced')
        checked.append(original)
    if len(checked) != 4:
        raise ValueError('Missing original consistency fields')
    return checked


def evaluate(root, repetitions=10000):
    files, entries, runs, context_scores = {}, {}, [], {}

    def read_file(path):
        path = Path(path)
        data = path.read_bytes()
        files[str(path.relative_to(root))] = sha256(data)
        return data

    def add_run(row, x, y, record, mean_key, generated=None, path_target=None):
        scores = trajectory_scores(x, y)
        row.update(contexts=len(y), members=x.shape[1], horizon=x.shape[2],
                   means=summarize_scores(scores),
                   reproduced_original_fields=check_existing(scores, record, mean_key))
        if generated is not None:
            if generated.shape != path_target.shape or generated.shape[1] != x.shape[-1]:
                raise ValueError('Single-path forecast/target dimensions differ')
            single = {}
            for k in (1, 2):
                single[f'R{k}_generated'] = roughness(generated, k)
                single[f'R{k}_target'] = roughness(path_target, k)
            row['single_path'] = {'contexts': len(generated), 'means': summarize_scores(single)}
            # Separate namespace makes the 1024-path cohort impossible to confuse
            # with the 256-context proper-score cohort.
            for key, value in single.items():
                context_scores[f"{row['id']}::single_path::{key}"] = value
        runs.append(row)
        for key, value in scores.items():
            context_scores[f"{row['id']}::ensemble::{key}"] = value
        return scores

    main = root/'results/ensemble-evaluation'
    config = json.loads(read_file(main/'config.json'))
    main_manifest = json.loads(read_file(main/'manifest.json'))
    common = read_npz(read_file(main/'common_inputs.npz'))
    if common['target'].shape != (config['n_histories'], 16):
        raise ValueError('Unexpected main target dimensions')
    if (common['history'].shape != (config['n_histories'], 32)
            or common['source'].shape != (config['n_histories'], config['ensemble_size'], 16)):
        raise ValueError('Unexpected main shared history/source dimensions')
    expected_main = {f'gp_{h}_seed{s}' for h in config['heads'] for s in config['seeds']}
    actual_main = {p.stem for p in (main/'samples').glob('gp_*.npz')}
    if actual_main != expected_main or len(expected_main) != 15:
        raise ValueError('Main run set is incomplete or changed')
    for name in sorted(expected_main):
        head, seed = name[3:].rsplit('_seed', 1)
        record = json.loads(read_file(main/'runs'/f'{name}.json'))
        if (record['name'] != name or record['head'] != head or record['seed'] != int(seed)
                or record['fingerprint'] != main_manifest['fingerprint']):
            raise ValueError('Main record identity/protocol fingerprint mismatch')
        x = read_npz(read_file(main/'samples'/f'{name}.npz'))['generated']
        if x.shape != (config['n_histories'], config['ensemble_size'], common['target'].shape[1]):
            raise ValueError('Unexpected main ensemble dimensions')
        add_run({'id': 'main/'+name, 'suite': 'main', 'group': head,
                 'pair_id': 'shared_test_draw/train_seed'+seed, 'seed': int(seed),
                 'data_seed': config['data_seed'], 'source_seed': config['source_seed'],
                 'euler_steps': config['euler_steps'], 'target_input': 'results/ensemble-evaluation/common_inputs.npz'},
                x, common['target'], record, 'means')

    transfer = root/'artifacts/followup-transfer'
    manifest = json.loads(read_file(transfer/'manifest.json'))
    if (manifest['model_count'] != 24 or len(manifest['models']) != 24
            or len({m['id'] for m in manifest['models']}) != 24):
        raise ValueError('Unexpected transfer model count')
    transferred = {(item['archive'], item['path']): item for item in manifest['transferred_files']}
    for item in manifest['archives']:
        actual = file_hash(transfer/item['path'])
        if actual != item['sha256']:
            raise ValueError('Transfer archive SHA256 mismatch: '+item['path'])
        files[str((transfer/item['path']).relative_to(root))] = actual

    with zipfile.ZipFile(transfer/'forecast_samples.zip') as forecasts, zipfile.ZipFile(transfer/'checkpoints.zip') as records:
        def read_entry(archive, name):
            container = forecasts if archive == 'forecast_samples.zip' else records
            data = container.read(name)
            expected = transferred[(archive, name)]
            if len(data) != expected['bytes'] or sha256(data) != expected['sha256']:
                raise ValueError('Transferred entry hash mismatch: '+name)
            entries[archive+'::'+name] = sha256(data)
            return data

        targets = {}
        for shared in manifest['shared_targets']:
            name = shared['path']
            arrays = read_npz(read_entry('forecast_samples.zip', name))
            if (not np.array_equal(arrays['ensemble_target'], arrays['path_target'][:256])
                    or len(arrays['path_history']) != len(arrays['path_target'])):
                raise ValueError('Transferred target prefix/history alignment check failed')
            for key, spec in shared['arrays'].items():
                arr = arrays[key]
                if (list(arr.shape) != spec['shape'] or arr.dtype.str != spec['dtype']
                        or sha256(arr.tobytes(order='C')) != spec['value_bytes_sha256_c_order']):
                    raise ValueError('Transferred target array mismatch: '+name+'::'+key)
            targets[name] = arrays
        for model in manifest['models']:
            name = model['forecast_transfer']['path']
            arrays = read_npz(read_entry('forecast_samples.zip', name))
            for key, spec in model['arrays'].items():
                arr = arrays[key]
                if (list(arr.shape) != spec['shape'] or arr.dtype.str != spec['dtype']
                        or sha256(arr.tobytes(order='C')) != spec['value_bytes_sha256_c_order']):
                    raise ValueError('Transferred forecast array mismatch: '+name+'::'+key)
            record = json.loads(read_entry('checkpoints.zip', model['run_source']['path']))
            if any(record[key] != model[key] for key in ('head', 'seed', 'data_seed', 'source')):
                raise ValueError('Transferred record/model identity mismatch')
            shared = targets[model['shared_targets_path']]
            suite = model['family']
            group = model['head']+'_'+model['label'] if suite == 'followup-label-noise' else model['head']
            add_run({'id': model['id'], 'suite': suite, 'group': group,
                     'pair_id': str(model['data_seed']), 'seed': model['seed'],
                     'data_seed': model['data_seed'], 'source': model['source'],
                     'selected_step': model['selected_step'],
                     'completed_training_budget': model['completed_training_budget'],
                     'euler_steps': model['generation']['steps'], 'target_input': model['shared_targets_path']},
                    arrays['ensemble'], shared['ensemble_target'], record, 'forecast_means',
                    arrays['generated'], shared['path_target'])

    groups = defaultdict(list)
    for row in runs:
        groups[(row['suite'], row['group'])].append(row)
    expected_groups = {'main': {'mlp', 's4', 'whitened_mlp'},
                       'followup-label-noise': {'mlp_stochastic', 'mlp_oracle', 's4_stochastic', 's4_oracle'},
                       'followup-capacity': {'mlp_original', 'mlp_rank_clear', 'mlp_wide', 's4'}}
    expected = {(suite, group) for suite, names in expected_groups.items() for group in names}
    if set(groups) != expected or any(len(rows) != (5 if suite == 'main' else 3)
                                      for (suite, group), rows in groups.items()):
        raise ValueError('Unexpected or incomplete trajectory-analysis groups')
    aggregates = []
    for (suite, group), rows in sorted(groups.items()):
        means = {key: float(np.mean([r['means'][key] for r in rows]))
                 for key in rows[0]['means'] if not key.endswith('_ratio')}
        for k in (1, 2):
            den = means[f'R{k}_target']
            means[f'R{k}_ratio'] = means[f'R{k}_generated']/den if den > 0 else None
        aggregate = {'suite': suite, 'group': group, 'n_runs': len(rows), 'means': means,
                     'run_sd': {key: float(np.std([r['means'][key] for r in rows], ddof=1))
                                for key in rows[0]['means']},
                     'run_min': {key: min(r['means'][key] for r in rows) for key in rows[0]['means']},
                     'run_max': {key: max(r['means'][key] for r in rows) for key in rows[0]['means']}}
        if 'single_path' in rows[0]:
            sm = {key: float(np.mean([r['single_path']['means'][key] for r in rows]))
                  for key in rows[0]['single_path']['means'] if not key.endswith('_ratio')}
            for k in (1, 2):
                sm[f'R{k}_ratio'] = sm[f'R{k}_generated']/sm[f'R{k}_target']
            aggregate['single_path_means'] = sm
        aggregates.append(aggregate)

    planned = {
        'main': [('s4', 'mlp'), ('whitened_mlp', 'mlp')],
        'followup-label-noise': [('mlp_oracle', 'mlp_stochastic'), ('s4_oracle', 's4_stochastic'),
                                 ('s4_stochastic', 'mlp_stochastic'), ('s4_oracle', 'mlp_oracle')],
        'followup-capacity': [('mlp_rank_clear', 'mlp_original'), ('mlp_wide', 'mlp_original'),
                              ('s4', 'mlp_original')],
    }
    contrasts = []
    def ensemble_scores(row):
        prefix = row['id']+'::ensemble::'
        return {key[len(prefix):]: value for key, value in context_scores.items() if key.startswith(prefix)}

    for suite, pairs in planned.items():
        for a, b in pairs:
            ar, br = groups[(suite, a)], groups[(suite, b)]
            aa, bb = {r['pair_id']: r for r in ar}, {r['pair_id']: r for r in br}
            if len(aa) != len(ar) or aa.keys() != bb.keys():
                raise ValueError('Unpaired/duplicate contrast runs')
            paired = []
            for pair_id in sorted(aa):
                ra, rb = aa[pair_id], bb[pair_id]
                if ra['target_input'] != rb['target_input'] or ra['seed'] != rb['seed']:
                    raise ValueError('Contrast target/model seed mismatch')
                per_pair = {'pair_id': pair_id, 'seed': ra['seed'],
                            'differences': {key: ra['means'][key]-rb['means'][key] for key in ra['means']}}
                if suite != 'main':
                    per_pair['conditional_context_bootstrap'] = paired_context_bootstrap(
                        ensemble_scores(ra), ensemble_scores(rb), repetitions=repetitions)
                paired.append(per_pair)
            contrast = {'suite': suite, 'a': a, 'b': b, 'direction': 'a minus b', 'paired_runs': paired,
                        'paired_run_summary': {key: {'mean': float(np.mean([p['differences'][key] for p in paired])),
                                                    'sd': float(np.std([p['differences'][key] for p in paired], ddof=1)),
                                                    'min': min(p['differences'][key] for p in paired),
                                                    'max': max(p['differences'][key] for p in paired)}
                                               for key in paired[0]['differences']}}
            if suite == 'main':
                averaged = []
                for rows in (ar, br):
                    ss = [ensemble_scores(r) for r in rows]
                    averaged.append({key: np.mean([s[key] for s in ss], axis=0) for key in ss[0]})
                contrast['conditional_context_bootstrap'] = paired_context_bootstrap(*averaged, repetitions=repetitions)
                contrast['uncertainty_unit'] = '256 paired contexts, conditional on all five fixed models and saved ensemble draws; seed variability is descriptive'
            else:
                contrast['uncertainty_unit'] = 'three independent paired data-draw/training-seed replicates; within-draw context CIs are conditional diagnostics, no pooled CI'
            contrasts.append(contrast)

    summary = {'schema_version': 1, 'analysis': 'Post-hoc, descriptive, no retraining/inference/new success threshold',
               'definitions': {'Rk': 'mean absolute kth time difference; unit time spacing',
                               'roughness_ratio': 'mean generated Rk / mean target Rk; group ratio recomputed after aggregation',
                               'variogram': {'p': 0.5, 'pairs': 'i<j', 'weights': '(1/(j-i))/sum_(i<j)(1/(j-i))',
                                             'plugin': 'sum w*(|y_i-y_j|^p - mean_m |x_mi-x_mj|^p)^2',
                                             'fair': 'plugin - sum w*var_m(|x_mi-x_mj|^p,ddof=1)/M',
                                             'clipping': 'none; fair estimates may be negative'},
                               'energy': 'Euclidean whole-path units, no division by sqrt(horizon)',
                               'crps': 'marginal absolute-distance CRPS, averaged over horizon'},
               'units': 'Saved history-normalized observation coordinates; compare within suite only',
               'bootstrap': {'seed': 90041, 'repetitions': repetitions, 'interval': 'percentile 95%; exploratory, no multiplicity adjustment'},
               'alignment': 'Transfer ensemble_target exactly equals path_target[:256]; arrays checked against manifest hashes and shapes; all within-pair models share target/history',
               'main_design': 'five training seeds evaluated on one common 256-context test draw and common sources',
               'followup_design': 'three independent data draws paired with three training seeds per group; draw and training seed effects cannot be separated',
               'runs': runs, 'groups': aggregates, 'contrasts': contrasts}
    provenance = {'files_sha256': files, 'zip_entries_sha256': entries,
                  'code_sha256': {str(p.relative_to(root)): file_hash(p) for p in
                                  [Path(__file__), root/'fm_stress/trajectory_metrics.py', root/'docs/trajectory_metrics.md']},
                  'numpy_version': np.__version__, 'python_version': sys.version.split()[0],
                  'forecast_runs': len(runs), 'new_forecasts': 0}
    return summary, context_scores, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT/'results/trajectory-metrics')
    parser.add_argument('--bootstrap-repetitions', type=int, default=10000)
    args = parser.parse_args()
    summary, scores, provenance = evaluate(args.root.resolve(), args.bootstrap_repetitions)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_bytes(json_bytes(summary))
    np.savez_compressed(args.output/'per_context.npz', **scores)
    for kind in ('runs', 'groups'):
        rows = []
        for row in summary[kind]:
            meta = {key: row[key] for key in ('id', 'suite', 'group', 'seed', 'data_seed', 'contexts', 'members', 'horizon', 'n_runs') if key in row}
            rows.append({**meta, **row['means']})
        with (args.output/f'{kind}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (args.output/'paired_contrasts.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['suite', 'a', 'b', 'metric', 'paired_mean_difference', 'paired_sd', 'paired_min', 'paired_max', 'context_ci95_low', 'context_ci95_high'])
        writer.writeheader()
        for contrast in summary['contrasts']:
            for metric, value in contrast['paired_run_summary'].items():
                ci = contrast.get('conditional_context_bootstrap', {}).get(metric, {}).get('context_ci95') or [None, None]
                writer.writerow(dict(suite=contrast['suite'], a=contrast['a'], b=contrast['b'], metric=metric,
                                     paired_mean_difference=value['mean'], paired_sd=value['sd'], paired_min=value['min'], paired_max=value['max'],
                                     context_ci95_low=ci[0], context_ci95_high=ci[1]))
    provenance['output_sha256'] = {p.name: file_hash(p) for p in sorted(args.output.iterdir()) if p.is_file() and p.name != 'provenance.json'}
    (args.output/'provenance.json').write_bytes(json_bytes(provenance))
    print(f"Scored {len(summary['runs'])} existing ensembles; {len(summary['groups'])} groups; {len(summary['contrasts'])} paired contrasts -> {args.output}")


if __name__ == '__main__':
    main()
