#!/usr/bin/env python3
"""Separate-agent, CPU-only array/provenance audit; no model integration.

Does not import experiment sampling or scoring implementations. Keeps the
author's separate CPU replay report intact.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/followup-label-solver'
BASE = ROOT / 'results/followup-label-noise'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def crps(ensemble, target):
    """Unbiased finite-ensemble score, direct all-pairs definition."""
    result = []
    for paths, truth in zip(ensemble.astype(np.float64), target):
        m = len(paths)
        first = np.abs(paths - truth).mean()
        second = sum(np.abs(paths[i] - paths[j]).mean()
                     for i in range(m) for j in range(i + 1, m))
        result.append(first - second / (m * (m - 1)))
    return np.array(result)


def covariance(size, data):
    difference = np.arange(size)[:, None] - np.arange(size)[None, :]
    return data['variance'] * np.exp(-0.5 * (difference / data['lengthscale']) ** 2) + data['nugget'] * np.eye(size)


def main():
    torch.set_num_threads(2)
    checks = 0
    failures = []
    max_errors = {}

    def check(ok, label):
        nonlocal checks
        checks += 1
        if not bool(ok):
            failures.append(label)

    def close(actual, expected, label, tolerance=1e-9):
        a, b = np.asarray(actual), np.asarray(expected)
        error = float(np.max(np.abs(a - b))) if a.shape == b.shape else float('inf')
        max_errors[label] = error
        check(np.isfinite(error) and error <= tolerance, label)
        return error

    config = read(OUT / 'config.json')
    manifest = read(OUT / 'manifest.json')
    summary = read(OUT / 'summary.json')
    training = read(BASE / 'config.json')
    analysis = read(ROOT / 'output/followup-label-solver/summary.json')
    check(config == manifest['config'], 'frozen config equality')
    fingerprint = hashlib.sha256(json.dumps({k: v for k, v in manifest.items() if k != 'fingerprint'}, sort_keys=True).encode()).hexdigest()
    check(fingerprint == manifest['fingerprint'], 'manifest fingerprint')
    for name, digest in manifest['inputs_sha256'].items():
        check(sha(ROOT / name) == digest, 'frozen hash: ' + name)
    check(config['steps'] == [64, 128] and config['contexts'] == 256 and config['members'] == 32, 'full fixed solver design')
    expected = {(rep['data_seed'], rep['seed'], source, head, label)
                for rep in training['replicates'] for source in training['sources']
                for head in training['heads'] for label in training['labels']}
    observed = set()
    rebuilt = {}
    pairing = {}
    records = []
    values = {}
    data = training['data']
    nc, ns, horizon, lookback = 256, 32, data['horizon'], data['lookback']
    for row in summary['rows']:
        identity = tuple(row[k] for k in ['data_seed', 'seed', 'source', 'head', 'label'])
        check(identity in expected and identity not in observed, 'unique expected identity: ' + str(identity))
        observed.add(identity)
        ds, seed, source, head, label = identity
        name = f'{head}_{label}_seed{seed}'
        key = f'{ds}_{source}_{name}'
        group = BASE / f'dataset_{ds}' / source
        original = read(group / 'runs' / f'{name}.json')
        check(row == read(OUT / f'{key}.json'), key + ': summary matches record')
        check(row['fingerprint'] == fingerprint, key + ': fingerprint')
        check(all(row[k] == original[k] for k in ['data_seed', 'seed', 'source', 'head', 'label', 'best_step']), key + ': original identity and selection')
        check(sha(OUT / f'{key}.npz') == row['arrays_sha256'], key + ': array hash')
        sample_path = group / 'samples' / f'{name}.npz'
        checkpoint_path = group / 'checkpoints' / f'{name}.pt'
        check(sha(sample_path) == row['original_samples_sha256'] == original['samples_sha256'], key + ': original sample hash')
        check(sha(checkpoint_path) == row['checkpoint_sha256'] == original['checkpoint_sha256'], key + ': checkpoint hash')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        check(checkpoint['step'] == row['best_step'] and checkpoint['fingerprint'] == original['fingerprint'], key + ': checkpoint metadata')
        close(checkpoint['validation'], original['best_validation_oracle_mse'], key + ': selected validation', 1e-12)
        with np.load(OUT / f'{key}.npz') as z:
            fine, initial, history, target = [z[k] for k in ['ensemble128', 'initial', 'history', 'target']]
        with np.load(sample_path) as z:
            coarse, original_initial = z['ensemble'], z['ensemble_source']
        check(fine.shape == coarse.shape == (nc, ns, horizon), key + ': ensemble shape')
        check(all(np.isfinite(x).all() for x in [fine, coarse, initial, history, target]), key + ': finite arrays')
        check(np.array_equal(initial, original_initial), key + ': identical initial states')
        if ds not in rebuilt:
            raw = np.random.default_rng(ds + 2).standard_normal((training['n_test'], lookback + horizon)) @ np.linalg.cholesky(covariance(lookback + horizon, data)).T
            past = raw[:nc, :lookback]
            loc = past.mean(1, keepdims=True)
            scale = past.std(1, keepdims=True)
            scale = np.where(scale > data['normalization_threshold'], scale, 1.)
            rebuilt[ds] = ((past - loc) / scale, (raw[:nc, lookback:] - loc) / scale)
        close(history, rebuilt[ds][0], key + ': independently rebuilt history', 1e-10)
        close(target, rebuilt[ds][1], key + ': independently rebuilt target', 1e-10)
        noise = np.random.default_rng(ds + 3000).standard_normal((nc * ns, horizon))
        eps = noise if source == 'white' else noise @ np.linalg.cholesky(covariance(horizon, data)).T
        close(initial, eps, key + ': independently rebuilt source', 1e-12)
        pair = (ds, source)
        if pair in pairing:
            check(all(np.array_equal(a, b) for a, b in zip(pairing[pair], [initial, history, target])), key + ': head and label pairing')
        else:
            pairing[pair] = [initial, history, target]
        scores64, scores128 = crps(coarse, target), crps(fine, target)
        # The new 64-step replay is not archived separately; original ensembles
        # provide an independent comparison within the documented replay tolerance.
        e64 = close(scores64, row['per_context64']['fair_crps'], key + ': CRPS64', 1e-6)
        e128 = close(scores128, row['per_context128']['fair_crps'], key + ': CRPS128')
        close(scores64.mean(), row['means64']['fair_crps'], key + ': mean64', 1e-6)
        close(scores128.mean(), row['means128']['fair_crps'], key + ': mean128')
        close(scores128.mean() - scores64.mean(), row['score_change']['fair_crps'], key + ': change', 1e-6)
        values[(ds, source, head, label)] = (float(scores64.mean()), float(scores128.mean()))
        records.append({'identity': list(identity), 'max_crps64_error': e64, 'max_crps128_error': e128, 'contexts': nc, 'members': ns})
    check(observed == expected and len(records) == 24, 'complete exact 24-model identity set')
    check(summary['complete'] and summary['expected'] == summary['observed'] == 24, 'summary completeness')
    for cell in analysis['cells']:
        matching = [v for (ds, s, h, l), v in values.items() if (s, h, l) == (cell['source'], cell['head'], cell['label'])]
        check(len(matching) == 3, 'cell has three paired draws: ' + str(cell['source']) + '/' + str(cell['head']) + '/' + str(cell['label']))
        for i, steps in enumerate([64, 128]):
            close(np.mean([v[i] for v in matching]), cell[f'means{steps}']['fair_crps'], 'aggregate ' + str((cell['source'], cell['head'], cell['label'], steps)), 1e-6 if i == 0 else 1e-9)
    contrasts = []
    for row in analysis['paired_label_contrasts']:
        if row['metric'] != 'fair_crps':
            continue
        identity = (row['data_seed'], row['source'], row['head'])
        ordinary, oracle = values[identity + ('stochastic',)], values[identity + ('oracle',)]
        d64, d128 = oracle[0] - ordinary[0], oracle[1] - ordinary[1]
        close([d64, d128], [row['label_difference64'], row['label_difference128']], 'paired label contrast ' + str(identity), 1e-6)
        check(bool(np.sign(d64) == np.sign(d128)) == row['sign_preserved'], 'paired contrast sign ' + str(identity))
        contrasts.append((d64, d128))
    check(len(contrasts) == 12, 'all twelve paired label contrasts')
    close(max(abs(b - a) for a, b in values.values()), analysis['max_abs_individual_crps_change'], 'maximum individual CRPS change', 1e-6)
    close(max(abs(b - a) for a, b in contrasts), analysis['max_abs_label_crps_contrast_change'], 'maximum contrast change', 1e-6)
    sign_changes = sum(np.sign(a) != np.sign(b) for a, b in contrasts)
    check(sign_changes == len(analysis['crps_label_contrast_sign_changes']), 'total sign changes')
    result = {'complete': observed == expected and not failures, 'status': 'passed' if not failures else 'failed', 'checks': checks, 'failures': failures, 'records': records, 'frozen_inputs_unchanged': not any(x.startswith('frozen hash:') for x in failures), 'manifest_sha256': sha(OUT / 'manifest.json'), 'summary_sha256': sha(OUT / 'summary.json'), 'analysis_sha256': sha(ROOT / 'output/followup-label-solver/summary.json'), 'checker_sha256': sha(Path(__file__)), 'max_direct_crps64_error': max(r['max_crps64_error'] for r in records), 'max_direct_crps128_error': max(r['max_crps128_error'] for r in records), 'scope': 'Separate-agent cross-review. Independent NumPy covariance, source and normalized-history reconstruction; direct pairwise fair CRPS for every original64 and saved128 ensemble; hashes, checkpoint metadata and paired summary contrasts. CPU only. No integration or training. Original64 arrays are used because the rerun64 ensembles were not archived separately. This complements, and does not overwrite, the author CPU replay report.'}
    (OUT / 'cross_review_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['complete', 'status', 'checks', 'failures', 'max_direct_crps64_error', 'max_direct_crps128_error']}))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
