#!/usr/bin/env python3
"""Independent array-only audit of all capacity Euler64/128 comparisons."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/followup-solver'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def direct_crps(ensemble, target):
    scores = []
    for paths, truth in zip(ensemble.astype(np.float64), target):
        m, f = paths.shape
        first = np.abs(paths - truth).mean()
        pairs = np.abs(paths[:, None, :] - paths[None, :, :]).sum()
        scores.append(first - pairs / (2 * m * (m - 1) * f))
    return np.array(scores)


def covariance(g, n):
    d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) / g['lengthscale']
    kernel = np.exp(-.5 * d ** 2) if g['kernel'] == 'rbf' else np.exp(-d)
    return g['variance'] * kernel + g['nugget'] * np.eye(n)


def main():
    checks, failures, records, values, rebuilt, pairing = 0, [], [], {}, {}, {}

    def check(ok, label):
        nonlocal checks
        checks += 1
        if not bool(ok):
            failures.append(label)

    def close(a, b, label, tolerance=1e-9):
        a, b = np.asarray(a), np.asarray(b)
        error = float(np.max(np.abs(a - b))) if a.shape == b.shape else float('inf')
        check(np.isfinite(error) and error <= tolerance, label)
        return error

    cfg = read(OUT / 'config.json')
    manifest = read(OUT / 'manifest.json')
    summary = read(OUT / 'summary.json')
    base = ROOT / cfg['capacity_root']
    capacity = read(base / 'config.json')
    cm = read(base / 'manifest.json')
    check(stable({k: v for k, v in manifest.items() if k != 'fingerprint'}) == manifest['fingerprint'], 'solver fingerprint')
    check(stable({k: v for k, v in cm.items() if k != 'fingerprint'}) == cm['fingerprint'], 'capacity fingerprint')
    check(sha(ROOT / 'configs/followup_solver.json') == manifest['config_sha256'], 'frozen solver config hash')
    check(cfg == read(ROOT / 'configs/followup_solver.json'), 'saved solver config value equality')
    check(sha(base / 'config.json') == manifest['capacity_config_sha256'] and stable(capacity) == cm['config_sha256'], 'capacity config hashes')
    check(sha(base / 'manifest.json') == manifest['capacity_manifest_sha256'], 'capacity manifest hash')
    for label, m in [('solver', manifest), ('capacity', cm)]:
        for name, digest in m['source_hashes'].items():
            check(sha(ROOT / name) == digest, label + ' frozen source: ' + name)
    check(cfg['selected_budget'] == 12000 and cfg['coarse_steps'] == 64 and cfg['fine_steps'] == 128, 'fixed budget and steps')
    expected = {(kernel, rep['data_seed'], source, head)
                for kernel, heads in capacity['kernel_heads'].items()
                for rep in capacity['replicates'] for source in capacity['sources'] for head in heads}
    observed = set()
    for row in summary['records']:
        identity = tuple(row['identity'])
        kernel, ds, source, head = identity
        name = f'{kernel}_draw{ds}_{source}_{head}'
        check(identity in expected and identity not in observed, name + ': unique expected identity')
        observed.add(identity)
        check(row == read(OUT / 'runs' / f'{name}.json'), name + ': summary matches record')
        check(row['fingerprint'] == summary['fingerprint'] == manifest['fingerprint'], name + ': fingerprint')
        check(row['budget'] == 12000 and row['contexts'] == 256 and row['members'] == 32 and row['steps'] == [64, 128], name + ': complete design metadata')
        group = base / kernel / 'F64_l8' / f'draw{ds}'
        stem = f'{source}_{head}_budget12000'
        original_path = group / 'runs' / f'{stem}.json'
        original = read(original_path)
        sample_path = group / 'samples' / f'{stem}.npz'
        checkpoint_path = group / 'checkpoints' / f'{stem}.pt'
        check(sha(original_path) == row['inputs']['run_sha256'], name + ': input run hash')
        check(sha(sample_path) == row['inputs']['samples64_sha256'] == original['samples_sha256'], name + ': input sample hash')
        check(sha(checkpoint_path) == row['inputs']['checkpoint_sha256'] == original['checkpoint_sha256'], name + ': checkpoint hash')
        check(row['selected_step'] == original['best_step'] and tuple(original[k] for k in ['kernel', 'data_seed', 'source', 'head']) == identity, name + ': unchanged checkpoint selection')
        check(row['input_hashes'] == original['test_input_hashes'], name + ': input identity metadata')
        arrays_path = ROOT / row['samples_path']
        check(sha(arrays_path) == row['samples128_sha256'], name + ': saved arrays hash')
        with np.load(arrays_path) as z:
            a = {k: z[k] for k in z.files}
        with np.load(sample_path) as z:
            check(np.array_equal(a['ensemble64'], z['ensemble']), name + ': original64 ensemble identity')
            check(np.array_equal(a['target'], z['ensemble_target']), name + ': original target identity')
        g = original['geometry']
        n, m, f, l = 256, 32, g['horizon'], g['lookback']
        check(a['ensemble64'].shape == a['ensemble128'].shape == a['source'].shape == (n, m, f), name + ': ensemble dimensions')
        check(all(np.isfinite(x).all() for x in a.values()), name + ': finite arrays')
        if (kernel, ds) not in rebuilt:
            raw = np.random.default_rng(ds + 2).standard_normal((n, l + f)) @ np.linalg.cholesky(covariance(g, l + f)).T
            past = raw[:, :l]
            loc, scale = past.mean(1, keepdims=True), past.std(1, keepdims=True)
            scale = np.where(scale > g['normalization_threshold'], scale, 1.)
            rebuilt[kernel, ds] = ((past - loc) / scale, (raw[:, l:] - loc) / scale)
        close(a['history'], rebuilt[kernel, ds][0], name + ': independent normalized history', 1e-10)
        close(a['target'], rebuilt[kernel, ds][1], name + ': independent target', 1e-10)
        noise = np.random.default_rng(ds + 1000).standard_normal((n * m, f))
        initial = noise if source == 'white' else noise @ np.linalg.cholesky(covariance(g, f)).T
        close(a['source'], initial.reshape(n, m, f), name + ': independent initial source', 1e-12)
        pair = (kernel, ds, source)
        state = [a[k] for k in ['source', 'history', 'target']]
        if pair in pairing:
            check(all(np.array_equal(x, y) for x, y in zip(pairing[pair], state)), name + ': head pairing')
        else:
            pairing[pair] = state
        scores = [direct_crps(a[f'ensemble{steps}'], a['target']) for steps in [64, 128]]
        errors = []
        for steps, score in zip([64, 128], scores):
            errors.append(close(score, a[f'score{steps}_fair_crps'], name + f': direct CRPS{steps}'))
            close(score.mean(), row['scores']['fair_crps'][f'mean{steps}'], name + f': mean CRPS{steps}')
        delta = scores[1] - scores[0]
        close(delta, a['delta_fair_crps'], name + ': saved per-context delta')
        close(delta.mean(), row['scores']['fair_crps']['delta128_minus64'], name + ': mean delta')
        close(np.abs(delta).max(), row['scores']['fair_crps']['max_context_abs_delta'], name + ': maximum context delta')
        values[identity] = tuple(float(s.mean()) for s in scores)
        records.append({'identity': list(identity), 'max_direct_crps64_error': errors[0], 'max_direct_crps128_error': errors[1], 'contexts': n, 'members': m})
    check(observed == expected and len(records) == 36, 'exact complete 36-model set')
    check(summary['complete'] and summary['models'] == summary['expected_models'] == 36, 'summary completeness')
    contrast_keys, recomputed = set(), []
    for contrast in summary['paired_head_contrasts']:
        kernel, source, head = [contrast[k] for k in ['kernel', 'source', 'head']]
        check((kernel, source, head) not in contrast_keys, 'unique contrast ' + str((kernel, source, head)))
        contrast_keys.add((kernel, source, head))
        draws = []
        for rep in capacity['replicates']:
            ds = rep['data_seed']
            left, right = values[kernel, ds, source, head], values[kernel, ds, source, 's4']
            d64, d128 = left[0] - right[0], left[1] - right[1]
            saved = [d for d in contrast['draws'] if d['draw'] == ds]
            check(len(saved) == 1, 'unique paired draw ' + str((kernel, source, head, ds)))
            if saved:
                close([d64, d128, d128 - d64], [saved[0]['head_minus_s4_64'], saved[0]['head_minus_s4_128'], saved[0]['contrast_change']], 'paired contrast ' + str((kernel, source, head, ds)))
                check(bool(np.sign(d64) != np.sign(d128)) == saved[0]['sign_flipped'], 'draw sign ' + str((kernel, source, head, ds)))
            draws.append((d64, d128))
        means = np.mean(draws, axis=0)
        close(means, [contrast['mean_head_minus_s4_64'], contrast['mean_head_minus_s4_128']], 'mean contrast ' + str((kernel, source, head)))
        close(means[1] - means[0], contrast['mean_contrast_change'], 'mean contrast change ' + str((kernel, source, head)))
        flip = bool(np.sign(means[0]) != np.sign(means[1]))
        check(flip == contrast['mean_sign_flipped'], 'mean sign ' + str((kernel, source, head)))
        recomputed.append({'kernel': kernel, 'source': source, 'head': head, 'mean64': float(means[0]), 'mean128': float(means[1]), 'mean_sign_flipped': flip, 'individual_sign_flips': sum(int(np.sign(x) != np.sign(y)) for x, y in draws)})
    expected_contrasts = {(k, s, h) for k, heads in capacity['kernel_heads'].items() for s in capacity['sources'] for h in heads if h != 's4'}
    check(contrast_keys == expected_contrasts, 'all eight planned head contrasts')
    result = {'complete': observed == expected and not failures, 'status': 'passed' if not failures else 'failed', 'checks': checks, 'failures': failures, 'records': records, 'paired_head_contrasts': recomputed, 'max_individual_mean_crps_shift': max(abs(b - a) for a, b in values.values()), 'mean_contrast_sign_flips': sum(x['mean_sign_flipped'] for x in recomputed), 'individual_draw_contrast_sign_flips': sum(x['individual_sign_flips'] for x in recomputed), 'max_direct_crps_error': max(max(r['max_direct_crps64_error'], r['max_direct_crps128_error']) for r in records), 'manifest_sha256': sha(OUT / 'manifest.json'), 'summary_sha256': sha(OUT / 'summary.json'), 'checker_sha256': sha(Path(__file__)), 'scope': 'Separate-agent CPU-only cross-audit: direct fair CRPS on every stored64/128 ensemble, independent normalized test/source reconstruction, pairing, frozen source/config hashes, original checkpoint/sample hashes and all paired draw/mean contrasts. No integration, model fitting or source changes.'}
    (OUT / 'cross_review_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['complete', 'checks', 'failures', 'max_individual_mean_crps_shift', 'mean_contrast_sign_flips', 'individual_draw_contrast_sign_flips', 'max_direct_crps_error']}))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
