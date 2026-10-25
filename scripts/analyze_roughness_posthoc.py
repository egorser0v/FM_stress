#!/usr/bin/env python3
"""Post-hoc R1/R2 analysis of archived univariate forecasts; no model execution.

R1/R2 are mean absolute first/second time differences, with unit time spacing.
Scores use the observation coordinates of each saved study. Ratios divide the
mean generated score by the mean observed-target score, not per-path ratios.
Seed ranges describe optimization variability, not confidence intervals.
No pass threshold is introduced and no original experiment record is changed.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def path_scores(paths, order):
    paths = np.asarray(paths, dtype=np.float64)
    if paths.ndim not in (2, 3) or paths.shape[-1] <= order or not np.isfinite(paths).all():
        raise ValueError('Expected finite [context,time] or [context,member,time] paths')
    return np.abs(np.diff(paths, n=order, axis=-1)).mean(axis=-1)


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/roughness-posthoc/summary.json')
    args = parser.parse_args()
    # Known-value checks distinguish a linear trend from alternating jumps.
    assert path_scores([[0, 1, 2, 3, 4]], 2)[0] == 0
    assert path_scores([[0, 1, 0, 1, 0]], 2)[0] == 2
    suites = {'main': 25, 'lengthscale6': 15, 'controlled_followup': 12,
              'ensemble-evaluation': 15}
    result = {'analysis': 'post-hoc; descriptive; no new success flags',
              'definition': 'Rk = mean(abs(diff(path, n=k))), unit time spacing',
              'ratio': 'mean generated Rk / mean observed-target Rk',
              'units': 'Saved normalized observation coordinates; compare within each suite',
              'uncertainty': 'Seed min/max are descriptive, not confidence intervals',
              'script_sha256': fingerprint(Path(__file__)), 'suites': {}}
    for suite, expected in suites.items():
        root = ROOT / 'results' / suite
        ensemble = suite == 'ensemble-evaluation'
        paths = sorted((root / 'samples').glob('gp_*.npz' if ensemble else '*.npz'))
        if len(paths) != expected:
            raise ValueError(f'{suite}: expected {expected} saved runs, got {len(paths)}')
        shared_target = None
        if ensemble:
            with np.load(root / 'common_inputs.npz', allow_pickle=False) as arrays:
                shared_target = arrays['target']
        runs, groups = [], defaultdict(list)
        for path in paths:
            cell, seed = path.stem.rsplit('_seed', 1)
            with np.load(path, allow_pickle=False) as arrays:
                generated = arrays['generated']
                target = shared_target if ensemble else arrays['target']
                if generated.shape[0] != target.shape[0] or generated.shape[-1] != target.shape[-1]:
                    raise ValueError(f'{path}: forecast/target dimensions differ')
                row = {'run': path.stem, 'cell': cell, 'seed': int(seed),
                       'archive': str(path.relative_to(ROOT)), 'sha256': fingerprint(path),
                       'contexts': len(target), 'members': generated.shape[1] if ensemble else 1}
                for order in (1, 2):
                    gs, ts = path_scores(generated, order), path_scores(target, order)
                    gm, tm = float(gs.mean()), float(ts.mean())
                    row[f'R{order}'] = {
                        'generated_mean': gm, 'target_mean': tm,
                        'ratio': gm / tm if tm > 0 else None,
                        'generated_path_quantiles_10_50_90': np.quantile(gs, [.1, .5, .9]).tolist(),
                        'target_path_quantiles_10_50_90': np.quantile(ts, [.1, .5, .9]).tolist()}
                if not ensemble:
                    record = json.loads((root / 'runs' / f'{path.stem}.json').read_text())
                    for key, saved_key in [('generated_mean', 'generated_roughness'),
                                           ('target_mean', 'target_roughness')]:
                        if not np.isclose(row['R1'][key], record['metrics'][saved_key], rtol=1e-6, atol=1e-10):
                            raise ValueError(f'{path}: R1 differs from original {saved_key}')
            runs.append(row)
            groups[cell].append(row)
        aggregates = {}
        for cell, rows in groups.items():
            aggregate = {'n_seeds': len(rows)}
            for order in (1, 2):
                key = f'R{order}'
                ratios = [r[key]['ratio'] for r in rows]
                aggregate[key] = {'ratio_mean': float(np.mean(ratios)),
                                  'ratio_seed_min': min(ratios), 'ratio_seed_max': max(ratios),
                                  'generated_mean': float(np.mean([r[key]['generated_mean'] for r in rows])),
                                  'target_mean': float(np.mean([r[key]['target_mean'] for r in rows]))}
            aggregates[cell] = aggregate
            print(f"{suite:20} {cell:20} R1={aggregate['R1']['ratio_mean']:.4f} "
                  f"R2={aggregate['R2']['ratio_mean']:.4f}")
        result['suites'][suite] = {'runs': runs, 'aggregates': aggregates}
        if ensemble:
            result['suites'][suite]['common_inputs_sha256'] = fingerprint(root / 'common_inputs.npz')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
