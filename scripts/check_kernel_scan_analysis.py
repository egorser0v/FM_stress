#!/usr/bin/env python3
"""Independently recompute the kernel-scan verdict quantities from the run records.

NumPy only; no fm_stress import. It re-derives the GP-source S4/MLP cell ratios
and labels, the axis trends, the OU-RBF contrast, the §5.3 verdict conditions,
the "keep TimeFlow's head" count and the exploratory oracle-excess ratios, then
compares them with output/kernel-scan/summary.json.
"""
from pathlib import Path
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / 'results/kernel-scan', ROOT / 'output/kernel-scan'
GEOMS = ('F16_l8', 'F32_l8', 'F64_l8', 'F16_l4', 'F16_l2')
RATIO = {'F16_l8': 2, 'F32_l8': 4, 'F64_l8': 8, 'F16_l4': 4, 'F16_l2': 8}
AXES = {'horizon': GEOMS[:3], 'lengthscale': ('F16_l8', 'F16_l4', 'F16_l2')}
KERNELS = ('rbf', 'matern52', 'matern32', 'ou')
DRAWS = (307041, 407041, 507041)
END = {'E1': ('residual', .8), 'E2': ('crps', .95)}


def run(k, g, d, s, h):
    return json.loads((RES / k / g / f'draw{d}' / 'runs' / f'{s}_{h}.json').read_text())


def value(r, kind):
    return r['metrics']['residual_mse'] if kind == 'residual' else r['forecast_means']['fair_crps']


def label(rs, thr):
    gm = math.exp(sum(map(math.log, rs)) / len(rs))
    if max(rs) < 1 and gm <= thr:
        return 'num', gm
    if min(rs) > 1 and gm >= 1 / thr:
        return 'den', gm
    return 'match', gm


def sign(v):
    return 's4' if max(v) < 0 else 'mlp' if min(v) > 0 else 'inconsistent'


def main():
    s = json.loads((OUT / 'summary.json').read_text())
    errors, checks = [], 0

    def same(a, b, what):
        nonlocal checks
        checks += 1
        if isinstance(a, float) or isinstance(b, float):
            ok = math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)
        else:
            ok = a == b
        if not ok:
            errors.append(f'{what}: {a} != {b}')

    rho = {}
    for e, (kind, thr) in END.items():
        for k in KERNELS:
            for g in GEOMS:
                rs = [value(run(k, g, d, 'gp', 's4'), kind) / value(run(k, g, d, 'gp', 'mlp'), kind) for d in DRAWS]
                rho[e, k, g] = rs
        for k in KERNELS:
            for axis, ids in AXES.items():
                slopes = []
                for i in range(3):
                    y = np.log([rho[e, k, g][i] for g in ids])
                    slopes.append(float(np.polyfit([1, 2, 3], y, 1)[0]))
                t = s['trends'][f'{e}|{k}|{axis}']
                for a, b in zip(slopes, t['slopes']):
                    same(a, b, f'slope {e} {k} {axis}')
                same(sign(slopes), t['favours'], f'trend label {e} {k} {axis}')
        contrast = [float(np.mean([math.log(rho[e, 'ou', g][i]) - math.log(rho[e, 'rbf', g][i]) for g in GEOMS]))
                    for i in range(3)]
        for a, b in zip(contrast, s['contrasts'][e]['ou_minus_rbf_mean_log_ratio']):
            same(a, b, f'OU-RBF contrast {e}')
        horizon = [k for k in KERNELS if sign([float(np.polyfit([1, 2, 3], np.log([rho[e, k, g][i] for g in AXES['horizon']]), 1)[0])
                                               for i in range(3)]) == 's4']
        s4_cells = [(k, g) for k in KERNELS for g in GEOMS if label(rho[e, k, g], thr)[0] == 'num']
        cond = s['verdicts'][e]['conditions']
        same(len(horizon) >= 3, cond['a_horizon_trend_s4_in_3_of_4'], f'condition a {e}')
        same(sign(contrast) == 's4', cond['b_ou_minus_rbf_favours_s4'], f'condition b {e}')
        same(any(RATIO[g] >= 4 for _, g in s4_cells), cond['c_s4_advantage_at_ratio_ge_4'], f'condition c {e}')
        same(sorted(map(list, s4_cells)), sorted(map(list, s['verdicts'][e]['s4_advantage_cells_gp'])), f'S4 cells {e}')
        # "Keep TimeFlow's head": GM cells with neither an S4 nor a whitened-MLP advantage over the MLP.
        gm_cells = [(k, g) for k in ('matern52', 'matern32', 'ou') for g in ('F32_l8', 'F64_l8')]
        quiet = 0
        for k, g in gm_cells:
            w = [value(run(k, g, d, 'gp', 'whitened_mlp'), kind) / value(run(k, g, d, 'gp', 'mlp'), kind) for d in DRAWS]
            quiet += label(rho[e, k, g], thr)[0] != 'num' and label(w, thr)[0] != 'num'
        if e == 'E1':
            keep = quiet >= 4
        else:
            keep = keep and quiet >= 4
        # Exploratory excess over the oracle.
        for k in KERNELS:
            for g in GEOMS:
                per = []
                for d in DRAWS:
                    o = json.loads((RES / k / g / f'draw{d}' / 'oracle_gp.json').read_text())
                    ref = o['velocity_groups']['residual'] if e == 'E1' else o['forecast_means']['fair_crps']
                    per.append((value(run(k, g, d, 'gp', 's4'), kind) - ref) / (value(run(k, g, d, 'gp', 'mlp'), kind) - ref))
                for a, b in zip(per, s['exploratory_excess_over_oracle']['cells'][f'{k}|{g}|{e}']['s4_over_mlp_excess']):
                    same(a, b, f'excess {e} {k} {g}')
    same(keep, s['recommendation']['flags']['keep_timeflow_head'], 'keep-head flag')
    result = {'complete': not errors, 'checks': checks, 'errors': errors}
    (ROOT / 'results/diagnostics/kernel_scan_analysis_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=1))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
