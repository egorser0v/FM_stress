#!/usr/bin/env python3
"""Exact head-Jacobian probe on the main study's checkpoints (no training).

For each of the 25 main checkpoints (MLP, S4, whitened MLP x white/GP prior x 5 seeds),
the full learned Jacobian J = dv/dx (16 x 16) is computed by autodiff at points
x_t = (1-t) eps + t y. Here eps is drawn from the model's own prior with a fixed seed,
y are the first 256 test targets, and t is in {0.1, 0.3, 0.5, 0.7, 0.9}.

The exact Jacobian of the ideal field is J* = U S^-1, with
S = t^2 C + (1-t)^2 C_prior and U = t C - (1-t) C_prior, where C is the exact
normalized conditional covariance given the raw history (data.conditional_target,
as in data.conditional_oracle_velocity).

Errors are reported overall and in blocks of the prior's own training-velocity PCA
basis: PC1-2 x PC1-2 ("main") and PC3-16 x PC3-16 ("residual"). The Toeplitz
deviation ||J - T(J)|| / ||J|| is also reported, where T(J) averages each diagonal.

Reading rule, fixed before the first run (the assignment's claim): with the GP prior,
the residual-block relative Jacobian error of MLP (averaged over t) exceeds that of S4
in every seed. Otherwise the claim is not supported.
"""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fm_stress.data import GPConfig, conditional_target, sample_dataset, source_covariance  # noqa: E402
from fm_stress.experiment import CoordinateModel  # noqa: E402
from fm_stress.models import create_model  # noqa: E402

MAIN = ROOT / 'results/main'
TIMES = (0.1, 0.3, 0.5, 0.7, 0.9)
N = 256


def learned_jacobian(model, x, t, h):
    """[N, F, F] Jacobian dv_i/dx_j: one backward pass per output coordinate (outputs are per-window)."""
    x = x.clone().requires_grad_(True)
    v = model(x, t, h)
    rows = [torch.autograd.grad(v[:, i].sum(), x, retain_graph=True)[0] for i in range(v.shape[1])]
    return torch.stack(rows, dim=1).detach().numpy()


def toeplitz_deviation(J):
    F = J.shape[-1]
    T = np.zeros_like(J)
    for k in range(-F + 1, F):
        idx = np.arange(max(0, -k), min(F, F - k))
        mean = J[:, idx, idx + k].mean(axis=1, keepdims=True)
        T[:, idx, idx + k] = mean
    return float((np.linalg.norm(J - T, axis=(1, 2)) / np.linalg.norm(J, axis=(1, 2))).mean())


def main():
    cfg = json.loads((MAIN / 'config.json').read_text())
    geometry = json.loads((MAIN / 'geometry.json').read_text())
    gp = GPConfig(**cfg['data'])
    test = sample_dataset(gp, cfg['n_test'], cfg['data_seed'] + 2)
    mean, covs = conditional_target(gp, test)
    covs, y, h_np = covs[:N], test.target[:N], test.context[:N]
    torch.set_num_threads(4)
    results, rows = {}, []
    for source in ('white', 'gp'):
        cs = source_covariance(gp, source)
        Q = np.asarray(geometry['pca'][source]['components'])           # rows = PCs of training velocity
        eps = np.random.default_rng(424242).standard_normal((N, gp.horizon)) @ np.linalg.cholesky(cs).T
        exact = {}
        for t in TIMES:
            S = t ** 2 * covs + (1 - t) ** 2 * cs
            U = t * covs - (1 - t) * cs
            exact[t] = np.linalg.solve(S.swapaxes(-1, -2), U.swapaxes(-1, -2)).swapaxes(-1, -2)   # U S^-1
        for kind in ('mlp', 's4', 'whitened_mlp'):
            if source == 'white' and kind == 'whitened_mlp':
                continue
            for seed in cfg['seeds']:
                state = torch.load(MAIN / 'checkpoints' / f'{source}_{kind}_seed{seed}.pt', map_location='cpu',
                                   weights_only=False)
                base = create_model('s4' if kind == 's4' else 'mlp', horizon=gp.horizon, lookback=gp.lookback,
                                    **cfg['model'])
                model = CoordinateModel(base)
                model.load_state_dict(state['state_dict'])
                model.eval()
                for t in TIMES:
                    x = torch.as_tensor((1 - t) * eps + t * y, dtype=torch.float32)
                    J = learned_jacobian(model, x, torch.full((N,), t), torch.as_tensor(h_np, dtype=torch.float32))
                    D = J - exact[t]
                    rot = lambda M: Q @ M @ Q.T
                    Dp, Ep = rot(D), rot(exact[t])
                    blk = lambda M, a, b: np.linalg.norm(M[:, a:b, a:b], axis=(1, 2))
                    rows.append({'source': source, 'kind': kind, 'seed': seed, 't': t,
                                 'overall': float((np.linalg.norm(D, axis=(1, 2)) / np.linalg.norm(exact[t], axis=(1, 2))).mean()),
                                 'main_block': float((blk(Dp, 0, 2) / blk(Ep, 0, 2)).mean()),
                                 'residual_block': float((blk(Dp, 2, 16) / blk(Ep, 2, 16)).mean()),
                                 'toeplitz_dev_learned': toeplitz_deviation(J),
                                 'toeplitz_dev_exact': toeplitz_deviation(exact[t])})
                print(source, kind, seed, 'done', flush=True)
    def agg(source, kind, key, t=None):
        sel = [r for r in rows if r['source'] == source and r['kind'] == kind and (t is None or r['t'] == t)]
        per_seed = [np.mean([r[key] for r in sel if r['seed'] == s]) for s in cfg['seeds']]
        return float(np.mean(per_seed)), per_seed
    print('\nmean over t and seeds  (relative error vs the exact Jacobian; Toeplitz deviation)')
    print('%-18s %9s %9s %9s %9s %9s' % ('model', 'overall', 'PC1-2', 'PC3-16', 'toepl.', 'toepl.*'))
    for source in ('white', 'gp'):
        for kind in ('mlp', 's4', 'whitened_mlp'):
            if source == 'white' and kind == 'whitened_mlp':
                continue
            vals = [agg(source, kind, k)[0] for k in ('overall', 'main_block', 'residual_block', 'toeplitz_dev_learned',
                                                       'toeplitz_dev_exact')]
            results[f'{source}|{kind}'] = {k: agg(source, kind, k) for k in ('overall', 'main_block', 'residual_block',
                                                                            'toeplitz_dev_learned')}
            print('%-18s %9.3f %9.3f %9.3f %9.3f %9.3f' % (f'{kind}/{source}', *vals))
    print('\nGP prior, residual-block error by t:')
    for kind in ('mlp', 's4', 'whitened_mlp'):
        print('  %-13s' % kind, '  '.join('t=%.1f %.3f' % (t, agg('gp', kind, 'residual_block', t)[0]) for t in TIMES))
    mlp_seed, s4_seed = agg('gp', 'mlp', 'residual_block')[1], agg('gp', 's4', 'residual_block')[1]
    supported = all(m > s for m, s in zip(mlp_seed, s4_seed))
    print('\nGP residual-block error per seed  MLP', [round(v, 3) for v in mlp_seed], ' S4', [round(v, 3) for v in s4_seed])
    print('assignment claim (MLP worse than S4 in every seed):', 'supported' if supported else 'not supported')
    (ROOT / 'results/diagnostics/exact_jacobian_probe.json').write_text(json.dumps(
        {'rows': rows, 'summary': results, 'claim_supported': supported}, indent=2) + '\n')


if __name__ == '__main__':
    main()
