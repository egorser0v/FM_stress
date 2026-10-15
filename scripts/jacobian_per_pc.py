#!/usr/bin/env python3
"""Per-PC learned vs exact Jacobian response on the main study's checkpoints (no training).

For every checkpoint (MLP and S4 x white/GP prior x 5 seeds) and every t in TIMES,
the learned Jacobian J and the exact J* = U S^-1 are rotated into the prior's
training-velocity PCA basis. Reported per seed and t:
  - residual-block (PC3-16) norm ratio ||J|| / ||J*|| and cosine(J, J*);
  - per-PC response gain_k = <J q_k, J* q_k> / ||J* q_k||^2, the projection of the
    learned column response onto the exact one (1 = learned exactly, 0 = no response).
Writes results/diagnostics/jacobian_per_pc.json and output/jacobian/per_pc_gain.png.
"""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from probe_exact_jacobian import N, TIMES, learned_jacobian  # noqa: E402
from fm_stress.data import GPConfig, conditional_target, sample_dataset, source_covariance  # noqa: E402
from fm_stress.experiment import CoordinateModel  # noqa: E402
from fm_stress.models import create_model  # noqa: E402

MAIN = ROOT / 'results/main'


def main():
    cfg = json.loads((MAIN / 'config.json').read_text())
    geo = json.loads((MAIN / 'geometry.json').read_text())
    gp = GPConfig(**cfg['data'])
    test = sample_dataset(gp, cfg['n_test'], cfg['data_seed'] + 2)
    _, covs = conditional_target(gp, test)
    covs, y, h = covs[:N], test.target[:N], test.context[:N]
    torch.set_num_threads(4)
    out = {}
    for source in ('white', 'gp'):
        cs = source_covariance(gp, source)
        Q = np.asarray(geo['pca'][source]['components'])
        eps = np.random.default_rng(424242).standard_normal((N, gp.horizon)) @ np.linalg.cholesky(cs).T
        for kind in ('mlp', 's4'):
            for seed in cfg['seeds']:
                st = torch.load(MAIN / 'checkpoints' / f'{source}_{kind}_seed{seed}.pt', map_location='cpu', weights_only=False)
                m = CoordinateModel(create_model('s4' if kind == 's4' else 'mlp', horizon=16, lookback=32, **cfg['model']))
                m.load_state_dict(st['state_dict'])
                m.eval()
                for t in TIMES:
                    S = t * t * covs + (1 - t) ** 2 * cs
                    U = t * covs - (1 - t) * cs
                    Je = np.linalg.solve(S.swapaxes(-1, -2), U.swapaxes(-1, -2)).swapaxes(-1, -2)
                    J = learned_jacobian(m, torch.as_tensor((1 - t) * eps + t * y, dtype=torch.float32),
                                         torch.full((N,), t), torch.as_tensor(h, dtype=torch.float32))
                    A, B = Q @ J @ Q.T, Q @ Je @ Q.T                      # PC basis
                    a, b = A[:, 2:, 2:], B[:, 2:, 2:]
                    ratio = float((np.linalg.norm(a, axis=(1, 2)) / np.linalg.norm(b, axis=(1, 2))).mean())
                    cos = float(((a * b).sum((1, 2)) / (np.linalg.norm(a, axis=(1, 2)) * np.linalg.norm(b, axis=(1, 2)))).mean())
                    gain = ((A * B).sum(axis=1) / (B * B).sum(axis=1)).mean(axis=0)   # per input PC k (columns)
                    out[f'{source}|{kind}|{seed}|{t}'] = {'ratio': ratio, 'cosine': cos, 'gain_per_pc': gain.tolist()}
                print(source, kind, seed, flush=True)
    (ROOT / 'results/diagnostics/jacobian_per_pc.json').write_text(json.dumps(out, indent=2) + '\n')
    seeds = cfg['seeds']
    print('\nresidual block PC3-16, mean over t; per seed')
    for source in ('white', 'gp'):
        for kind in ('mlp', 's4'):
            r = [np.mean([out[f'{source}|{kind}|{s}|{t}']['ratio'] for t in TIMES]) for s in seeds]
            c = [np.mean([out[f'{source}|{kind}|{s}|{t}']['cosine'] for t in TIMES]) for s in seeds]
            print(f'  {kind}/{source:5s} norm ratio {np.mean(r):.3f} {[round(x, 3) for x in r]}   cosine {np.mean(c):.3f} {[round(x, 3) for x in c]}')
    print('\nGP prior, residual-block norm ratio by t (mean over seeds):')
    for kind in ('mlp', 's4'):
        print(f'  {kind:4s}', '  '.join('t=%.1f %.3f' % (t, np.mean([out[f"gp|{kind}|{s}|{t}"]["ratio"] for s in seeds])) for t in TIMES))
    print('\nGP prior, per-PC gain (mean over seeds and t), PC1..16:')
    for kind in ('mlp', 's4'):
        g = np.mean([out[f'gp|{kind}|{s}|{t}']['gain_per_pc'] for s in seeds for t in TIMES], axis=0)
        print(f'  {kind:4s}', ' '.join('%.2f' % v for v in g))
    figure(out, seeds)


def figure(out, seeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'mlp': '#2a78d6', 's4': '#eb6834'}
    ink, muted, grid = '#1f1f1e', '#6b6a63', '#e4e3dc'
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, source in zip(axes, ('white', 'gp')):
        ax.axhline(1, color=muted, lw=1)
        ax.axhline(0, color=grid, lw=1)
        for kind in ('mlp', 's4'):
            per_seed = np.array([np.mean([out[f'{source}|{kind}|{s}|{t}']['gain_per_pc'] for t in TIMES], axis=0)
                                 for s in seeds])
            x = np.arange(1, 17)
            for row in per_seed:
                ax.plot(x, row, color=colors[kind], lw=0.8, alpha=.3)
            ax.plot(x, per_seed.mean(axis=0), color=colors[kind], lw=2, marker='o', ms=4, label=kind.upper())
        ax.axvspan(2.5, 16.5, color='#f1f0ea', zorder=0)
        ax.set_title(f"{'white' if source == 'white' else 'GP'} prior", fontsize=10, color=ink, loc='left')
        ax.set_xlabel('velocity principal component', fontsize=9, color=muted)
        ax.set_xticks([1, 2, 4, 8, 12, 16])
        ax.set_ylim(-0.3, 1.3)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=muted, labelsize=9)
        ax.grid(axis='y', color=grid, lw=.6)
    axes[0].set_ylabel('learned / exact Jacobian response', fontsize=9, color=muted)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle('How far into the small components each head learns the Jacobian (main study; thin lines = seeds; shaded = PC3-16)',
                 fontsize=10, color=ink, x=.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, .93))
    path = ROOT / 'output/jacobian/per_pc_gain.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)


if __name__ == '__main__':
    main()
