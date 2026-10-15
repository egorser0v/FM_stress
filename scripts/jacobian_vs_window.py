#!/usr/bin/env python3
"""Learned vs exact head Jacobian as a function of window length F (kernel scan; no training).

Question: does S4's advantage over the AdaLN MLP in the residual-block Jacobian grow with
the window length F? (Weight sharing across positions should matter more for longer patches.)

Data: the audited kernel-scan checkpoints (results/kernel-scan). Primary: RBF kernel, horizon
axis F16_l8 / F32_l8 / F64_l8 (F = 16 / 32 / 64, lookback 2F, l = 8), three data draws
(307041, 407041, 507041), white and GP prior, heads MLP and S4. Secondary (reported, never
used for the verdict): the same for the OU kernel.

For every (kernel, geometry, draw, prior, head) and t in {0.1, 0.3, 0.5, 0.7, 0.9}:
  - test windows: kernel_scan.sample_windows(geometry, n_test, data_seed + 2), first N = 256;
  - x_t = (1-t) eps + t y, eps from the model's own prior with a fixed seed (424242),
    y the test targets, h the test contexts;
  - learned Jacobian J = dv/dx by autodiff (probe_exact_jacobian.learned_jacobian);
  - exact Jacobian J* = U S^-1, S = t^2 C_n + (1-t)^2 C_prior, U = t C_n - (1-t) C_prior,
    C_n = residual / scale_n^2 from kernel_scan.conditional_moments (the oracle's field).
Both are rotated into the prior's training-velocity PCA basis Q (rows of geometry.json
sources[prior].pca.components): A = Q J Q^T, B = Q J* Q^T. Per window we compute the norm
ratio ||A_blk|| / ||B_blk|| and the cosine <A_blk, B_blk> / (||A_blk|| ||B_blk||) for
  residual        PCs 3..F,
  signal residual PCs 3..m (m = signal_pcs from geometry.json, via its pc_groups),
  tail            PCs m+1..F (null if empty),
  main            PCs 1..2 (extra context only),
then average over the N windows and over t.

PRE-DECLARED READING RULE (fixed before the first run of this script):
For the RBF kernel with the GP prior, define per draw
    R(F) = norm_ratio_S4(F) / norm_ratio_MLP(F)   on the residual block (PCs 3..F).
"S4's advantage grows with F" is
  SUPPORTED    if R(F64) > R(F16) in all 3 draws AND R(F64) > 1 in all 3 draws;
  REFUTED      if R(F64) <= R(F16) in at least 2 of 3 draws;
  INCONCLUSIVE otherwise.
R(F32) is also reported. The signal-residual / tail split, the cosines, the white prior and
the OU kernel are descriptive and do not enter the verdict.

Diagnostics added after the first run (descriptive only; the rule above is unchanged):
  - numeric comparison of the regenerated test windows with the saved path_target /
    path_history arrays (bitwise split hashes differ across platforms, as documented in
    scripts/check_kernel_scan.py, so the hash flags are kept only as 'bitwise' fields);
  - numerical rank of the learned Jacobian (singular values > 1e-5 x the largest) and the
    MLP's structural cap width - 2 (final LayerNorm without affine removes two directions);
  - energy shares of the signal x signal, tail x tail and signal x tail cross blocks in the
    residual block, for the exact and the learned Jacobian;
  - R on the signal-residual block, per draw.

Writes results/diagnostics/jacobian_vs_window.json and output/jacobian/vs_window.png.
`--figure-only` redraws the figure from the saved JSON.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from probe_exact_jacobian import learned_jacobian  # noqa: E402
from fm_stress import kernel_scan as ks  # noqa: E402
from fm_stress.experiment import CoordinateModel  # noqa: E402
from fm_stress.models import create_model  # noqa: E402

SCAN = ROOT / 'results/kernel-scan'
OUT_JSON = ROOT / 'results/diagnostics/jacobian_vs_window.json'
OUT_FIG = ROOT / 'output/jacobian/vs_window.png'
KERNELS = ('rbf', 'ou')                    # rbf primary, ou secondary
GEOMETRIES = ('F16_l8', 'F32_l8', 'F64_l8')
SOURCES = ('white', 'gp')
HEADS = ('mlp', 's4')
TIMES = (0.1, 0.3, 0.5, 0.7, 0.9)
N = 256
EPS_SEED = 424242
N_PRED_CHECK = 1024
BLOCK_NAMES = ('residual', 'signal_residual', 'tail', 'main')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_jacobian(covs, cs, t):
    """[N,F,F] U S^-1 in float64, plus the relative solve residual ||J S - U|| / ||U||."""
    S = t ** 2 * covs + (1 - t) ** 2 * cs
    U = t * covs - (1 - t) * cs
    J = np.linalg.solve(S.swapaxes(-1, -2), U.swapaxes(-1, -2)).swapaxes(-1, -2)
    resid = np.linalg.norm(J @ S - U, axis=(1, 2)) / np.linalg.norm(U, axis=(1, 2))
    return J, float(resid.max())


def block_stats(A, B, index):
    """Per-window norm ratio and cosine on the square block index x index (None if empty)."""
    if not index:
        return None
    ix = np.asarray(index)
    a, b = A[:, ix[:, None], ix[None, :]], B[:, ix[:, None], ix[None, :]]
    na, nb = np.linalg.norm(a, axis=(1, 2)), np.linalg.norm(b, axis=(1, 2))
    ratio = na / nb
    cos = (a * b).sum(axis=(1, 2)) / (na * nb)
    return {'norm_ratio': float(ratio.mean()), 'norm_ratio_median': float(np.median(ratio)),
            'cosine': float(cos.mean()), 'learned_sq_norm': float(np.mean(na ** 2)),
            'exact_sq_norm': float(np.mean(nb ** 2))}


def energy_shares(M, signal, tail):
    """Mean over windows of the signal/tail/cross share of the residual-block squared norm."""
    if not tail:
        return None
    s, tl = np.asarray(signal), np.asarray(tail)
    r = np.concatenate([s, tl])
    e = lambda a, b: (M[:, a[:, None], b[None, :]] ** 2).sum(axis=(1, 2))
    total = e(r, r)
    return {'signal': float(np.mean(e(s, s) / total)), 'tail': float(np.mean(e(tl, tl) / total)),
            'cross': float(np.mean((e(s, tl) + e(tl, s)) / total))}


def residual_numerics(geometry):
    """Two independent float64 routes to the conditional covariance; returns diagnostics."""
    k, l = geometry.covariance(), geometry.lookback
    a = np.linalg.solve(k[:l, :l], k[:l, l:])
    r1 = k[l:, l:] - k[l:, :l] @ a
    r1 = (r1 + r1.T) / 2
    p = np.linalg.inv(k)[l:, l:]
    r2 = np.linalg.inv((p + p.T) / 2)
    r2 = (r2 + r2.T) / 2
    e1, e2 = np.linalg.eigvalsh(r1), np.linalg.eigvalsh(r2)
    return {'joint_cov_condition': float(np.linalg.cond(k)),
            'residual_min_eig': float(e1.min()), 'residual_max_eig': float(e1.max()),
            'residual_min_eig_alt_route': float(e2.min()),
            'eig_rel_diff_max': float(np.max(np.abs(e1 - e2) / np.abs(e2))),
            'nugget': geometry.nugget}


def run_group(cfg, kernel, gid, data_seed):
    gdir = ks.group_dir(SCAN, kernel, gid, data_seed)
    rec = json.loads((gdir / 'geometry.json').read_text())
    geometry = ks.make_geometry(cfg, kernel, gid)
    F, L = geometry.horizon, geometry.lookback
    test = ks.sample_windows(geometry, cfg['n_test'], data_seed + ks.SPLIT_OFFSETS['test'])
    _, residual = ks.conditional_moments(geometry, test)
    scale = np.asarray(test.scale[:N], dtype=np.float64)
    covs = residual[None] / scale.reshape(-1, 1, 1) ** 2
    y, h = test.target[:N], test.context[:N]
    groups = rec['pc_groups']
    index = {'residual': groups['residual'], 'signal_residual': groups['residual_signal'],
             'tail': groups['residual_tail'], 'main': [0, 1]}
    numerics = residual_numerics(geometry)
    numerics['scale_min_first_N'] = float(scale.min())
    out = []
    for source in SOURCES:
        cs = ks.source_covariance(geometry, source)
        Q = np.asarray(rec['sources'][source]['pca']['components'], dtype=np.float64)
        q_orth = float(np.abs(Q @ Q.T - np.eye(F)).max())
        eps = np.random.default_rng(EPS_SEED).standard_normal((N, F)) @ np.linalg.cholesky(cs).T
        exact, solve_resid = {}, {}
        for t in TIMES:
            exact[t], solve_resid[t] = exact_jacobian(covs, cs, t)
        pairs = ks.fixed_pairs(geometry, test, source, data_seed, 'test')
        for head in HEADS:
            run = f'{source}_{head}'
            record = json.loads((gdir / 'runs' / f'{run}.json').read_text())
            ckpt = gdir / 'checkpoints' / f'{run}.pt'
            # Saved forecast archives are optional: the slim repository copy omits them. Without them the
            # comparisons against saved arrays are skipped and recorded as None; the Jacobians do not use them.
            sample_path = gdir / 'samples' / f'{run}.npz'
            samples = np.load(sample_path) if sample_path.exists() else None
            checks = {'checkpoint_sha256_match': sha256(ckpt) == record['checkpoint_sha256'],
                      'test_target_max_abs_diff_vs_saved': None, 'test_context_max_abs_diff_vs_saved': None,
                      'bitwise_test_raw_target_hash_match': ks.array_hash(test.raw_target) == record['test_input_hashes']['raw_target'],
                      'pca_orthonormality_max_abs_dev': q_orth,
                      'exact_solve_rel_residual_max': max(solve_resid.values())}
            if samples is not None:
                n_saved = len(samples['path_target'])
                checks['test_target_max_abs_diff_vs_saved'] = float(np.abs(test.target[:n_saved] - samples['path_target']).max())
                checks['test_context_max_abs_diff_vs_saved'] = float(np.abs(test.context[:n_saved] - samples['path_history']).max())
            state = torch.load(ckpt, map_location='cpu', weights_only=False)
            assert (state['kernel'], state['geometry_id'], state['data_seed'], state['source'], state['head']) == \
                (kernel, gid, data_seed, source, head), 'checkpoint metadata mismatch'
            base = create_model('s4' if head == 's4' else 'mlp', horizon=F, lookback=L,
                                width=ks.width_for(cfg, F, head), **cfg['model'])
            model = CoordinateModel(base)
            model.load_state_dict(state['state_dict'])
            model.eval()
            # Reload check: CPU predictions on the stored test pairs vs the saved (CUDA) predictions.
            with torch.no_grad():
                pred = model(torch.as_tensor(pairs[0][:N_PRED_CHECK], dtype=torch.float32),
                             torch.as_tensor(pairs[2][:N_PRED_CHECK], dtype=torch.float32),
                             torch.as_tensor(test.context[:N_PRED_CHECK], dtype=torch.float32)).numpy()
            checks['prediction_rel_rms_diff_vs_saved'] = None
            if samples is not None:
                saved = samples['prediction'][:N_PRED_CHECK]
                checks['prediction_rel_rms_diff_vs_saved'] = float(np.sqrt(np.mean((pred - saved) ** 2) / np.mean(saved ** 2)))
            by_t, shares, ranks = {}, {'exact': [], 'learned': []}, []
            finite = True
            for t in TIMES:
                x = torch.as_tensor((1 - t) * eps + t * y, dtype=torch.float32)
                J = learned_jacobian(model, x, torch.full((N,), t), torch.as_tensor(h, dtype=torch.float32))
                finite &= bool(np.isfinite(J).all())
                sv = np.linalg.svd(J.astype(np.float64), compute_uv=False)
                ranks.append((sv > 1e-5 * sv[:, :1]).sum(axis=1))
                A, B = Q @ J.astype(np.float64) @ Q.T, Q @ exact[t] @ Q.T
                by_t[str(t)] = {name: block_stats(A, B, index[name]) for name in BLOCK_NAMES}
                if index['tail']:
                    shares['exact'].append(energy_shares(B, index['signal_residual'], index['tail']))
                    shares['learned'].append(energy_shares(A, index['signal_residual'], index['tail']))
            checks['learned_jacobian_finite'] = finite
            ranks = np.concatenate(ranks)
            rank = {'min': int(ranks.min()), 'median': float(np.median(ranks)), 'max': int(ranks.max()),
                    'structural_cap': (ks.width_for(cfg, F, head) - 2) if head == 'mlp' else F}
            blocks = {}
            for name in BLOCK_NAMES:
                if by_t[str(TIMES[0])][name] is None:
                    blocks[name] = None
                    continue
                blocks[name] = {k: float(np.mean([by_t[str(t)][name][k] for t in TIMES]))
                                for k in ('norm_ratio', 'norm_ratio_median', 'cosine')}
            out.append({'kernel': kernel, 'geometry_id': gid, 'F': F, 'lookback': L, 'data_seed': data_seed,
                        'source': source, 'head': head, 'width': ks.width_for(cfg, F, head),
                        'signal_pcs': rec['signal_pcs'],
                        'block_sizes': {k: len(v) for k, v in index.items()},
                        'blocks': blocks, 'by_t': by_t,
                        'residual_energy_shares': (
                            {who: {part: float(np.mean([s[part] for s in v])) for part in ('signal', 'tail', 'cross')}
                             for who, v in shares.items()} if index['tail'] else None),
                        'learned_jacobian_rank': rank,
                        'checks': checks})
            print(f'{kernel} {gid} draw{data_seed} {run}: residual ratio {blocks["residual"]["norm_ratio"]:.3f} '
                  f'cos {blocks["residual"]["cosine"]:.3f}  pred-diff {checks["prediction_rel_rms_diff_vs_saved"]}',
                  flush=True)
    return out, numerics


def summarize(cells, cfg):
    draws = [r['data_seed'] for r in cfg['replicates']]
    pick = lambda k, g, s, hd, d: next(c for c in cells if (c['kernel'], c['geometry_id'], c['source'], c['head'],
                                                             c['data_seed']) == (k, g, s, hd, d))
    summary, ratios, verdicts, ratios_signal = {}, {}, {}, {}
    for kernel in KERNELS:
        for source in SOURCES:
            for head in HEADS:
                for gid in GEOMETRIES:
                    for block in BLOCK_NAMES:
                        for metric in ('norm_ratio', 'cosine'):
                            vals = [pick(kernel, gid, source, head, d)['blocks'][block] for d in draws]
                            vals = [None if v is None else v[metric] for v in vals]
                            key = f'{kernel}|{source}|{head}|{gid}|{block}|{metric}'
                            summary[key] = {'per_draw': vals,
                                            'mean': None if None in vals else float(np.mean(vals))}
            R = {}
            for d in draws:
                R[str(d)] = {gid: pick(kernel, gid, source, 's4', d)['blocks']['residual']['norm_ratio'] /
                             pick(kernel, gid, source, 'mlp', d)['blocks']['residual']['norm_ratio']
                             for gid in GEOMETRIES}
            ratios[f'{kernel}|{source}'] = R
            ratios_signal[f'{kernel}|{source}'] = {
                str(d): {gid: pick(kernel, gid, source, 's4', d)['blocks']['signal_residual']['norm_ratio'] /
                         pick(kernel, gid, source, 'mlp', d)['blocks']['signal_residual']['norm_ratio']
                         for gid in GEOMETRIES} for d in draws}
            grows = [R[str(d)]['F64_l8'] > R[str(d)]['F16_l8'] for d in draws]
            above = [R[str(d)]['F64_l8'] > 1 for d in draws]
            if all(grows) and all(above):
                v = 'supported'
            elif sum(not g for g in grows) >= 2:
                v = 'refuted'
            else:
                v = 'inconclusive'
            verdicts[f'{kernel}|{source}'] = {'R64_gt_R16': grows, 'R64_gt_1': above, 'verdict': v,
                                              'counts_for_verdict': (kernel, source) == ('rbf', 'gp')}
    return summary, ratios, verdicts, ratios_signal


def figure(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'mlp': '#2a78d6', 's4': '#eb6834'}
    ink, muted, grid = '#1f1f1e', '#6b6a63', '#e4e3dc'
    summary, draws = data['summary'], [str(d) for d in data['settings']['data_seeds']]
    Fs = [16, 32, 64]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    for ax, source in zip(axes, SOURCES):
        ax.axhline(1, color=muted, lw=1, zorder=1)
        top = 1.0
        for head in HEADS:
            per = np.array([summary[f'rbf|{source}|{head}|{g}|residual|norm_ratio']['per_draw'] for g in GEOMETRIES])
            for j in range(per.shape[1]):
                ax.plot(Fs, per[:, j], color=colors[head], lw=0.9, alpha=.35, zorder=2)
            mean = per.mean(axis=1)
            ax.plot(Fs, mean, color=colors[head], lw=2.2, marker='o', ms=6, zorder=3,
                    markeredgecolor='white', markeredgewidth=1.2, label='S4' if head == 's4' else 'AdaLN MLP')
            ax.annotate(f'{mean[-1]:.2f}', (Fs[-1], mean[-1]), xytext=(7, 0), textcoords='offset points',
                        va='center', fontsize=8.5, color=ink)
            top = max(top, per.max())
        if source == 'gp':
            for F, gid in zip(Fs, GEOMETRIES):
                r = np.mean([data['R'][f'rbf|{source}'][d][gid] for d in draws])
                ax.annotate(f'S4/MLP {r:.2f}', (F, 0.02), ha='center', va='bottom', fontsize=8, color=muted)
        ax.set_xscale('log', base=2)
        ax.set_xticks(Fs)
        ax.set_xticklabels([str(f) for f in Fs])
        ax.set_xlim(13.5, 80)
        ax.set_ylim(0, top * 1.12)
        ax.set_title(f"{'White' if source == 'white' else 'GP'} prior", fontsize=10, color=ink, loc='left')
        ax.set_xlabel('window length F  (lookback 2F, RBF, l = 8)', fontsize=9, color=muted)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=9)
        ax.grid(axis='y', color=grid, lw=.6, zorder=0)
    axes[0].set_ylabel('learned / exact Jacobian norm, PCs 3..F', fontsize=9, color=muted)
    axes[1].legend(frameon=False, fontsize=9, loc='upper right')
    fig.suptitle('Residual-block Jacobian norm ratio by window length (kernel scan, RBF; '
                 'thin = 3 draws, thick = mean; 1 = exact size)',
                 fontsize=10, color=ink, x=.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, .93))
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=160)
    print('wrote', OUT_FIG)


def print_tables(data):
    s, draws = data['summary'], data['settings']['data_seeds']
    for kernel in KERNELS:
        print(f'\n== {kernel.upper()} ==')
        for block in ('residual', 'signal_residual', 'tail'):
            print(f'  block {block}: norm ratio / cosine (mean [per draw])')
            for source in SOURCES:
                for head in HEADS:
                    for gid in GEOMETRIES:
                        r = s[f'{kernel}|{source}|{head}|{gid}|{block}|norm_ratio']
                        c = s[f'{kernel}|{source}|{head}|{gid}|{block}|cosine']
                        if r['mean'] is None:
                            print(f'    {source:5s} {head:3s} {gid}: null')
                            continue
                        print(f'    {source:5s} {head:3s} {gid}: ratio {r["mean"]:.3f} {[round(v, 3) for v in r["per_draw"]]}'
                              f'  cos {c["mean"]:.3f} {[round(v, 3) for v in c["per_draw"]]}')
        print('  learned-Jacobian rank (min/median/max, cap) and residual energy shares (exact | learned: sig/tail/cross)')
        for source in SOURCES:
            for head in HEADS:
                for gid in GEOMETRIES:
                    cs = [c for c in data['cells'] if (c['kernel'], c['source'], c['head'], c['geometry_id']) ==
                          (kernel, source, head, gid)]
                    rk = cs[0]['learned_jacobian_rank']
                    line = (f'    {source:5s} {head:3s} {gid}: rank {min(c["learned_jacobian_rank"]["min"] for c in cs)}/'
                            f'{np.median([c["learned_jacobian_rank"]["median"] for c in cs]):.0f}/'
                            f'{max(c["learned_jacobian_rank"]["max"] for c in cs)} cap {rk["structural_cap"]}')
                    if cs[0]['residual_energy_shares'] is not None:
                        sh = {w: [np.mean([c['residual_energy_shares'][w][p] for c in cs]) for p in ('signal', 'tail', 'cross')]
                              for w in ('exact', 'learned')}
                        line += ('   shares exact %.3f/%.3f/%.3f | learned %.3f/%.3f/%.3f' % (*sh['exact'], *sh['learned']))
                    print(line)
        for source in SOURCES:
            R = data['R'][f'{kernel}|{source}']
            Rs = data['R_signal_residual_descriptive'][f'{kernel}|{source}']
            v = data['verdicts'][f'{kernel}|{source}']
            print(f'  R(F) {source}:', {d: {g: round(x, 3) for g, x in R[str(d)].items()} for d in draws},
                  '->', v['verdict'], '(verdict)' if v['counts_for_verdict'] else '(descriptive only)')
            print(f'  R_signal(F) {source}:', {d: {g: round(x, 3) for g, x in Rs[str(d)].items()} for d in draws})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--figure-only', action='store_true')
    args = parser.parse_args()
    if args.figure_only:
        data = json.loads(OUT_JSON.read_text())
        print_tables(data)
        figure(data)
        return
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    cfg = json.loads((SCAN / 'config.json').read_text())
    started = time.perf_counter()
    cells, numerics = [], {}
    for kernel in KERNELS:
        for gid in GEOMETRIES:
            for rep in cfg['replicates']:
                part, num = run_group(cfg, kernel, gid, rep['data_seed'])
                cells += part
                numerics[f'{kernel}|{gid}|{rep["data_seed"]}'] = num
    summary, ratios, verdicts, ratios_signal = summarize(cells, cfg)
    data = {'description': 'Learned vs exact head Jacobian by window length F on kernel-scan checkpoints '
                           '(supplementary/exploratory; inference and autodiff only, no training).',
            'reading_rule': {'primary': 'rbf|gp', 'statistic': 'R(F) = norm_ratio_S4 / norm_ratio_MLP, residual block PCs 3..F',
                             'supported': 'R(F64) > R(F16) in all 3 draws and R(F64) > 1 in all 3 draws',
                             'refuted': 'R(F64) <= R(F16) in at least 2 of 3 draws', 'otherwise': 'inconclusive',
                             'secondary_not_for_verdict': ['ou kernel', 'white prior', 'signal_residual/tail split',
                                                           'cosines']},
            'verdict_primary': verdicts['rbf|gp']['verdict'],
            'settings': {'kernels': list(KERNELS), 'geometries': list(GEOMETRIES), 'sources': list(SOURCES),
                         'heads': list(HEADS), 'times': list(TIMES), 'n_windows': N, 'eps_seed': EPS_SEED,
                         'data_seeds': [r['data_seed'] for r in cfg['replicates']],
                         'per_window_then_mean': 'norm ratio and cosine computed per window, averaged over windows, then over t',
                         'device': 'cpu', 'torch': torch.__version__,
                         'kernel_scan_config_sha256': sha256(SCAN / 'config.json')},
            'summary': summary, 'R': ratios, 'verdicts': verdicts,
            'R_signal_residual_descriptive': ratios_signal, 'numerics': numerics, 'cells': cells,
            'seconds': time.perf_counter() - started}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=1) + '\n')
    print('wrote', OUT_JSON)
    print_tables(data)
    figure(data)


if __name__ == '__main__':
    main()
