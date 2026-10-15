#!/usr/bin/env python3
"""Rough-perturbation response during generation, learned heads vs the exact field (no training).

Question. The exact-Jacobian probes (scripts/probe_exact_jacobian.py, scripts/jacobian_per_pc.py)
found that under the GP prior both heads learn little of dv/dx in the velocity PCs 3-16, with S4
learning about twice as much correctly signed response as the MLP. Does that difference show up
during generation, when the trajectory is pushed off the smooth subspace?

Protocol (fixed before any model was run).
  * Main-study checkpoints only: {white, gp} prior x {mlp, s4} x seeds 0-4, plus the GP-prior
    whitened MLP as a supplementary head. No training; CPU inference only.
  * Windows: the first N = 256 test windows (sample_dataset(gp, n_test, data_seed + 2)).
  * Generation: the production Euler rule of fm_stress.experiment.generate, 64 steps,
    t_k = k/64, x <- x + v(x, t_k, h)/64. One prior draw per window, fixed seed, shared by
    every head and seed of that prior.
  * Perturbation: at the first step k with t_k >= t0 (t0 = 0.3 -> k = 20, t = 0.3125;
    t0 = 0.6 -> k = 39, t = 0.609375), delta is added to x before that step's velocity call.
    delta = r * u, where u is a unit vector drawn isotropically in the span of the prior's own
    training-velocity PCs 3-16 (results/main/geometry.json), fixed seed, one direction per
    window, shared by all heads, seeds, t0 and sizes. r = 0.1 (primary) and 0.5 (secondary).
  * Learned response: Delta_learned = x_1(perturbed) - x_1(unperturbed), same prior draw.
  * Ideal response: Delta_ideal = Phi(1, t_k) delta, Phi = prod_{j=k}^{63} (I + J*(t_j)/64),
    J*(t) = U S^-1, S = t^2 C + (1-t)^2 C_prior, U = t C - (1-t) C_prior, with C the exact
    normalized conditional covariance given the raw history (data.conditional_target). The
    exact field is affine in x, so this is its exact response on the same Euler grid.
  * Metrics per window, then averaged over windows (per seed): amplification
    ||Delta_learned||/||delta|| and ||Delta_ideal||/||delta||; response error
    ||Delta_learned - Delta_ideal|| / ||Delta_ideal||; cosine(Delta_learned, Delta_ideal).
    Supplementary: the same error divided by ||delta|| (well defined when the ideal response
    is small, as under the white prior), the projection gain <Delta_l, Delta_i>/||Delta_i||^2,
    the split of both responses into velocity PCs 1-2 vs 3-16, and a linearity check
    ||Delta_l(0.5)/0.5 - Delta_l(0.1)/0.1|| / ||Delta_l(0.1)/0.1||.
  * Reference "delta carried unchanged" (a field with zero Jacobian): error
    ||delta - Delta_ideal|| / ||Delta_ideal||.
  * Sanity check: the exact field (data.conditional_oracle_velocity, float64) is run through the
    same Euler loop; its finite-difference response must match Delta_ideal to rounding.
  * Secondary (ensemble): 64 contexts (first 64 test windows) x 32 members, every member
    perturbed (own direction, fixed seed), fair CRPS from ensemble_evaluation.context_scores
    before and after the perturbation, for every head and for the exact field.

Pre-declared reading rule. Primary setting: GP prior, ||delta|| = 0.1, t0 = 0.3. The statement
"S4 handles rough perturbations more like the ideal field than the MLP" is SUPPORTED if S4's
response error (window mean) is lower than the MLP's in all 5 seeds, REFUTED if it is lower in
at most 2 of 5 seeds, and INCONCLUSIVE otherwise (3 or 4 of 5). The white prior is a control,
where both heads are expected to be close to ideal. Other settings are descriptive only.

Disclosed before the model runs (model-free inspection of the exact field only): under the GP
prior the ideal response to a PC3-16 perturbation is strongly amplified (median about 17x at
t0 = 0.3) and lands mostly in velocity PCs 1-2; under the white prior it is contracted (median
about 0.12x), so the relative response error has a small denominator there. That is why the
error divided by ||delta|| is also reported. The ideal field uses privileged raw-history
statistics (per-window scale) that the networks cannot see, so it is a reference, not an
attainable target.

Writes results/diagnostics/perturbation_recovery.json and output/jacobian/perturbation_recovery.png.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fm_stress.data import (GPConfig, conditional_oracle_velocity, conditional_target,  # noqa: E402
                            sample_dataset, source_covariance)
from fm_stress.ensemble_evaluation import context_scores  # noqa: E402
from fm_stress.experiment import CoordinateModel, generate  # noqa: E402
from fm_stress.models import create_model  # noqa: E402

MAIN = ROOT / 'results/main'
OUT_JSON = ROOT / 'results/diagnostics/perturbation_recovery.json'
OUT_FIG = ROOT / 'output/jacobian/perturbation_recovery.png'
N = 256
T0S = (0.3, 0.6)
SIZES = (0.1, 0.5)
PRIMARY = ('gp', 0.3, 0.1)
HEADS = {'white': ('mlp', 's4'), 'gp': ('mlp', 's4', 'whitened_mlp')}
N_CTX, N_MEM = 64, 32
EPS_SEED, DELTA_SEED, ENS_EPS_SEED, ENS_DELTA_SEED = 731001, 731002, 731003, 731004


def start_step(t0, steps):
    """First Euler step index k with k/steps >= t0."""
    return int(math.ceil(t0 * steps - 1e-9))


def setting_key(t0, size):
    return f't0={t0}|size={size}'


@torch.no_grad()
def integrate(field, x, h, k_start, k_stop, steps):
    """The production Euler update of experiment.generate, over steps k_start..k_stop-1."""
    for k in range(k_start, k_stop):
        t = torch.full((len(x),), k / steps, dtype=x.dtype)
        x = x + field(x, t, h) / steps
    return x


def responses(field, x0, h, deltas, steps):
    """Unperturbed x_1 and, for every (t0, size), the perturbed-minus-unperturbed x_1."""
    ks = sorted({start_step(t0, steps) for t0 in T0S})
    states, x, k_prev = {}, x0.clone(), 0
    for k in ks:
        x = integrate(field, x, h, k_prev, k, steps)
        states[k], k_prev = x.clone(), k
    x1 = integrate(field, x, h, k_prev, steps, steps)
    out = {}
    for t0 in T0S:
        k = start_step(t0, steps)
        for size in SIZES:
            xp = integrate(field, states[k] + size * deltas, h, k, steps, steps)
            out[(t0, size)] = (xp - x1)
    return x1, out


class ExactField:
    """Production analytic oracle velocity as a field(x, t, h); per-row mean and covariance."""
    def __init__(self, mean, cov, prior_cov):
        self.mean, self.cov, self.prior_cov = mean, cov, prior_cov

    def __call__(self, x, t, h):
        v = conditional_oracle_velocity(x.numpy(), t.numpy(), self.mean, self.cov, self.prior_cov)
        return torch.as_tensor(v, dtype=torch.float64)


def exact_jacobians(cov, prior_cov, steps, k_min):
    jac = {}
    for j in range(k_min, steps):
        t = j / steps
        S = t * t * cov + (1 - t) ** 2 * prior_cov
        U = t * cov - (1 - t) * prior_cov
        jac[j] = np.linalg.solve(S.swapaxes(-1, -2), U.swapaxes(-1, -2)).swapaxes(-1, -2)   # U S^-1
    return jac


def ideal_response(delta, jac, k, steps):
    d = delta.copy()
    for j in range(k, steps):
        d = d + np.einsum('nij,nj->ni', jac[j], d) / steps
    return d


def norm(v):
    return np.linalg.norm(v, axis=-1)


def response_metrics(dl, di, delta, Q):
    """Per-window metrics; Q rows are the prior's velocity PCs."""
    nd, ni, nl = norm(delta), norm(di), norm(dl)
    e = dl - di
    pl, pi, pe = dl @ Q.T, di @ Q.T, e @ Q.T
    return {'amp_learned': nl / nd, 'amp_ideal': ni / nd,
            'rel_error': norm(e) / ni, 'abs_error_per_delta': norm(e) / nd,
            'cosine': (dl * di).sum(-1) / (nl * ni), 'gain': (dl * di).sum(-1) / ni ** 2,
            'learned_pc12': norm(pl[:, :2]) / nd, 'learned_pc3_16': norm(pl[:, 2:]) / nd,
            'ideal_pc12': norm(pi[:, :2]) / nd, 'ideal_pc3_16': norm(pi[:, 2:]) / nd,
            'rel_error_pc12_part': norm(pe[:, :2]) / ni, 'rel_error_pc3_16_part': norm(pe[:, 2:]) / ni}


def summarize(per_window):
    out = {k: float(np.mean(v)) for k, v in per_window.items()}
    out['rel_error_median'] = float(np.median(per_window['rel_error']))
    return out


def unit_directions(rng_seed, n, Q):
    d = np.random.default_rng(rng_seed).standard_normal((n, Q.shape[0] - 2)) @ Q[2:]
    return d / norm(d)[:, None]


def load_model(source, kind, seed, cfg):
    path = MAIN / 'checkpoints' / f'{source}_{kind}_seed{seed}.pt'
    state = torch.load(path, map_location='cpu', weights_only=False)
    model = CoordinateModel(create_model('s4' if kind == 's4' else 'mlp', horizon=16, lookback=32, **cfg['model']))
    model.load_state_dict(state['state_dict'])
    model.eval()
    return model, hashlib.sha256(path.read_bytes()).hexdigest()


def ensemble_scores(samples, target):
    s = context_scores(samples.reshape(N_CTX, N_MEM, -1), target)
    return float(s['fair_crps'].mean()), float(s['generated_roughness'].mean())


def main():
    started = time.time()
    torch.set_num_threads(4)
    cfg = json.loads((MAIN / 'config.json').read_text())
    geo = json.loads((MAIN / 'geometry.json').read_text())
    gp = GPConfig(**cfg['data'])
    steps = cfg['euler_steps']
    test = sample_dataset(gp, cfg['n_test'], cfg['data_seed'] + 2)
    means, covs = conditional_target(gp, test)
    mu, C, y, h_np = means[:N], covs[:N], test.target[:N], test.context[:N]
    h = torch.as_tensor(h_np, dtype=torch.float32)
    # Ensemble inputs: context-major rows (c * N_MEM + m).
    ctx_target = test.target[:N_CTX]
    mu_e, C_e = np.repeat(means[:N_CTX], N_MEM, 0), np.repeat(covs[:N_CTX], N_MEM, 0)
    h_e = torch.as_tensor(np.repeat(test.context[:N_CTX], N_MEM, 0), dtype=torch.float32)
    k_min = min(start_step(t0, steps) for t0 in T0S)

    record = {'protocol': {'n_windows': N, 'euler_steps': steps, 't0': list(T0S),
                           'start_step': {str(t0): start_step(t0, steps) for t0 in T0S},
                           'start_time': {str(t0): start_step(t0, steps) / steps for t0 in T0S},
                           'sizes': list(SIZES), 'primary': {'source': PRIMARY[0], 't0': PRIMARY[1], 'size': PRIMARY[2]},
                           'seeds': {'eps': EPS_SEED, 'delta': DELTA_SEED, 'ensemble_eps': ENS_EPS_SEED,
                                     'ensemble_delta': ENS_DELTA_SEED},
                           'ensemble': {'contexts': N_CTX, 'members': N_MEM},
                           'reading_rule': 'GP prior, size 0.1, t0 0.3: S4 rel_error < MLP rel_error in 5/5 seeds -> '
                                           'supported; <= 2/5 -> refuted; otherwise inconclusive'},
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'config_sha256': hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
              'runs': [], 'reference': {}, 'sanity': {}}

    for source in ('white', 'gp'):
        Cp = source_covariance(gp, source)
        chol = np.linalg.cholesky(Cp)
        Q = np.asarray(geo['pca'][source]['components'])
        eps = np.random.default_rng(EPS_SEED).standard_normal((N, gp.horizon)) @ chol.T
        u = unit_directions(DELTA_SEED, N, Q)
        eps_e = np.random.default_rng(ENS_EPS_SEED).standard_normal((N_CTX * N_MEM, gp.horizon)) @ chol.T
        u_e = unit_directions(ENS_DELTA_SEED, N_CTX * N_MEM, Q)
        jac = exact_jacobians(C, Cp, steps, k_min)
        ideal = {(t0, s): ideal_response(s * u, jac, start_step(t0, steps), steps) for t0 in T0S for s in SIZES}

        # Spread of x_t in the PC3-16 subspace under the exact marginal, for scale.
        spread = {}
        for t0 in T0S:
            t = start_step(t0, steps) / steps
            M = t * t * C + (1 - t) ** 2 * Cp
            spread[str(t0)] = float(np.median(np.sqrt(np.einsum('ki,nij,kj->n', Q[2:], M, Q[2:]))))
        ref = {}
        for (t0, s), di in ideal.items():
            ref[setting_key(t0, s)] = {'carried_unchanged_rel_error': float(np.mean(norm(s * u - di) / norm(di))),
                                       'amp_ideal': float(np.mean(norm(di) / s)),
                                       'amp_ideal_median': float(np.median(norm(di) / s)),
                                       'ideal_pc12_energy_fraction': float(np.mean(
                                           norm((di @ Q.T)[:, :2]) ** 2 / norm(di) ** 2))}
        record['reference'][source] = {'settings': ref,
                                       'median_rms_norm_of_x_t_in_pc3_16': spread,
                                       'velocity_pca_eigenvalues': geo['pca'][source]['eigenvalues']}

        # Sanity check: exact field through the same Euler loop, float64.
        exact = ExactField(mu, C, Cp)
        _, d_exact = responses(exact, torch.as_tensor(eps, dtype=torch.float64), None,
                               torch.as_tensor(u, dtype=torch.float64), steps)
        record['sanity'][source] = {setting_key(t0, s): {
            'max_rel_error': float(np.max(norm(d_exact[(t0, s)].numpy() - ideal[(t0, s)]) / norm(ideal[(t0, s)]))),
            'mean_rel_error': float(np.mean(norm(d_exact[(t0, s)].numpy() - ideal[(t0, s)]) / norm(ideal[(t0, s)])))}
            for t0 in T0S for s in SIZES}

        # Ensemble reference: exact field (float64).
        exact_e = ExactField(mu_e, C_e, Cp)
        x1e, de = responses(exact_e, torch.as_tensor(eps_e, dtype=torch.float64), None,
                            torch.as_tensor(u_e, dtype=torch.float64), steps)
        x1e = x1e.numpy()
        base_crps, base_rough = ensemble_scores(x1e, ctx_target)
        record['reference'][source]['ensemble_exact_field'] = {
            'crps_unperturbed': base_crps, 'roughness_unperturbed': base_rough,
            'settings': {setting_key(t0, s): dict(zip(('crps_perturbed', 'roughness_perturbed'),
                                                      ensemble_scores(x1e + de[(t0, s)].numpy(), ctx_target)))
                         for t0 in T0S for s in SIZES}}
        print(source, 'reference + sanity done', record['sanity'][source], flush=True)

        x0 = torch.as_tensor(eps, dtype=torch.float32)
        uu = torch.as_tensor(u, dtype=torch.float32)
        x0e = torch.as_tensor(eps_e, dtype=torch.float32)
        uue = torch.as_tensor(u_e, dtype=torch.float32)
        for kind in HEADS[source]:
            for seed in cfg['seeds']:
                model, ckpt_hash = load_model(source, kind, seed, cfg)
                x1, dl = responses(model, x0, h, uu, steps)
                # Check that the loop reproduces the production generator.
                prod = generate(model, x0, h, steps, batch_size=N)
                run = {'source': source, 'kind': kind, 'seed': seed, 'checkpoint_sha256': ckpt_hash,
                       'max_abs_diff_vs_production_generate': float(np.max(np.abs(prod - x1.numpy()))),
                       'settings': {}, 'linearity': {}, 'ensemble': {}}
                for t0 in T0S:
                    for s in SIZES:
                        m = response_metrics(dl[(t0, s)].numpy().astype(np.float64), ideal[(t0, s)], s * u, Q)
                        run['settings'][setting_key(t0, s)] = summarize(m)
                    a = dl[(t0, SIZES[1])].numpy() / SIZES[1]
                    b = dl[(t0, SIZES[0])].numpy() / SIZES[0]
                    run['linearity'][str(t0)] = float(np.mean(norm(a - b) / norm(b)))
                x1m, dm = responses(model, x0e, h_e, uue, steps)
                x1m = x1m.numpy().astype(np.float64)
                crps0, rough0 = ensemble_scores(x1m, ctx_target)
                run['ensemble'] = {'crps_unperturbed': crps0, 'roughness_unperturbed': rough0,
                                   'settings': {setting_key(t0, s): dict(zip(
                                       ('crps_perturbed', 'roughness_perturbed'),
                                       ensemble_scores(x1m + dm[(t0, s)].numpy(), ctx_target)))
                                       for t0 in T0S for s in SIZES}}
                record['runs'].append(run)
                p = run['settings'][setting_key(0.3, 0.1)]
                print(f"{source} {kind} seed{seed}: amp {p['amp_learned']:.3f} (ideal {p['amp_ideal']:.3f}) "
                      f"rel_err {p['rel_error']:.3f} cos {p['cosine']:.3f}", flush=True)

    record['summary'] = aggregate(record['runs'], cfg['seeds'])
    record['verdict'] = verdict(record['runs'], cfg['seeds'])
    record['seconds'] = time.time() - started
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    report(record)
    figure(record, cfg['seeds'])


def aggregate(runs, seeds):
    out = {}
    cells = sorted({(r['source'], r['kind']) for r in runs})
    for source, kind in cells:
        rows = {r['seed']: r for r in runs if r['source'] == source and r['kind'] == kind}
        cell = {}
        for t0 in T0S:
            for s in SIZES:
                key = setting_key(t0, s)
                metrics = rows[seeds[0]]['settings'][key].keys()
                cell[key] = {m: {'mean': float(np.mean([rows[sd]['settings'][key][m] for sd in seeds])),
                                 'per_seed': [rows[sd]['settings'][key][m] for sd in seeds]} for m in metrics}
                ens = [rows[sd]['ensemble']['settings'][key]['crps_perturbed'] - rows[sd]['ensemble']['crps_unperturbed']
                       for sd in seeds]
                cell[key]['crps_change'] = {'mean': float(np.mean(ens)), 'per_seed': ens}
        cell['crps_unperturbed'] = {'mean': float(np.mean([rows[sd]['ensemble']['crps_unperturbed'] for sd in seeds])),
                                    'per_seed': [rows[sd]['ensemble']['crps_unperturbed'] for sd in seeds]}
        cell['linearity'] = {str(t0): {'mean': float(np.mean([rows[sd]['linearity'][str(t0)] for sd in seeds])),
                                       'per_seed': [rows[sd]['linearity'][str(t0)] for sd in seeds]} for t0 in T0S}
        out[f'{source}|{kind}'] = cell
    return out


def verdict(runs, seeds):
    def per_seed(source, kind, t0, s):
        rows = {r['seed']: r for r in runs if r['source'] == source and r['kind'] == kind}
        return [rows[sd]['settings'][setting_key(t0, s)]['rel_error'] for sd in seeds]
    out = {}
    for source in ('white', 'gp'):
        for t0 in T0S:
            for s in SIZES:
                m, q = per_seed(source, 'mlp', t0, s), per_seed(source, 's4', t0, s)
                out[f'{source}|{setting_key(t0, s)}'] = {'s4_lower_in_seeds': int(sum(b < a for a, b in zip(m, q))),
                                                         'mlp': m, 's4': q}
    wins = out[f'{PRIMARY[0]}|{setting_key(PRIMARY[1], PRIMARY[2])}']['s4_lower_in_seeds']
    out['primary'] = {'s4_lower_in_seeds': wins,
                      'verdict': 'supported' if wins == 5 else ('refuted' if wins <= 2 else 'inconclusive')}
    return out


def report(record):
    print('\nSanity (exact field in the Euler loop vs the propagator), max relative error:')
    for source, rows in record['sanity'].items():
        print(' ', source, {k: f"{v['max_rel_error']:.2e}" for k, v in rows.items()})
    print('\nEnsemble CRPS without perturbation:',
          {cell: round(v['crps_unperturbed']['mean'], 4) for cell, v in record['summary'].items()},
          {f'{s}|exact': round(record['reference'][s]['ensemble_exact_field']['crps_unperturbed'], 4)
           for s in ('white', 'gp')})
    for key in [setting_key(t0, s) for t0 in T0S for s in SIZES]:
        print(f'\n{key}')
        print('  %-22s %8s %8s %9s %9s %8s %8s %9s' % ('cell', 'amp', 'amp*', 'rel_err', 'err/|d|', 'cosine', 'gain',
                                                     'dCRPS'))
        for source in ('white', 'gp'):
            ref = record['reference'][source]
            ens = ref['ensemble_exact_field']
            amp_ideal = ref['settings'][key]['amp_ideal']
            print('  %-22s %8.3f %8.3f %9.3f %9s %8s %8s %+9.4f' % (
                f'{source}|exact field', amp_ideal, amp_ideal, 0.0, '', '', '',
                ens['settings'][key]['crps_perturbed'] - ens['crps_unperturbed']))
            print('  %-22s %8.3f %8.3f %9.3f' % (f'{source}|zero-Jacobian', 1.0, amp_ideal,
                                                ref['settings'][key]['carried_unchanged_rel_error']))
            for kind in HEADS[source]:
                v = record['summary'][f'{source}|{kind}'][key]
                print('  %-22s %8.3f %8.3f %9.3f %9.3f %8.3f %8.3f %+9.4f' % (
                    f'{source}|{kind}', v['amp_learned']['mean'], v['amp_ideal']['mean'], v['rel_error']['mean'],
                    v['abs_error_per_delta']['mean'], v['cosine']['mean'], v['gain']['mean'],
                    v['crps_change']['mean']))
    print('\nverdict:', record['verdict']['primary'])


def figure(record, seeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    key = setting_key(PRIMARY[1], PRIMARY[2])
    colors = {'mlp': '#2a78d6', 's4': '#eb6834'}
    labels = {'mlp': 'MLP', 's4': 'S4'}
    ink, muted, grid = '#1f1f1e', '#6b6a63', '#e4e3dc'
    summ = record['summary']
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), gridspec_kw={'width_ratios': [1, 1, 1.3]})

    def style(ax):
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=9)
        ax.grid(axis='y', color=grid, lw=.6)
        ax.set_axisbelow(True)

    for ax, source in zip(axes[:2], ('white', 'gp')):
        kinds = ('mlp', 's4')
        vals = {k: np.asarray(summ[f'{source}|{k}'][key]['rel_error']['per_seed']) for k in kinds}
        for i in range(len(seeds)):     # same seed, MLP -> S4
            ax.plot([0, 1], [vals['mlp'][i], vals['s4'][i]], color='#c9c8c0', lw=1, zorder=1)
        for j, k in enumerate(kinds):
            ax.scatter(np.full(len(seeds), j), vals[k], s=38, color=colors[k], edgecolor='white', lw=1, zorder=3)
            ax.plot([j - .18, j + .18], [vals[k].mean()] * 2, color=ink, lw=2, zorder=4)
            ax.annotate(f'{vals[k].mean():.3f}', (j + (.22 if j else -.22), vals[k].mean()), va='center',
                        ha='left' if j else 'right', fontsize=9, color=ink)
        lo = min(v.min() for v in vals.values())
        hi = max(v.max() for v in vals.values())
        carried = record['reference'][source]['settings'][key]['carried_unchanged_rel_error']
        in_range = lo - .5 * (hi - lo) <= carried <= hi + .5 * (hi - lo)
        if in_range:
            lo, hi = min(lo, carried), max(hi, carried)
        pad = (hi - lo) * .15
        ax.set_ylim(lo - pad, hi + pad)
        if in_range:
            ax.axhline(carried, color=muted, lw=1, ls='--', zorder=0)
            ax.text(1.65, carried, f'zero-Jacobian field\n(δ carried unchanged): {carried:.3f}', fontsize=7.5,
                    color=muted, ha='right', va='bottom', bbox=dict(fc='white', ec='none', pad=1), zorder=2)
        else:
            ax.text(.98, .98, f'zero-Jacobian field\n(δ carried unchanged): {carried:.1f}, off scale', fontsize=7.5,
                    color=muted, ha='right', va='top', transform=ax.transAxes)
        wins = record['verdict'][f'{source}|{key}']['s4_lower_in_seeds']
        ax.text(.5, .02, f'S4 lower in {wins}/5 seeds', fontsize=8.5, color=ink, ha='center', va='bottom',
                transform=ax.transAxes)
        ax.set_xticks(range(len(kinds)), [labels[k] for k in kinds])
        ax.set_xlim(-.7, 1.7)
        style(ax)
        ax.set_title(f"{'White' if source == 'white' else 'GP'} prior: response error", fontsize=10, color=ink,
                     loc='left')
        ax.set_ylabel('‖Δ_learned − Δ_ideal‖ / ‖Δ_ideal‖', fontsize=9, color=muted)

    ax = axes[2]
    groups = [('white', 'ideal'), ('white', 'mlp'), ('white', 's4'), ('gp', 'ideal'), ('gp', 'mlp'), ('gp', 's4')]
    xs = [0, 1, 2, 3.6, 4.6, 5.6]
    for x, (source, k) in zip(xs, groups):
        if k == 'ideal':
            a = record['reference'][source]['settings'][key]['amp_ideal']
            ax.scatter([x], [a], marker='D', s=40, color=ink, zorder=3)
            ax.annotate(f'{a:.2f}' if a < 1 else f'{a:.1f}', (x + .2, a), va='center', fontsize=9, color=ink)
        else:
            v = np.asarray(summ[f'{source}|{k}'][key]['amp_learned']['per_seed'])
            ax.scatter(np.full(len(v), x), v, s=38, color=colors[k], edgecolor='white', lw=1, zorder=3)
            ax.annotate(f'{v.mean():.2f}', (x + .2, v.mean()), va='center', fontsize=9, color=ink)
    ax.set_yscale('log')
    ax.set_ylim(.1, 40)
    ax.axhline(1, color=muted, lw=1, ls=':', zorder=0)
    ax.text(6.3, 1, 'δ kept as is', fontsize=7.5, color=muted, ha='right', va='bottom')
    ax.set_xticks(xs, ['exact', 'MLP', 'S4'] * 2)
    ax.text(1, -0.19, 'white prior', transform=ax.get_xaxis_transform(), ha='center', fontsize=9, color=muted)
    ax.text(4.6, -0.19, 'GP prior', transform=ax.get_xaxis_transform(), ha='center', fontsize=9, color=muted)
    ax.set_xlim(-.5, 6.4)
    style(ax)
    ax.set_ylabel('‖Δ‖ / ‖δ‖  (log scale)', fontsize=9, color=muted)
    ax.set_title('Amplification of the perturbation at t = 1', fontsize=10, color=ink, loc='left')
    fig.suptitle('Response at t = 1 to a rough perturbation: ‖δ‖ = 0.1 in velocity PCs 3–16, added at t = 0.3125 '
                 '(main study, 256 test windows, 64 Euler steps)\ndots = seeds (grey lines pair the same seed), '
                 'bar = mean over seeds; the two error panels have different, zoomed y-scales',
                 fontsize=9.5, color=ink, x=.01, y=.98, ha='left', va='top')
    fig.subplots_adjust(left=.06, right=.99, top=.8, bottom=.16, wspace=.32)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=160)


if __name__ == '__main__':
    if '--figure-only' in sys.argv:          # redraw from the saved JSON
        saved = json.loads(OUT_JSON.read_text())
        report(saved)
        figure(saved, saved['runs'] and sorted({r['seed'] for r in saved['runs']}))
    else:
        main()
