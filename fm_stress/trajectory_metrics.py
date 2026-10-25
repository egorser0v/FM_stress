"""NumPy-only post-hoc scores for univariate forecast trajectories.

The last axis is time, never a collection of independent observations. Fair
estimators assume conditionally iid ensemble members; negative estimates are
valid and are never clipped. No inference or model-loading dependency is used.
"""
from __future__ import annotations

import numpy as np


def validate_ensemble(samples, target):
    x, y = np.asarray(samples, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if (x.ndim != 3 or x.shape[0] == 0 or x.shape[1] < 2 or x.shape[2] < 2
            or y.shape != (x.shape[0], x.shape[2])):
        raise ValueError("Expected samples [context,member,time], member/time>=2, target [context,time]")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Forecast and target must be finite")
    return x, y


def roughness(paths, order=1):
    """Per-path mean absolute first/second difference, with unit time spacing."""
    x = np.asarray(paths, dtype=np.float64)
    if (order not in (1, 2) or x.ndim not in (2, 3) or x.shape[-1] <= order
            or x.size == 0 or not np.isfinite(x).all()):
        raise ValueError("Expected finite paths [context,(member),time] with time>order, order 1 or 2")
    return np.abs(np.diff(x, n=order, axis=-1)).mean(axis=-1)


def variogram_weights(horizon):
    """Unordered i<j pairs and inverse time-lag weights, summing to one."""
    if not isinstance(horizon, (int, np.integer)) or horizon < 2:
        raise ValueError("Horizon must be an integer >=2")
    i, j = np.triu_indices(horizon, k=1)
    weights = 1.0 / (j - i)
    return i, j, weights / weights.sum()


def variogram_scores(samples, target, p=0.5, context_batch=32):
    """Per-context plug-in and fair VS, and the removed finite-M correction.

    z_mij=|x_mi-x_mj|**p, a_ij=|y_i-y_j|**p.
    plug-in=sum w_ij*(a_ij-mean_m z_mij)**2;
    fair=plug-in-sum w_ij*var_m(z_mij,ddof=1)/M.
    """
    x, y = validate_ensemble(samples, target)
    if not np.isfinite(p) or not 0 < p < 2 or not isinstance(context_batch, int) or context_batch < 1:
        raise ValueError("Require 0<p<2 and a positive integer context batch")
    i, j, w = variogram_weights(x.shape[-1])
    plugin, correction = np.empty(len(x)), np.empty(len(x))
    for start in range(0, len(x), context_batch):
        xx, yy = x[start:start+context_batch], y[start:start+context_batch]
        z = np.abs(xx[:, :, i] - xx[:, :, j]) ** p
        observed = np.abs(yy[:, i] - yy[:, j]) ** p
        plugin[start:start+len(xx)] = np.square(observed-z.mean(axis=1)) @ w
        correction[start:start+len(xx)] = (z.var(axis=1, ddof=1) / x.shape[1]) @ w
    return {"variogram_plugin": plugin, "variogram_fair": plugin-correction,
            "variogram_correction": correction}


def crps_scores(samples, target):
    """Marginal CRPS, averaged over time; empirical plug-in and fair estimator."""
    x, y = validate_ensemble(samples, target)
    m = x.shape[1]
    weights = 2*np.arange(1, m+1)-m-1
    pairs = (np.sort(x, axis=1)*weights[None, :, None]).sum(axis=1).mean(axis=1)
    first = np.abs(x-y[:, None]).mean(axis=(1, 2))
    return {"crps_plugin": first-pairs/m**2, "fair_crps": first-pairs/(m*(m-1))}


def energy_scores(samples, target):
    """Euclidean whole-path energy, with exactly the existing unscaled units."""
    x, y = validate_ensemble(samples, target)
    m = x.shape[1]
    i, j = np.triu_indices(m, k=1)
    first = np.linalg.norm(x-y[:, None], axis=-1).mean(axis=1)
    # One context at a time bounds memory for long horizons or large ensembles.
    pairs = np.array([np.linalg.norm(xx[i]-xx[j], axis=-1).sum() for xx in x])
    return {"energy_plugin": first-pairs/m**2,
            "fair_energy_score": first-pairs/(m*(m-1))}


def trajectory_scores(samples, target):
    """Equal-context scores; roughness ratios must be formed after averaging."""
    x, y = validate_ensemble(samples, target)
    if x.shape[-1] < 3:
        raise ValueError("R2 requires at least three time coordinates")
    scores = {**variogram_scores(x, y), **crps_scores(x, y), **energy_scores(x, y)}
    for k in (1, 2):
        scores[f"R{k}_generated"] = roughness(x, k).mean(axis=1)
        scores[f"R{k}_target"] = roughness(y, k)
    return scores


def summarize_scores(scores):
    means = {key: float(np.mean(value)) for key, value in scores.items()}
    for k in (1, 2):
        denominator = means[f"R{k}_target"]
        means[f"R{k}_ratio"] = means[f"R{k}_generated"]/denominator if denominator > 0 else None
    return means


def paired_context_bootstrap(a, b, seed=90041, repetitions=10000):
    """Conditional paired-context percentile intervals, including ratio contrasts.

    Inputs are scores for aligned histories, possibly already averaged over a
    fixed set of fitted models. A ratio is recomputed on every resample.
    """
    if set(a) != set(b) or not a:
        raise ValueError("Paired score sets must have the same nonempty keys")
    n = len(next(iter(a.values())))
    if repetitions < 1 or any(np.shape(v) != (n,) or not np.isfinite(v).all()
                              for v in [*a.values(), *b.values()]):
        raise ValueError("Expected aligned, finite, one-dimensional context scores")
    if n == 0:
        raise ValueError("No paired contexts")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(repetitions, n))
    result = {}
    for key in a:
        differences = np.asarray(a[key])-np.asarray(b[key])
        boot = differences[draws].mean(axis=1)
        result[key] = {"difference": float(differences.mean()),
                       "context_ci95": np.quantile(boot, [.025, .975]).tolist()}
    am, bm = summarize_scores(a), summarize_scores(b)
    for k in (1, 2):
        ag, at = a[f"R{k}_generated"][draws].mean(axis=1), a[f"R{k}_target"][draws].mean(axis=1)
        bg, bt = b[f"R{k}_generated"][draws].mean(axis=1), b[f"R{k}_target"][draws].mean(axis=1)
        if np.all(at > 0) and np.all(bt > 0):
            result[f"R{k}_ratio"] = {"difference": am[f"R{k}_ratio"]-bm[f"R{k}_ratio"],
                                      "context_ci95": np.quantile(ag/at-bg/bt, [.025, .975]).tolist()}
        else:
            result[f"R{k}_ratio"] = {"difference": None, "context_ci95": None}
    return result
