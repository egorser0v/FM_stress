"""Joint horizon/channel metrics for the explicitly separate extension suite."""
import numpy as np
from .ensemble_evaluation import fair_crps, fair_energy_score, interval_metrics


def forecast_scores(samples, target):
    """Return per-context scores; never difference across channel boundaries.

    Input [N,S,F,C], target [N,F,C], in train-standardized channel units.
    Energy is joint Euclidean energy / sqrt(F*C), marginal CRPS averages
    coordinates. Sum-CRPS uses the channel sum divided by sqrt(C).
    """
    x, y = np.asarray(samples, float), np.asarray(target, float)
    if x.ndim != 4 or y.shape != (x.shape[0], *x.shape[2:]) or x.shape[1] < 2:
        raise ValueError('Expected [N,S,F,C] samples and [N,F,C] targets')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite forecasts')
    n, s, f, c = x.shape
    xx, yy = x.reshape(n, s, -1), y.reshape(n, -1)
    cov, width = interval_metrics(xx, yy, .9)
    return {
        'fair_crps': fair_crps(xx, yy),
        'joint_energy_scaled': fair_energy_score(xx, yy) / np.sqrt(f*c),
        'channel_sum_crps_scaled': fair_crps(x.sum(-1)/np.sqrt(c), y.sum(-1)/np.sqrt(c)),
        'mean_forecast_mse': np.square(x.mean(1)-y).mean((1, 2)),
        'coverage_90': cov, 'width_90': width,
        'generated_roughness': np.abs(np.diff(x, axis=2)).mean((1, 2, 3)),
        'target_roughness': np.abs(np.diff(y, axis=1)).mean((1, 2)),
    }


def means(scores):
    out = {k: float(np.mean(v)) for k, v in scores.items()}
    out['roughness_ratio'] = out['generated_roughness']/max(out['target_roughness'], 1e-12)
    return out
