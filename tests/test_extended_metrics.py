import numpy as np
from fm_stress.extended_metrics import forecast_scores


def test_channel_boundaries_do_not_count_as_temporal_roughness():
    y = np.tile(np.array([0., 100.]), (3, 8, 1))
    x = np.repeat(y[:, None], 4, axis=1)
    m = forecast_scores(x, y)
    for key in ('fair_crps', 'joint_energy_scaled', 'mean_forecast_mse', 'generated_roughness', 'target_roughness'):
        np.testing.assert_array_equal(m[key], 0)


def test_joint_score_detects_cross_channel_dependence():
    # Same univariate marginals, different joint distributions.
    correct = np.array([[-1., -1.], [1., 1.]]*20)
    wrong = np.array([[-1., 1.], [1., -1.]]*20)
    target = np.tile([[-1., -1.], [1., 1.]], (20, 1))[:, None]
    a = np.tile(correct[None, :, None], (40, 1, 2, 1))
    b = np.tile(wrong[None, :, None], (40, 1, 2, 1))
    target = np.repeat(target, 2, axis=1)
    ma, mb = forecast_scores(a, target), forecast_scores(b, target)
    np.testing.assert_allclose(ma['fair_crps'], mb['fair_crps'])
    assert ma['joint_energy_scaled'].mean() < mb['joint_energy_scaled'].mean()
