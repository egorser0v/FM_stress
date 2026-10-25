"""Independent hand calculations and invariants for archived trajectory scores."""
import itertools
import numpy as np
import pytest

from fm_stress.trajectory_metrics import (crps_scores, energy_scores, roughness,
    trajectory_scores, variogram_scores, variogram_weights, summarize_scores,
    paired_context_bootstrap)


def test_roughness_known_paths_and_ratio_of_means():
    assert roughness([[0, 1, 2, 3]], 1)[0] == 1
    assert roughness([[0, 1, 2, 3]], 2)[0] == 0
    assert roughness([[0, 1, 0, 1]], 2)[0] == 2
    scores = {'R1_generated': np.array([2., 2.]), 'R1_target': np.array([1., 3.]),
              'R2_generated': np.array([2., 4.]), 'R2_target': np.array([1., 2.])}
    assert summarize_scores(scores)['R1_ratio'] == 1
    assert summarize_scores(scores)['R2_ratio'] == 2
    assert summarize_scores({**scores, 'R1_target': np.zeros(2)})['R1_ratio'] is None


def test_normalized_inverse_lag_weights():
    i, j, w = variogram_weights(3)
    np.testing.assert_array_equal(i, [0, 0, 1])
    np.testing.assert_array_equal(j, [1, 2, 2])
    np.testing.assert_allclose(w, [.4, .2, .4])
    assert np.isclose(variogram_weights(64)[2].sum(), 1)


def test_variogram_two_member_handcalc_and_negative_fair():
    # Only one time pair. z=[0,2], observed difference**.5=1.
    x, y = np.array([[[0., 0.], [0., 4.]]]), np.array([[0., 1.]])
    scores = variogram_scores(x, y)
    np.testing.assert_allclose(scores['variogram_plugin'], [0.])
    np.testing.assert_allclose(scores['variogram_correction'], [1.])
    np.testing.assert_allclose(scores['variogram_fair'], [-1.])


def test_variogram_unbiased_by_exhaustive_iid_ensemble_enumeration():
    # Enumerate every iid member draw from a two-atom forecast. Population
    # E[z]=1, observed a=0, so population score is 1; plugin bias is Var(z)/M.
    for m in (2, 3, 4):
        ensembles = np.array([[[0., 4.*bit] for bit in bits]
                              for bits in itertools.product((0, 1), repeat=m)])
        y = np.zeros((len(ensembles), 2))
        scores = variogram_scores(ensembles, y)
        np.testing.assert_allclose(scores['variogram_fair'].mean(), 1.)
        np.testing.assert_allclose(scores['variogram_plugin'].mean(), 1.+1./m)


def test_variogram_explicit_pairs_permutations_time_reversal_and_scale():
    rng = np.random.default_rng(41)
    x, y = rng.normal(size=(3, 5, 6)), rng.normal(size=(3, 6))
    expected_plugin, expected_correction = [], []
    for xx, yy in zip(x, y):
        pairs = [(i, j) for i in range(6) for j in range(i+1, 6)]
        norm = sum(1/(j-i) for i, j in pairs)
        plugin = correction = 0.
        for i, j in pairs:
            z = np.sqrt(np.abs(xx[:, i]-xx[:, j]))
            w = 1/(j-i)/norm
            plugin += w*(np.sqrt(abs(yy[i]-yy[j]))-z.mean())**2
            correction += w*z.var(ddof=1)/5
        expected_plugin.append(plugin)
        expected_correction.append(correction)
    baseline = variogram_scores(x, y, context_batch=1)
    np.testing.assert_allclose(baseline['variogram_plugin'], expected_plugin)
    np.testing.assert_allclose(baseline['variogram_correction'], expected_correction)
    for transformed in (variogram_scores(x[:, ::-1]+7, y+7),
                        variogram_scores(x[:, :, ::-1], y[:, ::-1])):
        for key in baseline:
            np.testing.assert_allclose(transformed[key], baseline[key], atol=1e-14)
    for key, value in variogram_scores(3*x, 3*y).items():
        np.testing.assert_allclose(value, 3*baseline[key])  # exponent 2*p=1


def test_crps_energy_explicit_pair_sums_and_scaling():
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=(3, 5, 4)), rng.normal(size=(3, 4))
    for function, distance, prefix in ((crps_scores, lambda a: np.abs(a).mean(), 'crps'),
                                      (energy_scores, np.linalg.norm, 'energy')):
        expected = []
        for xx, yy in zip(x, y):
            first = np.mean([distance(member-yy) for member in xx])
            pairs = sum(distance(xx[i]-xx[j]) for i in range(5) for j in range(i+1, 5))
            expected.append((first-pairs/25, first-pairs/20))
        actual = function(x, y)
        np.testing.assert_allclose(actual[prefix+'_plugin'], np.array(expected)[:, 0])
        fair_key = 'fair_crps' if prefix == 'crps' else 'fair_energy_score'
        np.testing.assert_allclose(actual[fair_key], np.array(expected)[:, 1])
        np.testing.assert_allclose(function(-2*x+7, -2*y+7)[fair_key], 2*actual[fair_key])


def test_identical_paths_zero_score_and_bootstrap_reproducibility():
    y = np.array([[0., 2., 1.], [3., 0., 2.]])
    x = np.repeat(y[:, None], 4, axis=1)
    scores = trajectory_scores(x, y)
    for key in ('variogram_plugin', 'variogram_fair', 'variogram_correction',
                'crps_plugin', 'fair_crps', 'energy_plugin', 'fair_energy_score'):
        np.testing.assert_allclose(scores[key], 0, atol=1e-15)
    paired = paired_context_bootstrap(scores, scores, repetitions=100)
    assert paired == paired_context_bootstrap(scores, scores, repetitions=100)
    assert all(v == {'difference': 0., 'context_ci95': [0., 0.]} for v in paired.values())


@pytest.mark.parametrize('shape', [(2, 1, 3), (0, 2, 3), (2, 2, 1), (2, 3)])
def test_reject_bad_ensemble_shapes(shape):
    with pytest.raises(ValueError):
        variogram_scores(np.zeros(shape), np.zeros((2, 3)))


def test_reject_nonfinite_and_invalid_order():
    with pytest.raises(ValueError):
        trajectory_scores(np.full((1, 2, 3), np.nan), np.zeros((1, 3)))
    with pytest.raises(ValueError):
        roughness([[0., 1.]], 2)
    with pytest.raises(ValueError):
        variogram_scores(np.zeros((1, 2, 3)), np.zeros((1, 3)), p=0)
