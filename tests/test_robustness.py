import numpy as np

from fm_stress.data import conditional_oracle_velocity
from fm_stress.robustness import (covariance_diagnostics, gaussian_expected_roughness,
                                  oracle_euler_transport)


def test_oracle_transport_matches_velocity_euler():
    covariance = np.array([[[1., .3], [.3, 2.]], [[.4, .1], [.1, 3.]]])
    source = np.array([[2., .2], [.2, .5]])
    mean = np.array([[1., 2.], [-.3, .7]])
    eps = np.array([[1., -2.], [.5, 1.]])
    x = eps.copy()
    steps = 17
    for j in range(steps):
        x += conditional_oracle_velocity(x, j/steps, mean, covariance, source)/steps
    transfer, endpoint_covariance = oracle_euler_transport(covariance, source, steps)
    np.testing.assert_allclose(x, mean + np.einsum("nij,nj->ni", transfer, eps), atol=1e-12)
    np.testing.assert_allclose(endpoint_covariance,
                               transfer @ source @ transfer.swapaxes(-1, -2), atol=1e-12)


def test_covariance_diagnostic_uses_generalized_eigenvalues():
    covariance = np.array([[[1., .3], [.3, 2.]], [[.4, .1], [.1, 3.]]])
    result = covariance_diagnostics(.25*covariance, covariance)
    np.testing.assert_allclose(result["whitened_covariance_eigenvalue_min"], .25)
    np.testing.assert_allclose(result["whitened_covariance_eigenvalue_max"], .25)
    assert result["fraction_conditional_directions_below_80pct_variance"] == 1


def test_expected_gaussian_roughness_has_correct_known_limits():
    # Independent unit-variance points: adjacent difference is N(0, 2).
    mean = np.zeros((2, 4))
    covariance = np.broadcast_to(np.eye(4), (2, 4, 4))
    np.testing.assert_allclose(gaussian_expected_roughness(mean, covariance), 2/np.sqrt(np.pi))
    # Deterministic straight line: each jump is 2.
    mean = np.array([[0., 2., 4., 6.], [1., -1., -3., -5.]])
    np.testing.assert_allclose(gaussian_expected_roughness(mean, np.zeros((2, 4, 4))), 2.)
