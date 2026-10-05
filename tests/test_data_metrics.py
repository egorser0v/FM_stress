"""Independent known-value checks, no GPU or external model needed."""
import unittest
import numpy as np
from fm_stress.data import (GPConfig, sundial_normalize, sample_dataset,
                            sample_source, source_covariance, interpolate,
                            conditional_target, conditional_oracle_velocity,
                            conditional_oracle_noise, sample_conditional_target)
from fm_stress.metrics import (PCA, fit_pca, spectrum, roughness,
                               velocity_metrics, success_rule, evaluate)


class TestData(unittest.TestCase):
    def test_context_only_population_normalization_and_constant_fallback(self):
        h = np.array([[1., 3.], [5., 5.], [0., .01]])
        y = np.array([[5., 7.], [6., 9.], [1., 2.]])
        hn, yn, mean, scale = sundial_normalize(h, y)
        np.testing.assert_allclose(scale[:, 0], [1., 1., 1.])
        np.testing.assert_allclose(hn[0], [-1., 1.])
        np.testing.assert_allclose(yn, [[3., 5.], [1., 4.], [.995, 1.995]])
        np.testing.assert_array_equal(sundial_normalize(h, y * 100)[0], hn)

    def test_joint_gp_reproducibility_and_normalization_roundtrip(self):
        c = GPConfig()
        a = sample_dataset(c, 500, 111)
        b = sample_dataset(c, 500, 111)
        other = sample_dataset(c, 500, 112)
        np.testing.assert_array_equal(a.target, b.target)
        self.assertFalse(np.array_equal(a.target, other.target))
        np.testing.assert_allclose(a.unnormalize(a.target), a.raw_target, atol=1e-14)
        # Adjacent across lookback/future boundary should be strongly correlated.
        self.assertGreater(np.corrcoef(a.raw_context[:, -1], a.raw_target[:, 0])[0, 1], .95)

    def test_independent_source_and_matched_kernel(self):
        c = GPConfig()
        d = sample_dataset(c, 12000, 71)
        eps = sample_source(c, 12000, "gp", 72)
        self.assertLess(abs(np.corrcoef(d.target[:, 0], eps[:, 0])[0, 1]), .04)
        np.testing.assert_allclose(np.cov(eps.T), source_covariance(c, "gp"), atol=.04)
        white = sample_source(c, 12000, "white", 73)
        self.assertLess(roughness(eps), roughness(white) / 5)

    def test_interpolant_endpoints(self):
        y = np.array([[2., 4., 6.], [1., 3., 5.]])
        e = np.ones_like(y)
        x, u = interpolate(y, e, [0., 1.])
        np.testing.assert_array_equal(x, [e[0], y[1]])
        np.testing.assert_array_equal(u, y - e)

    def test_analytic_scalar_oracle_embedded_diagonal_case(self):
        # y~N(2,4), e~N(0,1), t=.5: E[u|x]=2+1.2*(x-1).
        x = np.array([[0., 1., 2.], [3., 4., 5.]])
        mu = np.full_like(x, 2.)
        cy = np.broadcast_to(4 * np.eye(3), (2, 3, 3))
        ce = np.eye(3)
        pred = conditional_oracle_velocity(x, .5, mu, cy, ce)
        np.testing.assert_allclose(pred, 2 + 1.2 * (x - 1))
        noise = conditional_oracle_noise([.5, .5], cy, ce)
        np.testing.assert_allclose(noise, np.broadcast_to(3.2 * np.eye(3), (2, 3, 3)))
        np.testing.assert_allclose(conditional_oracle_velocity(x, 0., mu, cy, ce), mu - x)
        np.testing.assert_allclose(conditional_oracle_velocity(x, 1., mu, cy, ce), x)

    def test_conditional_samples_match_analytic_distribution(self):
        c = GPConfig(lookback=3, horizon=3, lengthscale=2.)
        d = sample_dataset(c, 1, 84)
        mu, cov = conditional_target(c, d)
        draws = sample_conditional_target(c, d, 25000, 85)[0]
        self.assertTrue(np.all(np.abs(draws.mean(0) - mu[0]) < 5 * np.sqrt(np.diag(cov[0]) / len(draws))))
        np.testing.assert_allclose(np.cov(draws.T), cov[0], rtol=.06, atol=.03)


class TestMetrics(unittest.TestCase):
    def test_projection_mse_does_not_remove_bias(self):
        q = np.array([[1., 1., 0.], [-1., 1., 0.], [0., 0., np.sqrt(2)]]) / np.sqrt(2)
        pca = PCA(np.array([7., 9., 11.]), q, np.array([4., 2., 1.]), 1e-8)
        # Constant biased error in PC coordinates is [2,3,4], not zero.
        pred = np.tile(np.array([2., 3., 4.]) @ q, (5, 1))
        result = velocity_metrics(pred, np.zeros_like(pred), pca)
        np.testing.assert_allclose(result['pc_mse'], [4., 9., 16.])
        self.assertAlmostEqual(result['residual_mse'], 16.)
        self.assertAlmostEqual(result['pc1_normalized_mse'], 1.)

    def test_residual_is_mean_per_coordinate_not_sum(self):
        pca = PCA(np.zeros(4), np.eye(4), np.ones(4), 1e-8)
        result = velocity_metrics([[1., 2., 3., 5.]], [[0., 0., 0., 0.]], pca)
        self.assertEqual(result['residual_mse'], 17.)
        self.assertEqual(result['pc2_mse'], 4.)

    def test_whitening_inverse_and_train_covariance(self):
        rng = np.random.default_rng(41)
        x = rng.normal(size=(1000, 4)) * [4., 3., 2., 1.] + [1., 2., 3., 4.]
        pca = fit_pca(x)
        np.testing.assert_allclose(np.cov(pca.whiten(x).T), np.eye(4), atol=1e-12)
        np.testing.assert_allclose(pca.unwhiten(pca.whiten(x)), x, atol=1e-12)
        np.testing.assert_allclose(pca.unwhiten(pca.whiten(x, False), False), x, atol=1e-12)
        self.assertGreater(pca.eigenvalues[0], pca.eigenvalues[-1])

    def test_heldout_energy_uses_train_axes(self):
        pca = PCA(np.zeros(3), np.eye(3), np.array([9., 4., 1.]), 1e-8)
        x = np.array([[1., 1., 10.], [-1., -1., -10.]])
        report = spectrum(x, pca)
        self.assertAlmostEqual(report['top2_energy'], 2 / 102)

    def test_roughness_known_values(self):
        self.assertEqual(roughness([[0., 1., 3.], [3., 3., 0.]]), 1.5)
        self.assertEqual(roughness(np.ones((3, 4, 16))), 0.)

    def test_literal_success_including_smaller_error_failure(self):
        self.assertTrue(success_rule(1., .8, 1.2, 1.)['works'])
        self.assertTrue(success_rule(1., 1.2, .8, 1.)['works'])
        self.assertFalse(success_rule(1., .1, 1., 1.)['works'])
        self.assertFalse(success_rule(0., 0., 0., 0.)['works'])
        self.assertIsNone(success_rule(0., 0., 0., 0.)['residual_to_pc1'])

    def test_default_geometry_passes_without_constant_collapse(self):
        c = GPConfig()
        train = sample_dataset(c, 4000, 801)
        valid = sample_dataset(c, 4000, 802)
        gps_train = sample_source(c, len(train), 'gp', 803)
        gps_valid = sample_source(c, len(valid), 'gp', 804)
        pca = fit_pca(train.target - gps_train)
        report = spectrum(valid.target - gps_valid, pca)
        self.assertGreater(report['top2_energy'], .9)
        self.assertGreater(report['mean_within_patch_std'], .1)
        white_pca = fit_pca(train.target - sample_source(c, len(train), 'white', 805))
        white_report = spectrum(valid.target - sample_source(c, len(valid), 'white', 806), white_pca)
        self.assertLess(white_report['top2_energy'], .9)


if __name__ == '__main__':
    unittest.main()
