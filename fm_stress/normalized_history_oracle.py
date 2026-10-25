"""Information-matched Bayes field for history-normalized Gaussian windows.

Only normalized history is observed. Integrating its unknown raw location and
scale gives a Student-t target (or a threshold-truncated scale mixture). The
Gaussian-source flow field is therefore a mixture, NOT a linear Gaussian oracle.
All calculations are float64 and retain the original kernel's nugget.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh
from scipy.special import roots_genlaguerre, gammaln, gammainc, gammaincc, gammaincinv, logsumexp
from numpy.polynomial.legendre import leggauss


@dataclass
class HistoryPosterior:
    mean: np.ndarray
    rate_twice: np.ndarray
    small_scale_branch: np.ndarray
    omitted_scale_probability: np.ndarray


class NormalizedHistoryOracle:
    """Exact conditioning law; finite quadrature approximates its flow field.

    For H=m*1+s*z, z has zero mean and population variance one. Put
    P=Khh^-1, c=1'P1, b=1'Pz, a=z'Pz-b²/c, A=Kyh P,
    d=A1-1 and V=Kyy-A Khy+dd'/c. Then
      s² | z ~ Gamma((L-1)/2, rate=a/2), restricted to s>threshold,
      Y_normalized | z,s ~ N(Az-d*b/c, V/s²).
    The untruncated target is t_(L-1)(mean, a*V/(L-1)).
    If the observed normalized history has std<=threshold, the normalizer used
    scale=1: its centered raw history is observed and the target is N(mean,V).
    """
    def __init__(self, joint_covariance, lookback, source_covariance, threshold=.01):
        joint = np.asarray(joint_covariance, dtype=np.float64)
        self.lookback = int(lookback)
        self.horizon = joint.shape[0] - lookback
        self.threshold = float(threshold)
        self.shape = (lookback - 1) / 2
        if lookback < 4 or self.horizon < 1 or threshold <= 0:
            raise ValueError('Need L>=4, F>=1, threshold>0')
        kh = joint[:lookback, :lookback]
        self.P = np.linalg.solve(kh, np.eye(lookback))
        self.P = (self.P + self.P.T) / 2
        ones = np.ones(lookback)
        self.p1 = self.P @ ones
        self.c = float(ones @ self.p1)
        self.A = np.linalg.solve(kh, joint[:lookback, lookback:]).T
        self.d = self.A @ ones - 1
        residual = joint[lookback:, lookback:] - self.A @ joint[:lookback, lookback:]
        self.V = residual + np.outer(self.d, self.d) / self.c
        self.V = (self.V + self.V.T) / 2
        self.C = np.asarray(source_covariance, dtype=np.float64)
        np.linalg.cholesky(self.V)
        np.linalg.cholesky(self.C)
        self.eigenvalues, self.E = eigh(self.V, self.C)
        if np.min(self.eigenvalues) <= 0:
            raise ValueError('Target mixture covariance is not positive definite')
        self.T = self.C @ self.E  # inverse of E.T; x_row -> x_row@E
        self.logdet_source = np.linalg.slogdet(self.C)[1]
        self._rules = {}

    def posterior(self, history):
        h = np.asarray(history, dtype=np.float64)
        if h.ndim != 2 or h.shape[1] != self.lookback or not np.isfinite(h).all():
            raise ValueError('history must be finite [N,L]')
        if np.max(np.abs(h.mean(1))) > 1e-7:
            raise ValueError('History must be centered by its own mean')
        std = h.std(1)
        small = std <= self.threshold * (1 + 1e-8)
        if np.any(~small & (np.abs(std - 1) > 1e-6)):
            raise ValueError('History does not follow the specified normalization')
        b = h @ self.p1
        a = np.einsum('ni,ij,nj->n', h, self.P, h) - b*b/self.c
        if np.any(a[~small] <= 0):
            raise ValueError('Nonpositive posterior scale rate')
        mean = h @ self.A.T - b[:, None] * self.d[None] / self.c
        tail = gammainc(self.shape, np.maximum(a, 0) * self.threshold**2 / 2)
        tail[small] = 0
        return HistoryPosterior(mean, a, small, tail)

    def target_moments(self, history):
        post = self.posterior(history)
        x0 = post.rate_twice * self.threshold**2 / 2
        inv_s2 = post.rate_twice / (2 * (self.shape - 1))
        inv_s2 *= gammaincc(self.shape - 1, x0) / gammaincc(self.shape, x0)
        inv_s2[post.small_scale_branch] = 1
        return post.mean, inv_s2[:, None, None] * self.V

    def _rule(self, order):
        if order not in self._rules:
            nodes, weights = roots_genlaguerre(order, self.shape - 1)
            ln = np.log(np.maximum(weights, np.finfo(float).tiny)) - gammaln(self.shape)
            ln -= logsumexp(ln)
            self._rules[order] = nodes, ln
        return self._rules[order]

    def flow(self, x, times, history, order=64, jacobian=True, noise_covariance=False):
        """Conditional velocity and optionally its full derivative and label noise.

        Jacobian includes derivatives of posterior mixture weights. Averaging the
        component Gaussian Jacobians alone would be incorrect. Laguerre drops a
        scale-threshold tail only if probability <=1e-12; otherwise a truncated
        Gamma-quantile Gauss-Legendre rule is used. Convergence must be checked.
        """
        x = np.asarray(x, dtype=np.float64)
        h = np.asarray(history, dtype=np.float64)
        t = np.broadcast_to(np.asarray(times, dtype=np.float64), (len(x),))
        if x.shape != (len(h), self.horizon) or np.any((t < 0) | (t > 1)):
            raise ValueError('Mismatched shape or time outside [0,1]')
        post = self.posterior(h)
        nodes, logw = self._rule(order)
        inv_s2 = post.rate_twice[:, None] / (2 * nodes[None])
        logs = np.broadcast_to(logw, inv_s2.shape).copy()
        truncated = (~post.small_scale_branch) & (post.omitted_scale_probability > 1e-12)
        if np.any(truncated):
            gn, gw = leggauss(order)
            u = (gn+1)/2
            lower = post.omitted_scale_probability[truncated, None]
            quantiles = gammaincinv(self.shape, lower + (1-lower)*u)
            inv_s2[truncated] = post.rate_twice[truncated, None]/(2*quantiles)
            logs[truncated] = np.log(gw/2)
        inv_s2[post.small_scale_branch] = 1
        mean_z = post.mean @ self.E
        center = x @ self.E - t[:, None]*mean_z
        k = inv_s2[:, :, None]*self.eigenvalues[None, None]
        den = t[:, None, None]**2*k + (1-t[:, None, None])**2
        beta = (t[:, None, None]*k - (1-t[:, None, None]))/den
        v_components = mean_z[:, None] + beta*center[:, None]
        log_density = -.5*(np.log(den).sum(2) + (center[:, None]**2/den).sum(2)
                             + self.horizon*np.log(2*np.pi) + self.logdet_source)
        norm = logsumexp(logs+log_density, axis=1)
        weights = np.exp(logs+log_density-norm[:, None])
        velocity_z = np.einsum('nq,nqf->nf', weights, v_components)
        result = {'velocity': velocity_z @ self.T.T, 'log_density': norm,
                  'quadrature_order': order,
                  'threshold_tail_probability': post.omitted_scale_probability,
                  'small_scale_branch': post.small_scale_branch,
                  'truncated_quantile_rule': truncated,
                  'weight_ess': 1/(weights*weights).sum(1)}
        centered_v = v_components-velocity_z[:, None]
        if jacobian:
            score = -center[:, None]/den
            Jz = np.einsum('nq,nqi,nqj->nij', weights, centered_v, score)
            diag = np.arange(self.horizon)
            Jz[:, diag, diag] += np.einsum('nq,nqf->nf', weights, beta)
            result['jacobian'] = self.T @ Jz @ self.E.T
            component_Jz = np.zeros_like(Jz)
            component_Jz[:, diag, diag] = np.einsum('nq,nqf->nf', weights, beta)
            result['component_average_jacobian'] = self.T @ component_Jz @ self.E.T
        if noise_covariance:
            covz = np.einsum('nq,nqi,nqj->nij', weights, centered_v, centered_v)
            diag = np.arange(self.horizon)
            covz[:, diag, diag] += np.einsum('nq,nqf->nf', weights, k/den)
            result['noise_covariance'] = self.T @ covz @ self.T.T
        return result

    def sample_target(self, history, samples, seed=0):
        post = self.posterior(history)
        rng = np.random.default_rng(seed)
        lower = post.omitted_scale_probability[:, None]
        gamma = gammaincinv(self.shape, lower+(1-lower)*rng.random((len(history), samples)))
        inv_s = np.sqrt(post.rate_twice[:, None]/(2*gamma))
        inv_s[post.small_scale_branch] = 1
        z = rng.normal(size=(len(history), samples, self.horizon))
        return post.mean[:, None] + (z @ np.linalg.cholesky(self.V).T)*inv_s[:, :, None]
