"""
ZINB Graphical Model log density for Pigeons.jl parallel tempering.

Implements the Zero-Inflated Negative Binomial (ZINB) pseudo-likelihood
graphical model as a log density suitable for use with Pigeons.jl.

Parameters are encoded in an unconstrained real-valued vector:
    [A_tril, log_mu, log_phi, logit_pi, gamma_mu, gamma_phi, gamma_pi]

where:
    A_tril   : n*(n-1)/2 entries — lower-triangular off-diagonal of Ω
    log_mu   : n entries — log of ZINB mean μ  (μ = exp(log_mu))
    log_phi  : n entries — log of ZINB dispersion φ (φ = exp(log_phi))
    logit_pi : n entries — logit of zero-inflation π (π = sigmoid(logit_pi))
    gamma_mu : 1 — scaling factor for μ interactions
    gamma_phi: 1 — scaling factor for φ interactions
    gamma_pi : 1 — scaling factor for π interactions

Greek Parameters:
    Ω (Omega)   : Symmetric interaction/precision matrix with off-diagonal
                  entries and unit diagonal.
    γ_μ         : Scaling factor for how Ω affects μ.
    γ_φ         : Scaling factor for how Ω affects φ.
    γ_π         : Scaling factor for how Ω affects π.
    μ (mu)      : Mean parameter of the ZINB distribution per feature.
    φ (phi)     : Dispersion parameter of the ZINB distribution per feature.
    π (pi_zero) : Zero-inflation probability per feature. Range [0, 1].
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, expit, betaln

from .priors import ZINBPriorConfig


def _nb_log_prob(x: np.ndarray, mu: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Negative binomial log probability.

    NB(x | μ, φ) where φ is the dispersion parameter (also called 'size').

    log P(x | φ, μ) = log Γ(x+φ) − log Γ(φ) − log Γ(x+1)
                    + φ · log(φ/(φ+μ)) + x · log(μ/(φ+μ))

    Args:
        x: Observed counts (non-negative), broadcastable array.
        mu: Mean parameter μ > 0, broadcastable array.
        phi: Dispersion parameter φ > 0, broadcastable array.

    Returns:
        Log probability array matching the broadcasted shape of x, mu, phi.
    """
    mu = np.clip(mu, 1e-8, 1e8)
    phi = np.clip(phi, 1e-8, 1e8)
    log_p = (
        gammaln(x + phi) - gammaln(phi) - gammaln(x + 1)
        + phi * np.log(phi / (phi + mu))
        + x * np.log(mu / (phi + mu))
    )
    return log_p


def _zinb_log_prob(
    x: np.ndarray,
    mu: np.ndarray,
    phi: np.ndarray,
    pi_zero: np.ndarray,
) -> np.ndarray:
    """Zero-Inflated Negative Binomial log probability.

    ZINB(x | μ, φ, π) = π · I(x=0) + (1−π) · NB(x | μ, φ)

    Args:
        x: Observed counts, broadcastable array.
        mu: Mean parameter μ > 0.
        phi: Dispersion parameter φ > 0.
        pi_zero: Zero-inflation probability π ∈ [0, 1].

    Returns:
        Log probability array matching the broadcasted shape.
    """
    pi = np.clip(pi_zero, 1e-8, 1.0 - 1e-8)
    log_pi = np.log(pi)
    log_1mpi = np.log(1.0 - pi)
    nb_lp = _nb_log_prob(x, mu, phi)

    # x = 0: log(π + (1−π) · NB(0))
    log_prob_zero = np.logaddexp(log_pi, log_1mpi + nb_lp)
    # x > 0: log(1−π) + log_NB(x)
    log_prob_pos = log_1mpi + nb_lp

    return np.where(x == 0, log_prob_zero, log_prob_pos)


def _build_omega(A_tril: np.ndarray, n_features: int) -> np.ndarray:
    """Build symmetric interaction matrix Ω from lower-triangular parameters.

    Off-diagonal: Ω_ij = A_tril[k] for i > j.
    Diagonal:     Ω_ii = 1 (fixed for identifiability).

    Args:
        A_tril: Lower-triangular parameters (excluding diagonal),
                length n_features*(n_features−1)/2.
        n_features: Number of features p.

    Returns:
        Omega: Symmetric (p × p) interaction matrix with unit diagonal.
    """
    Omega = np.zeros((n_features, n_features))
    idx = 0
    for i in range(n_features):
        for j in range(i):
            val = A_tril[idx]
            Omega[i, j] = val
            Omega[j, i] = val
            idx += 1
    np.fill_diagonal(Omega, 1.0)
    return Omega


class ZINBLogDensity:
    """Unnormalised log posterior for the ZINB pseudo-likelihood graphical model.

    Computes log π(x) = log prior(x) + log pseudo-likelihood(x | data)
    in the unconstrained parameter space required by Pigeons.jl.

    The parameter vector ``x`` passed to :meth:`log_density` is:

    .. code-block:: text

        index range         parameter
        ──────────────────────────────────────────────────────────────────
        0 : k               A_tril[0..k-1]  — off-diagonal entries of Ω
        k : k+n             log_mu          — log-mean parameters
        k+n : k+2n          log_phi         — log-dispersion parameters
        k+2n : k+3n         logit_pi        — logit-zero-inflation probs
        k+3n                gamma_mu        — μ interaction scaling
        k+3n+1              gamma_phi       — φ interaction scaling
        k+3n+2              gamma_pi        — π interaction scaling
        ──────────────────────────────────────────────────────────────────
        (k = n*(n-1)/2,  n = n_features,  total dim = k + 3n + 3)

    Priors are configured via :class:`~pigeoneer.ZINBPriorConfig`.  Defaults:
        A_tril[i]  ~ Normal(0, 0.1)
        log_mu[j]  ~ Normal(0, 1)          [μ ~ LogNormal(0, 1)]
        log_phi[j] ~ Normal(0, 1)          [φ ~ LogNormal(0, 1)]
        logit_pi[j]: Beta(1, 1) on π ⟹ Jacobian correction
        gamma_mu   ~ Normal(1, 0.5)
        gamma_phi  ~ Normal(0, 0.5)
        gamma_pi   ~ Normal(0, 0.5)

    Args:
        X: Count matrix of shape (n_samples, n_features).
        n_features: Number of features (genes/columns).
        prior: :class:`~pigeoneer.ZINBPriorConfig` instance with prior
            hyperparameters.  Uses default priors when ``None``.
        prior_a_scale: **Deprecated** shorthand for setting only
            ``prior.a_tril_scale``.  Ignored when ``prior`` is provided.
            Default: ``None`` (use value from ``prior``).
    """

    def __init__(
        self,
        X: np.ndarray,
        n_features: int,
        prior: ZINBPriorConfig | None = None,
        prior_a_scale: float | None = None,
    ) -> None:
        self.X = np.asarray(X, dtype=np.float64)
        self.n_features = n_features
        self.n_interactions = n_features * (n_features - 1) // 2

        # Build prior config — support legacy prior_a_scale kwarg.
        if prior is None:
            prior = ZINBPriorConfig()
            if prior_a_scale is not None:
                import dataclasses
                prior = dataclasses.replace(prior, a_tril_scale=prior_a_scale)
        self.prior = prior

        # Total dimension of the unconstrained parameter vector
        self.dim = self.n_interactions + 3 * n_features + 3

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode(self, x: np.ndarray):
        """Decode unconstrained parameter vector into named components.

        Returns:
            Tuple (A_tril, log_mu, log_phi, logit_pi,
                   mu, phi, pi_zero, gamma_mu, gamma_phi, gamma_pi)
        """
        n = self.n_features
        k = self.n_interactions

        A_tril = x[0:k]
        log_mu = x[k : k + n]
        log_phi = x[k + n : k + 2 * n]
        logit_pi = x[k + 2 * n : k + 3 * n]
        gamma_mu = float(x[k + 3 * n])
        gamma_phi = float(x[k + 3 * n + 1])
        gamma_pi = float(x[k + 3 * n + 2])

        mu = np.exp(np.clip(log_mu, -20.0, 20.0))
        phi = np.exp(np.clip(log_phi, -20.0, 20.0))
        pi_zero = expit(logit_pi)

        return (
            A_tril,
            log_mu,
            log_phi,
            logit_pi,
            mu,
            phi,
            pi_zero,
            gamma_mu,
            gamma_phi,
            gamma_pi,
        )

    # ------------------------------------------------------------------
    # Log prior
    # ------------------------------------------------------------------

    def log_prior(self, x: np.ndarray) -> float:
        """Log prior evaluated in unconstrained space.

        Includes Jacobian correction for the logit-π transformation.
        Uses hyperparameters from :attr:`prior`.

        Args:
            x: Unconstrained parameter vector of length :attr:`dim`.

        Returns:
            Scalar log prior value.
        """
        (
            A_tril,
            log_mu,
            log_phi,
            logit_pi,
            _mu,
            _phi,
            pi_zero,
            gamma_mu,
            gamma_phi,
            gamma_pi,
        ) = self._decode(x)

        p = self.prior

        # A_tril ~ Normal(0, a_tril_scale)
        lp = -0.5 * np.sum((A_tril / p.a_tril_scale) ** 2) - len(
            A_tril
        ) * np.log(p.a_tril_scale)

        # log_mu ~ Normal(mu_log_mean, mu_log_scale)  [μ ~ LogNormal]
        lp += -0.5 * np.sum(((log_mu - p.mu_log_mean) / p.mu_log_scale) ** 2) - len(
            log_mu
        ) * np.log(p.mu_log_scale)

        # log_phi ~ Normal(phi_log_mean, phi_log_scale)  [φ ~ LogNormal]
        lp += -0.5 * np.sum(((log_phi - p.phi_log_mean) / p.phi_log_scale) ** 2) - len(
            log_phi
        ) * np.log(p.phi_log_scale)

        # logit_pi: Beta(pi_alpha, pi_beta) on π with Jacobian correction.
        # log p(logit_pi) = (pi_alpha)*log(π) + (pi_beta)*log(1−π) − log B(pi_alpha, pi_beta)
        # The Jacobian of the logit transform dπ/d(logit_pi) = π(1−π) which gives log |J| = log π + log(1−π).
        # Combined: (pi_alpha)*log(π) + (pi_beta)*log(1−π)
        #           − log B(alpha,beta) + log(π) + log(1−π)
        #         = (pi_alpha+1)*log(π) + (pi_beta+1)*log(1−π) − log B(alpha,beta)
        pi_s = np.clip(pi_zero, 1e-8, 1.0 - 1e-8)
        log_beta_const = float(betaln(p.pi_alpha, p.pi_beta))
        lp += (
            p.pi_alpha * np.sum(np.log(pi_s))
            + p.pi_beta * np.sum(np.log(1.0 - pi_s))
            + np.sum(np.log(pi_s) + np.log(1.0 - pi_s))
            - len(pi_s) * log_beta_const
        )

        # gamma_mu ~ Normal(gamma_mu_mean, gamma_mu_scale)
        lp += -0.5 * ((gamma_mu - p.gamma_mu_mean) / p.gamma_mu_scale) ** 2 - np.log(
            p.gamma_mu_scale
        )

        # gamma_phi ~ Normal(gamma_phi_mean, gamma_phi_scale)
        lp += -0.5 * ((gamma_phi - p.gamma_phi_mean) / p.gamma_phi_scale) ** 2 - np.log(
            p.gamma_phi_scale
        )

        # gamma_pi ~ Normal(gamma_pi_mean, gamma_pi_scale)
        lp += -0.5 * ((gamma_pi - p.gamma_pi_mean) / p.gamma_pi_scale) ** 2 - np.log(
            p.gamma_pi_scale
        )

        return float(lp)

    # ------------------------------------------------------------------
    # Pseudo-log-likelihood
    # ------------------------------------------------------------------

    def pseudo_log_likelihood(self, x: np.ndarray) -> float:
        """ZINB pseudo-log-likelihood.

        For each feature j, conditions on all other features:

            effect_j  = Σ_{k≠j} Ω_{jk} · X_{:,k}
            μ_j^cond  = μ_j · exp(γ_μ · effect_j)
            φ_j^cond  = φ_j · exp(γ_φ · effect_j)
            π_j^cond  = sigmoid(logit(π_j) + γ_π · effect_j)
            log P(X_j | X_{-j}) = ZINB(X_j | μ_j^cond, φ_j^cond, π_j^cond)

        Args:
            x: Unconstrained parameter vector of length :attr:`dim`.

        Returns:
            Scalar pseudo-log-likelihood summed over all samples and features.
        """
        (
            A_tril,
            _lm,
            _lp,
            logit_pi,
            mu,
            phi,
            pi_zero,
            gamma_mu,
            gamma_phi,
            gamma_pi,
        ) = self._decode(x)

        Omega = _build_omega(A_tril, self.n_features)

        # Off-diagonal Ω for computing interaction effects
        Omega_od = Omega - np.diag(np.diag(Omega))  # zero diagonal
        effects = self.X @ Omega_od  # (n_samples, n_features)
        effects = np.clip(effects, -10.0, 10.0)

        # Conditional μ: log-link
        cond_mu = mu[np.newaxis, :] * np.exp(gamma_mu * effects)
        cond_mu = np.clip(cond_mu, 1e-8, 1e8)

        # Conditional φ: log-link
        cond_phi = phi[np.newaxis, :] * np.exp(gamma_phi * effects)
        cond_phi = np.clip(cond_phi, 1e-8, 1e8)

        # Conditional π: logit-link
        pi_s = np.clip(pi_zero, 1e-8, 1.0 - 1e-8)
        logit_pi_base = logit_pi[np.newaxis, :]  # broadcast over samples
        cond_logit_pi = logit_pi_base + gamma_pi * effects
        cond_logit_pi = np.clip(cond_logit_pi, -20.0, 20.0)
        cond_pi = expit(cond_logit_pi)

        log_probs = _zinb_log_prob(self.X, cond_mu, cond_phi, cond_pi)

        return float(np.sum(log_probs))

    # ------------------------------------------------------------------
    # Full log density (prior + likelihood)
    # ------------------------------------------------------------------

    def log_density(self, x) -> float:
        """Unnormalised log posterior: log prior + pseudo-log-likelihood.

        This is the function passed to Pigeons.jl as the target log density.
        Safe against NaN/Inf inputs — returns ``−1e30`` on any error.

        Args:
            x: Unconstrained parameter vector (Python list, numpy array,
               or juliacall-wrapped Julia vector).

        Returns:
            Scalar float.
        """
        try:
            x_np = np.asarray(x, dtype=np.float64)
            if not np.all(np.isfinite(x_np)):
                return -1e30
            lp = self.log_prior(x_np) + self.pseudo_log_likelihood(x_np)
            if not np.isfinite(lp):
                return -1e30
            return float(lp)
        except Exception:
            return -1e30

    # ------------------------------------------------------------------
    # Helpers for initialisation and decoding
    # ------------------------------------------------------------------

    def initial_params(self) -> np.ndarray:
        """Return a sensible initial parameter vector (prior means).

        Returns:
            Vector of length :attr:`dim` set to prior means in unconstrained
            space:

            * A_tril = 0 (Normal mean)
            * log_mu = ``prior.mu_log_mean``
            * log_phi = ``prior.phi_log_mean``
            * logit_pi = 0  (mid-point of Uniform(0,1))
            * gamma_mu = ``prior.gamma_mu_mean``
            * gamma_phi = ``prior.gamma_phi_mean``
            * gamma_pi = ``prior.gamma_pi_mean``
        """
        p = self.prior
        x0 = np.zeros(self.dim)
        k, n = self.n_interactions, self.n_features

        # log_mu and log_phi initialised to their prior means.
        x0[k : k + n] = p.mu_log_mean
        x0[k + n : k + 2 * n] = p.phi_log_mean
        # logit_pi stays at 0 (maps to π = 0.5, uninformative).

        # Gamma parameters initialised to their prior means.
        x0[k + 3 * n] = p.gamma_mu_mean
        x0[k + 3 * n + 1] = p.gamma_phi_mean
        x0[k + 3 * n + 2] = p.gamma_pi_mean

        return x0

    def decode_params(self, x) -> dict:
        """Decode an unconstrained sample vector into named parameters.

        Args:
            x: Unconstrained parameter vector.

        Returns:
            Dictionary with keys:
                ``'A_tril'``, ``'mu'``, ``'phi'``, ``'pi_zero'``,
                ``'gamma_mu'``, ``'gamma_phi'``, ``'gamma_pi'``, ``'omega'``.
        """
        x_np = np.asarray(x, dtype=np.float64)
        (
            A_tril,
            _lm,
            _lp,
            _lpi,
            mu,
            phi,
            pi_zero,
            gamma_mu,
            gamma_phi,
            gamma_pi,
        ) = self._decode(x_np)
        Omega = _build_omega(A_tril, self.n_features)
        return {
            "A_tril": A_tril,
            "mu": mu,
            "phi": phi,
            "pi_zero": pi_zero,
            "gamma_mu": float(gamma_mu),
            "gamma_phi": float(gamma_phi),
            "gamma_pi": float(gamma_pi),
            "omega": Omega,
        }
