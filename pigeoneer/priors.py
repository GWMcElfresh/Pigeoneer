"""
Prior configuration for the ZINB graphical model.

:class:`ZINBPriorConfig` is a dataclass that holds prior hyperparameters
for all model parameters.  An instance is passed to :class:`~pigeoneer.ZINBLogDensity`
to override the defaults.

Default priors match the original PreGraphModeling implementation:

+------------------+-------------------------------+
| Parameter        | Default prior                 |
+==================+===============================+
| A_tril           | Normal(0, 0.1)                |
+------------------+-------------------------------+
| μ  (log_mu)      | LogNormal(0, 1)               |
+------------------+-------------------------------+
| φ  (log_phi)     | LogNormal(0, 1)               |
+------------------+-------------------------------+
| π  (logit_pi)    | Beta(1, 1) = Uniform(0, 1)   |
+------------------+-------------------------------+
| γ_μ              | Normal(1, 0.5)                |
+------------------+-------------------------------+
| γ_φ              | Normal(0, 0.5)                |
+------------------+-------------------------------+
| γ_π              | Normal(0, 0.5)                |
+------------------+-------------------------------+

Example::

    from pigeoneer import ZINBLogDensity, ZINBPriorConfig
    import numpy as np

    prior = ZINBPriorConfig(
        a_tril_scale=0.5,          # wider prior on interaction strengths
        mu_log_mean=1.0,           # shift LogNormal mean for μ
        pi_alpha=2.0,              # Beta(2, 5) → prior toward low zero-inflation
        pi_beta=5.0,
    )

    X = np.random.negative_binomial(3, 0.5, size=(100, 10)).astype(float)
    model = ZINBLogDensity(X, n_features=10, prior=prior)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ZINBPriorConfig:
    """Prior hyperparameters for the ZINB graphical model.

    All parameters have defaults that reproduce the original PreGraphModeling
    priors; change only what you need.

    Attributes:
        a_tril_scale: Standard deviation of the Normal prior on A_tril
            (off-diagonal entries of Ω).  Smaller values shrink interactions
            toward zero.  Default: 0.1.
        mu_log_mean: Mean of the Normal prior on ``log_mu``
            (i.e., μ ~ LogNormal(mu_log_mean, mu_log_scale)).  Default: 0.0.
        mu_log_scale: Standard deviation of the Normal prior on ``log_mu``.
            Default: 1.0.
        phi_log_mean: Mean of the Normal prior on ``log_phi``
            (i.e., φ ~ LogNormal(phi_log_mean, phi_log_scale)).  Default: 0.0.
        phi_log_scale: Standard deviation of the Normal prior on ``log_phi``.
            Default: 1.0.
        pi_alpha: First shape parameter of the Beta prior on π (zero-inflation
            probability).  Default: 1.0 (= Uniform(0,1)).
        pi_beta: Second shape parameter of the Beta prior on π.
            Default: 1.0 (= Uniform(0,1)).
        gamma_mu_mean: Mean of the Normal prior on γ_μ.  Default: 1.0.
        gamma_mu_scale: Standard deviation of the Normal prior on γ_μ.
            Default: 0.5.
        gamma_phi_mean: Mean of the Normal prior on γ_φ.  Default: 0.0.
        gamma_phi_scale: Standard deviation of the Normal prior on γ_φ.
            Default: 0.5.
        gamma_pi_mean: Mean of the Normal prior on γ_π.  Default: 0.0.
        gamma_pi_scale: Standard deviation of the Normal prior on γ_π.
            Default: 0.5.
    """

    # A_tril ~ Normal(0, a_tril_scale)
    a_tril_scale: float = 0.1

    # μ ~ LogNormal(mu_log_mean, mu_log_scale)
    mu_log_mean: float = 0.0
    mu_log_scale: float = 1.0

    # φ ~ LogNormal(phi_log_mean, phi_log_scale)
    phi_log_mean: float = 0.0
    phi_log_scale: float = 1.0

    # π ~ Beta(pi_alpha, pi_beta)
    pi_alpha: float = 1.0
    pi_beta: float = 1.0

    # γ_μ ~ Normal(gamma_mu_mean, gamma_mu_scale)
    gamma_mu_mean: float = 1.0
    gamma_mu_scale: float = 0.5

    # γ_φ ~ Normal(gamma_phi_mean, gamma_phi_scale)
    gamma_phi_mean: float = 0.0
    gamma_phi_scale: float = 0.5

    # γ_π ~ Normal(gamma_pi_mean, gamma_pi_scale)
    gamma_pi_mean: float = 0.0
    gamma_pi_scale: float = 0.5

    def __post_init__(self) -> None:
        """Validate hyperparameter values."""
        for name, val in [
            ("a_tril_scale", self.a_tril_scale),
            ("mu_log_scale", self.mu_log_scale),
            ("phi_log_scale", self.phi_log_scale),
            ("gamma_mu_scale", self.gamma_mu_scale),
            ("gamma_phi_scale", self.gamma_phi_scale),
            ("gamma_pi_scale", self.gamma_pi_scale),
        ]:
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")
        for name, val in [
            ("pi_alpha", self.pi_alpha),
            ("pi_beta", self.pi_beta),
        ]:
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")
