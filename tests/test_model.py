"""
Tests for pigeoneer.model — no Julia required.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

# Expected keys in a decoded parameter dict from ZINBLogDensity.decode_params().
_DECODE_KEYS = frozenset(
    ("A_tril", "mu", "phi", "pi_zero", "gamma_mu", "gamma_phi", "gamma_pi", "omega")
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_X(n_samples=20, n_features=4, seed=42):
    rng = np.random.default_rng(seed)
    return rng.negative_binomial(3, 0.5, size=(n_samples, n_features)).astype(float)


def _default_model(n_features=4, seed=42):
    from pigeoneer.model import ZINBLogDensity
    X = _make_X(n_features=n_features, seed=seed)
    return ZINBLogDensity(X, n_features=n_features)


# ---------------------------------------------------------------------------
# _nb_log_prob
# ---------------------------------------------------------------------------

class TestNBLogProb:
    def test_finite_for_valid_inputs(self):
        from pigeoneer.model import _nb_log_prob
        x = np.array([0.0, 1.0, 5.0, 10.0])
        lp = _nb_log_prob(x, mu=3.0, phi=2.0)
        assert np.all(np.isfinite(lp)), "NB log prob should be finite for valid inputs"

    def test_non_positive_log_prob(self):
        from pigeoneer.model import _nb_log_prob
        x = np.array([1.0, 2.0, 3.0])
        lp = _nb_log_prob(x, mu=2.0, phi=1.0)
        assert np.all(lp <= 0), "Log probabilities must be ≤ 0"

    def test_x_zero_higher_prob_with_high_pi(self):
        """P(x=0) should increase as mu decreases."""
        from pigeoneer.model import _nb_log_prob
        lp_low_mu = _nb_log_prob(np.array([0.0]), mu=0.1, phi=1.0)
        lp_high_mu = _nb_log_prob(np.array([0.0]), mu=10.0, phi=1.0)
        assert lp_low_mu > lp_high_mu

    def test_broadcast_shapes(self):
        from pigeoneer.model import _nb_log_prob
        x = np.ones((10, 4))
        mu = np.array([1.0, 2.0, 3.0, 4.0])
        phi = np.array([1.0, 1.0, 1.0, 1.0])
        lp = _nb_log_prob(x, mu[np.newaxis, :], phi[np.newaxis, :])
        assert lp.shape == (10, 4)


# ---------------------------------------------------------------------------
# _zinb_log_prob
# ---------------------------------------------------------------------------

class TestZINBLogProb:
    def test_finite_for_zero_counts(self):
        from pigeoneer.model import _zinb_log_prob
        x = np.zeros(5)
        lp = _zinb_log_prob(x, mu=2.0, phi=1.0, pi_zero=0.3)
        assert np.all(np.isfinite(lp))

    def test_finite_for_positive_counts(self):
        from pigeoneer.model import _zinb_log_prob
        x = np.array([1.0, 3.0, 7.0])
        lp = _zinb_log_prob(x, mu=3.0, phi=2.0, pi_zero=0.1)
        assert np.all(np.isfinite(lp))
        assert np.all(lp <= 0)

    def test_high_pi_inflates_zero_prob(self):
        """P(x=0) should be higher when π is high."""
        from pigeoneer.model import _zinb_log_prob
        x = np.array([0.0])
        lp_high = _zinb_log_prob(x, mu=3.0, phi=1.0, pi_zero=0.9)
        lp_low = _zinb_log_prob(x, mu=3.0, phi=1.0, pi_zero=0.01)
        assert lp_high > lp_low

    def test_broadcast_shapes(self):
        from pigeoneer.model import _zinb_log_prob
        x = np.ones((10, 4))
        mu = np.ones((1, 4)) * 2.0
        phi = np.ones((1, 4))
        pi = np.ones((1, 4)) * 0.2
        lp = _zinb_log_prob(x, mu, phi, pi)
        assert lp.shape == (10, 4)


# ---------------------------------------------------------------------------
# _build_omega
# ---------------------------------------------------------------------------

class TestBuildOmega:
    def test_symmetry(self):
        from pigeoneer.model import _build_omega
        n = 5
        k = n * (n - 1) // 2
        A = np.random.default_rng(0).standard_normal(k)
        Omega = _build_omega(A, n)
        assert np.allclose(Omega, Omega.T), "Ω must be symmetric"

    def test_unit_diagonal(self):
        from pigeoneer.model import _build_omega
        n = 4
        k = n * (n - 1) // 2
        A = np.random.default_rng(1).standard_normal(k)
        Omega = _build_omega(A, n)
        assert np.allclose(np.diag(Omega), 1.0), "Diagonal of Ω must be 1"

    def test_shape(self):
        from pigeoneer.model import _build_omega
        n = 6
        k = n * (n - 1) // 2
        A = np.zeros(k)
        Omega = _build_omega(A, n)
        assert Omega.shape == (n, n)

    def test_off_diagonal_values(self):
        """Check that A_tril values end up in the correct off-diagonal positions."""
        from pigeoneer.model import _build_omega
        # For n=3: A_tril = [Ω_{10}, Ω_{20}, Ω_{21}]
        A = np.array([-1.0, 2.0, 0.5])
        Omega = _build_omega(A, 3)
        assert Omega[1, 0] == -1.0 and Omega[0, 1] == -1.0
        assert Omega[2, 0] == 2.0 and Omega[0, 2] == 2.0
        assert Omega[2, 1] == 0.5 and Omega[1, 2] == 0.5


# ---------------------------------------------------------------------------
# ZINBLogDensity
# ---------------------------------------------------------------------------

class TestZINBLogDensityInit:
    def test_dim_correct(self):
        model = _default_model(n_features=4)
        expected = 4 * 3 // 2 + 3 * 4 + 3  # 6 + 12 + 3 = 21
        assert model.dim == expected

    def test_n_interactions_correct(self):
        model = _default_model(n_features=5)
        assert model.n_interactions == 5 * 4 // 2  # 10

    def test_x_stored_as_float64(self):
        model = _default_model(n_features=3)
        assert model.X.dtype == np.float64


class TestZINBLogDensityLogPrior:
    def test_finite_at_initial_params(self):
        model = _default_model()
        x0 = model.initial_params()
        lp = model.log_prior(x0)
        assert np.isfinite(lp)

    def test_decreases_with_extreme_A(self):
        """Very large A_tril should give lower prior than zero A_tril."""
        model = _default_model()
        x0 = model.initial_params()
        x_extreme = x0.copy()
        x_extreme[: model.n_interactions] = 10.0
        lp0 = model.log_prior(x0)
        lp_extreme = model.log_prior(x_extreme)
        assert lp0 > lp_extreme

    def test_gamma_mu_prior_peak_at_one(self):
        """gamma_mu prior is Normal(1, 0.5), so lp is highest at 1."""
        model = _default_model()
        x0 = model.initial_params()  # gamma_mu = 1.0

        x_off = x0.copy()
        n, k = model.n_features, model.n_interactions
        x_off[k + 3 * n] = 5.0  # gamma_mu = 5.0

        assert model.log_prior(x0) > model.log_prior(x_off)


class TestZINBLogDensityPLL:
    def test_finite(self):
        model = _default_model()
        x0 = model.initial_params()
        pll = model.pseudo_log_likelihood(x0)
        assert np.isfinite(pll)

    def test_non_positive_per_sample(self):
        """Total log-likelihood should be negative (log of values ≤ 1)."""
        model = _default_model()
        x0 = model.initial_params()
        pll = model.pseudo_log_likelihood(x0)
        assert pll <= 0

    def test_varies_with_parameters(self):
        """Different parameter values should give different log-likelihoods."""
        model = _default_model()
        x0 = model.initial_params()
        x1 = x0.copy()
        x1[model.n_interactions] += 3.0  # shift log_mu[0]
        pll0 = model.pseudo_log_likelihood(x0)
        pll1 = model.pseudo_log_likelihood(x1)
        assert pll0 != pll1


class TestZINBLogDensityFull:
    def test_finite_at_initial_params(self):
        model = _default_model()
        x0 = model.initial_params()
        ld = model.log_density(x0)
        assert np.isfinite(ld)

    def test_returns_float(self):
        model = _default_model()
        x0 = model.initial_params()
        ld = model.log_density(x0)
        assert isinstance(ld, float)

    def test_safe_with_nan_input(self):
        model = _default_model()
        x_bad = np.full(model.dim, np.nan)
        ld = model.log_density(x_bad)
        assert ld == -1e30

    def test_safe_with_inf_input(self):
        model = _default_model()
        x_bad = np.full(model.dim, np.inf)
        ld = model.log_density(x_bad)
        assert ld == -1e30


class TestZINBLogDensityDecodeParams:
    def test_returns_dict_with_expected_keys(self):
        model = _default_model()
        x0 = model.initial_params()
        params = model.decode_params(x0)
        assert set(params.keys()) >= _DECODE_KEYS, f"Missing keys: {_DECODE_KEYS - set(params.keys())}"

    def test_omega_symmetric(self):
        model = _default_model()
        rng = np.random.default_rng(7)
        x = rng.standard_normal(model.dim)
        params = model.decode_params(x)
        assert np.allclose(params["omega"], params["omega"].T)

    def test_omega_unit_diagonal(self):
        model = _default_model()
        rng = np.random.default_rng(8)
        x = rng.standard_normal(model.dim)
        params = model.decode_params(x)
        assert np.allclose(np.diag(params["omega"]), 1.0)

    def test_mu_positive(self):
        model = _default_model()
        rng = np.random.default_rng(9)
        x = rng.standard_normal(model.dim)
        params = model.decode_params(x)
        assert np.all(params["mu"] > 0)

    def test_pi_in_unit_interval(self):
        model = _default_model()
        rng = np.random.default_rng(10)
        x = rng.standard_normal(model.dim)
        params = model.decode_params(x)
        assert np.all(params["pi_zero"] > 0) and np.all(params["pi_zero"] < 1)

    def test_accepts_python_list(self):
        model = _default_model()
        x0 = model.initial_params().tolist()
        params = model.decode_params(x0)
        assert "omega" in params

    def test_initial_params_gamma_mu(self):
        model = _default_model()
        x0 = model.initial_params()
        params = model.decode_params(x0)
        assert abs(params["gamma_mu"] - 1.0) < 1e-9


class TestZINBLogDensityInitialParams:
    def test_correct_length(self):
        model = _default_model(n_features=3)
        x0 = model.initial_params()
        assert len(x0) == model.dim

    def test_gamma_mu_is_prior_mean(self):
        """gamma_mu initial value should equal prior.gamma_mu_mean."""
        model = _default_model()
        x0 = model.initial_params()
        n, k = model.n_features, model.n_interactions
        assert x0[k + 3 * n] == model.prior.gamma_mu_mean

    def test_a_tril_zeros(self):
        model = _default_model()
        x0 = model.initial_params()
        assert np.all(x0[: model.n_interactions] == 0.0)

    def test_custom_prior_affects_initial_params(self):
        """initial_params reflects custom gamma_mu_mean and mu_log_mean."""
        from pigeoneer.priors import ZINBPriorConfig
        from pigeoneer.model import ZINBLogDensity
        prior = ZINBPriorConfig(gamma_mu_mean=2.5, mu_log_mean=1.0)
        X = _make_X()
        model = ZINBLogDensity(X, n_features=4, prior=prior)
        x0 = model.initial_params()
        n, k = model.n_features, model.n_interactions
        assert abs(x0[k + 3 * n] - 2.5) < 1e-9       # gamma_mu
        assert np.allclose(x0[k : k + n], 1.0)        # log_mu


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class TestLoadCountMatrix:
    def test_load_npy(self):
        from pigeoneer.data import load_count_matrix
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td) / "counts.npy"
            arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
            np.save(fp, arr)
            X = load_count_matrix(str(fp))
        assert X.shape == (2, 2)
        assert X.dtype == np.float64

    def test_load_npz(self):
        from pigeoneer.data import load_count_matrix
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td) / "counts.npz"
            arr = np.array([[5, 6], [7, 8]], dtype=np.float32)
            np.savez(fp, counts=arr)
            X = load_count_matrix(str(fp))
        assert X.shape == (2, 2)

    def test_load_csv(self):
        from pigeoneer.data import load_count_matrix
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td) / "counts.csv"
            fp.write_text(",gene1,gene2,gene3\nrow1,1,2,3\nrow2,4,5,6\n")
            X = load_count_matrix(str(fp))
        assert X.shape == (2, 3)

    def test_file_not_found(self):
        from pigeoneer.data import load_count_matrix
        with pytest.raises(FileNotFoundError):
            load_count_matrix("/no/such/file.npy")

    def test_unsupported_format(self):
        from pigeoneer.data import load_count_matrix
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td) / "data.xyz"
            fp.touch()
            with pytest.raises(ValueError, match="Unsupported file format"):
                load_count_matrix(str(fp))

    def test_1d_npy_becomes_column(self):
        from pigeoneer.data import load_count_matrix
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td) / "v.npy"
            np.save(fp, np.array([1.0, 2.0, 3.0]))
            X = load_count_matrix(str(fp))
        assert X.ndim == 2 and X.shape[1] == 1


# ---------------------------------------------------------------------------
# ZINBPriorConfig
# ---------------------------------------------------------------------------

class TestZINBPriorConfig:
    def test_defaults_are_original_values(self):
        """Default prior config reproduces the original hardcoded priors."""
        from pigeoneer.priors import ZINBPriorConfig
        p = ZINBPriorConfig()
        assert p.a_tril_scale == 0.1
        assert p.mu_log_mean == 0.0 and p.mu_log_scale == 1.0
        assert p.phi_log_mean == 0.0 and p.phi_log_scale == 1.0
        assert p.pi_alpha == 1.0 and p.pi_beta == 1.0
        assert p.gamma_mu_mean == 1.0 and p.gamma_mu_scale == 0.5
        assert p.gamma_phi_mean == 0.0 and p.gamma_phi_scale == 0.5
        assert p.gamma_pi_mean == 0.0 and p.gamma_pi_scale == 0.5

    def test_custom_values_stored(self):
        from pigeoneer.priors import ZINBPriorConfig
        p = ZINBPriorConfig(
            a_tril_scale=0.5,
            mu_log_mean=1.0,
            mu_log_scale=2.0,
            pi_alpha=2.0,
            pi_beta=5.0,
            gamma_mu_mean=0.0,
        )
        assert p.a_tril_scale == 0.5
        assert p.mu_log_mean == 1.0
        assert p.mu_log_scale == 2.0
        assert p.pi_alpha == 2.0
        assert p.pi_beta == 5.0
        assert p.gamma_mu_mean == 0.0

    def test_invalid_scale_raises(self):
        from pigeoneer.priors import ZINBPriorConfig
        with pytest.raises(ValueError, match="a_tril_scale"):
            ZINBPriorConfig(a_tril_scale=-0.1)

    def test_invalid_pi_alpha_raises(self):
        from pigeoneer.priors import ZINBPriorConfig
        with pytest.raises(ValueError, match="pi_alpha"):
            ZINBPriorConfig(pi_alpha=0.0)

    def test_log_prior_uses_custom_prior(self):
        """Custom prior shifts the peak of the log prior."""
        from pigeoneer.priors import ZINBPriorConfig
        from pigeoneer.model import ZINBLogDensity
        X = _make_X()
        # Prior with gamma_mu centered at 2.0
        prior_custom = ZINBPriorConfig(gamma_mu_mean=2.0)
        prior_default = ZINBPriorConfig()  # gamma_mu centered at 1.0

        model_c = ZINBLogDensity(X, n_features=4, prior=prior_custom)
        model_d = ZINBLogDensity(X, n_features=4, prior=prior_default)

        # At gamma_mu=2 custom prior should score higher than default prior
        x_at_2 = model_c.initial_params()  # starts at gamma_mu_mean = 2.0
        lp_custom = model_c.log_prior(x_at_2)
        lp_default = model_d.log_prior(x_at_2)
        assert lp_custom > lp_default

    def test_legacy_prior_a_scale_kwarg(self):
        """prior_a_scale kwarg is still accepted for backward compatibility."""
        from pigeoneer.model import ZINBLogDensity
        X = _make_X()
        model = ZINBLogDensity(X, n_features=4, prior_a_scale=0.5)
        assert model.prior.a_tril_scale == 0.5

    def test_prior_stored_on_model(self):
        from pigeoneer.priors import ZINBPriorConfig
        from pigeoneer.model import ZINBLogDensity
        p = ZINBPriorConfig(a_tril_scale=0.2)
        X = _make_X()
        model = ZINBLogDensity(X, n_features=4, prior=p)
        assert model.prior is p

    def test_custom_beta_prior_changes_log_prior(self):
        """Beta(2, 5) prior differs from Beta(1, 1) uniform."""
        from pigeoneer.priors import ZINBPriorConfig
        from pigeoneer.model import ZINBLogDensity
        X = _make_X()
        model_uniform = ZINBLogDensity(X, n_features=4, prior=ZINBPriorConfig(pi_alpha=1.0, pi_beta=1.0))
        model_beta = ZINBLogDensity(X, n_features=4, prior=ZINBPriorConfig(pi_alpha=2.0, pi_beta=5.0))
        x0 = model_uniform.initial_params()
        lp_u = model_uniform.log_prior(x0)
        lp_b = model_beta.log_prior(x0)
        assert lp_u != lp_b
