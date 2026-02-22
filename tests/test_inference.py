"""
Tests for pigeoneer.inference.

Julia and juliacall are required for the integration tests, which are
marked ``@pytest.mark.slow``.  The unit tests for ``_compute_summary``
run without Julia.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# _compute_summary (no Julia required)
# ---------------------------------------------------------------------------

class TestComputeSummary:
    def _make_decoded(self, n_samples=20, n_features=3, seed=0):
        from pigeoneer.model import ZINBLogDensity
        rng = np.random.default_rng(seed)
        X = rng.negative_binomial(3, 0.5, size=(50, n_features)).astype(float)
        model = ZINBLogDensity(X, n_features=n_features)
        samples = [
            model.decode_params(rng.standard_normal(model.dim))
            for _ in range(n_samples)
        ]
        omega_samples = np.stack([s["omega"] for s in samples])
        return samples, omega_samples, n_features

    def test_keys_present(self):
        from pigeoneer.inference import _compute_summary
        decoded, omega_samples, _ = self._make_decoded()
        summary = _compute_summary(decoded, omega_samples)
        for key in ("gamma_mu", "gamma_phi", "gamma_pi", "mu", "phi",
                    "pi_zero", "A_tril", "omega"):
            assert key in summary, f"Missing summary key: {key}"

    def test_scalar_stats(self):
        from pigeoneer.inference import _compute_summary
        decoded, omega_samples, _ = self._make_decoded()
        summary = _compute_summary(decoded, omega_samples)
        for key in ("gamma_mu", "gamma_phi", "gamma_pi"):
            for stat in ("mean", "std", "median", "q05", "q95"):
                assert stat in summary[key], f"Missing stat '{stat}' for {key}"
            assert np.isfinite(summary[key]["mean"])

    def test_omega_shape(self):
        from pigeoneer.inference import _compute_summary
        n = 4
        decoded, omega_samples, _ = self._make_decoded(n_features=n)
        summary = _compute_summary(decoded, omega_samples)
        assert summary["omega"]["mean"].shape == (n, n)
        assert summary["omega"]["std"].shape == (n, n)

    def test_empty_decoded_returns_empty(self):
        from pigeoneer.inference import _compute_summary
        summary = _compute_summary([], np.empty((0, 3, 3)))
        assert summary == {}


# ---------------------------------------------------------------------------
# run_pigeons_inference (requires Julia + Pigeons.jl — marked slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestRunPigeonsInference:
    """Integration tests that require juliacall and a Julia installation."""

    @pytest.fixture(autouse=True)
    def skip_without_juliacall(self):
        juliacall = pytest.importorskip(
            "juliacall",
            reason="juliacall not installed — skipping Pigeons.jl integration tests",
        )

    def _make_model(self, n_samples=30, n_features=3, seed=42):
        from pigeoneer.model import ZINBLogDensity
        rng = np.random.default_rng(seed)
        X = rng.negative_binomial(3, 0.5, size=(n_samples, n_features)).astype(float)
        return ZINBLogDensity(X, n_features=n_features)

    def test_result_keys(self):
        from pigeoneer.inference import run_pigeons_inference
        model = self._make_model()
        results = run_pigeons_inference(model, n_rounds=3, n_chains=2, seed=1)
        for key in ("samples", "decoded_samples", "omega_samples",
                    "summary", "log_marginal_likelihood", "pt"):
            assert key in results, f"Missing result key: {key}"

    def test_sample_shape(self):
        from pigeoneer.inference import run_pigeons_inference
        model = self._make_model(n_features=3)
        results = run_pigeons_inference(model, n_rounds=3, n_chains=2, seed=1)
        # Total samples = 2 ** n_rounds
        n_expected = 2 ** 3
        assert results["samples"].shape == (n_expected, model.dim)

    def test_omega_samples_shape(self):
        from pigeoneer.inference import run_pigeons_inference
        model = self._make_model(n_features=3)
        results = run_pigeons_inference(model, n_rounds=3, n_chains=2, seed=1)
        n_expected = 2 ** 3
        assert results["omega_samples"].shape == (n_expected, 3, 3)

    def test_omega_symmetric(self):
        from pigeoneer.inference import run_pigeons_inference
        model = self._make_model(n_features=3)
        results = run_pigeons_inference(model, n_rounds=3, n_chains=2, seed=1)
        omega_mean = results["summary"]["omega"]["mean"]
        assert np.allclose(omega_mean, omega_mean.T, atol=1e-8)

    def test_log_marginal_likelihood_finite(self):
        from pigeoneer.inference import run_pigeons_inference
        model = self._make_model()
        results = run_pigeons_inference(model, n_rounds=3, n_chains=2, seed=1)
        log_z = results["log_marginal_likelihood"]
        assert np.isfinite(log_z)

    def test_decoded_samples_count(self):
        from pigeoneer.inference import run_pigeons_inference
        model = self._make_model()
        results = run_pigeons_inference(model, n_rounds=3, n_chains=2, seed=1)
        n_expected = 2 ** 3
        assert len(results["decoded_samples"]) == n_expected

    def test_mu_positive_in_decoded(self):
        from pigeoneer.inference import run_pigeons_inference
        model = self._make_model()
        results = run_pigeons_inference(model, n_rounds=3, n_chains=2, seed=1)
        for ds in results["decoded_samples"]:
            assert np.all(ds["mu"] > 0)

    def test_pi_in_unit_interval(self):
        from pigeoneer.inference import run_pigeons_inference
        model = self._make_model()
        results = run_pigeons_inference(model, n_rounds=3, n_chains=2, seed=1)
        for ds in results["decoded_samples"]:
            assert np.all(ds["pi_zero"] > 0)
            assert np.all(ds["pi_zero"] < 1)
