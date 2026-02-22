"""
Pigeons.jl parallel tempering inference for the ZINB graphical model.

This module bridges Python to Julia via ``juliacall``, exposing Pigeons.jl's
parallel tempering (PT) algorithm for posterior sampling and marginal-likelihood
estimation.

The key advantage over standard MCMC is that PT simultaneously:
  * Provides posterior samples from the ZINB graphical model.
  * Estimates the log normalising constant (log marginal likelihood Z) via the
    stepping-stone estimator — without requiring a normalising constant.

Requirements
------------
* ``juliacall`` Python package (``pip install juliacall``).
* Julia ≥ 1.8 installed and on PATH (or managed by ``juliapkg``).
* Julia packages Pigeons.jl and LogDensityProblems.jl.
  These are declared in ``juliapkg.json`` and auto-installed by ``juliapkg``.

Julia bridge
------------
A thin Julia struct ``PythonLogDensity`` implements the
``LogDensityProblems`` interface by calling back into our Python log-density
function.  Pigeons.jl uses this struct as the sampling target.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .model import ZINBLogDensity

# Module-level flag so Julia is initialised at most once per process.
_JULIA_READY: bool = False

# Julia module reference, cached after first setup.
_JL: Any = None

# Julia code that defines the LogDensityProblems bridge struct.
# Evaluated once on first call to _setup_julia().
_JULIA_BRIDGE_CODE = """
struct PythonLogDensity
    py_fn::Any
    dim::Int
end

LogDensityProblems.logdensity(p::PythonLogDensity, x) =
    Float64(p.py_fn(x))

LogDensityProblems.dimension(p::PythonLogDensity) = p.dim

LogDensityProblems.capabilities(::Type{PythonLogDensity}) =
    LogDensityProblems.LogDensityOrder{0}()
"""


def _setup_julia():
    """Import Julia packages and define the log-density bridge (idempotent).

    Returns:
        juliacall ``Main`` module.

    Raises:
        ImportError: If ``juliacall`` is not installed.
        RuntimeError: If Julia or required packages cannot be loaded.
    """
    global _JULIA_READY, _JL

    if _JULIA_READY:
        return _JL

    try:
        from juliacall import Main as jl
    except ImportError as exc:
        raise ImportError(
            "juliacall is required for Pigeons.jl inference.\n"
            "Install it with:  pip install juliacall"
        ) from exc

    try:
        jl.seval("using Pigeons, LogDensityProblems")
    except Exception as exc:
        raise RuntimeError(
            "Failed to load Julia packages Pigeons and LogDensityProblems.\n"
            "Ensure Julia is installed and the packages are available.\n"
            "They are declared in juliapkg.json and should be auto-installed by "
            "juliapkg on first use.\n"
            f"Julia error: {exc}"
        ) from exc

    jl.seval(_JULIA_BRIDGE_CODE)

    _JL = jl
    _JULIA_READY = True
    return _JL


def run_pigeons_inference(
    model: ZINBLogDensity,
    n_rounds: int = 10,
    n_chains: int = 4,
    seed: int | None = 1,
) -> dict[str, Any]:
    """Run Pigeons.jl parallel tempering for the ZINB graphical model.

    Uses Pigeons.jl's parallel tempering (PT) algorithm to:

    * Draw posterior samples from the ZINB pseudo-likelihood graphical model.
    * Estimate the log marginal likelihood (log Z) via the stepping-stone
      estimator — which is a key advantage over Pyro-based MCMC/SVI.

    The total number of posterior samples returned is ``2 ** n_rounds``.

    Args:
        model: :class:`~pigeoneer.ZINBLogDensity` instance that encapsulates
            the count data and the ZINB pseudo-likelihood log density.
        n_rounds: Number of PT rounds.  Each round doubles the sample count.
            Total samples = ``2 ** n_rounds``.  Default: 10 (≈ 1 024 samples).
        n_chains: Number of parallel tempering chains.  More chains improve
            mixing but increase computational cost.  Default: 4.
        seed: Integer seed for reproducibility.  Pass ``None`` to use a
            random seed.  Default: 1.

    Returns:
        Dictionary with keys:

        ``'samples'`` : ``np.ndarray`` of shape ``(n_samples, dim)``
            Posterior samples in the unconstrained parameter space.

        ``'decoded_samples'`` : ``list[dict]``
            Each element is a dict with keys ``'A_tril'``, ``'mu'``,
            ``'phi'``, ``'pi_zero'``, ``'gamma_mu'``, ``'gamma_phi'``,
            ``'gamma_pi'``, ``'omega'``.

        ``'omega_samples'`` : ``np.ndarray`` of shape
            ``(n_samples, n_features, n_features)``
            Posterior samples of the interaction matrix Ω.

        ``'summary'`` : ``dict``
            Posterior summary statistics (mean, std, quantiles) for all
            parameters, including ``'omega'`` (mean and std matrices).

        ``'log_marginal_likelihood'`` : ``float``
            Log Z estimate from Pigeons.jl's stepping-stone estimator.
            Useful for Bayesian model comparison.

        ``'pt'`` : Pigeons.jl PT object
            The raw Pigeons.jl result for advanced diagnostics.

    Raises:
        ImportError: If ``juliacall`` is not installed.
        RuntimeError: If Julia or required packages cannot be loaded.

    Example::

        import numpy as np
        from pigeoneer import ZINBLogDensity, run_pigeons_inference

        rng = np.random.default_rng(0)
        X = rng.negative_binomial(5, 0.5, size=(100, 10)).astype(float)

        model = ZINBLogDensity(X, n_features=10)
        results = run_pigeons_inference(model, n_rounds=8, n_chains=4)

        omega_mean = results["summary"]["omega"]["mean"]
        log_z = results["log_marginal_likelihood"]
    """
    jl = _setup_julia()

    # Store the Python callable in Julia's namespace.
    jl.pigeoneer_log_density_fn = model.log_density
    jl.pigeoneer_dim = model.dim

    # Build the Julia target struct and run Pigeons.jl.
    # The seed argument is conditionally included so that passing seed=None
    # lets Pigeons.jl use its own default random seed.
    _common_args = (
        f"        target   = target,\n"
        f"        n_rounds = {n_rounds},\n"
        f"        n_chains = {n_chains},\n"
    )
    if seed is not None:
        _seed_arg = f"        seed     = {seed},\n"
    else:
        _seed_arg = ""
    script = (
        "let target = PythonLogDensity(pigeoneer_log_density_fn, pigeoneer_dim)\n"
        "    pigeons(\n"
        + _common_args
        + _seed_arg
        + "        record   = [Pigeons.record_samples(); Pigeons.record_default()...],\n"
        "    )\n"
        "end\n"
    )
    pt = jl.seval(script)

    # Extract posterior samples as a numpy matrix (n_samples × dim).
    sample_mat = np.array(jl.sample_array(pt), dtype=np.float64)

    # Decode each row into named parameters.
    decoded = [model.decode_params(sample_mat[i]) for i in range(len(sample_mat))]

    # Stack Ω matrices: (n_samples, n_features, n_features)
    omega_samples = np.stack([d["omega"] for d in decoded], axis=0)

    # Log marginal likelihood via stepping-stone estimator.
    log_z = float(jl.Pigeons.stepping_stone(pt))

    # Summary statistics.
    summary = _compute_summary(decoded, omega_samples)

    return {
        "samples": sample_mat,
        "decoded_samples": decoded,
        "omega_samples": omega_samples,
        "summary": summary,
        "log_marginal_likelihood": log_z,
        "pt": pt,
    }


def _compute_summary(
    decoded_samples: list[dict],
    omega_samples: np.ndarray,
) -> dict[str, Any]:
    """Compute posterior summary statistics.

    Args:
        decoded_samples: List of decoded parameter dicts from
            :meth:`~pigeoneer.ZINBLogDensity.decode_params`.
        omega_samples: Array of shape (n_samples, n_features, n_features).

    Returns:
        Dictionary mapping parameter names to summary dicts.
        Scalar parameters include ``'mean'``, ``'std'``, ``'median'``,
        ``'q05'``, ``'q95'``.  Vector/matrix parameters include
        ``'mean'`` and ``'std'``.
    """
    if not decoded_samples:
        return {}

    summary: dict[str, Any] = {}

    # Scalar parameters.
    for key in ("gamma_mu", "gamma_phi", "gamma_pi"):
        vals = np.array([d[key] for d in decoded_samples])
        summary[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "median": float(np.median(vals)),
            "q05": float(np.quantile(vals, 0.05)),
            "q95": float(np.quantile(vals, 0.95)),
        }

    # Vector parameters.
    for key in ("mu", "phi", "pi_zero", "A_tril"):
        vals = np.stack([d[key] for d in decoded_samples], axis=0)
        summary[key] = {
            "mean": vals.mean(axis=0),
            "std": vals.std(axis=0),
        }

    # Omega matrix.
    summary["omega"] = {
        "mean": omega_samples.mean(axis=0),
        "std": omega_samples.std(axis=0),
    }

    return summary
