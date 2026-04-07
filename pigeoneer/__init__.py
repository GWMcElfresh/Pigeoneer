"""
Pigeoneer: Python API for Pigeons.jl-based ZINB Graphical Model Inference.

This package provides a Python interface for fitting Zero-Inflated Negative
Binomial (ZINB) graphical models of gene expression using Pigeons.jl's
parallel tempering (PT) algorithm.

Key advantages over Pyro-based MCMC/SVI
-----------------------------------------
* **Marginal likelihood**: PT naturally estimates the log normalising constant
  (log Z) via the stepping-stone estimator, enabling Bayesian model comparison
  without additional computation.
* **Multimodal posteriors**: Replica exchange between chains at different
  temperatures allows the sampler to escape local modes.
* **Reproducibility**: Pigeons.jl guarantees bit-identical results regardless
  of the number of threads or machines used.

Quick start
-----------
::

    import numpy as np
    from pigeoneer import ZINBLogDensity, run_pigeons_inference, load_count_matrix

    # Load count matrix (genes × cells)
    X = load_count_matrix("counts.csv")

    # Define ZINB graphical model log density
    model = ZINBLogDensity(X, n_features=X.shape[1])

    # Run parallel tempering via Pigeons.jl
    results = run_pigeons_inference(model, n_rounds=10, n_chains=4)

    # Posterior mean of the interaction matrix Ω
    omega_mean = results["summary"]["omega"]["mean"]

    # Log marginal likelihood estimate
    log_z = results["log_marginal_likelihood"]

Requirements
------------
* ``numpy``, ``scipy``
* ``juliacall`` (``pip install juliacall``)
* Julia ≥ 1.8 with Pigeons.jl and LogDensityProblems.jl
  (auto-installed from ``juliapkg.json`` on first use)
"""

from .model import ZINBLogDensity, _build_omega, _nb_log_prob, _zinb_log_prob
from .inference import run_pigeons_inference, _compute_summary
from .data import load_count_matrix
from .priors import ZINBPriorConfig

__all__ = [
    "ZINBLogDensity",
    "ZINBPriorConfig",
    "run_pigeons_inference",
    "load_count_matrix",
    "_build_omega",
    "_nb_log_prob",
    "_zinb_log_prob",
    "_compute_summary",
]

__version__ = "0.1.0"
