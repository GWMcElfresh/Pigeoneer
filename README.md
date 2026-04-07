# Pigeoneer

A Python API for fitting **Zero-Inflated Negative Binomial (ZINB) graphical
models** of gene expression using
[Pigeons.jl](https://github.com/Julia-Tempering/Pigeons.jl)'s parallel
tempering algorithm.

Pigeoneer wraps the ZINB pseudo-likelihood graphical model from
[PreGraphModeling](https://github.com/GWMcElfresh/PreGraphModeling/tree/main/python)
and replaces the Pyro NUTS/SVI engine with Pigeons.jl, providing:

* **Marginal likelihood estimation** — the log normalising constant (log Z) is
  estimated for free by the stepping-stone estimator during parallel tempering,
  enabling principled Bayesian model comparison between different graph
  structures.
* **Better mixing** — replica exchange between chains at different temperatures
  escapes local modes that trap standard MCMC.
* **Reproducibility** — Pigeons.jl guarantees bit-identical results regardless
  of the number of threads or machines.

---

## Model

The ZINB pseudo-likelihood graphical model encodes conditional dependencies
between genes via a symmetric interaction matrix Ω.  For each gene *j*:

```
effect_j  = Σ_{k≠j} Ω_{jk} · X_{:,k}
μ_j^cond  = μ_j · exp(γ_μ · effect_j)
φ_j^cond  = φ_j · exp(γ_φ · effect_j)
π_j^cond  = sigmoid(logit(π_j) + γ_π · effect_j)

P(X_j | X_{-j})  =  ZINB(X_j | μ_j^cond, φ_j^cond, π_j^cond)
```

**Priors** (evaluated in unconstrained parameter space):

| Parameter | Prior | Unconstrained encoding |
|-----------|-------|------------------------|
| Ω off-diag (A_tril) | Normal(0, 0.1) | identity |
| μ | LogNormal(0, 1) | log_mu ~ Normal(0,1) |
| φ | LogNormal(0, 1) | log_phi ~ Normal(0,1) |
| π | Beta(1, 1) = Uniform(0,1) | logit_pi, with Jacobian |
| γ_μ | Normal(1, 0.5) | identity |
| γ_φ | Normal(0, 0.5) | identity |
| γ_π | Normal(0, 0.5) | identity |

---

## Installation

### Python dependencies

```bash
pip install pigeoneer
```

Or, from source:

```bash
git clone https://github.com/GWMcElfresh/Pigeoneer.git
cd Pigeoneer
pip install -e ".[dev]"
```

### Julia dependencies

Pigeoneer requires Julia ≥ 1.8 to be installed and available on `PATH`.
The required Julia packages (`Pigeons.jl`, `LogDensityProblems.jl`) are
declared in `juliapkg.json` and are automatically installed by
[juliapkg](https://github.com/JuliaPy/PythonCall.jl/tree/main/juliapkg)
on first use.

---

## Quick start

```python
import numpy as np
from pigeoneer import ZINBLogDensity, ZINBPriorConfig, run_pigeons_inference, load_count_matrix

# Load count matrix (cells × genes) from CSV / TSV / NPY / NPZ
X = load_count_matrix("counts.csv")

# (Optional) configure custom priors
prior = ZINBPriorConfig(
    a_tril_scale=0.5,     # wider prior on interaction strengths
    pi_alpha=2.0,         # Beta(2, 5) → favour low zero-inflation
    pi_beta=5.0,
)

# Define the ZINB graphical-model log density
model = ZINBLogDensity(X, n_features=X.shape[1], prior=prior)

# Run Pigeons.jl parallel tempering
# n_rounds=10  →  2^10 = 1024 posterior samples
results = run_pigeons_inference(model, n_rounds=10, n_chains=4, seed=1)

# Posterior mean of the interaction matrix Ω
omega_mean = results["summary"]["omega"]["mean"]

# Log marginal likelihood (for model comparison)
log_z = results["log_marginal_likelihood"]

# Per-gene posterior mean parameters
mu_mean    = results["summary"]["mu"]["mean"]
phi_mean   = results["summary"]["phi"]["mean"]
pi_mean    = results["summary"]["pi_zero"]["mean"]
```

---

## API reference

### `ZINBPriorConfig`

```python
ZINBPriorConfig(
    a_tril_scale=0.1,
    mu_log_mean=0.0, mu_log_scale=1.0,
    phi_log_mean=0.0, phi_log_scale=1.0,
    pi_alpha=1.0, pi_beta=1.0,
    gamma_mu_mean=1.0, gamma_mu_scale=0.5,
    gamma_phi_mean=0.0, gamma_phi_scale=0.5,
    gamma_pi_mean=0.0, gamma_pi_scale=0.5,
)
```

Dataclass holding prior hyperparameters for every model parameter.
Pass an instance to `ZINBLogDensity` to override the defaults.

| Field | Prior | Default |
|-------|-------|---------|
| `a_tril_scale` | A_tril ~ Normal(0, ·) | 0.1 |
| `mu_log_mean` / `mu_log_scale` | μ ~ LogNormal(·, ·) | 0, 1 |
| `phi_log_mean` / `phi_log_scale` | φ ~ LogNormal(·, ·) | 0, 1 |
| `pi_alpha` / `pi_beta` | π ~ Beta(·, ·) | 1, 1 (= Uniform) |
| `gamma_mu_mean` / `gamma_mu_scale` | γ_μ ~ Normal(·, ·) | 1, 0.5 |
| `gamma_phi_mean` / `gamma_phi_scale` | γ_φ ~ Normal(·, ·) | 0, 0.5 |
| `gamma_pi_mean` / `gamma_pi_scale` | γ_π ~ Normal(·, ·) | 0, 0.5 |

### `ZINBLogDensity`

```python
ZINBLogDensity(X, n_features, prior=None)
```

Encapsulates the data and the ZINB pseudo-likelihood log density for use
with Pigeons.jl.

| Method | Description |
|--------|-------------|
| `log_density(x)` | Unnormalised log posterior (prior + pseudo-LL). Called by Pigeons.jl. |
| `log_prior(x)` | Log prior in unconstrained space. |
| `pseudo_log_likelihood(x)` | ZINB pseudo-log-likelihood. |
| `initial_params()` | Prior-mean initial parameter vector. |
| `decode_params(x)` | Decode unconstrained vector → named-parameter dict. |

### `run_pigeons_inference`

```python
run_pigeons_inference(model, n_rounds=10, n_chains=4, seed=1)
```

Runs Pigeons.jl parallel tempering.

**Returns** a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `'samples'` | `(N, dim) ndarray` | Posterior samples in unconstrained space |
| `'decoded_samples'` | `list[dict]` | Decoded parameter dicts |
| `'omega_samples'` | `(N, p, p) ndarray` | Posterior Ω samples |
| `'summary'` | `dict` | Mean / std / quantile summary |
| `'log_marginal_likelihood'` | `float` | Stepping-stone log Z estimate |
| `'pt'` | Julia object | Raw Pigeons.jl result for diagnostics |

### `load_count_matrix`

```python
load_count_matrix(filepath)
```

Load a count matrix from `.csv`, `.tsv`, `.npy`, or `.npz` and return a
`float64` numpy array of shape `(n_samples, n_features)`.

---

## Docker

The repository ships a multi-stage `Dockerfile` with `deps` and `runtime`
targets.

```bash
# Build the deps image (dependencies only — cacheable)
docker build --target deps -t pigeoneer:deps .

# Build the full runtime image
docker build -t pigeoneer:latest .

# Run tests inside the container
docker run --rm pigeoneer:latest
```

---

## Testing

```bash
# Fast tests (no Julia required)
pytest tests/ -v -m "not slow"

# All tests including Pigeons.jl integration (requires Julia)
pytest tests/ -v
```

---

## Relationship to PreGraphModeling

This package takes the ZINB graphical model defined in
[PreGraphModeling/python](https://github.com/GWMcElfresh/PreGraphModeling/tree/main/python)
and replaces the Pyro NUTS/SVI inference engine with Pigeons.jl.  The key
difference is the availability of the **marginal likelihood** estimate, which
Pyro's NUTS/SVI does not provide directly.

| Feature | PreGraphModeling (Pyro) | Pigeoneer (Pigeons.jl) |
|---------|------------------------|------------------------|
| Sampler | NUTS / SVI | Parallel tempering |
| Marginal likelihood | ❌ | ✅ stepping-stone |
| GPU support | ✅ PyTorch | — |
| Python-only | ✅ | Requires Julia |