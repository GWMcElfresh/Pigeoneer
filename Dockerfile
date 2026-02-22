# syntax=docker/dockerfile:1
# Multi-stage Dockerfile for Pigeoneer.
#
# Stages:
#   deps    — installs all Python and Julia dependencies (cacheable layer)
#   runtime — copies the application code on top of the deps layer
#
# Build arguments:
#   DEPS_IMAGE   — (runtime stage) pre-built deps image to use as base.
#                  Set automatically by the docker-cache CI workflow.
#   PYTHON_VERSION — Python version to install (default: 3.11).
#   JULIA_VERSION  — Julia version to install (default: 1.10.0).
#   BASE_IMAGE     — (optional) monthly base image for layer reuse.
#   SKIP_BASE_DEPS — (optional) skip re-installing base OS deps when using
#                    a pre-built base image.

ARG PYTHON_VERSION=3.11
ARG JULIA_VERSION=1.10.0
ARG BASE_IMAGE=ubuntu:22.04

# ── deps ──────────────────────────────────────────────────────────────────────
FROM ${BASE_IMAGE} AS deps

ARG PYTHON_VERSION
ARG JULIA_VERSION
ARG SKIP_BASE_DEPS=false

ENV DEBIAN_FRONTEND=noninteractive \
    JULIA_DEPOT_PATH=/opt/julia_depot \
    PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring

# --- OS packages ---
RUN if [ "$SKIP_BASE_DEPS" != "true" ]; then \
      apt-get update && apt-get install -y --no-install-recommends \
        curl wget ca-certificates git \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip \
        build-essential && \
      apt-get clean && rm -rf /var/lib/apt/lists/*; \
    fi

# Make python3 / pip3 point to the requested version.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 10 2>/dev/null || true && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python${PYTHON_VERSION} 10 2>/dev/null || true

# --- Julia ---
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then JULIA_ARCH="x86_64"; \
    elif [ "$ARCH" = "aarch64" ]; then JULIA_ARCH="aarch64"; \
    else echo "Unsupported arch: $ARCH" && exit 1; fi && \
    curl -fsSL "https://julialang-s3.julialang.org/bin/linux/${JULIA_ARCH}/$(echo ${JULIA_VERSION} | cut -d. -f1-2)/julia-${JULIA_VERSION}-linux-${JULIA_ARCH}.tar.gz" \
        -o /tmp/julia.tar.gz && \
    tar -xzf /tmp/julia.tar.gz -C /opt && \
    ln -sf /opt/julia-${JULIA_VERSION}/bin/julia /usr/local/bin/julia && \
    rm /tmp/julia.tar.gz

# --- Python dependencies ---
WORKDIR /app
COPY pyproject.toml ./
# Install Python deps without the package itself so this layer is cached.
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir "numpy>=1.20.0" "scipy>=1.7.0" "juliacall>=0.9.0" pytest pytest-cov

# --- Julia dependencies (pre-warm the Julia package depot) ---
COPY juliapkg.json ./
# juliapkg reads juliapkg.json and resolves/downloads the Julia packages.
RUN python3 -c "import juliapkg; juliapkg.resolve()" || true

# ── runtime ───────────────────────────────────────────────────────────────────
ARG DEPS_IMAGE
FROM ${DEPS_IMAGE:-deps} AS runtime

WORKDIR /app
# Copy the package source on top of the pre-built deps layer.
COPY . .
RUN pip3 install --no-cache-dir -e ".[dev]"

CMD ["pytest", "tests/", "-v", "-m", "not slow"]
