"""
Data loading utilities for count matrices.

Supports CSV, TSV, NPY, and NPZ file formats.
Returns numpy arrays (float64) suitable for use with
:class:`~pigeoneer.ZINBLogDensity`.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np


def _load_delimited_with_row_names(filepath: Path, delimiter: str) -> np.ndarray:
    """Load a delimited file, skipping a header row and optional row-name column.

    Args:
        filepath: Path to the file.
        delimiter: Field delimiter character.

    Returns:
        2-D float64 array of shape (n_rows, n_cols).
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        first_line = fh.readline().strip()

    n_cols = len(first_line.split(delimiter))

    # Try to load all columns except the first (assumed row-name column).
    try:
        data = np.loadtxt(
            filepath,
            delimiter=delimiter,
            skiprows=1,
            usecols=range(1, n_cols),
        )
        return data
    except (ValueError, IndexError):
        # Fall back: load all columns (no row names).
        return np.loadtxt(filepath, delimiter=delimiter, skiprows=1)


def load_count_matrix(filepath: str) -> np.ndarray:
    """Load a count matrix from disk.

    Supported formats: ``.npy``, ``.npz``, ``.csv``, ``.tsv``, ``.txt``.

    CSV/TSV files are expected to have a header row with feature names and
    optionally a first column of row names (as produced by R's
    ``write.csv`` or pandas ``to_csv``).

    Args:
        filepath: Path to the count matrix file.

    Returns:
        Float64 numpy array of shape ``(n_samples, n_features)``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    fpath = Path(filepath)

    if not fpath.exists():
        raise FileNotFoundError(f"Count matrix file not found: {fpath}")

    suffix = fpath.suffix.lower()

    if suffix == ".npy":
        data = np.load(fpath)

    elif suffix == ".npz":
        npz = np.load(fpath)
        keys = list(npz.keys())
        if len(keys) == 1:
            data = npz[keys[0]]
        elif "counts" in keys:
            data = npz["counts"]
        elif "data" in keys:
            data = npz["data"]
        else:
            data = npz[keys[0]]

    elif suffix == ".csv":
        data = _load_delimited_with_row_names(fpath, delimiter=",")

    elif suffix in (".tsv", ".txt"):
        data = _load_delimited_with_row_names(fpath, delimiter="\t")

    else:
        raise ValueError(
            f"Unsupported file format: '{suffix}'. "
            "Supported formats: .npy, .npz, .csv, .tsv, .txt"
        )

    data = np.asarray(data, dtype=np.float64)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    return data
