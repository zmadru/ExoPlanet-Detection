"""Load B3 light-curve series exported as NumPy .npz files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_npz(path: str | Path) -> dict:
    """Load one ``.npz`` sample.

    Returns a dict with numpy arrays and string metadata:
    ``global_odd``, ``global_even``, ``local_odd``, ``local_even`` (1D float32),
    optional ``*_wl_a`` / ``*_wl_d`` if exported with wavelets.
    """
    with np.load(path, allow_pickle=False) as data:
        out = {key: data[key] for key in data.files}
    out["kepid"] = int(out["kepid"])
    out["kepler_name"] = str(out["kepler_name"])
    out["koi_class"] = str(out["koi_class"])
    return out


def load_index(data_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "index.csv")


def level0_sample_arrays(sample: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (global, local) with shape ``(2, L)`` for impar/par level 0."""
    global_series = np.stack([sample["global_odd"], sample["global_even"]], axis=0)
    local_series = np.stack([sample["local_odd"], sample["local_even"]], axis=0)
    return global_series, local_series
