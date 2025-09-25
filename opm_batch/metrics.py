from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, Dict
from .config import (
    SELECT_METRICS, METRIC_COLNAMES, NOISE_MODE, NOISE_BAND, NOISE_WINDOW,
    NOISE_AVERAGE, NOISE_LOCK_DF_HZ, NOISE_DETREND, NOISE_ENABLE_PREPROCESS,
    EXPERIMENT_MODE, CURRENT_OFFSET_MA, POWER_SCALE, POWER_OFFSET
)
from opmtool import noise_psd, noise_psd_lowband

def force_include_r2_columns(selected: set[str]) -> set[str]:
    sel = set(s.lower() for s in selected)
    if "slope" in sel: sel.add("sloper2")
    if {"voigtfwhm","voigtgamma","voigtsigma"} & sel: sel.add("voigtr2")
    if "lorentzfwhm" in sel: sel.add("lorentzr2")
    if "gaussianfwhm" in sel: sel.add("gaussianr2")
    return sel

def first_scalar(x: Any, default: float = np.nan) -> float:
    try:
        if x is None: return default
        if hasattr(x, "values"):
            arr = np.asarray(x.values, dtype=float)
            return float(arr.ravel()[0])
        if isinstance(x, (list, tuple)):
            return float(np.asarray(x, dtype=float).ravel()[0])
        arr = np.asarray(x)
        if arr.ndim >= 1:
            return float(arr.ravel()[0])
        return float(arr)
    except Exception:
        return default

def compute_noise(noise_df_std: pd.DataFrame) -> float:
    kwargs = dict(
        band=NOISE_BAND, window=NOISE_WINDOW, average=NOISE_AVERAGE,
        min_freq_resolution=NOISE_LOCK_DF_HZ, force_long_segments=False,
        detrend=NOISE_DETREND, snap_to_power_of_2=False
    )
    if NOISE_MODE == "lowband":
        _, mean_rms, _, _ = noise_psd_lowband(
            noise_df_std, enable_auto_preprocess=NOISE_ENABLE_PREPROCESS, **kwargs
        )
    else:
        _, mean_rms, _, _ = noise_psd(noise_df_std, **kwargs)
    return float(mean_rms)  # V/√Hz

def extract_sweep_value(label: str) -> float:
    import re
    m = re.search(r"([-+]?\d+(?:\.\d+)?)", label)
    val = float(m.group(1)) if m else np.nan
    mode = EXPERIMENT_MODE.lower()
    if mode == "current":
        return val + float(CURRENT_OFFSET_MA)
    elif mode == "power":
        return val * float(POWER_SCALE) + float(POWER_OFFSET)
    return np.nan
