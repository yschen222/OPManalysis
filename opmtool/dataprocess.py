# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 17:38:40 2025

@author: YiHsuanChen
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, Tuple, Union
from scipy.signal import welch, find_peaks

"""
Initial data processing
"""
def one_cycle_cut(
    data,
    B_range,
    cycles=1,
    segment='half',                # 'half' (rising-only / falling-only) or 'full' (rise+fall)
    direction='auto',              # 'auto' | 'rising' | 'falling' (used for 'half')
    time_col='time',
    ab_col='Ab',
    demod_col='demod',
    tri_col='tri',                 # required triangle/sawtooth column
    smooth_win=5,                  # smoothing window (samples) for the triangle
    anchor='abspeak',              # 'abspeak' | 'center' | 'first' | 'last'
    edge_clip_frac=0.02,           # clip 2% from both ends of the chosen segment(s)
    map_mode='global',             # 'global' (default) | 'local'
    require_tri=True,
    return_debug=False
):
    """
    Cut **exactly one sweep segment** from a measured triangle/sawtooth waveform
    and convert it into a magnetic-field axis according to the given (B_min, B_max).

    This version uses *robust peak/trough detection* to find the most reliable
    half- or full-cycle. It prevents duplicated or stitched segments and ensures
    only one continuous sweep is extracted.

    Parameters
    ----------
    data : pandas.DataFrame or ndarray
        - DataFrame must include columns: [time_col, ab_col, demod_col, tri_col].
        - ndarray must be shape (N, 4): [t, Ab, demod, tri].
    B_range : (float, float)
        (B_min, B_max) corresponding to the triangle extrema for *one sweep*.
        Units are arbitrary but should match your intended nT scale.
    cycles : int, optional
        (Reserved for backward compatibility; ignored in this version.)
        Always returns only one sweep segment.
    segment : {'half', 'full'}, optional
        - 'half': cut one strictly monotonic run (rising or falling).
        - 'full': cut one complete peak-to-peak window (rise + fall).
    direction : {'auto', 'rising', 'falling'}, optional
        Which direction to select when `segment='half'`.
        - 'auto' infers the dominant direction from the derivative sign of the triangle.
    time_col, ab_col, demod_col, tri_col : str
        Column names (used only when `data` is a DataFrame).
    smooth_win : int, optional
        Moving-average window (in samples) applied to the triangle before
        segmentation. Set ≤1 to disable smoothing.
    anchor : {'abspeak', 'center', 'first', 'last'}, optional
        Rule for choosing which detected segment to keep:
        - 'abspeak': segment containing the largest |Ab - baseline|.
        - 'center' : segment closest to the middle of the record.
        - 'first'  : earliest valid segment.
        - 'last'   : latest valid segment.
    edge_clip_frac : float, optional
        Fraction of samples to remove from both ends of the chosen segment
        to suppress turn-around artifacts and coil transients
        (e.g. 0.02 = trim 2% at each end).
    map_mode : {'global', 'local'}, optional
        Defines how the triangle waveform is mapped to (B_min, B_max):
        - 'global' (default): use the **pre-clipped** min/max of the chosen segment.
          The resulting field naturally stays within your commanded ±range.
        - 'local': rescale the **post-clipped** segment back to (B_min, B_max),
          effectively stretching it to full range.
    require_tri : bool, optional
        If True, raises an error if the triangle column is missing.
    return_debug : bool, optional
        If True, also returns a dictionary with segmentation diagnostics.

    Returns
    -------
    BField : ndarray
        Magnetic-field axis (in the same length as the cut signals),
        mapped according to `map_mode`.
    AbCut : ndarray
        Absorption (Channel A) signal corresponding to the selected segment.
    DemodCut : ndarray
        Demodulated (lock-in) signal corresponding to the selected segment.
    debug : dict, optional
        Returned only if `return_debug=True`. Includes:
        - 'segment'         : 'half' or 'full'
        - 'direction'       : 'rising' or 'falling'
        - 'spans'           : list of all candidate (start, end) segments
        - 'chosen'          : (start, end) indices of selected full segment
        - 'clipped'         : (start, end) indices after edge trimming
        - 'tri_span_full'   : (min, max) triangle values before clipping
        - 'tri_span_local'  : (min, max) triangle values after clipping
        - 'map_mode'        : 'global' or 'local'

    Notes
    -----
    * Segmentation is based entirely on the **triangle/sawtooth waveform**:
      - Peaks and troughs are detected with prominence filtering to reject noise.
      - Only one candidate segment is finally chosen according to `anchor`.
    * The absorption signal (`Ab`) is used **only** for segment selection when
      `anchor='abspeak'`, never for defining boundaries.
    * A final monotonicity check removes small nonmonotonic fragments
      (to prevent duplicated ramps in half-cycles).
    * Recommended defaults for typical OPM scans:
        `segment='half'`, `direction='auto'`, `map_mode='global'`,
        `edge_clip_frac=0.02`.
    """

    # ---------- extract columns ----------
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        must = [time_col, ab_col, demod_col]
        for c in must:
            if c not in df.columns:
                raise ValueError(f"DataFrame needs '{time_col}', '{ab_col}', '{demod_col}'.")
        if tri_col not in df.columns:
            if require_tri:
                raise ValueError(f"Missing triangle column '{tri_col}'.")
            raise ValueError("Triangle waveform required for reliable cutting.")
        t     = df[time_col].to_numpy(float)
        Ab    = df[ab_col].to_numpy(float)
        demod = df[demod_col].to_numpy(float)
        tri   = df[tri_col].to_numpy(float)
    else:
        arr = np.asarray(data, float)
        if arr.ndim < 2 or arr.shape[1] < 4:
            raise ValueError("ndarray must be (N,4): [t, Ab, demod, tri].")
        t, Ab, demod, tri = arr[:,0], arr[:,1], arr[:,2], arr[:,3]

    B_min, B_max = B_range
    N = len(Ab)
    if N < 16:
        raise ValueError("Not enough samples.")

    # ---------- helpers ----------
    def _smooth(x, w):
        if w is None or w <= 1: return x
        k = np.ones(int(w))/int(w)
        return np.convolve(x, k, mode='same')

    def _edge_baseline(y, frac=0.1):
        n = max(1, int(len(y)*frac))
        return float(np.median(np.r_[y[:n], y[-n:]]))

    def _pick_anchor_span(spans, mode='abspeak'):
        if not spans:
            raise ValueError("No candidate span found.")
        if mode == 'first':  return 0
        if mode == 'last':   return len(spans)-1
        if mode == 'center':
            mid = N//2
            return int(np.argmin([abs((s+e)//2 - mid) for s, e in spans]))
        # 'abspeak'
        c0 = _edge_baseline(Ab)
        k  = int(np.argmax(np.abs(Ab - c0)))
        for i, (s,e) in enumerate(spans):
            if s <= k < e:
                return i
        mid = N//2
        return int(np.argmin([abs((s+e)//2 - mid) for s, e in spans]))

    # ---------- triangle & robust extrema ----------
    tri_s = _smooth(tri, smooth_win)
    tri_lo, tri_hi = float(np.nanmin(tri_s)), float(np.nanmax(tri_s))
    tri_rng = max(1e-12, tri_hi - tri_lo)

    # robust peaks/troughs: require prominence and distance
    prom = 0.05 * tri_rng          # 5% of range
    min_len = max(16, N // 200)    # >=16 samples, or 0.5% of record
    pk, _ = find_peaks(tri_s, prominence=prom, distance=min_len)
    tr, _ = find_peaks(-tri_s, prominence=prom, distance=min_len)
    pk = np.asarray(pk, int)
    tr = np.asarray(tr, int)

    # ---------- build candidates ----------
    spans = []  # list[(s,e)] ; single span will be chosen later

    if segment == 'half':
        # determine direction if auto
        dtri = np.diff(tri_s)
        if direction == 'auto':
            dir_mode = 'rising' if np.sum(dtri > 0) >= np.sum(dtri < 0) else 'falling'
        else:
            dir_mode = 'rising' if direction == 'rising' else 'falling'

        # pair trough->peak (rising) or peak->trough (falling)
        ord_tp = np.sort(np.r_[pk, tr])
        for a, b in zip(ord_tp[:-1], ord_tp[1:]):
            if dir_mode == 'rising' and (a in tr) and (b in pk) and (b - a >= min_len):
                spans.append((int(a), int(b)))
            if dir_mode == 'falling' and (a in pk) and (b in tr) and (b - a >= min_len):
                spans.append((int(a), int(b)))

        # fallback if no robust pairs: longest strictly monotone run with span filter
        if not spans:
            good = (dtri > 0) if dir_mode == 'rising' else (dtri < 0)
            edges = np.diff(np.r_[False, good, False])
            st = np.where(edges ==  1)[0]
            en = np.where(edges == -1)[0]
            for s, e in zip(st, en):
                if (e - s) >= min_len and (np.ptp(tri_s[s:e]) >= 0.5 * tri_rng):
                    spans.append((int(s), int(e)))
            if not spans:
                spans = [(0, N-1)]  # absolute last resort

    else:  # 'full': peak->peak or trough->trough
        for arr in (np.sort(pk), np.sort(tr)):
            if len(arr) >= 2:
                for a, b in zip(arr[:-1], arr[1:]):
                    if (b - a) >= min_len and (np.ptp(tri_s[a:b]) >= 0.5 * tri_rng):
                        spans.append((int(a), int(b)))
        # last resort: adjacent extrema of any type
        if not spans:
            ord_tp = np.sort(np.r_[pk, tr])
            for a, b in zip(ord_tp[:-1], ord_tp[1:]):
                if (b - a) >= min_len and (np.ptp(tri_s[a:b]) >= 0.5 * tri_rng):
                    spans.append((int(a), int(b)))
        if not spans:
            spans = [(0, N-1)]

    # ---------- choose ONE span ----------
    i0 = _pick_anchor_span(spans, anchor)
    s_full, e_full = spans[i0]
    e_full = max(e_full, s_full + 1)

    # ---------- edge clipping ----------
    clip = int(max(0.0, float(edge_clip_frac)) * (e_full - s_full))
    s0 = min(max(0, s_full + clip), e_full - 1)
    e0 = max(min(N, e_full - clip), s0 + 1)

    tri_cut   = tri[s0:e0]
    AbCut     = Ab[s0:e0]
    DemodCut  = demod[s0:e0]

    # ---------- final monotonicity guard for HALF ----------
    if segment == 'half' and (e0 - s0) > 2:
        dseg = np.diff(tri_cut)
        want_up = (direction == 'rising') or (direction == 'auto' and tri_s[e_full] > tri_s[s_full])
        eps = 1e-12
        step_ok = (dseg > eps) if want_up else (dseg < -eps)
        keep = np.r_[True, step_ok]

        edges2 = np.diff(np.r_[False, keep, False])
        st2 = np.flatnonzero(edges2 == 1)
        en2 = np.flatnonzero(edges2 == -1)
        if len(st2) and len(en2):
            i_long = int(np.argmax(en2 - st2))
            sl = slice(st2[i_long], en2[i_long])
            tri_cut  = tri_cut[sl]; AbCut = AbCut[sl]; DemodCut = DemodCut[sl]
        elif keep.any():
            idx = np.flatnonzero(keep)
            tri_cut  = tri_cut[idx]; AbCut = AbCut[idx]; DemodCut = DemodCut[idx]
        # else: keep as-is

    # ---------- map triangle -> B ----------
    if map_mode == 'global':
        tr_lo = float(np.min(tri[s_full:e_full]))
        tr_hi = float(np.max(tri[s_full:e_full]))
    else:
        tr_lo = float(np.min(tri_cut)) if len(tri_cut) else np.nan
        tr_hi = float(np.max(tri_cut)) if len(tri_cut) else np.nan

    if not np.isfinite(tr_lo) or not np.isfinite(tr_hi) or tr_hi <= tr_lo:
        BField = np.full_like(tri_cut, (B_min + B_max)/2.0, dtype=float)
    else:
        scale  = (B_max - B_min) / (tr_hi - tr_lo + 1e-30)
        BField = B_min + (tri_cut - tr_lo) * scale
        lo, hi = (B_min, B_max) if B_min <= B_max else (B_max, B_min)
        BField = np.clip(BField, lo, hi)

    if return_debug:
        dbg = dict(
            spans=spans,
            chosen=(s_full, e_full),
            clipped=(s0, e0),
            tri_span_full=(float(np.min(tri[s_full:e_full])), float(np.max(tri[s_full:e_full]))),
            tri_span_local=(float(np.min(tri_cut)) if len(tri_cut) else np.nan,
                            float(np.max(tri_cut)) if len(tri_cut) else np.nan),
            segment=segment,
            direction=(direction if segment=='half' else 'full'),
        )
        return BField, AbCut, DemodCut, dbg
    return BField, AbCut, DemodCut



def noise_psd(
    data,
    band=None,                    # (f1, f2) in Hz, optional
    targetf=None, half_bw=None,   # center +/- half_bw, optional
    window='hann',
    average='median',
    min_freq_resolution=None,     # desired df (Hz); if None, auto from band
    force_long_segments=False,    # prioritize resolution over #segments
    detrend='linear',             # 'linear' for low-freq robustness
    snap_to_power_of_2=False      # optional FFT optimization
):
    """
    Welch PSD around a *user-chosen band*, with two selection modes:
      (1) Exact band edges: band=(f1, f2)
      (2) Center +/- half bandwidth: targetf, half_bw

    Parameters
    ----------
    data : DataFrame or ndarray
        Columns = [time, signal1, signal2]. Noise is taken from column index 2.
    band : tuple(float, float), optional
        Analysis band edges (f1, f2) in Hz. Use this OR (targetf, half_bw).
    targetf : float, optional
        Center frequency in Hz (used when `band` is not given).
    half_bw : float, optional
        Half-bandwidth in Hz (used with `targetf`).
    window : str | tuple | array_like, optional
        Welch window. Default 'hann' (good noise-floor tradeoff).
        Use 'flattop' for amplitude accuracy of single tones,
        'blackmanharris' for high dynamic range.
    average : {'mean','median'}, optional
        Welch average. Default 'median' for robustness to spurs.
    min_freq_resolution : float, optional
        Target frequency resolution (df) in Hz. If None, it is chosen from the
        requested analysis band (aiming ~8–16 bins across the band).
    force_long_segments : bool, optional
        If True, prioritize fine df (long nperseg) even if segments become few.
    detrend : {'linear','constant',None}, optional
        Detrend option for Welch. 'linear' recommended for low-frequency drift.
    snap_to_power_of_2 : bool, optional
        If True, adjust nperseg to nearest power of 2 for FFT efficiency.
        May slightly degrade frequency resolution.

    Returns
    -------
    pts : ndarray, shape (M, 2)
        [f, PSD] with PSD in V^2/Hz.
    mean_rms : float
        Mean ASD over the selected band: sqrt(mean(PSD_band)) in V/√Hz.
    fs : float
        Sampling frequency inferred from time vector (Hz).
    info : dict
        Analysis parameters and diagnostics

    Notes
    -----
    * If both `band` and `(targetf, half_bw)` are provided, `band` takes priority.
    * If neither is provided, a ValueError is raised.
    * The function does not resample. If your fs is very high with short records (3–8 s),
      consider downsampling to 1–2 kHz before calling for better segment counts.
    """
    # ---------- Load & validate ----------
    if isinstance(data, pd.DataFrame):
        arr = data.values
    else:
        arr = np.asarray(data)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("Data must have at least 3 columns: [time, signal1, signal2].")

    time  = np.asarray(arr[:, 0], dtype=float)
    noise = np.asarray(arr[:, 2], dtype=float)

    msk = np.isfinite(time) & np.isfinite(noise)
    time, noise = time[msk], noise[msk]
    if time.size < 100:
        raise ValueError("Insufficient valid data points (need at least 100).")

    # ---------- Band selection mode ----------
    if band is not None:
        # Validate band parameter
        if not isinstance(band, (tuple, list)) or len(band) != 2:
            raise ValueError("`band` must be a 2-element tuple/list: (f1, f2)")
        
        f1, f2 = float(band[0]), float(band[1])
        if not np.isfinite(f1) or not np.isfinite(f2):
            raise ValueError("Band frequencies must be finite numbers.")
        if f1 < 0 or f2 < 0:
            raise ValueError("Band frequencies must be non-negative.")
        if f2 <= f1:
            raise ValueError("Invalid `band`: must be (f1, f2) with f2 > f1.")
            
        band_lo_req, band_hi_req = f1, f2
        band_width = f2 - f1
        band_center = 0.5*(f1 + f2)
    else:
        if targetf is None or half_bw is None:
            raise ValueError("Provide either `band=(f1, f2)` or `targetf` with `half_bw`.")
        
        targetf = float(targetf)
        half_bw = float(half_bw)
        if not np.isfinite(targetf) or not np.isfinite(half_bw):
            raise ValueError("targetf and half_bw must be finite numbers.")
        if targetf < 0 or half_bw <= 0:
            raise ValueError("targetf must be non-negative and half_bw must be positive.")
            
        band_center = targetf
        band_width  = 2.0 * half_bw
        band_lo_req = band_center - half_bw
        band_hi_req = band_center + half_bw

    # ---------- Sampling rate ----------
    dt = np.median(np.diff(time))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid time axis: cannot determine sampling interval.")
    fs = 1.0 / dt
    N  = noise.size
    nyq = fs / 2.0

    # Validate that requested band makes sense for this sampling rate
    if band_hi_req >= nyq:
        warnings.warn(f"Requested upper frequency {band_hi_req:.1f} Hz >= Nyquist frequency "
                     f"{nyq:.1f} Hz. Will be capped to Nyquist.", RuntimeWarning)

    # ---------- Choose df and nperseg ----------
    if min_freq_resolution is None:
        target_bins = 12.0
        min_freq_resolution = max(band_width / target_bins, 1e-9)

    if not np.isfinite(min_freq_resolution) or min_freq_resolution <= 0:
        raise ValueError("`min_freq_resolution` must be positive and finite.")

    nperseg_res = int(fs / min_freq_resolution)
    nperseg_res = max(256, nperseg_res)
    nperseg_res = min(nperseg_res, N)    

    def seg_count(nps):
        if nps <= 0 or nps > N:
            return 0
        step = max(1, nps - nps//2)  # 50% overlap
        return int(np.floor((N - nps) / step)) + 1

    if force_long_segments:
        nperseg = nperseg_res
        min_segments = 2
    else:
        nperseg = nperseg_res
        min_segments = 4
        while nperseg >= 256 and seg_count(nperseg) < min_segments:
            nperseg = int(max(256, nperseg * 0.8))

    # Optional: snap to power of 2 for FFT efficiency
    if snap_to_power_of_2:
        original_nperseg = nperseg
        nperseg = int(2 ** np.floor(np.log2(max(256, nperseg))))
        if abs(nperseg - original_nperseg) / original_nperseg > 0.2:  # >20% change
            warnings.warn(
                f"Power-of-2 adjustment changed nperseg from {original_nperseg} to {nperseg}, "
                "affecting frequency resolution.", RuntimeWarning
            )

    nperseg = min(nperseg, N)  
    noverlap = nperseg // 2
    K = seg_count(nperseg)

    # start from resolution-driven nperseg, try to keep reasonable segments
    if force_long_segments:
        nperseg = nperseg_res
        min_segments = 2
    else:
        nperseg = nperseg_res
        min_segments = 4
        # reduce nperseg until we get enough segments, but not below 256
        while nperseg >= 256 and seg_count(nperseg) < min_segments:
            nperseg = int(max(256, nperseg * 0.8))

    # Optional: snap to power of two for FFT efficiency
    if snap_to_power_of_2:
        original_nperseg = nperseg
        nperseg = int(2 ** np.floor(np.log2(max(256, nperseg))))
        if abs(nperseg - original_nperseg) / original_nperseg > 0.2:  # >20% change
            warnings.warn(f"Power-of-2 adjustment changed nperseg from {original_nperseg} "
                         f"to {nperseg}, affecting frequency resolution.", RuntimeWarning)
    
    nperseg = min(nperseg, N)  # cannot exceed length
    noverlap = nperseg // 2
    K = seg_count(nperseg)
    
    if K < 2:
        warnings.warn(f"Only {K} Welch segments available; PSD may be very noisy.", RuntimeWarning)
    elif K < 4:
        warnings.warn(f"Only {K} Welch segments available; PSD may be noisy.", RuntimeWarning)

    # ---------- Welch PSD ----------
    try:
        f, Pxx = welch(
            noise, fs=fs, window=window,
            nperseg=nperseg, noverlap=noverlap,
            detrend=detrend, scaling='density',
            average=average
        )
    except Exception as e:
        raise RuntimeError(f"Welch PSD estimation failed: {e}")
        
    pts = np.column_stack((f, Pxx))

    # ---------- Effective band (cap to FFT grid and Nyquist) ----------
    # Skip DC bin when selecting band
    f_min_bin = f[1] if f.size > 1 else 0.0
    band_lo = max(band_lo_req, f_min_bin)
    band_hi = min(band_hi_req, nyq)
    
    # Select frequency mask
    mask = (f >= band_lo) & (f <= band_hi)
    
    if not np.any(mask):
        # fallback: nearest bin to requested center
        idx = int(np.argmin(np.abs(f - band_center)))
        mask = np.zeros_like(f, dtype=bool)
        mask[idx] = True
        warnings.warn(
            f"No frequency bins inside requested band [{band_lo_req:.3f}, {band_hi_req:.3f}] Hz; "
            f"using nearest bin at {f[idx]:.3f} Hz.",
            RuntimeWarning
        )

    # Extract band PSD values
    f_band = f[mask]
    P_band = Pxx[mask]
    good = np.isfinite(P_band)
    P_band = P_band[good]
    f_band = f_band[good]

    mean_rms = float(np.sqrt(np.mean(P_band))) if P_band.size else np.nan
    if not np.isfinite(mean_rms):
        warnings.warn("No valid PSD values in the selected band.", RuntimeWarning)

    # ---------- Information dictionary ----------
    df_bin = f[1] - f[0] if f.size > 1 else np.nan
    actual_resolution = float(df_bin) if np.isfinite(df_bin) else np.nan
    
    info = {
        'nperseg': int(nperseg),
        'noverlap': int(noverlap),
        'n_segments': int(K),
        'df': actual_resolution,
        'requested_df': float(min_freq_resolution),
        'resolution_achieved': actual_resolution <= min_freq_resolution * 1.1,  # within 10%
        'window': window,
        'average': average,
        'band_bins': int(np.sum(mask)),
        'band_range': [float(band_lo), float(band_hi)],
        'requested_band': [float(band_lo_req), float(band_hi_req)],
        'data_length_sec': float(N * dt),
        'freq_range': [float(f[1] if f.size > 1 else 0.0), float(f[-1] if f.size else 0.0)],
        'nyquist_freq': float(nyq),
        'sampling_rate': float(fs)
    }
    return pts, mean_rms, fs, info

def noise_psd_lowband(
    data: Union[pd.DataFrame, np.ndarray],
    band: Optional[Tuple[float, float]] = None,
    targetf: Optional[float] = None,
    half_bw: Optional[float] = None,
    *,
    # auto-preprocess (anti-alias LPF + downsample) controls
    enable_auto_preprocess: bool = True,
    lowband_hi: float = 150.0,          # define "low-frequency" as 0–150 Hz
    oversample: float = 5.0,            # target fs ≈ oversample * band_hi (recommended 4–6)
    fs_floor: float = 250.0,            # do not downsample below this fs (keeps df reasonable)
    fs_ceil: float = 800.0,             # do not keep fs higher than this for low-band tasks
    lp_margin: float = 0.30,            # LPF cutoff ~ (1+lp_margin) * band_hi
    fir_taps: int = 801,                # FIR length; longer = better stopband, slower
    # Welch/PSD defaults (low-frequency–friendly)
    min_freq_resolution: Optional[float] = None,
    force_long_segments: bool = False,
    window: str = "hann",
    average: str = "median",
    detrend: str = "linear",
    snap_to_power_of_2: bool = False,
) -> Tuple[np.ndarray, float, float, dict]:
    """
    Low-frequency–oriented PSD with automatic downsampling if the raw sample rate is excessive.

    Behavior:
      - If the analysis band upper edge (band_hi) ≤ lowband_hi (default 150 Hz) AND
        the raw sampling rate fs is much higher than necessary, the function will:
           (1) apply an anti-alias low-pass filter, then
           (2) downsample to a target rate ~ oversample * band_hi (bounded by [fs_floor, fs_ceil]).
      - Otherwise, it falls back to direct Welch PSD on the input data.

    Inputs
    ------
    data : DataFrame or ndarray
        Columns must be [time, signal1, signal2]; the PSD is computed on column #2.
    band : (f1, f2), optional
        Explicit analysis band edges in Hz.
    targetf, half_bw : optional
        Alternative band selection: center ± half bandwidth in Hz.
    enable_auto_preprocess : bool
        If True, auto LPF + downsample when band is in "low-frequency" (≤ lowband_hi) and fs is high.
    lowband_hi : float
        Upper edge that defines "low-frequency" analysis (default 150 Hz).
    oversample : float
        Target sampling rate ≈ oversample * band_hi (choose 4–6 for robust Welch).
    fs_floor, fs_ceil : float
        Clamp for the target sampling rate to avoid being too low or unnecessarily high.
    lp_margin : float
        Anti-alias LPF cutoff ≈ (1+lp_margin) * band_hi; keep some passband headroom.
    fir_taps : int
        FIR length for anti-alias LPF (used with filtfilt for zero-phase).

    Welch parameters (passed to `noise_psd`)
    ----------------------------------------
    min_freq_resolution, force_long_segments, window, average, detrend, snap_to_power_of_2

    Returns
    -------
    pts : ndarray (M, 2)
        Columns: [frequency, PSD] with PSD in V^2/Hz.
    mean_rms : float
        Mean ASD over the selected band: sqrt(mean(PSD_band)) in V/√Hz.
    fs_used : float
        Sampling rate used by Welch (raw or downsampled).
    info : dict
        Diagnostics (includes preprocessing flags and bands).
    """
    # --- import locals only when needed (keeps global namespace clean) ---
    from scipy.signal import firwin, filtfilt, resample_poly
    from fractions import Fraction

    # --- normalize input array ---
    if isinstance(data, pd.DataFrame):
        arr = data.values
    else:
        arr = np.asarray(data)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("Data must have at least 3 columns: [time, signal1, signal2].")

    time = np.asarray(arr[:, 0], dtype=float)
    y    = np.asarray(arr[:, 2], dtype=float)

    m = np.isfinite(time) & np.isfinite(y)
    time, y = time[m], y[m]
    if time.size < 100:
        raise ValueError("Insufficient valid data points (need at least 100).")

    dt = np.median(np.diff(time))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid time axis: cannot determine sampling interval.")
    fs_raw = 1.0 / dt

    # --- resolve analysis band and "band_hi" ---
    if band is not None:
        f1, f2 = float(band[0]), float(band[1])
        if f2 <= f1 or f1 < 0:
            raise ValueError("Invalid band; must satisfy 0 ≤ f1 < f2.")
        band_hi = f2
        fc_repr = 0.5 * (f1 + f2)
    else:
        if targetf is None or half_bw is None:
            raise ValueError("Provide either `band=(f1,f2)` or both `targetf` and `half_bw`.")
        band_hi = float(targetf) + float(half_bw)
        fc_repr = float(targetf)

    # --- set low-frequency–friendly default df if not provided ---
    if min_freq_resolution is None:
        if fc_repr < 2.0:
            min_freq_resolution = 0.05
        elif fc_repr < 10.0:
            min_freq_resolution = 0.10
        else:
            min_freq_resolution = 0.20

    # --- decide whether to preprocess (LPF + downsample) ---
    do_pp = False
    fs_target = fs_raw
    lp_cut = None

    if enable_auto_preprocess and (band_hi <= lowband_hi):
        # propose a target fs tailored to the selected low-frequency band
        fs_target = oversample * band_hi
        fs_target = max(fs_floor, min(fs_target, fs_ceil))  # clamp
        # only preprocess if we are actually reducing fs (avoid needless filtering)
        if fs_target < 0.95 * fs_raw:
            do_pp = True
            # choose anti-alias LPF cutoff (must be below old Nyquist and below new Nyquist)
            # give passband margin, but never exceed 0.45*fs_target
            lp_cut = min(0.45 * fs_target, (1.0 + lp_margin) * band_hi)

    if do_pp:
        # --- Anti-alias FIR LPF + zero-phase filtering ---
        # guard: if lp_cut is too close to old Nyquist, skip preprocess (shouldn't happen with 5 kHz raw)
        if lp_cut >= 0.48 * fs_raw:
            do_pp = False  # fall back to raw Welch
        else:
            b = firwin(int(fir_taps), lp_cut, fs=fs_raw)
            y_f = filtfilt(b, [1.0], y)

            # --- rational resampling factor (limits denominator for numerical stability) ---
            frac = Fraction(fs_target / fs_raw).limit_denominator(512)
            up, down = frac.numerator, frac.denominator
            y_ds = resample_poly(y_f, up, down)

            # rebuild uniform time axis
            t0 = time[0]
            fs_used = fs_raw * up / down
            time_ds = t0 + np.arange(y_ds.size) / fs_used

            data_pp = pd.DataFrame({
                "time":  time_ds,
                "x":     np.zeros_like(y_ds),
                "demod": y_ds
            })

            pts, mean_rms, fs_out, info = noise_psd(
                data=data_pp,
                band=band,
                targetf=targetf,
                half_bw=half_bw,
                window=window,
                average=average,
                min_freq_resolution=min_freq_resolution,
                force_long_segments=force_long_segments,
                detrend=detrend,
                snap_to_power_of_2=snap_to_power_of_2
            )
            # annotate preprocessing details
            info.update(dict(
                preprocessed=True,
                fs_raw=float(fs_raw),
                fs_target=float(fs_target),
                fs_used=float(fs_out),
                lp_cutoff=float(lp_cut),
                lowband_hi=float(lowband_hi),
                oversample=float(oversample)
            ))
            return pts, mean_rms, fs_out, info

    # --- No preprocessing path (either high-band analysis, or fs already modest) ---
    pts, mean_rms, fs_out, info = noise_psd(
        data=data,
        band=band,
        targetf=targetf,
        half_bw=half_bw,
        window=window,
        average=average,
        min_freq_resolution=min_freq_resolution,
        force_long_segments=force_long_segments,
        detrend=detrend,
        snap_to_power_of_2=snap_to_power_of_2
    )
    info.update(dict(
        preprocessed=False,
        fs_raw=float(fs_raw),
        fs_target=float(fs_target),
        fs_used=float(fs_out),
        lp_cutoff=(None if lp_cut is None else float(lp_cut)),
        lowband_hi=float(lowband_hi),
        oversample=float(oversample)
    ))
    return pts, mean_rms, fs_out, info
