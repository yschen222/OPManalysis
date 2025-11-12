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
    Welch PSD around a *user-chosen band*.
    Supports input with 2 columns [time, signal] or 3 columns [time, signal1, signal2].
    """
    # ---------- Load & validate ----------
    if isinstance(data, pd.DataFrame):
        arr = data.values
    else:
        arr = np.asarray(data)

    # ✅ 支援兩種輸入格式
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("Data must have 2 or 3 columns: [time, signal] or [time, signal1, signal2].")

    time = np.asarray(arr[:, 0], dtype=float)
    if arr.shape[1] == 2:
        noise = np.asarray(arr[:, 1], dtype=float)   # 兩欄格式
    else:
        noise = np.asarray(arr[:, 2], dtype=float)   # 三欄格式

    msk = np.isfinite(time) & np.isfinite(noise)
    time, noise = time[msk], noise[msk]
    if time.size < 100:
        raise ValueError("Insufficient valid data points (need at least 100).")

    # ---------- Band selection ----------
    if band is not None:
        f1, f2 = float(band[0]), float(band[1])
        if f2 <= f1 or f1 < 0:
            raise ValueError("Invalid band; must satisfy 0 ≤ f1 < f2.")
        band_lo_req, band_hi_req = f1, f2
        band_width = f2 - f1
        band_center = 0.5*(f1 + f2)
    else:
        if targetf is None or half_bw is None:
            raise ValueError("Provide either `band=(f1, f2)` or (`targetf`, `half_bw`).")
        band_center = float(targetf)
        band_width  = 2.0 * float(half_bw)
        band_lo_req = band_center - half_bw
        band_hi_req = band_center + half_bw

    # ---------- Sampling rate ----------
    dt = np.median(np.diff(time))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid time axis.")
    fs = 1.0 / dt
    N  = noise.size
    nyq = fs / 2.0

    if band_hi_req >= nyq:
        warnings.warn(f"Requested upper frequency {band_hi_req:.1f} Hz >= Nyquist ({nyq:.1f} Hz).", RuntimeWarning)

    # ---------- Welch config ----------
    if min_freq_resolution is None:
        target_bins = 12.0
        min_freq_resolution = max(band_width / target_bins, 1e-9)
    nperseg = int(max(256, min(fs / min_freq_resolution, N)))
    noverlap = nperseg // 2

    # ---------- Welch PSD ----------
    f, Pxx = welch(
        noise, fs=fs, window=window,
        nperseg=nperseg, noverlap=noverlap,
        detrend=detrend, scaling='density', average=average
    )
    pts = np.column_stack((f, Pxx))

    # ---------- Band mask ----------
    band_lo = max(band_lo_req, f[1] if f.size > 1 else 0.0)
    band_hi = min(band_hi_req, nyq)
    mask = (f >= band_lo) & (f <= band_hi)
    if not np.any(mask):
        idx = int(np.argmin(np.abs(f - band_center)))
        mask = np.zeros_like(f, dtype=bool)
        mask[idx] = True
        warnings.warn(f"No bins in band [{band_lo_req},{band_hi_req}] Hz; using nearest {f[idx]:.2f} Hz.", RuntimeWarning)

    f_band = f[mask]
    P_band = Pxx[mask]
    good = np.isfinite(P_band)
    P_band = P_band[good]

    mean_rms = float(np.sqrt(np.mean(P_band))) if P_band.size else np.nan
    df_bin = f[1] - f[0] if f.size > 1 else np.nan

    info = dict(
        nperseg=int(nperseg),
        noverlap=int(noverlap),
        df=float(df_bin),
        fs=float(fs),
        band_range=[float(band_lo), float(band_hi)],
        band_requested=[float(band_lo_req), float(band_hi_req)],
        nyquist=float(nyq),
        n_points=int(N)
    )
    return pts, mean_rms, fs, info

def noise_psd_lowband(
    data: Union[pd.DataFrame, np.ndarray],
    band: Optional[Tuple[float, float]] = None,
    targetf: Optional[float] = None,
    half_bw: Optional[float] = None,
    *,
    enable_auto_preprocess: bool = True,
    lowband_hi: float = 150.0,
    oversample: float = 5.0,
    fs_floor: float = 250.0,
    fs_ceil: float = 800.0,
    lp_margin: float = 0.30,
    fir_taps: int = 801,
    min_freq_resolution: Optional[float] = None,
    force_long_segments: bool = False,
    window: str = "hann",
    average: str = "median",
    detrend: str = "linear",
    snap_to_power_of_2: bool = False,
) -> Tuple[np.ndarray, float, float, dict]:
    """
    相容新版 2 欄 [time, signal] 以及舊版 3 欄 [time, signal1, signal2]。
    """
    from scipy.signal import firwin, filtfilt, resample_poly
    from fractions import Fraction

    # --- normalize input array ---
    if isinstance(data, pd.DataFrame):
        arr = data.values
    else:
        arr = np.asarray(data)

    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("Data must have 2 or 3 columns: [time, signal] or [time, signal1, signal2].")

    time = np.asarray(arr[:, 0], dtype=float)
    if arr.shape[1] == 2:
        y = np.asarray(arr[:, 1], dtype=float)
    else:
        y = np.asarray(arr[:, 2], dtype=float)

    m = np.isfinite(time) & np.isfinite(y)
    time, y = time[m], y[m]
    if time.size < 100:
        raise ValueError("Insufficient valid data points (need at least 100).")

    dt = np.median(np.diff(time))
    fs_raw = 1.0 / dt

    # --- resolve band ---
    if band is not None:
        f1, f2 = float(band[0]), float(band[1])
        band_hi = f2
        fc_repr = 0.5 * (f1 + f2)
    else:
        if targetf is None or half_bw is None:
            raise ValueError("Provide `band=(f1,f2)` or (`targetf`,`half_bw`).")
        band_hi = float(targetf) + float(half_bw)
        fc_repr = float(targetf)

    if min_freq_resolution is None:
        if fc_repr < 2.0:
            min_freq_resolution = 0.05
        elif fc_repr < 10.0:
            min_freq_resolution = 0.10
        else:
            min_freq_resolution = 0.20

    # --- 是否降頻 ---
    do_pp = False
    fs_target = fs_raw
    lp_cut = None

    if enable_auto_preprocess and (band_hi <= lowband_hi):
        fs_target = max(fs_floor, min(oversample * band_hi, fs_ceil))
        if fs_target < 0.95 * fs_raw:
            do_pp = True
            lp_cut = min(0.45 * fs_target, (1.0 + lp_margin) * band_hi)

    if do_pp:
        if lp_cut >= 0.48 * fs_raw:
            do_pp = False
        else:
            b = firwin(int(fir_taps), lp_cut, fs=fs_raw)
            y_f = filtfilt(b, [1.0], y)

            frac = Fraction(fs_target / fs_raw).limit_denominator(512)
            up, down = frac.numerator, frac.denominator
            y_ds = resample_poly(y_f, up, down)

            t0 = time[0]
            fs_used = fs_raw * up / down
            time_ds = t0 + np.arange(y_ds.size) / fs_used
            df_pp = pd.DataFrame({"time": time_ds, "signal": y_ds})

            pts, mean_rms, fs_out, info = noise_psd(
                df_pp, band=band, targetf=targetf, half_bw=half_bw,
                window=window, average=average,
                min_freq_resolution=min_freq_resolution,
                force_long_segments=force_long_segments,
                detrend=detrend, snap_to_power_of_2=snap_to_power_of_2
            )
            info.update(dict(
                preprocessed=True,
                fs_raw=float(fs_raw),
                fs_target=float(fs_target),
                fs_used=float(fs_out),
                lp_cutoff=float(lp_cut)
            ))
            return pts, mean_rms, fs_out, info

    # --- No preprocess ---
    pts, mean_rms, fs_out, info = noise_psd(
        data=data, band=band, targetf=targetf, half_bw=half_bw,
        window=window, average=average,
        min_freq_resolution=min_freq_resolution,
        force_long_segments=force_long_segments,
        detrend=detrend, snap_to_power_of_2=snap_to_power_of_2
    )
    info.update(dict(
        preprocessed=False,
        fs_raw=float(fs_raw),
        fs_target=float(fs_target),
        fs_used=float(fs_out),
        lp_cutoff=(None if lp_cut is None else float(lp_cut))
    ))
    return pts, mean_rms, fs_out, info
