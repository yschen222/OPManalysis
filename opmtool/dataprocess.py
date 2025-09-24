# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 17:38:40 2025

@author: YiHsuanChen
"""

import numpy as np
import pandas as pd
from scipy.signal import periodogram

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
    map_mode='global',             # 'global' (default) | 'local'  ← see docstring
    require_tri=True,
    return_debug=False
):
    """
    Cut one or more sweep segments **using the measured triangle/sawtooth waveform only**
    and convert the selected segment(s) to magnetic field values according to the given
    (B_min, B_max).

    This function is designed for OPM scans where each record often contains only a
    rising ramp or only a falling ramp (i.e., sawtooth). Segmentation relies solely
    on the triangle/sawtooth monotonic runs; the absorption signal is used only to
    choose an anchor segment when requested, and never to decide the boundaries.

    Parameters
    ----------
    data : pandas.DataFrame or ndarray
        - DataFrame must include columns: [time_col, ab_col, demod_col, tri_col].
        - ndarray must be shape (N,4): [t, Ab, demod, tri].
    B_range : (float, float)
        (B_min, B_max) corresponding to the triangle extrema for *one sweep*.
        Units are arbitrary but must match your intended nT scale.
    cycles : int, optional
        Number of consecutive segments to cut.
        * For segment='half', this is the number of half-cycles (rising-only or falling-only).
        * For segment='full', each cycle contains one rising + one falling run.
    segment : {'half','full'}, optional
        - 'half': cut strictly monotonic runs (min→max rising, or max→min falling).
        - 'full': cut peak-to-peak windows (rise+fall), detected by turning points.
    direction : {'auto','rising','falling'}, optional
        Which half to select when segment='half'.
        - 'auto' infers from the derivative sign distribution of the triangle.
    time_col, ab_col, demod_col, tri_col : str
        Column names (used only when `data` is a DataFrame).
    smooth_win : int, optional
        Moving-average window (in samples) for the triangle to make segmentation robust.
        Set <=1 to disable smoothing.
    anchor : {'abspeak','center','first','last'}, optional
        Which segment to start from before stitching `cycles` consecutive segments:
        - 'abspeak': the segment containing max |Ab - baseline| (baseline from edges).
        - 'center' : the segment near the middle of the record.
        - 'first'  : the earliest valid segment.
        - 'last'   : the latest valid segment.
    edge_clip_frac : float, optional
        Fraction of samples to drop from each end of the chosen segment(s) to avoid
        turn-around artifacts and coil dynamics (e.g., 0.02 = 2%).
    map_mode : {'global','local'}, optional
        How to map triangle values to B after edge clipping:
        - 'global' (recommended, default): use the **pre-clip** min/max of the chosen
          segment(s) to map to (B_min, B_max). This means after clipping 2% the resulting
          `BField` will **not** be stretched back to ±range; it naturally stays *within*
          your requested range (e.g., ±34 nT).
        - 'local': use the **post-clip** min/max to map to (B_min, B_max).
          This stretches the clipped segment back to full range (typically *not* what you want).
    require_tri : bool, optional
        If True, raise if triangle column is missing (recommended).
    return_debug : bool, optional
        If True, also return a debug dict with segmentation details.

    Returns
    -------
    BField : ndarray
        Magnetic field axis mapped from the triangle according to `map_mode`.
    AbCut : ndarray
        Absorption signal cut to the selected segment(s).
    DemodCut : ndarray
        Demodulated (lock-in) signal cut to the selected segment(s).
    debug : dict (only if return_debug=True)
        Keys include:
        - 'direction'     : 'rising' or 'falling' used for 'half'
        - 'runs'          : list of (start, end) monotonic runs on the triangle
        - 'chosen_full'   : (s_full, e_full) indices before edge clipping
        - 'chosen_clipped': (s0, e0) indices after edge clipping
        - 'tri_span_full' : (min, max) of the pre-clip triangle in the chosen runs
        - 'tri_span_local': (min, max) of the post-clip triangle
        - 'map_mode'      : 'global' or 'local'

    Notes
    -----
    * Segmentation is performed on the triangle/sawtooth only:
      - 'half' uses longest monotonic runs (d(tri)/dt > 0 for rising, < 0 for falling),
        robust to small noise via smoothing and minimal-length filtering.
      - 'full' uses turning points (sign changes in derivative) to get peak-to-peak windows.
    * The absorption signal is **not** used to define boundaries. It is only used by
      `anchor='abspeak'` to pick where to start.
    * If you need your entire B axis **strictly within** ±range after clipping, use
      the default `map_mode='global'`.
    """
    import numpy as np
    import pandas as pd

    # ---------- extract columns ----------
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        for c in [time_col, ab_col, demod_col]:
            if c not in df.columns:
                raise ValueError(f"DataFrame needs '{time_col}', '{ab_col}', '{demod_col}'.")
        if tri_col not in df.columns:
            if require_tri:
                raise ValueError(f"Missing triangle column '{tri_col}'.")
            else:
                raise ValueError("Triangle waveform required for reliable cutting.")
        t     = df[time_col].to_numpy()
        Ab    = df[ab_col].to_numpy()
        demod = df[demod_col].to_numpy()
        tri   = df[tri_col].to_numpy()
    else:
        arr = np.asarray(data)
        if arr.ndim < 2 or arr.shape[1] < 4:
            raise ValueError("ndarray must be (N,4): [t, Ab, demod, tri].")
        t, Ab, demod, tri = arr[:,0], arr[:,1], arr[:,2], arr[:,3]

    B_min, B_max = B_range
    N = len(Ab)
    if N < 16:
        raise ValueError("Not enough samples.")

    def _smooth(x, w):
        if w is None or w <= 1: return x
        k = np.ones(int(w))/int(w)
        return np.convolve(x, k, mode='same')

    def _edge_baseline(y, frac=0.1):
        n = max(1, int(len(y)*frac))
        return float(np.median(np.r_[y[:n], y[-n:]]))

    # ---------- build monotonic runs on the triangle ----------
    tri_s = _smooth(tri, smooth_win)
    dtri  = np.diff(tri_s)

    if direction == 'auto':
        dir_mode = 'rising' if np.sum(dtri > 0) >= np.sum(dtri < 0) else 'falling'
    else:
        dir_mode = 'rising' if direction == 'rising' else 'falling'

    good = (dtri > 0) if dir_mode == 'rising' else (dtri < 0)
    edges = np.diff(np.r_[False, good, False])
    starts = np.where(edges ==  1)[0]
    ends   = np.where(edges == -1)[0]
    runs = [(int(s), int(e)) for s, e in zip(starts, ends) if (e - s) >= 8]
    if not runs:  # last resort
        runs = [(0, N)]

    # full cycles: merge rise+fall via turning points
    if segment == 'full':
        sign = np.sign(dtri)
        tps = 1 + np.where(sign[:-1] * sign[1:] < 0)[0]  # turning points
        if len(tps) >= 2:
            cand = [(int(tps[i]), int(tps[i+1])) for i in range(len(tps)-1)]
            runs = [rc for rc in cand if (rc[1]-rc[0]) >= 8]

    # ---------- choose anchor run ----------
    if anchor == 'first':
        run0 = 0
    elif anchor == 'last':
        run0 = len(runs) - 1
    elif anchor == 'center':
        mid = N // 2
        run0 = int(np.argmin([abs((s+e)//2 - mid) for s, e in runs]))
    else:  # 'abspeak'
        c0 = _edge_baseline(Ab)
        idx_peak = int(np.argmax(np.abs(Ab - c0)))
        run0 = 0
        for i, (s, e) in enumerate(runs):
            if s <= idx_peak < e:
                run0 = i
                break

    # ---------- stitch requested number of runs ----------
    run1 = min(len(runs), run0 + int(cycles))
    s_full, e_full = runs[run0][0], runs[run1-1][1]

    # Record full (pre-clip) triangle span for 'global' mapping
    tri_full = tri[s_full:e_full]
    tr_min_full = float(np.min(tri_full))
    tr_max_full = float(np.max(tri_full))

    # edge clipping to avoid artefacts near boundaries
    clip = int(max(0.0, float(edge_clip_frac)) * (e_full - s_full))
    s0 = min(max(0, s_full + clip), e_full - 1)
    e0 = max(min(N, e_full - clip), s0 + 1)

    tri_cut   = tri[s0:e0]
    AbCut     = Ab[s0:e0]
    DemodCut  = demod[s0:e0]

    # ---------- map triangle -> B ----------
    if map_mode == 'global':
        # Use pre-clip extrema: clipping does NOT stretch back to ±range
        tr_lo, tr_hi = tr_min_full, tr_max_full
    else:  # 'local' (stretches the clipped segment to full range)
        tr_lo, tr_hi = float(np.min(tri_cut)), float(np.max(tri_cut))

    if not np.isfinite(tr_lo) or not np.isfinite(tr_hi) or tr_hi <= tr_lo:
        BField = np.full_like(tri_cut, (B_min + B_max)/2.0, dtype=float)
    else:
        scale  = (B_max - B_min) / (tr_hi - tr_lo + 1e-30)
        BField = B_min + (tri_cut - tr_lo) * scale
        lo, hi = (B_min, B_max) if B_min <= B_max else (B_max, B_min)
        BField = np.clip(BField, lo, hi)  # numeric safety

    if return_debug:
        dbg = dict(
            direction=dir_mode,
            runs=runs,
            chosen_full=(s_full, e_full),
            chosen_clipped=(s0, e0),
            tri_span_full=(tr_min_full, tr_max_full),
            tri_span_local=(float(np.min(tri_cut)) if len(tri_cut) else np.nan,
                            float(np.max(tri_cut)) if len(tri_cut) else np.nan),
            map_mode=map_mode
        )
        return BField, AbCut, DemodCut, dbg
    return BField, AbCut, DemodCut




def noise_psd(data, targetf, half_bw):
    """
    Estimate the periodogram and the RMS noise spectral density around a target frequency
    using the Welch periodogram method.

    Parameters:
        data : DataFrame or ndarray
            Columns = [time, signal1, signal2]
            Noise is assumed to be in column 2 (index 2), which is the demodulated signal after LIA.

        targetf : float
            Target frequency in Hz (e.g., modulation freq)

        half_bw : float
            Bandwidth half-width in Hz (e.g., ±1 Hz range)

    Returns:
        pts      : array of (frequency, PSD)
        mean_rms : mean RMS noise (V/√Hz) within selected band
        fs       : sampling frequency
    """
    if isinstance(data, pd.DataFrame):
        arr = data.values
    else:
        arr = np.asarray(data)

    time  = arr[:, 0]
    noise = arr[:, 2]

    dt = np.median(np.diff(time))
    fs = int(round(1.0 / dt))

    f, Pxx = periodogram(
        noise,
        fs=fs,
        window='boxcar',
        scaling='density',
        detrend=False
    )
    pts = np.column_stack((f, Pxx))

    mask = (f >= targetf - half_bw) & (f <= targetf + half_bw)
    mean_noise = Pxx[mask].mean()
    mean_rms = np.sqrt(mean_noise)              # V/√Hz

    return pts, mean_rms, fs
