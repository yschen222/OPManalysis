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

def one_cycle_cut(data, sweepB, sweepf):
    """
    Automatically extracts a full sweep cycle centered at the absorption peak, and converts the time axis into magnetic field values based on a known triangular ramp function.


    Parameters:
        data : DataFrame or ndarray
            Columns = [time, Ab, demod] or [t, Ab, demod]

        sweepB : float
            Sweep amplitude (e.g., nT)

        sweepf : float
            Sweep frequency of triangle ramp function (Hz)

    Returns:
        BField    : 1D ndarray of magnetic field values (centered at peak)
        AbCut     : Cut absorption signal
        DemodCut  : Cut demodulated signal
    """
    if isinstance(data, pd.DataFrame):
        t, Ab, demod = data.values.T
    else:
        t, Ab, demod = data.T

    N = len(Ab)
    center_idx = N // 2
    center_left  = center_idx - N // 3
    center_right = min(center_idx + N // 3, N - 1)

    # Smooth & pick peak
    smooth_Ab = np.convolve(Ab, np.ones(10)/10, mode='same')
    peak1 = center_left + np.argmax(smooth_Ab[center_left:center_right+1])
    peak2 = center_left + np.argmax(Ab[center_left:center_right+1])
    peak_idx = peak1 if Ab[peak1] > Ab[peak2] else peak2

    # Search valleys around peak
    range_size = N // 3
    left_r  = max(0, peak_idx - range_size)
    right_r = min(N - 1, peak_idx + range_size)

    left_idx  = left_r + np.argmin(Ab[left_r : peak_idx+1])
    right_idx = peak_idx + np.argmin(Ab[peak_idx : right_r+1])

    # Cut and convert
    TimeCut  = t[left_idx : right_idx+1]
    AbCut    = Ab[left_idx : right_idx+1]
    DemodCut = demod[left_idx : right_idx+1]

    stepB  = sweepB * sweepf * 4   # dB/dt in nT/s
    BField = (TimeCut - t[peak_idx]) * stepB

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
