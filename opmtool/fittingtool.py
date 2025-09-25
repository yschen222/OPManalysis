# -*- coding: utf-8 -*-
"""
The codes are used to fitting OPM signal by experiments.
Created on Tue Jul  8 16:37:15 2025

@author: YiHsuanChen
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
from lmfit import Model
from lmfit.models import GaussianModel, ConstantModel, VoigtModel
from scipy.special import wofz

# =========================
# Helpers
# =========================
def _edge_baseline(y, frac=0.1):
    """Estimate a constant baseline using the median of samples from both edges."""
    n = max(1, int(len(y)*frac))
    return float(np.median(np.r_[y[:n], y[-n:]]))

def _baseline_linear_edges(x, y, frac=0.15):
    """
    Estimate a linear baseline y ≈ c0 + c1*x using points taken from the
    left and right edges (each side uses a fraction 'frac' of the data).
    Returns (c0, c1).
    """
    n = max(3, int(len(x)*frac))
    X = np.r_[x[:n], x[-n:]]
    Y = np.r_[y[:n], y[-n:]]
    if len(X) < 3:
        return float(np.median(Y)), 0.0
    b1, b0 = np.polyfit(X, Y, 1)  # y = b1*x + b0
    return float(b0), float(b1)

def _refine_center_quadratic(x, y_corr, guess_idx, win=3):
    """
    Refine the resonance center near `guess_idx` by a local quadratic fit.

    Strategy
    --------
    1) Take a small window [guess_idx-win, guess_idx+win] around the provisional center.
    2) Guard against ill-conditioned fits (too few points, tiny dynamic range).
       If the window is not suitable, fall back to a 3-point parabolic vertex
       using (idx-1, idx, idx+1). If even that is unavailable, return x[guess_idx].
    3) Standardize X before quadratic fit to improve conditioning.
    4) Compute the vertex x* = -b/(2a) and map it back to the original scale.
       If the vertex lies outside the local window, return x[guess_idx].

    Parameters
    ----------
    x : array-like
        Field axis (1D).
    y_corr : array-like
        Baseline-corrected signal (1D), aligned with `x`.
    guess_idx : int
        Index of the provisional center (e.g., max |y_corr|).
    win : int, optional
        Half-window size for the local quadratic fit (default: 3).

    Returns
    -------
    float
        Refined center position in the same units as `x`.
    """
    import numpy as np, warnings

    i0 = max(0, guess_idx - win)
    i1 = min(len(x), guess_idx + win + 1)
    X = np.asarray(x[i0:i1], dtype=float)
    Y = np.asarray(y_corr[i0:i1], dtype=float)

    # Guard: insufficient points or negligible dynamic range
    if (len(X) < 3) or (np.ptp(X) < 1e-12) or (np.ptp(Y) < 10*np.finfo(float).eps):
        # 3-point parabolic fallback around guess_idx (if available)
        if 1 <= guess_idx < len(x) - 1:
            x1, x2, x3 = float(x[guess_idx-1]), float(x[guess_idx]), float(x[guess_idx+1])
            y1, y2, y3 = float(y_corr[guess_idx-1]), float(y_corr[guess_idx]), float(y_corr[guess_idx+1])
            denom = (y1 - 2.0*y2 + y3)
            if abs(denom) > 1e-20:
                # Vertex of a parabola through three equally spaced points
                dx = (x3 - x1) / 2.0
                xc = x2 + 0.5*(y1 - y3)/denom * dx
                # Clamp to neighbors for safety
                return float(min(max(xc, min(x1, x3)), max(x1, x3)))
        return float(x[guess_idx])

    # Standardize X to improve conditioning
    xm, xs = X.mean(), X.std()
    if xs < 1e-20:
        return float(x[guess_idx])
    Xn = (X - xm) / xs

    # Suppress RankWarning from poorly conditioned polyfits
    with warnings.catch_warnings():
        from numpy import RankWarning
        warnings.simplefilter("ignore", RankWarning)
        a, b, c = np.polyfit(Xn, Y, 2)

    if abs(a) < 1e-20:
        return float(x[guess_idx])

    # Vertex in normalized space, then map back
    xc_n = -b / (2.0*a)
    xc = xc_n * xs + xm

    # If vertex falls outside the local window, keep original guess
    if (xc < X.min()) or (xc > X.max()):
        return float(x[guess_idx])
    return float(xc)


def _halfmax_width(x, y, baseline):
    """
    Estimate FWHM via linear interpolation between half-maximum crossings
    along the direction of the dominant extremum. Returns NaN on failure.
    'baseline' should be a constant; if a linear baseline was removed,
    pass 0.0 here.
    """
    idx = np.argmax(np.abs(y - baseline))
    peak = y[idx]
    level = baseline + 0.5*(peak - baseline)
    try:
        # left crossing
        left = np.where((y[:idx]-level)*(np.roll(y, -1)[:idx]-level) <= 0)[0][-1]
        x1, x2, y1, y2 = x[left], x[left+1], y[left], y[left+1]
        xl = x1 + (level - y1) * (x2 - x1) / (y2 - y1 + 1e-15)
        # right crossing
        right = np.where((y[idx:]-level)*(np.roll(y, -1)[idx:]-level) <= 0)[0][0] + idx
        x1, x2, y1, y2 = x[right], x[right+1], y[right], y[right+1]
        xr = x1 + (level - y1) * (x2 - x1) / (y2 - y1 + 1e-15)
        return abs(xr - xl)
    except Exception:
        return np.nan

def _voigt_profile(x, sigma, gamma):
    """Area-normalized Voigt profile implemented via the Faddeeva function wofz."""
    sigma = max(1e-12, float(abs(sigma)))
    gamma = max(1e-12, float(abs(gamma)))
    z = (np.asarray(x) + 1j*gamma) / (sigma*np.sqrt(2.0))
    return np.real(wofz(z)) / (sigma*np.sqrt(2.0*np.pi))

def _r2(y, yfit):
    """Coefficient of determination R^2."""
    ss_res = np.sum((y - yfit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res/ss_tot if ss_tot > 0 else np.nan


# =========================
# Lorentz (with center shift)
# f(B) = a / (1 + (b*(B - center))^2) + c
# with FWHM = 2/|b| and HWHM (gamma) = 1/|b|
# =========================
def lorentz_fit(data, p0=None, bounds=None):
    """
    Fit a shifted Lorentzian with a constant baseline.

    Parameters
    ----------
    data : DataFrame or ndarray
        Must contain columns 'B' (x-axis) and 'Ab' (signal).
    p0 : None | tuple/list | dict
        Initial guess. If None, estimated automatically.
        Tuple/list: (a, b, center, c) or (a, b, c) for backward-compat.
        Dict keys: {'amplitude','b'|'gamma'|'FWHM','center','offset'}.
    bounds : dict | None
        Optional parameter bounds. Accepted keys:
        {'amplitude','b','gamma','FWHM','center','offset'}.

    Returns
    -------
    dict with keys:
        out, S_fit, Params, FWHM, sigma(None), gamma, Data, R2
        Params also includes 'b' for convenience.
    """
    # Input normalization
    if isinstance(data, np.ndarray):
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("numpy array must be N×2 (B, Ab)")
        df = pd.DataFrame(data[:, :2], columns=['B','Ab'])
    else:
        df = data.copy()
        if 'B' not in df.columns or 'Ab' not in df.columns:
            if df.shape[1] >= 2:
                df = df.copy()
                df.columns = ['B', 'Ab'] + list(df.columns[2:])
            else:
                raise ValueError("data must contain columns B and Ab")
    x = df['B'].values
    y = df['Ab'].values

    def f(B, a, b, center, c):
        return a / (1.0 + (b*(B - center))**2) + c

    model = Model(f)

    # Auto seed: remove linear baseline, refine center, estimate width
    c0_lin, c1_lin = _baseline_linear_edges(x, y, frac=0.15)
    y_corr = y - (c0_lin + c1_lin*x)
    idx = np.argmax(np.abs(y_corr))
    center0 = _refine_center_quadratic(x, y_corr, idx, win=3)
    a0 = float(y_corr[idx])
    fwhm_est = _halfmax_width(x, y_corr, baseline=0.0)
    if not np.isfinite(fwhm_est) or fwhm_est <= 0:
        fwhm_est = (x.max() - x.min())/5.0
    b0 = 2.0/max(1e-12, fwhm_est)
    c0 = _edge_baseline(y)

    # p0 handling
    if p0 is None:
        p0 = (a0, b0, center0, c0)
    elif isinstance(p0, (list, tuple)):
        if len(p0) == 3:
            p0 = (p0[0], p0[1], 0.0, p0[2])
        elif len(p0) != 4:
            raise ValueError("p0 tuple/list must be (a,b,center,c) or (a,b,c)")
    elif isinstance(p0, dict):
        a0_d = p0.get('amplitude', a0)
        center0_d = p0.get('center', center0)
        c0_d = p0.get('offset', c0)
        if 'b' in p0:
            b0_d = p0['b']
        elif 'gamma' in p0:
            b0_d = 1.0/max(1e-12, abs(p0['gamma']))
        elif 'FWHM' in p0:
            b0_d = 2.0/max(1e-12, abs(p0['FWHM']))
        else:
            b0_d = b0
        p0 = (a0_d, b0_d, center0_d, c0_d)
    else:
        raise ValueError("p0 must be None / tuple(list) / dict")

    pars = model.make_params(a=p0[0], b=p0[1], center=p0[2], c=p0[3])

    # Loose automatic bounds for center if not given
    span = (x.max() - x.min()) if x.max() > x.min() else 1.0
    if not (isinstance(bounds, dict) and 'center' in bounds):
        pars['center'].set(min=p0[2] - 0.25*span, max=p0[2] + 0.25*span)

    # Other bounds
    if isinstance(bounds, dict):
        if 'amplitude' in bounds:
            lo, hi = bounds['amplitude']; pars['a'].set(min=lo, max=hi)
        if 'center' in bounds:
            lo, hi = bounds['center']; pars['center'].set(min=lo, max=hi)
        if 'offset' in bounds:
            lo, hi = bounds['offset']; pars['c'].set(min=lo, max=hi)
        if 'b' in bounds:
            lo, hi = bounds['b']; pars['b'].set(min=lo, max=hi)
        else:
            # Infer symmetric bounds on b from gamma/FWHM if provided
            b_lo_mag, b_hi_mag = 0.0, np.inf
            if 'gamma' in bounds:
                glo, ghi = bounds['gamma']
                if np.isfinite(ghi) and ghi > 0: b_lo_mag = max(b_lo_mag, 1.0/ghi)
                if np.isfinite(glo) and glo > 0: b_hi_mag = min(b_hi_mag, 1.0/glo)
            if 'FWHM' in bounds:
                flo, fhi = bounds['FWHM']
                if np.isfinite(fhi) and fhi > 0: b_lo_mag = max(b_lo_mag, 2.0/fhi)
                if np.isfinite(flo) and flo > 0: b_hi_mag = min(b_hi_mag, 2.0/flo)
            if np.isfinite(b_hi_mag):
                pars['b'].set(min=-b_hi_mag, max=b_hi_mag)

    out = model.fit(y, pars, B=x)
    yfit = out.best_fit

    a = out.params['a'].value
    b = out.params['b'].value
    center = out.params['center'].value
    c = out.params['c'].value

    FWHM = 2.0/abs(b) if b != 0 else np.inf
    gamma = FWHM/2.0

    return {
        "out": out,
        "S_fit": yfit,
        "Params": {"amplitude": a, "center": center, "gamma": gamma, "offset": c, "b": b},
        "FWHM": FWHM,
        "sigma": None,
        "gamma": gamma,
        "Data": df.assign(S_fit=yfit),
        "R2": _r2(y, yfit)
    }


# =========================
# Gaussian + constant baseline
# =========================
def gauss_fit(data, p0=None, bounds=None):
    """
    Fit a Gaussian with a constant baseline.

    Parameters
    ----------
    data : DataFrame or ndarray with columns 'B','Ab'
    p0 : dict | None
        Optional initial guess with keys {'amplitude','center','sigma','offset'}.
        If None, seeds are estimated automatically.
    bounds : dict | None
        Optional bounds for the same keys.

    Returns
    -------
    dict with keys:
        out, S_fit, Params, FWHM, sigma, gamma(None), Data, R2
    """
    if isinstance(data, np.ndarray):
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("numpy array must be N×2 (B, Ab)")
        df = pd.DataFrame(data[:, :2], columns=['B','Ab'])
    else:
        df = data.copy()
        if 'B' not in df.columns or 'Ab' not in df.columns:
            if df.shape[1] >= 2:
                df = df.copy()
                df.columns = ['B', 'Ab'] + list(df.columns[2:])
            else:
                raise ValueError("data must contain columns B and Ab")
    x, y = df['B'].values, df['Ab'].values

    gmod = GaussianModel(prefix='g_')
    cmod = ConstantModel(prefix='c_')
    model = gmod + cmod

    # Auto seeds: remove linear baseline, refine center, estimate width
    c0_lin, c1_lin = _baseline_linear_edges(x, y, frac=0.15)
    y_corr = y - (c0_lin + c1_lin*x)
    idx = np.argmax(np.abs(y_corr))
    mu0 = _refine_center_quadratic(x, y_corr, idx, win=3)
    A0_height = y_corr[idx]
    fwhm_est = _halfmax_width(x, y_corr, baseline=0.0)
    if not np.isfinite(fwhm_est) or fwhm_est <= 0:
        fwhm_est = (x.max() - x.min())/5.0
    sigma0 = fwhm_est/(2*np.sqrt(2*np.log(2)))
    A0_area = A0_height * np.sqrt(2*np.pi) * abs(sigma0)

    pars = model.make_params(
        g_amplitude=A0_area,
        g_center=mu0,
        g_sigma=max(1e-12, abs(sigma0)),
        c_c=_edge_baseline(y)
    )

    # Loose automatic bounds for center if not given
    span = (x.max() - x.min()) if x.max() > x.min() else 1.0
    if not (isinstance(bounds, dict) and 'center' in bounds):
        pars['g_center'].set(min=mu0 - 0.25*span, max=mu0 + 0.25*span)

    # Manual p0 override
    if isinstance(p0, dict):
        if 'amplitude' in p0: pars['g_amplitude'].set(value=p0['amplitude'])
        if 'center'    in p0: pars['g_center'  ].set(value=p0['center'])
        if 'sigma'     in p0: pars['g_sigma'   ].set(value=max(1e-12, abs(p0['sigma'])))
        if 'offset'    in p0: pars['c_c'       ].set(value=p0['offset'])

    # Bounds
    if isinstance(bounds, dict):
        if 'amplitude' in bounds:
            lo, hi = bounds['amplitude']; pars['g_amplitude'].set(min=lo, max=hi)
        if 'center' in bounds:
            lo, hi = bounds['center']; pars['g_center'].set(min=lo, max=hi)
        if 'sigma' in bounds:
            lo, hi = bounds['sigma']; pars['g_sigma'].set(min=max(1e-12, lo), max=hi)
        if 'offset' in bounds:
            lo, hi = bounds['offset']; pars['c_c'].set(min=lo, max=hi)

    out = model.fit(y, pars, x=x)
    yfit = out.best_fit

    amp   = out.params['g_amplitude'].value
    center= out.params['g_center'].value
    sigma = abs(out.params['g_sigma'].value)
    offset= out.params['c_c'].value

    FWHM = 2*np.sqrt(2*np.log(2))*sigma

    return {
        "out": out,
        "S_fit": yfit,
        "Params": {"amplitude": amp, "center": center, "sigma": sigma, "offset": offset},
        "FWHM": FWHM,
        "sigma": sigma,
        "gamma": None,
        "Data": df.assign(S_fit=yfit),
        "R2": _r2(y, yfit)
    }


# =========================
# Voigt + constant baseline
# =========================
def voigt_fit(data, p0=None, bounds=None):
    """
    Fit a Voigt (Gaussian ⊗ Lorentzian) profile with a constant baseline.

    Parameters
    ----------
    data : DataFrame or ndarray with columns 'B','Ab'
    p0 : dict | None
        Optional initial guess with keys {'amplitude','center','sigma','gamma','offset'}.
        If None, seeds are estimated automatically.
    bounds : dict | None
        Optional bounds for the same keys.

    Returns
    -------
    dict with keys:
        out, S_fit, Params, FWHM (Olivero–Longbothum), sigma, gamma, Data, R2
    """
    if isinstance(data, np.ndarray):
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("numpy array must be N×2 (B, Ab)")
        df = pd.DataFrame(data[:, :2], columns=['B','Ab'])
    else:
        df = data.copy()
        if 'B' not in df.columns or 'Ab' not in df.columns:
            if df.shape[1] >= 2:
                df = df.copy()
                df.columns = ['B', 'Ab'] + list(df.columns[2:])
            else:
                raise ValueError("data must contain columns B and Ab")
    x, y = df['B'].values, df['Ab'].values

    vmod = VoigtModel(prefix='v_')
    cmod = ConstantModel(prefix='c_')
    model = vmod + cmod

    # Auto seeds: remove linear baseline, refine center; let vmod.guess help shape
    c0_lin, c1_lin = _baseline_linear_edges(x, y, frac=0.15)
    y_corr = y - (c0_lin + c1_lin*x)
    idx = np.argmax(np.abs(y_corr))
    center0 = _refine_center_quadratic(x, y_corr, idx, win=3)

    c0_const = _edge_baseline(y)  # initial constant offset for the ConstantModel
    try:
        vparams = vmod.guess(y - c0_const, x=x)
    except Exception:
        span = (x.max()-x.min()) if x.max()>x.min() else 1.0
        vparams = vmod.make_params(
            v_amplitude=(y[idx]-c0_const)*np.sqrt(2*np.pi)*(span/10),
            v_center=center0, v_sigma=span/20, v_gamma=span/20
        )

    pars = model.make_params()
    # Important: do NOT abs() amplitude/center; only enforce positivity on sigma/gamma
    pars['v_amplitude'].set(value=vparams['v_amplitude'].value)
    pars['v_center'   ].set(value=center0)
    pars['v_sigma'    ].set(value=max(1e-12, abs(vparams['v_sigma' ].value)))
    pars['v_gamma'    ].set(value=max(1e-12, abs(vparams['v_gamma' ].value)))
    pars['c_c'        ].set(value=c0_const)

    # Loose automatic bounds for center if not given
    span = (x.max() - x.min()) if x.max() > x.min() else 1.0
    if not (isinstance(bounds, dict) and 'center' in bounds):
        pars['v_center'].set(min=center0 - 0.25*span, max=center0 + 0.25*span)

    # Manual p0 override
    if isinstance(p0, dict):
        if 'amplitude' in p0: pars['v_amplitude'].set(value=p0['amplitude'])
        if 'center'    in p0: pars['v_center'   ].set(value=p0['center'])
        if 'sigma'     in p0: pars['v_sigma'    ].set(value=max(1e-12, abs(p0['sigma'])))
        if 'gamma'     in p0: pars['v_gamma'    ].set(value=max(1e-12, abs(p0['gamma'])))
        if 'offset'    in p0: pars['c_c'        ].set(value=p0['offset'])

    # Bounds
    if isinstance(bounds, dict):
        if 'amplitude' in bounds:
            lo, hi = bounds['amplitude']; pars['v_amplitude'].set(min=lo, max=hi)
        if 'center' in bounds:
            lo, hi = bounds['center']; pars['v_center'].set(min=lo, max=hi)
        if 'sigma' in bounds:
            lo, hi = bounds['sigma']; pars['v_sigma'].set(min=max(1e-12, lo), max=hi)
        if 'gamma' in bounds:
            lo, hi = bounds['gamma']; pars['v_gamma'].set(min=max(1e-12, lo), max=hi)
        if 'offset' in bounds:
            lo, hi = bounds['offset']; pars['c_c'].set(min=lo, max=hi)

    out = model.fit(y, pars, x=x)
    yfit = out.best_fit

    amp = out.params['v_amplitude'].value
    cen = out.params['v_center'].value
    sig = abs(out.params['v_sigma'].value)
    gam = abs(out.params['v_gamma'].value)
    off = out.params['c_c'].value

    # Voigt FWHM via Olivero–Longbothum approximation
    Gamma = 2.0*gam
    G = 2.0*np.sqrt(2.0*np.log(2.0))*sig
    FWHM = 0.5346*Gamma + np.sqrt(0.2166*Gamma**2 + G**2)

    return {
        "out": out,
        "S_fit": yfit,
        "Params": {"amplitude": amp, "center": cen, "sigma": sig, "gamma": gam, "offset": off},
        "FWHM": FWHM,
        "sigma": sig,
        "gamma": gam,
        "Data": df.assign(S_fit=yfit),
        "R2": _r2(y, yfit)
    }


# =========================
# Asymmetric Lorentz (different widths left/right, single a and c keep continuity at center)
# =========================
def asymmetric_lorentz_fit(data, p0=None, bounds=None):
    """
    Piecewise asymmetric Lorentzian:
      f(B) = a / (1 + ((B-center)/w_left)^2)   for B < center
           + a / (1 + ((B-center)/w_right)^2)  for B >= center  + c

    Parameters
    ----------
    data : DataFrame or ndarray with columns 'B','Ab'
    p0 : dict | None
        Optional seeds {'amplitude','center','w_left','w_right','offset'}.
        If None, estimated automatically (w_left=w_right initially).
    bounds : dict | None
        Optional bounds for the same keys.

    Returns
    -------
    dict with keys:
        out, S_fit, Params (includes FWHM_left/right), FWHM (mean), sigma(None),
        gamma (=FWHM/2), Data, R2, asymmetry.
    """
    if isinstance(data, np.ndarray):
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("numpy array must be N×2 (B, Ab)")
        df = pd.DataFrame(data[:, :2], columns=['B','Ab'])
    else:
        df = data.copy()
        if 'B' not in df.columns or 'Ab' not in df.columns:
            if df.shape[1] >= 2:
                df = df.copy()
                df.columns = ['B', 'Ab'] + list(df.columns[2:])
    x, y = df['B'].values, df['Ab'].values

    def asym_lorentz(B, a, center, w_left, w_right, c):
        w = np.where(B < center, w_left, w_right)
        w = np.clip(np.asarray(w, dtype=float), 1e-12, np.inf)
        return a / (1.0 + ((B - center)/w)**2) + c

    model = Model(asym_lorentz)

    # Auto seeds: remove linear baseline, refine center; start with symmetric widths
    c0_lin, c1_lin = _baseline_linear_edges(x, y, frac=0.15)
    y_corr = y - (c0_lin + c1_lin*x)
    idx = np.argmax(np.abs(y_corr))
    center0 = _refine_center_quadratic(x, y_corr, idx, win=3)
    a0 = float(y_corr[idx])
    fwhm_est = _halfmax_width(x, y_corr, baseline=0.0)
    if not np.isfinite(fwhm_est) or fwhm_est <= 0:
        fwhm_est = (x.max() - x.min())/5.0
    w0 = max(1e-12, fwhm_est/2.0)
    c0 = _edge_baseline(y)

    if p0 is None:
        init = dict(amplitude=a0, center=center0, w_left=w0, w_right=w0, offset=c0)
    elif isinstance(p0, dict):
        init = dict(amplitude=p0.get('amplitude', a0),
                    center=p0.get('center', center0),
                    w_left=max(1e-12, abs(p0.get('w_left', w0))),
                    w_right=max(1e-12, abs(p0.get('w_right', w0))),
                    offset=p0.get('offset', c0))
    else:
        raise ValueError("p0 must be dict or None for asymmetric_lorentz_fit")

    pars = model.make_params(a=init['amplitude'], center=init['center'],
                             w_left=init['w_left'], w_right=init['w_right'], c=init['offset'])

    # Loose automatic bounds for center if not given
    span = (x.max() - x.min()) if x.max() > x.min() else 1.0
    if not (isinstance(bounds, dict) and 'center' in bounds):
        pars['center'].set(min=init['center'] - 0.25*span, max=init['center'] + 0.25*span)

    # Bounds
    if isinstance(bounds, dict):
        if 'amplitude' in bounds:
            lo, hi = bounds['amplitude']; pars['a'].set(min=lo, max=hi)
        if 'center' in bounds:
            lo, hi = bounds['center']; pars['center'].set(min=lo, max=hi)
        if 'offset' in bounds:
            lo, hi = bounds['offset']; pars['c'].set(min=lo, max=hi)
        if 'w_left' in bounds:
            lo, hi = bounds['w_left']; pars['w_left'].set(min=max(1e-12, lo), max=hi)
        if 'w_right' in bounds:
            lo, hi = bounds['w_right']; pars['w_right'].set(min=max(1e-12, lo), max=hi)

    out = model.fit(y, pars, B=x)
    yfit = out.best_fit

    a    = out.params['a'].value
    ctr  = out.params['center'].value
    wl   = abs(out.params['w_left'].value)
    wr   = abs(out.params['w_right'].value)
    c    = out.params['c'].value

    FWHM_L = 2.0*wl
    FWHM_R = 2.0*wr
    FWHM   = 0.5*(FWHM_L + FWHM_R)
    gamma  = FWHM/2.0
    asym   = abs(wr - wl) / max(1e-30, (wr + wl))

    return {
        "out": out,
        "S_fit": yfit,
        "Params": {"amplitude": a, "center": ctr, "w_left": wl, "w_right": wr, "gamma": gamma, "offset": c,
                   "FWHM_left": FWHM_L, "FWHM_right": FWHM_R},
        "FWHM": FWHM,
        "sigma": None,
        "gamma": gamma,
        "Data": df.assign(S_fit=yfit),
        "R2": _r2(y, yfit),
        "asymmetry": asym
    }



"""
Fitting tools of the OPM signals after lock-in amplifier: Dispersion Lorentz, Linear region
"""

def dispersion_lorentz_fit(
    data, p0=None, bounds=None, max_rel_err=0.01,
    center_hint=None, smooth_win=9, robust_iters=1, mad_thresh=4.0
):
    """
    Dispersive Lorentzian with center shift (lock-in demod line shape).

        f(B) = a * b * (B - center) / (1 + (b*(B - center))^2) + c

    Notes
    -----
    - Valid near resonance (low-field / small-signal regime).
    - Slope at B ≈ center is a*b.
    - Underlying Lorentz HWHM (gamma) is 1/|b|, hence FWHM = 2/|b|.

    Parameters
    ----------
    data : DataFrame or ndarray
        Accepts columns:
          - ('x','y') or
          - ('B','S') or
          - ('Ab','demod')
        Internally mapped to 'x' (field) and 'y' (signal).
    p0 : None | tuple/list | dict, optional
        Initial guess.
        - None: auto-estimated.
        - tuple/list:
            * legacy (a, b, c)  -> center=0 assumed
            * new    (a, b, center, c)
        - dict: {'amplitude', 'b'|'gamma'|'FWHM', 'center', 'offset'}
    bounds : dict | None, optional
        Optional bounds for the same keys as above.
    max_rel_err : float, optional
        Small-signal criterion |(b*(B-center))^2| < max_rel_err used to report LinearRange.
    center_hint : float or None, optional
        If given, the zero-crossing closest to this value is preferred as the seed center.
    smooth_win : int, optional
        Moving-average window (samples) used only for seed estimation. Disable with <=1.
    robust_iters : int, optional
        Number of robust re-fits with MAD outlier rejection (0/1/2 ...).
    mad_thresh : float, optional
        Threshold in units of robust sigma (1.4826*MAD) for outlier rejection.

    Returns
    -------
    dict
        {
          'out'        : lmfit ModelResult,
          'S_fit'      : fitted y on the input grid,
          'Params'     : {'amplitude','center','b','gamma','offset','slope'},
          'FWHM'       : 2/|b|,
          'sigma'      : None,
          'gamma'      : 1/|b|,
          'Data'       : input DataFrame with S_fit column,
          'R2'         : coefficient of determination,
          'Slope'      : a*b  (first-order slope at center),
          'LinearRange': (Bmin, Bmax) for the given max_rel_err,
          'FitResult'  : callable f(B)
        }
    """
    # ---------- normalize input ----------
    if isinstance(data, np.ndarray):
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("numpy array must be N×2")
        df = pd.DataFrame(data[:, :2], columns=['x', 'y'])
    else:
        df = data.copy()
        cols = set(df.columns)
        if {'x','y'}.issubset(cols):
            df = df.rename(columns={'x':'x','y':'y'})
        elif {'B','S'}.issubset(cols):
            df = df.rename(columns={'B':'x','S':'y'})
        elif {'Ab','demod'}.issubset(cols):
            df = df.rename(columns={'Ab':'x','demod':'y'})
        else:
            if df.shape[1] >= 2:
                df = df.iloc[:, :2]; df.columns = ['x','y']
            else:
                raise ValueError("data must contain ('x','y') or ('B','S') or ('Ab','demod')")

    x_full = df['x'].to_numpy(float)
    y_full = df['y'].to_numpy(float)

    # ---------- model ----------
    def f(B, a, b, center, c):
        u = b*(B - center)
        return a * b * (B - center) / (1.0 + u*u) + c
    model = Model(f)

    # ---------- helpers ----------
    def _edge_baseline(y, frac=0.1):
        n = max(1, int(len(y)*frac))
        return float(np.median(np.r_[y[:n], y[-n:]]))

    def _baseline_linear_edges(x, y, frac=0.15):
        n = max(3, int(len(x)*frac))
        X = np.r_[x[:n], x[-n:]]
        Y = np.r_[y[:n], y[-n:]]
        if len(X) < 3:
            return float(np.median(Y)), 0.0
        b1, b0 = np.polyfit(X, Y, 1)  # y = b1*x + b0
        return float(b0), float(b1)

    def _movavg(v, w):
        if w is None or w <= 1: return v
        k = np.ones(int(w))/int(w)
        return np.convolve(v, k, mode='same')

    # ---------- seeds on a deduped, increasing, smoothed grid ----------
    c0_lin, c1_lin = _baseline_linear_edges(x_full, y_full, frac=0.15)
    y_lin = y_full - (c0_lin + c1_lin*x_full)

    # unique & increasing for seed estimation
    xu_all, iu_all = np.unique(x_full, return_index=True)
    yu_all = y_lin[iu_all]
    yu_all = _movavg(yu_all, smooth_win)

    # ignore edges (turn-around / noise)
    M = len(xu_all)
    if M >= 20:
        i0 = int(0.05*M); i1 = int(0.95*M)
    else:
        i0, i1 = 0, M
    xu = xu_all[i0:i1]; yu = yu_all[i0:i1]
    if len(xu) < 4 or np.ptp(xu) <= 0:
        xu, yu = xu_all, yu_all

    # slope by finite difference (no gradient)
    dx = np.diff(xu)
    valid = dx > 0
    if not np.all(valid):
        mask = np.r_[True, valid]
        xu = xu[mask]; yu = yu[mask]; dx = np.diff(xu)
    dydx = np.diff(yu) / np.where(np.abs(dx) < 1e-30, 1e-30, dx)

    # zero crossings on smoothed signal
    sgn = np.sign(yu)
    zc = np.where(sgn[:-1]*sgn[1:] <= 0)[0]  # crossing in [k,k+1]

    # pick center0
    if len(zc):
        if center_hint is not None and np.isfinite(center_hint):
            # crossing closest to hint
            zc_pos = xu[zc] - yu[zc]*(xu[zc+1]-xu[zc])/(yu[zc+1]-yu[zc] + 1e-15)
            center0 = float(zc_pos[np.argmin(np.abs(zc_pos - center_hint))])
        else:
            # crossing nearest to the location of max |dS/dB|
            i_dmax = int(np.argmax(np.abs(dydx)))
            k = int(zc[np.argmin(np.abs(zc - i_dmax))])
            x1,x2,y1,y2 = xu[k], xu[k+1], yu[k], yu[k+1]
            center0 = x1 + (-y1)*(x2 - x1)/(y2 - y1 + 1e-15)
    else:
        center0 = float(xu[np.argmin(np.abs(yu))])

    # extrema around center0
    sgd = np.sign(dydx)
    tp = np.where(sgd[:-1]*sgd[1:] <= 0)[0]
    tp_x = (xu[tp] + xu[tp+1]) * 0.5 if len(tp) else np.array([])
    if len(tp_x):
        left_mask  = tp_x <  center0
        right_mask = tp_x >= center0
        if np.any(left_mask) and np.any(right_mask):
            xl = tp_x[left_mask][-1]
            xr = tp_x[right_mask][0]
            sep = abs(xr - xl)                    # ~ 2/gamma
            gamma0 = max(1e-6, sep/2.0)
            b0 = 1.0 / gamma0
            il = int(np.argmin(np.abs(xu - xl)))
            ir = int(np.argmin(np.abs(xu - xr)))
            a0 = 2.0 * max(abs(yu[il]), abs(yu[ir]))  # |y_ext| ~ |a|/2
        else:
            span = max(1e-9, xu.max() - xu.min())
            b0 = 4.0 / span
            a0 = 2.0 * np.max(np.abs(yu))
    else:
        span = max(1e-9, xu.max() - xu.min())
        b0 = 4.0 / span
        a0 = 2.0 * np.max(np.abs(yu))

    c0 = _edge_baseline(y_full)

    # ---------- p0 handling ----------
    if p0 is None:
        seeds = dict(amplitude=a0, b=b0, center=center0, offset=c0)
    elif isinstance(p0, (list, tuple)):
        if len(p0) == 3:
            seeds = dict(amplitude=p0[0], b=p0[1], center=0.0, offset=p0[2])
        elif len(p0) == 4:
            seeds = dict(amplitude=p0[0], b=p0[1], center=p0[2], offset=p0[3])
        else:
            raise ValueError("p0 must be (a,b,c) or (a,b,center,c)")
    elif isinstance(p0, dict):
        seeds = dict(amplitude=p0.get('amplitude', a0),
                     b=p0.get('b', b0),
                     center=p0.get('center', center0),
                     offset=p0.get('offset', c0))
        if 'gamma' in p0 and p0['gamma'] not in (None, np.nan):
            seeds['b'] = 1.0 / max(1e-12, abs(p0['gamma']))
        if 'FWHM' in p0 and p0['FWHM'] not in (None, np.nan):
            seeds['b'] = 2.0 / max(1e-12, abs(p0['FWHM']))
    else:
        raise ValueError("p0 must be None, tuple/list or dict")

    pars = model.make_params(a=seeds['amplitude'],
                             b=seeds['b'],
                             center=seeds['center'],
                             c=seeds['offset'])

    # ---------- bounds ----------
    if isinstance(bounds, dict):
        if 'amplitude' in bounds:
            lo, hi = bounds['amplitude']; pars['a'].set(min=lo, max=hi)
        if 'center' in bounds:
            lo, hi = bounds['center'];   pars['center'].set(min=lo, max=hi)
        if 'offset' in bounds:
            lo, hi = bounds['offset'];   pars['c'].set(min=lo, max=hi)
        if 'b' in bounds:
            lo, hi = bounds['b'];        pars['b'].set(min=lo, max=hi)
        else:
            b_lo_mag, b_hi_mag = 0.0, np.inf
            if 'gamma' in bounds:
                glo, ghi = bounds['gamma']
                if np.isfinite(ghi) and ghi > 0: b_lo_mag = max(b_lo_mag, 1.0/ghi)
                if np.isfinite(glo) and glo > 0: b_hi_mag = min(b_hi_mag, 1.0/glo)
            if 'FWHM' in bounds:
                flo, fhi = bounds['FWHM']
                if np.isfinite(fhi) and fhi > 0: b_lo_mag = max(b_lo_mag, 2.0/fhi)
                if np.isfinite(flo) and flo > 0: b_hi_mag = min(b_hi_mag, 2.0/flo)
            if np.isfinite(b_hi_mag):
                pars['b'].set(min=-b_hi_mag, max=b_hi_mag)

    # ---------- robust fit (optional) ----------
    mask = np.ones_like(x_full, dtype=bool)
    out = None
    for it in range(max(1, int(robust_iters)+1)):
        out = model.fit(y_full[mask], pars, B=x_full[mask])
        a  = out.params['a'].value
        b  = out.params['b'].value
        c0p= out.params['c'].value
        ctr= out.params['center'].value

        if it >= robust_iters:
            break
        # MAD-based outlier rejection on residuals of current fit
        y_pred = f(x_full[mask], a, b, ctr, c0p)
        resid  = y_full[mask] - y_pred
        med    = np.median(resid)
        mad    = np.median(np.abs(resid - med))
        sigma  = 1.4826 * mad if mad > 0 else np.std(resid)
        if sigma <= 0:
            break
        keep = np.abs(resid - med) <= mad_thresh * sigma
        # rebuild mask on the full grid
        new_mask = np.zeros_like(mask, dtype=bool)
        new_mask[np.where(mask)[0][keep]] = True
        # if almost nothing changes, stop
        if new_mask.sum() < max(12, 0.2*len(x_full)) or np.array_equal(new_mask, mask):
            break
        mask = new_mask

    # predict on full grid to fill Data['S_fit'] with correct length
    a      = out.params['a'].value
    b      = out.params['b'].value
    center = out.params['center'].value
    coff   = out.params['c'].value
    yfit_full = f(x_full, a, b, center, coff)

    FWHM  = 2.0/abs(b) if b != 0 else np.inf
    gamma = 1.0/abs(b) if b != 0 else np.inf
    slope = a * b
    B_half = np.sqrt(max_rel_err) / max(1e-12, abs(b))
    linear_range = (center - B_half, center + B_half)

    ss_res = float(np.sum((y_full - yfit_full)**2))
    ss_tot = float(np.sum((y_full - np.mean(y_full))**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan

    return {
        "out": out,
        "S_fit": yfit_full,
        "Params": {
            "amplitude": a,
            "center": center,
            "b": b,
            "gamma": gamma,
            "offset": coff,
            "slope": slope
        },
        "FWHM": FWHM,
        "sigma": None,
        "gamma": gamma,
        "Data": df.assign(S_fit=yfit_full),
        "R2": r2,
        "Slope": slope,
        "LinearRange": linear_range,
        "FitResult": (lambda B: f(B, a, b, center, coff)),
    }

