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
    Refine the peak center by fitting a local quadratic (2*win+1 points)
    around the extremum on baseline-corrected data y_corr.
    Returns the vertex location if it lies inside the local window.
    """
    i0 = max(0, guess_idx - win)
    i1 = min(len(x), guess_idx + win + 1)
    X = x[i0:i1]; Y = y_corr[i0:i1]
    if len(X) < 3:
        return float(x[guess_idx])
    a, b, c = np.polyfit(X, Y, 2)
    if abs(a) < 1e-20:
        return float(x[guess_idx])
    xc = -b/(2*a)
    if xc < X.min() or xc > X.max():
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

def dispersion_lorentz_fit(data, p0=None, bounds=None, max_rel_err=0.01):
    """
    Dispersive Lorentzian with center shift (lock-in demod line shape).

      f(B) = a * b * (B - center) / (1 + (b*(B - center))^2) + c

    Notes
    -----
    - Valid in the low-field / small-signal regime near B ≈ center.
    - Slope at B ≈ center is a*b.
    - The underlying Lorentz HWHM (gamma) is 1/|b|, so FWHM = 2/|b|.

    Parameters
    ----------
    data : DataFrame or ndarray
        Accepts columns:
          - ('x','y') or
          - ('B','S') or
          - ('Ab','demod')
        The function internally maps to columns 'x' (field) and 'y' (signal).
    p0 : None | tuple/list | dict
        Initial guess (optional).
        - None: auto-estimated.
        - tuple/list:
            * legacy (a, b, c)  -> center=0 assumed
            * new    (a, b, center, c)
        - dict: {'amplitude','b'|'gamma'|'FWHM','center','offset'}
    bounds : dict | None
        Optional bounds for the same dict keys above.
    max_rel_err : float
        |(b*(B-center))^2| < max_rel_err defines a suggested linear range.

    Returns
    -------
    dict with keys (aligned with other fitters):
        out        : lmfit ModelResult
        S_fit      : best-fit y
        Params     : {'amplitude','center','b','gamma','offset','slope'}
        FWHM       : 2/|b|
        sigma      : None
        gamma      : 1/|b|
        Data       : DataFrame with columns x,y,S_fit
        R2         : coefficient of determination
        Slope      : a*b at B≈center
        LinearRange: (Bmin, Bmax) where |(b*(B-center))^2| < max_rel_err
    """
    # -------- normalize input columns -> 'x','y' --------
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
            # try fallback to first two columns
            if df.shape[1] >= 2:
                df = df.iloc[:, :2]
                df.columns = ['x','y']
            else:
                raise ValueError("data must contain ('x','y') or ('B','S') or ('Ab','demod')")
    x = df['x'].values
    y = df['y'].values

    # -------- model --------
    def f(B, a, b, center, c):
        u = b*(B - center)
        return a * b * (B - center) / (1.0 + u*u) + c

    model = Model(f)

    # -------- auto seeds --------
    # remove linear baseline for robust center/width/amplitude estimation
    c0_lin, c1_lin = _baseline_linear_edges(x, y, frac=0.15)
    y_lin = y - (c0_lin + c1_lin*x)

    # 1) zero-crossing as initial center
    sign = np.sign(y_lin)
    cross_idx = np.where(sign[:-1]*sign[1:] <= 0)[0]
    if len(cross_idx) > 0:
        # choose crossing with largest local slope
        dy = np.gradient(y_lin, x)
        cand = []
        for i in cross_idx:
            # linear interpolation zero
            xz = x[i] - y_lin[i]*(x[i+1]-x[i])/(y_lin[i+1]-y_lin[i] + 1e-30)
            slope_local = abs(dy[i] if abs(dy[i]) > abs(dy[min(i+1, len(dy)-1)]) else dy[min(i+1, len(dy)-1)])
            cand.append((slope_local, xz))
        center0 = float(sorted(cand, key=lambda t: -abs(t[0]))[0][1])
    else:
        # fallback: minimum |y_lin|
        center0 = float(x[np.argmin(np.abs(y_lin))])

    # 2) extrema to estimate width b and amplitude a
    i_max = int(np.argmax(y_lin))
    i_min = int(np.argmin(y_lin))
    x_max, x_min = float(x[i_max]), float(x[i_min])
    # distance from center to the two extrema (take average magnitude)
    dL = abs(x_min - center0)
    dR = abs(x_max - center0)
    d = np.nanmean([dL, dR]) if np.isfinite(dL) and np.isfinite(dR) else max(dL, dR)
    if not np.isfinite(d) or d <= 0:
        span = (x.max()-x.min()) if x.max()>x.min() else 1.0
        d = span/10.0
    b0 = 1.0/max(1e-12, d)  # for f(x) = a*b*x/(1+(b*x)^2), extrema at |x|=1/|b|
    # at extrema, |f| ≈ |a|/2  (after linear baseline removal)
    a0 = 2.0*max(abs(y_lin[i_max]), abs(y_lin[i_min]))
    # constant offset from edges (keep constant baseline in the model)
    c0 = _edge_baseline(y)

    # -------- p0 handling --------
    if p0 is None:
        seeds = dict(amplitude=a0, b=b0, center=center0, offset=c0)
    elif isinstance(p0, (list, tuple)):
        if len(p0) == 3:  # legacy (a, b, c)
            seeds = dict(amplitude=p0[0], b=p0[1], center=0.0, offset=p0[2])
        elif len(p0) == 4:  # (a, b, center, c)
            seeds = dict(amplitude=p0[0], b=p0[1], center=p0[2], offset=p0[3])
        else:
            raise ValueError("p0 must be (a,b,c) or (a,b,center,c)")
    elif isinstance(p0, dict):
        seeds = dict(amplitude=p0.get('amplitude', a0),
                     b=p0.get('b', b0),
                     center=p0.get('center', center0),
                     offset=p0.get('offset', c0))
        # allow gamma/FWHM instead of b
        if 'gamma' in p0 and p0['gamma'] not in (None, np.nan):
            seeds['b'] = 1.0/max(1e-12, abs(p0['gamma']))
        if 'FWHM' in p0 and p0['FWHM'] not in (None, np.nan):
            seeds['b'] = 2.0/max(1e-12, abs(p0['FWHM']))
    else:
        raise ValueError("p0 must be None, tuple/list or dict")

    pars = model.make_params(a=seeds['amplitude'],
                             b=seeds['b'],
                             center=seeds['center'],
                             c=seeds['offset'])

    # -------- bounds (dict style like other fitters) --------
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
            # infer symmetric bounds on b from gamma/FWHM if provided
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

    # -------- fit --------
    out = model.fit(y, pars, B=x)
    yfit = out.best_fit

    a     = out.params['a'].value
    b     = out.params['b'].value
    center= out.params['center'].value
    c     = out.params['c'].value

    # width proxies consistent with other models
    FWHM  = 2.0/abs(b) if b != 0 else np.inf
    gamma = 1.0/abs(b) if b != 0 else np.inf  # HWHM of underlying Lorentz

    # slope and linear range near center
    slope = a * b
    B_half = np.sqrt(max_rel_err) / max(1e-12, abs(b))
    linear_range = (center - B_half, center + B_half)

    # R^2
    r2 = _r2(y, yfit)

    return {
        "out": out,
        "S_fit": yfit,
        "Params": {
            "amplitude": a,
            "center": center,
            "b": b,
            "gamma": gamma,
            "offset": c,
            "slope": slope
        },
        "FWHM": FWHM,
        "gamma": gamma,
        "Data": df.assign(S_fit=yfit),
        "R2": r2,
        "Slope": slope,
        "LinearRange": linear_range
    }



def linear_region_fit(data, win=10, r2min=0.95):
    """
    Automatically find the strongest linear region in B vs. S data using sliding window and R² threshold.

    Parameters:
        data : DataFrame or ndarray
            Must include columns 'B' (x-axis) and 'S' (y-axis signal, e.g., demodulated response)

        win : int, default=10
            Initial window size to compute seed slopes

        r2min : float, default=0.95
            Minimum R² to allow linear region expansion

    Returns:
        result : dict {
            'Region'   : DataFrame containing the linear subset of data,
            'Slope'    : slope of best-fit line in the linear region,
            'R2'       : coefficient of determination for that region,
            'FitResult': callable linear function: f(x) = slope * x + intercept
        }
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=['B', 'S'])
    else:
        data = data.rename(columns=lambda c: {'B': 'B', 'demod': 'S'}.get(c, c))
    data = data.sort_values('B').reset_index(drop=True)

    B, S = data['B'].values, data['S'].values
    n = len(B)

    slopes = []
    for i in range(n - win + 1):
        seg = slice(i, i + win)
        res = linregress(B[seg], S[seg])
        slopes.append(res.slope)
    slopes = np.array(slopes)

    best = np.argmax(np.abs(slopes))
    left, right = best, best + win - 1

    while left > 0:
        res = linregress(B[left - 1 : right + 1], S[left - 1 : right + 1])
        if res.rvalue ** 2 < r2min:
            break
        left -= 1

    while right < n - 1:
        res = linregress(B[left : right + 2], S[left : right + 2])
        if res.rvalue ** 2 < r2min:
            break
        right += 1

    res = linregress(B[left : right + 1], S[left : right + 1])
    region = data.iloc[left : right + 1]

    return {
        "Region": region,
        "Slope": res.slope,
        "R2": res.rvalue ** 2,
        "FitResult": lambda x: res.intercept + res.slope * x
    }

