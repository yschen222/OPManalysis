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
from lmfit.models import GaussianModel, ConstantModel

"""
Fitting tools of absorption OPM signals: Lorentz, Gaussian
"""

def lorentz_fit(data, p0=(1, 0.1, 0)):
    """
    Fitting Lorentzian Absorption Form: f(B) = a / (1 + (bB)^2)+c

    Parameters:
        data: DataFrame or ndarray
              Must include 'B' (magnetic field, x-axis) and 'Ab' (absorption signal, y-axis)
        p0: tuple, defalt=(1, 0.1, 0)
            Initial guess for parameters (a, b, c)

    Returns:
        result: dict {
            'Params'   : (a, b, c), where
                         a: amplitude (peak height)
                         b: width factor (FWHM = 2 / b)
                         c: offset
            'FWHM'     : full width at half maximum of the resonance
            'R2'       : coefficient of determination for goodness-of-fit
            'FitResult': callable function f(B)
            'Data'     : DataFrame with original and fitted values (columns: B, Ab, S_fit)
        }
    """
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data, columns=['B', 'Ab'])
    else:
        df = data.rename(columns=lambda c: {'B': 'B', 'Ab': 'Ab'}.get(c, c))

    B = df['B'].values
    Ab = df['Ab'].values
    
    def lorentz_absorption(B, a, b, c):
        return a / (1 + (b * B)**2)+c

    # fitting
    popt, pcov = curve_fit(lorentz_absorption, B, Ab, p0=p0)
    a, b, c = popt
    S_fit = lorentz_absorption(B, a, b, c)

    # R²
    ss_res = np.sum((Ab - S_fit)**2)
    ss_tot = np.sum((Ab - np.mean(Ab))**2)
    r2 = 1 - ss_res / ss_tot

    return {
        "Params": (a, b, c),
        "FWHM": 2 / b,
        "R2": r2,
        "FitResult": lambda B: lorentz_absorption(B, a, b, c),
        "Data": df.assign(S_fit=S_fit)
    }



def gauss_fit(data):
    """
    Fitting Gaussian Absorption Form：f(B) = A exp(-(B - mu)^2 / (2 sigma^2)) + c

    Parameters:
        data: DataFrame or ndarray，including 'B'(x-axis) and 'Absorption'(y-axis)

    Returns:
        result: dict {
            'FitResult': lmfit ModelResult,
            'Params'   : dict of fitted values (amplitude, center, sigma, offset),
            'Sigma'    : sigma of Gaussian,
            'FWHM'     : full width at half max,
            'R2'       : coefficient of determination,
            'Data'     : DataFrame with fitted values
        }
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=['B', 'Ab'])
    else:
        data = data.rename(columns=lambda c: {'B': 'B', 'Ab': 'Ab'}.get(c, c))

    x, y = data['B'].values, data['Ab'].values

    gmod = GaussianModel(prefix='g_')
    cmod = ConstantModel(prefix='c_')
    model = gmod + cmod

    params = gmod.guess(y, x=x)
    params.update(cmod.make_params(c_c=y.min()))

    out = model.fit(y, params, x=x)
    y_fit = out.best_fit

    # R² caculation
    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot

    # parameters and results
    sigma = out.params['g_sigma'].value
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    params_dict = {
        "amplitude": out.params['g_amplitude'].value,
        "center": out.params['g_center'].value,
        "sigma": sigma,
        "offset": out.params['c_c'].value
    }

    return {
        "FitResult": out,
        "Params": params_dict,
        "Sigma": sigma,
        "FWHM": fwhm,
        "R2": r2,
        "Data": data.assign(S_fit=y_fit)
    }



"""
Fitting tools of the OPM signals after lock-in amplifier: Dispersion Lorentz, Linear region
"""

def dispersion_lorentz_fit(data, p0=(1, 0.01, 0), bounds=(-np.inf, np.inf), max_rel_err=0.01):
    """
    Fitting function: f(B) = a * b * B / (1 + (b * B)^2) + c

    Common in OPM analysis as an analytical solution in the low-field regime.
    Note: This model is only valid near B=0 where γB ≪ 1/τ.

    Parameters:
        data : DataFrame or ndarray
            Must include columns 'x' (field, e.g., Ab) and 'y' (signal, e.g., demod).
            Accepts column mapping from {'Ab': 'x', 'demod': 'y'} automatically.

        p0 : tuple, default=(1, 0.01, 0)
            Initial guess for parameters (a, b, c)

        bounds : 2-tuple of array_like
            Lower and upper bounds for parameters

        max_rel_err : float, optional
            Maximum tolerated relative error from linearity (default=0.01)

    Returns:
        result : dict {
            'Params'      : (a, b, c),
            'Cov'         : Covariance matrix from curve_fit,
            'R2'          : Coefficient of determination,
            'Slope'       : First-order slope at B≈0 (i.e., a·b),
            'LinearRange' : (Bmin, Bmax) range where |(bB)^2| < max_rel_err,
            'FitResult'   : Callable function f(B),
            'Residuals'   : Residuals y - yfit,
            'Data'        : DataFrame with x, y, and fitted yfit
        }
    """
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data, columns=['x', 'y'])
    else:
        df = data.rename(columns=lambda c: {'Ab': 'x', 'demod': 'y'}.get(c, c))
    x, y = df['x'].values, df['y'].values

    def model(x, a, b, c):
        return a * b * x / (1 + (b * x)**2) + c

    popt, pcov = curve_fit(model, x, y, p0=p0, bounds=bounds)
    a, b, c = popt

    yfit = model(x, a, b, c)
    residuals = y - yfit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot

    # First-order slope and suggested linear range
    slope = a * b
    Bmax = np.sqrt(max_rel_err) / b
    Bmin = -Bmax

    return {
        "Params": (a, b, c),
        "Cov": pcov,
        "R2": r2,
        "Slope": slope,
        "LinearRange": (Bmin, Bmax),
        "FitResult": lambda x: model(x, a, b, c),
        "Residuals": residuals,
        "Data": df.assign(S_fit=yfit)
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
