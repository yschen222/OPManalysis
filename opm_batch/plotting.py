from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from .config import PLOT_MAXPTS, PLOT_DPI, PLOT_FMT

def plot_save_dispersion(sub_dir: Path, B, demod, yfit, slope, r2, lin_range=None, ctr=None, offset=None):
    try:
        B = np.asarray(B); demod = np.asarray(demod); yfit = np.asarray(yfit)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6.0,4.0))
        plt.plot(B, demod, '.', ms=2, color='k', label='Demod (window)')
        if np.all(np.isfinite(yfit)) and len(yfit)==len(B):
            plt.plot(B, yfit, '-', lw=2, color='r', label='Dispersion fit')
        if lin_range and ctr is not None and offset is not None and np.all(np.isfinite(lin_range)):
            lo, hi = lin_range
            if lo < hi:
                plt.axvspan(lo, hi, color='tab:green', alpha=0.15, label='Linear range')
                Bb = np.linspace(lo, hi, 200)
                yl = offset + slope*(Bb - ctr)
                plt.plot(Bb, yl, 'g--', lw=1.8, label='Small-signal approx')
        plt.title(f"Dispersion fit  (R²={r2:.4f}, slope={slope*1e3:.3g} mV/nT)")
        plt.xlabel("Magnetic Field (nT)"); plt.ylabel("Demodulated Signal (V)")
        plt.legend(loc='best', fontsize=9); plt.tight_layout()
        out_png = Path(sub_dir) / "fit_dispersion.png"
        plt.savefig(out_png, dpi=180); plt.close()
    except Exception as e:
        print(f"[WARN] plot dispersion failed: {sub_dir.name}: {e}")

def plot_save_lineshapes(sub_dir: Path, B, Ab, want_lor, lf, want_gau, gf, want_voi, vf):
    """
    Plot lineshape fits (Lorentzian, Gaussian, Voigt) on absorption data.
    
    Parameters
    ----------
    sub_dir : Path
        Output directory for the plot
    B : array-like
        Magnetic field values (nT)
    Ab : array-like
        Absorption signal values (V)
    want_lor : bool
        Whether to plot Lorentzian fit
    lf : dict or None
        Lorentzian fit result
    want_gau : bool
        Whether to plot Gaussian fit
    gf : dict or None
        Gaussian fit result
    want_voi : bool
        Whether to plot Voigt fit
    vf : dict or None
        Voigt fit result (new structure with nested dicts)
    """
    try:
        import numpy as np, matplotlib.pyplot as plt
        from .metrics import first_scalar
        
        x = np.asarray(B); y = np.asarray(Ab)
        if len(x) > PLOT_MAXPTS:
            idx = np.linspace(0, len(x)-1, PLOT_MAXPTS).astype(int)
            x, y = x[idx], y[idx]
        
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(x, y, ".", ms=2, alpha=0.5, label="data", color='gray')
        xx = np.linspace(np.nanmin(B), np.nanmax(B), 1200)
        
        # === Lorentzian fit ===
        if want_lor and lf:
            p = lf.get("Params", {})
            a, b, c0, ctr = p.get("amplitude"), p.get("b"), p.get("offset"), p.get("center")
            if all(v is not None for v in (a, b, c0, ctr)):
                yy = a/(1.0+(b*(xx-ctr))**2) + c0
                r2 = first_scalar(lf.get('R2'))
                ax.plot(xx, yy, ":", lw=2, color='red', label=f"Lorentz (R²={r2:.4f})")
        
        # === Gaussian fit ===
        if want_gau and gf:
            p = gf.get("Params", {})
            A, mu, sig, c0 = p.get("amplitude"), p.get("center"), p.get("sigma"), p.get("offset")
            if all(v is not None for v in (A, mu, sig, c0)):
                yy = (A/(np.sqrt(2*np.pi)*sig))*np.exp(-0.5*((xx-mu)/sig)**2) + c0
                r2 = first_scalar(gf.get('R2'))
                ax.plot(xx, yy, ":", lw=2, color='blue', label=f"Gaussian (R²={r2:.4f})")
        
        # === Voigt fit (NEW STRUCTURE SUPPORT) ===
        if want_voi and vf:
            # Try new nested structure first
            if "Voigt" in vf:
                # New structure: vf["Voigt"]["Params"]
                p = vf["Voigt"].get("Params", {})
                A = p.get("amplitude")
                mu = p.get("center")
                sig = p.get("sigma")
                gam = p.get("gamma")
                c0 = p.get("offset")
                r2 = first_scalar(vf["Voigt"].get('R2'))
                model_selected = vf.get("ModelSelected", "")
                
                # Add model selection info to label if available
                if model_selected:
                    label_suffix = f" [Best: {model_selected}]" if model_selected == "Voigt" else ""
                else:
                    label_suffix = ""
                    
            else:
                # Fallback to old flat structure
                p = vf.get("Params", {})
                A = p.get("amplitude")
                mu = p.get("center")
                sig = p.get("sigma")
                gam = p.get("gamma")
                c0 = p.get("offset")
                r2 = first_scalar(vf.get('R2'))
                label_suffix = ""
            
            # Plot Voigt if all parameters are available
            if all(v is not None for v in (A, mu, sig, gam, c0)):
                from scipy.special import wofz
                def voigt_area(xx, sigma, gamma):
                    sigma = max(1e-12, sigma)
                    gamma = max(1e-12, gamma)
                    z = (xx + 1j*gamma) / (sigma*np.sqrt(2))
                    return np.real(wofz(z)) / (sigma*np.sqrt(2*np.pi))
                
                yy = A*voigt_area(xx-mu, sig, gam) + c0
                ax.plot(xx, yy, "-", lw=2.5, color='lime', 
                       label=f"Voigt (R²={r2:.4f}){label_suffix}")
        
        ax.set_xlabel("B (nT)", fontsize=11)
        ax.set_ylabel("Absorption (V)", fontsize=11)
        ax.set_title("Lineshape fits", fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        out = sub_dir / f"fit_lineshape.{PLOT_FMT}"
        fig.tight_layout()
        fig.savefig(out, dpi=PLOT_DPI)
        plt.close(fig)
        
    except Exception as e:
        print(f"[WARN] plot lineshape failed: {sub_dir.name}: {e}")

def plot_save_psd(noise_dir: Path, noise_df_std: pd.DataFrame) -> None:
    """
    Recompute PSD from standardized noise dataframe and save a figure to noise_dir/noise_psd.{fmt}.
    The mean ASD in the band is shown in the title (V/√Hz).
    """
    try:
        from .config import NOISE_MODE, NOISE_BAND, NOISE_WINDOW, NOISE_AVERAGE
        from .config import NOISE_LOCK_DF_HZ, NOISE_DETREND, NOISE_ENABLE_PREPROCESS
        from .metrics import noise_psd_lowband, noise_psd  # Assuming these are imported
        
        noise_dir.mkdir(parents=True, exist_ok=True)

        # --- Recompute PSD (consistent with compute_noise settings) ---
        kwargs = dict(
            band=NOISE_BAND,
            window=NOISE_WINDOW,
            average=NOISE_AVERAGE,
            min_freq_resolution=NOISE_LOCK_DF_HZ,
            force_long_segments=False,
            detrend=NOISE_DETREND,
            snap_to_power_of_2=False
        )

        if NOISE_MODE == "lowband":
            pts, asd_mean, fs_used, info = noise_psd_lowband(
                noise_df_std,
                enable_auto_preprocess=NOISE_ENABLE_PREPROCESS,
                **kwargs
            )
        else:
            pts, asd_mean, fs_used, info = noise_psd(
                noise_df_std,
                **kwargs
            )

        f = np.asarray(pts[:, 0])
        Pxx = np.asarray(pts[:, 1])
        asd = np.sqrt(np.maximum(Pxx, 0))

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(f, asd, "-", lw=1.5, label="ASD (V/√Hz)")

        f1, f2 = NOISE_BAND
        ax.axvspan(f1, f2, alpha=0.15, label=f"band {f1:g}–{f2:g} Hz")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("ASD (V/√Hz)")
        ax.set_title(f"Noise PSD (mean={asd_mean:.3g} V/√Hz)")
        ax.legend(loc="best")

        out = noise_dir / f"noise_psd.{PLOT_FMT}"
        fig.tight_layout()
        fig.savefig(out, dpi=PLOT_DPI)
        plt.close(fig)
        
    except Exception as e:
        print(f"[WARN] PSD plot failed ({noise_dir.name}): {e}")