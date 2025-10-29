from __future__ import annotations
from pathlib import Path
import warnings, numpy as np, pandas as pd
from typing import Dict
from opmtool import dispersion_lorentz_fit, voigt_fit, lorentz_fit, gauss_fit, one_cycle_cut
from .config import *
from .metrics import force_include_r2_columns, first_scalar, compute_noise, extract_sweep_value
from .io_utils import (
    read_csv_safely, pick_scan_csv, pick_noise_csv,
    standardize_scan_df, standardize_noise_df, load_experiment_rules, choose_rule_for_label
)
from .layout import detect_layout
from .reporting import write_outputs
from .plotting import plot_save_dispersion, plot_save_lineshapes

# expand metric selection (auto include R²)
_SELECT = force_include_r2_columns(set(SELECT_METRICS))

def _want(name: str) -> bool:
    return name.lower() in _SELECT

def process_session(session_dir: Path, rules_from_txt: list, root_dir: Path) -> pd.DataFrame:
    rows = []
    subdirs = [p for p in sorted(session_dir.iterdir()) if p.is_dir() and p.name != NOISE_SUBDIR]
    if not subdirs:
        subdirs = [session_dir]  # treat itself as one dataset

    for sub in subdirs:
        # choose B-range rule
        B_range_used, _ = choose_rule_for_label(rules_from_txt, sub.name)
        if not B_range_used:
            B_range_used = B_RANGE_NT

        scan_csv = pick_scan_csv(sub, root=session_dir)
        if scan_csv is None:
            continue

        try:
            df_raw = read_csv_safely(scan_csv)
            df_scan = standardize_scan_df(df_raw)
        except Exception as e:
            print(f"[WARN] {session_dir.name}/{sub.name}: scan parse failed: {e}")
            continue

        # map to B and cut one segment
        try:
            BField, AbCut, DemodCut = one_cycle_cut(
                df_scan.rename(columns={"tri":"tri"}),
                B_range=B_range_used, segment=LINE_SEGMENT, direction=LINE_DIRECTION,
                time_col="time", ab_col="Ab", demod_col="demod", tri_col="tri",
                map_mode='global', smooth_win=5
            )
            df_bs = pd.DataFrame({"B": BField, "Ab": AbCut, "demod": DemodCut})
        except Exception as e:
            print(f"[WARN] {session_dir.name}/{sub.name}: one_cycle_cut failed: {e}")
            df_bs = pd.DataFrame({"Ab": df_scan["Ab"], "demod": df_scan["demod"]})

        need_slope = _want("slope") or _want("sloper2") or _want("sensitivity")
        need_noise = _want("noisepsd") or _want("sensitivity")
        need_voigt = any(_want(k) for k in ("voigtfwhm","voigtgamma","voigtsigma","voigtr2"))
        need_lor   = any(_want(k) for k in ("lorentzfwhm","lorentzr2"))
        need_gau   = any(_want(k) for k in ("gaussianfwhm","gaussianr2"))

        slope_val = np.nan; slope_R2 = np.nan; disp = None
        if need_slope:
            try:
                if "B" in df_bs.columns:
                    B_all = df_bs["B"].to_numpy(float)
                    Y_all = df_bs["demod"].to_numpy(float)
                    Bspan = float(B_all.max() - B_all.min())
                    half  = max(1e-9, Bspan * (WIDTHRATIO/100.0))
                    m = (B_all >= (DISP_CENTER - half)) & (B_all <= (DISP_CENTER + half))
                    if m.sum() < 12:
                        for s in (1.5, 2.0, 3.0):
                            mm = (B_all >= (DISP_CENTER - half*s)) & (B_all <= (DISP_CENTER + half*s))
                            if mm.sum() >= 12:
                                m = mm; break
                        else:
                            m = np.ones_like(B_all, dtype=bool)
                    Bf, Yf = B_all[m], Y_all[m]
                    df_focus = pd.DataFrame({"x": Bf, "y": Yf})
                    disp = dispersion_lorentz_fit(df_focus, p0={'center': DISP_CENTER, 'amplitude': DISP_P0_AMPL}, max_rel_err=0.01)
                    slope_val = float(disp["Slope"]); slope_R2 = float(disp["R2"])
                    if PLOT_ENABLED and np.isfinite(slope_val):
                        try:
                            lin_lo, lin_hi = disp.get("LinearRange", (None, None))
                            ctr   = disp["Params"]["center"]; c0 = disp["Params"]["offset"]
                            yfit  = disp["Data"]["S_fit"]
                            plot_save_dispersion(sub, Bf, Yf, yfit, slope_val, slope_R2, (lin_lo, lin_hi), ctr, c0)
                        except Exception as e:
                            print(f"[WARN] slope plot failed: {sub.name}: {e}")
                else:
                    df_disp = df_bs.rename(columns={"Ab": "x", "demod": "y"})[["x", "y"]].dropna()
                    disp = dispersion_lorentz_fit(df_disp, p0={'center': 0.0, 'amplitude': DISP_P0_AMPL}, max_rel_err=0.01)
                    slope_val = float(disp["Slope"]); slope_R2  = float(disp["R2"])
            except Exception as e:
                print(f"[WARN] {sub.name}: dispersion fit failed: {e}")

        lorentz_FWHM = gaussian_FWHM = voigt_FWHM = np.nan
        lorentz_R2 = gaussian_R2 = vR2 = np.nan
        v_sigma = v_gamma = np.nan

        import warnings as _w
        _w.filterwarnings("ignore", message=r"Using UFloat objects with std_dev==0.*", category=UserWarning, module=r"uncertainties\.core")

        lf = gf = vf = None
        if "B" in df_bs.columns:
            # === Lorentzian fit ===
            if need_lor:
                try:
                    lf = lorentz_fit(df_bs[["B","Ab"]])
                    lorentz_FWHM = float(lf["FWHM"]) if "FWHM" in lf else 2.0*abs(float(lf["gamma"]))
                    lorentz_R2   = first_scalar(lf.get("R2"))
                except Exception as e:
                    print(f"[WARN] {sub.name}: Lorentz fit failed: {e}")
            
            # === Gaussian fit ===
            if need_gau:
                try:
                    gf = gauss_fit(df_bs[["B","Ab"]])
                    gaussian_FWHM = float(gf["FWHM"]) if "FWHM" in gf else 2.0*np.sqrt(2.0*np.log(2.0))*abs(float(gf["sigma"]))
                    gaussian_R2   = first_scalar(gf.get("R2"))
                except Exception as e:
                    print(f"[WARN] {sub.name}: Gaussian fit failed: {e}")
            
            # === Voigt fit (NEW VERSION) ===
            if need_voigt:
                try:
                    vf = voigt_fit(df_bs[["B","Ab"]])
                    
                    # Extract Voigt-specific results
                    # The new voigt_fit returns nested structure with Voigt/Gaussian/Lorentzian sub-dicts
                    if "Voigt" in vf:
                        # Use the Voigt model's parameters
                        voigt_params = vf["Voigt"]["Params"]
                        v_sigma = float(voigt_params.get("sigma", np.nan))
                        v_gamma = float(voigt_params.get("gamma", np.nan))
                        voigt_FWHM = float(vf["Voigt"].get("FWHM", np.nan))
                        vR2 = first_scalar(vf["Voigt"].get("R2", np.nan))
                    else:
                        # Fallback to top-level (backward compatibility)
                        v_sigma = float(vf.get("sigma", np.nan))
                        v_gamma = float(vf.get("gamma", np.nan))
                        vR2 = first_scalar(vf.get("R2", np.nan))
                        
                        # Recalculate FWHM if needed
                        if "FWHM" in vf:
                            voigt_FWHM = float(vf["FWHM"])
                        else:
                            L = 2.0*abs(v_gamma)
                            G = 2.0*np.sqrt(2.0*np.log(2.0))*abs(v_sigma)
                            voigt_FWHM = 0.5346*L + np.sqrt(0.2166*L*L + G*G)
                    
                    # Optional: Log model selection results
                    if "ModelSelected" in vf:
                        print(f"[INFO] {sub.name}: Best model = {vf['ModelSelected']} "
                              f"(AIC: G={vf['AIC_BIC']['AIC']['G']:.1f}, "
                              f"L={vf['AIC_BIC']['AIC']['L']:.1f}, "
                              f"V={vf['AIC_BIC']['AIC']['V']:.1f})")
                    
                except Exception as e:
                    print(f"[WARN] {sub.name}: Voigt fit failed: {e}")

            # === Plotting ===
            if PLOT_ENABLED and (need_lor or need_gau or need_voigt):
                plot_save_lineshapes(sub, df_bs["B"].to_numpy(), df_bs["Ab"].to_numpy(),
                                     need_lor, lf, need_gau, gf, need_voigt, vf)

        # === Noise calculation ===
        noise_rms = np.nan
        if need_noise:
            try:
                try:
                    noise_csv = pick_noise_csv(sub / NOISE_SUBDIR)
                except NameError:
                    noise_csv = None
                if noise_csv is not None:
                    noise_df_raw = read_csv_safely(noise_csv)
                    noise_df_std = standardize_noise_df(noise_df_raw)
                    noise_rms = compute_noise(noise_df_std)
                    # （如要畫 PSD，可在此呼叫你的 PSD 畫圖函數）
            except Exception as e:
                print(f"[WARN] {session_dir.name}/{sub.name}: noise calc failed: {e}")

        # === Sensitivity calculation ===
        sensitivity = np.nan
        if _want("sensitivity") and np.isfinite(noise_rms) and np.isfinite(slope_val) and slope_val != 0:
            sensitivity = noise_rms / abs(slope_val)

        # === Build output row ===
        row = {
            "label": sub.name,
            "folder_name": str(sub),
            SWEEP_COL_NAME: extract_sweep_value(sub.name),
            "B_range_used_nT": f"[{B_range_used[0]:.2f}, {B_range_used[1]:.2f}]",
        }
        if _want("slope"):        row[METRIC_COLNAMES["slope"]]       = slope_val * 1e3
        if _want("sloper2"):      row[METRIC_COLNAMES["sloper2"]]     = slope_R2
        if _want("noisepsd"):     row[METRIC_COLNAMES["noisepsd"]]    = noise_rms * 1e6
        if _want("sensitivity"):  row[METRIC_COLNAMES["sensitivity"]] = sensitivity * 1e3
        if _want("lorentzfwhm"):  row[METRIC_COLNAMES["lorentzfwhm"]] = lorentz_FWHM
        if _want("lorentzr2"):    row[METRIC_COLNAMES["lorentzr2"]]   = lorentz_R2
        if _want("gaussianfwhm"): row[METRIC_COLNAMES["gaussianfwhm"]]= gaussian_FWHM
        if _want("gaussianr2"):   row[METRIC_COLNAMES["gaussianr2"]]  = gaussian_R2
        if _want("voigtfwhm"):    row[METRIC_COLNAMES["voigtfwhm"]]   = voigt_FWHM
        if _want("voigtgamma"):   row[METRIC_COLNAMES["voigtgamma"]]  = v_gamma
        if _want("voigtsigma"):   row[METRIC_COLNAMES["voigtsigma"]]  = v_sigma
        if _want("voigtr2"):      row[METRIC_COLNAMES["voigtr2"]]     = vR2
        rows.append(row)

    dynamic_cols = ALWAYS_COLUMNS + [METRIC_COLNAMES[k] for k in METRIC_COLNAMES if k in _SELECT]
    return pd.DataFrame(rows, columns=dynamic_cols)

def process_root(root_dir: Path) -> pd.DataFrame:
    root_dir = root_dir.resolve()
    warnings.filterwarnings("ignore", message=r"Using UFloat objects with std_dev==0.*",
                            category=UserWarning, module=r"uncertainties\.core")

    rules_from_txt = load_experiment_rules(root_dir)

    layout, sessions, sheet_mode = detect_layout(root_dir)
    total = len(sessions)
    print(f"[ROOT] layout={layout}; sessions={len(sessions)}; mode={sheet_mode}")

    sheets: Dict[str, pd.DataFrame] = {}
    for i, sess in enumerate(sessions, start=1):
        tag = (sess.name if sess != root_dir else root_dir.name)
        print(f"[ROOT] start: {tag}  ({i-1}/{total})")
        df_sess = process_session(sess, rules_from_txt, root_dir)
        if len(df_sess):
            sheets[tag] = df_sess
        print(f"[ROOT] done : {tag}  ({i}/{total})")

    # 寫檔 + 在終端印出完整 pandas 表格
    return write_outputs(root_dir, sheets, sheet_mode=sheet_mode)