from __future__ import annotations
from pathlib import Path
import pandas as pd, numpy as np, re
from typing import Tuple, List, Optional
from .config import (
    SCAN_CSV_GLOB, NOISE_SUBDIR, SCAN_USE_POSITION, NOISE_USE_POSITION,
    SCAN_POS_MAP, NOISE_POS_MAP, USER_SCAN_COLNAMES, USER_NOISE_COLNAMES
)

def read_csv_safely(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, engine="python", sep=None, comment="%", skip_blank_lines=True, encoding="utf-8-sig")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    df.columns = [c.strip() for c in df.columns]
    return df.reset_index(drop=True)

def pick_scan_csv(folder: Path, root: Path) -> Optional[Path]:
    cands = []
    for csv in folder.glob(SCAN_CSV_GLOB):
        if csv.is_dir(): continue
        if csv.parent.name == NOISE_SUBDIR: continue
        if csv.name == f"{root.name}.csv": continue
        cands.append(csv)
    return sorted(cands)[0] if cands else None

def pick_noise_csv(noise_dir: Path) -> Optional[Path]:
    if not noise_dir.exists() or not noise_dir.is_dir(): return None
    for csv in sorted(noise_dir.glob("*.csv")):
        if csv.is_file(): return csv
    return None

def standardize_scan_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_raw = df_raw.copy(); out_names = USER_SCAN_COLNAMES
    if SCAN_USE_POSITION:
        try:
            cols = {k: pd.to_numeric(df_raw.iloc[:, idx], errors="coerce") for k, idx in SCAN_POS_MAP.items()}
            df = pd.DataFrame(cols).dropna(how="any")
            return df.rename(columns={k: out_names.get(k, k) for k in cols})
        except Exception:
            pass
    norm = {re.sub(r"\s+", "", c.lower()): c for c in df_raw.columns}
    def pick(key): return next((orig for k, orig in norm.items() if key in k), None)
    tcol = pick("time"); Acol = pick("channela(") or pick("channela")
    Bcol = pick("channelb(") or pick("channelb")
    Ccol = pick("channelc(") or pick("channelc")
    if not (tcol and Acol and Bcol and Ccol):
        raise ValueError("Scan CSV needs Time and Channel A/B/C (or provide positions).")
    df = pd.DataFrame({
        out_names.get("time","time"):  pd.to_numeric(df_raw[tcol], errors="coerce"),
        out_names.get("Ab","Ab"):      pd.to_numeric(df_raw[Acol], errors="coerce"),
        out_names.get("demod","demod"):pd.to_numeric(df_raw[Bcol], errors="coerce"),
        out_names.get("tri","tri"):    pd.to_numeric(df_raw[Ccol], errors="coerce"),
    }).dropna(how="any")
    return df

def standardize_noise_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_raw = df_raw.copy(); out_names = USER_NOISE_COLNAMES
    if NOISE_USE_POSITION:
        try:
            t_idx = NOISE_POS_MAP["time"]; d_idx = NOISE_POS_MAP["demod"]
            time = pd.to_numeric(df_raw.iloc[:, t_idx], errors="coerce")
            demod = pd.to_numeric(df_raw.iloc[:, d_idx], errors="coerce")
            a_val = None
            if "Ab" in NOISE_POS_MAP and NOISE_POS_MAP["Ab"] < df_raw.shape[1]:
                a_val = pd.to_numeric(df_raw.iloc[:, NOISE_POS_MAP["Ab"]], errors="coerce")
            df = pd.DataFrame({
                out_names.get("time","time"): time,
                "sig1": (a_val if a_val is not None else 0.0),
                "sig2": demod,
            }).dropna(how="any")
            return df
        except Exception:
            pass
    norm = {re.sub(r"\s+", "", c.lower()): c for c in df_raw.columns}
    def pick(key): return next((orig for k, orig in norm.items() if key in k), None)
    tcol = pick("time"); Bcol = pick("channelb(") or pick("channelb")
    if not (tcol and Bcol):
        raise ValueError("Noise CSV needs at least Time and Channel B (or provide positions).")
    df = pd.DataFrame({
        out_names.get("time","time"):  pd.to_numeric(df_raw[tcol], errors="coerce"),
        "sig2":  pd.to_numeric(df_raw[Bcol], errors="coerce"),
    }).dropna(how="any")
    return df

def load_experiment_rules(root_dir: Path):
    p = root_dir / "實驗記錄.txt"
    if not p.exists(): return []
    txt = p.read_text(encoding="utf-8-sig")
    pattern = re.compile(
        r"(?P<lo>\d+(?:\.\d+)?)\s*~\s*(?P<hi>\d+(?:\.\d+)?)"
        r".*?掃磁範圍.*?(?:\+/-|±)\s*(?P<Bpm>\d+(?:\.\d+)?)\s*nT"
        r"\s*\(\s*(?P<mVpp>\d+(?:\.\d+)?)\s*mVpp\s*\)"
        r"[，,]\s*(?P<fmHz>\d+(?:\.\d+)?)\s*mHz",
        re.VERBOSE | re.IGNORECASE
    )
    rules = []
    for line in txt.splitlines():
        m = pattern.search(line)
        if not m: continue
        lo, hi = float(m.group("lo")), float(m.group("hi"))
        lo, hi = (lo, hi) if lo <= hi else (hi, lo)
        rules.append(dict(lo=lo, hi=hi, Bpm=float(m.group("Bpm")),
                          mVpp=float(m.group("mVpp")), fmHz=float(m.group("fmHz"))))
    return rules

def choose_rule_for_label(rules: list, label: str):
    m = re.search(r"([-+]?\d+(?:\.\d+)?)", label)
    val = float(m.group(1)) if m else None
    if val is None or not rules: return None, None
    hits = [r for r in rules if r["lo"] <= val <= r["hi"]]
    if not hits: return None, None
    best = min(hits, key=lambda r: (r["hi"] - r["lo"], -r["Bpm"]))
    Bpm = best["Bpm"]
    return (-Bpm, Bpm), {"mVpp": best["mVpp"], "fmHz": best["fmHz"], "lo": best["lo"], "hi": best["hi"]}
