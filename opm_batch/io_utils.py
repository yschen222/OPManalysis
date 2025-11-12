# io_utils.py
from __future__ import annotations
from pathlib import Path
import pandas as pd, numpy as np, re
from typing import Tuple, List, Optional, Dict, Iterable
from .config import (
    SCAN_CSV_GLOB, NOISE_SUBDIR, SCAN_USE_POSITION, NOISE_USE_POSITION,
    SCAN_POS_MAP, NOISE_POS_MAP, USER_SCAN_COLNAMES, USER_NOISE_COLNAMES
)

# -------------------------------
# 基本讀檔
# -------------------------------
def read_csv_safely(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, engine="python", sep=None, comment="%", skip_blank_lines=True, encoding="utf-8-sig")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    df.columns = [c.strip() for c in df.columns]
    return df.reset_index(drop=True)

# -------------------------------
# 尋找檔案
# -------------------------------
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

# -------------------------------
# 內部工具：強制映射 + 別名自動辨識
# -------------------------------
def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", "", str(name).strip().lower())

def _apply_force_map(
    df_raw: pd.DataFrame,
    roles: Iterable[str],
    force_map: Optional[Dict[str, str]] = None,
    drop_cols: Optional[Iterable[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    force_map 例:
      {"time":"time_s", "Ab":"pd_V", "demod":"demod_V", "tri":"Z_coil_V_calc"}
    drop_cols 例:
      ["Z_coil_V_measured"]
    """
    if not force_map:
        return None
    # 檢查指定來源欄是否存在
    missing = [role for role in roles if (role in force_map) and (force_map[role] not in df_raw.columns)]
    if missing:
        # 指定了但找不到，回傳 None 讓外層走自動流程或報錯
        return None

    df = df_raw.copy()
    # 先丟掉不要的欄（若有）
    if drop_cols:
        df = df.drop(columns=list(drop_cols), errors="ignore")

    # 建立 rename：來源欄名 -> 標準欄名
    rename_map = {force_map[r]: r for r in roles if r in force_map}
    df = df.rename(columns=rename_map)

    # 只保留我們關心的欄位（有指定者）
    keep = [r for r in roles if r in df.columns]
    if not keep:
        return None
    return df[keep].copy()

def _auto_pick_with_alias(df_raw: pd.DataFrame, role_patterns: Dict[str, Iterable[str]]) -> Dict[str, str]:
    """
    依據每個 role 的 regex patterns，自動從 df_raw 中找第一個匹配的欄位。
    回傳：{role: 原始欄名}
    """
    # 預處理：做「規範化名 -> 原名」索引，加速查找
    norm_map = {_normalize_name(c): c for c in df_raw.columns}

    resolved = {}
    for role, patterns in role_patterns.items():
        found_orig = None
        # 先 exact norm 匹配，再 regex
        for pat in patterns:
            pat_norm = _normalize_name(pat)
            if pat_norm in norm_map:
                found_orig = norm_map[pat_norm]
                break
        if not found_orig:
            # regex on normalized names
            for norm_key, orig in norm_map.items():
                if any(re.search(pat, norm_key) for pat in patterns):
                    found_orig = orig
                    break
        if found_orig:
            resolved[role] = found_orig
    return resolved

def _to_numeric_df(df: pd.DataFrame, numeric_cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(how="any")

# -------------------------------
# Scan 標準化
# -------------------------------
def standardize_scan_df(
    df_raw: pd.DataFrame,
    *,
    force_map: Optional[Dict[str, str]] = None,
    drop_cols: Optional[Iterable[str]] = ("Z_coil_V_measured",),
) -> pd.DataFrame:
    """
    產出標準欄位：time, Ab, demod, tri（有 tri 則附上）
    - 先嘗試 force_map；失敗再走位置指定；最後用別名自動辨識。
    - USER_SCAN_COLNAMES 可把標準名輸出成你偏好的欄名（例如 'time' -> 'time'、'Ab' -> 'Ab'）
    """
    roles = ("time", "Ab", "demod", "tri")  # tri 可缺
    out_names = USER_SCAN_COLNAMES  # 例如 {'time':'time','Ab':'Ab','demod':'demod','tri':'tri'}

    # 1) 強制映射（若有）
    forced = _apply_force_map(df_raw, roles, force_map=force_map, drop_cols=drop_cols)
    if forced is not None:
        # 轉數值
        forced = _to_numeric_df(forced, numeric_cols=["time","Ab","demod","tri"])
        # rename to user names
        rename_map = {k: out_names.get(k, k) for k in forced.columns}
        return forced.rename(columns=rename_map)

    # 2) 位置指定（舊功能）
    if SCAN_USE_POSITION:
        try:
            # 允許 tri 缺席
            cols = {
                "time":  df_raw.iloc[:, SCAN_POS_MAP["time"]],
                "Ab":    df_raw.iloc[:, SCAN_POS_MAP["Ab"]],
                "demod": df_raw.iloc[:, SCAN_POS_MAP["demod"]],
            }
            if "tri" in SCAN_POS_MAP and SCAN_POS_MAP["tri"] < df_raw.shape[1]:
                cols["tri"] = df_raw.iloc[:, SCAN_POS_MAP["tri"]]
            df = pd.DataFrame(cols)
            df = _to_numeric_df(df, numeric_cols=list(cols.keys()))
            return df.rename(columns={k: out_names.get(k, k) for k in df.columns})
        except Exception:
            pass  # 改走別名自動辨識

    # 3) 別名自動辨識（更強 alias 覆蓋你的實務欄名）
    #   - time：time, time_s, t, seconds, sec
    #   - Ab：pd, pd_v, pdmv, pd_volt, channela, a, ab
    #   - demod：demod, demod_v, demod_volt, demodvz..., channelb, b
    #   - tri（掃場/三角波/Z_coil 計算值）：tri, triangle, z_coil_v_calc, zcoilcalc, channelc, c
    role_patterns = {
        "time":  [r"time", r"time_s", r"\bt\b", r"seconds?", r"\bsec\b"],
        "Ab":    [r"^ab$", r"\bpd\b", r"pd_v", r"pdmv", r"pdvolt", r"photodiode", r"channela\(?", r"^a$"],
        "demod": [r"^demod$", r"demod_v", r"demodvolt", r"demod_vz.*", r"channelb\(?", r"^b$"],
        "tri":   [r"^tri$", r"triangle", r"z_coil_v_calc", r"zcoil.*calc", r"channelc\(?", r"^c$"],
    }
    # 丟掉不需要的欄位（例如 Z_coil_V_measured）
    df0 = df_raw.drop(columns=list(drop_cols or ()), errors="ignore").copy()

    resolved = _auto_pick_with_alias(df0, role_patterns)

    # 必要欄位檢查
    if not all(k in resolved for k in ("time","Ab","demod")):
        raise ValueError(
            "Scan CSV needs Time + Ab + demod columns.\n"
            f"Resolved: {resolved}\nColumns: {list(df_raw.columns)}"
        )

    data = {
        "time":  df0[resolved["time"]],
        "Ab":    df0[resolved["Ab"]],
        "demod": df0[resolved["demod"]],
    }
    if "tri" in resolved:
        data["tri"] = df0[resolved["tri"]]

    df = pd.DataFrame(data)
    df = _to_numeric_df(df, numeric_cols=list(data.keys()))
    return df.rename(columns={k: out_names.get(k, k) for k in df.columns})

# -------------------------------
# Noise 標準化
# -------------------------------
def standardize_noise_df(
    df_raw: pd.DataFrame,
    *,
    force_map: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    產出標準欄位：time, signal（你可在 USER_NOISE_COLNAMES 把 'signal' 改成你要的名稱）
    - 先嘗試 force_map；失敗再走位置指定；最後用別名自動辨識。
    """
    roles = ("time", "signal")
    out_names = USER_NOISE_COLNAMES  # 例如 {'time':'time','signal':'signal'}

    # 1) 強制映射
    forced = _apply_force_map(df_raw, roles, force_map=force_map, drop_cols=None)
    if forced is not None:
        forced = _to_numeric_df(forced, numeric_cols=["time","signal"])
        return forced.rename(columns={k: out_names.get(k, k) for k in forced.columns})

    # 2) 位置指定
    if NOISE_USE_POSITION:
        try:
            time = df_raw.iloc[:, NOISE_POS_MAP["time"]]
            sig  = df_raw.iloc[:, NOISE_POS_MAP["demod"]]  # 仍沿用你原本 "demod" 的鍵名
            df = pd.DataFrame({"time": time, "signal": sig})
            df = _to_numeric_df(df, numeric_cols=["time","signal"])
            return df.rename(columns={k: out_names.get(k, k) for k in df.columns})
        except Exception:
            pass

    # 3) 別名自動辨識
    role_patterns = {
        "time":   [r"time", r"time_s", r"\bt\b", r"seconds?", r"\bsec\b"],
        "signal": [r"^demod$", r"demod_v", r"demodvolt", r"channelb\(?", r"^b$", r"signal", r"sig2"],
    }
    resolved = _auto_pick_with_alias(df_raw, role_patterns)

    if not all(k in resolved for k in roles):
        raise ValueError(
            "Noise CSV needs at least time + signal columns.\n"
            f"Resolved: {resolved}\nColumns: {list(df_raw.columns)}"
        )

    df = pd.DataFrame({
        "time":   df_raw[resolved["time"]],
        "signal": df_raw[resolved["signal"]],
    })
    df = _to_numeric_df(df, numeric_cols=["time","signal"])
    return df.rename(columns={k: out_names.get(k, k) for k in df.columns})

# -------------------------------
# 規則檔解析（原樣）
# -------------------------------
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