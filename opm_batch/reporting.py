from __future__ import annotations
from pathlib import Path
import pandas as pd, re

def write_outputs(root_dir: Path, sheets: dict[str, pd.DataFrame], sheet_mode: str = "per_session") -> pd.DataFrame:
    if not sheets:
        print("[WARN] No data to write."); return pd.DataFrame()

    df_all = pd.concat(sheets.values(), ignore_index=True)

    # Plain pandas table in terminal (clean & readable)
    print("\n=== ALL (summary) ===")
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 180,
        "display.colheader_justify", "left"
    ):
        print(df_all.to_string(index=False))

    # CSV (always write ALL)
    csv_path = root_dir / f"{root_dir.name}.csv"
    df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[OK] CSV written: {csv_path}")

    # XLSX (optional per mode)
    xlsx_path = root_dir / f"{root_dir.name}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        if sheet_mode == "all_only":
            df_all.to_excel(xw, sheet_name="ALL", index=False)
        else:
            for name, df in sheets.items():
                sheet = re.sub(r'[\\/*?:\[\]]', '_', name)[:31] or "Sheet"
                df.to_excel(xw, sheet_name=sheet, index=False)
            if len(sheets) > 1:
                df_all.to_excel(xw, sheet_name="ALL", index=False)
    print(f"[OK] XLSX written: {xlsx_path}")
    return df_all
