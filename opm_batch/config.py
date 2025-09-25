import matplotlib
matplotlib.use("Agg")  # headless backend before pyplot import
import matplotlib.pyplot as plt

# ---- Output control ----
XLSX_SHEET_MODE = "all_only"   # "all_only" | "per_session"
PRINT_SUMMARY_TO_STDOUT = True
STDOUT_MAX_ROWS = 200

# ---- Plot settings ----
PLOT_ENABLED = True
PLOT_DPI     = 140
PLOT_FMT     = "png"
PLOT_STYLE   = "default"
PLOT_MAXPTS  = 4000
plt.style.use(PLOT_STYLE)

# ---- Experiment mode ----
EXPERIMENT_MODE = "current"  # "current" | "power"
CURRENT_OFFSET_MA = 0.4

POWER_UNIT   = "mV"
POWER_SCALE  = 1.0
POWER_OFFSET = 0.0

# ---- B-sweep mapping / segmentation ----
B_RANGE_NT     = (-32.12, 32.12)
LINE_SEGMENT   = "half"     # 'half' | 'full'
LINE_DIRECTION = "auto"     # 'auto' | 'rising' | 'falling'

# ---- Dispersion fit window ----
DISP_CENTER  = 0.0
WIDTHRATIO   = 10.0
DISP_P0_AMPL = 0.01

# ---- Noise calc ----
NOISE_MODE              = "lowband"  # "lowband" | "plain"
NOISE_BAND              = (3.0, 80.0)
NOISE_LOCK_DF_HZ        = 0.20
NOISE_DETREND           = "linear"   # "linear" | "constant" | None
NOISE_ENABLE_PREPROCESS = True
NOISE_WINDOW            = "hann"
NOISE_AVERAGE           = "median"

# ---- File selection ----
SCAN_CSV_GLOB   = "*.csv"
NOISE_SUBDIR    = "noise"
SESSION_NAME_PATTERN = r"^\d{2,}$"
EXCEL_PER_SESSION    = True

# ---- Metrics selection ----
SELECT_METRICS = [
    "slope", "sloper2", "noisepsd", "sensitivity",
    "lorentzfwhm", "gaussianfwhm", "voigtfwhm"
]

# ---- Column names ----
if EXPERIMENT_MODE.lower() == "current":
    SWEEP_COL_NAME = "current_mA"
elif EXPERIMENT_MODE.lower() == "power":
    SWEEP_COL_NAME = f"power_ctrl_{POWER_UNIT}"
else:
    raise ValueError("EXPERIMENT_MODE must be 'current' or 'power'")

ALWAYS_COLUMNS = ["label", "folder_name", SWEEP_COL_NAME, "B_range_used_nT"]
METRIC_COLNAMES = {
    "slope":        "slope_mV_per_nT",
    "sloper2":      "slope_R2",
    "noisepsd":     "noise_rms_uV_per_rtHz",
    "sensitivity":  "sensitivity_pT_per_rtHz",
    "lorentzfwhm":  "lorentz_FWHM_nT",
    "lorentzr2":    "lorentz_R2",
    "gaussianfwhm": "gaussian_FWHM_nT",
    "gaussianr2":   "gaussian_R2",
    "voigtfwhm":    "voigt_FWHM_nT",
    "voigtgamma":   "voigt_gamma_nT",
    "voigtsigma":   "voigt_sigma_nT",
    "voigtr2":      "voigt_R2",
}

# ---- Column position maps ----
SCAN_USE_POSITION  = True
NOISE_USE_POSITION = True
SCAN_POS_MAP  = { "time": 0, "Ab": 1, "demod": 2, "tri": 3 }
NOISE_POS_MAP = { "time": 0, "Ab": 1, "demod": 2 }
USER_SCAN_COLNAMES  = { "time": "time", "Ab": "Ab", "demod": "demod", "tri": "tri" }
USER_NOISE_COLNAMES = { "time": "time", "Ab": "Ab", "demod": "demod" }
