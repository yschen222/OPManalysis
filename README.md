# 🧪 OPManalysis

**A lightweight Python toolkit for analyzing optically pumped magnetometer (OPM) experimental data**,  
including signal preprocessing, curve fitting, and sensitivity estimation.

This package is designed for experiments using setups like **Moku:Pro**, and supports typical signal structures:  
absorption signals, demodulated signals, and magnetic field sweeps.

---

## 📦 Features

- ✂️ `one_cycle_cut`: Automatically extract one full sweep cycle centered at absorption peak
- 📉 `lorentz_fit`, `gauss_fit`: Fit absorption signals using Lorentzian or Gaussian models
- 📐 `dispersion_lorentz_fit`: Analytical model for low-field demodulated signal in SERF regime
- 📊 `linear_region_fit`: Automatically find linear region and slope in dispersion-shaped data
- 📈 `noise_psd`: Estimate RMS noise from PSD near modulation frequency
- ⚙️ Designed for **easy use in Jupyter Notebooks**

---

## 🚀 Installation

In your working directory (where `pyproject.toml` is located), run:

```bash
pip install git+https://github.com/yschen222/OPManalysis .
```

> You can now import the package with:
>
> ```python
> from opmantool import lorentz_fit, one_cycle_cut, noise_psd, ...
> ```

---

## 📂 Example Notebook

An interactive example is provided in:

```bash
examples/OPM_analysis_demo.ipynb
```

It demonstrates:

- Data loading and cutting
- Absorption signal fitting (Lorentzian and Gaussian)
- Demodulated signal fitting (analytical low-field model)
- Linear region detection
- PSD-based noise analysis
- Sensitivity estimation

The example uses demo data in:

```bash
examples/Demo_data/
```

---

## 📘 Dependencies

- `numpy`, `scipy`, `pandas`, `matplotlib`
- `lmfit` (for Gaussian fitting)

Install all with:

```bash
pip install -r requirements.txt
```

---

## 🧑‍🔬 Author

Developed by **Yi-Hsuan Chen**  
If used in academic work, please consider citing or acknowledging the project.

