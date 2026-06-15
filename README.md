# Waveform morphology predicts network connectivity in human sEEG

Status: under Manuscript preparation

This repository contains all analysis code, figure-generation scripts, and manuscript
files for:

> **Waveform morphology predicts network connectivity in human
> stereoelectroencephalography**


---

## Overview

We show that the shape of individual alpha oscillatory cycles—quantified by the
ascending-to-descending phase duration ratio (asc2desc) and peak-to-trough duration
ratio (peak2trough)—independently predicts imaginary coherence (ImCoh), directional
drive asymmetry (gPDC), and temporal irreversibility (TRA) across n = 1,009 sEEG
electrodes in eight participants who received electrical brain stimulation at 1 Hz and
50 Hz.

Key findings:
- **Stimulation selectively prolongs the ascending phase** at stimulated electrodes
  (Δasc2desc stim = 0.016 ± 0.020 vs. non-stim = −0.001 ± 0.011; p = 5.1 × 10⁻⁸).
- **asc2desc independently predicts all three connectivity measures** in LME models
  controlling for brain region, stimulation frequency, and channel status.
- Effects are consistent across 1 Hz and 50 Hz protocols and across frontal,
  temporal, insula, and occipital regions.

---

## Repository Structure

```
seeg_waveform_github/
├── environment.yml          # Conda environment (Python 3.11)
├── pyproject.toml           # Package install config
├── .gitignore
│
├── src/seeg_waveform/       # Core Python package
│   ├── config.py            # Paths, parameters, random seed
│   ├── io.py                # Data loading / caching (HDF5, npz)
│   ├── preprocess.py        # CAR referencing, notch filter, artefact rejection
│   ├── pipeline.py          # Run-level orchestration
│   ├── quinn.py             # Masked-EMD + waveform feature extraction
│   ├── connectivity.py      # ImCoh (MNE), gPDC (VAR), TRA
│   ├── imf_connectivity.py  # IMF-specific connectivity
│   ├── all_pairs_waveform_connectivity.py  # All-electrode-pairs pipeline
│   ├── stats.py             # Spearman, Wilcoxon, FDR, LME wrappers
│   ├── plotting.py          # Shared figure utilities
│   ├── control_point_figures.py
│   ├── connectivity_figures.py
│   ├── primary_car_mne.py   # CAR-montage analysis entry point
│   ├── primary_publication.py
│   ├── publication_addons.py
│   ├── region_labeling.py   # AAL atlas region assignment
│   ├── region_imf_analysis.py
│   ├── region_waveform.py
│   ├── region_quinn_figures.py
│   ├── three_epoch_pairwise.py
│   └── spectral.py
│
├── scripts/
│   ├── analysis/            # Pipeline entry-point scripts (run these)
│   │   ├── run_waveform.py              # 1. EMD decomposition + waveform features
│   │   ├── run_primary_car_mne.py       # 2. CAR connectivity (ImCoh, gPDC, TRA)
│   │   ├── run_lme_analysis.py          # 3. Linear mixed-effects models
│   │   ├── run_rise_time_ratio.py       # 4. Rise-time ratio analysis
│   │   ├── run_electrode_time_windows.py
│   │   ├── run_all_pairs_waveform_connectivity.py
│   │   ├── run_connectivity.py
│   │   ├── run_imf_connectivity.py
│   │   ├── run_mixed_effects.py
│   │   ├── run_region_imf_analysis.py
│   │   ├── run_three_epoch_pairwise_extension.py
│   │   ├── run_spectral_analysis.py
│   │   ├── run_sensitivity.py
│   │   └── smoke_test.py
│   │
│   └── figures/             # Figure generation (Figs 1–7 + supplementary)
│       ├── plot_CAR_publication.py      # Main: generates Figs 1–7 + Supp S1
│       ├── plot_figure1_emd_pipeline.py
│       ├── plot_figure2_waveform_change.py
│       ├── plot_figure3_main_scatter.py
│       ├── plot_lme_figures.py          # Figs 4–5
│       └── ...
│
├── configs/                 # YAML parameter files
│   ├── analysis_config.yaml
│   ├── primary_car_mne_config.yaml
│   ├── connectivity_config.yaml
│   └── ...
│
├── manuscript/
│   ├── main.tex             # Full LaTeX manuscript
│   ├── references.bib       # 22 references
│   ├── manuscript.docx      # Word version
│   └── tables/
│       ├── table1_demographics.tex
│       └── table2_lme_results.tex
│
├── figures/
│   ├── main/                # Figs 1–7 (PNG + PDF, 300 dpi)
│   └── supplementary/       # Supp S1 panels (ImCoh, PDC, TRA)
│
└── docs/
    └── methods.md           # Expanded methods notes
```

---

## Quickstart

### 1. Install the environment

```bash
conda env create -f environment.yml
conda activate seeg
pip install -e .          # installs src/seeg_waveform as an editable package
```

### 2. Configure paths

Edit `configs/analysis_config.yaml` to point to your data directory
(raw `.mat` files are not included; see **Data Availability** below).

### 3. Run the analysis pipeline

```bash
# Step 1 – EMD decomposition and waveform feature extraction
python scripts/analysis/run_waveform.py

# Step 2 – CAR-referenced connectivity (ImCoh, gPDC, TRA)
python scripts/analysis/run_primary_car_mne.py

# Step 3 – Linear mixed-effects models
python scripts/analysis/run_lme_analysis.py

# Step 4 – Rise-time ratio analysis
python scripts/analysis/run_rise_time_ratio.py
```

### 4. Reproduce publication figures

```bash
# Generate all Figures 1–7 + Supplementary S1
python scripts/figures/plot_CAR_publication.py
```

---

## Methods Summary

| Step | Script | Description |
|------|--------|-------------|
| Pre-processing | `preprocess.py` | Common average reference (CAR), notch filter (60/120/180 Hz), artefact rejection |
| Waveform decomposition | `quinn.py` | Masked-SIFT EMD, IMF 4 = alpha (9.9 ± 0.8 Hz) |
| Feature extraction | `quinn.py` | IF, asc2desc, peak2trough per cycle |
| ImCoh | `connectivity.py` | MNE multitaper, 8–11 Hz, |ImCoh| per electrode |
| gPDC | `connectivity.py` | Bivariate VAR, AIC order selection, 8–11 Hz |
| TRA | `connectivity.py` | Envelope cross-correlation asymmetry |
| Statistics | `stats.py` | Spearman ρ, Wilcoxon, BH-FDR, LME |

**Reference method:** Quinn, A.J. et al. (2021). Within-cycle instantaneous frequency
profiles report oscillatory waveform dynamics. *J. Neurophysiol.*, 126, 1190–1208.

---

## Data Availability

Raw sEEG recordings contain identifiable patient information and are not publicly shared.
Requests for de-identified data should be directed to the corresponding author.
Derived electrode-level summary statistics (waveform features + connectivity measures)
will be deposited on [OSF / Zenodo] upon acceptance.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.11 | |
| numpy | latest | Numerical computation |
| scipy | latest | Signal processing, statistics |
| pandas | latest | Data management |
| matplotlib | latest | Figures (Arial font, 300 dpi) |
| emd | latest | Masked-EMD (Quinn et al. 2021) |
| mne-connectivity | latest | Imaginary coherence |
| statsmodels | latest | VAR models, LME |
| nibabel / nilearn | latest | AAL atlas region labeling |

---

## Citation

pending

---

## License

Code released under the **MIT License** (see `LICENSE`). Manuscript text and figures
are copyright the authors.
