# Methods: Waveform Shape and Network Connectivity in sEEG

## 1. Participants and Recordings

Stereo-EEG (sEEG) recordings were obtained from eight participants (Subject01, Subject04, Subject07, HEJ_Subject12–16) undergoing intracranial monitoring. Each participant received electrical brain stimulation at two frequencies: **1 Hz** and **50 Hz**. Recordings were stored as MATLAB HDF5 (.mat) files and loaded directly using `h5py`. One recording (Subject04, 50 Hz stimulation, run 2) was excluded from all analyses due to data quality issues.

---

## 2. Referencing Schemes

All analyses were performed under two independent referencing schemes to assess robustness:

- **Bipolar (adjacent)**: Each channel was re-referenced by subtracting the signal of the immediately adjacent electrode on the same depth shaft. Adjacent bipolar pairs sharing a constituent contact were excluded from pair-level analyses.
- **Common Average Reference (CAR)**: Each channel was re-referenced by subtracting the instantaneous mean across all channels simultaneously recorded. A minimum of five channels was required for a valid CAR computation. Figures presented in this paper use the CAR montage.

---

## 3. Epoch Definition

For each recording, two analysis epochs were defined relative to the stimulation period:

| Epoch | Definition |
|---|---|
| **Pre-stimulation** | −30 s to −2 s before stimulation onset (28 s window; the 2 s guard excluded stimulation-anticipation artifacts) |
| **Post-stimulation** | Stimulation end + 1 s artifact buffer, for 28 s duration |

Stimulation end times were frequency-dependent:
- **1 Hz**: stimulation ended at 30 s → post window: 31 s to 59 s
- **50 Hz**: stimulation ended at 5 s → post window: 6 s to 34 s

Primary results are reported for the **post-stimulation epoch**; pre-stimulation data were computed but are not the focus of this paper.

---

## 4. Signal Preprocessing

Each electrode's signal within the selected epoch was preprocessed as follows:

1. **Demeaning**: the epoch mean was subtracted.
2. **Notch filtering**: a second-order IIR notch filter (quality factor Q = 30) was applied at 60 Hz, 120 Hz, and 180 Hz to suppress line noise.
3. **Drift detection and removal**: a linear trend was fitted to the signal. If the slope-to-IQR ratio exceeded 0.5, the signal was linearly detrended.
4. **Noise flagging**: channels were excluded if they showed:
   - Any non-finite values
   - Peak-to-peak amplitude exceeding 50× the inter-quartile range
   - Power in a 1 Hz band around line frequency exceeding a threshold (flat-spectrum detection)

Channels failing any noise criterion were excluded from all downstream analyses for that epoch.

---

## 5. Empirical Mode Decomposition (EMD)

The preprocessed signal from each eligible electrode was decomposed using **Masked Empirical Mode Decomposition** (mask-SIFT), as implemented in the `emd` Python package (Quinn et al., 2021). This method extracts a set of **Intrinsic Mode Functions (IMFs)** — amplitude- and frequency-modulated oscillatory components ordered from high to low frequency.

### 5.1 Decomposition Parameters

| Parameter | Value |
|---|---|
| Algorithm | Masked SIFT (`emd.sift.mask_sift`) |
| Maximum IMFs | 6 |
| Number of mask phases | 4 |
| Mask amplitude mode | `ratio_sig` (ratio to signal amplitude) |
| Mask amplitude ratio | 1 |
| Mask step factor | 2 |
| Sifting threshold | 1 × 10⁻⁸ |
| IMF stop criterion | Rilling criterion (sd_thresh = 0.05) |
| Envelope interpolation | Monotone PCHIP |
| Mask frequencies (normalised) | [80, 40, 20, 10, 5, 2, 1] / Fs Hz |

### 5.2 IMF Selection

**IMF 4** (zero-indexed: index 3) was selected as the component of interest for all waveform and connectivity analyses. IMF 4 captured the dominant oscillation in the alpha band (mean instantaneous frequency 9.9 ± 0.8 Hz, range 6.4–11.9 Hz across electrodes), consistent with prior work (Quinn et al., 2021).

---

## 6. Instantaneous Phase, Frequency, and Amplitude

For IMF 4, the **Hilbert transform** was applied to obtain:

- **Instantaneous phase (IP)**: the analytic phase of the IMF
- **Instantaneous frequency (IF)**: the time derivative of the instantaneous phase (in Hz), smoothed using a 7-sample sliding window (`smooth_phase=3`, half-width 3 samples)
- **Instantaneous amplitude (IA)**: the analytic envelope of the IMF

These were computed using `emd.spectra.frequency_transform(imf, fs, 'hilbert', smooth_phase=3)`.

---

## 7. Cycle Detection and Waveform Feature Extraction

### 7.1 Cycle Detection

Cycles were detected from the instantaneous phase of IMF 4 using `emd.cycles.Cycles`. Each cycle spanned one full 2π phase rotation. A cycle was retained in the analysis only if it passed all of the following quality criteria:

| Criterion | Threshold |
|---|---|
| Cycle completeness (`is_good`) | Must be a complete cycle |
| Maximum amplitude | > 10th percentile of instantaneous amplitude in the epoch |
| Maximum instantaneous frequency | < 20 Hz |
| IF range within cycle | < 20 Hz |
| Minimum cycles per electrode per epoch | ≥ 5 |

### 7.2 Per-Cycle Waveform Metrics

Three shape metrics were computed for each accepted cycle using the Quinn et al. (2021) method:

#### Instantaneous Frequency (IF)
The **mean instantaneous frequency** of IMF 4 across all time points within the cycle, in Hz. Reflects how fast the oscillation cycles.

#### Ascent/Descent Ratio (`asc2desc`)
The fraction of the cycle's total duration spent in the **ascending phase**:

$$\text{asc2desc} = \frac{t_\text{peak} + (T - t_\text{trough})}{T}$$

where $T$ is the total cycle length in samples, $t_\text{peak}$ is the peak sample (detected by parabolic interpolation via `emd.cycles.cf_peak_sample`), and $t_\text{trough}$ is the trough sample (`emd.cycles.cf_trough_sample`). Because each cycle is delimited by ascending zero-crossings, the ascending phase is measured as two non-contiguous intervals: from cycle onset to the peak ($t_\text{peak}$ samples), and from the trough to cycle end ($T - t_\text{trough}$ samples). Values > 0.5 indicate a slow rise and fast fall; values < 0.5 indicate a fast rise and slow fall.

#### Peak/Trough Ratio (`peak2trough`)
The **position of the descending zero-crossing as a fraction of total cycle length**:

$$\text{peak2trough} = \frac{t_\text{desc\_zero}}{T}$$

where $t_\text{desc\_zero}$ is the descending zero-crossing sample (`emd.cycles.cf_descending_zero_sample`). Values near 0.5 indicate a symmetric waveform; values below 0.5 indicate an early descending zero-crossing (sharp peak); values above 0.5 indicate a late descending zero-crossing (blunt peak).

### 7.3 Per-Electrode Feature

Each of the three cycle-level metrics was **averaged across all accepted cycles** within the epoch to yield a single per-electrode, per-epoch value:

- `mean_if`: mean instantaneous frequency (Hz)
- `asc2desc`: mean ascent/descent ratio
- `peak2trough`: mean peak/trough ratio

---

## 8. Connectivity Measures

Three pairwise connectivity measures were computed for every eligible channel pair (A, B) within each epoch.

### 8.1 Imaginary Coherence (ImCoh)

**Imaginary coherence** was computed from the **imaginary part of the cross-spectrum** between channels A and B, averaged over the alpha band (8–11 Hz). It is robust to volume conduction because spurious zero-lag synchrony contributes only to the real part of the coherence.

Computation used the **multitaper method** (`mne-connectivity`):
- Frequency band: **8–11 Hz** (averaged across the band)
- Segment length: **4 seconds**
- Number of segments: **7** (consecutive non-overlapping, spanning the full 28-second epoch)
- Multitaper adaptive weighting: disabled

The signed ImCoh and its absolute value (|ImCoh|) were both retained.

### 8.2 Generalised Partial Directed Coherence (gPDC)

**Generalised PDC** quantifies the **directional** causal influence between pairs of channels. For each pair (A, B):

1. The IMF 4 signals from both channels were **downsampled to 128 Hz** using polyphase resampling (`scipy.signal.resample_poly`).
2. A **bivariate Vector Autoregressive (VAR)** model was fitted to the pair using AIC order selection, with maximum order capped at `min(30, n_samples/20)`.
3. VAR stability was verified (`fit.is_stable()`).
4. Residual whiteness was tested using the **Ljung–Box test** (lag = min(20, n_samples/5)); pairs failing this test (p < 0.01) were flagged as invalid and excluded.
5. The **gPDC spectrum** was evaluated at 64 frequency points within 8–11 Hz and averaged:
   - `gpdc_a_to_b`: normalised power flowing from A to B
   - `gpdc_b_to_a`: normalised power flowing from B to A
   - `gpdc_log_ratio`: log(gpdc_a_to_b / gpdc_b_to_a) — positive values indicate A drives B

VAR parameters:

| Parameter | Value |
|---|---|
| Target sampling rate | 128 Hz |
| Maximum VAR order | 30 |
| Order selection criterion | AIC |
| Whiteness test | Ljung–Box, p < 0.01 |
| Frequency band | 8–11 Hz |
| Evaluation points | 64 linearly spaced |

### 8.3 Temporal Irreversibility (TRA)

**Temporal irreversibility** (unsigned) quantifies whether the cross-correlation between the Hilbert envelopes of A and B looks different forwards vs backwards in time — a signature that their interaction is time-asymmetric.

The measure follows the invariant bivariate feature from the MATLAB reference (`invariant_features_bivariate_v2.m`, S=0):

1. The **Hilbert envelope** of each channel's IMF 4 was computed.
2. Envelopes were normalised by their standard deviation for numerical conditioning (not centred, to preserve the MATLAB equivalence).
3. The **cross-correlation** was computed via FFT at lags 1 to `floor(Fs × 1.0)` samples (up to 1 second):

$$\text{TRA} = \sqrt{ \frac{1}{\det(\Sigma)} \cdot \frac{1}{L} \sum_{\ell=1}^{L} \left( \frac{[y \star x]_\ell - [x \star y]_\ell}{N - \ell} \right)^2 }$$

where $\Sigma$ is the 2×2 covariance matrix of the normalised envelopes, $N$ is the signal length, and $L$ = `floor(Fs × 1.0)` is the maximum lag.

Pairs were excluded if the covariance matrix was degenerate (determinant ≤ 1 × 10⁻¹⁴).

---

## 9. Electrode-Level Aggregation

To move from pair-level to **electrode-level** measures, the pairwise connectivity values were aggregated across all pairs in which each electrode appeared:

| Measure | Aggregation |
|---|---|
| **Mean \|ImCoh\|** | Mean of `abs_imcoh` across all pairs involving the electrode |
| **PDC asymmetry** | Mean outgoing gPDC − mean incoming gPDC across all valid pairs |
| **Mean TRA** | Mean of `tra_unsigned` across all valid pairs |

Only pairs with a valid gPDC fit (VAR stable + residuals white) contributed to the PDC asymmetry. Only pairs with a valid TRA estimate (non-degenerate covariance) contributed to the mean TRA.

This yielded one row per electrode per epoch, with three waveform shape features and three connectivity summaries.

---

## 10. Statistical Analysis

### 10.1 Within-Electrode Spearman Correlation

For each electrode within each participant, recording, and epoch, a **Spearman rank correlation** was computed between a waveform feature and a connectivity measure. Spearman's ρ was used because both variables are non-normally distributed and the relationship is not assumed to be linear.

Three waveform features × three connectivity measures = **nine hypothesis tests** per participant per montage.

### 10.2 Group-Level Inference

Per-participant Spearman ρ values were tested against zero using a **Wilcoxon signed-rank test** across participants (n = 8). This tests whether the median ρ across participants is consistently non-zero. With n = 8 participants, the Wilcoxon test is sensitive to large, consistent effects (minimum achievable two-tailed p ≈ 0.008) but has limited power to detect small effects; results should be interpreted accordingly.

### 10.3 Multiple Comparison Correction

**Benjamini–Hochberg False Discovery Rate (FDR)** correction was applied within each montage × epoch combination across all nine hypotheses. FDR-corrected p-values are reported in the supplementary statistics table; uncorrected Spearman ρ and p-values are shown in the figures.

### 10.4 Subgroup Analyses

The following subgroup analyses were performed on the CAR post-stimulation data:

- **Stimulated vs non-stimulated electrodes**: Spearman ρ computed separately for electrodes designated as stimulation targets and all remaining electrodes.
- **Stimulation frequency**: analyses repeated separately for 1 Hz and 50 Hz stimulation protocols.
- **Brain region**: electrodes were assigned to anatomical regions (frontal, temporal, insula, hippocampus, parietal, occipital, amygdala) using the AAL atlas. Spearman ρ was computed within each region and FDR-corrected across regions.

---

## 11. Confirmatory Linear Mixed Effects Analysis

To assess whether waveform shape features predicted connectivity **after controlling for anatomical and experimental confounds**, a Linear Mixed Effects (LME) model was fitted for each connectivity measure using `statsmodels` MixedLM:

$$\text{connectivity}_z \sim \beta_1 \cdot \text{IF}_z + \beta_2 \cdot \text{asc2desc}_z + \beta_3 \cdot \text{peak2trough}_z + \text{region} + \text{stim\_freq} + \text{is\_stim} + (1 \mid \text{subject})$$

All continuous predictors and outcomes were z-scored prior to fitting so that β coefficients are directly comparable across features. Fixed-effect confounds were: brain region (seven-level categorical, reference = frontal; amygdala collapsed into "other" due to n = 2), stimulation frequency (1 Hz vs 50 Hz), and stimulated-channel status (binary). A random intercept per subject was included to account for between-subject baseline differences.

Models were fitted using REML with the Powell optimiser. Where the random-effect variance estimate collapsed to the boundary (< 10⁻⁶), the model was re-fitted using maximum likelihood. **Benjamini–Hochberg FDR correction** was applied jointly across all nine waveform-feature fixed effects (3 features × 3 outcomes). This analysis uses the same 995 CAR post-stimulation electrodes as the primary Spearman analysis.

---

## 12. Visualisation

All figures used the **common-average reference (CAR)** montage and the **post-stimulation epoch**. Each scatter plot shows one point per electrode. The regression line is an **ordinary least-squares (OLS)** fit to the raw values, included solely as a visual aid to the direction and magnitude of the relationship; the OLS line is not tested and no inference is drawn from its slope or intercept. The annotated statistic is always Spearman ρ. Figures were produced in Python using `matplotlib` with Arial font at 300 dpi, with fonts embedded in PDF (fonttype 42) for publication compatibility.

---

## 13. Software and Reproducibility

| Component | Package / version |
|---|---|
| EMD decomposition | `emd` (Quinn et al., 2021) |
| Connectivity (ImCoh, Coh) | `mne-connectivity` |
| VAR models (gPDC) | `statsmodels` |
| Signal processing | `scipy` |
| Statistical tests | `scipy.stats`, `statsmodels.stats` |
| Numerical computation | `numpy`, `pandas` |
| Visualisation | `matplotlib` |
| Configuration | YAML (PyYAML) |
| Parallel computation | `joblib` |

All analysis code, configuration files, and figure scripts are version-controlled in the project repository. Random seeds are fixed (`seed = 20260612`) for all permutation and bootstrap procedures.

---

## References

Quinn, A. J., Lopes-dos-Santos, V., Huang, N., Liang, W.-K., Juan, C.-H., Yeh, J.-R., Nobre, A. C., Dupret, D., & Woolrich, M. W. (2021). Within-cycle instantaneous frequency profiles report oscillatory waveform dynamics. *Journal of Neurophysiology*, 126(4), 1190–1208.
