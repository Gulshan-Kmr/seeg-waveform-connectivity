"""
Linear Mixed Effects Model analysis.
Predicts connectivity from waveform shape features, controlling for
brain region, stimulation frequency, and stimulated-channel status.

One model per connectivity measure (3 total):
  connectivity_z ~ IF_z + asc2desc_z + peak2trough_z
                 + C(region) + C(stim_freq) + is_stim
                 + (1 | subject)

Outputs:
  outputs/lme_analysis/lme_results.csv        — fixed-effect estimates + FDR
  outputs/lme_analysis/lme_partial_resid.csv  — partial residuals for figures
  outputs/lme_analysis/lme_model_summary.txt
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "outputs" / "lme_analysis"
OUT.mkdir(parents=True, exist_ok=True)

# ── 1. Load and filter ────────────────────────────────────────────────────────
elec = pd.read_csv(ROOT / "outputs" / "electrode_level_analysis" / "electrode_level_summary.csv")
df = elec[(elec["montage"] == "car") & (elec["epoch"] == "post")].copy()

KEY_COLS = [
    "mean_if", "asc2desc", "peak2trough",
    "mean_abs_imcoh", "pdc_asymmetry", "mean_tra",
    "region_family", "stim_frequency_hz", "is_stimulated_channel", "subject",
]
df = df.dropna(subset=KEY_COLS).copy()
print(f"N electrodes: {len(df)}  |  N subjects: {df['subject'].nunique()}")

# ── 2. Prepare variables ──────────────────────────────────────────────────────
# Merge amygdala (n=2) into other — too few for a separate dummy
df["region"] = df["region_family"].replace({"amygdala": "other"})

# Categorical: stim frequency as string so C() treats it as factor
df["stim_freq"] = df["stim_frequency_hz"].astype(str)   # "1.0" vs "50.0"

# Boolean → int for is_stimulated_channel
df["is_stim"] = df["is_stimulated_channel"].astype(int)

# Z-score all continuous variables (features + outcomes)
WAVEFORM = ["mean_if", "asc2desc", "peak2trough"]
OUTCOMES  = ["mean_abs_imcoh", "pdc_asymmetry", "mean_tra"]

scale_stats = {}
for col in WAVEFORM + OUTCOMES:
    mu, sd = df[col].mean(), df[col].std()
    scale_stats[col] = (mu, sd)
    df[col + "_z"] = (df[col] - mu) / sd

# ── 3. Fit models ─────────────────────────────────────────────────────────────
OUTCOME_LABELS = {
    "mean_abs_imcoh": "|ImCoh|",
    "pdc_asymmetry" : "PDC asymmetry",
    "mean_tra"      : "TRA",
}
FEATURE_LABELS = {
    "mean_if_z"    : "Instantaneous frequency",
    "asc2desc_z"   : "Ascent/descent ratio",
    "peak2trough_z": "Peak/trough ratio",
}

records   = []
model_fits = {}
summary_lines = []

for outcome_col, outcome_label in OUTCOME_LABELS.items():
    formula = (
        f"{outcome_col}_z ~ mean_if_z + asc2desc_z + peak2trough_z"
        f" + C(region, Treatment('frontal'))"
        f" + C(stim_freq, Treatment('1.0'))"
        f" + is_stim"
    )
    model = smf.mixedlm(formula, df, groups=df["subject"])

    # REML with powell; if random-effect variance collapses to boundary use ML
    fit = model.fit(reml=True, method="powell", maxiter=3000, disp=False)
    if fit.cov_re.values[0][0] < 1e-6:
        fit = model.fit(reml=False, method="powell", maxiter=3000, disp=False)

    model_fits[outcome_col] = fit

    summary_lines.append(f"\n{'='*70}\nOutcome: {outcome_label}\n{'='*70}")
    summary_lines.append(str(fit.summary()))

    ci = fit.conf_int()
    for feat_col, feat_label in FEATURE_LABELS.items():
        records.append({
            "outcome"      : outcome_col,
            "outcome_label": outcome_label,
            "feature"      : feat_col,
            "feature_label": feat_label,
            "beta"         : fit.params[feat_col],
            "se"           : fit.bse[feat_col],
            "ci_lo"        : ci.loc[feat_col, 0],
            "ci_hi"        : ci.loc[feat_col, 1],
            "p"            : fit.pvalues[feat_col],
        })

# ── 4. FDR correction across 9 waveform-feature tests ────────────────────────
res = pd.DataFrame(records)
_, p_fdr, _, _ = multipletests(res["p"].values, method="fdr_bh")
res["p_fdr"] = p_fdr

res.to_csv(OUT / "lme_results.csv", index=False)

# Save text summaries
with open(OUT / "lme_model_summary.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(summary_lines))

# ── 5. Print summary table ────────────────────────────────────────────────────
print("\nLME fixed effects — waveform features only")
print("-" * 72)
print(f"{'Outcome':<18} {'Feature':<28} {'beta':>7} {'95% CI':>18} {'p':>8} {'FDR-p':>8}")
print("-" * 72)
for _, row in res.iterrows():
    sig = ("***" if row.p_fdr < 0.001 else
           "**"  if row.p_fdr < 0.01  else
           "*"   if row.p_fdr < 0.05  else "")
    print(f"{row.outcome_label:<18} {row.feature_label:<28} "
          f"{row.beta:+7.3f}  [{row.ci_lo:+.3f}, {row.ci_hi:+.3f}]  "
          f"{row.p:8.4f}  {row.p_fdr:8.4f} {sig}")

print(f"\nResults saved to {OUT}")

# ── 6. Save partial residuals for each feature × outcome combination ──────────
# Partial residual for feature j = LME residuals + beta_j * x_j
# This is what the figure script plots in the partial regression scatter.
partial_rows = []
for outcome_col in OUTCOME_LABELS:
    fit   = model_fits[outcome_col]
    resid = fit.resid.values
    for feat_col in FEATURE_LABELS:
        beta_j   = fit.params[feat_col]
        x_vals   = df[feat_col].values
        y_partial = resid + beta_j * x_vals
        partial_rows.append(
            df[["subject"]].assign(
                outcome=outcome_col,
                feature=feat_col,
                x_z=x_vals,
                y_partial=y_partial,
            )
        )

partial_df = pd.concat(partial_rows, ignore_index=True)
partial_df.to_csv(OUT / "lme_partial_resid.csv", index=False)
print("lme_partial_resid.csv saved.")
