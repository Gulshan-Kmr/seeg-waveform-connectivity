"""
Rise-time ratio analysis.

Hypothesis (from project brief): if channel A has a longer rise time than
channel B, A tends to drive B (i.e. gpdc_a_to_b > gpdc_b_to_a).

Rise time for a channel = ascending-phase fraction / instantaneous frequency
  = asc2desc / mean_if  (seconds per cycle spent rising)

Log rise-time ratio for pair (A, B):
  log(rise_time_A / rise_time_B)
  = log(asc2desc_A / asc2desc_B) - log(mean_if_A / mean_if_B)
  = asc2desc_log_ratio - mean_if_log_ratio   (both already in the table)

Directional PDC log ratio: gpdc_log_ratio = log(gpdc_a_to_b / gpdc_b_to_a)
  positive  →  A drives B
  negative  →  B drives A

Statistical framework:
  - Spearman rho(rise_time_log_ratio, gpdc_log_ratio) per subject per run
  - Wilcoxon signed-rank across subjects (n = 8) to test median rho != 0

Outputs:
  outputs/rise_time_ratio/rise_time_ratio_results.csv   per-subject rho
  outputs/rise_time_ratio/rise_time_ratio_inference.csv group inference
  figures: scatter + forest plot
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "outputs" / "rise_time_ratio"
OUT.mkdir(parents=True, exist_ok=True)

# ── 1. Load pair-level data ───────────────────────────────────────────────────
pairs = pd.read_csv(
    ROOT / "outputs" / "publication_all_pairs_imf4" / "tables" / "all_pairs_connectivity.csv"
)

# Filter: CAR montage, post-stimulation, valid PDC
df = pairs[
    (pairs["montage"] == "car") &
    (pairs["epoch"]   == "post") &
    (pairs["pdc_valid"] == True)
].copy()

print(f"Valid CAR post-stimulation pairs with PDC: {len(df)}")
print(f"Subjects: {sorted(df['subject'].unique())}")

# ── 2. Compute rise-time log ratio ────────────────────────────────────────────
# log(rise_time_A / rise_time_B) = asc2desc_log_ratio - mean_if_log_ratio
df["rise_time_log_ratio"] = df["asc2desc_log_ratio"] - df["mean_if_log_ratio"]

# Drop rows where either ratio is non-finite
finite_mask = (
    np.isfinite(df["rise_time_log_ratio"]) &
    np.isfinite(df["gpdc_log_ratio"])
)
df = df[finite_mask].copy()
print(f"Pairs after finiteness filter: {len(df)}")

# ── 3. Per-subject Spearman rho ───────────────────────────────────────────────
rho_rows = []
for (subj, freq), grp in df.groupby(["subject", "stim_frequency_hz"]):
    if len(grp) < 5:
        continue
    rho, p = stats.spearmanr(grp["rise_time_log_ratio"], grp["gpdc_log_ratio"])
    rho_rows.append({
        "subject"           : subj,
        "stim_frequency_hz" : freq,
        "n_pairs"           : len(grp),
        "rho"               : rho,
        "p"                 : p,
    })

rho_df = pd.DataFrame(rho_rows)
rho_df.to_csv(OUT / "rise_time_ratio_per_subject.csv", index=False)

# Per-subject rho collapsed across frequencies (mean)
subj_rho = rho_df.groupby("subject")["rho"].mean().reset_index()
subj_rho.columns = ["subject", "mean_rho"]

# ── 4. Group-level Wilcoxon ───────────────────────────────────────────────────
rho_vals = subj_rho["mean_rho"].values
wil_stat, wil_p = stats.wilcoxon(rho_vals)

inference = pd.DataFrame([{
    "n_subjects"   : len(rho_vals),
    "median_rho"   : float(np.median(rho_vals)),
    "mean_rho"     : float(np.mean(rho_vals)),
    "wilcoxon_stat": wil_stat,
    "wilcoxon_p"   : wil_p,
}])
inference.to_csv(OUT / "rise_time_ratio_inference.csv", index=False)

print("\nRise-time ratio → PDC log ratio")
print(f"  Per-subject rho (mean ± SD): "
      f"{rho_vals.mean():.3f} ± {rho_vals.std():.3f}")
print(f"  Wilcoxon: stat={wil_stat:.1f}, p={wil_p:.4f}")
print(f"  Per-subject rho values: {[f'{r:.3f}' for r in rho_vals]}")

# ── 5. Figure A — Scatter (pooled) ───────────────────────────────────────────
COLOR = "#7B4FA6"

fig, ax = plt.subplots(figsize=(6, 5))

# Sample for scatter density
rng = np.random.default_rng(42)
n   = len(df)
idx = rng.choice(n, size=min(n, 1500), replace=False)

ax.scatter(
    df["rise_time_log_ratio"].values[idx],
    df["gpdc_log_ratio"].values[idx],
    s=10, alpha=0.25, color=COLOR, linewidths=0, zorder=2,
)

# OLS line (visual guide)
x = df["rise_time_log_ratio"].values
y = df["gpdc_log_ratio"].values
m, b = np.polyfit(x, y, 1)
x_line = np.array([np.percentile(x, 1), np.percentile(x, 99)])
ax.plot(x_line, m * x_line + b, color="#C44536", lw=2, zorder=4)

# Pooled Spearman
rho_pool, p_pool = stats.spearmanr(x, y)
ax.text(0.04, 0.97,
        f"ρ = {rho_pool:.3f}\np = {p_pool:.4f}\nn = {len(df):,} pairs",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.75", alpha=0.9))

ax.axhline(0, color="0.7", lw=0.8, ls="--")
ax.axvline(0, color="0.7", lw=0.8, ls="--")
ax.set_xlabel("Log rise-time ratio  log(T_rise_A / T_rise_B)", fontsize=10)
ax.set_ylabel("PDC log ratio  log(A→B / B→A)", fontsize=10)
ax.set_title("Rise-time ratio vs directional PDC\n(post-stimulation, valid PDC pairs)", fontsize=10)

fig.tight_layout()
fig.savefig(OUT / "fig_rise_time_A_scatter.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig_rise_time_A_scatter.pdf", bbox_inches="tight")
plt.close(fig)
print("\nFigure A (scatter) saved.")

# ── 6. Figure B — Forest plot of per-subject rho ─────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4.5))

subjects = subj_rho["subject"].values
rhos     = subj_rho["mean_rho"].values
y_pos    = np.arange(len(subjects))

ax.scatter(rhos, y_pos, s=60, color=COLOR, zorder=3)
ax.axvline(0, color="0.5", lw=1, ls="--")
ax.axvline(rho_vals.mean(), color=COLOR, lw=1.5, ls=":", alpha=0.7,
           label=f"Mean ρ = {rho_vals.mean():.3f}")

for yi, (rho_v, subj) in enumerate(zip(rhos, subjects)):
    ax.text(rho_v + 0.005, yi, f"{rho_v:+.3f}", va="center", fontsize=8)

sig_label = ("***" if wil_p < 0.001 else "**" if wil_p < 0.01 else
             "*"   if wil_p < 0.05  else "ns")
ax.set_title(
    f"Per-subject Spearman ρ: rise-time ratio vs PDC direction\n"
    f"Wilcoxon p = {wil_p:.4f} {sig_label}  (n = {len(rho_vals)} subjects)",
    fontsize=9,
)
ax.set_yticks(y_pos)
ax.set_yticklabels([s.replace("HEJ_", "") for s in subjects], fontsize=9)
ax.set_xlabel("Spearman ρ", fontsize=10)
ax.legend(fontsize=8, frameon=False)

fig.tight_layout()
fig.savefig(OUT / "fig_rise_time_B_forest.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig_rise_time_B_forest.pdf", bbox_inches="tight")
plt.close(fig)
print("Figure B (forest) saved.")

print(f"\nAll rise-time ratio outputs saved to {OUT}")
