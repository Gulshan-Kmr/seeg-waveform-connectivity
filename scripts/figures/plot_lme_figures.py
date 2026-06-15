"""
Figures for Linear Mixed Effects Model analysis.
Reads CSVs produced by run_lme_analysis.py — no model objects needed.

Figure A — Coefficient plot:
  3 panels (one per connectivity measure), waveform features on Y-axis,
  standardised beta +/- 95% CI on X-axis. Filled = FDR < 0.05.

Figure B — Partial regression scatter (3x3 grid):
  Augmented residuals (residuals + predictor contribution) vs z-scored
  feature. The slope of the OLS line in this plot equals beta exactly,
  so the line and the annotated coefficient are fully consistent.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
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
OUT  = ROOT / "outputs" / "lme_analysis"

res     = pd.read_csv(OUT / "lme_results.csv")
partial = pd.read_csv(OUT / "lme_partial_resid.csv")

OUTCOME_LABELS = {
    "mean_abs_imcoh": "|ImCoh|",
    "pdc_asymmetry" : "PDC asymmetry",
    "mean_tra"      : "TRA",
}
FEATURE_LABELS = {
    "mean_if_z"    : "Instantaneous\nFrequency",
    "asc2desc_z"   : "Ascent/Descent\nRatio",
    "peak2trough_z": "Peak/Trough\nRatio",
}
FEATURE_XLABELS = {
    "mean_if_z"    : "Instantaneous frequency (z)",
    "asc2desc_z"   : "Ascent/descent ratio (z)",
    "peak2trough_z": "Peak/trough ratio (z)",
}
OUTCOME_COLORS = {
    "mean_abs_imcoh": "#4C8BB5",
    "pdc_asymmetry" : "#E8A838",
    "mean_tra"      : "#5BAD6F",
}

def sig_stars(p: float) -> str:
    if not np.isfinite(p): return ""
    return ("***" if p < 0.001 else "**" if p < 0.01 else
            "*"   if p < 0.05  else "ns")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE A — Coefficient plot
# ════════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)

feat_keys   = list(FEATURE_LABELS.keys())[::-1]   # reverse: IF on top
feat_labels = [FEATURE_LABELS[k] for k in feat_keys]
y_pos = np.arange(len(feat_keys))

for ax, (outcome_col, outcome_label) in zip(axes, OUTCOME_LABELS.items()):
    color  = OUTCOME_COLORS[outcome_col]
    subset = res[res["outcome"] == outcome_col].set_index("feature")

    for yi, feat_col in enumerate(feat_keys):
        row  = subset.loc[feat_col]
        xerr = np.array([[row.beta - row.ci_lo], [row.ci_hi - row.beta]])
        fc   = color if row.p_fdr < 0.05 else "white"
        ax.errorbar(row.beta, yi, xerr=xerr,
                    fmt="o", color=color, markerfacecolor=fc,
                    markeredgecolor=color, markersize=9,
                    elinewidth=1.8, capsize=5, capthick=1.8, zorder=3)
        stars = sig_stars(row.p_fdr)
        if stars and stars != "ns":
            ax.text(row.ci_hi + 0.02, yi, stars,
                    va="center", fontsize=9, color=color, fontweight="bold")

    ax.axvline(0, color="0.5", lw=1, ls="--", zorder=1)
    ax.set_title(outcome_label, fontsize=11, fontweight="bold", color=color, pad=8)
    ax.set_xlabel("Standardised β  (95% CI)", fontsize=9)
    ax.set_xlim(-0.40, 0.40)

axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(feat_labels, fontsize=9)

legend_handles = [
    Line2D([0],[0], marker="o", color="0.4", markerfacecolor="0.4",
           markersize=8, lw=0, label="FDR < 0.05"),
    Line2D([0],[0], marker="o", color="0.4", markerfacecolor="white",
           markeredgecolor="0.4", markersize=8, lw=0, label="ns"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2,
           fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))

fig.suptitle(
    "LME: waveform shape predicts connectivity  (standardised β, post-stimulation)\n"
    "controlling for brain region, stimulation frequency, stimulated-channel status",
    fontsize=10, y=1.01,
)
fig.tight_layout()
fig.savefig(OUT / "fig_lme_A_coefficients.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig_lme_A_coefficients.pdf", bbox_inches="tight")
plt.close(fig)
print("Figure A saved.")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE B — Partial regression scatter (3 × 3 grid)
# ════════════════════════════════════════════════════════════════════════════════
feat_keys_b  = list(FEATURE_LABELS.keys())
outcome_keys = list(OUTCOME_LABELS.keys())

fig, axes = plt.subplots(3, 3, figsize=(13, 11))
rng = np.random.default_rng(42)

for col_i, outcome_col in enumerate(outcome_keys):
    color = OUTCOME_COLORS[outcome_col]

    for row_i, feat_col in enumerate(feat_keys_b):
        ax = axes[row_i][col_i]

        sub = partial[
            (partial["outcome"] == outcome_col) &
            (partial["feature"] == feat_col)
        ]
        x_vals   = sub["x_z"].values
        y_partial = sub["y_partial"].values

        # Sample for scatter (keeps PDF size manageable)
        n   = len(x_vals)
        idx = (rng.choice(n, size=min(n, 600), replace=False)
               if n > 600 else np.arange(n))

        ax.scatter(x_vals[idx], y_partial[idx],
                   s=14, alpha=0.35, color=color, zorder=2, linewidths=0)

        # OLS line — slope equals beta exactly in partial residual space
        row_stat = res[
            (res["outcome"] == outcome_col) & (res["feature"] == feat_col)
        ].iloc[0]
        x_line = np.array([x_vals.min(), x_vals.max()])
        ax.plot(x_line, row_stat.beta * x_line, color="#C44536", lw=2, zorder=4)

        stars = sig_stars(row_stat.p_fdr)
        txt = (f"β = {row_stat.beta:+.3f}\n"
               f"p = {row_stat.p:.4f}\n"
               f"FDR-p = {row_stat.p_fdr:.4f}"
               + (f" {stars}" if stars != "ns" else ""))
        ax.text(0.04, 0.97, txt, transform=ax.transAxes, va="top",
                fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.9))

        ax.axhline(0, color="0.75", lw=0.8, ls="--")
        ax.axvline(0, color="0.75", lw=0.8, ls="--")

        if row_i == 0:
            ax.set_title(OUTCOME_LABELS[outcome_col], fontsize=10,
                         fontweight="bold", color=color, pad=6)
        if col_i == 0:
            ax.set_ylabel(
                FEATURE_LABELS[feat_col].replace("\n", " ") + "\n(partial residuals)",
                fontsize=8,
            )
        if row_i == 2:
            ax.set_xlabel(FEATURE_XLABELS[feat_col], fontsize=8)

fig.suptitle(
    "LME partial regression: waveform shape vs connectivity  (confounds removed)\n"
    "x = z-scored feature;  y = LME residuals + predictor contribution",
    fontsize=10, y=1.01,
)
fig.tight_layout()
fig.savefig(OUT / "fig_lme_B_partial_regression.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig_lme_B_partial_regression.pdf", bbox_inches="tight")
plt.close(fig)
print("Figure B saved.")

print(f"\nAll LME figures saved to {OUT}")
