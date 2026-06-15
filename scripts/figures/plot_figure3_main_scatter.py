"""
Figure 3 — Waveform shape predicts network connectivity (main result).

3 × 3 grid: rows = connectivity measure (ImCoh, PDC, TRA),
            cols = waveform feature (IF, asc2desc, peak2trough).

Each panel shows:
  • Pooled scatter (all post-stimulation electrodes, n ≈ 1009)
  • OLS trend line (visualisation aid)
  • Stats box: pooled Spearman ρ · within-subject median ρ ·
               Wilcoxon signed-rank p across n=8 subjects
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
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "font.family": "Arial",
    "font.size"  : 10,
    "axes.linewidth"   : 0.8,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
})

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "outputs" / "publication_figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#4C8BB5"
RED  = "#C44536"

# ── load data ──────────────────────────────────────────────────────────────────
elec_all = pd.read_csv(
    ROOT / "outputs" / "electrode_level_analysis" / "electrode_level_summary.csv"
)
post = elec_all[
    (elec_all["epoch"] == "post") & (elec_all["montage"] == "car")
].copy()

# ── axis definitions ───────────────────────────────────────────────────────────
WAVEFORM = [
    {"col": "mean_if",     "label": "Instantaneous\nfrequency (Hz)"},
    {"col": "asc2desc",    "label": "Ascent/descent\nratio"},
    {"col": "peak2trough", "label": "Peak/trough\nratio"},
]
CONNECTIVITY = [
    {"col": "mean_abs_imcoh", "label": "Mean |ImCoh|",           "short": "ImCoh"},
    {"col": "pdc_asymmetry",  "label": "PDC asymmetry (out−in)", "short": "PDC"},
    {"col": "mean_tra",       "label": "Mean TRA",               "short": "TRA"},
]

# ── helpers ────────────────────────────────────────────────────────────────────
def clean(df, x, y):
    return df[[x, y]].dropna()

def spearman_rho(df, x, y):
    c = clean(df, x, y)
    if len(c) < 5:
        return np.nan, np.nan, 0
    r, p = stats.spearmanr(c[x], c[y])
    return r, p, len(c)

def ols_line(x_arr, y_arr):
    m, b = np.polyfit(x_arr, y_arr, 1)
    xs = np.array([x_arr.min(), x_arr.max()])
    return xs, m * xs + b

def stars(p):
    if not np.isfinite(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def within_subject_stats(df, x, y):
    """Spearman ρ per subject → Wilcoxon signed-rank vs 0."""
    rhos = []
    for _, grp in df.groupby("subject"):
        r, p, n = spearman_rho(grp, x, y)
        if np.isfinite(r):
            rhos.append(r)
    if len(rhos) < 3:
        return np.nan, np.nan, 0, 0
    rhos = np.array(rhos)
    _, p_wilcox = stats.wilcoxon(rhos, alternative="two-sided")
    n_sig = int(np.sum(rhos > 0))   # count subjects with positive ρ
    return float(np.median(rhos)), p_wilcox, len(rhos), n_sig

def panel_label(ax, letter, x=-0.13, y=1.05):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")

# ── figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 12))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

panel_letters = iter("ABCDEFGHI")

for row, conn in enumerate(CONNECTIVITY):
    for col, wf in enumerate(WAVEFORM):
        ax = fig.add_subplot(gs[row, col])
        letter = next(panel_letters)

        c = clean(post, wf["col"], conn["col"])

        # ── scatter ──
        ax.scatter(c[wf["col"]], c[conn["col"]],
                   s=12, alpha=0.30, color=BLUE,
                   linewidths=0, zorder=2)

        # ── OLS trend line ──
        if len(c) >= 5:
            xs, ys = ols_line(c[wf["col"]].values, c[conn["col"]].values)
            ax.plot(xs, ys, color=RED, lw=2.0, zorder=4)

        # ── pooled Spearman ──
        rho_pool, p_pool, n_pool = spearman_rho(post, wf["col"], conn["col"])

        # ── within-subject stats ──
        med_rho, p_wilcox, n_subj, n_pos = within_subject_stats(
            post, wf["col"], conn["col"]
        )

        # ── stats annotation ──
        txt = (
            f"ρ = {rho_pool:.3f} (pooled)\n"
            f"med ρ = {med_rho:.3f} (within-subject)\n"
            f"Wilcoxon p = {p_wilcox:.3f} {stars(p_wilcox)}\n"
            f"n = {n_pool} electrodes"
        )
        ax.text(0.03, 0.97, txt,
                transform=ax.transAxes, va="top", ha="left",
                fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec="0.65", alpha=0.93, lw=0.7))

        # ── zero reference for PDC ──
        if conn["col"] == "pdc_asymmetry":
            ax.axhline(0, color="0.75", lw=0.7, ls="--", zorder=1)

        # ── axis labels (only outer edges) ──
        if row == 2:
            ax.set_xlabel(wf["label"], fontsize=9)
        else:
            ax.set_xlabel("")
        if col == 0:
            ax.set_ylabel(conn["label"], fontsize=9)
        else:
            ax.set_ylabel("")

        # ── column title on top row ──
        if row == 0:
            ax.set_title(wf["label"].replace("\n", " "),
                         fontsize=9, fontweight="bold", pad=6)

        panel_label(ax, letter)

# ── row labels (connectivity measure) on right side ──
for row, conn in enumerate(CONNECTIVITY):
    ax_last = fig.axes[row * 3 + 2]
    ax_last.annotate(
        conn["short"],
        xy=(1.04, 0.5), xycoords="axes fraction",
        fontsize=11, fontweight="bold",
        rotation=270, va="center", ha="left",
    )

fig.suptitle(
    "Waveform shape predicts network connectivity — post-stimulation electrodes\n"
    "Rows: connectivity measure   ·   Columns: waveform feature",
    fontsize=11, y=1.01,
)

fig.savefig(OUT / "fig3_main_scatter.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig3_main_scatter.pdf",           bbox_inches="tight")
plt.close(fig)
print(f"Figure 3 saved to {OUT}")
