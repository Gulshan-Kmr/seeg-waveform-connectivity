"""
Figures for electrode-level time-window analysis.
Reads outputs/electrode_level_analysis/electrode_time_window_spearman.csv
produced by run_electrode_time_windows.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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

OUT  = ROOT / "outputs" / "electrode_level_analysis"
elec = pd.read_csv(OUT / "electrode_time_windows.csv")
corr = pd.read_csv(OUT / "electrode_time_window_spearman.csv")

WIN_ORDER  = ["0-10s", "10-20s", "20-28s"]
WIN_COLORS = ["#2166AC", "#4DAC26", "#D6604D"]
WIN_LABELS = {"0-10s": "First 10 s", "10-20s": "Middle 10 s", "20-28s": "Last 8 s"}

KEY_HYPS = [
    {"name": "asc_pdc",  "x": "asc2desc",   "y": "pdc_asymmetry",
     "xlabel": "Ascent/descent ratio",       "ylabel": "PDC asymmetry (out−in)",
     "title": "Asc/Desc Ratio  vs  PDC Asymmetry"},
    {"name": "if_tra",   "x": "mean_if",     "y": "mean_tra",
     "xlabel": "Instantaneous frequency (Hz)", "ylabel": "Mean TRA",
     "title": "Instantaneous Frequency  vs  TRA"},
]
ALL_HYPS = [
    {"name": "asc_pdc",   "label": "Asc/Desc vs PDC"},
    {"name": "if_tra",    "label": "IF vs TRA"},
    {"name": "if_imcoh",  "label": "IF vs ImCoh"},
    {"name": "asc_imcoh", "label": "Asc/Desc vs ImCoh"},
    {"name": "pt_tra",    "label": "Peak/Trough vs TRA"},
]

subjects = sorted(elec["subject"].unique())
sub_palette = plt.cm.tab10(np.linspace(0, 0.9, len(subjects)))
sub_color   = dict(zip(subjects, sub_palette))


def ols_line(x, y):
    m, b = np.polyfit(x, y, 1)
    xs = np.array([x.min(), x.max()])
    return xs, m * xs + b


def sig_stars(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


# ── FIGURE A: Scatter — two key results × three time windows (CAR) ───────────
fig, axes = plt.subplots(len(KEY_HYPS), 3, figsize=(15, 9), squeeze=False)
for row, hyp in enumerate(KEY_HYPS):
    for col, win in enumerate(WIN_ORDER):
        ax = axes[row][col]
        data = elec[(elec["montage"] == "car") & (elec["time_window"] == win)]
        for subj in subjects:
            grp = data[data["subject"] == subj][[hyp["x"], hyp["y"]]].dropna()
            if len(grp) > 0:
                ax.scatter(grp[hyp["x"]], grp[hyp["y"]], s=16, alpha=0.45,
                           color=sub_color[subj], label=subj.replace("HEJ_", ""), zorder=2)
        clean = data[[hyp["x"], hyp["y"]]].dropna()
        if len(clean) >= 5:
            xs, ys = ols_line(clean[hyp["x"]].values, clean[hyp["y"]].values)
            rho, p_val = stats.spearmanr(clean[hyp["x"]], clean[hyp["y"]])
            ax.plot(xs, ys, color="black", lw=2, zorder=4)
            rows_fdr = corr[(corr["montage"] == "car") & (corr["time_window"] == win)
                            & (corr["hypothesis"] == hyp["name"])]
            mean_fdr = rows_fdr["p_fdr"].mean()
            txt = (f"rho = {rho:.3f}\n"
                   f"p = {p_val:.4f} ({sig_stars(p_val)})\n"
                   f"FDR-p = {mean_fdr:.4f} ({sig_stars(mean_fdr)})\n"
                   f"n = {len(clean)} electrodes")
            ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.9))
        ax.axhline(0, color="0.7", lw=0.8, ls="--")
        ax.set_title(WIN_LABELS[win], fontsize=10, fontweight="bold",
                     color=WIN_COLORS[col])
        if col == 0:
            ax.set_ylabel(f"{hyp['title']}\n{hyp['ylabel']}", fontsize=9)
        if row == len(KEY_HYPS) - 1:
            ax.set_xlabel(hyp["xlabel"], fontsize=9)

handles = ([Line2D([0],[0], marker="o", color="w", markerfacecolor=sub_color[s],
                   markersize=8, label=s.replace("HEJ_","")) for s in subjects]
           + [Line2D([0],[0], color="black", lw=2, label="OLS trend (pooled)")])
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("CAR — electrode-level waveform vs connectivity\nacross three post-stimulation time windows",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT / "figA_time_window_scatter_CAR.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "figA_time_window_scatter_CAR.pdf", bbox_inches="tight")
plt.close(fig)
print("Figure A saved")


# ── FIGURE B: rho trajectory across time windows (line plot, all hypotheses) ─
for montage in ("bipolar", "car"):
    fig, ax = plt.subplots(figsize=(9, 5))
    hyp_colors = plt.cm.tab10(np.linspace(0, 0.9, len(ALL_HYPS)))
    for hyp, color in zip(ALL_HYPS, hyp_colors):
        mean_rhos, cis, win_labels = [], [], []
        for win in WIN_ORDER:
            rows = corr[(corr["montage"] == montage) & (corr["time_window"] == win)
                        & (corr["hypothesis"] == hyp["name"])]["rho"].dropna()
            if len(rows) == 0:
                continue
            mean_rho = rows.mean()
            se = rows.std() / np.sqrt(len(rows)) if len(rows) > 1 else 0
            mean_rhos.append(mean_rho)
            cis.append(1.96 * se)
            win_labels.append(win)
        x_pos = np.arange(len(win_labels))
        ax.plot(x_pos, mean_rhos, marker="o", color=color, lw=2, label=hyp["label"])
        ax.fill_between(x_pos,
                        np.array(mean_rhos) - np.array(cis),
                        np.array(mean_rhos) + np.array(cis),
                        color=color, alpha=0.15)
    ax.axhline(0, color="0.5", lw=1, ls="--")
    ax.set_xticks(range(len(WIN_ORDER)), [WIN_LABELS[w] for w in WIN_ORDER])
    ax.set_xlabel("Post-stimulation time window")
    ax.set_ylabel("Mean Spearman rho across subjects")
    ax.set_title(f"Electrode-level waveform–connectivity rho over time — {montage.upper()}\n"
                 f"(mean ± 95% CI across subjects)", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / f"figB_rho_trajectory_{montage}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"figB_rho_trajectory_{montage}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure B ({montage}) saved")


# ── FIGURE C: Forest plot per time window (3 panels per montage) ─────────────
for montage in ("bipolar", "car"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    for ax, win, win_color in zip(axes, WIN_ORDER, WIN_COLORS):
        sub = corr[(corr["montage"] == montage) & (corr["time_window"] == win)]
        for yi, hyp in enumerate(ALL_HYPS):
            rows = sub[sub["hypothesis"] == hyp["name"]]["rho"].dropna()
            for xi, rho in enumerate(rows):
                offset = (xi - len(rows)/2 + 0.5) * 0.07
                ax.scatter(rho, yi + offset, s=40, color=win_color, alpha=0.8, zorder=3)
            if len(rows) >= 2:
                mean_rho = rows.mean()
                se = rows.std() / np.sqrt(len(rows))
                ax.errorbar(mean_rho, yi, xerr=1.96*se, fmt="D", color="black",
                            markersize=6, lw=2, zorder=4, capsize=4)
                try:
                    _, wil_p = stats.wilcoxon(rows.values)
                except Exception:
                    wil_p = np.nan
                mean_fdr = sub[sub["hypothesis"] == hyp["name"]]["p_fdr"].mean()
                ax.text(0.62, yi,
                        f"p={wil_p:.3f}{sig_stars(wil_p)}\nFDR={mean_fdr:.3f}{sig_stars(mean_fdr)}",
                        va="center", fontsize=7, transform=ax.get_yaxis_transform(),
                        bbox=dict(boxstyle="round,pad=0.2", fc="#f9f9f9", ec="0.8", alpha=0.9))
        ax.axvline(0, color="0.5", lw=1, ls="--")
        ax.set_yticks(range(len(ALL_HYPS)), [h["label"] for h in ALL_HYPS])
        ax.set_xlabel("Spearman rho")
        ax.set_title(WIN_LABELS[win], fontweight="bold", color=win_color, fontsize=11)
        ax.set_xlim(-0.65, 0.95)
    fig.suptitle(f"Electrode-level associations by post-stimulation window — {montage.upper()}\n"
                 f"(dots = subjects; diamond = mean; Wilcoxon & FDR-p shown)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / f"figC_forest_time_windows_{montage}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"figC_forest_time_windows_{montage}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure C ({montage}) saved")


# ── FIGURE D: Heatmap rho — hypothesis × time_window (per montage) ───────────
for montage in ("bipolar", "car"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, agg_fn, title in zip(axes,
                                  [lambda r: r.mean(), lambda r: (r < 0.05).mean()],
                                  ["Mean Spearman rho", "Fraction subjects p<0.05"]):
        mat = np.full((len(ALL_HYPS), len(WIN_ORDER)), np.nan)
        for ri, hyp in enumerate(ALL_HYPS):
            for ci, win in enumerate(WIN_ORDER):
                rows = corr[(corr["montage"] == montage) & (corr["time_window"] == win)
                            & (corr["hypothesis"] == hyp["name"])]
                if not rows.empty:
                    mat[ri, ci] = agg_fn(rows["rho"] if title.startswith("Mean") else rows["p"])
        vmax = 0.4 if title.startswith("Mean") else 1.0
        vmin = -0.4 if title.startswith("Mean") else 0.0
        cmap = "RdBu_r" if title.startswith("Mean") else "YlOrRd"
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(WIN_ORDER)), [WIN_LABELS[w] for w in WIN_ORDER], rotation=15)
        ax.set_yticks(range(len(ALL_HYPS)), [h["label"] for h in ALL_HYPS])
        ax.set_title(title, fontweight="bold")
        for ri in range(mat.shape[0]):
            for ci in range(mat.shape[1]):
                if np.isfinite(mat[ri, ci]):
                    ax.text(ci, ri, f"{mat[ri,ci]:.2f}", ha="center", va="center",
                            fontsize=8, color="white" if abs(mat[ri,ci]) > vmax*0.6 else "black")
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"Time-window summary heatmap — {montage.upper()}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / f"figD_heatmap_{montage}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"figD_heatmap_{montage}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure D ({montage}) saved")

print("\nAll time-window figures saved to:", OUT)
