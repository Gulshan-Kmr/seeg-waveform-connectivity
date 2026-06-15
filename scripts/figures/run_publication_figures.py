"""
Publication-ready electrode-level figures covering:
  1. Polished main results (CAR, post, key hypotheses)
  2. Stimulated vs non-stimulated electrodes
  3. Region-level analysis
  4. 1 Hz vs 50 Hz comparison
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
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "Arial", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "figure.dpi": 150,
})

OUT = ROOT / "outputs" / "electrode_level_analysis" / "publication"
OUT.mkdir(parents=True, exist_ok=True)

elec_all = pd.read_csv(ROOT / "outputs" / "electrode_level_analysis" / "electrode_level_summary.csv")
# post-stimulation only
elec = elec_all[elec_all["epoch"] == "post"].copy()

KEY_HYPS = [
    {"name": "asc_pdc",  "x": "asc2desc",    "y": "pdc_asymmetry",
     "xlabel": "Ascent/descent ratio",         "ylabel": "PDC asymmetry (out − in)",
     "title": "Asc/Desc Ratio vs PDC Asymmetry"},
    {"name": "if_tra",   "x": "mean_if",      "y": "mean_tra",
     "xlabel": "Instantaneous frequency (Hz)", "ylabel": "Mean temporal irreversibility",
     "title": "Instantaneous Frequency vs TRA"},
    {"name": "if_imcoh", "x": "mean_if",      "y": "mean_abs_imcoh",
     "xlabel": "Instantaneous frequency (Hz)", "ylabel": "Mean |ImCoh|",
     "title": "Instantaneous Frequency vs ImCoh"},
]

MONTAGE_LABEL = {"bipolar": "Bipolar", "car": "CAR"}
FREQ_COLORS   = {1.0: "#2166AC", 50.0: "#D6604D"}
FREQ_LABELS   = {1.0: "1 Hz", 50.0: "50 Hz"}
STIM_COLORS   = {True: "#C44536", False: "#276FBF"}
STIM_LABELS   = {True: "Stimulated", False: "Non-stimulated"}

REGIONS_KEEP  = ["frontal", "temporal", "insula", "hippocampus",
                 "parietal", "occipital", "amygdala"]
REGION_COLORS = plt.cm.Set2(np.linspace(0, 1, len(REGIONS_KEEP)))
REGION_COLOR_MAP = dict(zip(REGIONS_KEEP, REGION_COLORS))

subjects = sorted(elec["subject"].unique())
SUB_COLORS = dict(zip(subjects, plt.cm.tab10(np.linspace(0, 0.9, len(subjects)))))


# ── helpers ───────────────────────────────────────────────────────────────────
def ols_line(x, y):
    m, b = np.polyfit(x, y, 1)
    xs = np.array([x.min(), x.max()])
    return xs, m * xs + b

def sig_stars(p):
    if not np.isfinite(p): return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

def spearman_summary(x, y):
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(clean) < 5:
        return np.nan, np.nan, 0
    rho, p = stats.spearmanr(clean["x"], clean["y"])
    return rho, p, len(clean)

def annot_box(ax, rho, p, fdr_p, n, loc="upper left"):
    stars  = sig_stars(p)
    fstars = sig_stars(fdr_p)
    txt = (f"rho = {rho:.3f}\n"
           f"p = {p:.4f} {stars}\n"
           f"FDR-p = {fdr_p:.4f} {fstars}\n"
           f"n = {n}")
    x0, y0, ha = (0.03, 0.97, "left") if "left" in loc else (0.97, 0.97, "right")
    va = "top"
    ax.text(x0, y0, txt, transform=ax.transAxes, va=va, ha=ha, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.65", alpha=0.92))

def subject_fdr(data, x_col, y_col):
    rows = []
    for subj, grp in data.groupby("subject"):
        rho, p, n = spearman_summary(grp[x_col], grp[y_col])
        rows.append({"subject": subj, "rho": rho, "p": p, "n": n})
    df = pd.DataFrame(rows).dropna(subset=["p"])
    if len(df) > 1:
        df["p_fdr"] = multipletests(df["p"].values, method="fdr_bh")[1]
    else:
        df["p_fdr"] = df["p"]
    return df

def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Polished main results: 3 hypotheses × 2 montages
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 1: polished main results …")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
panel = iter("abcdef")
for row, montage in enumerate(("bipolar", "car")):
    data_m = elec[elec["montage"] == montage]
    fdr_df = subject_fdr(data_m, KEY_HYPS[0]["x"], KEY_HYPS[0]["y"])  # placeholder
    for col, hyp in enumerate(KEY_HYPS):
        ax = axes[row][col]
        data = data_m[[hyp["x"], hyp["y"], "subject"]].dropna()
        # per-subject scatter
        for subj, grp in data.groupby("subject"):
            ax.scatter(grp[hyp["x"]], grp[hyp["y"]], s=18, alpha=0.40,
                       color=SUB_COLORS[subj], zorder=2)
        # pooled OLS line
        rho, p, n = spearman_summary(data[hyp["x"]], data[hyp["y"]])
        if n >= 5:
            xs, ys = ols_line(data[hyp["x"]].values, data[hyp["y"]].values)
            ax.plot(xs, ys, color="black", lw=2.2, zorder=4)
        # per-subject FDR
        sdf = subject_fdr(data_m, hyp["x"], hyp["y"])
        mean_fdr = sdf["p_fdr"].mean()
        annot_box(ax, rho, p, mean_fdr, n)
        ax.axhline(0, color="0.75", lw=0.8, ls="--")
        ax.set_xlabel(hyp["xlabel"], fontsize=9)
        ax.set_ylabel(hyp["ylabel"] if col == 0 else "", fontsize=9)
        if row == 0:
            ax.set_title(hyp["title"], fontsize=10, fontweight="bold", pad=6)
        letter = next(panel)
        ax.text(-0.10, 1.04, letter, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom")
        if col == 0:
            ax.set_ylabel(f"{MONTAGE_LABEL[montage]}\n{hyp['ylabel']}", fontsize=9)

sub_handles = [Line2D([0],[0], marker="o", color="w",
                markerfacecolor=SUB_COLORS[s], markersize=8,
                label=s.replace("HEJ_","")) for s in subjects]
sub_handles.append(Line2D([0],[0], color="black", lw=2.2, label="OLS trend"))
fig.legend(handles=sub_handles, loc="lower center", ncol=5, fontsize=8,
           frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Electrode-level waveform vs connectivity — post-stimulation",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, "fig1_main_results")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Stimulated vs non-stimulated electrodes
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 2: stimulated vs non-stimulated …")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
panel = iter("abcdef")
for row, montage in enumerate(("bipolar", "car")):
    data_m = elec[elec["montage"] == montage]
    for col, hyp in enumerate(KEY_HYPS):
        ax = axes[row][col]
        rhos, ps = {}, {}
        for is_stim in (False, True):
            grp = data_m[data_m["is_stimulated_channel"] == is_stim][[hyp["x"], hyp["y"]]].dropna()
            if len(grp) >= 5:
                ax.scatter(grp[hyp["x"]], grp[hyp["y"]], s=18,
                           alpha=0.45, color=STIM_COLORS[is_stim],
                           label=STIM_LABELS[is_stim], zorder=2)
                xs, ys = ols_line(grp[hyp["x"]].values, grp[hyp["y"]].values)
                ax.plot(xs, ys, color=STIM_COLORS[is_stim], lw=2.2, zorder=4)
                rhos[is_stim], ps[is_stim], _ = spearman_summary(grp[hyp["x"]], grp[hyp["y"]])
        # difference test
        n_stim    = (data_m["is_stimulated_channel"] == True).sum()
        n_nonstim = (data_m["is_stimulated_channel"] == False).sum()
        txt_lines = []
        for is_stim in (False, True):
            if is_stim in rhos:
                s = sig_stars(ps[is_stim])
                lbl = "Stim" if is_stim else "Non-stim"
                txt_lines.append(f"{lbl}: rho={rhos[is_stim]:.2f} p={ps[is_stim]:.3f}{s}")
        txt_lines.append(f"n stim={n_stim}  non-stim={n_nonstim}")
        ax.text(0.03, 0.97, "\n".join(txt_lines), transform=ax.transAxes,
                va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.65", alpha=0.92))
        ax.axhline(0, color="0.75", lw=0.8, ls="--")
        ax.set_xlabel(hyp["xlabel"], fontsize=9)
        if row == 0:
            ax.set_title(hyp["title"], fontsize=10, fontweight="bold", pad=6)
        letter = next(panel)
        ax.text(-0.10, 1.04, letter, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom")
        if col == 0:
            ax.set_ylabel(f"{MONTAGE_LABEL[montage]}\n{hyp['ylabel']}", fontsize=9)
        if row == 0 and col == 0:
            ax.legend(fontsize=8, frameon=False)

fig.suptitle("Stimulated vs non-stimulated electrodes — post-stimulation",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0.02, 1, 1))
save(fig, "fig2_stimulated_vs_nonstimulated")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Region-level analysis (bar + scatter per region)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 3: region-level analysis …")
# 3a: rho per region per hypothesis (CAR, pooled across subjects)
data_m = elec[elec["montage"] == "car"].copy()
data_m = data_m[data_m["region_family"].isin(REGIONS_KEEP)]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, hyp in zip(axes, KEY_HYPS):
    region_rhos, region_ps, region_ns = [], [], []
    for region in REGIONS_KEEP:
        grp = data_m[data_m["region_family"] == region][[hyp["x"], hyp["y"]]].dropna()
        rho, p, n = spearman_summary(grp[hyp["x"]], grp[hyp["y"]])
        region_rhos.append(rho); region_ps.append(p); region_ns.append(n)
    # FDR across regions
    valid = [np.isfinite(p) for p in region_ps]
    p_fdr = [np.nan] * len(region_ps)
    if sum(valid) > 1:
        corrected = multipletests([p for p, v in zip(region_ps, valid) if v], method="fdr_bh")[1]
        ci = 0
        for i, v in enumerate(valid):
            if v:
                p_fdr[i] = corrected[ci]; ci += 1
    colors = [REGION_COLOR_MAP[r] for r in REGIONS_KEEP]
    bars = ax.barh(REGIONS_KEEP, region_rhos, color=colors, alpha=0.85, edgecolor="0.3", lw=0.6)
    for yi, (rho, p, fdr, n) in enumerate(zip(region_rhos, region_ps, p_fdr, region_ns)):
        if np.isfinite(rho):
            stars = sig_stars(fdr)
            ax.text(rho + (0.01 if rho >= 0 else -0.01), yi,
                    f" {stars} n={n}", va="center", fontsize=7,
                    ha="left" if rho >= 0 else "right")
    ax.axvline(0, color="0.4", lw=1, ls="--")
    ax.set_xlabel("Spearman rho")
    ax.set_title(hyp["title"], fontsize=10, fontweight="bold")
    ax.set_xlim(-0.6, 0.6)

fig.suptitle("Region-level waveform–connectivity associations — CAR, post-stimulation\n"
             "(FDR stars shown per region; n = electrodes in region)",
             fontsize=11, fontweight="bold")
fig.tight_layout()
save(fig, "fig3a_region_bar_CAR")

# 3b: scatter coloured by region (CAR, key hypothesis)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, hyp in zip(axes, KEY_HYPS[:2]):
    data_r = data_m[[hyp["x"], hyp["y"], "region_family"]].dropna()
    for region in REGIONS_KEEP:
        grp = data_r[data_r["region_family"] == region]
        if len(grp):
            ax.scatter(grp[hyp["x"]], grp[hyp["y"]], s=20, alpha=0.55,
                       color=REGION_COLOR_MAP[region], label=region, zorder=2)
    clean = data_r[[hyp["x"], hyp["y"]]].dropna()
    if len(clean) >= 5:
        xs, ys = ols_line(clean[hyp["x"]].values, clean[hyp["y"]].values)
        ax.plot(xs, ys, color="black", lw=2, zorder=4, label="OLS (all regions)")
    rho, p, n = spearman_summary(clean[hyp["x"]], clean[hyp["y"]])
    ax.text(0.03, 0.97, f"rho={rho:.3f}  p={p:.4f} {sig_stars(p)}\nn={n}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.65", alpha=0.92))
    ax.axhline(0, color="0.75", lw=0.8, ls="--")
    ax.set_xlabel(hyp["xlabel"]); ax.set_ylabel(hyp["ylabel"])
    ax.set_title(hyp["title"], fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, frameon=False, loc="lower right", ncol=2)

fig.suptitle("Electrode scatter coloured by brain region — CAR, post-stimulation",
             fontsize=11, fontweight="bold")
fig.tight_layout()
save(fig, "fig3b_region_scatter_CAR")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — 1 Hz vs 50 Hz comparison
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 4: 1 Hz vs 50 Hz …")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
panel = iter("abcdef")
for row, montage in enumerate(("bipolar", "car")):
    data_m = elec[elec["montage"] == montage]
    for col, hyp in enumerate(KEY_HYPS):
        ax = axes[row][col]
        rho_by_freq = {}
        for freq in (1.0, 50.0):
            grp = data_m[data_m["stim_frequency_hz"] == freq][[hyp["x"], hyp["y"]]].dropna()
            if len(grp) >= 5:
                ax.scatter(grp[hyp["x"]], grp[hyp["y"]], s=16, alpha=0.40,
                           color=FREQ_COLORS[freq], label=FREQ_LABELS[freq], zorder=2)
                xs, ys = ols_line(grp[hyp["x"]].values, grp[hyp["y"]].values)
                ax.plot(xs, ys, color=FREQ_COLORS[freq], lw=2.2, zorder=4)
                rho, p, n = spearman_summary(grp[hyp["x"]], grp[hyp["y"]])
                rho_by_freq[freq] = (rho, p, n)
        txt_lines = []
        for freq in (1.0, 50.0):
            if freq in rho_by_freq:
                rho, p, n = rho_by_freq[freq]
                txt_lines.append(f"{FREQ_LABELS[freq]}: rho={rho:.2f} p={p:.3f}{sig_stars(p)} n={n}")
        ax.text(0.03, 0.97, "\n".join(txt_lines), transform=ax.transAxes,
                va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.65", alpha=0.92))
        ax.axhline(0, color="0.75", lw=0.8, ls="--")
        ax.set_xlabel(hyp["xlabel"], fontsize=9)
        if row == 0:
            ax.set_title(hyp["title"], fontsize=10, fontweight="bold", pad=6)
        letter = next(panel)
        ax.text(-0.10, 1.04, letter, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom")
        if col == 0:
            ax.set_ylabel(f"{MONTAGE_LABEL[montage]}\n{hyp['ylabel']}", fontsize=9)
        if row == 0 and col == 0:
            ax.legend(fontsize=8, frameon=False)

fig.suptitle("1 Hz vs 50 Hz stimulation — electrode-level waveform–connectivity associations",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0.02, 1, 1))
save(fig, "fig4_1hz_vs_50hz")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Summary forest plot: all comparisons in one panel
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 5: summary forest plot …")

def compute_rho_group(data, x, y, group_col, group_vals):
    """Spearman rho per subject, then mean ± CI across subjects."""
    out = {}
    for val in group_vals:
        grp = data[data[group_col] == val]
        sdf = subject_fdr(grp, x, y)
        rhos = sdf["rho"].dropna().values
        if len(rhos) == 0:
            out[val] = (np.nan, np.nan, np.nan, np.nan)
            continue
        mean_r = rhos.mean()
        se = rhos.std() / np.sqrt(len(rhos)) if len(rhos) > 1 else 0
        try:
            _, wp = stats.wilcoxon(rhos) if len(rhos) >= 4 else (np.nan, np.nan)
        except Exception:
            wp = np.nan
        out[val] = (mean_r, 1.96 * se, wp, sdf["p_fdr"].mean())
    return out

fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
y_labels, y_groups = [], []

for hyp in KEY_HYPS:
    y_labels.append(f"— {hyp['title']} —")
    y_groups.append("header")
    for montage in ("bipolar", "car"):
        y_labels.append(f"  {MONTAGE_LABEL[montage]}")
        y_groups.append((hyp, montage))

y_pos = list(range(len(y_labels)))

for ax_col, (split_col, split_vals, split_colors, split_labels) in enumerate([
    ("stim_frequency_hz", [1.0, 50.0], [FREQ_COLORS[1.0], FREQ_COLORS[50.0]], ["1 Hz", "50 Hz"]),
    ("is_stimulated_channel", [False, True], [STIM_COLORS[False], STIM_COLORS[True]], ["Non-stim", "Stim"]),
]):
    ax = axes[ax_col]
    for yi, (label, group) in enumerate(zip(y_labels, y_groups)):
        if group == "header":
            ax.text(-0.62, yi, label, va="center", fontsize=9, fontweight="bold", color="0.3")
            continue
        hyp, montage = group
        data_m = elec[elec["montage"] == montage]
        results = compute_rho_group(data_m, hyp["x"], hyp["y"], split_col, split_vals)
        offsets = [-0.12, 0.12]
        for offset, val, color in zip(offsets, split_vals, split_colors):
            mean_r, ci, wp, fdr_p = results[val]
            if not np.isfinite(mean_r):
                continue
            ax.errorbar(mean_r, yi + offset, xerr=ci, fmt="o",
                        color=color, capsize=3, markersize=6, lw=1.8, zorder=3)
            if np.isfinite(wp) and wp < 0.05:
                ax.scatter(mean_r, yi + offset + 0.18, marker="*",
                           s=40, color=color, zorder=4)
    ax.axvline(0, color="0.5", lw=1, ls="--")
    ax.set_xlabel("Mean subject Spearman rho (± 95% CI)")
    ax.set_xlim(-0.55, 0.55)
    split_legend = [Line2D([0],[0], marker="o", color="w", markerfacecolor=c,
                           markersize=8, label=l)
                    for c, l in zip(split_colors, split_labels)]
    split_legend.append(Line2D([0],[0], marker="*", color="0.5",
                               markersize=8, label="Wilcoxon p<0.05", lw=0))
    ax.legend(handles=split_legend, fontsize=8, frameon=False, loc="lower right")
    titles = ["1 Hz vs 50 Hz stimulation", "Stimulated vs Non-stimulated"]
    ax.set_title(titles[ax_col], fontsize=11, fontweight="bold")

axes[0].set_yticks(y_pos, ["" if g == "header" else f"  {MONTAGE_LABEL[g[1]]}" for g in y_groups], fontsize=9)
fig.suptitle("Summary: electrode-level waveform–connectivity associations — post-stimulation\n"
             "(*  = Wilcoxon p < 0.05 across subjects)",
             fontsize=11, fontweight="bold")
fig.tight_layout()
save(fig, "fig5_summary_forest")

print(f"\nAll figures saved to {OUT}")
