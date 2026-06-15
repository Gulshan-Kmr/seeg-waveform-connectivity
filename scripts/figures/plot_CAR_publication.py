"""
CAR-only publication figures covering all three connectivity measures:
ImCoh, PDC asymmetry, TRA — post-stimulation, electrode level.

Figures produced:
  CAR_fig1_main_scatter          — 3 hypotheses, all electrodes, per-subject colour
  CAR_fig2_imcoh                 — ImCoh dedicated panel (3 waveform features vs ImCoh)
  CAR_fig3_pdc                   — PDC dedicated panel (3 waveform features vs PDC)
  CAR_fig4_tra                   — TRA dedicated panel (3 waveform features vs TRA)
  CAR_fig5_stim_vs_nonstim       — Stimulated vs non-stimulated (all 3 connectivity)
  CAR_fig6_region                — Rho per brain region (all 3 connectivity)
  CAR_fig7_1hz_vs_50hz           — Protocol comparison (all 3 connectivity)
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
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
})

OUT = ROOT / "outputs" / "electrode_level_analysis" / "CAR_publication"
OUT.mkdir(parents=True, exist_ok=True)

elec_all = pd.read_csv(ROOT / "outputs" / "electrode_level_analysis" / "electrode_level_summary.csv")
post = elec_all[(elec_all["epoch"] == "post") & (elec_all["montage"] == "car")].copy()

# ── axis definitions ──────────────────────────────────────────────────────────
WAVEFORM_FEATURES = [
    {"col": "mean_if",     "label": "Instantaneous frequency (Hz)"},
    {"col": "asc2desc",    "label": "Ascent/descent ratio"},
    {"col": "peak2trough", "label": "Peak/trough ratio"},
]
CONNECTIVITY = [
    {"col": "mean_abs_imcoh", "label": "Mean |ImCoh|",           "short": "ImCoh"},
    {"col": "pdc_asymmetry",  "label": "PDC asymmetry (out−in)", "short": "PDC"},
    {"col": "mean_tra",       "label": "Mean TRA",               "short": "TRA"},
]

REGIONS_KEEP  = ["frontal","temporal","insula","hippocampus","parietal","occipital","amygdala"]
REGION_COLORS = dict(zip(REGIONS_KEEP, plt.cm.Set2(np.linspace(0,1,len(REGIONS_KEEP)))))
FREQ_COLORS   = {1.0: "#2166AC", 50.0: "#D6604D"}
FREQ_LABELS   = {1.0: "1 Hz", 50.0: "50 Hz"}
STIM_COLORS   = {True: "#C44536", False: "#276FBF"}
STIM_LABELS   = {True: "Stimulated", False: "Non-stimulated"}

subjects = sorted(post["subject"].unique())
SUB_COLORS = dict(zip(subjects, plt.cm.tab10(np.linspace(0, 0.9, len(subjects)))))

# ── helpers ───────────────────────────────────────────────────────────────────
def clean(data, x, y):
    return data[[x, y]].dropna()

def spearman(data, x, y):
    c = clean(data, x, y)
    if len(c) < 5:
        return np.nan, np.nan, 0
    r, p = stats.spearmanr(c[x], c[y])
    return r, p, len(c)

def ols(x_arr, y_arr):
    m, b = np.polyfit(x_arr, y_arr, 1)
    xs = np.array([x_arr.min(), x_arr.max()])
    return xs, m * xs + b

def stars(p):
    if not np.isfinite(p): return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

def fdr_across_subjects(data, x, y):
    rows = []
    for subj, grp in data.groupby("subject"):
        r, p, n = spearman(grp, x, y)
        if np.isfinite(p):
            rows.append({"subject": subj, "rho": r, "p": p, "n": n})
    if not rows:
        return pd.DataFrame(columns=["subject","rho","p","n","p_fdr"])
    df = pd.DataFrame(rows)
    if len(df) > 1:
        df["p_fdr"] = multipletests(df["p"].values, method="fdr_bh")[1]
    else:
        df["p_fdr"] = df["p"]
    return df

def annotate(ax, rho, p, n, loc="upper left"):
    s = stars(p)
    txt = f"rho = {rho:.3f}\np = {p:.4f} {s}\nn = {n} electrodes"
    x0 = 0.03 if "left" in loc else 0.97
    ha = "left" if "left" in loc else "right"
    ax.text(x0, 0.97, txt, transform=ax.transAxes, va="top", ha=ha, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.6", alpha=0.93, lw=0.7))

def scatter_with_ols(ax, data, x, y, color, label=None, alpha=0.45, s=18):
    c = clean(data, x, y)
    if len(c) < 2:
        return
    ax.scatter(c[x], c[y], s=s, alpha=alpha, color=color, label=label, zorder=2, linewidths=0)
    if len(c) >= 5:
        xs, ys = ols(c[x].values, c[y].values)
        ax.plot(xs, ys, color=color, lw=2.2, zorder=4)

def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")

def panel_letter(ax, letter):
    ax.text(-0.12, 1.05, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom", ha="left")


# ═══════════════════════════════════════════════════════════════════════════════
# CAR FIG 1 — Main scatter: 3 waveform × 3 connectivity (9 panels)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 1: main 3x3 scatter …")
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
letters = iter("ABCDEFGHI")
for row, conn in enumerate(CONNECTIVITY):
    for col, wf in enumerate(WAVEFORM_FEATURES):
        ax = axes[row][col]
        # pooled scatter — all electrodes, single colour
        c = clean(post, wf["col"], conn["col"])
        ax.scatter(c[wf["col"]], c[conn["col"]], s=14, alpha=0.35,
                   color="#4C8BB5", linewidths=0, zorder=2)
        rho, p, n = spearman(post, wf["col"], conn["col"])
        if n >= 5:
            xs, ys = ols(c[wf["col"]].values, c[conn["col"]].values)
            ax.plot(xs, ys, color="#C44536", lw=2.2, zorder=4)
        annotate(ax, rho, p, n)
        ax.axhline(0, color="0.75", lw=0.7, ls="--")
        if row == 2:
            ax.set_xlabel(wf["label"], fontsize=9)
        if col == 0:
            ax.set_ylabel(conn["label"], fontsize=9)
        if row == 0:
            ax.set_title(wf["label"], fontsize=9, fontweight="bold", pad=5)
        panel_letter(ax, next(letters))

# row labels on right
for row, conn in enumerate(CONNECTIVITY):
    axes[row][-1].annotate(
        conn["short"], xy=(1.03, 0.5), xycoords="axes fraction",
        fontsize=11, fontweight="bold", rotation=270, va="center", ha="left",
    )

fig.suptitle("Electrode-level waveform vs connectivity (post-stimulation)\n"
             "Rows: connectivity measure   Columns: waveform feature",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0.02, 0.97, 1))
save(fig, "fig1_main_scatter")


# ═══════════════════════════════════════════════════════════════════════════════
# CAR FIG 2 — ImCoh dedicated (3 waveform features vs |ImCoh|)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 2: ImCoh …")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
conn = CONNECTIVITY[0]  # ImCoh
for ax, wf, letter in zip(axes, WAVEFORM_FEATURES, "ABC"):
    c = clean(post, wf["col"], conn["col"])
    ax.scatter(c[wf["col"]], c[conn["col"]], s=14, alpha=0.35,
               color="#4C8BB5", linewidths=0, zorder=2)
    if len(c) >= 5:
        xs, ys = ols(c[wf["col"]].values, c[conn["col"]].values)
        ax.plot(xs, ys, color="#C44536", lw=2.2, zorder=4)
    rho, p, n = spearman(post, wf["col"], conn["col"])
    annotate(ax, rho, p, n)
    ax.set_xlabel(wf["label"], fontsize=9)
    ax.set_ylabel(conn["label"] if letter == "A" else "")
    panel_letter(ax, letter)

fig.suptitle("Imaginary Coherence (|ImCoh|) vs waveform features\nPost-stimulation, electrode level",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "fig2_imcoh")


# ═══════════════════════════════════════════════════════════════════════════════
# CAR FIG 3 — PDC dedicated (3 waveform features vs PDC asymmetry)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 3: PDC …")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
conn = CONNECTIVITY[1]  # PDC
for ax, wf, letter in zip(axes, WAVEFORM_FEATURES, "ABC"):
    c = clean(post, wf["col"], conn["col"])
    ax.scatter(c[wf["col"]], c[conn["col"]], s=14, alpha=0.35,
               color="#4C8BB5", linewidths=0, zorder=2)
    if len(c) >= 5:
        xs, ys = ols(c[wf["col"]].values, c[conn["col"]].values)
        ax.plot(xs, ys, color="#C44536", lw=2.2, zorder=4)
    rho, p, n = spearman(post, wf["col"], conn["col"])
    annotate(ax, rho, p, n)
    ax.axhline(0, color="0.75", lw=0.7, ls="--")
    ax.set_xlabel(wf["label"], fontsize=9)
    ax.set_ylabel(conn["label"] if letter == "A" else "")
    panel_letter(ax, letter)

fig.suptitle("Partial Directed Coherence asymmetry vs waveform features\n"
             "Post-stimulation, electrode level   (positive = net driver)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "fig3_pdc")


# ═══════════════════════════════════════════════════════════════════════════════
# CAR FIG 4 — TRA dedicated (3 waveform features vs TRA)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 4: TRA …")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
conn = CONNECTIVITY[2]  # TRA
for ax, wf, letter in zip(axes, WAVEFORM_FEATURES, "ABC"):
    c = clean(post, wf["col"], conn["col"])
    ax.scatter(c[wf["col"]], c[conn["col"]], s=14, alpha=0.35,
               color="#4C8BB5", linewidths=0, zorder=2)
    if len(c) >= 5:
        xs, ys = ols(c[wf["col"]].values, c[conn["col"]].values)
        ax.plot(xs, ys, color="#C44536", lw=2.2, zorder=4)
    rho, p, n = spearman(post, wf["col"], conn["col"])
    annotate(ax, rho, p, n)
    ax.set_xlabel(wf["label"], fontsize=9)
    ax.set_ylabel(conn["label"] if letter == "A" else "")
    panel_letter(ax, letter)

fig.suptitle("Temporal Irreversibility (TRA) vs waveform features\nPost-stimulation, electrode level",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "fig4_tra")


# ═══════════════════════════════════════════════════════════════════════════════
# CAR FIG 5 — Stimulated vs non-stimulated (3 connectivity measures)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 5: stimulated vs non-stimulated …")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
# use IF as the x-axis (strongest result)
wf = WAVEFORM_FEATURES[0]  # mean_if
for ax, conn, letter in zip(axes, CONNECTIVITY, "ABC"):
    for is_stim in (False, True):
        grp = post[post["is_stimulated_channel"] == is_stim]
        c = clean(grp, wf["col"], conn["col"])
        if len(c) >= 5:
            ax.scatter(c[wf["col"]], c[conn["col"]], s=18, alpha=0.5,
                       color=STIM_COLORS[is_stim],
                       label=STIM_LABELS[is_stim], zorder=2, linewidths=0)
            xs, ys = ols(c[wf["col"]].values, c[conn["col"]].values)
            ax.plot(xs, ys, color=STIM_COLORS[is_stim], lw=2.2, zorder=4)
            rho, p, n = spearman(grp, wf["col"], conn["col"])
            s_str = stars(p)
            lbl = "Stim" if is_stim else "Non-stim"
            yoff = 0.97 if not is_stim else 0.78
            ax.text(0.03, yoff,
                    f"{lbl}: rho={rho:.3f}, p={p:.4f} {s_str}, n={n}",
                    transform=ax.transAxes, va="top", fontsize=8,
                    color=STIM_COLORS[is_stim],
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=STIM_COLORS[is_stim],
                              alpha=0.9, lw=0.8))
    ax.axhline(0, color="0.75", lw=0.7, ls="--")
    ax.set_xlabel(wf["label"], fontsize=9)
    ax.set_ylabel(conn["label"] if letter == "a" else "")
    ax.set_title(conn["short"], fontsize=10, fontweight="bold")
    panel_letter(ax, letter)
    if letter == "A":
        ax.legend(fontsize=8, frameon=False, loc="lower right")

fig.suptitle("Stimulated vs non-stimulated electrodes\n"
             f"X-axis: {wf['label']} — post-stimulation, electrode level",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "fig5_stim_vs_nonstim")


# ═══════════════════════════════════════════════════════════════════════════════
# CAR FIG 6 — Region-level (bar chart: rho per region, 3 connectivity measures)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 6: region-level …")
data_r = post[post["region_family"].isin(REGIONS_KEEP)].copy()
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, conn, letter in zip(axes, CONNECTIVITY, "ABC"):
    region_rhos, region_ps, region_ns = [], [], []
    for region in REGIONS_KEEP:
        grp = data_r[data_r["region_family"] == region]
        # use IF as x for all regions (strongest result)
        rho, p, n = spearman(grp, "mean_if", conn["col"])
        region_rhos.append(rho); region_ps.append(p); region_ns.append(n)
    # FDR across regions
    valid_mask = [np.isfinite(p) for p in region_ps]
    p_fdr = [np.nan] * len(region_ps)
    valid_ps = [p for p, v in zip(region_ps, valid_mask) if v]
    if len(valid_ps) > 1:
        corrected = multipletests(valid_ps, method="fdr_bh")[1]
        ci = 0
        for i, v in enumerate(valid_mask):
            if v: p_fdr[i] = corrected[ci]; ci += 1

    colors = [REGION_COLORS[r] for r in REGIONS_KEEP]
    y_pos = np.arange(len(REGIONS_KEEP))
    bars = ax.barh(y_pos, region_rhos, color=colors, alpha=0.85,
                   edgecolor="0.3", lw=0.6, height=0.6)
    for yi, (rho, p, fdr, n) in enumerate(zip(region_rhos, region_ps, p_fdr, region_ns)):
        if not np.isfinite(rho):
            continue
        s_str = stars(fdr)
        x_off = 0.01 if rho >= 0 else -0.01
        ha = "left" if rho >= 0 else "right"
        ax.text(rho + x_off, yi, f"{s_str}  n={n}", va="center",
                ha=ha, fontsize=8)
    ax.set_yticks(y_pos, REGIONS_KEEP if letter == "A" else [""] * len(REGIONS_KEEP))
    ax.axvline(0, color="0.4", lw=0.8, ls="--")
    ax.set_xlabel("Spearman rho")
    ax.set_title(f"IF vs {conn['short']}", fontsize=10, fontweight="bold")
    ax.set_xlim(-0.55, 0.55)
    panel_letter(ax, letter)

fig.suptitle("Waveform–connectivity association by brain region\n"
             "X: Instantaneous frequency vs each connectivity measure — post-stimulation\n"
             "(* = FDR-corrected p<0.05 across regions)",
             fontsize=11, fontweight="bold")
fig.tight_layout()
save(fig, "fig6_region")


# ═══════════════════════════════════════════════════════════════════════════════
# CAR FIG 7 — 1 Hz vs 50 Hz (3 connectivity measures)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 7: 1 Hz vs 50 Hz …")
wf = WAVEFORM_FEATURES[0]  # mean_if
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, conn, letter in zip(axes, CONNECTIVITY, "ABC"):
    for freq in (1.0, 50.0):
        grp = post[post["stim_frequency_hz"] == freq]
        c = clean(grp, wf["col"], conn["col"])
        if len(c) >= 5:
            ax.scatter(c[wf["col"]], c[conn["col"]], s=18, alpha=0.45,
                       color=FREQ_COLORS[freq], label=FREQ_LABELS[freq],
                       zorder=2, linewidths=0)
            xs, ys = ols(c[wf["col"]].values, c[conn["col"]].values)
            ax.plot(xs, ys, color=FREQ_COLORS[freq], lw=2.2, zorder=4)
            rho, p, n = spearman(grp, wf["col"], conn["col"])
            s_str = stars(p)
            yoff = 0.97 if freq == 1.0 else 0.78
            ax.text(0.03, yoff,
                    f"{FREQ_LABELS[freq]}: rho={rho:.3f}, p={p:.4f} {s_str}, n={n}",
                    transform=ax.transAxes, va="top", fontsize=8,
                    color=FREQ_COLORS[freq],
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=FREQ_COLORS[freq], alpha=0.9, lw=0.8))
    ax.axhline(0, color="0.75", lw=0.7, ls="--")
    ax.set_xlabel(wf["label"], fontsize=9)
    ax.set_ylabel(conn["label"] if letter == "a" else "")
    ax.set_title(conn["short"], fontsize=10, fontweight="bold")
    panel_letter(ax, letter)
    if letter == "A":
        ax.legend(fontsize=8, frameon=False, loc="lower right")

fig.suptitle("1 Hz vs 50 Hz stimulation\n"
             f"X-axis: {wf['label']} — post-stimulation, electrode level",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "fig7_1hz_vs_50hz")

print(f"\nAll CAR publication figures → {OUT}")
