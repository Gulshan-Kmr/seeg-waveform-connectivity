"""
Figure 1 — EMD pipeline illustration (Quinn-style).

Panel A : Raw sEEG signal + 6 IMFs stacked, IMF4 highlighted (post-stimulation).
Panel B : Single representative cycle with peak / trough / descending
          zero-crossing annotated, illustrating asc2desc and peak2trough.
Panel C : Phase-aligned instantaneous-frequency (PA-IF) heatmap across all
          post-stimulation cycles (Quinn et al. 2021, Fig 5 style).
Panel D : Distribution of the three waveform features across all CAR
          post-stimulation electrodes (n = 995).

Representative electrode: O5, Subject 16, 1 Hz stimulation, 87 post cycles.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

plt.rcParams.update({
    "font.family": "Arial",
    "font.size"  : 10,
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "outputs" / "publication_figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── electrode choice ──────────────────────────────────────────────────────────
FILE_NAME = "stims_HEJ_Subject16_F=1_I=3"
CHANNEL   = "O5"

NPZ  = ROOT / "quinn_batch_outputs" / FILE_NAME / f"{CHANNEL}_IMF4_outputs.npz"
CYCS = ROOT / "quinn_batch_outputs" / FILE_NAME / f"{CHANNEL}_IMF4_post_cycles.csv"

data  = np.load(NPZ, allow_pickle=True)
cyc_df = pd.read_csv(CYCS)
elec_df = pd.read_csv(ROOT / "outputs" / "electrode_level_analysis" /
                       "electrode_level_summary.csv")
pop = elec_df[(elec_df["montage"] == "car") & (elec_df["epoch"] == "post")].copy()

# ── unpack arrays ─────────────────────────────────────────────────────────────
t_post   = data["t_post"]
x_post   = data["x_post"]
imf_post = data["imf_post"]          # (N, 6)
IF_post  = data["IF_post"]           # (N, 6)
IP_post  = data["IP_post"]           # (N, 6)  instantaneous phase [-π, π]
pa_post  = data["pa_post"]           # (48 phase-bins, n_cycles)  PA-IF

fs       = 1.0 / (t_post[1] - t_post[0])   # sampling rate
n_bins   = pa_post.shape[0]
n_cycles = pa_post.shape[1]

# ── helpers ───────────────────────────────────────────────────────────────────
IMF4_IDX = 3     # zero-indexed

BLUE  = "#4C8BB5"
GREY  = "#888888"
RED   = "#C44536"
GREEN = "#2E8B57"
GOLD  = "#E8A838"

def panel_label(ax, letter, x=-0.10, y=1.05):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")

# ════════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ════════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 13))
gs  = gridspec.GridSpec(
    3, 3,
    figure=fig,
    height_ratios=[2.2, 1.6, 1.2],
    hspace=0.45, wspace=0.38,
)
ax_A  = fig.add_subplot(gs[0, :])          # full-width signal panel
ax_B  = fig.add_subplot(gs[1, 0])          # single cycle
ax_C  = fig.add_subplot(gs[1, 1:])         # PA-IF heatmap
ax_D1 = fig.add_subplot(gs[2, 0])          # IF distribution
ax_D2 = fig.add_subplot(gs[2, 1])          # asc2desc distribution
ax_D3 = fig.add_subplot(gs[2, 2])          # peak2trough distribution

# ════════════════════════════════════════════════════════════════════════════════
# PANEL A — raw signal + IMFs stacked
# ════════════════════════════════════════════════════════════════════════════════
SHOW_SEC = 6.0
n_show   = int(SHOW_SEC * fs)
t_s      = t_post[:n_show]
x_s      = x_post[:n_show]

# Signals to stack (top → bottom): raw, IMF1, IMF2, IMF3, IMF4, IMF5, IMF6
labels   = ["Raw", "IMF 1", "IMF 2", "IMF 3", "IMF 4 (α)", "IMF 5", "IMF 6"]
signals  = [x_s] + [imf_post[:n_show, i] for i in range(6)]
n_traces = len(signals)

# Normalise each trace independently to fixed height, then stack with uniform gap
TRACE_H  = 1.0   # each trace occupies ±0.5 in normalised units
GAP      = 0.35  # gap between trace tops/bottoms

norm_sigs = []
for s in signals:
    rng = np.ptp(s)
    norm_sigs.append(s / rng if rng > 0 else s)

offsets = [0.0]
for i in range(1, n_traces):
    offsets.append(offsets[-1] - (TRACE_H + GAP))

for i, (sig, off, lab) in enumerate(zip(norm_sigs, offsets, labels)):
    color = BLUE if i == 4 else ("0.15" if i == 0 else GREY)
    lw    = 1.6  if i == 4 else (1.1 if i == 0 else 0.9)
    zord  = 3    if i == 4 else 2
    ax_A.plot(t_s, sig + off, color=color, lw=lw, zorder=zord)
    ax_A.text(t_s[0] - 0.18, off, lab, ha="right", va="center",
              fontsize=8.5, color=color if i in (0, 4) else GREY,
              fontweight="bold" if i in (0, 4) else "normal")

# Scale bar
bar_len  = 1.0   # 1 second
bar_y    = offsets[-1] - 0.55
ax_A.plot([t_s[-1] - bar_len, t_s[-1]], [bar_y, bar_y],
          color="k", lw=2, solid_capstyle="butt", clip_on=False)
ax_A.text(t_s[-1] - bar_len/2, bar_y - 0.25, "1 s",
          ha="center", va="top", fontsize=8)

ax_A.set_xlim(t_s[0], t_s[-1])
ax_A.set_xlabel("Time (s)", fontsize=10)
ax_A.set_yticks([])
ax_A.spines["left"].set_visible(False)
# Highlight IMF4 row with light shading
imf4_off = offsets[4]
ax_A.axhspan(imf4_off - 0.55, imf4_off + 0.55,
             color=BLUE, alpha=0.07, zorder=0)
panel_label(ax_A, "A")

# ════════════════════════════════════════════════════════════════════════════════
# PANEL B — single cycle with feature annotation
# ════════════════════════════════════════════════════════════════════════════════
phase4 = IP_post[:, IMF4_IDX]   # instantaneous phase, [0, 2π]
imf4   = imf_post[:, IMF4_IDX]

# Phase wrap-arounds = ascending zero-crossings (phase jumps from ~2π back to ~0)
azc = np.where(np.diff(phase4) < -np.pi)[0]

# Pick the cycle closest to the mean asc2desc of the electrode
good_cyc = cyc_df[cyc_df["is_good"] == 1].reset_index(drop=True)
target_a2d = good_cyc["asc2desc"].median()

best_k = None
best_diff = np.inf
for k in range(len(azc) - 1):
    # find which cycle entry matches this region
    mid_sample = (azc[k] + azc[k+1]) // 2
    idx_in_df  = int(np.digitize(k, np.arange(len(good_cyc)))) - 1
    if 0 <= idx_in_df < len(good_cyc):
        diff = abs(good_cyc.loc[idx_in_df, "asc2desc"] - target_a2d)
        if diff < best_diff and (azc[k+1] - azc[k]) > 10:
            best_diff = diff
            best_k    = k

k = best_k if best_k is not None else (len(azc) // 2)
seg   = imf4[azc[k] : azc[k+1] + 1]
T     = len(seg)
x_cyc = np.linspace(0, 1, T)

# Normalise cycle to [-1, 1]
seg_n = (seg - seg.min()) / (seg.max() - seg.min()) * 2 - 1

# Key points
i_peak = int(np.argmax(seg_n))
i_tro  = int(np.argmin(seg_n))
# Descending zero crossing: first zero crossing after peak where signal goes negative
desc_zc = None
for j in range(i_peak + 1, T - 1):
    if seg_n[j] >= 0 and seg_n[j+1] < 0:
        desc_zc = j
        break
if desc_zc is None:
    desc_zc = i_peak + (i_tro - i_peak) // 2

# Plot background guide cycles (thin grey)
for kk in range(max(0, k-3), min(len(azc)-1, k+3)):
    if kk == k:
        continue
    s = imf4[azc[kk]:azc[kk+1]+1]
    if len(s) < 5:
        continue
    s_n = (s - s.min()) / (s.max() - s.min() + 1e-14) * 2 - 1
    ax_B.plot(np.linspace(0, 1, len(s_n)), s_n, color=GREY, lw=0.6, alpha=0.4)

# Main cycle
ax_B.plot(x_cyc, seg_n, color=BLUE, lw=2, zorder=3)

# Key landmarks
ax_B.scatter([x_cyc[i_peak]], [seg_n[i_peak]], s=60, color=RED,   zorder=5, label="Peak")
ax_B.scatter([x_cyc[i_tro]],  [seg_n[i_tro]],  s=60, color=GOLD,  zorder=5, label="Trough")
ax_B.scatter([x_cyc[desc_zc]],[seg_n[desc_zc]], s=60, color=GREEN, zorder=5, label="Desc. zero")

# asc2desc annotation: double-headed bracket
asc_frac = (i_peak + (T - i_tro)) / T
ax_B.annotate("", xy=(1.0, -1.32), xytext=(0.0, -1.32),
              arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5))
ax_B.text(0.5 * x_cyc[i_peak], -1.52,
          f"asc2desc = {asc_frac:.3f}", ha="center", fontsize=7.5, color=RED)

# peak2trough annotation: descending ZC position
pt_frac = x_cyc[desc_zc]
ax_B.annotate("", xy=(pt_frac, -0.18), xytext=(0.0, -0.18),
              arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.5))
ax_B.text(pt_frac / 2, -0.04,
          f"peak2trough\n= {pt_frac:.3f}", ha="center", fontsize=7, color=GREEN)

ax_B.axhline(0, color="0.75", lw=0.8, ls="--", zorder=1)
ax_B.set_xlim(-0.02, 1.05)
ax_B.set_ylim(-1.75, 1.25)
ax_B.set_xlabel("Cycle phase (0 → 1)", fontsize=9)
ax_B.set_ylabel("Normalised amplitude", fontsize=9)
ax_B.set_title("Representative cycle", fontsize=9, pad=4)
ax_B.legend(fontsize=7.5, frameon=False, loc="upper right",
            handlelength=0.8, handletextpad=0.4)
panel_label(ax_B, "B")

# ════════════════════════════════════════════════════════════════════════════════
# PANEL C — phase-aligned IF heatmap
# ════════════════════════════════════════════════════════════════════════════════
# pa_post: (n_phase_bins, n_cycles) — IF in Hz
# Show cycles on y-axis, phase on x-axis
phase_bins = np.linspace(0, 360, n_bins + 1)   # degrees for x-axis

# Clip extreme values for better colour range
vmin, vmax = np.percentile(pa_post, [2, 98])

im = ax_C.imshow(
    pa_post.T,                   # (n_cycles, n_phase_bins)
    aspect="auto",
    origin="lower",
    cmap="RdBu_r",
    vmin=vmin, vmax=vmax,
    extent=[0, 360, 0.5, n_cycles + 0.5],
    interpolation="nearest",
)
# Mean IF profile as white dashed line — scaled to cycle-index axis
mean_if_profile = pa_post.mean(axis=1)
# Rescale mean IF to cycle-index axis for overlay
if_range = vmax - vmin
cycle_range = n_cycles
mean_scaled = (mean_if_profile - vmin) / if_range * cycle_range * 0.35 + n_cycles * 0.6
ax_C.plot(np.linspace(0, 360, n_bins), mean_scaled,
          color="white", lw=2, ls="--", zorder=5, label="Mean IF")
ax_C.legend(fontsize=7.5, frameon=True, loc="upper right",
            facecolor="0.2", edgecolor="none", labelcolor="white")

cbar = fig.colorbar(im, ax=ax_C, pad=0.02, shrink=0.9)
cbar.set_label("Instantaneous\nfrequency (Hz)", fontsize=8)

ax_C.set_xlabel("Phase (°)", fontsize=9)
ax_C.set_ylabel("Cycle index", fontsize=9)
ax_C.set_xticks([0, 90, 180, 270, 360])
ax_C.set_title("Phase-aligned IF profile across cycles", fontsize=9, pad=4)
panel_label(ax_C, "C", x=-0.06)

# ════════════════════════════════════════════════════════════════════════════════
# PANEL D — feature distributions across all electrodes
# ════════════════════════════════════════════════════════════════════════════════
FEAT_CONFIGS = [
    ("mean_if",     ax_D1, "Instantaneous\nfrequency (Hz)", BLUE,  (6, 14)),
    ("asc2desc",    ax_D2, "Ascent/descent\nratio",          RED,   (0.46, 0.56)),
    ("peak2trough", ax_D3, "Peak/trough\nratio",            GREEN, (0.46, 0.52)),
]

for feat, ax, xlabel, color, xlim in FEAT_CONFIGS:
    vals = pop[feat].dropna().values
    ax.hist(vals, bins=35, color=color, alpha=0.75, edgecolor="white", lw=0.4)
    ax.axvline(np.median(vals), color="k", lw=1.5, ls="--", zorder=5)
    ax.text(0.97, 0.95, f"median\n{np.median(vals):.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.9))
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Electrode count" if ax is ax_D1 else "", fontsize=9)
    ax.set_xlim(xlim)
    ax.set_title(f"n = {len(vals):,} electrodes", fontsize=8, pad=3)

panel_label(ax_D1, "D")

# ════════════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════════════
fig.suptitle(
    "EMD waveform analysis pipeline — representative sEEG electrode",
    fontsize=11, y=0.995,
)

fig.savefig(OUT / "fig1_emd_pipeline.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig1_emd_pipeline.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Figure 1 saved to {OUT}")
