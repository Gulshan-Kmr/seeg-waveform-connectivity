"""
Figure 2 — Waveform shape change after brain stimulation (grand average).

Panel A : Grand mean normalised waveform pre vs post across ALL stimulated
          electrodes (n = 42, all subjects). Mean ± SEM across electrodes.
Panel B : Same for ALL non-stimulated electrodes (n = 1248). Shows no change,
          confirming specificity of the stimulation effect.
Panel C : Δasc2desc (post − pre) violin: stimulated vs non-stimulated.
Panel D : Δpeak2trough (post − pre) violin: stimulated vs non-stimulated.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "font.family": "Arial",
    "font.size"  : 10,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "outputs" / "publication_figures"
OUT.mkdir(parents=True, exist_ok=True)

PRE_COLOR  = "#4C8BB5"
POST_COLOR = "#C44536"

def panel_label(ax, letter, x=-0.13, y=1.05):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

# ── load master ───────────────────────────────────────────────────────────────
master = pd.read_csv(ROOT / "quinn_batch_outputs" / "MASTER_all_channels_IMF4.csv")

def npz_path(r):
    return ROOT / "quinn_batch_outputs" / r["file_name"] / f"{r['channel_name']}_IMF4_outputs.npz"

stim    = master[master["is_stimulated"] == True].copy()
nonstim = master[master["is_stimulated"] == False].copy()

stim["npz"]    = stim.apply(npz_path, axis=1)
nonstim["npz"] = nonstim.apply(npz_path, axis=1)
stim    = stim[stim["npz"].apply(lambda p: p.exists()) &
               (stim["pre_n_cycles"] >= 10) & (stim["post_n_cycles"] >= 10)].reset_index(drop=True)
nonstim = nonstim[nonstim["npz"].apply(lambda p: p.exists()) &
                  (nonstim["pre_n_cycles"] >= 10) & (nonstim["post_n_cycles"] >= 10)].reset_index(drop=True)

print(f"Stimulated electrodes: {len(stim)}")
print(f"Non-stimulated electrodes: {len(nonstim)}")

# ── compute grand average waveforms ──────────────────────────────────────────
def load_mean_waveforms(df):
    """Returns (n_elec, 49) arrays for pre and post mean waveforms."""
    pre_list, post_list = [], []
    for _, row in df.iterrows():
        d = np.load(row["npz"], allow_pickle=True)
        np_arr = d["norm_pre"]   # (49, n_pre_cycles)
        po_arr = d["norm_post"]  # (49, n_post_cycles)
        # Each electrode contributes its per-cycle mean
        if np_arr.shape[1] >= 1 and po_arr.shape[1] >= 1:
            pre_list.append(np_arr.mean(axis=1))
            post_list.append(po_arr.mean(axis=1))
    return np.stack(pre_list), np.stack(post_list)   # (n_elec, 49)

print("Loading stimulated waveforms...")
stim_pre,    stim_post    = load_mean_waveforms(stim)
print("Loading non-stimulated waveforms...")
nonstim_pre, nonstim_post = load_mean_waveforms(nonstim)
print("Done.")

# ── layout ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
ax_A = fig.add_subplot(gs[0, 0])
ax_B = fig.add_subplot(gs[0, 1])
ax_C = fig.add_subplot(gs[1, 0])
ax_D = fig.add_subplot(gs[1, 1])

x_phase = np.linspace(0, 1, stim_pre.shape[1])

# ════════════════════════════════════════════════════════════════════════════════
# PANELS A & B — grand mean waveforms
# ════════════════════════════════════════════════════════════════════════════════
def plot_waveform_panel(ax, pre_arr, post_arr, title, label):
    n = pre_arr.shape[0]

    mean_pre  = pre_arr.mean(axis=0)
    sem_pre   = pre_arr.std(axis=0)  / np.sqrt(n)
    mean_post = post_arr.mean(axis=0)
    sem_post  = post_arr.std(axis=0) / np.sqrt(n)

    ax.fill_between(x_phase, mean_pre - sem_pre, mean_pre + sem_pre,
                    color=PRE_COLOR, alpha=0.20)
    ax.fill_between(x_phase, mean_post - sem_post, mean_post + sem_post,
                    color=POST_COLOR, alpha=0.20)
    ax.plot(x_phase, mean_pre,  color=PRE_COLOR,  lw=2.2,
            label=f"Pre  (n = {n} electrodes)")
    ax.plot(x_phase, mean_post, color=POST_COLOR, lw=2.2,
            label=f"Post (n = {n} electrodes)")

    ax.axhline(0, color="0.75", lw=0.8, ls="--")
    ax.set_xlabel("Cycle phase  (0 → 1)", fontsize=10)
    ax.set_ylabel("Normalised amplitude", fontsize=10)
    ax.set_title(title, fontsize=9, pad=6)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    panel_label(ax, label)

plot_waveform_panel(
    ax_A, stim_pre, stim_post,
    f"Stimulated electrodes  (n = {len(stim)})\nGrand mean ± SEM pre vs post stimulation",
    "A",
)
plot_waveform_panel(
    ax_B, nonstim_pre, nonstim_post,
    f"Non-stimulated electrodes  (n = {len(nonstim)})\nGrand mean ± SEM pre vs post stimulation",
    "B",
)

# Match y-axis limits across A and B
y_all = [ax_A.get_ylim(), ax_B.get_ylim()]
y_min = min(y[0] for y in y_all)
y_max = max(y[1] for y in y_all)
ax_A.set_ylim(y_min, y_max)
ax_B.set_ylim(y_min, y_max)

# ════════════════════════════════════════════════════════════════════════════════
# PANELS C & D — Δ violin plots
# ════════════════════════════════════════════════════════════════════════════════
VIOLIN_CFGS = [
    (ax_C, "delta_asc2desc",    "Δ Ascent/descent ratio\n(post − pre)", "C"),
    (ax_D, "delta_peak2trough", "Δ Peak/trough ratio\n(post − pre)",    "D"),
]

stim_full    = master[master["is_stimulated"] == True].dropna(
                   subset=["delta_asc2desc", "delta_peak2trough"])
nonstim_full = master[master["is_stimulated"] == False].dropna(
                   subset=["delta_asc2desc", "delta_peak2trough"])

NSTIM_COLOR = "#4C8BB5"
STIM_COLOR  = "#C44536"

for ax, feat, ylabel, letter in VIOLIN_CFGS:
    ns_vals = nonstim_full[feat].values
    st_vals = stim_full[feat].values

    parts = ax.violinplot(
        [ns_vals, st_vals], positions=[0, 1],
        widths=0.55, showmedians=False, showextrema=False,
    )
    for body, col in zip(parts["bodies"], [NSTIM_COLOR, STIM_COLOR]):
        body.set_facecolor(col)
        body.set_alpha(0.35)
        body.set_edgecolor(col)
        body.set_linewidth(1)

    rng = np.random.default_rng(42)
    for xi, vals, col in zip([0, 1], [ns_vals, st_vals], [NSTIM_COLOR, STIM_COLOR]):
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([xi - 0.08, xi + 0.08], [med, med], color=col, lw=2.5, zorder=4)
        ax.plot([xi, xi], [q1, q3], color=col, lw=6,
                solid_capstyle="butt", alpha=0.6, zorder=3)
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(xi + jitter, vals, s=8, color=col,
                   alpha=0.35, zorder=2, linewidths=0)

    ax.axhline(0, color="0.6", lw=1, ls="--", zorder=1)

    _, p_mw = stats.mannwhitneyu(st_vals, ns_vals, alternative="two-sided")
    stars    = sig_stars(p_mw)
    y_top    = max(ns_vals.max(), st_vals.max())
    y_sig    = y_top * 1.08
    ax.plot([0, 0, 1, 1], [y_top, y_sig, y_sig, y_top], color="0.3", lw=1.2)
    ax.text(0.5, y_sig * 1.04,
            f"p = {p_mw:.2e}  {stars}",
            ha="center", fontsize=8.5, color="0.2")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [f"Non-stimulated\n(n = {len(ns_vals)})",
         f"Stimulated\n(n = {len(st_vals)})"],
        fontsize=9,
    )
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlim(-0.5, 1.5)
    panel_label(ax, letter)

# ════════════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════════════
fig.suptitle(
    "Waveform shape change after brain stimulation — all subjects",
    fontsize=12, y=1.01,
)
fig.savefig(OUT / "fig2_waveform_change.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig2_waveform_change.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Figure 2 saved to {OUT}")
