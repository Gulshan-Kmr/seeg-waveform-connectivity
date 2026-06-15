from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seeg_waveform.quinn_figures import set_publication_style


OUT_ROOT = ROOT / "outputs" / "publication_bipolar_mne" / "primary_28s_guarded"
TABLE_DIR = OUT_ROOT / "tables"
SUPPORT = TABLE_DIR / "figure_support"
FIG_DIR = OUT_ROOT / "figures"
COLORS = {1.0: "#2A9D8F", 50.0: "#D62828"}
LABELS = {1.0: "1 Hz stimulation protocol", 50.0: "50 Hz stimulation protocol"}
METRICS = {
    "delta_abs_imcoh": "Absolute ImCoh",
    "delta_imcoh_signed": "Signed ImCoh",
    "delta_coh": "Coherence",
}


def _panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")


def _sem(values: pd.Series) -> float:
    clean = values.dropna().astype(float)
    return float(clean.sem()) if len(clean) > 1 else np.nan


def _included_primary_seed_edges(edges: pd.DataFrame) -> pd.DataFrame:
    return edges.loc[
        edges["imf"].eq(4)
        & edges["imf_role"].eq("primary")
        & edges["edge_type"].eq("seed_to_target")
        & edges["analysis_include"].fillna(False).astype(bool)
    ].copy()


def _joined_target_edges() -> pd.DataFrame:
    edges = pd.read_csv(TABLE_DIR / "connectivity_edges.csv")
    waveform = pd.read_csv(TABLE_DIR / "waveform_nodes.csv")
    seed = _included_primary_seed_edges(edges)
    target_wf = waveform.loc[
        waveform["analysis_include"].fillna(False).astype(bool),
        ["file", "channel", "region_family", "delta_mean_if"],
    ].rename(columns={"channel": "target_channel", "region_family": "target_region_family"})
    joined = seed.merge(target_wf, on=["file", "target_channel"], how="inner")
    joined.to_csv(SUPPORT / "figure4_target_edge_waveform_connectivity_join.csv", index=False)
    return joined


def _metric_coupling(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for metric in METRICS:
        for keys, group in joined.groupby(["file", "subject", "stim_frequency_hz"], dropna=False):
            clean = group[["delta_mean_if", metric]].replace([np.inf, -np.inf], np.nan).dropna()
            rho = stats.spearmanr(clean["delta_mean_if"], clean[metric]).statistic if len(clean) >= 4 else np.nan
            rows.append(
                {
                    "file": keys[0],
                    "subject": keys[1],
                    "stim_frequency_hz": keys[2],
                    "metric": metric,
                    "metric_label": METRICS[metric],
                    "n_edges": len(clean),
                    "spearman_rho": rho,
                }
            )
    run = pd.DataFrame(rows)
    subject = (
        run.dropna(subset=["spearman_rho"])
        .groupby(["subject", "stim_frequency_hz", "metric", "metric_label"], as_index=False)
        .agg(spearman_rho=("spearman_rho", "mean"), n_runs=("file", "nunique"))
    )
    run.to_csv(SUPPORT / "figure4_metric_coupling_run_values.csv", index=False)
    subject.to_csv(SUPPORT / "figure4_metric_coupling_subject_values.csv", index=False)
    return run, subject


def _region_coupling(joined: pd.DataFrame) -> pd.DataFrame:
    run_rows: list[dict] = []
    for keys, group in joined.groupby(["file", "subject", "stim_frequency_hz", "target_region_family"], dropna=False):
        clean = group[["delta_mean_if", "delta_abs_imcoh"]].replace([np.inf, -np.inf], np.nan).dropna()
        rho = stats.spearmanr(clean["delta_mean_if"], clean["delta_abs_imcoh"]).statistic if len(clean) >= 4 else np.nan
        run_rows.append(
            {
                "file": keys[0],
                "subject": keys[1],
                "stim_frequency_hz": keys[2],
                "target_region_family": keys[3],
                "spearman_rho": rho,
                "n_edges": len(clean),
            }
        )
    run_region = pd.DataFrame(run_rows)
    subject_region = (
        run_region.dropna(subset=["spearman_rho"])
        .groupby(["subject", "stim_frequency_hz", "target_region_family"], as_index=False)
        .agg(
            spearman_rho=("spearman_rho", "mean"),
            n_runs=("file", "nunique"),
            n_edges=("n_edges", "sum"),
        )
    )
    summary = (
        subject_region.groupby(["stim_frequency_hz", "target_region_family"], as_index=False)
        .agg(
            mean_spearman_rho=("spearman_rho", "mean"),
            sem_spearman_rho=("spearman_rho", _sem),
            n_subjects=("subject", "nunique"),
        )
    )
    run_region.to_csv(SUPPORT / "figure4_region_coupling_run_values.csv", index=False)
    subject_region.to_csv(SUPPORT / "figure4_region_coupling_subject_values.csv", index=False)
    summary.to_csv(SUPPORT / "figure4_region_coupling_summary.csv", index=False)
    return summary


def _region_bivariate_values(joined: pd.DataFrame) -> pd.DataFrame:
    run_region = (
        joined.groupby(["file", "subject", "stim_frequency_hz", "target_region_family"], as_index=False)
        .agg(
            delta_mean_if=("delta_mean_if", "mean"),
            delta_abs_imcoh=("delta_abs_imcoh", "mean"),
            n_edges=("target_channel", "nunique"),
        )
    )
    subject_region = (
        run_region.groupby(["subject", "stim_frequency_hz", "target_region_family"], as_index=False)
        .agg(
            delta_mean_if=("delta_mean_if", "mean"),
            delta_abs_imcoh=("delta_abs_imcoh", "mean"),
            n_runs=("file", "nunique"),
            n_edges=("n_edges", "sum"),
        )
    )
    subject_region.to_csv(SUPPORT / "figure4_region_bivariate_subject_values.csv", index=False)
    return subject_region


def _draw_scatter(ax: plt.Axes, run: pd.DataFrame, subject: pd.DataFrame) -> None:
    for freq in (1.0, 50.0):
        color = COLORS[freq]
        sub_run = run.loc[run["stim_frequency_hz"].eq(freq)].copy()
        sub_subject = subject.loc[subject["stim_frequency_hz"].eq(freq)].copy()
        ax.scatter(
            sub_run["delta_mean_if"],
            sub_run["delta_abs_imcoh"],
            s=28,
            facecolor="none",
            edgecolor=color,
            alpha=0.45,
            linewidth=0.9,
            label=f"{freq:g} Hz run summary",
        )
        ax.scatter(
            sub_subject["delta_mean_if"],
            sub_subject["delta_abs_imcoh"],
            s=56,
            marker="D",
            color=color,
            edgecolor="white",
            linewidth=0.45,
            label=f"{freq:g} Hz subject summary",
        )
    clean = subject[["delta_mean_if", "delta_abs_imcoh"]].dropna()
    rho = stats.spearmanr(clean["delta_mean_if"], clean["delta_abs_imcoh"]).statistic if len(clean) >= 4 else np.nan
    ax.axhline(0, color="0.45", lw=0.8, ls="--")
    ax.axvline(0, color="0.45", lw=0.8, ls="--")
    ax.set_xlabel("Post - pre IMF4 mean instantaneous frequency (Hz)")
    ax.set_ylabel("Post - pre seed-to-target absolute ImCoh")
    ax.set_title(f"Run- and subject-level summaries\nsubject Spearman rho = {rho:.2f}")
    ax.legend(frameon=False, fontsize=6, loc="best")


def _draw_correlation_points(ax: plt.Axes, subject_metric: pd.DataFrame, metric: str, title: str) -> None:
    values = subject_metric.loc[subject_metric["metric"].eq(metric)].copy()
    positions = {1.0: 0.0, 50.0: 1.0}
    for freq in (1.0, 50.0):
        sub = values.loc[values["stim_frequency_hz"].eq(freq), "spearman_rho"].dropna().astype(float)
        x = positions[freq]
        jitter = np.linspace(-0.045, 0.045, len(sub)) if len(sub) > 1 else np.array([0.0])
        ax.scatter(
            np.repeat(x, len(sub)) + jitter,
            sub,
            s=38,
            color=COLORS[freq],
            alpha=0.68,
            edgecolor="white",
            linewidth=0.35,
        )
        if len(sub):
            ax.errorbar(x, sub.mean(), yerr=sub.sem() if len(sub) > 1 else np.nan, fmt="D", color=COLORS[freq], capsize=2)
            ax.text(x, ax.get_ylim()[1], f"n={sub.notna().sum()}", ha="center", va="top", fontsize=7, color=COLORS[freq])
    ax.axhline(0, color="0.45", lw=0.8, ls="--")
    ax.set_xticks([0, 1], ["1 Hz", "50 Hz"])
    ax.set_ylabel("Within-run Spearman rho")
    ax.set_title(title)


def _draw_region_summary(ax: plt.Axes, summary: pd.DataFrame) -> None:
    preferred = ["frontal", "insula", "temporal", "hippocampus", "parietal", "other"]
    available = [region for region in preferred if region in set(summary["target_region_family"])]
    y_lookup = {region: index for index, region in enumerate(available[::-1])}
    offsets = {1.0: -0.17, 50.0: 0.17}
    for freq in (1.0, 50.0):
        sub = summary.loc[summary["stim_frequency_hz"].eq(freq) & summary["target_region_family"].isin(preferred)].copy()
        for row in sub.itertuples():
            if row.n_subjects < 2:
                continue
            y = y_lookup[row.target_region_family] + offsets[freq]
            ax.errorbar(
                row.mean_spearman_rho,
                y,
                xerr=0 if not np.isfinite(row.sem_spearman_rho) else row.sem_spearman_rho,
                fmt="D",
                color=COLORS[freq],
                alpha=0.85,
                capsize=2,
                ms=4.5,
            )
            ax.text(
                ax.get_xlim()[1],
                y,
                f"n={int(row.n_subjects)}",
                ha="right",
                va="center",
                fontsize=6,
                color=COLORS[freq],
            )
    ax.axvline(0, color="0.45", lw=0.8, ls="--")
    ax.set_yticks(range(len(available)), [region.title() for region in available[::-1]])
    ax.set_xlabel("Within-region Spearman rho")
    ax.set_ylabel("Target region family")
    ax.set_title("Region-wise waveform/connectivity coupling")


def _draw_metric_sensitivity(ax: plt.Axes, subject_metric: pd.DataFrame) -> None:
    metric_order = list(METRICS)
    offsets = {1.0: -0.16, 50.0: 0.16}
    for index, metric in enumerate(metric_order):
        for freq in (1.0, 50.0):
            sub = subject_metric.loc[
                subject_metric["metric"].eq(metric) & subject_metric["stim_frequency_hz"].eq(freq),
                "spearman_rho",
            ].dropna().astype(float)
            x = index + offsets[freq]
            jitter = np.linspace(-0.035, 0.035, len(sub)) if len(sub) > 1 else np.array([0.0])
            ax.scatter(np.repeat(x, len(sub)) + jitter, sub, s=25, color=COLORS[freq], alpha=0.45)
            if len(sub):
                ax.errorbar(x, sub.mean(), yerr=sub.sem() if len(sub) > 1 else np.nan, fmt="D", color=COLORS[freq], capsize=2)
    ax.axhline(0, color="0.45", lw=0.8, ls="--")
    ax.set_xticks(range(len(metric_order)), [METRICS[m] for m in metric_order], rotation=20, ha="right")
    ax.set_ylabel("Within-run Spearman rho")
    ax.set_title("Connectivity metric sensitivity")


def _make_region_small_multiples(subject_region: pd.DataFrame) -> Path:
    preferred = ["frontal", "insula", "temporal", "hippocampus", "parietal", "other"]
    regions = [
        region
        for region in preferred
        if subject_region.loc[subject_region["target_region_family"].eq(region), "subject"].nunique() >= 2
    ]
    set_publication_style()
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.4), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    xlim = np.nanpercentile(subject_region["delta_mean_if"], [2, 98])
    ylim = np.nanpercentile(subject_region["delta_abs_imcoh"], [2, 98])
    x_pad = max(0.05, float((xlim[1] - xlim[0]) * 0.12))
    y_pad = max(0.01, float((ylim[1] - ylim[0]) * 0.12))
    for ax, region in zip(axes_flat, regions):
        sub = subject_region.loc[subject_region["target_region_family"].eq(region)].copy()
        for freq in (1.0, 50.0):
            freq_sub = sub.loc[sub["stim_frequency_hz"].eq(freq)]
            ax.scatter(
                freq_sub["delta_mean_if"],
                freq_sub["delta_abs_imcoh"],
                s=36,
                color=COLORS[freq],
                alpha=0.72,
                edgecolor="white",
                linewidth=0.35,
                label=LABELS[freq],
            )
            if len(freq_sub) >= 2:
                ax.scatter(
                    [freq_sub["delta_mean_if"].mean()],
                    [freq_sub["delta_abs_imcoh"].mean()],
                    s=72,
                    marker="D",
                    color=COLORS[freq],
                    edgecolor="white",
                    linewidth=0.45,
                )
        n1 = sub.loc[sub["stim_frequency_hz"].eq(1.0), "subject"].nunique()
        n50 = sub.loc[sub["stim_frequency_hz"].eq(50.0), "subject"].nunique()
        ax.axhline(0, color="0.55", lw=0.7, ls="--")
        ax.axvline(0, color="0.55", lw=0.7, ls="--")
        ax.set_title(f"{region.title()} (n={n1}/{n50})")
        ax.set_xlim(xlim[0] - x_pad, xlim[1] + x_pad)
        ax.set_ylim(ylim[0] - y_pad, ylim[1] + y_pad)
    for ax in axes_flat[len(regions) :]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Post - pre absolute ImCoh")
    for ax in axes[-1, :]:
        ax.set_xlabel("Post - pre IMF4 mean IF (Hz)")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.53, 0.985))
    fig.suptitle("Region-wise relationship between IMF4 waveform and seed-to-target connectivity change", y=0.935)
    fig.subplots_adjust(top=0.82, hspace=0.32, wspace=0.08)
    out = FIG_DIR / "figure_04_regionwise_waveform_connectivity_small_multiples.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    global OUT_ROOT, TABLE_DIR, SUPPORT, FIG_DIR
    parser = argparse.ArgumentParser(description="Create waveform-connectivity coupling figures.")
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=OUT_ROOT,
        help="Analysis root containing tables/ and figures/.",
    )
    args = parser.parse_args()
    OUT_ROOT = args.analysis_root
    TABLE_DIR = OUT_ROOT / "tables"
    SUPPORT = TABLE_DIR / "figure_support"
    FIG_DIR = OUT_ROOT / "figures"
    SUPPORT.mkdir(parents=True, exist_ok=True)
    joined = _joined_target_edges()
    _, subject_metric = _metric_coupling(joined)
    region_summary = _region_coupling(joined)
    subject_region = _region_bivariate_values(joined)
    run = pd.read_csv(TABLE_DIR / "waveform_connectivity_run_summary.csv")
    subject = pd.read_csv(TABLE_DIR / "waveform_connectivity_subject_protocol_summary.csv")

    set_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.85), constrained_layout=True)
    _draw_scatter(axes[0], run, subject)
    _draw_correlation_points(axes[1], subject_metric, "delta_abs_imcoh", "Target-level waveform/connectivity coupling")
    _draw_metric_sensitivity(axes[2], subject_metric)
    for ax, label in zip(axes, "abc"):
        _panel(ax, label)
    fig.suptitle("Relationship between IMF4 waveform dynamics and connectivity change", y=1.05)
    out = FIG_DIR / "figure_04_waveform_connectivity_coupling.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out)
    print(_make_region_small_multiples(subject_region))


if __name__ == "__main__":
    main()
