from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seeg_waveform.quinn_figures import set_publication_style


OUT_ROOT = ROOT / "outputs" / "publication_bipolar_mne" / "primary_28s_guarded"
SUPPORT = OUT_ROOT / "tables" / "figure_support"
FIG_DIR = OUT_ROOT / "figures" / "corrected_addons"
COLORS = {1.0: "#2A9D8F", 50.0: "#D62828"}
FREQ_LABELS = {1.0: "1 Hz stimulation protocol", 50.0: "50 Hz stimulation protocol"}
PREFERRED_REGIONS = ["frontal", "insula", "temporal", "hippocampus", "parietal", "occipital", "other"]
DISTANCE_BIN_LABELS = ["<30", "30-<60", "60-<90", ">=90"]


def _nice_region(value: str) -> str:
    return str(value).replace("_", " ").title()


def _sem(values: pd.Series) -> float:
    clean = values.dropna().astype(float)
    return float(clean.sem()) if len(clean) > 1 else np.nan


def _panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.1, 1.025, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")


def _summarize(values: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        values.groupby(group_cols + ["stim_frequency_hz"], as_index=False)
        .agg(
            mean_delta_abs_imcoh=("delta_abs_imcoh", "mean"),
            sem_delta_abs_imcoh=("delta_abs_imcoh", _sem),
            n_subjects=("subject", "nunique"),
        )
    )


def _draw_region_forest(
    ax: plt.Axes,
    values: pd.DataFrame,
    label_col: str,
    labels: list[str],
    title: str,
    ylabel: str,
) -> None:
    y_lookup = {label: index for index, label in enumerate(labels[::-1])}
    offsets = {1.0: -0.17, 50.0: 0.17}
    summary = _summarize(values, [label_col])

    for freq in [1.0, 50.0]:
        color = COLORS[freq]
        sub = values.loc[values["stim_frequency_hz"].eq(freq)].copy()
        for label, group in sub.groupby(label_col):
            if label not in y_lookup:
                continue
            y = y_lookup[label] + offsets[freq]
            jitter = np.linspace(-0.045, 0.045, len(group)) if len(group) > 1 else np.array([0.0])
            ax.scatter(
                group["delta_abs_imcoh"],
                y + jitter,
                s=24,
                color=color,
                alpha=0.47,
                edgecolor="white",
                linewidth=0.35,
                zorder=2,
            )
        freq_summary = summary.loc[summary["stim_frequency_hz"].eq(freq)]
        for row in freq_summary.itertuples():
            label = getattr(row, label_col)
            if label not in y_lookup or int(row.n_subjects) < 2:
                continue
            y = y_lookup[label] + offsets[freq]
            err = 0.0 if not np.isfinite(row.sem_delta_abs_imcoh) else float(row.sem_delta_abs_imcoh)
            ax.errorbar(
                float(row.mean_delta_abs_imcoh),
                y,
                xerr=err,
                fmt="D",
                ms=4.2,
                color=color,
                ecolor=color,
                elinewidth=1.1,
                capsize=2.0,
                zorder=4,
                label=FREQ_LABELS[freq] if label == labels[0] else None,
            )
            ax.text(
                ax.get_xlim()[1],
                y,
                f"n={int(row.n_subjects)}",
                ha="right",
                va="center",
                fontsize=6,
                color=color,
            )

    ax.axvline(0, color="0.35", lw=0.8, ls="--")
    ax.set_yticks(range(len(labels)), [_nice_region(label) for label in labels[::-1]])
    ax.set_xlabel("Post - pre absolute imaginary coherence")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_distance_profile(ax: plt.Axes, values: pd.DataFrame, title: str, xlabel: str) -> None:
    positions = np.arange(len(DISTANCE_BIN_LABELS), dtype=float)
    offsets = {1.0: -0.1, 50.0: 0.1}
    tick_labels: list[str] = []
    for label in DISTANCE_BIN_LABELS:
        counts = [
            int(
                values.loc[
                    values["stim_frequency_hz"].eq(freq) & values["distance_bin_mm"].eq(label),
                    "subject",
                ].nunique()
            )
            for freq in (1.0, 50.0)
        ]
        tick_labels.append(f"{label}\nn={counts[0]}/{counts[1]}")
    for freq in (1.0, 50.0):
        means: list[float] = []
        sems: list[float] = []
        for index, label in enumerate(DISTANCE_BIN_LABELS):
            sub = values.loc[
                values["stim_frequency_hz"].eq(freq) & values["distance_bin_mm"].eq(label),
                "delta_abs_imcoh",
            ].dropna().astype(float)
            x = positions[index] + offsets[freq]
            jitter = np.linspace(-0.035, 0.035, len(sub)) if len(sub) > 1 else np.array([0.0])
            ax.scatter(
                np.repeat(x, len(sub)) + jitter,
                sub,
                s=23,
                color=COLORS[freq],
                alpha=0.42,
                edgecolor="white",
                linewidth=0.25,
            )
            means.append(float(sub.mean()) if len(sub) else np.nan)
            sems.append(float(sub.sem()) if len(sub) > 1 else np.nan)
        ax.errorbar(
            positions + offsets[freq],
            means,
            yerr=sems,
            marker="D",
            ms=4,
            lw=1.4,
            capsize=2,
            color=COLORS[freq],
            label=FREQ_LABELS[freq],
        )
    ax.axhline(0, color="0.35", lw=0.8, ls="--")
    ax.set_xticks(positions, tick_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Post - pre absolute imaginary coherence")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_seed_region_figure() -> Path:
    values = pd.read_csv(SUPPORT / "region_seed_connectivity_subject_values.csv")
    values = values.loc[values["delta_abs_imcoh"].notna()].copy()
    support = values.groupby("region_family")["subject"].nunique()
    labels = [region for region in PREFERRED_REGIONS if support.get(region, 0) >= 2]

    set_publication_style()
    fig, ax = plt.subplots(figsize=(6.5, max(3.2, 0.43 * len(labels) + 1.1)), constrained_layout=True)
    _draw_region_forest(
        ax,
        values,
        "region_family",
        labels,
        "Stimulated seed-to-target IMF4 connectivity change by target region",
        "Target region family",
    )
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("Subject-balanced regional seed-to-target imaginary-coherence changes", y=1.02)
    out = FIG_DIR / "figure_03_regionwise_seed_target_imcoh_forest.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def make_main_figure3(max_pairs: int = 8) -> Path:
    seed_values = pd.read_csv(SUPPORT / "region_seed_connectivity_subject_values.csv")
    seed_values = seed_values.loc[seed_values["delta_abs_imcoh"].notna()].copy()
    seed_support = seed_values.groupby("region_family")["subject"].nunique()
    seed_labels = [region for region in PREFERRED_REGIONS if seed_support.get(region, 0) >= 2]

    remote_values = pd.read_csv(SUPPORT / "remote_region_poststim_subject_values.csv")
    remote_values = remote_values.loc[remote_values["delta_abs_imcoh"].notna()].copy()
    remote_values["region_pair"] = remote_values["region_a"].astype(str) + " - " + remote_values["region_b"].astype(str)
    support = remote_values.groupby("region_pair")["subject"].nunique()
    effect = remote_values.groupby("region_pair")["delta_abs_imcoh"].mean().abs()
    candidates = support.loc[support >= 2].index
    remote_labels = (
        pd.DataFrame({"support": support.loc[candidates], "effect": effect.loc[candidates]})
        .sort_values(["support", "effect"], ascending=[False, False])
        .head(max_pairs)
        .sort_values("effect", ascending=True)
        .index.tolist()
    )

    seed_distance = pd.read_csv(OUT_ROOT / "tables" / "seed_distance_bin_subject_summary.csv")
    remote_proximity = pd.read_csv(OUT_ROOT / "tables" / "remote_proximity_bin_subject_summary.csv")

    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), constrained_layout=True)
    _draw_region_forest(
        axes[0, 0],
        seed_values,
        "region_family",
        seed_labels,
        "Seed-to-target change by target region",
        "Target region family",
    )
    _draw_distance_profile(
        axes[0, 1],
        seed_distance,
        "Seed-to-target change by distance",
        "Seed-target distance from stimulated seed (mm)\n(n = 1 Hz / 50 Hz subjects)",
    )
    _draw_region_forest(
        axes[1, 0],
        remote_values,
        "region_pair",
        remote_labels,
        "Non-stimulated network change by region pair",
        "Region-family pair",
    )
    _draw_distance_profile(
        axes[1, 1],
        remote_proximity,
        "Remote-network change by proximity to seed",
        "Minimum recording-site distance to stimulated seed (mm)\n(n = 1 Hz / 50 Hz subjects)",
    )
    for ax, label in zip(axes.ravel(), "abcd"):
        _panel(ax, label)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.53, 1.03))
    fig.suptitle("IMF4 imaginary-coherence changes in stimulation-anchored and recorded networks", y=1.0)
    out = OUT_ROOT / "figures" / "figure_03_connectivity_region_distance.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def make_remote_region_pair_figure(max_pairs: int = 12) -> Path:
    values = pd.read_csv(SUPPORT / "remote_region_poststim_subject_values.csv")
    values = values.loc[values["delta_abs_imcoh"].notna()].copy()
    values["region_pair"] = values["region_a"].astype(str) + " - " + values["region_b"].astype(str)

    support = values.groupby("region_pair")["subject"].nunique()
    effect = values.groupby("region_pair")["delta_abs_imcoh"].mean().abs()
    candidates = support.loc[support >= 2].index
    labels = (
        pd.DataFrame({"support": support.loc[candidates], "effect": effect.loc[candidates]})
        .sort_values(["support", "effect"], ascending=[False, False])
        .head(max_pairs)
        .sort_values("effect", ascending=True)
        .index.tolist()
    )

    set_publication_style()
    fig, ax = plt.subplots(figsize=(7.0, max(3.5, 0.42 * len(labels) + 1.2)), constrained_layout=True)
    _draw_region_forest(
        ax,
        values,
        "region_pair",
        labels,
        "Non-stimulated recorded-network IMF4 connectivity change by region pair",
        "Region-family pair",
    )
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("Subject-balanced remote-network imaginary-coherence changes", y=1.02)
    out = FIG_DIR / "figure_03_regionwise_remote_network_imcoh_forest.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def make_combined_figure(max_pairs: int = 10) -> Path:
    seed_values = pd.read_csv(SUPPORT / "region_seed_connectivity_subject_values.csv")
    seed_values = seed_values.loc[seed_values["delta_abs_imcoh"].notna()].copy()
    seed_support = seed_values.groupby("region_family")["subject"].nunique()
    seed_labels = [region for region in PREFERRED_REGIONS if seed_support.get(region, 0) >= 2]

    remote_values = pd.read_csv(SUPPORT / "remote_region_poststim_subject_values.csv")
    remote_values = remote_values.loc[remote_values["delta_abs_imcoh"].notna()].copy()
    remote_values["region_pair"] = remote_values["region_a"].astype(str) + " - " + remote_values["region_b"].astype(str)
    support = remote_values.groupby("region_pair")["subject"].nunique()
    effect = remote_values.groupby("region_pair")["delta_abs_imcoh"].mean().abs()
    candidates = support.loc[support >= 2].index
    remote_labels = (
        pd.DataFrame({"support": support.loc[candidates], "effect": effect.loc[candidates]})
        .sort_values(["support", "effect"], ascending=[False, False])
        .head(max_pairs)
        .sort_values("effect", ascending=True)
        .index.tolist()
    )

    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True)
    _draw_region_forest(
        axes[0],
        seed_values,
        "region_family",
        seed_labels,
        "Stimulated seed-to-target",
        "Target region family",
    )
    _draw_region_forest(
        axes[1],
        remote_values,
        "region_pair",
        remote_labels,
        "Non-stimulated recorded network",
        "Region-family pair",
    )
    _panel(axes[0], "a")
    _panel(axes[1], "b")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.52, -0.045))
    fig.suptitle("Region-wise IMF4 absolute imaginary-coherence change", y=1.025)
    out = FIG_DIR / "figure_03_regionwise_imcoh_forest.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    global OUT_ROOT, SUPPORT, FIG_DIR
    parser = argparse.ArgumentParser(description="Create region-wise IMF4 connectivity figures.")
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=OUT_ROOT,
        help="Analysis root containing tables/ and figures/.",
    )
    args = parser.parse_args()
    OUT_ROOT = args.analysis_root
    SUPPORT = OUT_ROOT / "tables" / "figure_support"
    FIG_DIR = OUT_ROOT / "figures" / "corrected_addons"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = [make_seed_region_figure(), make_remote_region_pair_figure(), make_combined_figure(), make_main_figure3()]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
