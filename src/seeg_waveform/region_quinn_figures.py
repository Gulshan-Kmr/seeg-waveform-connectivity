from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config
from .pipeline import safe_name
from .quinn_figures import (
    _add_empty_message,
    _collect_channel_means,
    _eligible_for_average,
    _legend_if_any,
    set_publication_style,
)
from .region_waveform import add_region_family


def _mean_sem(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(arr, axis=0)
    sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(max(arr.shape[0], 1))
    return mean, sem


def _plot_profiles(ax: plt.Axes, x: np.ndarray, pre: np.ndarray, post: np.ndarray, ylabel: str) -> None:
    if not pre.size or not post.size:
        _add_empty_message(ax)
        return
    for arr, color, name in [(pre, "tab:blue", "Pre"), (post, "tab:red", "Post")]:
        ax.plot(x, arr.T, color=color, alpha=0.035, linewidth=0.3)
        mean, sem = _mean_sem(arr)
        ax.plot(x, mean, color=color, linewidth=2.0, label=f"{name} mean")
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.18, linewidth=0)
    ax.set_ylabel(ylabel)
    _legend_if_any(ax)


def plot_region_quinn_group(
    group: pd.DataFrame,
    out_root: Path,
    imf_index: int,
    label: str,
    out_path: Path,
    dpi: int = 300,
) -> Path:
    set_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.0), constrained_layout=True)
    n_subjects = group["subject_number"].nunique() if "subject_number" in group else group["subject"].nunique()
    fig.suptitle(f"Quinn-style regional average | {label} | n={len(group)} contacts, {n_subjects} subjects")

    pre_if, post_if = _collect_channel_means(group, out_root, imf_index, "pa_if")
    phase = np.linspace(0, 2 * np.pi, pre_if.shape[1]) if pre_if.size else np.linspace(0, 2 * np.pi, 64)
    _plot_profiles(axes[0], phase, pre_if, post_if, "IF (Hz)")
    axes[0].set_title("Phase-aligned IF")
    axes[0].set_xlabel("Phase (rad)")

    pre_norm, post_norm = _collect_channel_means(group, out_root, imf_index, "norm_waveform")
    x = np.linspace(0, 1, pre_norm.shape[1]) if pre_norm.size else np.linspace(0, 1, 64)
    _plot_profiles(axes[1], x, pre_norm, post_norm, "Amplitude")
    axes[1].plot(np.linspace(0, 1, 200), np.sin(np.linspace(0, 2 * np.pi, 200)), "k--", linewidth=0.8, label="Sine ref")
    axes[1].set_title("Normalized waveform")
    axes[1].set_xlabel("Normalized cycle")

    metrics = ["delta_mean_if", "delta_asc2desc", "delta_peak2trough"]
    vals = [group[m].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float) for m in metrics]
    parts = axes[2].violinplot(vals, showmeans=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("0.65")
        body.set_edgecolor("0.2")
        body.set_alpha(0.55)
    if "cmeans" in parts:
        parts["cmeans"].set_color("black")
        parts["cmeans"].set_linewidth(1.2)
    axes[2].axhline(0, color="0.2", linewidth=0.7)
    axes[2].set_xticks([1, 2, 3])
    axes[2].set_xticklabels(["mean IF", "asc2desc", "peak2trough"], rotation=25, ha="right")
    axes[2].set_title("Post-pre metrics")
    axes[2].set_ylabel("Delta")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_region_quinn_figures(
    config_path: str | Path,
    merged_csv: str | Path,
    common_csv: str | Path,
    output_dir: str | Path | None = None,
    min_contacts: int = 5,
) -> list[Path]:
    cfg = load_config(config_path)
    out_root = Path(cfg["project"]["output_dir"])
    imf_index = int(cfg["emd"].get("imf_index", 3))
    dpi = int(cfg["outputs"].get("figure_dpi", 300))
    merged = pd.read_csv(merged_csv)
    merged = add_region_family(merged)
    eligible = _eligible_for_average(merged, cfg)
    eligible = eligible.loc[
        eligible["region_analysis_include"].fillna(False)
        & eligible["analysis_gray_matter_flag"].fillna(False)
    ].copy()
    common = pd.read_csv(common_csv)
    fig_dir = Path(output_dir) if output_dir else out_root / "figures" / "region_quinn"
    fig_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    # Specific clinically meaningful family requested by the user.
    group_specs: list[tuple[str, pd.DataFrame]] = [
        ("hippocampus_all", eligible.loc[eligible["region_family"].eq("hippocampus")]),
        (
            "hippocampus_nonstimulated",
            eligible.loc[eligible["region_family"].eq("hippocampus") & ~eligible["is_stimulated_channel"].astype(bool)],
        ),
        (
            "hippocampus_stimulated",
            eligible.loc[eligible["region_family"].eq("hippocampus") & eligible["is_stimulated_channel"].astype(bool)],
        ),
    ]

    family_candidates = []
    for contact_group, base in [
        ("stimulated", eligible.loc[eligible["is_stimulated_channel"].astype(bool)]),
        ("nonstimulated", eligible.loc[~eligible["is_stimulated_channel"].astype(bool)]),
    ]:
        fam = (
            base.groupby("region_family", dropna=False)
            .agg(n_contacts=("channel_norm", "size"), n_subjects=("subject_number", "nunique"))
            .reset_index()
        )
        fam = fam.loc[(fam["region_family"].ne("non_gray")) & (fam["n_subjects"] >= 2)]
        fam = fam.sort_values(["n_subjects", "n_contacts"], ascending=False).head(8)
        for _, row in fam.iterrows():
            family_candidates.append((contact_group, row["region_family"]))
    for contact_group, family in family_candidates:
        mask = eligible["region_family"].eq(family)
        if contact_group == "stimulated":
            mask &= eligible["is_stimulated_channel"].astype(bool)
        else:
            mask &= ~eligible["is_stimulated_channel"].astype(bool)
        group_specs.append((f"{contact_group}_{family}_family", eligible.loc[mask]))

    selected_regions = common.loc[
        common["analysis_gray_matter_flag"].fillna(False)
        & common["contact_group"].isin(["stimulated", "nonstimulated"])
    ].copy()
    selected_regions = selected_regions.sort_values(["contact_group", "n_subjects", "n_contacts"], ascending=[True, False, False])
    for _, row in selected_regions.iterrows():
        region = row["primary_region"]
        contact_group = row["contact_group"]
        mask = eligible["primary_region"].eq(region)
        if contact_group == "stimulated":
            mask &= eligible["is_stimulated_channel"].astype(bool)
        elif contact_group == "nonstimulated":
            mask &= ~eligible["is_stimulated_channel"].astype(bool)
        sub = eligible.loc[mask]
        group_specs.append((f"{contact_group}_{region}", sub))

    seen: set[str] = set()
    for label, group in group_specs:
        if len(group) < min_contacts:
            continue
        key = safe_name(label)
        if key in seen:
            continue
        seen.add(key)
        paths.append(plot_region_quinn_group(group, out_root, imf_index, label, fig_dir / f"{key}_quinn_region_average.png", dpi=dpi))
    return paths
