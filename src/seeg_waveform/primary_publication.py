from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .quinn_figures import set_publication_style
from .spatial_analysis import bh_fdr


BANDS = ["delta", "theta", "alpha", "beta", "low_gamma", "high_gamma"]
BAND_LABELS = {
    "delta": "Delta",
    "theta": "Theta",
    "alpha": "Alpha",
    "beta": "Beta",
    "low_gamma": "Low gamma",
    "high_gamma": "High gamma",
}


def _clean_waveform(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the waveform rows used for primary publication summaries."""
    ok = df.loc[df["status"].eq("ok")].copy()
    ok = ok.loc[~ok["pre_noise_flag"].fillna(False).astype(bool)]
    ok = ok.loc[~ok["post_noise_flag"].fillna(False).astype(bool)]
    return ok


def _mean_sem(values: pd.Series) -> tuple[float, float, int]:
    """Return mean, SEM, and n for plotting subject/file-level effects."""
    x = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if x.empty:
        return np.nan, np.nan, 0
    sem = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return float(x.mean()), float(sem), int(len(x))


def _wilcoxon(values: pd.Series) -> float:
    """Use a signed-rank test only when there are enough non-identical observations."""
    x = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(x) < 4 or x.nunique() <= 1:
        return np.nan
    try:
        return float(stats.wilcoxon(x).pvalue)
    except ValueError:
        return np.nan


def make_cohort_table(waveform: pd.DataFrame, out_dir: Path) -> Path:
    """Summarize files, subjects, frequencies, and stimulation current."""
    rows = []
    files = waveform.drop_duplicates("file").copy()
    for freq, group in files.groupby("stim_frequency_hz", dropna=False):
        rows.append(
            {
                "stim_frequency_hz": freq,
                "n_files": int(group["file"].nunique()),
                "n_subjects": int(group["subject"].nunique()),
                "current_ma_mean": float(group["stim_amplitude_ma"].mean()),
                "current_ma_median": float(group["stim_amplitude_ma"].median()),
                "current_ma_min": float(group["stim_amplitude_ma"].min()),
                "current_ma_max": float(group["stim_amplitude_ma"].max()),
                "n_channels_total": int(waveform.loc[waveform["stim_frequency_hz"].eq(freq), "channel"].nunique()),
            }
        )
    out = pd.DataFrame(rows)
    out_path = out_dir / "table_01_cohort_stimulation_current.csv"
    out.to_csv(out_path, index=False)
    return out_path


def make_primary_waveform_table(waveform_regions: pd.DataFrame, out_dir: Path) -> Path:
    """Primary waveform table after collapsing contacts within each file and region.

    This keeps the primary table closer to independent stimulation files and avoids
    treating every electrode contact as an independent participant.
    """
    df = _clean_waveform(waveform_regions)
    if "region_analysis_include" in df.columns:
        df = df.loc[df["region_analysis_include"].fillna(False)].copy()
    metrics = ["delta_mean_if", "delta_asc2desc", "delta_peak2trough"]
    file_region = (
        df.groupby(["file", "subject", "stim_frequency_hz", "stim_amplitude_ma", "is_stimulated_channel", "region_family"], dropna=False)
        .agg({metric: "mean" for metric in metrics} | {"channel": "nunique"})
        .reset_index()
        .rename(columns={"channel": "n_contacts"})
    )
    rows = []
    for metric in metrics:
        for keys, group in file_region.groupby(["stim_frequency_hz", "is_stimulated_channel", "region_family"], dropna=False):
            values = group[metric]
            mean, sem, n = _mean_sem(values)
            rows.append(
                {
                    "metric": metric,
                    "stim_frequency_hz": keys[0],
                    "is_stimulated_channel": keys[1],
                    "region_family": keys[2],
                    "n": n,
                    "n_subjects": int(group["subject"].nunique()),
                    "n_files": int(group["file"].nunique()),
                    "n_contacts_total": int(group["n_contacts"].sum()),
                    "mean_delta": mean,
                    "sem_delta": sem,
                    "wilcoxon_p": _wilcoxon(values),
                    "current_ma_mean": float(group["stim_amplitude_ma"].mean()),
                }
            )
    out = pd.DataFrame(rows)
    out["wilcoxon_p_fdr"] = bh_fdr(out["wilcoxon_p"])
    out_path = out_dir / "table_02_primary_waveform_by_region.csv"
    out.to_csv(out_path, index=False)
    return out_path


def make_primary_imf_table(imf_joined: pd.DataFrame, out_dir: Path) -> Path:
    """Primary IMF table: stimulated seeds to non-stimulated target regions."""
    df = imf_joined.loc[imf_joined["analysis_pair_include"].fillna(False)].copy()
    df = df.loc[df["target_region_family"].notna() & df["target_region_family"].ne("non_gray")]
    df["imf_group"] = np.where(df["imf"].isin([1, 2, 3]), "IMF1-3", "IMF4-6")
    file_target = (
        df.groupby(["file", "subject", "stim_frequency_hz", "stim_amplitude_ma", "target_region_family", "imf_group"], dropna=False)
        .agg(
            delta_imcoh=("delta_imcoh", "mean"),
            delta_coh=("delta_coh", "mean"),
            seed_target_distance_mni_mm=("seed_target_distance_mni_mm", "mean"),
            delta_mean_if=("delta_mean_if", "mean"),
            delta_asc2desc=("delta_asc2desc", "mean"),
            delta_peak2trough=("delta_peak2trough", "mean"),
        )
        .reset_index()
    )
    rows = []
    for keys, group in file_target.groupby(["stim_frequency_hz", "target_region_family", "imf_group"], dropna=False):
        for metric in ["delta_imcoh", "delta_coh"]:
            values = group[metric]
            mean, sem, n = _mean_sem(values)
            rows.append(
                {
                    "metric": metric,
                    "stim_frequency_hz": keys[0],
                    "target_region_family": keys[1],
                    "imf_group": keys[2],
                    "n": n,
                    "n_subjects": int(group["subject"].nunique()),
                    "n_files": int(group["file"].nunique()),
                    "mean_delta": mean,
                    "sem_delta": sem,
                    "wilcoxon_p": _wilcoxon(values),
                    "distance_mean_mm": float(group["seed_target_distance_mni_mm"].mean()),
                    "current_ma_mean": float(group["stim_amplitude_ma"].mean()),
                }
            )
    out = pd.DataFrame(rows)
    out["wilcoxon_p_fdr"] = bh_fdr(out["wilcoxon_p"])
    out_path = out_dir / "table_03_primary_imf_connectivity_by_region.csv"
    out.to_csv(out_path, index=False)
    return out_path


def make_distance_table(distance_summary: pd.DataFrame, out_dir: Path) -> Path:
    """Copy the distance-bin connectivity result into the primary table set."""
    cols = [
        "stim_frequency_hz",
        "distance_bin",
        "imf",
        "n",
        "n_subjects",
        "n_files",
        "distance_mean_mm",
        "delta_imcoh_mean",
        "delta_imcoh_sem",
        "delta_coh_mean",
        "delta_coh_sem",
    ]
    out = distance_summary[[c for c in cols if c in distance_summary.columns]].copy()
    out_path = out_dir / "table_04_connectivity_by_distance_bin.csv"
    out.to_csv(out_path, index=False)
    return out_path


def make_waveform_connectivity_table(imf_joined: pd.DataFrame, out_dir: Path) -> Path:
    """Relate target waveform changes to seed-target IMF imaginary-coherence changes."""
    df = imf_joined.loc[imf_joined["analysis_pair_include"].fillna(False)].copy()
    df = df.loc[df["target_region_family"].notna() & df["target_region_family"].ne("non_gray")]
    df["imf_group"] = np.where(df["imf"].isin([1, 2, 3]), "IMF1-3", "IMF4-6")
    # Collapse seed pairs so each file-target-IMF-group contributes once.
    file_target = (
        df.groupby(["file", "subject", "stim_frequency_hz", "target_norm", "target_region_family", "imf_group"], dropna=False)
        .agg(
            delta_imcoh=("delta_imcoh", "mean"),
            delta_coh=("delta_coh", "mean"),
            seed_target_distance_mni_mm=("seed_target_distance_mni_mm", "mean"),
            stim_amplitude_ma=("stim_amplitude_ma", "mean"),
            delta_mean_if=("delta_mean_if", "first"),
            delta_asc2desc=("delta_asc2desc", "first"),
            delta_peak2trough=("delta_peak2trough", "first"),
        )
        .reset_index()
    )
    rows = []
    for keys, group in file_target.groupby(["stim_frequency_hz", "target_region_family", "imf_group"], dropna=False):
        for y in ["delta_mean_if", "delta_asc2desc", "delta_peak2trough"]:
            sub = group[["delta_imcoh", y]].replace([np.inf, -np.inf], np.nan).dropna()
            row = dict(zip(["stim_frequency_hz", "target_region_family", "imf_group"], keys))
            row.update({"x": "delta_imcoh", "y": y, "n": int(len(sub)), "n_subjects": int(group["subject"].nunique())})
            if len(sub) >= 5 and sub["delta_imcoh"].nunique() > 1 and sub[y].nunique() > 1:
                rho, p = stats.spearmanr(sub["delta_imcoh"], sub[y])
                row["spearman_rho"] = float(rho)
                row["p_value"] = float(p)
            else:
                row["spearman_rho"] = np.nan
                row["p_value"] = np.nan
            rows.append(row)
    out = pd.DataFrame(rows)
    out["p_fdr"] = bh_fdr(out["p_value"])
    out_path = out_dir / "table_05_waveform_connectivity_relationship.csv"
    out.to_csv(out_path, index=False)
    return out_path


def make_primary_spectral_table(spectral_regions: pd.DataFrame, out_dir: Path) -> Path:
    """Primary spectral table collapsed within each file and region family."""
    df = spectral_regions.loc[spectral_regions["spectral_region_include"].fillna(False)].copy()
    spectral_metrics = [f"delta_{band}_log_power" for band in BANDS] + [f"delta_{band}_relative_power" for band in BANDS]
    file_region = (
        df.groupby(["file", "subject", "stim_frequency_hz", "stim_amplitude_ma", "is_stimulated_channel", "region_family"], dropna=False)
        .agg({metric: "mean" for metric in spectral_metrics} | {"channel": "nunique"})
        .reset_index()
        .rename(columns={"channel": "n_contacts"})
    )
    rows = []
    for band in BANDS:
        for metric in [f"delta_{band}_log_power", f"delta_{band}_relative_power"]:
            for keys, group in file_region.groupby(["stim_frequency_hz", "is_stimulated_channel", "region_family"], dropna=False):
                values = group[metric]
                mean, sem, n = _mean_sem(values)
                rows.append(
                    {
                        "band": band,
                        "metric": metric,
                        "stim_frequency_hz": keys[0],
                        "is_stimulated_channel": keys[1],
                        "region_family": keys[2],
                        "n": n,
                        "n_subjects": int(group["subject"].nunique()),
                        "n_files": int(group["file"].nunique()),
                        "n_contacts_total": int(group["n_contacts"].sum()),
                        "mean_delta": mean,
                        "sem_delta": sem,
                        "wilcoxon_p": _wilcoxon(values),
                        "current_ma_mean": float(group["stim_amplitude_ma"].mean()),
                    }
                )
    out = pd.DataFrame(rows)
    out["wilcoxon_p_fdr"] = bh_fdr(out["wilcoxon_p"])
    out_path = out_dir / "table_06_primary_spectral_by_region.csv"
    out.to_csv(out_path, index=False)
    return out_path


def _plot_subject_bars(ax, data: pd.DataFrame, value_col: str, group_col: str, title: str, ylabel: str) -> None:
    """Plot group means with SEM and overlaid subject/file-level points."""
    groups = list(dict.fromkeys(data[group_col].dropna().tolist()))
    x = np.arange(len(groups))
    means = []
    sems = []
    for group in groups:
        values = data.loc[data[group_col].eq(group), value_col]
        mean, sem, _ = _mean_sem(values)
        means.append(mean)
        sems.append(sem)
    ax.bar(x, means, yerr=sems, color="0.72", edgecolor="0.2", linewidth=0.7, capsize=2)
    for i, group in enumerate(groups):
        values = data.loc[data[group_col].eq(group), value_col].replace([np.inf, -np.inf], np.nan).dropna()
        jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) else []
        ax.scatter(np.full(len(values), i) + jitter, values, s=18, color="0.15", alpha=0.75, linewidth=0)
    ax.axhline(0, color="0.25", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def make_figure_01_overview(waveform: pd.DataFrame, spectral: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 1: dataset, stimulation current, QC, and analysis schematic."""
    set_publication_style()
    files = waveform.drop_duplicates("file").copy()
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), constrained_layout=True)
    freq_counts = files["stim_frequency_hz"].value_counts().sort_index()
    axes[0, 0].bar([f"{x:g} Hz" for x in freq_counts.index], freq_counts.values, color="#4C78A8")
    axes[0, 0].set_title("Stimulation files")
    axes[0, 0].set_ylabel("Files")
    axes[0, 0].bar_label(axes[0, 0].containers[0], padding=2, fontsize=7)

    for freq, group in files.groupby("stim_frequency_hz"):
        axes[0, 1].scatter(
            np.full(len(group), float(freq)),
            group["stim_amplitude_ma"],
            s=24,
            alpha=0.75,
            label=f"{freq:g} Hz",
        )
    axes[0, 1].set_xticks(sorted(files["stim_frequency_hz"].dropna().unique()))
    axes[0, 1].set_xlabel("Stimulation frequency (Hz)")
    axes[0, 1].set_ylabel("Current (mA)")
    axes[0, 1].set_title("Stimulation current")

    status_counts = waveform["status"].value_counts()
    axes[1, 0].barh(status_counts.index.astype(str), status_counts.values, color="0.45")
    axes[1, 0].set_title("Waveform QC status")
    axes[1, 0].set_xlabel("Rows")

    # This panel is a compact schematic of the shared analysis windows.
    axes[1, 1].hlines(1, -10, 41, color="0.3", linewidth=1.2)
    axes[1, 1].axvspan(-10, 0, color="#4C78A8", alpha=0.25, label="Pre")
    axes[1, 1].axvspan(6, 16, color="#E45756", alpha=0.25, label="50 Hz post")
    axes[1, 1].axvspan(31, 41, color="#72B7B2", alpha=0.25, label="1 Hz post")
    axes[1, 1].axvline(0, color="0.2", linewidth=0.8)
    axes[1, 1].set_yticks([])
    axes[1, 1].set_xlabel("Time from stimulation onset (s)")
    axes[1, 1].set_title("Primary analysis windows")
    axes[1, 1].legend(frameon=False, fontsize=7)
    fig.suptitle("Dataset and primary analysis design")
    out_path = out_dir / "figure_01_overview.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_figure_02_waveform(waveform_regions: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 2: subject/file-level primary waveform effects."""
    df = _clean_waveform(waveform_regions)
    if "region_analysis_include" in df.columns:
        df = df.loc[df["region_analysis_include"].fillna(False)].copy()
    file_level = (
        df.groupby(["file", "subject", "stim_frequency_hz"], dropna=False)
        .agg(delta_mean_if=("delta_mean_if", "mean"), delta_asc2desc=("delta_asc2desc", "mean"), delta_peak2trough=("delta_peak2trough", "mean"))
        .reset_index()
    )
    file_level["frequency"] = file_level["stim_frequency_hz"].map(lambda x: f"{x:g} Hz")
    set_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2), constrained_layout=True)
    _plot_subject_bars(axes[0], file_level, "delta_mean_if", "frequency", "Mean IF", "Post - pre IF (Hz)")
    _plot_subject_bars(axes[1], file_level, "delta_asc2desc", "frequency", "Ascent/descent", "Post - pre ratio")
    _plot_subject_bars(axes[2], file_level, "delta_peak2trough", "frequency", "Peak/trough", "Post - pre ratio")
    fig.suptitle("Primary Quinn-style waveform changes")
    out_path = out_dir / "figure_02_primary_waveform.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_figure_03_connectivity_distance(imf_joined: pd.DataFrame, distance_summary: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 3: IMF imaginary coherence by region and distance."""
    df = imf_joined.loc[imf_joined["analysis_pair_include"].fillna(False)].copy()
    df = df.loc[df["target_region_family"].notna() & df["target_region_family"].ne("non_gray")]
    df["imf_group"] = np.where(df["imf"].isin([1, 2, 3]), "IMF1-3", "IMF4-6")
    region = (
        df.groupby(["file", "subject", "stim_frequency_hz", "target_region_family", "imf_group"], dropna=False)
        .agg(delta_imcoh=("delta_imcoh", "mean"))
        .reset_index()
    )
    top_regions = region.groupby("target_region_family")["subject"].nunique().sort_values(ascending=False).head(6).index
    region = region.loc[region["target_region_family"].isin(top_regions)]
    summary = (
        region.groupby(["stim_frequency_hz", "target_region_family", "imf_group"], dropna=False)
        .agg(mean=("delta_imcoh", "mean"), sem=("delta_imcoh", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan))
        .reset_index()
    )
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)
    x = np.arange(len(top_regions))
    width = 0.18
    specs = [(1.0, "IMF1-3", -1.5 * width, "#2a9d8f"), (50.0, "IMF1-3", -0.5 * width, "#d62828"), (1.0, "IMF4-6", 0.5 * width, "#8dd3c7"), (50.0, "IMF4-6", 1.5 * width, "#f4a3a3")]
    for freq, imf_group, offset, color in specs:
        vals = summary.loc[summary["stim_frequency_hz"].eq(freq) & summary["imf_group"].eq(imf_group)].set_index("target_region_family").reindex(top_regions)
        axes[0].bar(x + offset, vals["mean"], width=width, yerr=vals["sem"], color=color, capsize=2, label=f"{freq:g} Hz {imf_group}")
    axes[0].axhline(0, color="0.25", linewidth=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(top_regions, rotation=35, ha="right")
    axes[0].set_ylabel("Post - pre imaginary coherence")
    axes[0].set_title("Seed-to-target IMF connectivity")
    axes[0].legend(frameon=False, fontsize=7)

    dist = distance_summary.copy()
    dist = dist.loc[dist["imf"].isin([1, 2, 3])]
    profile = dist.groupby(["stim_frequency_hz", "distance_bin"], observed=False).agg(mean=("delta_imcoh_mean", "mean"), sem=("delta_imcoh_mean", "sem")).reset_index()
    bins = ["near", "middle", "far"]
    for freq, color in [(1.0, "#2a9d8f"), (50.0, "#d62828")]:
        vals = profile.loc[profile["stim_frequency_hz"].eq(freq)].set_index("distance_bin").reindex(bins)
        axes[1].errorbar(np.arange(len(bins)), vals["mean"], yerr=vals["sem"], marker="o", color=color, label=f"{freq:g} Hz")
    axes[1].axhline(0, color="0.25", linewidth=0.7)
    axes[1].set_xticks(np.arange(len(bins)))
    axes[1].set_xticklabels([b.title() for b in bins])
    axes[1].set_xlabel("Seed-target distance bin")
    axes[1].set_ylabel("Post - pre imaginary coherence")
    axes[1].set_title("Connectivity as a function of distance")
    axes[1].legend(frameon=False)
    fig.suptitle("Primary IMF connectivity findings")
    out_path = out_dir / "figure_03_imf_connectivity_distance.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_figure_04_spectral_coupling(spectral_regions: pd.DataFrame, imf_joined: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 4: spectral band-power changes and waveform-connectivity coupling."""
    spec = spectral_regions.loc[spectral_regions["spectral_region_include"].fillna(False)].copy()
    file_level = (
        spec.groupby(["file", "subject", "stim_frequency_hz", "is_stimulated_channel"], dropna=False)
        .agg(delta_beta_log_power=("delta_beta_log_power", "mean"), delta_high_gamma_log_power=("delta_high_gamma_log_power", "mean"), delta_delta_log_power=("delta_delta_log_power", "mean"))
        .reset_index()
    )
    file_level["contact_class"] = np.where(file_level["is_stimulated_channel"], "Stimulated", "Non-stimulated")
    file_level["frequency"] = file_level["stim_frequency_hz"].map(lambda x: f"{x:g} Hz")

    conn = imf_joined.loc[imf_joined["analysis_pair_include"].fillna(False) & imf_joined["imf"].isin([1, 2, 3])].copy()
    coupling = (
        conn.groupby(["file", "subject", "stim_frequency_hz", "target_norm"], dropna=False)
        .agg(delta_imcoh=("delta_imcoh", "mean"), delta_mean_if=("delta_mean_if", "first"), distance=("seed_target_distance_mni_mm", "mean"))
        .reset_index()
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["delta_imcoh", "delta_mean_if"])
    )
    set_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), constrained_layout=True)
    # Show the most interpretable spectral bands for the primary figure.
    _plot_subject_bars(axes[0], file_level.loc[~file_level["is_stimulated_channel"]], "delta_beta_log_power", "frequency", "Non-stim beta power", "Post - pre log10 power")
    _plot_subject_bars(axes[1], file_level.loc[~file_level["is_stimulated_channel"]], "delta_high_gamma_log_power", "frequency", "Non-stim high gamma", "Post - pre log10 power")
    for freq, color in [(1.0, "#2a9d8f"), (50.0, "#d62828")]:
        sub = coupling.loc[coupling["stim_frequency_hz"].eq(freq)]
        axes[2].scatter(sub["delta_imcoh"], sub["delta_mean_if"], s=14, alpha=0.45, color=color, edgecolor="none", label=f"{freq:g} Hz")
    axes[2].axhline(0, color="0.25", linewidth=0.7)
    axes[2].axvline(0, color="0.25", linewidth=0.7)
    axes[2].set_xlabel("IMF1-3 post - pre imaginary coherence")
    axes[2].set_ylabel("Post - pre mean IF (Hz)")
    axes[2].set_title("Waveform-connectivity relationship")
    axes[2].legend(frameon=False)
    fig.suptitle("Spectral validation and waveform-connectivity coupling")
    out_path = out_dir / "figure_04_spectral_waveform_connectivity.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_extended_data_manifest(project_root: Path, extended_dir: Path) -> Path:
    """Copy broad exploratory figures into an Extended Data folder without deleting originals."""
    sources = [
        ("waveform_region_quinn", project_root / "outputs" / "waveform_relaxed" / "region_quinn" / "figures"),
        ("connectivity_exploratory", project_root / "outputs" / "connectivity" / "figures"),
        ("imf_connectivity_exploratory", project_root / "outputs" / "imf_connectivity" / "region_imf" / "figures"),
        ("spectral_exploratory", project_root / "outputs" / "spectral" / "figures"),
        ("localization_subject13", project_root / "outputs" / "localization" / "sub_13"),
    ]
    rows = []
    for label, folder in sources:
        if not folder.exists():
            continue
        target = extended_dir / label
        target.mkdir(parents=True, exist_ok=True)
        for path in folder.rglob("*.png"):
            copied = target / path.name
            if path.resolve() != copied.resolve():
                shutil.copy2(path, copied)
            rows.append({"category": label, "source": str(path), "extended_data_copy": str(copied)})
    manifest = pd.DataFrame(rows)
    out_path = extended_dir / "extended_data_manifest.csv"
    manifest.to_csv(out_path, index=False)
    return out_path


def run_primary_publication_outputs(project_root: Path) -> dict[str, Path]:
    """Generate final primary tables, primary figures, and Extended Data copies."""
    out_root = project_root / "outputs" / "publication"
    tables_dir = out_root / "primary_tables"
    figs_dir = out_root / "primary_figures"
    extended_dir = out_root / "extended_data"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    extended_dir.mkdir(parents=True, exist_ok=True)

    waveform = pd.read_csv(project_root / "outputs" / "waveform_relaxed" / "MASTER_all_channels_summary.csv")
    waveform_regions = pd.read_csv(project_root / "outputs" / "waveform_relaxed" / "region_quinn" / "MASTER_waveform_with_brainstorm_regions.csv")
    imf_joined = pd.read_csv(project_root / "outputs" / "imf_connectivity" / "region_imf" / "imf_connectivity_waveform_regions_join.csv")
    distance_summary = pd.read_csv(project_root / "outputs" / "imf_connectivity" / "region_imf" / "distance_bin_region_imf_summary.csv")
    spectral_regions = pd.read_csv(project_root / "outputs" / "spectral" / "MASTER_spectral_band_power_with_regions.csv")

    paths = {
        "cohort_table": make_cohort_table(waveform, tables_dir),
        "waveform_table": make_primary_waveform_table(waveform_regions, tables_dir),
        "imf_connectivity_table": make_primary_imf_table(imf_joined, tables_dir),
        "distance_table": make_distance_table(distance_summary, tables_dir),
        "waveform_connectivity_table": make_waveform_connectivity_table(imf_joined, tables_dir),
        "spectral_table": make_primary_spectral_table(spectral_regions, tables_dir),
        "figure_01": make_figure_01_overview(waveform, spectral_regions, figs_dir),
        "figure_02": make_figure_02_waveform(waveform_regions, figs_dir),
        "figure_03": make_figure_03_connectivity_distance(imf_joined, distance_summary, figs_dir),
        "figure_04": make_figure_04_spectral_coupling(spectral_regions, imf_joined, figs_dir),
        "extended_data_manifest": make_extended_data_manifest(project_root, extended_dir),
    }
    summary = pd.DataFrame([{"name": key, "path": str(value)} for key, value in paths.items()])
    summary_path = out_root / "primary_publication_outputs_manifest.csv"
    summary.to_csv(summary_path, index=False)
    paths["manifest"] = summary_path
    return paths
