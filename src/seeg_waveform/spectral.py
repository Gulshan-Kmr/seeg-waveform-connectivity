from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats

from .config import load_config
from .io import iter_mat_files, load_recording
from .localization import normalize_channel_name
from .pipeline import channel_is_stimulated
from .preprocess import post_window_for_frequency, preprocess_segment, window_mask
from .quinn_figures import set_publication_style
from .region_waveform import add_region_family, subject_number
from .spatial_analysis import bh_fdr


BAND_LABELS = {
    "delta": "Delta",
    "theta": "Theta",
    "alpha": "Alpha",
    "beta": "Beta",
    "low_gamma": "Low gamma",
    "high_gamma": "High gamma",
}


def _band_label(name: object) -> str:
    return BAND_LABELS.get(str(name), str(name).replace("_", " ").title())


def _metric_label(metric: str) -> str:
    if metric == "log_power":
        return "log10 band power"
    if metric == "relative_power":
        return "relative band power"
    return metric.replace("_", " ")


def _band_power(freqs: np.ndarray, pxx: np.ndarray, band: tuple[float, float]) -> float:
    lo, hi = band
    mask = (freqs >= lo) & (freqs < hi) & np.isfinite(pxx)
    if mask.sum() < 2:
        return np.nan
    return float(np.trapezoid(pxx[mask], freqs[mask]))


def _welch_psd(x: np.ndarray, fs: float, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(len(x), max(8, int(round(float(cfg["welch_nperseg_sec"]) * fs))))
    noverlap = int(round(nperseg * float(cfg.get("welch_overlap", 0.5))))
    freqs, pxx = signal.welch(
        np.asarray(x, dtype=float),
        fs=fs,
        nperseg=nperseg,
        noverlap=min(noverlap, nperseg - 1),
        detrend=False,
        scaling="density",
    )
    return freqs, pxx


def _spectral_features(x: np.ndarray, fs: float, spectral_cfg: dict) -> dict[str, float]:
    freqs, pxx = _welch_psd(x, fs, spectral_cfg)
    bands = {name: tuple(vals) for name, vals in spectral_cfg["frequency_bands"].items()}
    total_band = tuple(spectral_cfg.get("total_power_band", [1.0, min(120.0, fs / 2.0)]))
    total_power = _band_power(freqs, pxx, total_band)
    out: dict[str, float] = {"total_power": total_power, "log_total_power": np.log10(total_power) if total_power > 0 else np.nan}
    for name, band in bands.items():
        power = _band_power(freqs, pxx, band)
        out[f"{name}_power"] = power
        out[f"{name}_log_power"] = np.log10(power) if power > 0 else np.nan
        out[f"{name}_relative_power"] = power / total_power if total_power and total_power > 0 else np.nan
    return out


def analyze_spectral_file(path: str | Path, cfg: dict) -> pd.DataFrame:
    rec = load_recording(path)
    min_channels = int(cfg.get("preprocessing", {}).get("min_channels_per_file", 1))
    if rec.signal.shape[1] < min_channels:
        return pd.DataFrame(
            [
                {
                    "file": rec.file_stem,
                    "path": str(rec.path),
                    "subject": rec.subject,
                    "status": "excluded_few_channels",
                    "n_channels": rec.signal.shape[1],
                }
            ]
        )
    pre_start, pre_stop = cfg["windows"]["pre"]
    post_start, post_stop = post_window_for_frequency(
        rec.stim_frequency_hz,
        cfg["windows"]["stim_end_by_frequency"],
        cfg["windows"]["post_duration_sec"],
        cfg["windows"]["post_artifact_buffer_sec"],
    )
    pre_idx = window_mask(rec.time, pre_start, pre_stop)
    post_idx = window_mask(rec.time, post_start, post_stop)
    rows = []
    for channel_index, channel in enumerate(rec.channel_names):
        pre_x, pre_drift, pre_noise = preprocess_segment(
            rec.signal[pre_idx, channel_index], rec.time[pre_idx], rec.fs, cfg["preprocessing"]
        )
        post_x, post_drift, post_noise = preprocess_segment(
            rec.signal[post_idx, channel_index], rec.time[post_idx], rec.fs, cfg["preprocessing"]
        )
        row = {
            "file": rec.file_stem,
            "path": str(rec.path),
            "subject": rec.subject,
            "subject_number": subject_number(rec.subject),
            "entry_index": rec.entry_index,
            "stim_frequency_hz": rec.stim_frequency_hz,
            "stim_amplitude_ma": rec.stim_amplitude_ma,
            "channel": channel,
            "channel_norm": normalize_channel_name(channel),
            "channel_index": channel_index,
            "is_stimulated_channel": channel_is_stimulated(channel, rec.stimulated_channel_names),
            "fs": rec.fs,
            "pre_window_start": pre_start,
            "pre_window_stop": pre_stop,
            "post_window_start": post_start,
            "post_window_stop": post_stop,
            "pre_drift_flag": pre_drift.flagged,
            "post_drift_flag": post_drift.flagged,
            "pre_drift_ratio": pre_drift.slope_iqr_ratio,
            "post_drift_ratio": post_drift.slope_iqr_ratio,
            "pre_noise_flag": pre_noise.flagged,
            "post_noise_flag": post_noise.flagged,
            "pre_line_noise_ratio": pre_noise.line_noise_ratio,
            "post_line_noise_ratio": post_noise.line_noise_ratio,
            "status": "ok_with_noise_flag" if (pre_noise.flagged or post_noise.flagged) else "ok",
        }
        pre_features = _spectral_features(pre_x, rec.fs, cfg["spectral"])
        post_features = _spectral_features(post_x, rec.fs, cfg["spectral"])
        for key, value in pre_features.items():
            row[f"pre_{key}"] = value
        for key, value in post_features.items():
            row[f"post_{key}"] = value
        for key in pre_features:
            row[f"delta_{key}"] = post_features[key] - pre_features[key]
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_spectral_dataset(config_path: str | Path, max_files: int | None = None) -> pd.DataFrame:
    cfg = load_config(config_path)
    files = list(iter_mat_files(cfg["project"]["data_dir"]))
    if max_files is not None:
        files = files[:max_files]
    frames = [analyze_spectral_file(path, cfg) for path in files]
    master = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_root = Path(cfg["project"]["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_root / "MASTER_spectral_band_power.csv", index=False)
    return master


def summarize_spectral(master: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    bands = list(cfg["spectral"]["frequency_bands"])
    ok = master.loc[
        master["status"].astype(str).str.startswith("ok")
        & ~master["pre_noise_flag"].fillna(False).astype(bool)
        & ~master["post_noise_flag"].fillna(False).astype(bool)
    ].copy()
    rows = []
    group_defs = [
        ("all", []),
        ("stim_frequency_hz", ["stim_frequency_hz"]),
        ("contact_class", ["is_stimulated_channel"]),
        ("frequency_contact", ["stim_frequency_hz", "is_stimulated_channel"]),
    ]
    for band in bands:
        for metric in [f"{band}_log_power", f"{band}_relative_power"]:
            delta_col = f"delta_{metric}"
            for level, group_cols in group_defs:
                groups = [((), ok)] if not group_cols else ok.groupby(group_cols, dropna=False)
                for keys, group in groups:
                    keys = keys if isinstance(keys, tuple) else (keys,)
                    values = group[delta_col].replace([np.inf, -np.inf], np.nan).dropna()
                    row = {
                        "band": band,
                        "metric": metric,
                        "level": level,
                        "n": int(len(values)),
                        "n_subjects": int(group["subject"].nunique()),
                        "n_files": int(group["file"].nunique()),
                        "mean_delta": float(values.mean()) if len(values) else np.nan,
                        "median_delta": float(values.median()) if len(values) else np.nan,
                    }
                    for col, key in zip(group_cols, keys):
                        row[col] = key
                    if len(values) >= 4 and values.nunique() > 1:
                        row["wilcoxon_p"] = float(stats.wilcoxon(values).pvalue)
                    else:
                        row["wilcoxon_p"] = np.nan
                    rows.append(row)
    out = pd.DataFrame(rows)
    out["wilcoxon_p_fdr"] = bh_fdr(out["wilcoxon_p"])
    return out


def join_spectral_to_regions(spectral: pd.DataFrame, waveform_regions_csv: Path) -> pd.DataFrame:
    regions = pd.read_csv(waveform_regions_csv)
    keep = [
        "subject_number",
        "file",
        "channel_norm",
        "primary_region",
        "region_family",
        "analysis_gray_matter_flag",
        "hemisphere",
        "x_mni",
        "y_mni",
        "z_mni",
        "delta_mean_if",
        "delta_asc2desc",
        "delta_peak2trough",
    ]
    regions = add_region_family(regions)
    merged = spectral.merge(
        regions[[c for c in keep if c in regions.columns]].drop_duplicates(["subject_number", "file", "channel_norm"]),
        on=["subject_number", "file", "channel_norm"],
        how="left",
    )
    merged["spectral_region_include"] = (
        merged["status"].astype(str).str.startswith("ok")
        & merged["analysis_gray_matter_flag"].fillna(False).astype(bool)
        & ~merged["pre_noise_flag"].fillna(False).astype(bool)
        & ~merged["post_noise_flag"].fillna(False).astype(bool)
    )
    return merged


def summarize_spectral_regions(merged: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    bands = list(cfg["spectral"]["frequency_bands"])
    df = merged.loc[merged["spectral_region_include"].fillna(False)].copy()
    rows = []
    for band in bands:
        for metric in [f"delta_{band}_log_power", f"delta_{band}_relative_power"]:
            for keys, group in df.groupby(["stim_frequency_hz", "region_family", "is_stimulated_channel"], dropna=False):
                values = group[metric].replace([np.inf, -np.inf], np.nan).dropna()
                row = dict(zip(["stim_frequency_hz", "region_family", "is_stimulated_channel"], keys))
                row.update(
                    {
                        "band": band,
                        "metric": metric,
                        "n": int(len(values)),
                        "n_subjects": int(group["subject"].nunique()),
                        "n_files": int(group["file"].nunique()),
                        "mean_delta": float(values.mean()) if len(values) else np.nan,
                        "median_delta": float(values.median()) if len(values) else np.nan,
                    }
                )
                if len(values) >= 4 and values.nunique() > 1:
                    row["wilcoxon_p"] = float(stats.wilcoxon(values).pvalue)
                else:
                    row["wilcoxon_p"] = np.nan
                rows.append(row)
    out = pd.DataFrame(rows)
    out["wilcoxon_p_fdr"] = bh_fdr(out["wilcoxon_p"])
    return out


def compare_frequency_spectral(master: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    bands = list(cfg["spectral"]["frequency_bands"])
    ok = master.loc[
        master["status"].astype(str).str.startswith("ok")
        & ~master["pre_noise_flag"].fillna(False).astype(bool)
        & ~master["post_noise_flag"].fillna(False).astype(bool)
    ].copy()
    file_level = (
        ok.groupby(["file", "subject", "stim_frequency_hz", "is_stimulated_channel"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    rows = []
    for band in bands:
        for metric in [f"delta_{band}_log_power", f"delta_{band}_relative_power"]:
            for stim_state, group in file_level.groupby("is_stimulated_channel", dropna=False):
                one = group.loc[group["stim_frequency_hz"].eq(1.0), metric].replace([np.inf, -np.inf], np.nan).dropna()
                fifty = group.loc[group["stim_frequency_hz"].eq(50.0), metric].replace([np.inf, -np.inf], np.nan).dropna()
                row = {
                    "band": band,
                    "metric": metric,
                    "is_stimulated_channel": stim_state,
                    "n_1hz": int(len(one)),
                    "n_50hz": int(len(fifty)),
                    "n_subjects_1hz": int(group.loc[group["stim_frequency_hz"].eq(1.0), "subject"].nunique()),
                    "n_subjects_50hz": int(group.loc[group["stim_frequency_hz"].eq(50.0), "subject"].nunique()),
                    "mean_1hz": float(one.mean()) if len(one) else np.nan,
                    "mean_50hz": float(fifty.mean()) if len(fifty) else np.nan,
                }
                row["mean_50_minus_1hz"] = row["mean_50hz"] - row["mean_1hz"]
                if len(one) >= 3 and len(fifty) >= 3 and one.nunique() > 1 and fifty.nunique() > 1:
                    row["mannwhitney_p"] = float(stats.mannwhitneyu(fifty, one, alternative="two-sided").pvalue)
                else:
                    row["mannwhitney_p"] = np.nan
                rows.append(row)
    out = pd.DataFrame(rows)
    out["mannwhitney_p_fdr"] = bh_fdr(out["mannwhitney_p"])
    return out


def plot_spectral_band_bars(summary: pd.DataFrame, out_path: Path, metric: str = "log_power") -> Path:
    set_publication_style()
    df = summary.loc[
        summary["level"].eq("frequency_contact")
        & summary["metric"].eq(f"{summary['band'].iloc[0]}_{metric}")  # placeholder overwritten below
    ].copy()
    frames = []
    for band in summary["band"].dropna().unique():
        frames.append(
            summary.loc[
                summary["level"].eq("frequency_contact") & summary["metric"].eq(f"{band}_{metric}")
            ].assign(plot_band=band)
        )
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    bands = list(dict.fromkeys(df["plot_band"].tolist()))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True, constrained_layout=True)
    colors = {False: "#4C78A8", True: "#E45756"}
    labels = {False: "Non-stimulated", True: "Stimulated"}
    for ax, freq in zip(axes, sorted(df["stim_frequency_hz"].dropna().unique())):
        sub = df.loc[df["stim_frequency_hz"].eq(freq)]
        x = np.arange(len(bands))
        width = 0.36
        for offset, stim_state in [(-width / 2, False), (width / 2, True)]:
            vals = (
                sub.loc[sub["is_stimulated_channel"].eq(stim_state)]
                .set_index("plot_band")
                .reindex(bands)["mean_delta"]
                .to_numpy(float)
            )
            ax.bar(x + offset, vals, width=width, color=colors[stim_state], label=labels[stim_state])
        ax.axhline(0, color="0.25", linewidth=0.7)
        ax.set_title(f"{freq:g} Hz")
        ax.set_xticks(x)
        ax.set_xticklabels([_band_label(b) for b in bands], rotation=35, ha="right")
        ax.set_xlabel("Frequency band")
        ax.set_ylabel(f"Post - pre {_metric_label(metric)}")
    axes[0].legend(frameon=False)
    fig.suptitle("Pre-to-post spectral band-power change")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_spectral_frequency_contrast(contrast: pd.DataFrame, out_path: Path, metric: str = "log_power") -> Path:
    set_publication_style()
    df = contrast.loc[contrast["metric"].str.endswith(f"_{metric}")].copy()
    bands = list(dict.fromkeys(df["band"].tolist()))
    fig, ax = plt.subplots(figsize=(6.8, 3.4), constrained_layout=True)
    x = np.arange(len(bands))
    width = 0.36
    colors = {False: "#4C78A8", True: "#E45756"}
    labels = {False: "Non-stimulated", True: "Stimulated"}
    for offset, stim_state in [(-width / 2, False), (width / 2, True)]:
        sub = df.loc[df["is_stimulated_channel"].eq(stim_state)].set_index("band").reindex(bands)
        vals = sub["mean_50_minus_1hz"].to_numpy(float)
        ax.bar(x + offset, vals, width=width, color=colors[stim_state], label=labels[stim_state])
        for xi, value, pval in zip(x + offset, vals, sub["mannwhitney_p_fdr"].to_numpy(float)):
            if np.isfinite(value) and np.isfinite(pval) and pval < 0.05:
                ax.text(xi, value, "*", ha="center", va="bottom" if value >= 0 else "top", fontsize=10)
    ax.axhline(0, color="0.25", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([_band_label(b) for b in bands], rotation=35, ha="right")
    ax.set_xlabel("Frequency band")
    ax.set_ylabel(f"50 Hz - 1 Hz change in {_metric_label(metric)}")
    ax.set_title("Stimulation-frequency contrast")
    ax.legend(frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _metric_band_name(metric: str) -> str:
    text = str(metric)
    if text.startswith("delta_"):
        text = text[len("delta_") :]
    if text.endswith("_log_power"):
        text = text[: -len("_log_power")]
    return text


def plot_region_spectral_heatmap(region_summary: pd.DataFrame, out_path: Path, metric: str = "delta_alpha_log_power") -> Path:
    set_publication_style()
    df = region_summary.loc[
        region_summary["metric"].eq(metric)
        & region_summary["region_family"].notna()
        & region_summary["region_family"].ne("non_gray")
        & region_summary["is_stimulated_channel"].eq(False)
    ].copy()
    families = df.groupby("region_family")["n_subjects"].max().sort_values(ascending=False).head(8).index.tolist()
    freqs = sorted(df["stim_frequency_hz"].dropna().unique())
    fig, axes = plt.subplots(1, len(freqs), figsize=(4.0 * len(freqs), 3.8), constrained_layout=True)
    axes = np.atleast_1d(axes)
    vmax = np.nanpercentile(np.abs(df["mean_delta"]), 95) if df["mean_delta"].notna().any() else 1.0
    vmax = max(float(vmax), 1e-6)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#e6e6e6")
    im = None
    bands = [_metric_band_name(b) for b in region_summary["metric"].unique() if str(b).endswith("_log_power")]
    for ax, freq in zip(axes, freqs):
        sub = region_summary.loc[
            region_summary["stim_frequency_hz"].eq(freq)
            & region_summary["metric"].str.endswith("_log_power")
            & region_summary["is_stimulated_channel"].eq(False)
        ].copy()
        sub["band"] = sub["metric"].map(_metric_band_name)
        pivot = sub.pivot_table(index="region_family", columns="band", values="mean_delta", aggfunc="mean")
        pivot = pivot.reindex(index=families, columns=bands)
        im = ax.imshow(np.ma.masked_invalid(pivot.to_numpy(float)), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"{freq:g} Hz non-stim targets")
        ax.set_yticks(np.arange(len(families)))
        ax.set_yticklabels(families)
        ax.set_xticks(np.arange(len(bands)))
        ax.set_xlabel("Frequency band")
        ax.set_xticklabels([_band_label(b) for b in bands], rotation=35, ha="right")
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.8, label="Post - pre log10 band power")
    fig.suptitle("Region-wise pre-to-post spectral power change")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_spectral_analysis(
    config_path: str | Path,
    waveform_regions_csv: str | Path | None = None,
    max_files: int | None = None,
) -> dict[str, Path]:
    cfg = load_config(config_path)
    out_root = Path(cfg["project"]["output_dir"])
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    master = analyze_spectral_dataset(config_path, max_files=max_files)
    summary = summarize_spectral(master, cfg)
    summary_path = out_root / "spectral_band_power_summary.csv"
    summary.to_csv(summary_path, index=False)
    contrast = compare_frequency_spectral(master, cfg)
    contrast_path = out_root / "spectral_50hz_vs_1hz_contrast.csv"
    contrast.to_csv(contrast_path, index=False)
    bar_log = plot_spectral_band_bars(summary, fig_dir / "spectral_log_power_bars.png", metric="log_power")
    bar_rel = plot_spectral_band_bars(summary, fig_dir / "spectral_relative_power_bars.png", metric="relative_power")
    contrast_log = plot_spectral_frequency_contrast(
        contrast, fig_dir / "spectral_50hz_vs_1hz_log_power_contrast.png", metric="log_power"
    )
    contrast_rel = plot_spectral_frequency_contrast(
        contrast, fig_dir / "spectral_50hz_vs_1hz_relative_power_contrast.png", metric="relative_power"
    )
    paths = {
        "master": out_root / "MASTER_spectral_band_power.csv",
        "summary": summary_path,
        "frequency_contrast": contrast_path,
        "log_power_bars": bar_log,
        "relative_power_bars": bar_rel,
        "log_power_contrast": contrast_log,
        "relative_power_contrast": contrast_rel,
    }
    if waveform_regions_csv is not None and Path(waveform_regions_csv).exists():
        merged = join_spectral_to_regions(master, Path(waveform_regions_csv))
        merged_path = out_root / "MASTER_spectral_band_power_with_regions.csv"
        merged.to_csv(merged_path, index=False)
        region_summary = summarize_spectral_regions(merged, cfg)
        region_summary_path = out_root / "spectral_region_family_summary.csv"
        region_summary.to_csv(region_summary_path, index=False)
        heat = plot_region_spectral_heatmap(region_summary, fig_dir / "spectral_region_log_power_heatmap.png")
        paths.update({"region_master": merged_path, "region_summary": region_summary_path, "region_heatmap": heat})
    return paths
