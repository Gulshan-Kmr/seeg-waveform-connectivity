from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config
from .io import load_recording
from .pipeline import channel_diagnostics, safe_name


def set_publication_style() -> None:
    """Use a restrained, journal-like matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _selected_imf(result, imf_index: int) -> np.ndarray | None:
    if result.imf is None or result.imf.shape[1] <= imf_index:
        return None
    return result.imf[:, imf_index]


def _selected_2d(arr: np.ndarray | None, imf_index: int) -> np.ndarray | None:
    if arr is None or arr.ndim < 2 or arr.shape[1] <= imf_index:
        return None
    return arr[:, imf_index]


def _mean_and_sem(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(mat, axis=1)
    n = np.sum(np.isfinite(mat), axis=1)
    sd = np.nanstd(mat, axis=1, ddof=1)
    sem = sd / np.sqrt(np.maximum(n, 1))
    return mean, sem


def _mean_vectors(phase_profiles: np.ndarray) -> np.ndarray:
    phase = np.linspace(0, 2 * np.pi, phase_profiles.shape[0], endpoint=False)
    weights = np.exp(1j * phase)[:, None]
    centered = phase_profiles - np.nanmean(phase_profiles, axis=0, keepdims=True)
    denom = np.nansum(np.abs(centered), axis=0) + np.finfo(float).eps
    return np.nansum(centered * weights, axis=0) / denom


def _eligible_for_average(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    min_cycles = int(cfg["emd"].get("publication_min_cycles", 5))
    out = df[df["status"].eq("ok")].copy()
    if {"pre_n_cycles", "post_n_cycles"}.issubset(out.columns):
        out = out[(out["pre_n_cycles"] >= min_cycles) & (out["post_n_cycles"] >= min_cycles)]
    for col in ["pre_noise_flag", "post_noise_flag"]:
        if col in out.columns:
            out = out[~out[col].fillna(False).astype(bool)]
    return out


def _add_empty_message(ax, message: str = "No valid cycles selected") -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="0.35")
    ax.set_xticks([])
    ax.set_yticks([])


def _legend_if_any(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False)


def plot_channel_quinn_diagnostic(diag: dict, out_path: str | Path, dpi: int = 300) -> Path:
    set_publication_style()
    rec = diag["recording"]
    channel = diag["channel_name"]
    imf_index = diag["imf_index"]
    is_stimulated = channel in rec.stimulated_channel_names
    pre = diag["pre"]
    post = diag["post"]
    pre_res = pre["result"]
    post_res = post["result"]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 18), constrained_layout=True)
    gs = fig.add_gridspec(6, 2)

    fig.suptitle(
        f"{rec.file_stem} | {channel} | IMF{imf_index + 1} | "
        f"{rec.stim_frequency_hz:g} Hz, {rec.stim_amplitude_ma:g} mA | "
        f"{'stimulated contact' if is_stimulated else 'non-stimulated contact'}",
        fontsize=10,
    )

    # Raw + selected IMF.
    for col, segment, color, title in [(0, pre, "tab:blue", "Pre"), (1, post, "tab:red", "Post")]:
        ax = fig.add_subplot(gs[0, col])
        res = segment["result"]
        imf = _selected_imf(res, imf_index)
        t = segment["time"]
        ax.plot(t, segment["processed"], color="0.25", linewidth=0.8, label="Processed SEEG")
        if imf is not None:
            ax.plot(t[: imf.size], imf, color=color, linewidth=1.0, label=f"IMF{imf_index + 1}")
        ax.set_title(f"{title}: processed signal + selected IMF")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        _legend_if_any(ax)

    # Phase and instantaneous frequency.
    for col, segment, color, title in [(0, pre, "tab:blue", "Pre"), (1, post, "tab:red", "Post")]:
        ax = fig.add_subplot(gs[1, col])
        res = segment["result"]
        t = segment["time"]
        ip = _selected_2d(res.ip, imf_index)
        ifreq = _selected_2d(res.ifreq, imf_index)
        if ip is not None:
            ax.plot(t[: ip.size], np.angle(np.exp(1j * ip)), color=color, alpha=0.5, linewidth=0.7, label="Wrapped phase")
        ax2 = ax.twinx()
        if ifreq is not None:
            ax2.plot(t[: ifreq.size], ifreq, color="0.15", linewidth=0.8, label="IF")
        if ip is None and ifreq is None:
            _add_empty_message(ax, "No phase/IF output")
        ax.set_title(f"{title}: phase and instantaneous frequency")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Phase (rad)")
        ax2.set_ylabel("IF (Hz)")

    # Control-point metrics.
    ax = fig.add_subplot(gs[2, 0])
    any_points = False
    for table, color, label in [(pre_res.cycle_table, "tab:blue", "Pre"), (post_res.cycle_table, "tab:red", "Post")]:
        if table is not None and {"peak2trough", "asc2desc"}.issubset(table.columns):
            ax.scatter(table["peak2trough"], table["asc2desc"], s=14, alpha=0.45, color=color, label=label)
            any_points = True
    if not any_points:
        _add_empty_message(ax)
    ax.set_xlabel("Peak-to-trough ratio")
    ax.set_ylabel("Ascent-to-descent ratio")
    ax.set_title("Cycle-wise control-point ratios")
    _legend_if_any(ax)

    ax = fig.add_subplot(gs[2, 1])
    box_data = []
    labels = []
    for table, prefix in [(pre_res.cycle_table, "Pre"), (post_res.cycle_table, "Post")]:
        if table is not None:
            for metric in ["peak2trough", "asc2desc"]:
                if metric in table:
                    box_data.append(table[metric].dropna().to_numpy())
                    labels.append(f"{prefix}\n{metric}")
    if box_data:
        ax.boxplot(box_data, labels=labels, showfliers=False)
    else:
        _add_empty_message(ax)
    ax.set_title("Control-point distributions")
    ax.set_ylabel("Ratio")

    # Phase-aligned IF heatmaps.
    for col, res, title in [(0, pre_res, "Pre"), (1, post_res, "Post")]:
        ax = fig.add_subplot(gs[3, col])
        if res.phase_aligned_if is not None:
            phase = np.linspace(0, 2 * np.pi, res.phase_aligned_if.shape[0])
            im = ax.imshow(
                res.phase_aligned_if.T,
                aspect="auto",
                origin="lower",
                extent=[phase[0], phase[-1], 1, res.phase_aligned_if.shape[1]],
            )
            fig.colorbar(im, ax=ax, shrink=0.85, label="IF (Hz)")
        else:
            _add_empty_message(ax)
        ax.set_title(f"{title}: phase-aligned IF")
        ax.set_xlabel("Phase (rad)")
        ax.set_ylabel("Cycle")

    # Mean phase-aligned IF and normalized waveform.
    ax = fig.add_subplot(gs[4, 0])
    any_line = False
    for res, color, label in [(pre_res, "tab:blue", "Pre"), (post_res, "tab:red", "Post")]:
        if res.phase_aligned_if is not None:
            phase = np.linspace(0, 2 * np.pi, res.phase_aligned_if.shape[0])
            ax.plot(phase, res.phase_aligned_if, color=color, alpha=0.08, linewidth=0.45)
            mean, sem = _mean_and_sem(res.phase_aligned_if)
            ax.plot(phase, mean, color=color, linewidth=2.0, label=f"{label} mean")
            ax.fill_between(phase, mean - sem, mean + sem, color=color, alpha=0.2, linewidth=0)
            any_line = True
    if not any_line:
        _add_empty_message(ax)
    ax.set_title("Mean phase-aligned IF")
    ax.set_xlabel("Phase (rad)")
    ax.set_ylabel("IF (Hz)")
    _legend_if_any(ax)

    ax = fig.add_subplot(gs[4, 1])
    any_line = False
    for res, color, label in [(pre_res, "tab:blue", "Pre"), (post_res, "tab:red", "Post")]:
        if res.normalized_waveform is not None:
            x = np.linspace(0, 1, res.normalized_waveform.shape[0])
            ax.plot(x, res.normalized_waveform, color=color, alpha=0.08, linewidth=0.45)
            mean, sem = _mean_and_sem(res.normalized_waveform)
            ax.plot(x, mean, color=color, linewidth=2.0, label=f"{label} mean")
            ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.2, linewidth=0)
            any_line = True
    ax.plot(np.linspace(0, 1, 200), np.sin(np.linspace(0, 2 * np.pi, 200)), "k--", linewidth=0.9, label="Sine ref")
    if not any_line:
        _add_empty_message(ax)
    ax.set_title("Mean normalized waveform")
    ax.set_xlabel("Normalized cycle")
    ax.set_ylabel("Amplitude")
    _legend_if_any(ax)

    # Mean-vector distributions and QC text.
    ax = fig.add_subplot(gs[5, 0])
    any_points = False
    for res, color, label in [(pre_res, "tab:blue", "Pre"), (post_res, "tab:red", "Post")]:
        if res.phase_aligned_if is not None:
            vectors = _mean_vectors(res.phase_aligned_if)
            ax.scatter(np.real(vectors), np.imag(vectors), s=12, alpha=0.45, color=color, label=label)
            any_points = True
    if not any_points:
        _add_empty_message(ax)
    ax.axhline(0, color="0.7", linewidth=0.8)
    ax.axvline(0, color="0.7", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Cycle IF-profile mean vectors")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    _legend_if_any(ax)

    ax = fig.add_subplot(gs[5, 1])
    ax.axis("off")
    text = "\n".join(
        [
            f"Pre status: {pre_res.status}, cycles: {pre_res.n_cycles}",
            f"Post status: {post_res.status}, cycles: {post_res.n_cycles}",
            f"Contact type: {'stimulated' if is_stimulated else 'non-stimulated'}",
            f"Stimulated pair: {'; '.join(rec.stimulated_channel_names)}",
            f"Pre drift: {pre['drift'].flagged}, ratio: {pre['drift'].slope_iqr_ratio:.3f}",
            f"Post drift: {post['drift'].flagged}, ratio: {post['drift'].slope_iqr_ratio:.3f}",
            f"Pre noise: {pre['noise'].flagged}, line ratio: {pre['noise'].line_noise_ratio:.4f}",
            f"Post noise: {post['noise'].flagged}, line ratio: {post['noise'].line_noise_ratio:.4f}",
            f"Pre window: {pre['window'][0]:.1f} to {pre['window'][1]:.1f} s",
            f"Post window: {post['window'][0]:.1f} to {post['window'][1]:.1f} s",
        ]
    )
    ax.text(0.0, 1.0, text, va="top", ha="left", family="monospace", fontsize=10)

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def make_selected_quinn_figures(
    config_path: str | Path,
    master_csv: str | Path | None = None,
    top_n: int = 10,
    include_noise_flagged: bool = True,
) -> list[Path]:
    cfg = load_config(config_path)
    out_root = Path(cfg["project"]["output_dir"])
    master_csv = Path(master_csv) if master_csv else out_root / "MASTER_all_channels_summary.csv"
    df = pd.read_csv(master_csv)

    ok = df[df["status"].eq("ok")].copy()
    selected = []
    for metric in ["pa_if_change_score", "norm_waveform_change_score"]:
        if metric in ok:
            selected.append(ok.sort_values(metric, ascending=False).head(top_n))

    if include_noise_flagged:
        noise_cols = [c for c in ["pre_noise_flag", "post_noise_flag"] if c in df.columns]
        if noise_cols:
            noise = df[df[noise_cols].fillna(False).astype(bool).any(axis=1)].copy()
            selected.append(noise.head(top_n))

    selected_df = pd.concat(selected, ignore_index=True).drop_duplicates(["file", "channel"])
    out_dir = out_root / "figures" / "quinn_diagnostics"
    paths: list[Path] = []
    for _, row in selected_df.iterrows():
        rec = load_recording(row["path"])
        diag = channel_diagnostics(rec, row["channel"], cfg)
        name = f"{safe_name(row['file'])}__{safe_name(row['channel'])}__quinn_diagnostic.png"
        paths.append(plot_channel_quinn_diagnostic(diag, out_dir / name, dpi=cfg["outputs"].get("figure_dpi", 300)))
    return paths


def _channel_npz_path(out_root: Path, row: pd.Series, imf_index: int) -> Path:
    return out_root / row["file"] / f"{safe_name(row['channel'])}_IMF{imf_index + 1}_outputs.npz"


def _collect_channel_means(df: pd.DataFrame, out_root: Path, imf_index: int, key: str) -> tuple[np.ndarray, np.ndarray]:
    pre_means = []
    post_means = []
    for _, row in df.iterrows():
        npz_path = _channel_npz_path(out_root, row, imf_index)
        if not npz_path.exists():
            continue
        z = np.load(npz_path, allow_pickle=True)
        pre = z.get(f"pre_{key}")
        post = z.get(f"post_{key}")
        if pre is None or post is None:
            continue
        if pre.ndim != 2 or post.ndim != 2:
            continue
        pre_means.append(np.nanmean(pre, axis=1))
        post_means.append(np.nanmean(post, axis=1))
    if not pre_means:
        return np.empty((0, 0)), np.empty((0, 0))
    min_len = min([x.size for x in pre_means + post_means])
    return np.vstack([x[:min_len] for x in pre_means]), np.vstack([x[:min_len] for x in post_means])


def plot_population_averages(config_path: str | Path, master_csv: str | Path | None = None) -> list[Path]:
    set_publication_style()
    cfg = load_config(config_path)
    out_root = Path(cfg["project"]["output_dir"])
    master_csv = Path(master_csv) if master_csv else out_root / "MASTER_all_channels_summary.csv"
    df = pd.read_csv(master_csv)
    df = _eligible_for_average(df, cfg)
    imf_index = int(cfg["emd"].get("imf_index", 3))
    out_dir = out_root / "figures" / "population_averages"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    group_specs = [("all", df)]
    for freq in sorted(df["stim_frequency_hz"].dropna().unique()):
        group_specs.append((f"{freq:g}Hz", df[df["stim_frequency_hz"].eq(freq)]))
    if "is_stimulated_channel" in df:
        group_specs.append(("stimulated_contacts", df[df["is_stimulated_channel"].astype(bool)]))
        group_specs.append(("nonstimulated_contacts", df[~df["is_stimulated_channel"].astype(bool)]))
        for freq in sorted(df["stim_frequency_hz"].dropna().unique()):
            freq_df = df[df["stim_frequency_hz"].eq(freq)]
            group_specs.append(
                (
                    f"{freq:g}Hz_stimulated_contacts",
                    freq_df[freq_df["is_stimulated_channel"].astype(bool)],
                )
            )
            group_specs.append(
                (
                    f"{freq:g}Hz_nonstimulated_contacts",
                    freq_df[~freq_df["is_stimulated_channel"].astype(bool)],
                )
            )

    for label, group in group_specs:
        if group.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
        fig.suptitle(f"Population average Quinn profiles | {label} | n={len(group)} channels")

        pre_if, post_if = _collect_channel_means(group, out_root, imf_index, "pa_if")
        ax = axes[0]
        if pre_if.size and post_if.size:
            phase = np.linspace(0, 2 * np.pi, pre_if.shape[1])
            for arr, color, name in [(pre_if, "tab:blue", "Pre"), (post_if, "tab:red", "Post")]:
                ax.plot(phase, arr.T, color=color, alpha=0.045, linewidth=0.35)
                mean = np.nanmean(arr, axis=0)
                sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
                ax.plot(phase, mean, color=color, linewidth=2.0, label=f"{name} mean")
                ax.fill_between(phase, mean - sem, mean + sem, color=color, alpha=0.2, linewidth=0)
        else:
            _add_empty_message(ax)
        ax.set_title("Phase-aligned IF")
        ax.set_xlabel("Phase (rad)")
        ax.set_ylabel("IF (Hz)")
        _legend_if_any(ax)

        pre_norm, post_norm = _collect_channel_means(group, out_root, imf_index, "norm_waveform")
        ax = axes[1]
        if pre_norm.size and post_norm.size:
            x = np.linspace(0, 1, pre_norm.shape[1])
            for arr, color, name in [(pre_norm, "tab:blue", "Pre"), (post_norm, "tab:red", "Post")]:
                ax.plot(x, arr.T, color=color, alpha=0.045, linewidth=0.35)
                mean = np.nanmean(arr, axis=0)
                sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
                ax.plot(x, mean, color=color, linewidth=2.0, label=f"{name} mean")
                ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.2, linewidth=0)
            ax.plot(np.linspace(0, 1, 200), np.sin(np.linspace(0, 2 * np.pi, 200)), "k--", linewidth=0.8, label="Sine ref")
        else:
            _add_empty_message(ax)
        ax.set_title("Normalized waveform")
        ax.set_xlabel("Normalized cycle")
        ax.set_ylabel("Amplitude")
        _legend_if_any(ax)

        out_path = out_dir / f"{safe_name(label)}_population_average.png"
        fig.savefig(out_path, dpi=cfg["outputs"].get("figure_dpi", 300))
        plt.close(fig)
        paths.append(out_path)
    return paths
