from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats

from .config import load_config
from .io import SeegRecording, iter_mat_files, load_recording
from .pipeline import channel_is_stimulated
from .preprocess import post_window_for_frequency, preprocess_segment, window_mask
from .quinn import SegmentResult, analyze_segment
from .quinn_figures import set_publication_style


@dataclass
class ChannelImfs:
    channel: str
    channel_index: int
    result: SegmentResult
    drift_flag: bool
    noise_flag: bool


def _epoch_starts(n_samples: int, fs: float, epoch_length_sec: float, overlap: float) -> list[tuple[int, int]]:
    nper = int(round(epoch_length_sec * fs))
    if nper <= 0 or n_samples < nper:
        return []
    step = max(1, int(round(nper * (1.0 - overlap))))
    return [(start, start + nper) for start in range(0, n_samples - nper + 1, step)]


def _as_column(result: SegmentResult, imf_number: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    idx = int(imf_number) - 1
    if result.imf is None or result.imf.ndim != 2 or result.imf.shape[1] <= idx:
        return None, None
    imf = np.asarray(result.imf[:, idx], dtype=float)
    ifreq = None
    if result.ifreq is not None and result.ifreq.ndim == 2 and result.ifreq.shape[1] > idx:
        ifreq = np.asarray(result.ifreq[:, idx], dtype=float)
    return imf, ifreq


def _spectral_pair(x: np.ndarray, y: np.ndarray, fs: float, nperseg_sec: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nperseg = min(len(x), max(8, int(round(nperseg_sec * fs))))
    freqs, pxx = signal.welch(x, fs=fs, nperseg=nperseg)
    _, pyy = signal.welch(y, fs=fs, nperseg=nperseg)
    _, pxy = signal.csd(x, y, fs=fs, nperseg=nperseg)
    coherency = pxy / (np.sqrt(pxx * pyy) + np.finfo(float).eps)
    return freqs, np.abs(coherency) ** 2, np.abs(np.imag(coherency))


def _epochwise_imf_connectivity(
    seed: np.ndarray,
    target: np.ndarray,
    fs: float,
    cfg: dict,
) -> dict[str, float]:
    conn = cfg["connectivity"]
    epochs = _epoch_starts(seed.size, fs, float(conn["epoch_length_sec"]), float(conn["epoch_overlap"]))
    if len(epochs) < int(conn.get("min_epochs", 4)):
        return {"n_epochs": len(epochs), "coh": np.nan, "imcoh": np.nan}
    nperseg_sec = float(conn.get("coherence", {}).get("nperseg_sec", 1.0))
    coh = []
    imcoh = []
    for start, stop in epochs:
        _, c, ic = _spectral_pair(seed[start:stop], target[start:stop], fs, nperseg_sec)
        coh.append(float(np.nanmean(c)))
        imcoh.append(float(np.nanmean(ic)))
    return {"n_epochs": len(epochs), "coh": float(np.nanmean(coh)), "imcoh": float(np.nanmean(imcoh))}


def _decompose_window(rec: SeegRecording, mask: np.ndarray, cfg: dict) -> list[ChannelImfs]:
    rows = []
    for channel_index, channel in enumerate(rec.channel_names):
        x, drift, noise = preprocess_segment(rec.signal[mask, channel_index], rec.time[mask], rec.fs, cfg["preprocessing"])
        result = analyze_segment(x, rec.fs, cfg["emd"])
        rows.append(
            ChannelImfs(
                channel=channel,
                channel_index=channel_index,
                result=result,
                drift_flag=drift.flagged,
                noise_flag=noise.flagged,
            )
        )
    return rows


def _seed_target_indices(rec: SeegRecording) -> tuple[list[int], list[int]]:
    seeds = [i for i, ch in enumerate(rec.channel_names) if channel_is_stimulated(ch, rec.stimulated_channel_names)]
    targets = [i for i in range(len(rec.channel_names)) if i not in seeds]
    return seeds, targets


def _imf_frequency_rows(rec: SeegRecording, epoch: str, decomposed: list[ChannelImfs], cfg: dict) -> list[dict]:
    rows = []
    for ch in decomposed:
        for imf_number in cfg["connectivity"]["imfs"]:
            imf, ifreq = _as_column(ch.result, int(imf_number))
            if imf is None:
                rows.append(
                    {
                        "file": rec.file_stem,
                        "subject": rec.subject,
                        "epoch": epoch,
                        "channel": ch.channel,
                        "channel_index": ch.channel_index,
                        "imf": int(imf_number),
                        "status": "missing_imf",
                        "noise_flag": ch.noise_flag,
                        "drift_flag": ch.drift_flag,
                    }
                )
                continue
            finite_if = ifreq[np.isfinite(ifreq) & (ifreq > 0)] if ifreq is not None else np.array([])
            rows.append(
                {
                    "file": rec.file_stem,
                    "subject": rec.subject,
                    "epoch": epoch,
                    "channel": ch.channel,
                    "channel_index": ch.channel_index,
                    "imf": int(imf_number),
                    "status": ch.result.status,
                    "noise_flag": ch.noise_flag,
                    "drift_flag": ch.drift_flag,
                    "mean_if_hz": float(np.nanmean(finite_if)) if finite_if.size else np.nan,
                    "median_if_hz": float(np.nanmedian(finite_if)) if finite_if.size else np.nan,
                    "if_iqr_hz": float(np.subtract(*np.nanpercentile(finite_if, [75, 25]))) if finite_if.size else np.nan,
                    "rms_amp": float(np.sqrt(np.nanmean(imf**2))),
                }
            )
    return rows


def analyze_imf_connectivity_file(path: str | Path, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rec = load_recording(path)
    min_channels = int(cfg.get("preprocessing", {}).get("min_channels_per_file", 1))
    if rec.signal.shape[1] < min_channels:
        row = {
            "file": rec.file_stem,
            "status": "excluded_few_channels",
            "n_channels": rec.signal.shape[1],
        }
        return pd.DataFrame([row]), pd.DataFrame()

    seeds, targets = _seed_target_indices(rec)
    if not seeds or not targets:
        row = {"file": rec.file_stem, "status": "missing_seed_or_target", "n_channels": rec.signal.shape[1]}
        return pd.DataFrame([row]), pd.DataFrame()

    pre_start, pre_stop = cfg["windows"]["pre"]
    post_start, post_stop = post_window_for_frequency(
        rec.stim_frequency_hz,
        cfg["windows"]["stim_end_by_frequency"],
        cfg["windows"]["post_duration_sec"],
        cfg["windows"]["post_artifact_buffer_sec"],
    )
    pre_mask = window_mask(rec.time, pre_start, pre_stop)
    post_mask = window_mask(rec.time, post_start, post_stop)
    pre_dec = _decompose_window(rec, pre_mask, cfg)
    post_dec = _decompose_window(rec, post_mask, cfg)

    freq_rows = _imf_frequency_rows(rec, "pre", pre_dec, cfg)
    freq_rows.extend(_imf_frequency_rows(rec, "post", post_dec, cfg))

    pair_rows = []
    for seed_idx in seeds:
        for target_idx in targets:
            for imf_number in cfg["connectivity"]["imfs"]:
                pre_seed, _ = _as_column(pre_dec[seed_idx].result, int(imf_number))
                pre_target, _ = _as_column(pre_dec[target_idx].result, int(imf_number))
                post_seed, _ = _as_column(post_dec[seed_idx].result, int(imf_number))
                post_target, _ = _as_column(post_dec[target_idx].result, int(imf_number))
                row = {
                    "file": rec.file_stem,
                    "path": str(rec.path),
                    "subject": rec.subject,
                    "stim_frequency_hz": rec.stim_frequency_hz,
                    "stim_amplitude_ma": rec.stim_amplitude_ma,
                    "seed_channel": rec.channel_names[seed_idx],
                    "target_channel": rec.channel_names[target_idx],
                    "seed_index": seed_idx,
                    "target_index": target_idx,
                    "imf": int(imf_number),
                    "pre_noise_pair": bool(pre_dec[seed_idx].noise_flag or pre_dec[target_idx].noise_flag),
                    "post_noise_pair": bool(post_dec[seed_idx].noise_flag or post_dec[target_idx].noise_flag),
                }
                if any(v is None for v in [pre_seed, pre_target, post_seed, post_target]):
                    row["status"] = "missing_imf"
                    pair_rows.append(row)
                    continue
                pre_conn = _epochwise_imf_connectivity(pre_seed, pre_target, rec.fs, cfg)
                post_conn = _epochwise_imf_connectivity(post_seed, post_target, rec.fs, cfg)
                row["status"] = "ok"
                for key, value in pre_conn.items():
                    row[f"pre_{key}"] = value
                for key, value in post_conn.items():
                    row[f"post_{key}"] = value
                for method in ["coh", "imcoh"]:
                    row[f"delta_{method}"] = row.get(f"post_{method}", np.nan) - row.get(f"pre_{method}", np.nan)
                pair_rows.append(row)
    return pd.DataFrame(pair_rows), pd.DataFrame(freq_rows)


def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan)
    valid = np.isfinite(pvals)
    if not valid.any():
        return out
    pv = pvals[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * len(ranked) / (np.arange(len(ranked)) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    tmp = np.empty_like(adjusted)
    tmp[order] = np.clip(adjusted, 0, 1)
    out[valid] = tmp
    return out


def _paired_summary(df: pd.DataFrame, method: str, level: str, imf: int) -> dict:
    pre_col = f"pre_{method}"
    post_col = f"post_{method}"
    mask = np.isfinite(df[pre_col]) & np.isfinite(df[post_col])
    pre = df.loc[mask, pre_col].to_numpy(float)
    post = df.loc[mask, post_col].to_numpy(float)
    delta = post - pre
    out = {"method": method, "level": level, "imf": imf, "n": int(delta.size)}
    if delta.size < 2:
        return out
    ttest = stats.ttest_rel(post, pre)
    try:
        wil = stats.wilcoxon(delta)
        wil_p = float(wil.pvalue)
    except ValueError:
        wil_p = np.nan
    out.update(
        {
            "pre_mean": float(np.nanmean(pre)),
            "post_mean": float(np.nanmean(post)),
            "mean_delta": float(np.nanmean(delta)),
            "median_delta": float(np.nanmedian(delta)),
            "paired_t_p": float(ttest.pvalue),
            "wilcoxon_p": wil_p,
        }
    )
    return out


def summarize_imf_connectivity(pairs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ok = pairs[pairs["status"].eq("ok")].copy()
    ok = ok[~ok["pre_noise_pair"].fillna(False) & ~ok["post_noise_pair"].fillna(False)]
    rows = []
    for imf in cfg["connectivity"]["imfs"]:
        sub = ok[ok["imf"].eq(int(imf))]
        for method in ["coh", "imcoh"]:
            row = _paired_summary(sub, method, "pair", int(imf))
            row["group"] = "all"
            rows.append(row)
            file_means = sub.groupby(["file", "stim_frequency_hz"], as_index=False)[
                [f"pre_{method}", f"post_{method}"]
            ].mean()
            row = _paired_summary(file_means, method, "file_mean", int(imf))
            row["group"] = "all"
            rows.append(row)
            for freq in sorted(sub["stim_frequency_hz"].dropna().unique()):
                freq_sub = sub[sub["stim_frequency_hz"].eq(freq)]
                row = _paired_summary(freq_sub, method, "pair", int(imf))
                row["group"] = f"{freq:g}Hz"
                rows.append(row)
                freq_file_means = file_means[file_means["stim_frequency_hz"].eq(freq)]
                row = _paired_summary(freq_file_means, method, "file_mean", int(imf))
                row["group"] = f"{freq:g}Hz"
                rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and "wilcoxon_p" in out:
        out["wilcoxon_p_fdr"] = _fdr_bh(out["wilcoxon_p"].to_numpy())
    return out


def summarize_existing_imf_connectivity(config_path: str | Path) -> pd.DataFrame:
    cfg = load_config(config_path)
    out_root = Path(cfg["project"]["output_dir"])
    pairs = pd.read_csv(out_root / "imf_seed_to_nonstim_target_connectivity.csv")
    stats_df = summarize_imf_connectivity(pairs, cfg)
    stats_df.to_csv(out_root / "imf_seed_to_nonstim_target_stats.csv", index=False)
    return stats_df


def analyze_imf_connectivity_dataset(
    config_path: str | Path,
    max_files: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = load_config(config_path)
    files = list(iter_mat_files(cfg["project"]["data_dir"]))
    if max_files is not None:
        files = files[:max_files]
    pair_frames = []
    freq_frames = []
    for path in files:
        pairs, freqs = analyze_imf_connectivity_file(path, cfg)
        pair_frames.append(pairs)
        if not freqs.empty:
            freq_frames.append(freqs)
    pair_df = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    freq_df = pd.concat(freq_frames, ignore_index=True) if freq_frames else pd.DataFrame()
    stats_df = summarize_imf_connectivity(pair_df, cfg)

    out_root = Path(cfg["project"]["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(out_root / "imf_seed_to_nonstim_target_connectivity.csv", index=False)
    freq_df.to_csv(out_root / "imf_frequency_qc.csv", index=False)
    stats_df.to_csv(out_root / "imf_seed_to_nonstim_target_stats.csv", index=False)
    return pair_df, freq_df, stats_df


def plot_imf_connectivity_outputs(config_path: str | Path) -> list[Path]:
    cfg = load_config(config_path)
    out_root = Path(cfg["project"]["output_dir"])
    stats_df = pd.read_csv(out_root / "imf_seed_to_nonstim_target_stats.csv")
    freq_df = pd.read_csv(out_root / "imf_frequency_qc.csv")
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    set_publication_style()
    paths: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for ax, epoch in zip(axes, ["pre", "post"]):
        sub = freq_df[freq_df["epoch"].eq(epoch) & np.isfinite(freq_df["median_if_hz"])]
        data = [sub[sub["imf"].eq(imf)]["median_if_hz"].to_numpy() for imf in cfg["connectivity"]["imfs"]]
        parts = ax.violinplot(data, showmedians=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor("#4C78A8" if epoch == "pre" else "#E45756")
            body.set_alpha(0.65)
        ax.set_title(f"{epoch.capitalize()} IMF frequency QC")
        ax.set_xlabel("IMF")
        ax.set_ylabel("Median IF (Hz)")
        ax.set_xticks(range(1, len(data) + 1))
        ax.set_xticklabels([str(i) for i in cfg["connectivity"]["imfs"]])
        ax.set_ylim(0, 100)
    path = fig_dir / "imf_frequency_qc_violins.png"
    fig.savefig(path, dpi=cfg["outputs"].get("figure_dpi", 300))
    plt.close(fig)
    paths.append(path)

    for level in ["pair", "file_mean"]:
        sub = stats_df[stats_df["level"].eq(level)]
        if "group" in sub.columns:
            sub = sub[sub["group"].eq("all")]
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
        for ax, method, color in [(axes[0], "coh", "#4C78A8"), (axes[1], "imcoh", "#F58518")]:
            local = sub[sub["method"].eq(method)].sort_values("imf")
            x = local["imf"].to_numpy()
            y = local["mean_delta"].to_numpy()
            bars = ax.bar(x, y, color=color)
            ax.axhline(0, color="0.25", linewidth=0.8)
            ax.set_title(f"{method.upper()} ({level})")
            ax.set_xlabel("IMF")
            ax.set_ylabel("Post - pre")
            for bar, p in zip(bars, local["wilcoxon_p_fdr"].to_numpy()):
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                if star:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        star,
                        ha="center",
                        va="bottom" if bar.get_height() >= 0 else "top",
                    )
        path = fig_dir / f"imf_connectivity_delta_{level}.png"
        fig.savefig(path, dpi=cfg["outputs"].get("figure_dpi", 300))
        plt.close(fig)
        paths.append(path)

    if "group" in stats_df.columns:
        for level in ["pair", "file_mean"]:
            sub = stats_df[stats_df["level"].eq(level) & stats_df["group"].isin(["1Hz", "50Hz"])]
            fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), constrained_layout=True)
            x = np.arange(len(cfg["connectivity"]["imfs"]))
            width = 0.38
            for ax, method, color_pair in [
                (axes[0], "coh", ("#54A24B", "#B279A2")),
                (axes[1], "imcoh", ("#54A24B", "#B279A2")),
            ]:
                for offset, group, color in [(-width / 2, "1Hz", color_pair[0]), (width / 2, "50Hz", color_pair[1])]:
                    local = sub[sub["method"].eq(method) & sub["group"].eq(group)].sort_values("imf")
                    vals = []
                    stars = []
                    for imf in cfg["connectivity"]["imfs"]:
                        row = local[local["imf"].eq(int(imf))]
                        vals.append(float(row["mean_delta"].iloc[0]) if not row.empty else 0.0)
                        p = float(row["wilcoxon_p_fdr"].iloc[0]) if not row.empty else 1.0
                        stars.append("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")
                    bars = ax.bar(x + offset, vals, width=width, color=color, label=group)
                    for bar, star in zip(bars, stars):
                        if star:
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,
                                bar.get_height(),
                                star,
                                ha="center",
                                va="bottom" if bar.get_height() >= 0 else "top",
                                fontsize=8,
                            )
                ax.axhline(0, color="0.25", linewidth=0.8)
                ax.set_title(f"{method.upper()} ({level})")
                ax.set_xlabel("IMF")
                ax.set_ylabel("Post - pre")
                ax.set_xticks(x)
                ax.set_xticklabels([str(i) for i in cfg["connectivity"]["imfs"]])
                ax.legend(frameon=False)
            path = fig_dir / f"imf_connectivity_delta_by_frequency_{level}.png"
            fig.savefig(path, dpi=cfg["outputs"].get("figure_dpi", 300))
            plt.close(fig)
            paths.append(path)
    return paths
