from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from .io import SeegRecording, iter_mat_files, load_recording
from .preprocess import post_window_for_frequency, preprocess_segment, window_mask
from .quinn import SegmentResult, analyze_segment


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_. -]+", "_", str(text)).strip()


def channel_is_stimulated(channel_name: str, stimulated: list[str]) -> bool:
    norm = re.sub(r"\s+", "", channel_name).upper()
    return any(re.sub(r"\s+", "", s).upper() == norm for s in stimulated)


def summarize_segment(prefix: str, result: SegmentResult) -> dict:
    out = {f"{prefix}_status": result.status, f"{prefix}_n_cycles": result.n_cycles}
    table = result.cycle_table
    if table is not None and len(table) > 0:
        for col in ("mean_if", "max_amp", "if_range", "asc2desc", "peak2trough"):
            if col in table:
                out[f"{prefix}_{col}"] = float(np.nanmean(table[col]))
    return out


def waveform_change(pre: SegmentResult, post: SegmentResult) -> dict:
    out = {}
    if pre.phase_aligned_if is not None and post.phase_aligned_if is not None:
        pre_mean = np.nanmean(pre.phase_aligned_if, axis=1)
        post_mean = np.nanmean(post.phase_aligned_if, axis=1)
        n = min(pre_mean.size, post_mean.size)
        out["pa_if_change_score"] = float(np.nanmean(np.abs(post_mean[:n] - pre_mean[:n])))
    if pre.normalized_waveform is not None and post.normalized_waveform is not None:
        pre_mean = np.nanmean(pre.normalized_waveform, axis=1)
        post_mean = np.nanmean(post.normalized_waveform, axis=1)
        n = min(pre_mean.size, post_mean.size)
        out["norm_waveform_change_score"] = float(np.nanmean(np.abs(post_mean[:n] - pre_mean[:n])))
    return out


def summarize_noise(prefix: str, report) -> dict:
    return {
        f"{prefix}_noise_flag": report.flagged,
        f"{prefix}_nan_fraction": report.nan_fraction,
        f"{prefix}_robust_std": report.robust_std,
        f"{prefix}_peak_to_peak": report.peak_to_peak,
        f"{prefix}_peak_to_peak_iqr_ratio": report.peak_to_peak_iqr_ratio,
        f"{prefix}_line_noise_ratio": report.line_noise_ratio,
        f"{prefix}_flat_flag": report.flat_flag,
        f"{prefix}_extreme_amplitude_flag": report.extreme_amplitude_flag,
        f"{prefix}_nonfinite_flag": report.nonfinite_flag,
    }


def save_channel_outputs(out_dir: Path, channel: str, pre: SegmentResult, post: SegmentResult, cfg: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = safe_name(channel) + f"_IMF{int(cfg['emd'].get('imf_index', 3)) + 1}"
    if cfg["outputs"].get("save_cycle_csv", True):
        if pre.cycle_table is not None:
            pre.cycle_table.to_csv(out_dir / f"{prefix}_pre_cycles.csv", index=False)
        if post.cycle_table is not None:
            post.cycle_table.to_csv(out_dir / f"{prefix}_post_cycles.csv", index=False)
    if cfg["outputs"].get("save_channel_npz", True):
        np.savez_compressed(
            out_dir / f"{prefix}_outputs.npz",
            pre_pa_if=pre.phase_aligned_if,
            post_pa_if=post.phase_aligned_if,
            pre_norm_waveform=pre.normalized_waveform,
            post_norm_waveform=post.normalized_waveform,
        )


def analyze_channel(rec: SeegRecording, channel_index: int, cfg: dict, file_out_dir: Path | None = None) -> dict:
    windows = cfg["windows"]
    pre_start, pre_stop = windows["pre"]
    post_start, post_stop = post_window_for_frequency(
        rec.stim_frequency_hz,
        windows["stim_end_by_frequency"],
        windows["post_duration_sec"],
        windows["post_artifact_buffer_sec"],
    )

    channel_name = rec.channel_names[channel_index]
    x = rec.signal[:, channel_index]
    pre_idx = window_mask(rec.time, pre_start, pre_stop)
    post_idx = window_mask(rec.time, post_start, post_stop)

    if pre_idx.sum() < rec.fs * 2 or post_idx.sum() < rec.fs * 2:
        return {
            "file": rec.file_stem,
            "channel": channel_name,
            "status": "window_too_short",
            "pre_samples": int(pre_idx.sum()),
            "post_samples": int(post_idx.sum()),
        }

    pre_x, pre_drift, pre_noise = preprocess_segment(x[pre_idx], rec.time[pre_idx], rec.fs, cfg["preprocessing"])
    post_x, post_drift, post_noise = preprocess_segment(x[post_idx], rec.time[post_idx], rec.fs, cfg["preprocessing"])
    pre = analyze_segment(pre_x, rec.fs, cfg["emd"])
    post = analyze_segment(post_x, rec.fs, cfg["emd"])

    noise_failed = pre_noise.flagged or post_noise.flagged
    if pre.status == "ok" and post.status == "ok" and not noise_failed:
        status = "ok"
    elif pre.status == "ok" and post.status == "ok" and noise_failed:
        status = "ok_with_noise_flag"
    else:
        status = "partial_or_failed"
    row = {
        "file": rec.file_stem,
        "path": str(rec.path),
        "subject": rec.subject,
        "entry_index": rec.entry_index,
        "stim_frequency_hz": rec.stim_frequency_hz,
        "stim_amplitude_ma": rec.stim_amplitude_ma,
        "stimulated_channels": ";".join(rec.stimulated_channel_names),
        "channel": channel_name,
        "channel_index": channel_index,
        "is_stimulated_channel": channel_is_stimulated(channel_name, rec.stimulated_channel_names),
        "fs": rec.fs,
        "pre_window_start": pre_start,
        "pre_window_stop": pre_stop,
        "post_window_start": post_start,
        "post_window_stop": post_stop,
        "pre_samples": int(pre_idx.sum()),
        "post_samples": int(post_idx.sum()),
        "pre_drift_flag": pre_drift.flagged,
        "post_drift_flag": post_drift.flagged,
        "pre_drift_ratio": pre_drift.slope_iqr_ratio,
        "post_drift_ratio": post_drift.slope_iqr_ratio,
        "status": status,
    }
    row.update(summarize_noise("pre", pre_noise))
    row.update(summarize_noise("post", post_noise))
    row.update(summarize_segment("pre", pre))
    row.update(summarize_segment("post", post))
    row.update(waveform_change(pre, post))
    for metric in ("mean_if", "asc2desc", "peak2trough"):
        a = row.get(f"pre_{metric}")
        b = row.get(f"post_{metric}")
        if a is not None and b is not None:
            row[f"delta_{metric}"] = b - a

    if file_out_dir is not None and status == "ok":
        save_channel_outputs(file_out_dir, channel_name, pre, post, cfg)
    return row


def channel_diagnostics(rec: SeegRecording, channel_name: str, cfg: dict) -> dict:
    """Return detailed pre/post arrays for Quinn-style diagnostic plotting."""
    if channel_name not in rec.channel_names:
        raise ValueError(f"{channel_name!r} not found in {rec.file_stem}")
    channel_index = rec.channel_names.index(channel_name)

    windows = cfg["windows"]
    pre_start, pre_stop = windows["pre"]
    post_start, post_stop = post_window_for_frequency(
        rec.stim_frequency_hz,
        windows["stim_end_by_frequency"],
        windows["post_duration_sec"],
        windows["post_artifact_buffer_sec"],
    )
    x = rec.signal[:, channel_index]
    pre_idx = window_mask(rec.time, pre_start, pre_stop)
    post_idx = window_mask(rec.time, post_start, post_stop)

    pre_x, pre_drift, pre_noise = preprocess_segment(x[pre_idx], rec.time[pre_idx], rec.fs, cfg["preprocessing"])
    post_x, post_drift, post_noise = preprocess_segment(x[post_idx], rec.time[post_idx], rec.fs, cfg["preprocessing"])
    pre = analyze_segment(pre_x, rec.fs, cfg["emd"])
    post = analyze_segment(post_x, rec.fs, cfg["emd"])

    imf_index = int(cfg["emd"].get("imf_index", 3))
    return {
        "recording": rec,
        "channel_name": channel_name,
        "channel_index": channel_index,
        "imf_index": imf_index,
        "pre": {
            "time": rec.time[pre_idx],
            "raw": x[pre_idx],
            "processed": pre_x,
            "drift": pre_drift,
            "noise": pre_noise,
            "result": pre,
            "window": (pre_start, pre_stop),
        },
        "post": {
            "time": rec.time[post_idx],
            "raw": x[post_idx],
            "processed": post_x,
            "drift": post_drift,
            "noise": post_noise,
            "result": post,
            "window": (post_start, post_stop),
        },
    }


def analyze_file(path: str | Path, cfg: dict, max_channels: int | None = None) -> pd.DataFrame:
    rec = load_recording(path)
    out_root = Path(cfg["project"]["output_dir"])
    file_out = out_root / rec.file_stem
    min_channels = int(cfg.get("preprocessing", {}).get("min_channels_per_file", 1))
    if rec.signal.shape[1] < min_channels:
        df = pd.DataFrame(
            [
                {
                    "file": rec.file_stem,
                    "path": str(rec.path),
                    "subject": rec.subject,
                    "entry_index": rec.entry_index,
                    "stim_frequency_hz": rec.stim_frequency_hz,
                    "stim_amplitude_ma": rec.stim_amplitude_ma,
                    "n_channels": rec.signal.shape[1],
                    "status": "excluded_few_channels",
                    "message": f"Excluded because n_channels < {min_channels}",
                }
            ]
        )
        file_out.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_out / f"{rec.file_stem}_all_channels_summary.csv", index=False)
        return df
    rows = []
    n_channels = rec.signal.shape[1] if max_channels is None else min(max_channels, rec.signal.shape[1])
    for idx in range(n_channels):
        try:
            rows.append(analyze_channel(rec, idx, cfg, file_out))
        except Exception as exc:
            rows.append({"file": rec.file_stem, "channel_index": idx, "status": "exception", "message": str(exc)})
    df = pd.DataFrame(rows)
    file_out.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_out / f"{rec.file_stem}_all_channels_summary.csv", index=False)
    return df


def analyze_dataset(cfg: dict, max_files: int | None = None, max_channels: int | None = None) -> pd.DataFrame:
    files = list(iter_mat_files(cfg["project"]["data_dir"]))
    if max_files is not None:
        files = files[:max_files]
    frames = [analyze_file(path, cfg, max_channels=max_channels) for path in files]
    master = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_root = Path(cfg["project"]["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_root / "MASTER_all_channels_summary.csv", index=False)
    return master
