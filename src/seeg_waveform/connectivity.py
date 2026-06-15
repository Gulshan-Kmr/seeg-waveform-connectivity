from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats

from .config import load_config
from .io import SeegRecording, iter_mat_files, load_recording
from .pipeline import channel_is_stimulated
from .preprocess import post_window_for_frequency, preprocess_segment, window_mask


@dataclass
class PairSpectra:
    freqs: np.ndarray
    coherence: np.ndarray
    imcoh: np.ndarray


def _epoch_starts(n_samples: int, fs: float, epoch_length_sec: float, overlap: float) -> list[tuple[int, int]]:
    nper = int(round(epoch_length_sec * fs))
    if nper <= 0 or n_samples < nper:
        return []
    step = max(1, int(round(nper * (1.0 - overlap))))
    return [(start, start + nper) for start in range(0, n_samples - nper + 1, step)]


def _band_mean(freqs: np.ndarray, values: np.ndarray, band: tuple[float, float]) -> float:
    lo, hi = band
    mask = (freqs >= lo) & (freqs < hi) & np.isfinite(values)
    if not np.any(mask):
        return np.nan
    return float(np.nanmean(values[mask]))


def _pair_spectra(x: np.ndarray, y: np.ndarray, fs: float, nperseg_sec: float) -> PairSpectra:
    nperseg = min(len(x), max(8, int(round(nperseg_sec * fs))))
    freqs, pxx = signal.welch(x, fs=fs, nperseg=nperseg)
    _, pyy = signal.welch(y, fs=fs, nperseg=nperseg)
    _, pxy = signal.csd(x, y, fs=fs, nperseg=nperseg)
    denom = np.sqrt(pxx * pyy) + np.finfo(float).eps
    coherency = pxy / denom
    coherence = np.abs(coherency) ** 2
    imcoh = np.abs(np.imag(coherency))
    return PairSpectra(freqs=freqs, coherence=coherence, imcoh=imcoh)


def _preprocess_matrix(rec: SeegRecording, idx: np.ndarray, cfg: dict) -> tuple[np.ndarray, list[dict]]:
    signals = []
    qc_rows = []
    for channel_index, channel in enumerate(rec.channel_names):
        x, drift, noise = preprocess_segment(rec.signal[idx, channel_index], rec.time[idx], rec.fs, cfg["preprocessing"])
        signals.append(x)
        qc_rows.append(
            {
                "channel": channel,
                "channel_index": channel_index,
                "drift_flag": drift.flagged,
                "drift_ratio": drift.slope_iqr_ratio,
                "noise_flag": noise.flagged,
                "line_noise_ratio": noise.line_noise_ratio,
                "peak_to_peak_iqr_ratio": noise.peak_to_peak_iqr_ratio,
            }
        )
    return np.column_stack(signals), qc_rows


def _seed_target_indices(rec: SeegRecording) -> tuple[list[int], list[int]]:
    seed_indices = [
        i for i, ch in enumerate(rec.channel_names) if channel_is_stimulated(ch, rec.stimulated_channel_names)
    ]
    target_indices = [i for i in range(len(rec.channel_names)) if i not in seed_indices]
    return seed_indices, target_indices


def _summarize_pair_epochwise(
    data: np.ndarray,
    fs: float,
    seed_index: int,
    target_index: int,
    conn_cfg: dict,
) -> dict[str, float]:
    epochs = _epoch_starts(data.shape[0], fs, conn_cfg["epoch_length_sec"], conn_cfg["epoch_overlap"])
    if len(epochs) < int(conn_cfg.get("min_epochs", 4)):
        return {"n_epochs": len(epochs)}
    bands = {name: tuple(vals) for name, vals in conn_cfg["frequency_bands"].items()}
    nperseg_sec = float(conn_cfg.get("coherence", {}).get("nperseg_sec", 1.0))
    rows = []
    for start, stop in epochs:
        spec = _pair_spectra(data[start:stop, seed_index], data[start:stop, target_index], fs, nperseg_sec)
        row = {}
        for band_name, band in bands.items():
            row[f"coh_{band_name}"] = _band_mean(spec.freqs, spec.coherence, band)
            row[f"imcoh_{band_name}"] = _band_mean(spec.freqs, spec.imcoh, band)
        rows.append(row)
    epoch_df = pd.DataFrame(rows)
    out = {"n_epochs": len(epochs)}
    for col in epoch_df.columns:
        out[col] = float(epoch_df[col].mean())
    return out


def _adf_stationary(x: np.ndarray, alpha: float) -> bool:
    try:
        from statsmodels.tsa.stattools import adfuller

        return bool(adfuller(x, autolag="AIC")[1] < alpha)
    except Exception:
        return False


def _pdc_pair(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]],
    pdc_cfg: dict,
) -> dict[str, float | bool | int | str]:
    """Fit a bivariate VAR and estimate seed-to-target PDC.

    Variables are ordered [seed, target]. PDC seed->target is A[target, seed].
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.api import VAR

    alpha = float(pdc_cfg.get("stationarity_alpha", 0.05))
    if not (_adf_stationary(x, alpha) and _adf_stationary(y, alpha)):
        return {"pdc_valid": False, "pdc_reason": "nonstationary_adf"}

    arr = np.column_stack([x, y])
    arr = signal.detrend(arr, axis=0, type="linear")
    arr = arr - arr.mean(axis=0, keepdims=True)
    max_order = int(pdc_cfg.get("max_order", 20))
    max_order = max(1, min(max_order, arr.shape[0] // int(pdc_cfg.get("min_samples_per_parameter", 10))))
    if max_order < 1:
        return {"pdc_valid": False, "pdc_reason": "too_few_samples"}

    try:
        order_res = VAR(arr).select_order(maxlags=max_order)
        order = int(order_res.aic or 1)
        order = max(1, min(order, max_order))
        fit = VAR(arr).fit(order)
    except Exception as exc:
        return {"pdc_valid": False, "pdc_reason": f"var_failed:{exc}"[:120]}

    if not fit.is_stable(verbose=False):
        return {"pdc_valid": False, "pdc_reason": "unstable_var", "pdc_order": order}

    whiteness_alpha = float(pdc_cfg.get("residual_whiteness_alpha", 0.01))
    try:
        pvals = []
        for col in range(fit.resid.shape[1]):
            lb = acorr_ljungbox(fit.resid[:, col], lags=[min(10, max(1, fit.resid.shape[0] // 5))], return_df=True)
            pvals.append(float(lb["lb_pvalue"].iloc[0]))
        residual_white = min(pvals) > whiteness_alpha
    except Exception:
        residual_white = False

    freqs = np.linspace(0, min(100.0, fs / 2.0), 256)
    coefs = fit.coefs
    values = []
    for f in freqs:
        amat = np.eye(2, dtype=complex)
        for lag in range(order):
            amat -= coefs[lag] * np.exp(-2j * np.pi * f * (lag + 1) / fs)
        denom = np.sqrt(np.sum(np.abs(amat[:, 0]) ** 2)) + np.finfo(float).eps
        values.append(np.abs(amat[1, 0]) / denom)
    values = np.asarray(values)

    out: dict[str, float | bool | int | str] = {
        "pdc_valid": bool(residual_white),
        "pdc_reason": "ok" if residual_white else "residual_autocorrelation",
        "pdc_order": order,
    }
    for band_name, band in bands.items():
        out[f"pdc_{band_name}"] = _band_mean(freqs, values, band)
    return out


def analyze_connectivity_file(path: str | Path, cfg: dict, include_pdc: bool = False) -> pd.DataFrame:
    rec = load_recording(path)
    min_channels = int(cfg.get("preprocessing", {}).get("min_channels_per_file", 1))
    if rec.signal.shape[1] < min_channels:
        return pd.DataFrame(
            [
                {
                    "file": rec.file_stem,
                    "status": "excluded_few_channels",
                    "n_channels": rec.signal.shape[1],
                    "message": f"Excluded because n_channels < {min_channels}",
                }
            ]
        )

    seed_indices, target_indices = _seed_target_indices(rec)
    if not seed_indices or not target_indices:
        return pd.DataFrame(
            [{"file": rec.file_stem, "status": "missing_seed_or_target", "n_channels": rec.signal.shape[1]}]
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
    pre_data, pre_qc = _preprocess_matrix(rec, pre_idx, cfg)
    post_data, post_qc = _preprocess_matrix(rec, post_idx, cfg)
    pre_noise = {row["channel_index"]: row["noise_flag"] for row in pre_qc}
    post_noise = {row["channel_index"]: row["noise_flag"] for row in post_qc}

    conn_cfg = cfg["connectivity"]
    bands = {name: tuple(vals) for name, vals in conn_cfg["frequency_bands"].items()}
    max_pdc = conn_cfg.get("pdc", {}).get("max_pairs_per_file")
    max_pdc = None if max_pdc is None else int(max_pdc)

    rows = []
    pdc_count = 0
    for seed_index in seed_indices:
        for target_index in target_indices:
            seed = rec.channel_names[seed_index]
            target = rec.channel_names[target_index]
            row = {
                "file": rec.file_stem,
                "path": str(rec.path),
                "subject": rec.subject,
                "stim_frequency_hz": rec.stim_frequency_hz,
                "stim_amplitude_ma": rec.stim_amplitude_ma,
                "seed_channel": seed,
                "target_channel": target,
                "seed_index": seed_index,
                "target_index": target_index,
                "pre_noise_pair": bool(pre_noise.get(seed_index, False) or pre_noise.get(target_index, False)),
                "post_noise_pair": bool(post_noise.get(seed_index, False) or post_noise.get(target_index, False)),
                "status": "ok",
            }
            pre = _summarize_pair_epochwise(pre_data, rec.fs, seed_index, target_index, conn_cfg)
            post = _summarize_pair_epochwise(post_data, rec.fs, seed_index, target_index, conn_cfg)
            for key, value in pre.items():
                row[f"pre_{key}"] = value
            for key, value in post.items():
                row[f"post_{key}"] = value
            for band_name in bands:
                for method in ["coh", "imcoh"]:
                    a = row.get(f"pre_{method}_{band_name}")
                    b = row.get(f"post_{method}_{band_name}")
                    if a is not None and b is not None:
                        row[f"delta_{method}_{band_name}"] = b - a

            if include_pdc and (max_pdc is None or pdc_count < max_pdc):
                pre_pdc = _pdc_pair(pre_data[:, seed_index], pre_data[:, target_index], rec.fs, bands, conn_cfg["pdc"])
                post_pdc = _pdc_pair(post_data[:, seed_index], post_data[:, target_index], rec.fs, bands, conn_cfg["pdc"])
                for key, value in pre_pdc.items():
                    row[f"pre_{key}"] = value
                for key, value in post_pdc.items():
                    row[f"post_{key}"] = value
                for band_name in bands:
                    a = row.get(f"pre_pdc_{band_name}")
                    b = row.get(f"post_pdc_{band_name}")
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        row[f"delta_pdc_{band_name}"] = b - a
                pdc_count += 1
            rows.append(row)
    return pd.DataFrame(rows)


def _paired_summary(df: pd.DataFrame, value_col: str, level: str) -> dict:
    pre_col = "pre_" + value_col
    post_col = "post_" + value_col
    mask = np.isfinite(df[pre_col]) & np.isfinite(df[post_col])
    pre = df.loc[mask, pre_col].to_numpy(float)
    post = df.loc[mask, post_col].to_numpy(float)
    delta = post - pre
    if delta.size < 2:
        return {"metric": value_col, "level": level, "n": int(delta.size)}
    ttest = stats.ttest_rel(post, pre)
    try:
        wil = stats.wilcoxon(delta)
        wil_p = float(wil.pvalue)
    except ValueError:
        wil_p = np.nan
    return {
        "metric": value_col,
        "level": level,
        "n": int(delta.size),
        "pre_mean": float(np.nanmean(pre)),
        "post_mean": float(np.nanmean(post)),
        "mean_delta": float(np.nanmean(delta)),
        "median_delta": float(np.nanmedian(delta)),
        "paired_t": float(ttest.statistic),
        "paired_t_p": float(ttest.pvalue),
        "wilcoxon_p": wil_p,
    }


def summarize_connectivity(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ok = df[df["status"].eq("ok")].copy()
    ok = ok[~ok["pre_noise_pair"].fillna(False) & ~ok["post_noise_pair"].fillna(False)]
    rows = []
    for method in ["coh", "imcoh", "pdc"]:
        for band in cfg["connectivity"]["frequency_bands"]:
            col = f"{method}_{band}"
            if f"pre_{col}" in ok and f"post_{col}" in ok:
                row = _paired_summary(ok, col, "pair")
                row["group"] = "all"
                rows.append(row)
                file_means = ok.groupby(["file", "stim_frequency_hz"], as_index=False)[
                    [f"pre_{col}", f"post_{col}"]
                ].mean()
                row = _paired_summary(file_means, col, "file_mean")
                row["group"] = "all"
                rows.append(row)
                for freq in sorted(ok["stim_frequency_hz"].dropna().unique()):
                    freq_ok = ok[ok["stim_frequency_hz"].eq(freq)]
                    row = _paired_summary(freq_ok, col, "pair")
                    row["group"] = f"{freq:g}Hz"
                    rows.append(row)
                    freq_file_means = file_means[file_means["stim_frequency_hz"].eq(freq)]
                    row = _paired_summary(freq_file_means, col, "file_mean")
                    row["group"] = f"{freq:g}Hz"
                    rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and "wilcoxon_p" in out:
        p = out["wilcoxon_p"].to_numpy(float)
        valid = np.isfinite(p)
        q = np.full_like(p, np.nan)
        if valid.any():
            order = np.argsort(p[valid])
            ranked = p[valid][order]
            n = len(ranked)
            adj = ranked * n / (np.arange(n) + 1)
            adj = np.minimum.accumulate(adj[::-1])[::-1]
            tmp = np.empty_like(adj)
            tmp[order] = np.clip(adj, 0, 1)
            q[valid] = tmp
        out["wilcoxon_p_fdr"] = q
    return out


def analyze_connectivity_dataset(
    config_path: str | Path,
    max_files: int | None = None,
    include_pdc: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config(config_path)
    files = list(iter_mat_files(cfg["project"]["data_dir"]))
    if max_files is not None:
        files = files[:max_files]
    frames = [analyze_connectivity_file(path, cfg, include_pdc=include_pdc) for path in files]
    master = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_root = Path(cfg["project"]["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    master_path = out_root / "stim_seed_to_nonstim_target_connectivity.csv"
    master.to_csv(master_path, index=False)
    summary = summarize_connectivity(master, cfg)
    summary_path = out_root / "stim_seed_to_nonstim_target_stats.csv"
    summary.to_csv(summary_path, index=False)
    return master, summary
