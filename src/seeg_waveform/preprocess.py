from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class DriftReport:
    flagged: bool
    slope_iqr_ratio: float
    slope: float
    intercept: float


@dataclass
class NoiseReport:
    flagged: bool
    nan_fraction: float
    robust_std: float
    peak_to_peak: float
    peak_to_peak_iqr_ratio: float
    line_noise_ratio: float
    flat_flag: bool
    extreme_amplitude_flag: bool
    nonfinite_flag: bool


def window_mask(time: np.ndarray, start: float, stop: float) -> np.ndarray:
    return (time >= start) & (time < stop)


def post_window_for_frequency(
    stim_frequency_hz: float,
    stim_end_by_frequency: dict[str, float],
    duration_sec: float,
    artifact_buffer_sec: float,
) -> tuple[float, float]:
    key = str(int(round(stim_frequency_hz)))
    if key not in stim_end_by_frequency:
        raise ValueError(f"No stimulation end time configured for {stim_frequency_hz:g} Hz")
    start = float(stim_end_by_frequency[key]) + float(artifact_buffer_sec)
    return start, start + float(duration_sec)


def notch_filter(x: np.ndarray, fs: float, freqs: list[float], q: float = 30.0) -> np.ndarray:
    y = np.asarray(x, dtype=float)
    for f0 in freqs:
        if 0 < f0 < fs / 2:
            b, a = signal.iirnotch(float(f0), float(q), fs=float(fs))
            y = signal.filtfilt(b, a, y)
    return y


def detect_drift(x: np.ndarray, t: np.ndarray, threshold: float = 0.5) -> DriftReport:
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    centered_t = t - np.nanmean(t)
    slope, intercept = np.polyfit(centered_t, x, 1)
    fitted_change = abs(slope) * (np.nanmax(centered_t) - np.nanmin(centered_t))
    iqr = np.subtract(*np.nanpercentile(x, [75, 25]))
    ratio = float(fitted_change / (iqr + np.finfo(float).eps))
    return DriftReport(flagged=ratio > threshold, slope=float(slope), intercept=float(intercept), slope_iqr_ratio=ratio)


def estimate_line_noise_ratio(x: np.ndarray, fs: float, freqs: list[float], bandwidth_hz: float = 1.0) -> float:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() < 4:
        return np.nan
    f, pxx = signal.welch(x[finite], fs=fs, nperseg=min(1024, finite.sum()))
    total = np.trapezoid(pxx, f)
    if total <= 0:
        return np.nan
    line_power = 0.0
    for f0 in freqs:
        if 0 < f0 < fs / 2:
            mask = (f >= f0 - bandwidth_hz) & (f <= f0 + bandwidth_hz)
            if np.any(mask):
                line_power += float(np.trapezoid(pxx[mask], f[mask]))
    return float(line_power / total)


def detect_noise(x: np.ndarray, fs: float, cfg: dict) -> NoiseReport:
    x = np.asarray(x, dtype=float)
    noise_cfg = cfg.get("noise", {})
    notch_cfg = cfg.get("notch", {})
    finite = np.isfinite(x)
    nan_fraction = float(1.0 - finite.mean()) if x.size else 1.0
    values = x[finite]
    if values.size == 0:
        return NoiseReport(True, nan_fraction, np.nan, np.nan, np.nan, np.nan, True, False, True)

    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    robust_std = float(1.4826 * mad)
    q75, q25 = np.nanpercentile(values, [75, 25])
    iqr = float(q75 - q25)
    peak_to_peak = float(np.nanmax(values) - np.nanmin(values))
    peak_to_peak_iqr_ratio = float(peak_to_peak / (iqr + np.finfo(float).eps))
    line_noise_ratio = estimate_line_noise_ratio(
        values,
        fs,
        notch_cfg.get("freqs", [60.0]),
        noise_cfg.get("line_noise_bandwidth_hz", 1.0),
    )

    flat_flag = robust_std <= float(noise_cfg.get("flat_mad_epsilon", 1e-12))
    nonfinite_flag = nan_fraction > float(noise_cfg.get("max_nan_fraction", 0.0))
    extreme_flag = peak_to_peak_iqr_ratio > float(noise_cfg.get("extreme_peak_to_peak_iqr_factor", 50.0))
    flagged = bool(flat_flag or nonfinite_flag or extreme_flag)
    return NoiseReport(
        flagged=flagged,
        nan_fraction=nan_fraction,
        robust_std=robust_std,
        peak_to_peak=peak_to_peak,
        peak_to_peak_iqr_ratio=peak_to_peak_iqr_ratio,
        line_noise_ratio=line_noise_ratio,
        flat_flag=bool(flat_flag),
        extreme_amplitude_flag=bool(extreme_flag),
        nonfinite_flag=bool(nonfinite_flag),
    )


def preprocess_segment(x: np.ndarray, t: np.ndarray, fs: float, cfg: dict) -> tuple[np.ndarray, DriftReport, NoiseReport]:
    y = np.asarray(x, dtype=float).copy()
    noise_report = detect_noise(y, fs, cfg)
    if cfg.get("demean", True):
        y = y - np.nanmean(y)

    notch_cfg = cfg.get("notch", {})
    if notch_cfg.get("enabled", True):
        y = notch_filter(y, fs, notch_cfg.get("freqs", [60.0]), notch_cfg.get("q", 30.0))

    drift_cfg = cfg.get("drift", {})
    report = detect_drift(y, t, drift_cfg.get("slope_iqr_ratio_threshold", 0.5))
    if drift_cfg.get("enabled", True) and drift_cfg.get("detrend_when_flagged", True) and report.flagged:
        y = signal.detrend(y, type="linear")
        y = y - np.nanmean(y)
    return y, report, noise_report
