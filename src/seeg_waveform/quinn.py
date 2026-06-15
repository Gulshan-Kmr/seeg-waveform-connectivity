from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SegmentResult:
    status: str
    n_cycles: int
    cycle_table: object | None
    imf: np.ndarray | None
    ip: np.ndarray | None
    ifreq: np.ndarray | None
    iamp: np.ndarray | None
    phase_aligned_if: np.ndarray | None
    normalized_waveform: np.ndarray | None
    message: str = ""


def build_sift_config(fs: float, max_imfs: int = 6, nphases: int = 4) -> dict:
    return {
        "mask_amp_mode": "ratio_sig",
        "mask_amp": 1,
        "mask_step_factor": 2,
        "ret_mask_freq": True,
        "max_imfs": int(max_imfs),
        "sift_thresh": 1e-8,
        "nphases": int(nphases),
        "imf_opts": {"sd_thresh": 0.05, "env_step_size": 1, "stop_method": "rilling"},
        "envelope_opts": {"interp_method": "mono_pchip"},
        "extrema_opts": {"pad_width": 2, "parabolic_extrema": False},
        "mask_freqs": np.array([80, 40, 20, 10, 5, 2, 1], dtype=float) / float(fs),
    }


def _as_2d_imf(imf: np.ndarray) -> np.ndarray:
    arr = np.asarray(imf)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    return arr


def compute_range(x: np.ndarray) -> float:
    return float(np.nanmax(x) - np.nanmin(x))


def asc2desc(x: np.ndarray) -> float:
    import emd

    pt = emd.cycles.cf_peak_sample(x, interp=True)
    tt = emd.cycles.cf_trough_sample(x, interp=True)
    if pt is None or tt is None:
        return np.nan
    asc = pt + (len(x) - tt)
    return float(asc / len(x))


def peak2trough(x: np.ndarray) -> float:
    import emd

    des = emd.cycles.cf_descending_zero_sample(x, interp=True)
    if des is None:
        return np.nan
    return float(des / len(x))


def analyze_segment(x: np.ndarray, fs: float, cfg: dict) -> SegmentResult:
    import emd

    x = np.asarray(x, dtype=float)
    if x.size < int(fs * 2) or not np.isfinite(x).any():
        return SegmentResult("too_short", 0, None, None, None, None, None, None, None, "Segment too short")

    try:
        sift_cfg = build_sift_config(fs, cfg.get("max_imfs", 6), cfg.get("nphases", 4))
        imf_raw, _ = emd.sift.mask_sift(x[:, None], **sift_cfg)
        imf = _as_2d_imf(imf_raw)
    except Exception as exc:
        return SegmentResult("emd_failed", 0, None, None, None, None, None, None, None, str(exc))

    imf_index = int(cfg.get("imf_index", 3))
    if imf.shape[1] <= imf_index:
        return SegmentResult("missing_imf", 0, None, imf, None, None, None, None, None, f"Only {imf.shape[1]} IMFs")

    try:
        ip, ifreq, iamp = emd.spectra.frequency_transform(
            imf, fs, "hilbert", smooth_phase=int(cfg.get("smooth_phase", 3))
        )
        ip = _as_2d_imf(ip)
        ifreq = _as_2d_imf(ifreq)
        iamp = _as_2d_imf(iamp)

        cycles = emd.cycles.Cycles(ip[:, imf_index])
        cycles.compute_cycle_metric("max_amp", iamp[:, imf_index], np.max)
        cycles.compute_cycle_metric("mean_if", ifreq[:, imf_index], np.mean)
        cycles.compute_cycle_metric("max_if", ifreq[:, imf_index], np.max)
        cycles.compute_cycle_metric("if_range", ifreq[:, imf_index], compute_range)
        cycles.compute_cycle_metric("asc2desc", imf[:, imf_index], asc2desc)
        cycles.compute_cycle_metric("peak2trough", imf[:, imf_index], peak2trough)

        amp_thresh = np.nanpercentile(iamp[:, imf_index], float(cfg.get("amp_percentile", 25.0)))
        conditions = [
            "is_good==1",
            f"max_amp>{amp_thresh}",
            f"max_if<{float(cfg.get('max_if_hz', 20.0))}",
            f"if_range<{float(cfg.get('max_if_range_hz', 10.0))}",
        ]
        cycles.pick_cycle_subset(conditions)
        table = cycles.get_metric_dataframe(subset=True)
    except Exception as exc:
        return SegmentResult("cycle_failed", 0, None, imf, None, None, None, None, None, str(exc))

    min_cycles = int(cfg.get("min_cycles_required", 3))
    if len(table) < min_cycles:
        return SegmentResult("too_few_cycles", len(table), table, imf, ip, ifreq, iamp, None, None)

    try:
        pa_if, _ = emd.cycles.phase_align(ip[:, imf_index], ifreq[:, imf_index], cycles=cycles.iterate(through="subset"))
        norm_waveform, _ = emd.cycles.normalised_waveform(pa_if)
    except Exception as exc:
        return SegmentResult("align_failed", len(table), table, imf, ip, ifreq, iamp, None, None, str(exc))

    return SegmentResult("ok", len(table), table, imf, ip, ifreq, iamp, pa_if, norm_waveform)


def summarize_decomposed_imf(result: SegmentResult, imf_index: int, cfg: dict) -> SegmentResult:
    """Compute cycle-level waveform profiles for one IMF from an existing decomposition."""
    import emd

    if result.imf is None or result.ip is None or result.ifreq is None or result.iamp is None:
        return SegmentResult("missing_decomposition", 0, None, result.imf, result.ip, result.ifreq, result.iamp, None, None)
    if result.imf.shape[1] <= imf_index:
        return SegmentResult(
            "missing_imf",
            0,
            None,
            result.imf,
            result.ip,
            result.ifreq,
            result.iamp,
            None,
            None,
            f"Only {result.imf.shape[1]} IMFs",
        )

    try:
        cycles = emd.cycles.Cycles(result.ip[:, imf_index])
        cycles.compute_cycle_metric("max_amp", result.iamp[:, imf_index], np.max)
        cycles.compute_cycle_metric("mean_if", result.ifreq[:, imf_index], np.mean)
        cycles.compute_cycle_metric("max_if", result.ifreq[:, imf_index], np.max)
        cycles.compute_cycle_metric("if_range", result.ifreq[:, imf_index], compute_range)
        cycles.compute_cycle_metric("asc2desc", result.imf[:, imf_index], asc2desc)
        cycles.compute_cycle_metric("peak2trough", result.imf[:, imf_index], peak2trough)

        amp_thresh = np.nanpercentile(result.iamp[:, imf_index], float(cfg.get("amp_percentile", 25.0)))
        conditions = [
            "is_good==1",
            f"max_amp>{amp_thresh}",
            f"max_if<{float(cfg.get('max_if_hz', 20.0))}",
            f"if_range<{float(cfg.get('max_if_range_hz', 10.0))}",
        ]
        cycles.pick_cycle_subset(conditions)
        table = cycles.get_metric_dataframe(subset=True)
    except Exception as exc:
        return SegmentResult("cycle_failed", 0, None, result.imf, result.ip, result.ifreq, result.iamp, None, None, str(exc))

    min_cycles = int(cfg.get("min_cycles_required", 3))
    if len(table) < min_cycles:
        return SegmentResult("too_few_cycles", len(table), table, result.imf, result.ip, result.ifreq, result.iamp, None, None)

    try:
        pa_if, _ = emd.cycles.phase_align(
            result.ip[:, imf_index],
            result.ifreq[:, imf_index],
            cycles=cycles.iterate(through="subset"),
        )
        norm_waveform, _ = emd.cycles.normalised_waveform(pa_if)
    except Exception as exc:
        return SegmentResult(
            "align_failed",
            len(table),
            table,
            result.imf,
            result.ip,
            result.ifreq,
            result.iamp,
            None,
            None,
            str(exc),
        )

    return SegmentResult(
        "ok",
        len(table),
        table,
        result.imf,
        result.ip,
        result.ifreq,
        result.iamp,
        pa_if,
        norm_waveform,
    )
