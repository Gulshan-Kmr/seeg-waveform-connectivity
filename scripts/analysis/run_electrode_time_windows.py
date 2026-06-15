"""
Electrode-level waveform vs connectivity analysis for three post-stimulation
time windows: 0-10s, 10-20s, 20-28s (relative to post-window start).
"""
from __future__ import annotations

import os
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MNE_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "outputs" / ".numba_cache"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import signal as sp_signal, stats
from statsmodels.stats.multitest import multipletests

from seeg_waveform.config import load_config
from seeg_waveform.io import iter_mat_files
from seeg_waveform.preprocess import (
    post_window_for_frequency,
    window_mask,
)
from seeg_waveform.primary_bipolar_mne import (
    _decompose_window,
    _metric_mean,
    _mne_values,
    _segmented_imfs,
    _share_constituent_contact,
    make_bipolar_run,
)
from seeg_waveform.primary_car_mne import make_car_run
from seeg_waveform.all_pairs_waveform_connectivity import (
    FEATURES,
    ROOT_CONFIGS,
    generalized_pdc_pair,
    temporal_irreversibility_unsigned,
)

OUT = ROOT / "outputs" / "electrode_level_analysis"
OUT.mkdir(parents=True, exist_ok=True)

ALL_CFG = load_config(ROOT / "configs" / "all_pairs_waveform_connectivity_config.yaml")
EXCLUDED = set(ALL_CFG["analysis"].get("excluded_files", []))

# Three non-overlapping 10-second windows (offsets from post-window start)
TIME_WINDOWS = [
    ("0-10s",  0.0, 10.0),
    ("10-20s", 10.0, 20.0),
    ("20-28s", 20.0, 28.0),
]

PDC_SETTINGS = {
    "pdc_downsample_hz": ALL_CFG["analysis"]["pdc_downsample_hz"],
    "pdc_max_order":     ALL_CFG["analysis"]["pdc_max_order"],
    "pdc_residual_whiteness_alpha": ALL_CFG["analysis"]["pdc_residual_whiteness_alpha"],
    "pdc_epsilon":       ALL_CFG["analysis"]["pdc_epsilon"],
    "tra_max_lag_sec":   ALL_CFG["analysis"]["tra_max_lag_sec"],
    "tra_covariance_epsilon": ALL_CFG["analysis"]["tra_covariance_epsilon"],
}


def process_file(path: Path, montage: str) -> list[dict]:
    base_cfg = load_config(ROOT_CONFIGS[montage])
    cfg_win = base_cfg["windows"]

    if montage == "car":
        run, _, _ = make_car_run(path, base_cfg)
    else:
        run = make_bipolar_run(path, base_cfg)
        if run.status != "ok":
            return []
    if run.status != "ok":
        return []

    # Determine the post-window start
    post_start, _ = post_window_for_frequency(
        run.recording.stim_frequency_hz,
        cfg_win["stim_end_by_frequency"],
        cfg_win["post_duration_sec"],
        cfg_win["post_artifact_buffer_sec"],
    )

    rows = []
    for win_label, rel_start, rel_end in TIME_WINDOWS:
        abs_start = post_start + rel_start
        abs_end = post_start + rel_end
        mask = window_mask(run.recording.time, abs_start, abs_end)
        if mask.sum() < int(run.recording.fs * 5):
            continue

        windows = _decompose_window(run.recording, mask, base_cfg)
        eligible = [
            i for i, node in run.nodes.iterrows()
            if bool(node["primary_region_include"])
            and not windows[i].noise_flag
            and windows[i].result.status == "ok"
        ]
        if len(eligible) < 2:
            continue

        pairs = [
            (a, b) for a, b in combinations(eligible, 2)
            if montage == "car" or not _share_constituent_contact(
                run.nodes.iloc[a], run.nodes.iloc[b]
            )
        ]
        local = {idx: pos for pos, idx in enumerate(eligible)}
        mne_pairs = [(local[a], local[b]) for a, b in pairs]
        segmented = _segmented_imfs(windows, eligible, run.recording, base_cfg, 4)
        conn = _mne_values(segmented, mne_pairs, run.recording.fs, base_cfg)

        signals_map  = {i: np.asarray(windows[i].result.imf[:, 3], float) for i in eligible}
        envelopes_map = {i: np.abs(sp_signal.hilbert(v)) for i, v in signals_map.items()}

        # per-electrode waveform features
        feat = {}
        for i in eligible:
            node = run.nodes.iloc[i]
            feat[i] = {
                "file": run.recording.file_stem,
                "subject": run.recording.subject,
                "stim_frequency_hz": run.recording.stim_frequency_hz,
                "montage": montage,
                "time_window": win_label,
                "channel": node["channel"],
                **{m: _metric_mean(windows[i].result, m) for m in FEATURES},
            }

        # pair-level connectivity → aggregate to electrode level
        elec_imcoh_sum   = {i: [] for i in eligible}
        elec_pdc_out_sum = {i: [] for i in eligible}
        elec_pdc_in_sum  = {i: [] for i in eligible}
        elec_tra_sum     = {i: [] for i in eligible}

        for pi, (a, b) in enumerate(pairs):
            imcoh = abs(float(conn["imcoh"][pi]))
            elec_imcoh_sum[a].append(imcoh)
            elec_imcoh_sum[b].append(imcoh)

            pdc_result = generalized_pdc_pair(
                signals_map[a], signals_map[b], run.recording.fs,
                target_fs=PDC_SETTINGS["pdc_downsample_hz"],
                max_order=PDC_SETTINGS["pdc_max_order"],
                whiteness_alpha=PDC_SETTINGS["pdc_residual_whiteness_alpha"],
                epsilon=PDC_SETTINGS["pdc_epsilon"],
            )
            if pdc_result.get("pdc_valid", False):
                elec_pdc_out_sum[a].append(pdc_result["gpdc_a_to_b"])
                elec_pdc_in_sum[a].append(pdc_result["gpdc_b_to_a"])
                elec_pdc_out_sum[b].append(pdc_result["gpdc_b_to_a"])
                elec_pdc_in_sum[b].append(pdc_result["gpdc_a_to_b"])

            tra, tra_status = temporal_irreversibility_unsigned(
                envelopes_map[a], envelopes_map[b],
                int(round(PDC_SETTINGS["tra_max_lag_sec"] * run.recording.fs)),
                PDC_SETTINGS["tra_covariance_epsilon"],
            )
            if tra_status == "ok":
                elec_tra_sum[a].append(tra)
                elec_tra_sum[b].append(tra)

        for i in eligible:
            row = feat[i].copy()
            row["mean_abs_imcoh"] = np.mean(elec_imcoh_sum[i]) if elec_imcoh_sum[i] else np.nan
            out_vals = elec_pdc_out_sum[i]
            in_vals  = elec_pdc_in_sum[i]
            row["mean_gpdc_out"]  = np.mean(out_vals) if out_vals else np.nan
            row["mean_gpdc_in"]   = np.mean(in_vals)  if in_vals  else np.nan
            row["pdc_asymmetry"]  = (np.mean(out_vals) - np.mean(in_vals)) if (out_vals and in_vals) else np.nan
            row["mean_tra"] = np.mean(elec_tra_sum[i]) if elec_tra_sum[i] else np.nan
            rows.append(row)

    return rows


def main():
    files = [p for p in iter_mat_files(ROOT / "data") if p.name not in EXCLUDED]
    files.sort(key=lambda p: (p.stat().st_size, p.name))
    print(f"Processing {len(files)} files across bipolar + CAR …")

    all_rows = []
    for path in files:
        for montage in ("bipolar", "car"):
            print(f"  {montage}: {path.name}", flush=True)
            try:
                all_rows.extend(process_file(path, montage))
            except Exception as exc:
                print(f"    ERROR: {exc}")

    elec = pd.DataFrame(all_rows)
    elec.to_csv(OUT / "electrode_time_windows.csv", index=False)
    print(f"\nSaved {len(elec)} electrode rows → {OUT / 'electrode_time_windows.csv'}")

    # Spearman per subject x montage x time_window x hypothesis
    hypotheses = [
        {"name": "asc_pdc",  "x": "asc2desc",    "y": "pdc_asymmetry"},
        {"name": "if_tra",   "x": "mean_if",      "y": "mean_tra"},
        {"name": "if_imcoh", "x": "mean_if",      "y": "mean_abs_imcoh"},
        {"name": "asc_imcoh","x": "asc2desc",     "y": "mean_abs_imcoh"},
        {"name": "pt_tra",   "x": "peak2trough",  "y": "mean_tra"},
    ]
    corr_rows = []
    for (montage, win, subject), grp in elec.groupby(["montage", "time_window", "subject"]):
        for hyp in hypotheses:
            clean = grp[[hyp["x"], hyp["y"]]].dropna()
            if len(clean) < 5:
                continue
            rho, p = stats.spearmanr(clean[hyp["x"]], clean[hyp["y"]])
            corr_rows.append({
                "montage": montage, "time_window": win, "subject": subject,
                "hypothesis": hyp["name"], "rho": rho, "p": p, "n": len(clean),
            })

    corr = pd.DataFrame(corr_rows)
    corr["p_fdr"] = np.nan
    for (montage, win, hyp), idx in corr.groupby(["montage", "time_window", "hypothesis"]).groups.items():
        vals = corr.loc[idx, "p"].values
        valid = np.isfinite(vals)
        if valid.sum() > 1:
            corr.loc[corr.index[idx][valid], "p_fdr"] = multipletests(vals[valid], method="fdr_bh")[1]

    corr.to_csv(OUT / "electrode_time_window_spearman.csv", index=False)
    print("Saved Spearman results →", OUT / "electrode_time_window_spearman.csv")
    print(corr.groupby(["montage", "time_window", "hypothesis"])[["rho", "p_fdr"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
