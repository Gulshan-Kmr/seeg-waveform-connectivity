from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seeg_waveform.config import load_config
from seeg_waveform.io import iter_mat_files
from seeg_waveform.preprocess import window_mask
from seeg_waveform.primary_bipolar_mne import (
    _decompose_window,
    analysis_windows,
    duplicate_recording_exclusions,
    make_bipolar_run,
)
from seeg_waveform.primary_car_mne import make_car_run
from seeg_waveform.quinn import summarize_decomposed_imf
from seeg_waveform.quinn_figures import set_publication_style


COLORS = {"pre": "#277da1", "post": "#d1495b"}


def _all_imf_cfg(cfg: dict) -> dict:
    """Use broad cycle QC for descriptive IMF1-IMF6 waveform profiles."""
    out = dict(cfg["emd"])
    out["max_if_hz"] = float(cfg["emd"].get("all_imf_max_if_hz", 120.0))
    out["max_if_range_hz"] = float(cfg["emd"].get("all_imf_max_if_range_hz", 120.0))
    return out


def _profile_rows_for_run(path: Path, cfg: dict) -> list[dict]:
    if str(cfg["design"].get("montage", "")).lower() == "common_average_unipolar":
        run, pre, post = make_car_run(path, cfg)
        if run.status != "ok" or pre is None or post is None:
            return []
    else:
        run = make_bipolar_run(path, cfg)
        if run.status != "ok":
            return []
        pre_window, post_window = analysis_windows(run.recording, cfg)
        pre_mask = window_mask(run.recording.time, *pre_window)
        post_mask = window_mask(run.recording.time, *post_window)
        pre = _decompose_window(run.recording, pre_mask, cfg)
        post = _decompose_window(run.recording, post_mask, cfg)
    if run.status != "ok":
        return []
    imf_cfg = _all_imf_cfg(cfg)
    rows: list[dict] = []
    for node_index, node in run.nodes.iterrows():
        include = (
            bool(node["primary_region_include"])
            and not pre[node_index].noise_flag
            and not post[node_index].noise_flag
        )
        if not include:
            continue
        for imf_number in range(1, 7):
            for epoch, window in [("pre", pre[node_index]), ("post", post[node_index])]:
                result = summarize_decomposed_imf(window.result, imf_number - 1, imf_cfg)
                if result.status != "ok" or result.phase_aligned_if is None or result.normalized_waveform is None:
                    continue
                if_profile = np.nanmean(result.phase_aligned_if, axis=1)
                wave_profile = np.nanmean(result.normalized_waveform, axis=1)
                for phase_index, (if_value, wave_value) in enumerate(zip(if_profile, wave_profile)):
                    rows.append(
                        {
                            "file": run.recording.file_stem,
                            "subject": run.recording.subject,
                            "stim_frequency_hz": run.recording.stim_frequency_hz,
                            "channel": node["channel"],
                            "region_family": node["region_family"],
                            "is_stimulated_channel": bool(node["is_stimulated_channel"]),
                            "imf": imf_number,
                            "epoch": epoch,
                            "phase_index": phase_index,
                            "phase_rad": 2 * np.pi * phase_index / len(if_profile),
                            "mean_if_hz": float(if_value),
                            "normalized_waveform": float(wave_value),
                            "n_cycles": int(result.n_cycles),
                        }
                    )
    return rows


def build_imf_waveform_profiles(cfg_path: Path, analysis_set: str) -> pd.DataFrame:
    cfg = load_config(cfg_path)
    out_dir = Path(cfg["project"]["output_dir"]) / analysis_set / "tables" / "figure_support"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "imf1_imf6_waveform_profiles.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)
    files = list(iter_mat_files(Path(cfg["project"]["data_dir"])))
    exclusions = duplicate_recording_exclusions(files)
    rows: list[dict] = []
    for path in files:
        if path in exclusions:
            continue
        rows.extend(_profile_rows_for_run(path, cfg))
    profiles = pd.DataFrame(rows)
    profiles.to_csv(cache_path, index=False)
    return profiles


def _subject_balanced_profiles(profiles: pd.DataFrame, include_seed: bool = False) -> pd.DataFrame:
    data = profiles.copy()
    if not include_seed:
        data = data.loc[~data["is_stimulated_channel"].astype(bool)].copy()
    run_node = (
        data.groupby(
            ["file", "subject", "stim_frequency_hz", "channel", "imf", "epoch", "phase_rad"],
            as_index=False,
        )[["mean_if_hz", "normalized_waveform"]]
        .mean()
    )
    run = (
        run_node.groupby(["file", "subject", "stim_frequency_hz", "imf", "epoch", "phase_rad"], as_index=False)[
            ["mean_if_hz", "normalized_waveform"]
        ]
        .mean()
    )
    return (
        run.groupby(["subject", "stim_frequency_hz", "imf", "epoch", "phase_rad"], as_index=False)[
            ["mean_if_hz", "normalized_waveform"]
        ]
        .mean()
    )


def _draw_profile(ax: plt.Axes, data: pd.DataFrame, metric: str) -> None:
    for epoch, color in COLORS.items():
        selected = data.loc[data["epoch"].eq(epoch)]
        for _, curve in selected.groupby("subject"):
            ax.plot(curve["phase_rad"], curve[metric], color=color, lw=0.35, alpha=0.14)
        summary = selected.groupby("phase_rad")[metric].agg(mean="mean", sem="sem").reset_index()
        x = summary["phase_rad"].to_numpy()
        mean = summary["mean"].to_numpy()
        sem = summary["sem"].fillna(0).to_numpy()
        if metric == "normalized_waveform" and x.size and x[-1] < 2 * np.pi:
            x = np.append(x, 2 * np.pi)
            mean = np.append(mean, mean[0])
            sem = np.append(sem, sem[0])
        ax.plot(x, mean, color=color, lw=1.3, label=f"{epoch.capitalize()}-stimulation interval")
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.15, lw=0)
    if metric == "normalized_waveform":
        phase = np.linspace(0, 2 * np.pi, 160)
        ax.plot(phase, np.sin(phase), color="0.2", lw=0.6, ls="--", label="Sinusoidal reference")
    ax.set_xticks([0, np.pi, 2 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
    ax.grid(axis="y", color="0.92", lw=0.45)


def plot_imf_waveform_profiles(
    profiles: pd.DataFrame,
    cfg_path: Path,
    analysis_set: str,
    include_seed: bool = False,
    out_name: str = "imf1_imf6_within_cycle_waveform_profiles_nonstimulated.png",
) -> Path:
    cfg = load_config(cfg_path)
    subject = _subject_balanced_profiles(profiles, include_seed=include_seed)
    fig_dir = Path(cfg["project"]["output_dir"]) / analysis_set / "figures" / "corrected_addons"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / out_name
    montage = str(cfg["design"].get("montage", "bipolar")).lower()
    node_kind = "CAR-referenced contacts" if montage == "common_average_unipolar" else "recording sites"
    node_label = f"stimulated {node_kind}" if include_seed else f"non-stimulated {node_kind}"
    set_publication_style()
    fig, axes = plt.subplots(6, 4, figsize=(12.0, 12.4), sharex=True, constrained_layout=True)
    for row, imf_number in enumerate(range(1, 7)):
        for col, (freq, metric, title) in enumerate(
            [
                (1.0, "mean_if_hz", "1 Hz IF"),
                (1.0, "normalized_waveform", "1 Hz waveform"),
                (50.0, "mean_if_hz", "50 Hz IF"),
                (50.0, "normalized_waveform", "50 Hz waveform"),
            ]
        ):
            ax = axes[row, col]
            selected = subject.loc[subject["imf"].eq(imf_number) & subject["stim_frequency_hz"].eq(freq)]
            _draw_profile(ax, selected, metric)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"IMF{imf_number}\nIF (Hz)")
            elif col == 2:
                ax.set_ylabel("IF (Hz)")
            elif col in (1, 3):
                ax.set_ylabel("Norm. waveform")
            if row == 5:
                ax.set_xlabel("Phase (radians)")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"IMF1-IMF6 within-cycle waveform profiles in {node_label}", y=1.035)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    selected_profiles = profiles.loc[profiles["is_stimulated_channel"].astype(bool)].copy() if include_seed else profiles.loc[
        ~profiles["is_stimulated_channel"].astype(bool)
    ].copy()
    counts = (
        selected_profiles
        .groupby(["stim_frequency_hz", "imf"])[["subject", "file", "channel"]]
        .nunique()
        .reset_index()
    )
    support_dir = Path(cfg["project"]["output_dir"]) / analysis_set / "tables" / "figure_support"
    suffix = "seed" if include_seed else "nonstimulated"
    counts.to_csv(support_dir / f"imf1_imf6_waveform_profile_support_{suffix}.csv", index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create IMF1-IMF6 within-cycle waveform Extended Data figure.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "primary_bipolar_mne_config.yaml")
    parser.add_argument("--analysis-set", default="primary_28s_guarded")
    args = parser.parse_args()
    profiles = build_imf_waveform_profiles(args.config, args.analysis_set)
    nonstim_path = plot_imf_waveform_profiles(
        profiles,
        args.config,
        args.analysis_set,
        include_seed=False,
        out_name="imf1_imf6_within_cycle_waveform_profiles_nonstimulated.png",
    )
    seed_path = plot_imf_waveform_profiles(
        profiles,
        args.config,
        args.analysis_set,
        include_seed=True,
        out_name="imf1_imf6_within_cycle_waveform_profiles_stimulated_seed.png",
    )
    print(f"profiles: {len(profiles)} rows")
    print(f"nonstimulated_figure: {nonstim_path}")
    print(f"stimulated_seed_figure: {seed_path}")


if __name__ == "__main__":
    main()
