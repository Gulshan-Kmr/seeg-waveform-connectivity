from __future__ import annotations

import argparse
from itertools import product
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
from seeg_waveform.quinn import SegmentResult, summarize_decomposed_imf
from seeg_waveform.quinn_figures import set_publication_style


COLORS = {1.0: "#277da1", 50.0: "#d1495b"}
METRICS = {
    "delta_asc2desc": "Post - pre ascent/descent (x10$^{-3}$)",
    "delta_peak2trough": "Post - pre peak/trough (x10$^{-3}$)",
}
NODE_GROUPS = {
    "nonstimulated": "Non-stimulated nodes",
    "seed": "Stimulated seed nodes",
}


def exact_signflip(values: pd.Series | np.ndarray) -> float:
    x = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if x.size == 0:
        return np.nan
    observed = abs(float(x.mean()))
    means = [abs(float(np.mean(x * np.asarray(signs)))) for signs in product([-1.0, 1.0], repeat=x.size)]
    return float(np.mean(np.asarray(means) >= observed - np.finfo(float).eps))


def metric_mean(result: SegmentResult, metric: str) -> float:
    if result.cycle_table is None or metric not in result.cycle_table:
        return np.nan
    return float(pd.to_numeric(result.cycle_table[metric], errors="coerce").mean())


def all_imf_cfg(cfg: dict) -> dict:
    """Use broad cycle QC so IMF1-IMF6 shape summaries are descriptive and comparable."""
    out = dict(cfg["emd"])
    out["max_if_hz"] = float(cfg["emd"].get("all_imf_max_if_hz", 120.0))
    out["max_if_range_hz"] = float(cfg["emd"].get("all_imf_max_if_range_hz", 120.0))
    return out


def control_point_rows_for_run(path: Path, cfg: dict) -> list[dict]:
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
    imf_cfg = all_imf_cfg(cfg)
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
            pre_result = summarize_decomposed_imf(pre[node_index].result, imf_number - 1, imf_cfg)
            post_result = summarize_decomposed_imf(post[node_index].result, imf_number - 1, imf_cfg)
            if pre_result.status != "ok" or post_result.status != "ok":
                continue
            row = {
                "file": run.recording.file_stem,
                "subject": run.recording.subject,
                "stim_frequency_hz": run.recording.stim_frequency_hz,
                "stim_amplitude_ma": run.recording.stim_amplitude_ma,
                "channel": node["channel"],
                "region_family": node["region_family"],
                "node_group": "seed" if bool(node["is_stimulated_channel"]) else "nonstimulated",
                "imf": imf_number,
                "pre_n_cycles": int(pre_result.n_cycles),
                "post_n_cycles": int(post_result.n_cycles),
            }
            for metric in ("asc2desc", "peak2trough"):
                row[f"pre_{metric}"] = metric_mean(pre_result, metric)
                row[f"post_{metric}"] = metric_mean(post_result, metric)
                row[f"delta_{metric}"] = row[f"post_{metric}"] - row[f"pre_{metric}"]
            rows.append(row)
    return rows


def build_control_point_table(cfg_path: Path, analysis_set: str) -> pd.DataFrame:
    cfg = load_config(cfg_path)
    support_dir = Path(cfg["project"]["output_dir"]) / analysis_set / "tables" / "figure_support"
    support_dir.mkdir(parents=True, exist_ok=True)
    cache_path = support_dir / "imf1_imf6_control_point_node_values.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)
    files = list(iter_mat_files(Path(cfg["project"]["data_dir"])))
    exclusions = duplicate_recording_exclusions(files)
    rows: list[dict] = []
    for path in files:
        if path in exclusions:
            continue
        rows.extend(control_point_rows_for_run(path, cfg))
    table = pd.DataFrame(rows)
    table.to_csv(cache_path, index=False)
    return table


def subject_values(table: pd.DataFrame) -> pd.DataFrame:
    metrics = list(METRICS)
    run = (
        table.groupby(["file", "subject", "stim_frequency_hz", "node_group", "imf"], as_index=False)[metrics]
        .mean(numeric_only=True)
    )
    return (
        run.groupby(["subject", "stim_frequency_hz", "node_group", "imf"], as_index=False)[metrics]
        .mean(numeric_only=True)
        .sort_values(["node_group", "stim_frequency_hz", "imf", "subject"])
    )


def summarize_stats(subject: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (node_group, imf, freq), group in subject.groupby(["node_group", "imf", "stim_frequency_hz"]):
        row = {
            "node_group": node_group,
            "imf": int(imf),
            "stim_frequency_hz": float(freq),
            "comparison": f"{freq:g}Hz_post_minus_pre",
            "n_subjects": int(group["subject"].nunique()),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_median"] = float(values.median()) if len(values) else np.nan
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_exact_signflip_p"] = exact_signflip(values)
        rows.append(row)
    return pd.DataFrame(rows)


def _arrays_for(subject: pd.DataFrame, node_group: str, metric: str, imf_number: int) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for freq in (1.0, 50.0):
        values = subject.loc[
            subject["node_group"].eq(node_group)
            & subject["imf"].eq(imf_number)
            & subject["stim_frequency_hz"].eq(freq),
            metric,
        ]
        arrays.append(values.dropna().to_numpy(float) * 1000.0)
    return arrays


def _ylim(arrays: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([array[np.isfinite(array)] for array in arrays if array.size])
    if values.size == 0:
        return (-1.0, 1.0)
    limit = max(float(np.nanmax(np.abs(values))), 1e-6) * 1.18
    return (-limit, limit)


def _draw_metric_panel(ax: plt.Axes, arrays: list[np.ndarray], ylabel: str, title: str, stats_text: list[str]) -> None:
    positions = [1, 2]
    nonempty = [(pos, arr) for pos, arr in zip(positions, arrays) if arr.size]
    if nonempty:
        violins = ax.violinplot(
            [arr for _, arr in nonempty],
            positions=[pos for pos, _ in nonempty],
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.55,
        )
        for body, (pos, _) in zip(violins["bodies"], nonempty):
            freq = 1.0 if pos == 1 else 50.0
            body.set_facecolor(COLORS[freq])
            body.set_edgecolor(COLORS[freq])
            body.set_alpha(0.24)
    rng = np.random.default_rng(616)
    for pos, freq, arr, text in zip(positions, (1.0, 50.0), arrays, stats_text):
        if not arr.size:
            continue
        ax.scatter(
            np.full(arr.size, pos) + rng.normal(0, 0.035, arr.size),
            arr,
            s=18,
            color=COLORS[freq],
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        ax.hlines(np.median(arr), pos - 0.22, pos + 0.22, color=COLORS[freq], linewidth=1.8)
        ax.text(pos, 0.98, text, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.8)
    ax.axhline(0, color="0.45", linestyle="--", linewidth=0.65)
    ax.set_xticks(positions, ["1 Hz", "50 Hz"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(_ylim(arrays))
    ax.grid(axis="y", color="0.91", linewidth=0.4)


def plot_imf_control_points(subject: pd.DataFrame, stats: pd.DataFrame, out_path: Path, node_group: str) -> Path:
    set_publication_style()
    fig, axes = plt.subplots(2, 6, figsize=(15.0, 5.8), constrained_layout=True)
    for row_index, (metric, ylabel) in enumerate(METRICS.items()):
        for imf_number in range(1, 7):
            ax = axes[row_index, imf_number - 1]
            arrays = _arrays_for(subject, node_group, metric, imf_number)
            texts: list[str] = []
            for freq in (1.0, 50.0):
                row = stats.loc[
                    stats["node_group"].eq(node_group)
                    & stats["imf"].eq(imf_number)
                    & stats["comparison"].eq(f"{freq:g}Hz_post_minus_pre")
                ]
                if row.empty:
                    texts.append("n=0")
                else:
                    p = row.iloc[0][f"{metric}_exact_signflip_p"]
                    n = int(row.iloc[0]["n_subjects"])
                    texts.append(f"n={n}\np={p:.3f}" if np.isfinite(p) else f"n={n}")
            _draw_metric_panel(ax, arrays, ylabel if imf_number == 1 else "", f"IMF{imf_number}", texts)
            if row_index == 1:
                ax.set_xlabel("Protocol")
            if row_index == 0:
                ax.set_xticklabels([])
    fig.suptitle(f"IMF1-IMF6 within-cycle control-point changes: {NODE_GROUPS[node_group]}", y=1.03)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create IMF1-IMF6 control-point violin figures without mean IF.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "primary_bipolar_mne_config.yaml")
    parser.add_argument("--analysis-set", default="primary_28s_guarded")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["project"]["output_dir"]) / args.analysis_set
    support_dir = root / "tables" / "figure_support"
    fig_dir = root / "figures" / "corrected_addons"
    support_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    table = build_control_point_table(args.config, args.analysis_set)
    subject = subject_values(table)
    stats = summarize_stats(subject)
    subject_path = support_dir / "imf1_imf6_control_point_subject_values.csv"
    stats_path = support_dir / "imf1_imf6_control_point_stats.csv"
    subject.to_csv(subject_path, index=False)
    stats.to_csv(stats_path, index=False)
    nonstim_path = plot_imf_control_points(
        subject, stats, fig_dir / "imf1_imf6_control_point_violins_nonstimulated.png", "nonstimulated"
    )
    seed_path = plot_imf_control_points(
        subject, stats, fig_dir / "imf1_imf6_control_point_violins_stimulated_seed.png", "seed"
    )
    print(f"node_values: {support_dir / 'imf1_imf6_control_point_node_values.csv'}")
    print(f"subject_values: {subject_path}")
    print(f"stats: {stats_path}")
    print(f"nonstimulated_figure: {nonstim_path}")
    print(f"stimulated_seed_figure: {seed_path}")


if __name__ == "__main__":
    main()
