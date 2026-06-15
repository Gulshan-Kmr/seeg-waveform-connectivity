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
from seeg_waveform.quinn_figures import set_publication_style


COLORS = {1.0: "#277da1", 50.0: "#d1495b"}
METRICS = {
    "delta_mean_if": {
        "title": "Mean instantaneous frequency",
        "ylabel": "Post - pre mean IF (Hz)",
        "scale": 1.0,
    },
    "delta_asc2desc": {
        "title": "Ascent/descent duration ratio",
        "ylabel": "Post - pre ascent/descent (x10$^{-3}$)",
        "scale": 1000.0,
    },
    "delta_peak2trough": {
        "title": "Peak/trough duration ratio",
        "ylabel": "Post - pre peak/trough (x10$^{-3}$)",
        "scale": 1000.0,
    },
}
NODE_GROUPS = {
    "nonstimulated": "Non-stimulated recording sites",
    "seed": "Stimulated seed contacts",
}


def exact_signflip(values: pd.Series | np.ndarray) -> float:
    x = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if x.size == 0:
        return np.nan
    observed = abs(float(x.mean()))
    means = [abs(float(np.mean(x * np.asarray(signs)))) for signs in product([-1.0, 1.0], repeat=x.size)]
    return float(np.mean(np.asarray(means) >= observed - np.finfo(float).eps))


def waveform_subject_values(waveform: pd.DataFrame) -> pd.DataFrame:
    clean = waveform.loc[waveform["analysis_include"].fillna(False).astype(bool)].copy()
    clean["node_group"] = np.where(clean["is_stimulated_channel"].fillna(False).astype(bool), "seed", "nonstimulated")
    metrics = list(METRICS)
    run = (
        clean.groupby(["file", "subject", "stim_frequency_hz", "node_group"], as_index=False)[metrics]
        .mean(numeric_only=True)
    )
    return (
        run.groupby(["subject", "stim_frequency_hz", "node_group"], as_index=False)[metrics]
        .mean(numeric_only=True)
        .sort_values(["node_group", "stim_frequency_hz", "subject"])
    )


def summarize_stats(subject: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for node_group, group_df in subject.groupby("node_group"):
        for freq, freq_df in group_df.groupby("stim_frequency_hz"):
            row = {
                "node_group": node_group,
                "comparison": f"{freq:g}Hz_post_minus_pre",
                "stim_frequency_hz": freq,
                "n_subjects": int(freq_df["subject"].nunique()),
            }
            for metric in METRICS:
                values = pd.to_numeric(freq_df[metric], errors="coerce").dropna()
                row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
                row[f"{metric}_median"] = float(values.median()) if len(values) else np.nan
                row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
                row[f"{metric}_exact_signflip_p"] = exact_signflip(values)
            rows.append(row)
        pivot = group_df.pivot_table(index="subject", columns="stim_frequency_hz", values=list(METRICS), aggfunc="mean")
        row = {
            "node_group": node_group,
            "comparison": "50Hz_minus_1Hz_paired",
            "stim_frequency_hz": np.nan,
            "n_subjects": 0,
        }
        for metric in METRICS:
            if metric in pivot and 1.0 in pivot[metric] and 50.0 in pivot[metric]:
                diffs = (pivot[metric][50.0] - pivot[metric][1.0]).dropna()
            else:
                diffs = pd.Series(dtype=float)
            row["n_subjects"] = max(int(row["n_subjects"]), int(len(diffs)))
            row[f"{metric}_mean"] = float(diffs.mean()) if len(diffs) else np.nan
            row[f"{metric}_median"] = float(diffs.median()) if len(diffs) else np.nan
            row[f"{metric}_sd"] = float(diffs.std(ddof=1)) if len(diffs) > 1 else np.nan
            row[f"{metric}_exact_signflip_p"] = exact_signflip(diffs)
        rows.append(row)
    return pd.DataFrame(rows)


def _scaled_values(subject: pd.DataFrame, metric: str, node_group: str, freq: float) -> np.ndarray:
    values = subject.loc[
        subject["node_group"].eq(node_group) & subject["stim_frequency_hz"].eq(freq), metric
    ].dropna()
    return values.to_numpy(float) * float(METRICS[metric]["scale"])


def _symmetric_ylim(arrays: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([array[np.isfinite(array)] for array in arrays if array.size])
    if values.size == 0:
        return (-1.0, 1.0)
    limit = max(float(np.nanmax(np.abs(values))), float(np.nanpercentile(np.abs(values), 90)) * 1.35, 1e-6)
    return (-limit * 1.18, limit * 1.18)


def _draw_violin(ax: plt.Axes, arrays: list[np.ndarray], ylabel: str, stats_text: list[str]) -> None:
    positions = [1, 2]
    nonempty_arrays = [array for array in arrays if array.size]
    nonempty_positions = [pos for pos, array in zip(positions, arrays) if array.size]
    if nonempty_arrays:
        violins = ax.violinplot(
            nonempty_arrays,
            positions=nonempty_positions,
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.55,
        )
        for body, pos in zip(violins["bodies"], nonempty_positions):
            freq = 1.0 if pos == 1 else 50.0
            body.set_facecolor(COLORS[freq])
            body.set_edgecolor(COLORS[freq])
            body.set_alpha(0.24)
            body.set_linewidth(1.0)
    rng = np.random.default_rng(202)
    for pos, freq, array, text in zip(positions, (1.0, 50.0), arrays, stats_text):
        if not array.size:
            continue
        jitter = rng.normal(0, 0.035, len(array))
        ax.scatter(
            np.full(len(array), pos) + jitter,
            array,
            s=28,
            color=COLORS[freq],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.hlines(np.median(array), pos - 0.24, pos + 0.24, color=COLORS[freq], linewidth=2.1)
        ax.text(pos, 0.98, text, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.6)
    ax.axhline(0, color="0.45", linestyle="--", linewidth=0.7)
    ax.set_xticks(positions, ["1 Hz", "50 Hz"])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="0.9", linewidth=0.45)
    ax.set_ylim(_symmetric_ylim(arrays))


def plot_control_point_figure(subject: pd.DataFrame, stats: pd.DataFrame, out_path: Path) -> Path:
    set_publication_style()
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.2), constrained_layout=True)
    for row_index, (node_group, group_label) in enumerate(NODE_GROUPS.items()):
        for col_index, (metric, meta) in enumerate(METRICS.items()):
            ax = axes[row_index, col_index]
            arrays = [_scaled_values(subject, metric, node_group, freq) for freq in (1.0, 50.0)]
            stats_text = []
            for freq in (1.0, 50.0):
                row = stats.loc[
                    stats["node_group"].eq(node_group)
                    & stats["comparison"].eq(f"{freq:g}Hz_post_minus_pre")
                ]
                if row.empty:
                    stats_text.append("n=0")
                else:
                    p = row.iloc[0][f"{metric}_exact_signflip_p"]
                    n = int(row.iloc[0]["n_subjects"])
                    stats_text.append(f"n={n}\np={p:.3f}" if np.isfinite(p) else f"n={n}")
            _draw_violin(ax, arrays, meta["ylabel"], stats_text)
            letter = chr(97 + row_index * 3 + col_index)
            ax.set_title(f"{letter}  {meta['title']}")
            if col_index == 0:
                ax.text(
                    -0.28,
                    0.5,
                    group_label,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )
    fig.suptitle("IMF4 within-cycle waveform control-point changes", y=1.02)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create manuscript-ready waveform control-point violin figure.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "primary_bipolar_mne_config.yaml")
    parser.add_argument("--analysis-set", default="primary_28s_guarded")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["project"]["output_dir"]) / args.analysis_set
    waveform = pd.read_csv(root / "tables" / "waveform_nodes.csv")
    subject = waveform_subject_values(waveform)
    stats = summarize_stats(subject)
    fig_dir = root / "figures" / "corrected_addons"
    support_dir = root / "tables" / "figure_support"
    fig_dir.mkdir(parents=True, exist_ok=True)
    support_dir.mkdir(parents=True, exist_ok=True)
    subject_path = support_dir / "figure2_control_point_subject_values.csv"
    stats_path = support_dir / "figure2_control_point_stats.csv"
    fig_path = fig_dir / "figure_02c_control_point_violins.png"
    subject.to_csv(subject_path, index=False)
    stats.to_csv(stats_path, index=False)
    plot_control_point_figure(subject, stats, fig_path)
    print(f"subject_values: {subject_path}")
    print(f"stats: {stats_path}")
    print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
