from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_top_changes(master_csv: str | Path, out_dir: str | Path, top_n: int = 30, dpi: int = 300) -> None:
    df = pd.read_csv(master_csv)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for metric in ["pa_if_change_score", "norm_waveform_change_score", "delta_asc2desc", "delta_peak2trough"]:
        if metric not in df:
            continue
        sub = df[df["status"].eq("ok")].dropna(subset=[metric]).copy()
        if sub.empty:
            continue
        sub["abs_metric"] = sub[metric].abs()
        sub = sub.sort_values("abs_metric", ascending=False).head(top_n)
        labels = sub["file"].astype(str) + " | " + sub["channel"].astype(str)

        fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(sub))))
        ax.barh(labels, sub[metric])
        ax.axvline(0, color="0.2", linewidth=0.8)
        ax.invert_yaxis()
        ax.set_xlabel(metric)
        ax.set_title(f"Top {len(sub)} channels by {metric}")
        fig.tight_layout()
        fig.savefig(out / f"top_{metric}.png", dpi=dpi)
        plt.close(fig)


def plot_group_summary(master_csv: str | Path, out_dir: str | Path, dpi: int = 300) -> None:
    df = pd.read_csv(master_csv)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = ["delta_mean_if", "delta_asc2desc", "delta_peak2trough"]
    for metric in metrics:
        if metric not in df:
            continue
        sub = df[df["status"].eq("ok")].dropna(subset=[metric])
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        sub.boxplot(column=metric, by="stim_frequency_hz", ax=ax)
        ax.axhline(0, color="0.2", linewidth=0.8)
        ax.set_title(metric)
        ax.set_xlabel("Stimulation frequency (Hz)")
        ax.set_ylabel("Post - pre")
        fig.suptitle("")
        fig.tight_layout()
        fig.savefig(out / f"{metric}_by_frequency.png", dpi=dpi)
        plt.close(fig)
