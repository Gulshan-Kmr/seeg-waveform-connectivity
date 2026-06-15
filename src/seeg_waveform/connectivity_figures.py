from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config
from .quinn_figures import set_publication_style


def _p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _eligible_pairs(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"].eq("ok")].copy()
    return ok[~ok["pre_noise_pair"].fillna(False) & ~ok["post_noise_pair"].fillna(False)]


def _file_means(df: pd.DataFrame, bands: list[str], method: str) -> pd.DataFrame:
    cols = []
    for band in bands:
        cols.extend([f"pre_{method}_{band}", f"post_{method}_{band}", f"delta_{method}_{band}"])
    return df.groupby(["file", "stim_frequency_hz"], as_index=False)[cols].mean()


def plot_file_level_paired(df: pd.DataFrame, stats_df: pd.DataFrame, cfg: dict, out_dir: Path) -> list[Path]:
    bands = list(cfg["connectivity"]["frequency_bands"].keys())
    labels = [b.replace("_", "\n") for b in bands]
    paths = []
    set_publication_style()
    for method, title, color in [("coh", "Coherence", "#4C78A8"), ("imcoh", "Imaginary coherence", "#F58518")]:
        fm = _file_means(df, bands, method)
        fig, axes = plt.subplots(1, len(bands), figsize=(10.5, 3.2), sharey=True, constrained_layout=True)
        for ax, band, label in zip(axes, bands, labels):
            pre = fm[f"pre_{method}_{band}"].to_numpy(float)
            post = fm[f"post_{method}_{band}"].to_numpy(float)
            for a, b in zip(pre, post):
                ax.plot([0, 1], [a, b], color="0.70", linewidth=0.8, zorder=1)
            ax.scatter(np.zeros_like(pre), pre, color="0.35", s=16, alpha=0.85, zorder=2)
            ax.scatter(np.ones_like(post), post, color=color, s=16, alpha=0.85, zorder=2)
            row = stats_df[
                stats_df["level"].eq("file_mean") & stats_df["metric"].eq(f"{method}_{band}")
            ]
            star = _p_to_stars(float(row["wilcoxon_p_fdr"].iloc[0])) if not row.empty else ""
            if star:
                ymin, ymax = ax.get_ylim()
                ax.text(0.5, ymax, star, ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax.set_title(label)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Pre", "Post"])
            ax.tick_params(axis="x", rotation=0)
        axes[0].set_ylabel(title)
        fig.suptitle(f"File-level stimulated-to-nonstimulated {title.lower()} | n={len(fm)} files")
        path = out_dir / f"file_level_paired_{method}.png"
        fig.savefig(path, dpi=cfg["outputs"].get("figure_dpi", 300), bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_pair_level_distributions(df: pd.DataFrame, stats_df: pd.DataFrame, cfg: dict, out_dir: Path) -> list[Path]:
    bands = list(cfg["connectivity"]["frequency_bands"].keys())
    labels = [b.replace("_", "\n") for b in bands]
    paths = []
    set_publication_style()
    for method, title, color in [("coh", "Coherence", "#4C78A8"), ("imcoh", "Imaginary coherence", "#F58518")]:
        data = [df[f"delta_{method}_{band}"].dropna().to_numpy(float) for band in bands]
        fig, ax = plt.subplots(figsize=(7.4, 3.5), constrained_layout=True)
        parts = ax.violinplot(data, showmedians=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.62)
            body.set_edgecolor("0.25")
        if "cmedians" in parts:
            parts["cmedians"].set_color("0.05")
            parts["cmedians"].set_linewidth(1.2)
        ax.axhline(0, color="0.25", linestyle="--", linewidth=0.8)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Post - pre")
        ax.set_title(f"Pair-level delta distributions | {title}")
        ymin, ymax = ax.get_ylim()
        for xpos, band in enumerate(bands, start=1):
            row = stats_df[stats_df["level"].eq("pair") & stats_df["metric"].eq(f"{method}_{band}")]
            star = _p_to_stars(float(row["wilcoxon_p_fdr"].iloc[0])) if not row.empty else ""
            if star:
                ax.text(xpos, ymax, star, ha="center", va="bottom", fontsize=9, fontweight="bold")
        path = out_dir / f"pair_level_delta_violin_{method}.png"
        fig.savefig(path, dpi=cfg["outputs"].get("figure_dpi", 300), bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_file_delta_heatmap(df: pd.DataFrame, cfg: dict, out_dir: Path) -> list[Path]:
    bands = list(cfg["connectivity"]["frequency_bands"].keys())
    labels = [b.replace("_", " ") for b in bands]
    paths = []
    set_publication_style()
    for method, title in [("coh", "Coherence"), ("imcoh", "Imaginary coherence")]:
        fm = _file_means(df, bands, method).sort_values(["stim_frequency_hz", "file"])
        mat = fm[[f"delta_{method}_{band}" for band in bands]].to_numpy(float)
        max_abs = np.nanpercentile(np.abs(mat), 95)
        max_abs = max(max_abs, np.finfo(float).eps)
        fig, ax = plt.subplots(figsize=(7.5, max(4.2, 0.22 * len(fm))), constrained_layout=True)
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ylabels = [f"{row.file} ({row.stim_frequency_hz:g} Hz)" for row in fm.itertuples()]
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=5)
        ax.set_title(f"File-level post-pre delta heatmap | {title}")
        fig.colorbar(im, ax=ax, label="Post - pre")
        path = out_dir / f"file_level_delta_heatmap_{method}.png"
        fig.savefig(path, dpi=cfg["outputs"].get("figure_dpi", 300), bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def make_publication_connectivity_figures(config_path: str | Path) -> list[Path]:
    cfg = load_config(config_path)
    out_root = Path(cfg["project"]["output_dir"])
    df = pd.read_csv(out_root / "stim_seed_to_nonstim_target_connectivity.csv")
    stats_df = pd.read_csv(out_root / "stim_seed_to_nonstim_target_stats.csv")
    df = _eligible_pairs(df)
    out_dir = out_root / "figures" / "publication"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    paths.extend(plot_file_level_paired(df, stats_df, cfg, out_dir))
    paths.extend(plot_pair_level_distributions(df, stats_df, cfg, out_dir))
    paths.extend(plot_file_delta_heatmap(df, cfg, out_dir))
    return paths
