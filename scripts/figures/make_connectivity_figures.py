from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import pandas as pd

from seeg_waveform.config import load_config
from seeg_waveform.quinn_figures import set_publication_style


def main() -> None:
    cfg = load_config(ROOT / "configs" / "connectivity_config.yaml")
    out_root = Path(cfg["project"]["output_dir"])
    stats_path = out_root / "stim_seed_to_nonstim_target_stats.csv"
    stats = pd.read_csv(stats_path)
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    set_publication_style()
    bands = list(cfg["connectivity"]["frequency_bands"].keys())
    labels = [b.replace("_", " ") for b in bands]

    group_col = "group" if "group" in stats.columns else None

    for level in ["pair", "file_mean"]:
        sub = stats[stats["level"].eq(level)].copy()
        if group_col:
            sub = sub[sub[group_col].eq("all")]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
        for ax, method, title in [
            (axes[0], "coh", "Coherence"),
            (axes[1], "imcoh", "Imaginary coherence"),
        ]:
            vals = []
            stars = []
            for band in bands:
                row = sub[sub["metric"].eq(f"{method}_{band}")]
                vals.append(float(row["mean_delta"].iloc[0]) if not row.empty else 0.0)
                p = float(row["wilcoxon_p_fdr"].iloc[0]) if not row.empty else 1.0
                stars.append("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")
            bars = ax.bar(labels, vals, color="#4C78A8" if method == "coh" else "#F58518")
            ax.axhline(0, color="0.2", linewidth=0.8)
            ax.set_title(title)
            ax.set_ylabel("Post - pre")
            ax.tick_params(axis="x", rotation=35)
            for bar, star in zip(bars, stars):
                if star:
                    y = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, y, star, ha="center", va="bottom" if y >= 0 else "top")
        out = fig_dir / f"stim_seed_to_nonstim_{level}_connectivity_delta.png"
        fig.savefig(out, dpi=cfg["outputs"].get("figure_dpi", 300))
        plt.close(fig)
        print(out)

    if group_col:
        for level in ["pair", "file_mean"]:
            sub = stats[(stats["level"].eq(level)) & (stats[group_col].isin(["1Hz", "50Hz"]))].copy()
            if sub.empty:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)
            x = range(len(labels))
            width = 0.38
            for ax, method, title in [
                (axes[0], "coh", "Coherence"),
                (axes[1], "imcoh", "Imaginary coherence"),
            ]:
                for offset, group, color in [(-width / 2, "1Hz", "#54A24B"), (width / 2, "50Hz", "#B279A2")]:
                    vals = []
                    stars = []
                    for band in bands:
                        row = sub[(sub[group_col].eq(group)) & (sub["metric"].eq(f"{method}_{band}"))]
                        vals.append(float(row["mean_delta"].iloc[0]) if not row.empty else 0.0)
                        p = float(row["wilcoxon_p_fdr"].iloc[0]) if not row.empty else 1.0
                        stars.append("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")
                    bars = ax.bar([i + offset for i in x], vals, width=width, color=color, label=group)
                    for bar, star in zip(bars, stars):
                        if star:
                            y = bar.get_height()
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,
                                y,
                                star,
                                ha="center",
                                va="bottom" if y >= 0 else "top",
                                fontsize=8,
                            )
                ax.axhline(0, color="0.2", linewidth=0.8)
                ax.set_title(title)
                ax.set_ylabel("Post - pre")
                ax.set_xticks(list(x))
                ax.set_xticklabels(labels, rotation=35, ha="right")
                ax.legend(frameon=False)
            out = fig_dir / f"stim_seed_to_nonstim_{level}_connectivity_delta_by_frequency.png"
            fig.savefig(out, dpi=cfg["outputs"].get("figure_dpi", 300))
            plt.close(fig)
            print(out)


if __name__ == "__main__":
    main()
