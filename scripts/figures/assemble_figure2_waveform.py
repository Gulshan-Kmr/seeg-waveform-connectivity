from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seeg_waveform.quinn_figures import set_publication_style


def add_image_panel(ax: plt.Axes, path: Path, label: str, title: str | None = None) -> None:
    image = mpimg.imread(path)
    ax.imshow(image)
    ax.axis("off")
    ax.text(
        0.005,
        0.995,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )
    if title:
        ax.text(
            0.5,
            1.015,
            title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
        )


def assemble(root: Path) -> Path:
    fig_dir = root / "figures" / "corrected_addons"
    out_path = root / "figures" / "figure_02_within_cycle_waveform_dynamics.png"
    paths = {
        "a": fig_dir / "figure_02a_within_cycle_waveform_schematic.png",
        "b": fig_dir / "figure_02a_quinn_population_averages.png",
        "c": fig_dir / "figure_02c_control_point_violins.png",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Figure 2 component(s):\n" + "\n".join(missing))
    set_publication_style()
    fig = plt.figure(figsize=(12.5, 10.8), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=[1.25, 1.05, 1.1])
    add_image_panel(fig.add_subplot(grid[0, 0]), paths["a"], "a")
    add_image_panel(fig.add_subplot(grid[1, 0]), paths["b"], "b")
    add_image_panel(fig.add_subplot(grid[2, 0]), paths["c"], "c")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def assemble_motifs(root: Path) -> Path:
    fig_dir = root / "figures" / "corrected_addons"
    out_path = root / "figures" / "figure_02_companion_pca_waveform_motifs.png"
    paths = {
        "a": fig_dir / "imf4_pca_waveform_motif_components.png",
        "b": fig_dir / "imf4_pca_waveform_motif_score_violins.png",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing PCA motif component(s):\n" + "\n".join(missing))
    set_publication_style()
    fig = plt.figure(figsize=(12.5, 10.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[0.85, 1.55])
    add_image_panel(fig.add_subplot(grid[0, 0]), paths["a"], "a")
    add_image_panel(fig.add_subplot(grid[1, 0]), paths["b"], "b")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble manuscript Figure 2 from waveform component panels.")
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=ROOT / "outputs" / "publication_bipolar_mne" / "primary_28s_guarded",
    )
    args = parser.parse_args()
    print(assemble(args.analysis_root))
    print(assemble_motifs(args.analysis_root))


if __name__ == "__main__":
    main()
