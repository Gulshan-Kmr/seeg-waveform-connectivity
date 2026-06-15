from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seeg_waveform.quinn_figures import set_publication_style


ANALYSES = {
    "Primary adjacent bipolar": ROOT / "outputs" / "publication_bipolar_mne" / "primary_28s_guarded",
    "CAR sensitivity": ROOT / "outputs" / "publication_car_reference" / "primary_28s_guarded",
}
COLORS = {1.0: "#2A9D8F", 50.0: "#D62828"}


def _panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")


def _load_subject_table(root: Path) -> pd.DataFrame:
    table = pd.read_csv(root / "tables" / "waveform_connectivity_subject_protocol_summary.csv")
    return table[["subject", "stim_frequency_hz", "delta_mean_if", "delta_abs_imcoh"]].dropna()


def _spearman_text(x: pd.Series, y: pd.Series) -> str:
    clean = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 3 or clean["x"].nunique() < 2 or clean["y"].nunique() < 2:
        return "rho = NA"
    rho, p = stats.spearmanr(clean["x"], clean["y"])
    return f"rho = {rho:.2f}, p = {p:.3f}, n = {len(clean)}"


def _fit_line(ax: plt.Axes, x: pd.Series, y: pd.Series, color: str, alpha: float = 1.0) -> None:
    clean = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 3 or clean["x"].nunique() < 2:
        return
    fit = stats.linregress(clean["x"], clean["y"])
    xs = np.linspace(float(clean["x"].min()), float(clean["x"].max()), 100)
    ax.plot(xs, fit.intercept + fit.slope * xs, color=color, lw=1.8, alpha=alpha)


def _draw_measured(ax: plt.Axes, table: pd.DataFrame, title: str) -> None:
    for freq in (1.0, 50.0):
        sub = table.loc[table["stim_frequency_hz"].eq(freq)]
        ax.scatter(
            sub["delta_mean_if"],
            sub["delta_abs_imcoh"],
            s=60,
            marker="D",
            color=COLORS[freq],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.85,
            label=f"{freq:g} Hz: {_spearman_text(sub['delta_mean_if'], sub['delta_abs_imcoh'])}",
        )
        _fit_line(ax, sub["delta_mean_if"], sub["delta_abs_imcoh"], COLORS[freq])
    ax.axhline(0, color="0.55", lw=0.8, ls="--")
    ax.axvline(0, color="0.55", lw=0.8, ls="--")
    ax.set_xlabel("Within-subject post - pre IMF4 mean IF (Hz)")
    ax.set_ylabel("Within-subject post - pre seed-to-target absolute ImCoh")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7)


def _draw_rank(ax: plt.Axes, table: pd.DataFrame, title: str) -> None:
    for freq in (1.0, 50.0):
        sub = table.loc[table["stim_frequency_hz"].eq(freq)].copy()
        sub["rank_waveform"] = sub["delta_mean_if"].rank(method="average")
        sub["rank_imcoh"] = sub["delta_abs_imcoh"].rank(method="average")
        ax.scatter(
            sub["rank_waveform"],
            sub["rank_imcoh"],
            s=60,
            marker="D",
            color=COLORS[freq],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.85,
            label=f"{freq:g} Hz rank relationship",
        )
        _fit_line(ax, sub["rank_waveform"], sub["rank_imcoh"], COLORS[freq])
    ax.set_xlabel("Within-protocol rank of waveform change")
    ax.set_ylabel("Within-protocol rank of ImCoh change")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7)


def main() -> None:
    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), constrained_layout=True)
    summary_rows: list[dict] = []
    for row, (label, root) in enumerate(ANALYSES.items()):
        table = _load_subject_table(root)
        _draw_measured(axes[row, 0], table, f"{label}: measured subject/protocol changes")
        _draw_rank(axes[row, 1], table, f"{label}: Spearman rank view")
        for freq in (1.0, 50.0):
            sub = table.loc[table["stim_frequency_hz"].eq(freq)]
            clean = sub[["delta_mean_if", "delta_abs_imcoh"]].dropna()
            if len(clean) >= 3 and clean["delta_mean_if"].nunique() > 1 and clean["delta_abs_imcoh"].nunique() > 1:
                rho, p_value = stats.spearmanr(clean["delta_mean_if"], clean["delta_abs_imcoh"])
            else:
                rho, p_value = np.nan, np.nan
            summary_rows.append(
                {
                    "analysis_reference": label,
                    "stim_frequency_hz": freq,
                    "n_subject_protocol_points": int(len(clean)),
                    "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                    "spearman_p_descriptive": float(p_value) if np.isfinite(p_value) else np.nan,
                }
            )
    for ax, letter in zip(axes.ravel(), "abcd"):
        _panel(ax, letter)
    fig.suptitle("Within-subject relationship between IMF4 waveform and seed-to-target imaginary coherence", y=1.02)

    out_root = ROOT / "outputs" / "publication_comparison" / "within_subject_analysis"
    fig_dir = out_root / "figures"
    table_dir = out_root / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    png = fig_dir / "figure_07_within_subject_waveform_imcoh_spearman_rank.png"
    pdf = fig_dir / "figure_07_within_subject_waveform_imcoh_spearman_rank.pdf"
    csv = table_dir / "within_subject_waveform_imcoh_spearman_summary.csv"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(summary_rows).to_csv(csv, index=False)
    print(f"png: {png}")
    print(f"pdf: {pdf}")
    print(f"summary: {csv}")


if __name__ == "__main__":
    main()
