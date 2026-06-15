from __future__ import annotations

from math import ceil
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seeg_waveform.quinn_figures import set_publication_style


ANALYSES = {
    "Adjacent bipolar": ROOT / "outputs" / "publication_bipolar_mne" / "primary_28s_guarded",
    "Common-average reference": ROOT / "outputs" / "publication_car_reference" / "primary_28s_guarded",
}
PROTOCOL_COLORS = {1.0: "#2A9D8F", 50.0: "#D62828"}
SUBJECT_COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#D55E00",
    "#56B4E9",
    "#F0E442",
    "#6A3D9A",
]


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().eq("true")


def _load_electrode_table(root: Path, reference: str) -> pd.DataFrame:
    """Join each target electrode's IMF4 waveform change to its seed-edge change."""
    table_dir = root / "tables"
    waveform = pd.read_csv(table_dir / "waveform_nodes.csv", low_memory=False)
    edges = pd.read_csv(table_dir / "connectivity_edges.csv", low_memory=False)

    waveform = waveform.loc[
        _as_bool(waveform["analysis_include"]),
        [
            "file",
            "subject",
            "stim_frequency_hz",
            "channel",
            "region_family",
            "delta_mean_if",
        ],
    ].rename(columns={"channel": "target_channel"})

    edges = edges.loc[
        _as_bool(edges["analysis_include"])
        & edges["edge_type"].eq("seed_to_target")
        & edges["imf"].eq(4),
        [
            "file",
            "subject",
            "stim_frequency_hz",
            "seed_channel",
            "target_channel",
            "node_b_region_family",
            "delta_abs_imcoh",
        ],
    ]

    joined = edges.merge(
        waveform,
        on=["file", "subject", "stim_frequency_hz", "target_channel"],
        how="inner",
        validate="many_to_one",
    )
    joined["region_family"] = joined["region_family"].fillna(joined["node_b_region_family"])
    joined["analysis_reference"] = reference

    # Repeated runs are not treated as independent electrode observations.
    electrode = (
        joined.groupby(
            [
                "analysis_reference",
                "subject",
                "stim_frequency_hz",
                "target_channel",
                "region_family",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            delta_mean_if=("delta_mean_if", "mean"),
            delta_abs_imcoh=("delta_abs_imcoh", "mean"),
            n_runs=("file", "nunique"),
            seed_channels=("seed_channel", lambda values: ";".join(sorted(set(map(str, values))))),
        )
    )
    return electrode.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["delta_mean_if", "delta_abs_imcoh"]
    )


def _correlations(electrode: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, group in electrode.groupby(
        ["analysis_reference", "subject", "stim_frequency_hz"], dropna=False
    ):
        valid = (
            len(group) >= 4
            and group["delta_mean_if"].nunique() >= 2
            and group["delta_abs_imcoh"].nunique() >= 2
        )
        rho, p_value = stats.spearmanr(
            group["delta_mean_if"], group["delta_abs_imcoh"]
        ) if valid else (np.nan, np.nan)
        rows.append(
            {
                "analysis_reference": keys[0],
                "subject": keys[1],
                "stim_frequency_hz": keys[2],
                "n_electrodes": len(group),
                "spearman_rho": rho,
                "p_uncorrected_descriptive": p_value,
            }
        )

    result = pd.DataFrame(rows)
    result["p_fdr_within_reference"] = np.nan
    for reference, index in result.groupby("analysis_reference").groups.items():
        finite = result.loc[index, "p_uncorrected_descriptive"].notna()
        selected = result.loc[index].index[finite]
        if len(selected):
            result.loc[selected, "p_fdr_within_reference"] = multipletests(
                result.loc[selected, "p_uncorrected_descriptive"], method="fdr_bh"
            )[1]
    return result


def _fit_visual_line(ax: plt.Axes, group: pd.DataFrame, color: str) -> None:
    """Draw a visual linear trend; inference remains the reported Spearman test."""
    if len(group) < 4 or group["delta_mean_if"].nunique() < 2:
        return
    fit = stats.linregress(group["delta_mean_if"], group["delta_abs_imcoh"])
    x_values = np.linspace(group["delta_mean_if"].min(), group["delta_mean_if"].max(), 100)
    ax.plot(x_values, fit.intercept + fit.slope * x_values, color=color, lw=1.6, alpha=0.9)


def _draw_reference(
    electrode: pd.DataFrame, correlations: pd.DataFrame, reference: str, output_dir: Path
) -> tuple[Path, Path]:
    data = electrode.loc[electrode["analysis_reference"].eq(reference)]
    panels = list(
        data[["subject", "stim_frequency_hz"]]
        .drop_duplicates()
        .sort_values(["subject", "stim_frequency_hz"])
        .itertuples(index=False, name=None)
    )
    n_cols = 4
    n_rows = ceil(len(panels) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(14.2, max(5.8, n_rows * 2.65)),
        constrained_layout=True,
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    panel = 0
    for subject, protocol in panels:
        ax = axes.ravel()[panel]
        group = data.loc[
            data["subject"].eq(subject)
            & data["stim_frequency_hz"].eq(protocol)
        ]
        stat_row = correlations.loc[
            correlations["analysis_reference"].eq(reference)
            & correlations["subject"].eq(subject)
            & correlations["stim_frequency_hz"].eq(protocol)
        ]
        color = PROTOCOL_COLORS[protocol]
        ax.scatter(
            group["delta_mean_if"],
            group["delta_abs_imcoh"],
            s=27,
            facecolor=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.78,
        )
        _fit_visual_line(ax, group, color)
        ax.axhline(0, color="0.65", lw=0.7, ls="--")
        ax.axvline(0, color="0.65", lw=0.7, ls="--")
        row = stat_row.iloc[0]
        rho = row["spearman_rho"]
        p_value = row["p_uncorrected_descriptive"]
        q_value = row["p_fdr_within_reference"]
        annotation = (
            f"rho = {rho:.2f}\np = {p_value:.3g}, q = {q_value:.3g}\n"
            f"n = {int(row['n_electrodes'])} electrodes"
            if np.isfinite(rho)
            else f"rho = NA\nn = {len(group)} electrodes"
        )
        ax.text(
            0.03,
            0.97,
            annotation,
            transform=ax.transAxes,
            va="top",
            fontsize=7.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76},
        )
        ax.set_title(f"{subject} | {protocol:g} Hz", fontsize=9)
        ax.set_xlabel("Target-electrode IMF4 mean IF change (Hz)", fontsize=7.5)
        ax.set_ylabel("Seed-to-target |ImCoh| change", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.text(
            -0.15,
            1.04,
            chr(ord("a") + panel),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=10,
        )
        panel += 1

    for ax in axes.ravel()[panel:]:
        ax.set_axis_off()

    fig.suptitle(
        f"{reference}: within-subject electrode-level waveform-connectivity relationships",
        fontsize=13,
    )
    stem = reference.lower().replace("-", "_").replace(" ", "_")
    png = output_dir / f"figure_09_{stem}_electrode_waveform_imcoh_spearman.png"
    pdf = output_dir / f"figure_09_{stem}_electrode_waveform_imcoh_spearman.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def _pooled_correlations(electrode: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, group in electrode.groupby(
        ["analysis_reference", "stim_frequency_hz"], dropna=False
    ):
        rho, p_value = stats.spearmanr(
            group["delta_mean_if"], group["delta_abs_imcoh"]
        )
        subject_rhos = []
        for _, subject_group in group.groupby("subject"):
            if (
                len(subject_group) >= 4
                and subject_group["delta_mean_if"].nunique() >= 2
                and subject_group["delta_abs_imcoh"].nunique() >= 2
            ):
                subject_rhos.append(
                    stats.spearmanr(
                        subject_group["delta_mean_if"],
                        subject_group["delta_abs_imcoh"],
                    ).statistic
                )
        rows.append(
            {
                "analysis_reference": keys[0],
                "stim_frequency_hz": keys[1],
                "n_electrodes": len(group),
                "n_subjects": group["subject"].nunique(),
                "pooled_spearman_rho_descriptive": rho,
                "pooled_p_uncorrected_descriptive": p_value,
                "median_within_subject_rho": (
                    float(np.median(subject_rhos)) if subject_rhos else np.nan
                ),
                "n_subjects_with_estimable_rho": len(subject_rhos),
            }
        )
    result = pd.DataFrame(rows)
    result["pooled_p_fdr_all_panels"] = multipletests(
        result["pooled_p_uncorrected_descriptive"], method="fdr_bh"
    )[1]
    return result


def _draw_pooled_reference(
    electrode: pd.DataFrame,
    pooled_stats: pd.DataFrame,
    reference: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    data = electrode.loc[electrode["analysis_reference"].eq(reference)]
    subjects = sorted(data["subject"].unique())
    subject_colors = {
        subject: SUBJECT_COLORS[index % len(SUBJECT_COLORS)]
        for index, subject in enumerate(subjects)
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.6))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.28, top=0.84, wspace=0.20)

    for panel, protocol in enumerate((1.0, 50.0)):
        ax = axes[panel]
        group = data.loc[data["stim_frequency_hz"].eq(protocol)]
        for subject, subject_group in group.groupby("subject"):
            ax.scatter(
                subject_group["delta_mean_if"],
                subject_group["delta_abs_imcoh"],
                s=23,
                facecolor=subject_colors[subject],
                edgecolor="white",
                linewidth=0.3,
                alpha=0.72,
                label=subject,
            )

        # The line is a visual least-squares summary; the annotation reports Spearman rho.
        _fit_visual_line(ax, group, PROTOCOL_COLORS[protocol])
        ax.axhline(0, color="0.65", lw=0.7, ls="--")
        ax.axvline(0, color="0.65", lw=0.7, ls="--")
        row = pooled_stats.loc[
            pooled_stats["analysis_reference"].eq(reference)
            & pooled_stats["stim_frequency_hz"].eq(protocol)
        ].iloc[0]
        ax.text(
            0.03,
            0.97,
            (
                f"Pooled descriptive rho = {row['pooled_spearman_rho_descriptive']:.2f}\n"
                f"p = {row['pooled_p_uncorrected_descriptive']:.3g}, "
                f"q = {row['pooled_p_fdr_all_panels']:.3g}\n"
                f"{int(row['n_electrodes'])} electrodes, "
                f"{int(row['n_subjects'])} participants\n"
                f"Median participant rho = {row['median_within_subject_rho']:.2f}"
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
        ax.set_title(f"{protocol:g} Hz stimulation")
        ax.set_xlabel("Target-electrode IMF4 mean IF change (Hz)")
        ax.set_ylabel("Seed-to-target |ImCoh| change")
        ax.text(
            -0.12,
            1.04,
            chr(ord("a") + panel),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=11,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=min(4, len(by_label)),
        frameon=False,
        fontsize=8,
        title="Participant",
    )
    fig.suptitle(
        f"{reference}: pooled electrode-level waveform-connectivity relationships",
        fontsize=13,
    )
    stem = reference.lower().replace("-", "_").replace(" ", "_")
    png = output_dir / f"figure_10_{stem}_pooled_electrode_waveform_imcoh.png"
    pdf = output_dir / f"figure_10_{stem}_pooled_electrode_waveform_imcoh.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    set_publication_style()
    out_root = ROOT / "outputs" / "publication_comparison" / "within_subject_analysis"
    figure_dir = out_root / "figures"
    table_dir = out_root / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    electrode = pd.concat(
        [_load_electrode_table(root, label) for label, root in ANALYSES.items()],
        ignore_index=True,
    )
    correlations = _correlations(electrode)
    pooled_stats = _pooled_correlations(electrode)

    electrode_csv = table_dir / "within_subject_electrode_waveform_imcoh_data.csv"
    stats_csv = table_dir / "within_subject_electrode_waveform_imcoh_spearman_stats.csv"
    pooled_stats_csv = table_dir / "pooled_electrode_waveform_imcoh_spearman_stats.csv"
    electrode.to_csv(electrode_csv, index=False)
    correlations.to_csv(stats_csv, index=False)
    pooled_stats.to_csv(pooled_stats_csv, index=False)

    figure_paths: list[Path] = []
    for reference in ANALYSES:
        figure_paths.extend(_draw_reference(electrode, correlations, reference, figure_dir))
        figure_paths.extend(
            _draw_pooled_reference(electrode, pooled_stats, reference, figure_dir)
        )

    print(f"Electrode observations: {len(electrode)}")
    print(f"Subject-protocol correlations: {len(correlations)}")
    print(f"Electrode data: {electrode_csv}")
    print(f"Statistics: {stats_csv}")
    print(f"Pooled statistics: {pooled_stats_csv}")
    for path in figure_paths:
        print(f"Figure: {path}")


if __name__ == "__main__":
    main()
