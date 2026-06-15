from __future__ import annotations

from itertools import product
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
CONNECTIVITY_METRICS = {
    "post_abs_imcoh": "Post-stimulation absolute imaginary coherence",
    "post_imcoh_signed": "Post-stimulation signed imaginary coherence",
    "post_coh": "Post-stimulation ordinary coherence",
}
PROTOCOL_COLORS = {1.0: "#2A9D8F", 50.0: "#D62828"}
N_PERMUTATIONS = 20_000
N_BOOTSTRAPS = 20_000
RANDOM_SEED = 20260611


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().eq("true")


def _load_pairs(reference: str, root: Path) -> pd.DataFrame:
    waveform = pd.read_csv(root / "tables" / "waveform_nodes.csv", low_memory=False)
    edges = pd.read_csv(root / "tables" / "connectivity_edges.csv", low_memory=False)
    waveform_columns = [
        "file",
        "subject",
        "stim_frequency_hz",
        "channel",
        "region_family",
        "post_mean_if",
        "post_asc2desc",
        "post_n_cycles",
        "post_status",
        "post_noise_flag",
        "primary_region_include",
    ]
    waveform = waveform[waveform_columns]
    edges = edges.loc[
        _as_bool(edges["analysis_include"])
        & edges["edge_type"].eq("seed_to_target")
        & edges["imf"].eq(4),
        [
            "file",
            "subject",
            "stim_frequency_hz",
            "stim_amplitude_ma",
            "seed_channel",
            "target_channel",
            "node_b_region_family",
            "distance_mm",
            "post_abs_imcoh",
            "post_imcoh_signed",
            "post_coh",
        ],
    ]

    post_clean = waveform["post_status"].eq("ok") & ~_as_bool(
        waveform["post_noise_flag"]
    )
    seed = waveform.loc[post_clean].rename(
        columns={
            "channel": "seed_channel",
            "region_family": "seed_region_family",
            "post_mean_if": "seed_post_mean_if_hz",
            "post_asc2desc": "seed_post_ascent_fraction",
            "post_n_cycles": "seed_post_n_cycles",
        }
    )
    target = waveform.loc[
        post_clean & _as_bool(waveform["primary_region_include"])
    ].rename(
        columns={
            "channel": "target_channel",
            "region_family": "target_region_family",
            "post_mean_if": "target_post_mean_if_hz",
            "post_asc2desc": "target_post_ascent_fraction",
            "post_n_cycles": "target_post_n_cycles",
        }
    )
    pairs = edges.merge(
        seed,
        on=["file", "subject", "stim_frequency_hz", "seed_channel"],
        how="inner",
        validate="many_to_one",
    ).merge(
        target,
        on=["file", "subject", "stim_frequency_hz", "target_channel"],
        how="inner",
        validate="many_to_one",
    )
    pairs["target_region_family"] = pairs["target_region_family"].fillna(
        pairs["node_b_region_family"]
    )
    pairs["seed_post_rise_time_ms"] = (
        1000.0
        * pairs["seed_post_ascent_fraction"]
        / pairs["seed_post_mean_if_hz"]
    )
    pairs["target_post_rise_time_ms"] = (
        1000.0
        * pairs["target_post_ascent_fraction"]
        / pairs["target_post_mean_if_hz"]
    )
    pairs["post_rise_time_ratio_seed_over_target"] = (
        pairs["seed_post_rise_time_ms"] / pairs["target_post_rise_time_ms"]
    )
    pairs["post_log_rise_time_ratio"] = np.log(
        pairs["post_rise_time_ratio_seed_over_target"]
    )
    pairs["post_absolute_log_rise_time_ratio"] = pairs[
        "post_log_rise_time_ratio"
    ].abs()
    pairs["analysis_reference"] = reference
    numeric = [
        "seed_post_rise_time_ms",
        "target_post_rise_time_ms",
        "post_rise_time_ratio_seed_over_target",
        "post_log_rise_time_ratio",
        "post_absolute_log_rise_time_ratio",
        *CONNECTIVITY_METRICS,
    ]
    pairs[numeric] = pairs[numeric].replace([np.inf, -np.inf], np.nan)
    return pairs.dropna(subset=numeric)


def _spearman_rows(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    x_metrics = [
        "post_rise_time_ratio_seed_over_target",
        "post_log_rise_time_ratio",
        "post_absolute_log_rise_time_ratio",
        "target_post_rise_time_ms",
    ]
    for keys, group in pairs.groupby(
        ["analysis_reference", "file", "subject", "stim_frequency_hz"],
        dropna=False,
    ):
        for x_metric, y_metric in product(x_metrics, CONNECTIVITY_METRICS):
            clean = group[[x_metric, y_metric]].dropna()
            if (
                len(clean) >= 4
                and clean[x_metric].nunique() >= 2
                and clean[y_metric].nunique() >= 2
            ):
                rho, p_value = stats.spearmanr(clean[x_metric], clean[y_metric])
            else:
                rho, p_value = np.nan, np.nan
            rows.append(
                {
                    "analysis_reference": keys[0],
                    "file": keys[1],
                    "subject": keys[2],
                    "stim_frequency_hz": keys[3],
                    "x_metric": x_metric,
                    "connectivity_metric": y_metric,
                    "n_seed_target_pairs": len(clean),
                    "spearman_rho": rho,
                    "p_uncorrected_descriptive": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["p_fdr_within_reference"] = np.nan
    for _, index in result.groupby("analysis_reference").groups.items():
        selected = result.loc[index].index[
            result.loc[index, "p_uncorrected_descriptive"].notna()
        ]
        if len(selected):
            result.loc[selected, "p_fdr_within_reference"] = multipletests(
                result.loc[selected, "p_uncorrected_descriptive"], method="fdr_bh"
            )[1]
    return result


def _exact_signflip(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if not len(values):
        return np.nan
    observed = abs(values.mean())
    signs = np.array(list(product([-1.0, 1.0], repeat=len(values))))
    null = np.abs((signs * values).mean(axis=1))
    return float((null >= observed - 1e-15).mean())


def _subject_inference(run_stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject = (
        run_stats.groupby(
            [
                "analysis_reference",
                "subject",
                "stim_frequency_hz",
                "x_metric",
                "connectivity_metric",
            ],
            as_index=False,
        )
        .agg(
            mean_run_spearman_rho=("spearman_rho", "mean"),
            n_runs=("file", "nunique"),
        )
    )
    rows: list[dict] = []
    for keys, group in subject.groupby(
        [
            "analysis_reference",
            "stim_frequency_hz",
            "x_metric",
            "connectivity_metric",
        ],
        dropna=False,
    ):
        values = group["mean_run_spearman_rho"].dropna()
        rows.append(
            {
                "analysis_reference": keys[0],
                "stim_frequency_hz": keys[1],
                "x_metric": keys[2],
                "connectivity_metric": keys[3],
                "n_subjects": len(values),
                "mean_subject_rho": values.mean(),
                "median_subject_rho": values.median(),
                "exact_signflip_p": _exact_signflip(values),
            }
        )
    inference = pd.DataFrame(rows)
    inference["p_fdr_all_tests"] = np.nan
    finite = inference["exact_signflip_p"].notna()
    inference.loc[finite, "p_fdr_all_tests"] = multipletests(
        inference.loc[finite, "exact_signflip_p"], method="fdr_bh"
    )[1]
    return subject, inference


def _standardized_ranks(values: pd.Series) -> np.ndarray:
    ranks = stats.rankdata(pd.to_numeric(values, errors="coerce").to_numpy(float))
    ranks = ranks - ranks.mean()
    norm = np.sqrt(np.sum(ranks**2))
    return ranks / norm if norm > 0 else np.full_like(ranks, np.nan)


def _hierarchical_test_one(
    data: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    rng: np.random.Generator,
) -> dict:
    """Shuffle x within seed/run and aggregate seed -> run -> subject with equal weights."""
    groups: list[dict] = []
    for keys, group in data.groupby(["subject", "file", "seed_channel"], dropna=False):
        clean = group[[x_metric, y_metric]].dropna()
        if (
            len(clean) < 4
            or clean[x_metric].nunique() < 2
            or clean[y_metric].nunique() < 2
        ):
            continue
        x_rank = _standardized_ranks(clean[x_metric])
        y_rank = _standardized_ranks(clean[y_metric])
        if not np.isfinite(x_rank).all() or not np.isfinite(y_rank).all():
            continue
        groups.append(
            {
                "subject": keys[0],
                "file": keys[1],
                "seed_channel": keys[2],
                "x_rank": x_rank,
                "y_rank": y_rank,
                "rho": float(np.dot(x_rank, y_rank)),
                "n_pairs": len(clean),
            }
        )
    if not groups:
        return {}

    group_table = pd.DataFrame(
        [{key: value for key, value in group.items() if key not in {"x_rank", "y_rank"}} for group in groups]
    )
    seed_counts = group_table.groupby(["subject", "file"])["seed_channel"].transform("count")
    run_counts = group_table.groupby("subject")["file"].transform("nunique")
    n_subjects = group_table["subject"].nunique()
    weights = 1.0 / (n_subjects * run_counts.to_numpy(float) * seed_counts.to_numpy(float))
    observed = float(np.dot(weights, group_table["rho"].to_numpy(float)))

    null = np.zeros(N_PERMUTATIONS, dtype=float)
    batch_size = 500
    for group, weight in zip(groups, weights):
        x_rank = group["x_rank"]
        y_rank = group["y_rank"]
        n_values = len(x_rank)
        for start in range(0, N_PERMUTATIONS, batch_size):
            stop = min(start + batch_size, N_PERMUTATIONS)
            random_order = np.argsort(
                rng.random((stop - start, n_values)), axis=1
            )
            null[start:stop] += weight * (x_rank[random_order] @ y_rank)
    p_value = float(
        (1 + np.count_nonzero(np.abs(null) >= abs(observed) - 1e-15))
        / (N_PERMUTATIONS + 1)
    )

    run_values = (
        group_table.groupby(["subject", "file"], as_index=False)["rho"].mean()
    )
    subject_values = run_values.groupby("subject", as_index=False)["rho"].mean()
    subject_rhos = subject_values["rho"].to_numpy(float)
    bootstrap_indices = rng.integers(
        0, len(subject_rhos), size=(N_BOOTSTRAPS, len(subject_rhos))
    )
    bootstrap_means = subject_rhos[bootstrap_indices].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "n_seed_target_pairs": int(group_table["n_pairs"].sum()),
        "n_seeds": int(len(group_table)),
        "n_runs": int(run_values["file"].nunique()),
        "n_subjects": int(len(subject_values)),
        "hierarchical_mean_spearman_rho": observed,
        "participant_bootstrap_ci_low": float(ci_low),
        "participant_bootstrap_ci_high": float(ci_high),
        "hierarchical_permutation_p": p_value,
        "n_permutations": N_PERMUTATIONS,
        "n_bootstraps": N_BOOTSTRAPS,
    }


def _hierarchical_inference(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    rng = np.random.default_rng(RANDOM_SEED)
    for reference, protocol, y_metric in product(
        ANALYSES, (1.0, 50.0), CONNECTIVITY_METRICS
    ):
        data = pairs.loc[
            pairs["analysis_reference"].eq(reference)
            & pairs["stim_frequency_hz"].eq(protocol)
        ]
        result = _hierarchical_test_one(
            data,
            "post_rise_time_ratio_seed_over_target",
            y_metric,
            rng,
        )
        rows.append(
            {
                "analysis_reference": reference,
                "stim_frequency_hz": protocol,
                "x_metric": "post_rise_time_ratio_seed_over_target",
                "connectivity_metric": y_metric,
                **result,
            }
        )
    output = pd.DataFrame(rows)
    output["p_fdr_hierarchical_all_panels"] = multipletests(
        output["hierarchical_permutation_p"], method="fdr_bh"
    )[1]
    return output


def _visual_line(ax: plt.Axes, x: pd.Series, y: pd.Series, color: str) -> None:
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(clean) < 4 or clean["x"].nunique() < 2:
        return
    fit = stats.linregress(clean["x"], clean["y"])
    xx = np.linspace(clean["x"].min(), clean["x"].max(), 100)
    ax.plot(xx, fit.intercept + fit.slope * xx, color=color, lw=1.5)


def _run_panel_figure(
    pairs: pd.DataFrame,
    run_stats: pd.DataFrame,
    reference: str,
    figure_dir: Path,
) -> tuple[Path, Path]:
    data = pairs.loc[pairs["analysis_reference"].eq(reference)]
    panels = list(
        data[["file", "subject", "stim_frequency_hz"]]
        .drop_duplicates()
        .sort_values(["subject", "stim_frequency_hz", "file"])
        .itertuples(index=False, name=None)
    )
    n_cols = 4
    n_rows = ceil(len(panels) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(14.2, n_rows * 2.7), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()
    x_metric = "post_rise_time_ratio_seed_over_target"
    y_metric = "post_abs_imcoh"
    for panel, (file_name, subject, protocol) in enumerate(panels):
        ax = axes[panel]
        group = data.loc[data["file"].eq(file_name)]
        color = PROTOCOL_COLORS[protocol]
        ax.scatter(
            group[x_metric],
            group[y_metric],
            s=27,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.78,
        )
        _visual_line(ax, group[x_metric], group[y_metric], color)
        stat_row = run_stats.loc[
            run_stats["analysis_reference"].eq(reference)
            & run_stats["file"].eq(file_name)
            & run_stats["x_metric"].eq(x_metric)
            & run_stats["connectivity_metric"].eq(y_metric)
        ].iloc[0]
        ax.text(
            0.03,
            0.97,
            (
                f"rho = {stat_row['spearman_rho']:.2f}\n"
                f"p = {stat_row['p_uncorrected_descriptive']:.3g}, "
                f"q = {stat_row['p_fdr_within_reference']:.3g}\n"
                f"n = {int(stat_row['n_seed_target_pairs'])} pairs"
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=7.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )
        ax.set_title(f"{subject} | {protocol:g} Hz", fontsize=9)
        ax.set_xlabel("Post IMF4 rise-time ratio: seed / target", fontsize=7.5)
        ax.set_ylabel("Post |imaginary coherence|", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.text(
            -0.15,
            1.04,
            chr(ord("a") + panel),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=10,
        )
    for ax in axes[len(panels) :]:
        ax.set_axis_off()
    fig.suptitle(
        f"{reference}: post-stimulation rise-time ratio and seed-to-target connectivity",
        fontsize=13,
    )
    stem = reference.lower().replace("-", "_").replace(" ", "_")
    png = figure_dir / f"figure_01_{stem}_run_level_rise_ratio_abs_imcoh.png"
    pdf = figure_dir / f"figure_01_{stem}_run_level_rise_ratio_abs_imcoh.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def _summary_figure(
    pairs: pd.DataFrame,
    hierarchical: pd.DataFrame,
    reference: str,
    figure_dir: Path,
) -> tuple[Path, Path]:
    data = pairs.loc[pairs["analysis_reference"].eq(reference)]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.0), constrained_layout=True)
    metrics = list(CONNECTIVITY_METRICS)
    for row, protocol in enumerate((1.0, 50.0)):
        protocol_data = data.loc[data["stim_frequency_hz"].eq(protocol)]
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            for subject, group in protocol_data.groupby("subject"):
                ax.scatter(
                    group["post_rise_time_ratio_seed_over_target"],
                    group[metric],
                    s=18,
                    alpha=0.58,
                    label=subject,
                )
            rho, p_value = stats.spearmanr(
                protocol_data["post_rise_time_ratio_seed_over_target"],
                protocol_data[metric],
            )
            hierarchy = hierarchical.loc[
                hierarchical["analysis_reference"].eq(reference)
                & hierarchical["stim_frequency_hz"].eq(protocol)
                & hierarchical["connectivity_metric"].eq(metric)
            ].iloc[0]
            _visual_line(
                ax,
                protocol_data["post_rise_time_ratio_seed_over_target"],
                protocol_data[metric],
                PROTOCOL_COLORS[protocol],
            )
            ax.text(
                0.03,
                0.97,
                (
                    f"Descriptive pooled rho = {rho:.2f}, p = {p_value:.3g}\n"
                    f"Hierarchical rho = "
                    f"{hierarchy['hierarchical_mean_spearman_rho']:.2f} "
                    f"[{hierarchy['participant_bootstrap_ci_low']:.2f}, "
                    f"{hierarchy['participant_bootstrap_ci_high']:.2f}]\n"
                    f"Permutation p = {hierarchy['hierarchical_permutation_p']:.3g}, "
                    f"FDR q = {hierarchy['p_fdr_hierarchical_all_panels']:.3g}\n"
                    f"{len(protocol_data)} pairs, "
                    f"{int(hierarchy['n_subjects'])} participants"
                ),
                transform=ax.transAxes,
                va="top",
                fontsize=7.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
            )
            ax.set_title(f"{protocol:g} Hz | {CONNECTIVITY_METRICS[metric]}", fontsize=9)
            ax.set_xlabel("Post IMF4 rise-time ratio: seed / target", fontsize=8)
            ax.set_ylabel(CONNECTIVITY_METRICS[metric], fontsize=8)
    fig.suptitle(
        f"{reference}: post-stimulation waveform timing and connectivity",
        fontsize=13,
    )
    stem = reference.lower().replace("-", "_").replace(" ", "_")
    png = figure_dir / f"figure_03_{stem}_hierarchical_connectivity_metrics.png"
    pdf = figure_dir / f"figure_03_{stem}_hierarchical_connectivity_metrics.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    set_publication_style()
    out_root = (
        ROOT
        / "outputs"
        / "publication_comparison"
        / "post_rise_time_connectivity"
    )
    table_dir = out_root / "tables"
    figure_dir = out_root / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.concat(
        [_load_pairs(reference, root) for reference, root in ANALYSES.items()],
        ignore_index=True,
    )
    run_stats = _spearman_rows(pairs)
    subject_stats, inference = _subject_inference(run_stats)
    hierarchical = _hierarchical_inference(pairs)

    pair_path = table_dir / "post_imf4_rise_time_seed_target_pairs.csv"
    run_path = table_dir / "post_imf4_rise_time_connectivity_run_spearman.csv"
    subject_path = table_dir / "post_imf4_rise_time_connectivity_subject_summary.csv"
    inference_path = table_dir / "post_imf4_rise_time_connectivity_subject_inference.csv"
    hierarchical_path = (
        table_dir / "post_imf4_rise_time_connectivity_hierarchical_permutation.csv"
    )
    pairs.to_csv(pair_path, index=False)
    run_stats.to_csv(run_path, index=False)
    subject_stats.to_csv(subject_path, index=False)
    inference.to_csv(inference_path, index=False)
    hierarchical.to_csv(hierarchical_path, index=False)

    figures: list[Path] = []
    for reference in ANALYSES:
        figures.extend(_summary_figure(pairs, hierarchical, reference, figure_dir))

    primary = inference.loc[
        inference["x_metric"].eq("post_rise_time_ratio_seed_over_target")
        & inference["connectivity_metric"].eq("post_abs_imcoh")
    ]
    print(f"Seed-target pairs: {len(pairs)}")
    print(f"Runs: {pairs['file'].nunique()}")
    print(f"Participants: {pairs['subject'].nunique()}")
    print(f"Run-level tests: {len(run_stats)}")
    print("Primary subject-level summaries:")
    print(primary.to_string(index=False))
    print(f"Pair table: {pair_path}")
    print(f"Run statistics: {run_path}")
    print(f"Subject inference: {inference_path}")
    print(f"Hierarchical inference: {hierarchical_path}")
    print("Hierarchical results:")
    print(hierarchical.to_string(index=False))
    for path in figures:
        print(f"Figure: {path}")


if __name__ == "__main__":
    main()
