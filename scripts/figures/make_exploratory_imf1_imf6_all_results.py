from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs" / "publication_bipolar_mne" / "primary_28s_guarded"
TABLES = BASE / "tables"
SUPPORT = TABLES / "figure_support"
OUT = BASE / "exploratory_imf1_imf6_all_results"

IMFS = [1, 2, 3, 4, 5, 6]
PROTOCOLS = [1.0, 50.0]
COLORS = {1.0: "#2A9D8F", 50.0: "#D62828"}
REGION_ORDER = ["frontal", "insula", "temporal", "hippocampus", "parietal", "occipital", "other"]
DISTANCE_BINS = ["<30", "30-<60", "60-<90", ">=90"]
CONNECTIVITY_METRICS = ["delta_abs_imcoh", "delta_imcoh_signed", "delta_coh"]
WAVEFORM_METRICS = ["delta_mean_if", "delta_asc2desc", "delta_peak2trough"]


@dataclass
class LoadedTables:
    csv_count: int
    waveform: pd.DataFrame
    connectivity: pd.DataFrame


def exact_signflip(values: pd.Series) -> float:
    x = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if x.size == 0:
        return np.nan
    observed = abs(float(x.mean()))
    means = [abs(float(np.mean(x * np.asarray(signs)))) for signs in product([-1.0, 1.0], repeat=x.size)]
    return float(np.mean(np.asarray(means) >= observed - np.finfo(float).eps))


def bh_fdr(p_values: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().astype(float).sort_values()
    if valid.empty:
        return out
    n = len(valid)
    raw = np.asarray([p * n / rank for rank, p in enumerate(valid, start=1)], dtype=float)
    adjusted = np.minimum.accumulate(raw[::-1])[::-1]
    out.loc[valid.index] = np.minimum(adjusted, 1.0)
    return out


def sem(values: pd.Series) -> float:
    clean = values.dropna().astype(float)
    return float(clean.sem()) if len(clean) > 1 else np.nan


def direction(value: float) -> str:
    if not np.isfinite(value) or np.isclose(value, 0.0):
        return "no_clear_direction"
    return "increase" if value > 0 else "decrease"


def interpretation_level(analysis_family: str, imf: int, metric: str) -> str:
    if imf == 4 and analysis_family == "waveform_nonstimulated" and metric == "delta_mean_if":
        return "primary"
    if imf == 4 and analysis_family == "seed_to_target_connectivity" and metric == "delta_abs_imcoh":
        return "primary"
    return "exploratory"


def test_subject_values(
    subject_values: pd.DataFrame,
    analysis_family: str,
    metric: str,
    imf: int,
    region: str = "",
    distance_bin: str = "",
) -> list[dict]:
    rows: list[dict] = []
    for protocol in PROTOCOLS:
        values = subject_values.loc[subject_values["stim_frequency_hz"].eq(protocol), "value"].dropna().astype(float)
        if values.empty:
            continue
        rows.append(
            {
                "analysis_family": analysis_family,
                "imf": imf,
                "protocol": f"{protocol:g} Hz",
                "metric": metric,
                "region": region,
                "distance_bin": distance_bin,
                "contrast": "post_minus_pre",
                "n_subjects": int(values.size),
                "mean_delta": float(values.mean()),
                "median_delta": float(values.median()),
                "sd_delta": float(values.std(ddof=1)) if values.size > 1 else np.nan,
                "sem_delta": float(values.sem()) if values.size > 1 else np.nan,
                "p_uncorrected": exact_signflip(values),
            }
        )
    paired = subject_values.pivot_table(index="subject", columns="stim_frequency_hz", values="value", aggfunc="mean")
    if 1.0 in paired.columns and 50.0 in paired.columns:
        diffs = (paired[50.0] - paired[1.0]).dropna().astype(float)
        if not diffs.empty:
            rows.append(
                {
                    "analysis_family": analysis_family,
                    "imf": imf,
                    "protocol": "paired",
                    "metric": metric,
                    "region": region,
                    "distance_bin": distance_bin,
                    "contrast": "50Hz_minus_1Hz",
                    "n_subjects": int(diffs.size),
                    "mean_delta": float(diffs.mean()),
                    "median_delta": float(diffs.median()),
                    "sd_delta": float(diffs.std(ddof=1)) if diffs.size > 1 else np.nan,
                    "sem_delta": float(diffs.sem()) if diffs.size > 1 else np.nan,
                    "p_uncorrected": exact_signflip(diffs),
                }
            )
    for row in rows:
        row["direction"] = direction(row["mean_delta"])
        row["interpretation_level"] = interpretation_level(analysis_family, imf, metric)
    return rows


def run_to_subject(run_values: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        run_values.dropna(subset=[metric])
        .groupby(["subject", "stim_frequency_hz"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: "value"})
    )


def load_waveform_values() -> pd.DataFrame:
    profiles = pd.read_csv(SUPPORT / "imf1_imf6_waveform_profiles.csv")
    control = pd.read_csv(SUPPORT / "imf1_imf6_control_point_node_values.csv")

    if_means = (
        profiles.loc[profiles["node_group"].eq("nonstimulated") if "node_group" in profiles.columns else ~profiles["is_stimulated_channel"].astype(bool)]
        if "node_group" in profiles.columns
        else profiles.loc[~profiles["is_stimulated_channel"].astype(bool)]
    )
    if_means = (
        if_means.groupby(
            ["file", "subject", "stim_frequency_hz", "channel", "region_family", "imf", "epoch"],
            as_index=False,
        )["mean_if_hz"]
        .mean()
    )
    if_wide = if_means.pivot_table(
        index=["file", "subject", "stim_frequency_hz", "channel", "region_family", "imf"],
        columns="epoch",
        values="mean_if_hz",
        aggfunc="mean",
    ).reset_index()
    if_wide["delta_mean_if"] = if_wide.get("post", np.nan) - if_wide.get("pre", np.nan)

    control = control.loc[control["node_group"].eq("nonstimulated")].copy()
    merged = control.merge(
        if_wide[["file", "channel", "imf", "delta_mean_if"]],
        on=["file", "channel", "imf"],
        how="left",
    )
    return merged


def load_tables() -> LoadedTables:
    csv_count = len(list(BASE.rglob("*.csv")))
    waveform = load_waveform_values()
    connectivity = pd.read_csv(TABLES / "connectivity_edges.csv")
    return LoadedTables(csv_count=csv_count, waveform=waveform, connectivity=connectivity)


def waveform_results(waveform: pd.DataFrame) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    rows: list[dict] = []
    support: dict[str, pd.DataFrame] = {}

    for metric in WAVEFORM_METRICS:
        run = (
            waveform.groupby(["file", "subject", "stim_frequency_hz", "imf"], as_index=False)[metric]
            .mean()
        )
        support[f"waveform_nonstimulated_{metric}_run"] = run
        for imf in IMFS:
            subject = run_to_subject(run.loc[run["imf"].eq(imf)], metric)
            rows.extend(test_subject_values(subject, "waveform_nonstimulated", metric, imf))

        reg_run = (
            waveform.groupby(["file", "subject", "stim_frequency_hz", "region_family", "imf"], as_index=False)[metric]
            .mean()
        )
        support[f"region_waveform_{metric}_run"] = reg_run
        for (imf, region), sub in reg_run.groupby(["imf", "region_family"]):
            if pd.isna(region):
                continue
            subject = run_to_subject(sub, metric)
            rows.extend(test_subject_values(subject, "region_waveform", metric, int(imf), region=str(region)))
    return rows, support


def included_seed_edges(connectivity: pd.DataFrame) -> pd.DataFrame:
    return connectivity.loc[
        connectivity["edge_type"].eq("seed_to_target")
        & connectivity["analysis_include"].fillna(False).astype(bool)
        & connectivity["imf"].isin(IMFS)
    ].copy()


def included_remote_edges(connectivity: pd.DataFrame) -> pd.DataFrame:
    return connectivity.loc[
        connectivity["edge_type"].eq("remote_to_remote")
        & connectivity["analysis_include"].fillna(False).astype(bool)
        & connectivity["imf"].isin(IMFS)
    ].copy()


def distance_bin(values: pd.Series) -> pd.Series:
    return pd.cut(values.astype(float), bins=[0.0, 30.0, 60.0, 90.0, np.inf], labels=DISTANCE_BINS, right=False)


def connectivity_results(connectivity: pd.DataFrame) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    rows: list[dict] = []
    support: dict[str, pd.DataFrame] = {}
    seed = included_seed_edges(connectivity)
    remote = included_remote_edges(connectivity)
    seed["distance_bin"] = distance_bin(seed["distance_mm"])
    remote["distance_bin"] = distance_bin(remote["edge_distance_to_seed_mm"])

    for metric in CONNECTIVITY_METRICS:
        seed_run = (
            seed.groupby(["file", "subject", "stim_frequency_hz", "imf"], as_index=False)[metric]
            .mean()
        )
        support[f"seed_to_target_connectivity_{metric}_run"] = seed_run
        for imf in IMFS:
            subject = run_to_subject(seed_run.loc[seed_run["imf"].eq(imf)], metric)
            rows.extend(test_subject_values(subject, "seed_to_target_connectivity", metric, imf))

        region_run = (
            seed.groupby(["file", "subject", "stim_frequency_hz", "node_b_region_family", "imf"], as_index=False)[metric]
            .mean()
        )
        support[f"region_seed_connectivity_{metric}_run"] = region_run
        for (imf, region), sub in region_run.groupby(["imf", "node_b_region_family"]):
            subject = run_to_subject(sub, metric)
            rows.extend(test_subject_values(subject, "region_seed_connectivity", metric, int(imf), region=str(region)))

        dist_run = (
            seed.dropna(subset=["distance_bin"])
            .groupby(["file", "subject", "stim_frequency_hz", "distance_bin", "imf"], as_index=False, observed=True)[metric]
            .mean()
        )
        support[f"distance_seed_connectivity_{metric}_run"] = dist_run
        for (imf, dist), sub in dist_run.groupby(["imf", "distance_bin"], observed=True):
            subject = run_to_subject(sub, metric)
            rows.extend(test_subject_values(subject, "distance_seed_connectivity", metric, int(imf), distance_bin=str(dist)))

        remote_run = (
            remote.groupby(["file", "subject", "stim_frequency_hz", "imf"], as_index=False)[metric]
            .mean()
        )
        support[f"remote_network_connectivity_{metric}_run"] = remote_run
        for imf in IMFS:
            subject = run_to_subject(remote_run.loc[remote_run["imf"].eq(imf)], metric)
            rows.extend(test_subject_values(subject, "remote_network_connectivity", metric, imf))
    return rows, support


def coupling_results(waveform: pd.DataFrame, connectivity: pd.DataFrame) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    rows: list[dict] = []
    support: dict[str, pd.DataFrame] = {}
    wf = waveform[["file", "subject", "stim_frequency_hz", "channel", "region_family", "imf", "delta_mean_if"]].copy()
    seed = included_seed_edges(connectivity)
    joined = seed.merge(
        wf.rename(columns={"channel": "target_channel", "region_family": "target_region_family"}),
        on=["file", "subject", "stim_frequency_hz", "target_channel", "imf"],
        how="inner",
    )
    support["waveform_connectivity_edge_join"] = joined
    for metric in CONNECTIVITY_METRICS:
        run_rows: list[dict] = []
        for keys, group in joined.groupby(["file", "subject", "stim_frequency_hz", "imf"], dropna=False):
            clean = group[["delta_mean_if", metric]].replace([np.inf, -np.inf], np.nan).dropna()
            rho = stats.spearmanr(clean["delta_mean_if"], clean[metric]).statistic if len(clean) >= 4 else np.nan
            run_rows.append(
                {
                    "file": keys[0],
                    "subject": keys[1],
                    "stim_frequency_hz": keys[2],
                    "imf": int(keys[3]),
                    f"rho_{metric}": rho,
                    "n_edges": len(clean),
                }
            )
        run = pd.DataFrame(run_rows)
        support[f"waveform_connectivity_coupling_rho_{metric}_run"] = run
        rho_metric = f"rho_{metric}"
        for imf in IMFS:
            subject = run_to_subject(run.loc[run["imf"].eq(imf)], rho_metric)
            rows.extend(test_subject_values(subject, "waveform_connectivity_coupling", rho_metric, imf))
    return rows, support


def finalize_results(rows: list[dict]) -> pd.DataFrame:
    results = pd.DataFrame(rows)
    if results.empty:
        return results
    results["p_fdr_within_family"] = np.nan
    for family, index in results.groupby("analysis_family").groups.items():
        results.loc[index, "p_fdr_within_family"] = bh_fdr(results.loc[index, "p_uncorrected"])
    results["p_fdr_all_exploratory"] = np.nan
    exploratory = results["interpretation_level"].eq("exploratory")
    results.loc[exploratory, "p_fdr_all_exploratory"] = bh_fdr(results.loc[exploratory, "p_uncorrected"])
    results["significant_uncorrected_0_05"] = results["p_uncorrected"] < 0.05
    results["significant_fdr_family_0_05"] = results["p_fdr_within_family"] < 0.05
    cols = [
        "analysis_family",
        "imf",
        "protocol",
        "metric",
        "region",
        "distance_bin",
        "contrast",
        "n_subjects",
        "mean_delta",
        "median_delta",
        "sd_delta",
        "sem_delta",
        "p_uncorrected",
        "p_fdr_within_family",
        "p_fdr_all_exploratory",
        "direction",
        "significant_uncorrected_0_05",
        "significant_fdr_family_0_05",
        "interpretation_level",
    ]
    for col in cols:
        if col not in results:
            results[col] = ""
    return results[cols].sort_values(["analysis_family", "imf", "metric", "region", "distance_bin", "protocol"])


def save_support_tables(support: dict[str, pd.DataFrame]) -> None:
    support_dir = OUT / "support_tables"
    support_dir.mkdir(parents=True, exist_ok=True)
    for name, table in support.items():
        if not table.empty:
            table.to_csv(support_dir / f"{name}.csv", index=False)


def matrix_from_results(results: pd.DataFrame, family: str, metric: str, protocol: str, value: str = "mean_delta") -> pd.DataFrame:
    sub = results.loc[
        results["analysis_family"].eq(family)
        & results["metric"].eq(metric)
        & results["protocol"].eq(protocol)
        & results["contrast"].eq("post_minus_pre")
    ].copy()
    return sub


def savefig(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_metric_lines(results: pd.DataFrame, family: str, metrics: list[str], title: str, name: str, ylabel: str) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        for protocol in ["1 Hz", "50 Hz"]:
            sub = matrix_from_results(results, family, metric, protocol)
            sub = sub.loc[sub["region"].eq("") & sub["distance_bin"].eq("")]
            ax.plot(sub["imf"], sub["mean_delta"], marker="o", color=COLORS[1.0 if protocol == "1 Hz" else 50.0], label=protocol)
            ax.fill_between(
                sub["imf"].to_numpy(float),
                (sub["mean_delta"] - sub["sem_delta"]).to_numpy(float),
                (sub["mean_delta"] + sub["sem_delta"]).to_numpy(float),
                color=COLORS[1.0 if protocol == "1 Hz" else 50.0],
                alpha=0.12,
                lw=0,
            )
        ax.axhline(0, color="0.45", lw=0.8, ls="--")
        ax.set_xticks(IMFS)
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("IMF")
        ax.set_ylabel(ylabel)
    axes[0].legend(frameon=False)
    fig.suptitle(title, y=1.05)
    savefig(fig, name)


def plot_region_heatmap(results: pd.DataFrame, family: str, metric: str, title: str, name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8), constrained_layout=True, sharey=True)
    arrays = []
    for protocol in ["1 Hz", "50 Hz"]:
        sub = matrix_from_results(results, family, metric, protocol)
        piv = sub.pivot_table(index="region", columns="imf", values="mean_delta", aggfunc="mean").reindex(REGION_ORDER)
        piv = piv.reindex(columns=IMFS)
        arrays.append(piv.to_numpy(float))
    finite = np.concatenate([a[np.isfinite(a)] for a in arrays if np.isfinite(a).any()])
    limit = max(float(np.nanmax(np.abs(finite))) if finite.size else 1.0, 1e-6)
    for ax, protocol, array in zip(axes, ["1 Hz", "50 Hz"], arrays):
        image = ax.imshow(array, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_xticks(np.arange(len(IMFS)), [str(i) for i in IMFS])
        ax.set_yticks(np.arange(len(REGION_ORDER)), [r.title() for r in REGION_ORDER])
        ax.set_xlabel("IMF")
        ax.set_title(protocol)
    fig.colorbar(image, ax=axes, label="Mean post - pre change", shrink=0.8)
    fig.suptitle(title, y=1.04)
    savefig(fig, name)


def plot_significance(results: pd.DataFrame) -> None:
    sub = results.loc[results["interpretation_level"].eq("exploratory")].copy()
    sub = sub.dropna(subset=["p_uncorrected"])
    sub["signed_logp"] = -np.log10(sub["p_uncorrected"].clip(lower=np.finfo(float).tiny)) * np.sign(sub["mean_delta"])
    top = sub.sort_values("p_uncorrected").head(40).copy()
    labels = (
        top["analysis_family"].str.replace("_", " ")
        + " | IMF"
        + top["imf"].astype(str)
        + " | "
        + top["metric"].str.replace("_", " ")
        + " | "
        + top["protocol"].astype(str)
        + top["region"].where(top["region"].eq(""), " | " + top["region"])
        + top["distance_bin"].where(top["distance_bin"].eq(""), " | " + top["distance_bin"])
        + " | "
        + top["contrast"].astype(str)
    )
    fig, ax = plt.subplots(figsize=(10, max(5, 0.18 * len(top))), constrained_layout=True)
    colors = np.where(top["p_fdr_within_family"] < 0.05, "#6A3D9A", "#4C78A8")
    ax.barh(np.arange(len(top)), top["signed_logp"], color=colors)
    ax.axvline(0, color="0.25", lw=0.8)
    ax.set_yticks(np.arange(len(top)), labels, fontsize=5.8)
    ax.invert_yaxis()
    ax.set_xlabel("Signed -log10(uncorrected p)")
    ax.set_title("Exploratory IMF1-IMF6 significance overview\nPurple indicates FDR within-family p < 0.05")
    savefig(fig, "Figure_S6_exploratory_significance_overview")


def make_figures(results: pd.DataFrame) -> None:
    plot_metric_lines(
        results,
        "waveform_nonstimulated",
        WAVEFORM_METRICS,
        "Figure S1. IMF1-IMF6 waveform and shape-ratio summary",
        "Figure_S1_imf1_imf6_waveform_shape_summary",
        "Mean post - pre change",
    )
    plot_metric_lines(
        results,
        "seed_to_target_connectivity",
        CONNECTIVITY_METRICS,
        "Figure S2. IMF1-IMF6 seed-to-target connectivity summary",
        "Figure_S2_imf1_imf6_seed_connectivity_summary",
        "Mean post - pre change",
    )
    plot_region_heatmap(
        results,
        "region_waveform",
        "delta_mean_if",
        "Figure S3. Region-wise waveform mean IF change across IMF1-IMF6",
        "Figure_S3_region_waveform_imf_heatmap",
    )
    plot_region_heatmap(
        results,
        "region_seed_connectivity",
        "delta_abs_imcoh",
        "Figure S4. Region-wise seed-to-target absolute ImCoh change across IMF1-IMF6",
        "Figure_S4_region_seed_connectivity_imf_heatmap",
    )
    plot_metric_lines(
        results,
        "waveform_connectivity_coupling",
        [f"rho_{m}" for m in CONNECTIVITY_METRICS],
        "Figure S5. Waveform-connectivity coupling across IMF1-IMF6",
        "Figure_S5_imf1_imf6_waveform_connectivity_coupling",
        "Mean within-run Spearman rho",
    )
    plot_significance(results)


def write_summary(results: pd.DataFrame, loaded: LoadedTables, paths: list[Path]) -> Path:
    summary_path = OUT / "SUPERVISOR_SUMMARY.txt"
    exploratory = results.loc[results["interpretation_level"].eq("exploratory")].copy()
    uncorrected = exploratory.loc[exploratory["significant_uncorrected_0_05"]]
    family = exploratory.loc[exploratory["significant_fdr_family_0_05"]]
    all_fdr = exploratory.loc[exploratory["p_fdr_all_exploratory"] < 0.05]
    top = exploratory.dropna(subset=["p_uncorrected"]).sort_values(["p_uncorrected", "mean_delta"], ascending=[True, False]).head(20)

    lines = [
        "SUPERVISOR SUMMARY: EXPLORATORY IMF1-IMF6 ALL RESULTS",
        "",
        f"Existing CSV files detected under primary_28s_guarded: {loaded.csv_count}",
        f"Waveform rows loaded for exploratory IMF summaries: {len(loaded.waveform):,}",
        f"Connectivity rows loaded: {len(loaded.connectivity):,}",
        f"Total tests in master table: {len(results):,}",
        f"Exploratory tests: {len(exploratory):,}",
        f"Uncorrected exploratory p < .05: {len(uncorrected):,}",
        f"FDR within-family exploratory p < .05: {len(family):,}",
        f"FDR across all exploratory tests p < .05: {len(all_fdr):,}",
        "",
        "UNCORRECTED p < .05 EFFECTS",
    ]
    if uncorrected.empty:
        lines.append("None.")
    else:
        for row in uncorrected.sort_values("p_uncorrected").itertuples():
            lines.append(
                f"- {row.analysis_family}, IMF{row.imf}, {row.protocol}, {row.metric}, "
                f"region={row.region or 'NA'}, distance={row.distance_bin or 'NA'}, "
                f"contrast={row.contrast}, mean={row.mean_delta:.4g}, p={row.p_uncorrected:.4g}"
            )
    lines.extend(["", "FDR WITHIN-FAMILY p < .05 EFFECTS"])
    if family.empty:
        lines.append("None.")
    else:
        for row in family.sort_values("p_fdr_within_family").itertuples():
            lines.append(
                f"- {row.analysis_family}, IMF{row.imf}, {row.protocol}, {row.metric}, "
                f"region={row.region or 'NA'}, distance={row.distance_bin or 'NA'}, "
                f"mean={row.mean_delta:.4g}, p_fdr_family={row.p_fdr_within_family:.4g}"
            )
    lines.extend(["", "FDR ACROSS ALL EXPLORATORY TESTS p < .05 EFFECTS"])
    if all_fdr.empty:
        lines.append("None.")
    else:
        for row in all_fdr.sort_values("p_fdr_all_exploratory").itertuples():
            lines.append(
                f"- {row.analysis_family}, IMF{row.imf}, {row.protocol}, {row.metric}, "
                f"region={row.region or 'NA'}, distance={row.distance_bin or 'NA'}, "
                f"mean={row.mean_delta:.4g}, p_fdr_all={row.p_fdr_all_exploratory:.4g}"
            )
    lines.extend(["", "TOP 20 STRONGEST EXPLORATORY EFFECTS"])
    for rank, row in enumerate(top.itertuples(), start=1):
        lines.append(
            f"{rank:02d}. {row.analysis_family}, IMF{row.imf}, {row.protocol}, {row.metric}, "
            f"region={row.region or 'NA'}, distance={row.distance_bin or 'NA'}, contrast={row.contrast}, "
            f"n={row.n_subjects}, mean={row.mean_delta:.4g}, p={row.p_uncorrected:.4g}, "
            f"family_FDR={row.p_fdr_within_family:.4g}, all_FDR={row.p_fdr_all_exploratory:.4g}"
        )
    lines.extend(
        [
            "",
            "CAUTION",
            "IMF4 delta_mean_if and IMF4 seed-to-target delta_abs_imcoh remain the only prespecified primary endpoints.",
            "All other IMF1-IMF6 findings are exploratory, hypothesis-generating, and should not be promoted as confirmatory.",
            "",
            "OUTPUT FILES",
        ]
    )
    for path in paths:
        lines.append(f"- {path}")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = load_tables()
    rows: list[dict] = []
    support: dict[str, pd.DataFrame] = {}

    wf_rows, wf_support = waveform_results(loaded.waveform)
    rows.extend(wf_rows)
    support.update(wf_support)

    conn_rows, conn_support = connectivity_results(loaded.connectivity)
    rows.extend(conn_rows)
    support.update(conn_support)

    coupling_rows, coupling_support = coupling_results(loaded.waveform, loaded.connectivity)
    rows.extend(coupling_rows)
    support.update(coupling_support)

    results = finalize_results(rows)
    master_path = OUT / "EXPLORATORY_IMF1_IMF6_ALL_RESULTS.csv"
    results.to_csv(master_path, index=False)
    save_support_tables(support)
    make_figures(results)

    figure_paths = sorted(OUT.glob("Figure_S*.png")) + sorted(OUT.glob("Figure_S*.pdf"))
    summary_path = write_summary(results, loaded, [master_path, *figure_paths])

    exploratory = results.loc[results["interpretation_level"].eq("exploratory")]
    uncorrected_n = int(exploratory["significant_uncorrected_0_05"].sum())
    fdr_family_n = int(exploratory["significant_fdr_family_0_05"].sum())
    fdr_all_n = int((exploratory["p_fdr_all_exploratory"] < 0.05).sum())
    print(f"CSV files detected: {loaded.csv_count}")
    print(f"Waveform rows loaded: {len(loaded.waveform)}")
    print(f"Connectivity rows loaded: {len(loaded.connectivity)}")
    print(f"Tests performed: {len(results)}")
    print(f"Uncorrected significant exploratory effects: {uncorrected_n}")
    print(f"FDR within-family significant exploratory effects: {fdr_family_n}")
    print(f"FDR across all exploratory significant effects: {fdr_all_n}")
    print(f"Output CSV: {master_path}")
    print(f"Summary text: {summary_path}")
    print("Figures:")
    for path in figure_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
