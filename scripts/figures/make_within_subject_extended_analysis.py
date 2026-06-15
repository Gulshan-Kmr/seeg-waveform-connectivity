from __future__ import annotations

from itertools import combinations, product
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
ENDPOINTS = {
    "imf4_mean_if": "IMF4 mean IF",
    "imf4_asc_desc_ratio": "IMF4 ascent/descent",
    "imf4_peak_trough_ratio": "IMF4 peak/trough",
    "imf4_abs_imcoh": "Seed-to-target abs ImCoh",
    "imf4_remote_abs_imcoh": "Remote-network abs ImCoh",
    "imf4_waveform_imcoh_coupling": "Waveform-ImCoh coupling",
    "distance_imcoh_rho": "Distance-ImCoh rho",
}
COLORS = {1.0: "#2A9D8F", 50.0: "#D62828"}
REF_MARKERS = {"Primary adjacent bipolar": "o", "CAR sensitivity": "D"}
REGION_ORDER = ["frontal", "insula", "temporal", "hippocampus", "parietal", "occipital", "other"]
DISTANCE_BINS = [0.0, 30.0, 60.0, 90.0, np.inf]
DISTANCE_LABELS = ["<30", "30-<60", "60-<90", ">=90"]


def _exact_signflip(values: pd.Series | np.ndarray) -> float:
    x = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if x.size == 0:
        return np.nan
    observed = abs(float(x.mean()))
    means = [abs(float(np.mean(x * np.asarray(signs)))) for signs in product([-1.0, 1.0], repeat=x.size)]
    return float(np.mean(np.asarray(means) >= observed - np.finfo(float).eps))


def _sem(values: pd.Series | np.ndarray) -> float:
    clean = pd.Series(values).dropna().astype(float)
    return float(clean.sem()) if len(clean) > 1 else np.nan


def _load_tables(root: Path) -> dict[str, pd.DataFrame]:
    tables = root / "tables"
    out = {
        "waveform": pd.read_csv(tables / "waveform_nodes.csv", low_memory=False),
        "connectivity": pd.read_csv(tables / "connectivity_edges.csv", low_memory=False),
        "coupling": pd.read_csv(tables / "waveform_connectivity_coupling.csv", low_memory=False),
    }
    return out


def _subject_values_from_core() -> pd.DataFrame:
    path = ROOT / "outputs" / "publication_comparison" / "within_subject_analysis" / "tables" / "within_subject_subject_protocol_values.csv"
    return pd.read_csv(path)


def _distance_subject_values() -> pd.DataFrame:
    path = ROOT / "outputs" / "publication_comparison" / "within_subject_analysis" / "tables" / "within_subject_distance_imcoh_subject_spearman.csv"
    if not path.exists():
        path = ROOT / "outputs" / "publication_comparison" / "distance_imcoh_correlation" / "tables" / "distance_imcoh_subject_spearman.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"spearman_rho": "delta"})
    df["endpoint"] = "distance_imcoh_rho"
    df["pre_value"] = np.nan
    df["post_value"] = np.nan
    return df[["analysis_reference", "endpoint", "subject", "stim_frequency_hz", "delta", "pre_value", "post_value"]]


def all_subject_endpoint_values() -> pd.DataFrame:
    values = _subject_values_from_core()
    dist = _distance_subject_values()
    keep = ["analysis_reference", "endpoint", "subject", "stim_frequency_hz", "delta", "pre_value", "post_value"]
    return pd.concat([values[keep], dist[keep]], ignore_index=True, sort=False)


def direction_consistency(values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ref, endpoint, freq), group in values.groupby(["analysis_reference", "endpoint", "stim_frequency_hz"]):
        delta = group["delta"].dropna()
        pos = int((delta > 0).sum())
        neg = int((delta < 0).sum())
        zero = int((delta == 0).sum())
        n = pos + neg
        p_binom = np.nan
        if n:
            p_binom = float(stats.binomtest(max(pos, neg), n=n, p=0.5, alternative="two-sided").pvalue)
        rows.append(
            {
                "analysis_reference": ref,
                "endpoint": endpoint,
                "endpoint_label": ENDPOINTS.get(endpoint, endpoint),
                "stim_frequency_hz": float(freq),
                "n_positive": pos,
                "n_negative": neg,
                "n_zero": zero,
                "n_nonzero": n,
                "dominant_direction": "positive" if pos > neg else ("negative" if neg > pos else "tie"),
                "binomial_sign_p": p_binom,
                "mean_delta": float(delta.mean()) if len(delta) else np.nan,
                "exact_signflip_p": _exact_signflip(delta),
            }
        )
    return pd.DataFrame(rows)


def leave_one_subject_out(values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ref, endpoint, freq), group in values.groupby(["analysis_reference", "endpoint", "stim_frequency_hz"]):
        subjects = sorted(group["subject"].dropna().unique())
        for subject in subjects:
            remain = group.loc[~group["subject"].eq(subject), "delta"].dropna()
            rows.append(
                {
                    "analysis_reference": ref,
                    "endpoint": endpoint,
                    "endpoint_label": ENDPOINTS.get(endpoint, endpoint),
                    "stim_frequency_hz": float(freq),
                    "left_out_subject": subject,
                    "n_remaining_subjects": int(remain.size),
                    "mean_delta_without_subject": float(remain.mean()) if len(remain) else np.nan,
                    "exact_signflip_p_without_subject": _exact_signflip(remain),
                }
            )
    return pd.DataFrame(rows)


def protocol_paired_table(values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ref, endpoint), group in values.groupby(["analysis_reference", "endpoint"]):
        pivot = group.pivot_table(index="subject", columns="stim_frequency_hz", values="delta", aggfunc="mean")
        if 1.0 not in pivot.columns or 50.0 not in pivot.columns:
            continue
        for subject, row in pivot.dropna(subset=[1.0, 50.0]).iterrows():
            diff = row[50.0] - row[1.0]
            rows.append(
                {
                    "analysis_reference": ref,
                    "endpoint": endpoint,
                    "endpoint_label": ENDPOINTS.get(endpoint, endpoint),
                    "subject": subject,
                    "value_1hz": float(row[1.0]),
                    "value_50hz": float(row[50.0]),
                    "difference_50hz_minus_1hz": float(diff),
                    "direction": "higher_50hz" if diff > 0 else ("lower_50hz" if diff < 0 else "no_difference"),
                }
            )
    return pd.DataFrame(rows)


def reference_robustness(values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (endpoint, freq), group in values.groupby(["endpoint", "stim_frequency_hz"]):
        pivot = group.pivot_table(index="subject", columns="analysis_reference", values="delta", aggfunc="mean")
        if not set(ANALYSES).issubset(pivot.columns):
            continue
        clean = pivot.dropna(subset=list(ANALYSES))
        if len(clean) >= 3 and clean[list(ANALYSES)[0]].nunique() > 1 and clean[list(ANALYSES)[1]].nunique() > 1:
            rho, p = stats.spearmanr(clean[list(ANALYSES)[0]], clean[list(ANALYSES)[1]])
        else:
            rho, p = np.nan, np.nan
        rows.append(
            {
                "endpoint": endpoint,
                "endpoint_label": ENDPOINTS.get(endpoint, endpoint),
                "stim_frequency_hz": float(freq),
                "n_subjects": int(len(clean)),
                "spearman_rho_bipolar_vs_car": float(rho) if np.isfinite(rho) else np.nan,
                "spearman_p_descriptive": float(p) if np.isfinite(p) else np.nan,
                "mean_bipolar_minus_car": float((clean[list(ANALYSES)[0]] - clean[list(ANALYSES)[1]]).mean()) if len(clean) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def region_waveform(tables: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for ref, t in tables.items():
        wf = t["waveform"]
        clean = wf.loc[wf["analysis_include"].fillna(False).astype(bool)].copy()
        run = (
            clean.groupby(["file", "subject", "stim_frequency_hz", "region_family"], as_index=False)
            .agg(delta_mean_if=("delta_mean_if", "mean"), n_sites=("channel", "nunique"))
        )
        subject = run.groupby(["subject", "stim_frequency_hz", "region_family"], as_index=False).agg(
            delta_mean_if=("delta_mean_if", "mean"), n_runs=("file", "nunique"), n_sites=("n_sites", "sum")
        )
        subject["analysis_reference"] = ref
        rows.append(subject)
    return pd.concat(rows, ignore_index=True, sort=False)


def region_connectivity(tables: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for ref, t in tables.items():
        edges = t["connectivity"]
        seed = edges.loc[
            edges["imf"].eq(4)
            & edges["edge_type"].eq("seed_to_target")
            & edges["analysis_include"].fillna(False).astype(bool)
        ].copy()
        run = (
            seed.groupby(["file", "subject", "stim_frequency_hz", "node_b_region_family"], as_index=False)
            .agg(delta_abs_imcoh=("delta_abs_imcoh", "mean"), n_edges=("target_channel", "nunique"))
            .rename(columns={"node_b_region_family": "region_family"})
        )
        subject = run.groupby(["subject", "stim_frequency_hz", "region_family"], as_index=False).agg(
            delta_abs_imcoh=("delta_abs_imcoh", "mean"), n_runs=("file", "nunique"), n_edges=("n_edges", "sum")
        )
        subject["analysis_reference"] = ref
        rows.append(subject)
    return pd.concat(rows, ignore_index=True, sort=False)


def remote_region_pairs(tables: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for ref, t in tables.items():
        edges = t["connectivity"]
        remote = edges.loc[
            edges["imf"].eq(4)
            & edges["edge_type"].eq("remote_to_remote")
            & edges["analysis_include"].fillna(False).astype(bool)
        ].copy()
        if remote.empty:
            continue
        pairs = remote.apply(lambda r: sorted([str(r["node_a_region_family"]), str(r["node_b_region_family"])]), axis=1)
        remote["region_a"] = pairs.map(lambda x: x[0])
        remote["region_b"] = pairs.map(lambda x: x[1])
        run = remote.groupby(["file", "subject", "stim_frequency_hz", "region_a", "region_b"], as_index=False).agg(
            delta_abs_imcoh=("delta_abs_imcoh", "mean"), n_edges=("delta_abs_imcoh", "size")
        )
        subject = run.groupby(["subject", "stim_frequency_hz", "region_a", "region_b"], as_index=False).agg(
            delta_abs_imcoh=("delta_abs_imcoh", "mean"), n_runs=("file", "nunique"), n_edges=("n_edges", "sum")
        )
        subject["analysis_reference"] = ref
        rows.append(subject)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def distance_bins(tables: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for ref, t in tables.items():
        seed = t["connectivity"].loc[
            t["connectivity"]["imf"].eq(4)
            & t["connectivity"]["edge_type"].eq("seed_to_target")
            & t["connectivity"]["analysis_include"].fillna(False).astype(bool)
        ].copy()
        seed["distance_bin_mm"] = pd.cut(seed["distance_mm"], bins=DISTANCE_BINS, labels=DISTANCE_LABELS, right=False)
        run = (
            seed.dropna(subset=["distance_bin_mm"])
            .groupby(["file", "subject", "stim_frequency_hz", "distance_bin_mm"], observed=True, as_index=False)
            .agg(delta_abs_imcoh=("delta_abs_imcoh", "mean"), n_edges=("delta_abs_imcoh", "size"))
        )
        subject = run.groupby(["subject", "stim_frequency_hz", "distance_bin_mm"], observed=True, as_index=False).agg(
            delta_abs_imcoh=("delta_abs_imcoh", "mean"), n_runs=("file", "nunique"), n_edges=("n_edges", "sum")
        )
        subject["analysis_reference"] = ref
        rows.append(subject)
    return pd.concat(rows, ignore_index=True, sort=False)


def stimulated_vs_nonstim_waveform(tables: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for ref, t in tables.items():
        wf = t["waveform"].loc[t["waveform"]["analysis_include"].fillna(False).astype(bool)].copy()
        wf["site_group"] = np.where(wf["is_stimulated_channel"].fillna(False).astype(bool), "stimulated", "non_stimulated")
        run = (
            wf.groupby(["file", "subject", "stim_frequency_hz", "site_group"], as_index=False)
            .agg(
                delta_mean_if=("delta_mean_if", "mean"),
                delta_asc2desc=("delta_asc2desc", "mean"),
                delta_peak2trough=("delta_peak2trough", "mean"),
                n_sites=("channel", "nunique"),
            )
        )
        subject = run.groupby(["subject", "stim_frequency_hz", "site_group"], as_index=False).agg(
            delta_mean_if=("delta_mean_if", "mean"),
            delta_asc2desc=("delta_asc2desc", "mean"),
            delta_peak2trough=("delta_peak2trough", "mean"),
            n_runs=("file", "nunique"),
            n_sites=("n_sites", "sum"),
        )
        subject["analysis_reference"] = ref
        rows.append(subject)
    return pd.concat(rows, ignore_index=True, sort=False)


def imf1_imf6_profiles(tables: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for ref, t in tables.items():
        edges = t["connectivity"]
        seed = edges.loc[
            edges["edge_type"].eq("seed_to_target") & edges["analysis_include"].fillna(False).astype(bool)
        ].copy()
        if seed.empty:
            continue
        run = seed.groupby(["file", "subject", "stim_frequency_hz", "imf"], as_index=False).agg(
            delta_abs_imcoh=("delta_abs_imcoh", "mean"), n_edges=("delta_abs_imcoh", "size")
        )
        subject = run.groupby(["subject", "stim_frequency_hz", "imf"], as_index=False).agg(
            delta_abs_imcoh=("delta_abs_imcoh", "mean"), n_runs=("file", "nunique"), n_edges=("n_edges", "sum")
        )
        subject["analysis_reference"] = ref
        rows.append(subject)
    return pd.concat(rows, ignore_index=True, sort=False)


def current_sensitivity(tables: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for ref, t in tables.items():
        wf = t["waveform"].loc[t["waveform"]["analysis_include"].fillna(False).astype(bool)].copy()
        run = wf.groupby(["file", "subject", "stim_frequency_hz", "stim_amplitude_ma"], as_index=False).agg(
            delta_mean_if=("delta_mean_if", "mean")
        )
        for freq, group in run.groupby("stim_frequency_hz"):
            clean = group[["stim_amplitude_ma", "delta_mean_if"]].dropna()
            rho, p = (np.nan, np.nan)
            if len(clean) >= 4 and clean["stim_amplitude_ma"].nunique() > 1:
                rho, p = stats.spearmanr(clean["stim_amplitude_ma"], clean["delta_mean_if"])
            rows.append(
                {
                    "analysis_reference": ref,
                    "stim_frequency_hz": float(freq),
                    "n_runs": int(len(clean)),
                    "spearman_rho_current_delta_mean_if": float(rho) if np.isfinite(rho) else np.nan,
                    "spearman_p_descriptive": float(p) if np.isfinite(p) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")


def _scatter_violin(ax: plt.Axes, data: pd.DataFrame, x_col: str, y_col: str, order: list, title: str, ylabel: str) -> None:
    rng = np.random.default_rng(101)
    x_lookup = {v: i for i, v in enumerate(order)}
    for value in order:
        vals = data.loc[data[x_col].eq(value), y_col].dropna().astype(float)
        if not len(vals):
            continue
        xpos = x_lookup[value]
        parts = ax.violinplot([vals.to_numpy()], positions=[xpos], widths=0.72, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor("0.75")
            body.set_edgecolor("0.35")
            body.set_alpha(0.22)
        ax.scatter(np.repeat(xpos, len(vals)) + rng.normal(0, 0.035, len(vals)), vals, s=24, color="0.25", alpha=0.75)
        ax.scatter([xpos], [vals.mean()], marker="D", color="#4C78A8", edgecolor="white", s=58, zorder=4)
    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.set_xticks(range(len(order)), [str(v).replace("_", " ").title() for v in order], rotation=35, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="0.9", lw=0.45)


def plot_direction_and_forest(values: pd.DataFrame, direction: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), constrained_layout=True)
    endpoints = ["imf4_mean_if", "imf4_abs_imcoh", "imf4_waveform_imcoh_coupling", "distance_imcoh_rho"]
    for ax, endpoint, letter in zip(axes.ravel(), endpoints, "abcd"):
        sub = values.loc[values["endpoint"].eq(endpoint)]
        for ref, marker in REF_MARKERS.items():
            ref_sub = sub.loc[sub["analysis_reference"].eq(ref)]
            for freq, color in COLORS.items():
                vals = ref_sub.loc[ref_sub["stim_frequency_hz"].eq(freq), ["subject", "delta"]].dropna()
                x = np.arange(len(vals))
                offset = -0.12 if ref == "Primary adjacent bipolar" else 0.12
                ax.scatter(
                    x + offset + (0 if freq == 1.0 else 0.34),
                    vals["delta"],
                    color=color,
                    marker=marker,
                    s=32,
                    alpha=0.82,
                    edgecolor="white",
                    lw=0.35,
                    label=f"{ref}, {freq:g} Hz" if endpoint == endpoints[0] else None,
                )
        ax.axhline(0, color="0.4", lw=0.8, ls="--")
        ax.set_title(ENDPOINTS[endpoint])
        ax.set_ylabel("Subject-level value")
        ax.set_xticks([])
        _panel(ax, letter)
    fig.legend(frameon=False, fontsize=7, loc="lower center", ncol=2)
    fig.suptitle("Subject-level effect-size forest summaries", y=1.02)
    path = out_dir / "extended_01_subject_effect_size_forest.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    pivot = direction.pivot_table(
        index=["endpoint_label", "analysis_reference"], columns="stim_frequency_hz", values="n_positive", aggfunc="first"
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    heat = direction.copy()
    heat["signed_fraction_positive"] = (heat["n_positive"] - heat["n_negative"]) / heat["n_nonzero"].replace(0, np.nan)
    mat = heat.pivot_table(index="endpoint_label", columns=["analysis_reference", "stim_frequency_hz"], values="signed_fraction_positive")
    im = ax.imshow(mat.to_numpy(float), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7)
    ax.set_yticks(range(mat.shape[0]), mat.index)
    ax.set_xticks(range(mat.shape[1]), [f"{a}\n{f:g} Hz" for a, f in mat.columns], rotation=35, ha="right")
    ax.set_title("Direction consistency: positive versus negative subject changes")
    fig.colorbar(im, ax=ax, label="Signed fraction positive")
    path = out_dir / "extended_02_direction_consistency_heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def plot_region_distance_stim_imf(region_wf: pd.DataFrame, region_conn: pd.DataFrame, dist: pd.DataFrame, stim: pd.DataFrame, imf: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.4), constrained_layout=True)
    _scatter_violin(axes[0, 0], region_wf, "region_family", "delta_mean_if", REGION_ORDER, "Region-wise waveform change", "Delta mean IF (Hz)")
    _scatter_violin(axes[0, 1], region_conn, "region_family", "delta_abs_imcoh", REGION_ORDER, "Region-wise seed-to-target ImCoh", "Delta abs ImCoh")
    _scatter_violin(axes[1, 0], dist, "distance_bin_mm", "delta_abs_imcoh", DISTANCE_LABELS, "Distance-bin seed-to-target ImCoh", "Delta abs ImCoh")
    _scatter_violin(axes[1, 1], stim, "site_group", "delta_mean_if", ["stimulated", "non_stimulated"], "Stimulated versus non-stimulated waveform", "Delta mean IF (Hz)")
    for ax, letter in zip(axes.ravel(), "abcd"):
        _panel(ax, letter)
    fig.suptitle("Within-subject regional, distance, and stimulation-site summaries", y=1.02)
    path = out_dir / "extended_03_region_distance_stimulated_summaries.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for ref, linestyle in [("Primary adjacent bipolar", "-"), ("CAR sensitivity", "--")]:
        for freq, color in COLORS.items():
            sub = imf.loc[imf["analysis_reference"].eq(ref) & imf["stim_frequency_hz"].eq(freq)]
            summary = sub.groupby("imf", as_index=False).agg(mean=("delta_abs_imcoh", "mean"), sem=("delta_abs_imcoh", _sem))
            ax.errorbar(summary["imf"], summary["mean"], yerr=summary["sem"], marker="o", color=color, ls=linestyle, capsize=2, label=f"{ref}, {freq:g} Hz")
    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.set_xticks([1, 2, 3, 4, 5, 6], [f"IMF{i}" for i in range(1, 7)])
    ax.set_ylabel("Seed-to-target delta abs ImCoh")
    ax.set_title("Within-subject IMF1-IMF6 connectivity profile")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    path = out_dir / "extended_04_imf1_imf6_within_subject_connectivity_profile.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def plot_reference_current_loo(refrob: pd.DataFrame, current: pd.DataFrame, loo: pd.DataFrame, paired: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.4), constrained_layout=True)
    selected = refrob.loc[refrob["endpoint"].isin(["imf4_mean_if", "imf4_abs_imcoh", "distance_imcoh_rho", "imf4_waveform_imcoh_coupling"])]
    for ax, freq, letter in zip(axes[0], [1.0, 50.0], "ab"):
        sub = selected.loc[selected["stim_frequency_hz"].eq(freq)]
        ax.barh(np.arange(len(sub)), sub["spearman_rho_bipolar_vs_car"], color="#4C78A8", alpha=0.75)
        ax.axvline(0, color="0.4", lw=0.8)
        ax.set_yticks(np.arange(len(sub)), sub["endpoint_label"])
        ax.set_xlabel("Spearman rho")
        ax.set_title(f"Bipolar-CAR robustness, {freq:g} Hz")
        _panel(ax, letter)
    for ax, freq, letter in zip(axes[1], [1.0, 50.0], "cd"):
        sub = current.loc[current["stim_frequency_hz"].eq(freq)]
        ax.bar(np.arange(len(sub)), sub["spearman_rho_current_delta_mean_if"], color="#F58518", alpha=0.75)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_xticks(np.arange(len(sub)), sub["analysis_reference"], rotation=20, ha="right")
        ax.set_ylabel("rho(current, delta mean IF)")
        ax.set_title(f"Current sensitivity, {freq:g} Hz")
        _panel(ax, letter)
    fig.suptitle("Reference robustness and current sensitivity", y=1.02)
    path = out_dir / "extended_05_reference_robustness_current_sensitivity.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True)
    loo_sel = loo.loc[loo["endpoint"].isin(["imf4_mean_if", "imf4_abs_imcoh"]) & loo["analysis_reference"].eq("Primary adjacent bipolar")]
    for ax, endpoint, letter in zip(axes, ["imf4_mean_if", "imf4_abs_imcoh"], "ab"):
        sub = loo_sel.loc[loo_sel["endpoint"].eq(endpoint)]
        for freq, color in COLORS.items():
            vals = sub.loc[sub["stim_frequency_hz"].eq(freq), "mean_delta_without_subject"].dropna()
            ax.scatter(np.repeat(freq, len(vals)), vals, color=color, s=30, alpha=0.8)
        ax.axhline(0, color="0.4", lw=0.8, ls="--")
        ax.set_xticks([1, 50], ["1 Hz", "50 Hz"])
        ax.set_ylabel("Mean delta after leaving one subject out")
        ax.set_title(f"Leave-one-subject-out: {ENDPOINTS[endpoint]}")
        _panel(ax, letter)
    path = out_dir / "extended_06_leave_one_subject_out_primary_endpoints.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    return paths


def write_summary(out_path: Path, tables: dict[str, pd.DataFrame]) -> None:
    lines = [
        "Within-subject extended sensitivity analyses",
        "==========================================",
        "",
        "All analyses are exploratory/sensitivity summaries. Subject-level summaries are used; contacts and edges are not treated as independent inference units.",
        "",
        "Generated analysis families:",
        "1. Subject-level effect-size forest plots",
        "2. Direction consistency/sign-test summaries",
        "3. Region-wise waveform summaries",
        "4. Region-wise seed-to-target connectivity summaries",
        "5. Distance-bin seed-to-target connectivity summaries",
        "6. Stimulated versus non-stimulated waveform summaries",
        "7. IMF1-IMF6 seed-to-target connectivity profile",
        "8. Stimulation-current sensitivity",
        "9. Bipolar-CAR reference robustness",
        "10. Leave-one-subject-out and paired subject trajectory tables",
        "",
    ]
    direction = tables["direction_consistency"]
    sig_dir = direction.loc[direction["binomial_sign_p"].lt(0.05, fill_value=False)]
    lines.append(f"Direction-consistency rows with uncorrected binomial p < .05: {len(sig_dir)}")
    for row in sig_dir.itertuples():
        lines.append(f"- {row.analysis_reference}; {row.endpoint_label}; {row.stim_frequency_hz:g} Hz; {row.dominant_direction}; p={row.binomial_sign_p:.4g}")
    refrob = tables["reference_robustness"].dropna(subset=["spearman_rho_bipolar_vs_car"])
    lines.append("")
    lines.append("Strongest bipolar-CAR robustness correlations:")
    for row in refrob.reindex(refrob["spearman_rho_bipolar_vs_car"].abs().sort_values(ascending=False).index).head(10).itertuples():
        lines.append(f"- {row.endpoint_label}; {row.stim_frequency_hz:g} Hz: rho={row.spearman_rho_bipolar_vs_car:.3g}, p={row.spearman_p_descriptive:.3g}, n={row.n_subjects}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out_root = ROOT / "outputs" / "publication_comparison" / "within_subject_analysis" / "extended_analyses"
    table_dir = out_root / "tables"
    figure_dir = out_root / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    tables = {ref: _load_tables(root) for ref, root in ANALYSES.items()}
    values = all_subject_endpoint_values()
    outputs = {
        "all_subject_endpoint_values": values,
        "direction_consistency": direction_consistency(values),
        "leave_one_subject_out": leave_one_subject_out(values),
        "protocol_paired_subject_trajectories": protocol_paired_table(values),
        "reference_robustness": reference_robustness(values),
        "region_waveform_subject_values": region_waveform(tables),
        "region_connectivity_subject_values": region_connectivity(tables),
        "remote_region_pair_subject_values": remote_region_pairs(tables),
        "distance_bin_subject_values": distance_bins(tables),
        "stimulated_vs_nonstim_waveform_subject_values": stimulated_vs_nonstim_waveform(tables),
        "imf1_imf6_connectivity_subject_values": imf1_imf6_profiles(tables),
        "current_sensitivity_run_spearman": current_sensitivity(tables),
    }
    for name, table in outputs.items():
        table.to_csv(table_dir / f"{name}.csv", index=False)
    paths = []
    paths += plot_direction_and_forest(values, outputs["direction_consistency"], figure_dir)
    paths += plot_region_distance_stim_imf(
        outputs["region_waveform_subject_values"],
        outputs["region_connectivity_subject_values"],
        outputs["distance_bin_subject_values"],
        outputs["stimulated_vs_nonstim_waveform_subject_values"],
        outputs["imf1_imf6_connectivity_subject_values"],
        figure_dir,
    )
    paths += plot_reference_current_loo(
        outputs["reference_robustness"],
        outputs["current_sensitivity_run_spearman"],
        outputs["leave_one_subject_out"],
        outputs["protocol_paired_subject_trajectories"],
        figure_dir,
    )
    write_summary(out_root / "WITHIN_SUBJECT_EXTENDED_SUMMARY.txt", outputs)
    print(f"output_root: {out_root}")
    for name in outputs:
        print(f"table: {table_dir / f'{name}.csv'}")
    print(f"summary: {out_root / 'WITHIN_SUBJECT_EXTENDED_SUMMARY.txt'}")
    for path in paths:
        print(f"figure: {path}")


if __name__ == "__main__":
    main()
