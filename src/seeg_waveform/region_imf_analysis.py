from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .localization import normalize_channel_name
from .quinn_figures import set_publication_style
from .region_waveform import add_region_family, subject_number
from .spatial_analysis import bh_fdr


def join_imf_connectivity_to_regions(
    imf_connectivity_csv: Path,
    waveform_regions_csv: Path,
) -> pd.DataFrame:
    imf = pd.read_csv(imf_connectivity_csv)
    imf["subject_number"] = imf["subject"].map(subject_number)
    imf["seed_norm"] = imf["seed_channel"].map(normalize_channel_name)
    imf["target_norm"] = imf["target_channel"].map(normalize_channel_name)
    regions = pd.read_csv(waveform_regions_csv)
    regions = add_region_family(regions)
    region_cols = [
        "subject_number",
        "file",
        "channel_norm",
        "primary_region",
        "region_family",
        "analysis_gray_matter_flag",
        "hemisphere",
        "x_mni",
        "y_mni",
        "z_mni",
        "delta_mean_if",
        "delta_asc2desc",
        "delta_peak2trough",
        "pre_noise_flag",
        "post_noise_flag",
    ]
    target = regions[region_cols].drop_duplicates(["subject_number", "file", "channel_norm"]).rename(
        columns={
            "channel_norm": "target_norm",
            "primary_region": "target_region",
            "region_family": "target_region_family",
            "analysis_gray_matter_flag": "target_gray_matter_flag",
            "hemisphere": "target_hemisphere",
            "x_mni": "target_x_mni",
            "y_mni": "target_y_mni",
            "z_mni": "target_z_mni",
            "pre_noise_flag": "target_pre_noise_flag",
            "post_noise_flag": "target_post_noise_flag",
        }
    )
    seed = regions[
        [
            "subject_number",
            "file",
            "channel_norm",
            "primary_region",
            "region_family",
            "analysis_gray_matter_flag",
            "x_mni",
            "y_mni",
            "z_mni",
        ]
    ].drop_duplicates(["subject_number", "file", "channel_norm"]).rename(
        columns={
            "channel_norm": "seed_norm",
            "primary_region": "seed_region",
            "region_family": "seed_region_family",
            "analysis_gray_matter_flag": "seed_gray_matter_flag",
            "x_mni": "seed_x_mni",
            "y_mni": "seed_y_mni",
            "z_mni": "seed_z_mni",
        }
    )
    out = imf.merge(seed, on=["subject_number", "file", "seed_norm"], how="left").merge(
        target, on=["subject_number", "file", "target_norm"], how="left"
    )
    out["seed_target_distance_mni_mm"] = np.linalg.norm(
        out[["seed_x_mni", "seed_y_mni", "seed_z_mni"]].to_numpy(float)
        - out[["target_x_mni", "target_y_mni", "target_z_mni"]].to_numpy(float),
        axis=1,
    )
    out["analysis_pair_include"] = (
        out["status"].eq("ok")
        & out["target_gray_matter_flag"].fillna(False)
        & ~out["target_pre_noise_flag"].fillna(False).astype(bool)
        & ~out["target_post_noise_flag"].fillna(False).astype(bool)
    )
    return out


def summarize_region_imf(joined: pd.DataFrame, level: str = "pair") -> pd.DataFrame:
    df = joined.loc[joined["analysis_pair_include"].fillna(False)].copy()
    group_cols = ["stim_frequency_hz", "target_region_family", "imf"]
    if level == "file_mean":
        df = (
            df.groupby(["file", "subject", *group_cols], dropna=False)
            .agg(
                pre_coh=("pre_coh", "mean"),
                post_coh=("post_coh", "mean"),
                delta_coh=("delta_coh", "mean"),
                pre_imcoh=("pre_imcoh", "mean"),
                post_imcoh=("post_imcoh", "mean"),
                delta_imcoh=("delta_imcoh", "mean"),
                n_pairs=("target_norm", "size"),
            )
            .reset_index()
        )
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["level"] = level
        row["n"] = int(len(group))
        row["n_subjects"] = int(group["subject"].nunique()) if "subject" in group else np.nan
        row["n_files"] = int(group["file"].nunique()) if "file" in group else np.nan
        for metric in ["coh", "imcoh"]:
            pre = group[f"pre_{metric}"].replace([np.inf, -np.inf], np.nan)
            post = group[f"post_{metric}"].replace([np.inf, -np.inf], np.nan)
            delta = group[f"delta_{metric}"].replace([np.inf, -np.inf], np.nan)
            row[f"pre_{metric}_mean"] = pre.mean()
            row[f"post_{metric}_mean"] = post.mean()
            row[f"delta_{metric}_mean"] = delta.mean()
            valid = delta.dropna()
            row[f"delta_{metric}_median"] = valid.median()
            if len(valid) >= 4:
                row[f"delta_{metric}_wilcoxon_p"] = stats.wilcoxon(valid).pvalue if np.any(valid != 0) else np.nan
            else:
                row[f"delta_{metric}_wilcoxon_p"] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    for metric in ["coh", "imcoh"]:
        pcol = f"delta_{metric}_wilcoxon_p"
        if pcol in out:
            out[f"{pcol}_fdr"] = bh_fdr(out[pcol])
    return out


def imf_waveform_coupling(joined: pd.DataFrame, level: str = "pair") -> pd.DataFrame:
    df = joined.loc[joined["analysis_pair_include"].fillna(False)].copy()
    if level == "file_target_mean":
        df = (
            df.groupby(["file", "subject", "stim_frequency_hz", "target_norm", "target_region_family", "imf"], dropna=False)
            .agg(
                delta_coh=("delta_coh", "mean"),
                delta_imcoh=("delta_imcoh", "mean"),
                seed_target_distance_mni_mm=("seed_target_distance_mni_mm", "mean"),
                delta_mean_if=("delta_mean_if", "first"),
                delta_asc2desc=("delta_asc2desc", "first"),
                delta_peak2trough=("delta_peak2trough", "first"),
            )
            .reset_index()
        )
    rows = []
    for keys, group in df.groupby(["stim_frequency_hz", "target_region_family", "imf"], dropna=False):
        base = dict(zip(["stim_frequency_hz", "target_region_family", "imf"], keys))
        for x in ["delta_coh", "delta_imcoh"]:
            for y in ["delta_mean_if", "delta_asc2desc", "delta_peak2trough"]:
                sub = group[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
                row = dict(base)
                row["level"] = level
                row["x"] = x
                row["y"] = y
                row["n"] = int(len(sub))
                row["n_subjects"] = int(group["subject"].nunique()) if "subject" in group else np.nan
                if len(sub) >= 5 and sub[x].nunique() > 1 and sub[y].nunique() > 1:
                    rho, p = stats.spearmanr(sub[x], sub[y])
                    row["spearman_rho"] = float(rho)
                    row["p_value"] = float(p)
                else:
                    row["spearman_rho"] = np.nan
                    row["p_value"] = np.nan
                rows.append(row)
    out = pd.DataFrame(rows)
    out["p_fdr"] = bh_fdr(out["p_value"])
    return out


def compare_frequency_region_imf(joined: pd.DataFrame) -> pd.DataFrame:
    df = joined.loc[
        joined["analysis_pair_include"].fillna(False)
        & joined["target_region_family"].notna()
        & joined["target_region_family"].ne("non_gray")
    ].copy()
    file_level = (
        df.groupby(["file", "subject", "stim_frequency_hz", "target_region_family", "imf"], dropna=False)
        .agg(delta_coh=("delta_coh", "mean"), delta_imcoh=("delta_imcoh", "mean"))
        .reset_index()
    )
    rows = []
    for keys, group in file_level.groupby(["target_region_family", "imf"], dropna=False):
        base = dict(zip(["target_region_family", "imf"], keys))
        for metric in ["delta_coh", "delta_imcoh"]:
            one = group.loc[group["stim_frequency_hz"].eq(1.0), metric].dropna()
            fifty = group.loc[group["stim_frequency_hz"].eq(50.0), metric].dropna()
            row = dict(base)
            row["metric"] = metric
            row["n_1hz"] = int(len(one))
            row["n_50hz"] = int(len(fifty))
            row["n_subjects_1hz"] = int(group.loc[group["stim_frequency_hz"].eq(1.0), "subject"].nunique())
            row["n_subjects_50hz"] = int(group.loc[group["stim_frequency_hz"].eq(50.0), "subject"].nunique())
            row["mean_1hz"] = float(one.mean()) if len(one) else np.nan
            row["mean_50hz"] = float(fifty.mean()) if len(fifty) else np.nan
            row["mean_50_minus_1hz"] = row["mean_50hz"] - row["mean_1hz"]
            if len(one) >= 3 and len(fifty) >= 3 and one.nunique() > 1 and fifty.nunique() > 1:
                row["mannwhitney_p"] = float(stats.mannwhitneyu(fifty, one, alternative="two-sided").pvalue)
            else:
                row["mannwhitney_p"] = np.nan
            rows.append(row)
    out = pd.DataFrame(rows)
    out["mannwhitney_p_fdr"] = bh_fdr(out["mannwhitney_p"])
    return out


def _file_target_imf_table(joined: pd.DataFrame) -> pd.DataFrame:
    df = joined.loc[
        joined["analysis_pair_include"].fillna(False)
        & joined["target_region_family"].notna()
        & joined["target_region_family"].ne("non_gray")
    ].copy()
    out = (
        df.groupby(["file", "subject", "stim_frequency_hz", "target_norm", "target_region_family", "imf"], dropna=False)
        .agg(
            delta_coh=("delta_coh", "mean"),
            delta_imcoh=("delta_imcoh", "mean"),
            seed_target_distance_mni_mm=("seed_target_distance_mni_mm", "mean"),
            delta_mean_if=("delta_mean_if", "first"),
            delta_asc2desc=("delta_asc2desc", "first"),
            delta_peak2trough=("delta_peak2trough", "first"),
            n_seed_pairs=("seed_norm", "nunique"),
        )
        .reset_index()
    )
    out["subject"] = out["subject"].astype(str)
    out["stim_frequency_hz"] = out["stim_frequency_hz"].astype(float)
    out["imf"] = out["imf"].astype(int)
    dist = out["seed_target_distance_mni_mm"].replace([np.inf, -np.inf], np.nan)
    out["distance_z"] = (dist - dist.mean()) / dist.std(ddof=0)
    return out


def subject_level_region_imf_summary(joined: pd.DataFrame) -> pd.DataFrame:
    data = _file_target_imf_table(joined)
    return (
        data.groupby(["subject", "stim_frequency_hz", "target_region_family", "imf"], dropna=False)
        .agg(
            n_files=("file", "nunique"),
            n_targets=("target_norm", "nunique"),
            delta_coh_mean=("delta_coh", "mean"),
            delta_imcoh_mean=("delta_imcoh", "mean"),
            distance_mean_mm=("seed_target_distance_mni_mm", "mean"),
            delta_mean_if_mean=("delta_mean_if", "mean"),
            delta_asc2desc_mean=("delta_asc2desc", "mean"),
            delta_peak2trough_mean=("delta_peak2trough", "mean"),
        )
        .reset_index()
    )


def leave_one_subject_out_frequency_effect(joined: pd.DataFrame) -> pd.DataFrame:
    data = _file_target_imf_table(joined)
    subjects = sorted(data["subject"].dropna().unique())
    rows = []
    for held_out in [None, *subjects]:
        sub = data.copy() if held_out is None else data.loc[~data["subject"].eq(held_out)].copy()
        for metric in ["delta_imcoh", "delta_coh"]:
            for imf_group, imfs in {"all": [1, 2, 3, 4, 5, 6], "low_imf_1_3": [1, 2, 3], "high_imf_4_6": [4, 5, 6]}.items():
                g = sub.loc[sub["imf"].isin(imfs)]
                one = g.loc[g["stim_frequency_hz"].eq(1.0), metric].dropna()
                fifty = g.loc[g["stim_frequency_hz"].eq(50.0), metric].dropna()
                row = {
                    "held_out_subject": "none" if held_out is None else held_out,
                    "metric": metric,
                    "imf_group": imf_group,
                    "n_1hz": int(len(one)),
                    "n_50hz": int(len(fifty)),
                    "n_subjects": int(g["subject"].nunique()),
                    "mean_1hz": float(one.mean()) if len(one) else np.nan,
                    "mean_50hz": float(fifty.mean()) if len(fifty) else np.nan,
                }
                row["mean_50_minus_1hz"] = row["mean_50hz"] - row["mean_1hz"]
                if len(one) >= 3 and len(fifty) >= 3 and one.nunique() > 1 and fifty.nunique() > 1:
                    row["mannwhitney_p"] = float(stats.mannwhitneyu(fifty, one, alternative="two-sided").pvalue)
                else:
                    row["mannwhitney_p"] = np.nan
                rows.append(row)
    out = pd.DataFrame(rows)
    out["mannwhitney_p_fdr"] = bh_fdr(out["mannwhitney_p"])
    return out


def distance_bin_region_imf_summary(joined: pd.DataFrame) -> pd.DataFrame:
    data = _file_target_imf_table(joined)
    finite = data["seed_target_distance_mni_mm"].replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        data["distance_bin"] = np.nan
    else:
        labels = ["near", "middle", "far"]
        try:
            data["distance_bin"] = pd.qcut(data["seed_target_distance_mni_mm"], q=3, labels=labels, duplicates="drop")
        except ValueError:
            data["distance_bin"] = pd.cut(data["seed_target_distance_mni_mm"], bins=3, labels=labels)
    out = (
        data.groupby(["stim_frequency_hz", "distance_bin", "imf"], dropna=False, observed=False)
        .agg(
            n=("delta_imcoh", "size"),
            n_subjects=("subject", "nunique"),
            n_files=("file", "nunique"),
            distance_mean_mm=("seed_target_distance_mni_mm", "mean"),
            delta_imcoh_mean=("delta_imcoh", "mean"),
            delta_imcoh_sem=("delta_imcoh", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
            delta_coh_mean=("delta_coh", "mean"),
            delta_coh_sem=("delta_coh", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
        )
        .reset_index()
    )
    return out


def plot_distance_bin_profiles(distance_summary: pd.DataFrame, out_path: Path, metric: str = "delta_imcoh") -> Path:
    set_publication_style()
    mean_col = f"{metric}_mean"
    sem_col = f"{metric}_sem"
    bins = [x for x in ["near", "middle", "far"] if x in distance_summary["distance_bin"].astype(str).unique()]
    fig, axes = plt.subplots(1, len(bins), figsize=(3.0 * max(len(bins), 1), 3.0), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    colors = {1.0: "#2a9d8f", 50.0: "#d62828"}
    for ax, bin_name in zip(axes, bins):
        sub = distance_summary.loc[distance_summary["distance_bin"].astype(str).eq(bin_name)]
        for freq, group in sub.groupby("stim_frequency_hz"):
            group = group.sort_values("imf")
            x = group["imf"].to_numpy(float)
            y = group[mean_col].to_numpy(float)
            sem = group[sem_col].to_numpy(float)
            color = colors.get(float(freq), "0.35")
            ax.plot(x, y, marker="o", linewidth=1.4, markersize=3.5, color=color, label=f"{freq:g} Hz")
            ok = np.isfinite(sem)
            ax.fill_between(x[ok], y[ok] - sem[ok], y[ok] + sem[ok], color=color, alpha=0.15, linewidth=0)
        ax.axhline(0, color="0.25", linewidth=0.7)
        ax.set_title(f"{bin_name} targets")
        ax.set_xlabel("IMF")
        ax.set_xticks(range(1, 7))
    axes[0].set_ylabel(metric.replace("_", " "))
    axes[0].legend(frameon=False)
    fig.suptitle("IMF connectivity change by seed-target distance")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_leave_one_subject_out(sensitivity: pd.DataFrame, out_path: Path, metric: str = "delta_imcoh") -> Path:
    set_publication_style()
    df = sensitivity.loc[
        sensitivity["metric"].eq(metric) & sensitivity["imf_group"].isin(["all", "low_imf_1_3", "high_imf_4_6"])
    ].copy()
    order = ["none"] + sorted([x for x in df["held_out_subject"].unique() if x != "none"])
    groups = ["all", "low_imf_1_3", "high_imf_4_6"]
    colors = {"all": "0.25", "low_imf_1_3": "#d62828", "high_imf_4_6": "#457b9d"}
    fig, ax = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    for group in groups:
        sub = df.loc[df["imf_group"].eq(group)].set_index("held_out_subject").reindex(order)
        x = np.arange(len(order))
        ax.plot(x, sub["mean_50_minus_1hz"], marker="o", linewidth=1.3, markersize=3.5, color=colors[group], label=group)
    ax.axhline(0, color="0.25", linewidth=0.7)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel(f"{metric.replace('_', ' ')}: 50 Hz - 1 Hz")
    ax.set_xlabel("Held-out subject")
    ax.set_title("Leave-one-subject-out frequency-effect sensitivity")
    ax.legend(frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_region_imf_mixed_models(joined: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.formula.api as smf

    data = _file_target_imf_table(joined)
    data = data.replace([np.inf, -np.inf], np.nan)
    families = data.groupby("target_region_family")["subject"].nunique()
    keep_families = families[families >= 2].index
    data = data.loc[data["target_region_family"].isin(keep_families)].copy()
    data["stim_frequency_hz"] = data["stim_frequency_hz"].astype("category")
    data["imf"] = data["imf"].astype("category")
    data["target_region_family"] = data["target_region_family"].astype("category")

    specs = []
    for outcome in ["delta_imcoh", "delta_coh"]:
        specs.extend(
            [
                (
                    outcome,
                    "frequency_main_effect",
                    f"{outcome} ~ C(stim_frequency_hz) + C(imf) + C(target_region_family) + distance_z",
                ),
                (
                    outcome,
                    "frequency_by_imf",
                    f"{outcome} ~ C(stim_frequency_hz) * C(imf) + C(target_region_family) + distance_z",
                ),
                (
                    outcome,
                    "frequency_by_region",
                    f"{outcome} ~ C(stim_frequency_hz) * C(target_region_family) + C(imf) + distance_z",
                ),
            ]
        )
    for outcome in ["delta_mean_if", "delta_asc2desc", "delta_peak2trough"]:
        specs.append(
            (
                outcome,
                "connectivity_waveform_coupling",
                f"{outcome} ~ delta_imcoh + delta_coh + C(stim_frequency_hz) + C(imf) + "
                "C(target_region_family) + distance_z",
            )
        )

    rows = []
    for outcome, model_name, formula in specs:
        cols = [outcome, "subject", "stim_frequency_hz", "imf", "target_region_family", "distance_z"]
        if "delta_imcoh" in formula:
            cols.append("delta_imcoh")
        if "delta_coh" in formula:
            cols.append("delta_coh")
        model_data = data[list(dict.fromkeys(cols))].dropna().copy()
        row_base = {
            "outcome": outcome,
            "model_name": model_name,
            "formula": formula,
            "n": int(len(model_data)),
            "n_subjects": int(model_data["subject"].nunique()) if len(model_data) else 0,
            "n_files": int(data.loc[model_data.index, "file"].nunique()) if len(model_data) else 0,
        }
        if len(model_data) < 20 or model_data["subject"].nunique() < 2:
            rows.append({**row_base, "model": "not_fit", "term": "", "estimate": np.nan, "se": np.nan, "z_or_t": np.nan, "p": np.nan, "converged": False, "note": "Insufficient rows or subjects"})
            continue
        try:
            fit = smf.mixedlm(formula, model_data, groups=model_data["subject"]).fit(
                method="lbfgs", reml=False, maxiter=300, disp=False
            )
            for term, estimate in fit.params.items():
                if term == "Group Var":
                    continue
                rows.append(
                    {
                        **row_base,
                        "model": "mixedlm_subject_random_intercept",
                        "term": term,
                        "estimate": float(estimate),
                        "se": float(fit.bse.get(term, np.nan)),
                        "z_or_t": float(fit.tvalues.get(term, np.nan)),
                        "p": float(fit.pvalues.get(term, np.nan)),
                        "converged": bool(fit.converged),
                        "note": "",
                    }
                )
        except Exception as exc:
            try:
                fit = smf.ols(formula, model_data).fit(cov_type="cluster", cov_kwds={"groups": model_data["subject"]})
                for term, estimate in fit.params.items():
                    rows.append(
                        {
                            **row_base,
                            "model": "ols_clustered_by_subject_fallback",
                            "term": term,
                            "estimate": float(estimate),
                            "se": float(fit.bse.get(term, np.nan)),
                            "z_or_t": float(fit.tvalues.get(term, np.nan)),
                            "p": float(fit.pvalues.get(term, np.nan)),
                            "converged": False,
                            "note": str(exc)[:240],
                        }
                    )
            except Exception as exc2:
                rows.append(
                    {
                        **row_base,
                        "model": "failed",
                        "term": "",
                        "estimate": np.nan,
                        "se": np.nan,
                        "z_or_t": np.nan,
                        "p": np.nan,
                        "converged": False,
                        "note": f"{str(exc)[:120]} | fallback: {str(exc2)[:120]}",
                    }
                )
    out = pd.DataFrame(rows)
    if "p" in out:
        out["p_fdr"] = bh_fdr(out["p"])
    return out


def plot_frequency_contrast_heatmap(contrast: pd.DataFrame, out_path: Path, metric: str = "delta_imcoh") -> Path:
    set_publication_style()
    df = contrast.loc[contrast["metric"].eq(metric)].copy()
    families = (
        df.groupby("target_region_family")[["n_subjects_1hz", "n_subjects_50hz"]]
        .max()
        .min(axis=1)
        .sort_values(ascending=False)
        .head(8)
        .index.tolist()
    )
    columns = list(range(1, 7))
    pivot = df.pivot_table(index="target_region_family", columns="imf", values="mean_50_minus_1hz", aggfunc="mean")
    pvals = df.pivot_table(index="target_region_family", columns="imf", values="mannwhitney_p_fdr", aggfunc="min")
    n1 = df.pivot_table(index="target_region_family", columns="imf", values="n_subjects_1hz", aggfunc="max")
    n50 = df.pivot_table(index="target_region_family", columns="imf", values="n_subjects_50hz", aggfunc="max")
    pivot = pivot.reindex(index=families, columns=columns)
    pvals = pvals.reindex(index=families, columns=columns)
    n1 = n1.reindex(index=families, columns=columns)
    n50 = n50.reindex(index=families, columns=columns)
    vmax = np.nanpercentile(np.abs(pivot.to_numpy(float)), 95) if pivot.notna().any().any() else 1.0
    vmax = max(float(vmax), 1e-6)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#e6e6e6")
    fig, ax = plt.subplots(figsize=(5.0, 4.2), constrained_layout=True)
    im = ax.imshow(np.ma.masked_invalid(pivot.to_numpy(float)), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("50 Hz minus 1 Hz IMF connectivity change")
    ax.set_xlabel("IMF")
    ax.set_yticks(np.arange(len(families)))
    ax.set_yticklabels(families)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns)
    for row_i, family in enumerate(families):
        for col_i, imf in enumerate(columns):
            value = pivot.loc[family, imf]
            if pd.isna(value):
                ax.text(col_i, row_i, "NA", ha="center", va="center", fontsize=6, color="0.25")
                continue
            if pvals.loc[family, imf] < 0.05:
                ax.text(col_i, row_i, "*", ha="center", va="center", fontsize=10, color="black")
            elif min(n1.loc[family, imf], n50.loc[family, imf]) < 2:
                ax.text(col_i, row_i, "n<2", ha="center", va="center", fontsize=6, color="0.25")
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(families), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(im, ax=ax, shrink=0.8, label=f"{metric.replace('_', ' ')}: 50 Hz - 1 Hz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_region_imf_heatmap(
    summary: pd.DataFrame,
    out_path: Path,
    metric: str = "delta_imcoh_mean",
    min_subjects: int = 2,
) -> Path:
    set_publication_style()
    df = summary.copy()
    df = df.loc[(df["level"].eq("file_mean")) & df["target_region_family"].notna()]
    df = df.loc[df["target_region_family"].ne("non_gray")]
    families = (
        df.groupby("target_region_family")["n_subjects"].max().sort_values(ascending=False).head(8).index.tolist()
    )
    df = df.loc[df["target_region_family"].isin(families)]
    freqs = sorted(df["stim_frequency_hz"].dropna().unique())
    fig, axes = plt.subplots(1, len(freqs), figsize=(4.4 * len(freqs), 4.2), constrained_layout=True)
    axes = np.atleast_1d(axes)
    vmax = np.nanpercentile(np.abs(df[metric]), 95) if df[metric].notna().any() else 1.0
    vmax = max(float(vmax), 1e-6)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#e6e6e6")
    im = None
    for ax, freq in zip(axes, freqs):
        freq_df = df.loc[df["stim_frequency_hz"].eq(freq)]
        pivot = freq_df.pivot_table(index="target_region_family", columns="imf", values=metric, aggfunc="mean")
        n_subjects = freq_df.pivot_table(index="target_region_family", columns="imf", values="n_subjects", aggfunc="max")
        columns = list(range(1, 7))
        pivot = pivot.reindex(index=families, columns=columns)
        n_subjects = n_subjects.reindex(index=families, columns=columns)
        values = np.ma.masked_invalid(pivot.to_numpy(float))
        im = ax.imshow(values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"{freq:g} Hz")
        ax.set_xlabel("IMF")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([int(x) for x in pivot.columns])
        ax.set_xlim(-0.5, len(columns) - 0.5)
        ax.set_ylim(len(families) - 0.5, -0.5)
        for row_i, family in enumerate(families):
            for col_i, imf in enumerate(columns):
                value = pivot.loc[family, imf]
                n_sub = n_subjects.loc[family, imf]
                if pd.isna(value):
                    ax.text(col_i, row_i, "NA", ha="center", va="center", fontsize=6, color="0.25")
                elif pd.isna(n_sub) or n_sub < min_subjects:
                    rect = plt.Rectangle(
                        (col_i - 0.5, row_i - 0.5),
                        1,
                        1,
                        fill=False,
                        hatch="///",
                        edgecolor="0.35",
                        linewidth=0.0,
                    )
                    ax.add_patch(rect)
                    ax.text(col_i, row_i, f"n={int(n_sub)}", ha="center", va="center", fontsize=6, color="0.15")
        ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(families), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.8, label=metric.replace("_", " "))
    fig.suptitle("Region-family IMF connectivity change")
    fig.text(
        0.02,
        -0.02,
        "Gray cells/NA indicate no included data; hatched cells indicate fewer than "
        f"{min_subjects} subjects and are descriptive only.",
        fontsize=7,
        color="0.25",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_region_imf_file_profiles(joined: pd.DataFrame, out_path: Path, metric: str = "delta_imcoh") -> Path:
    set_publication_style()
    df = joined.loc[
        joined["analysis_pair_include"].fillna(False)
        & joined["target_region_family"].notna()
        & joined["target_region_family"].ne("non_gray")
    ].copy()
    top_families = (
        df.groupby("target_region_family")["subject"].nunique().sort_values(ascending=False).head(6).index.tolist()
    )
    file_level = (
        df.loc[df["target_region_family"].isin(top_families)]
        .groupby(["file", "subject", "stim_frequency_hz", "target_region_family", "imf"], dropna=False)
        .agg(value=(metric, "mean"))
        .reset_index()
    )
    plot_df = (
        file_level.groupby(["stim_frequency_hz", "target_region_family", "imf"], dropna=False)
        .agg(
            mean=("value", "mean"),
            sem=("value", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
            n_files=("file", "nunique"),
            n_subjects=("subject", "nunique"),
        )
        .reset_index()
    )
    ncols = 3
    nrows = int(np.ceil(len(top_families) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.2, 2.35 * nrows), sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    colors = {1.0: "#2a9d8f", 50.0: "#d62828"}
    for ax, family in zip(axes, top_families):
        sub = plot_df.loc[plot_df["target_region_family"].eq(family)]
        for freq, group in sub.groupby("stim_frequency_hz"):
            group = group.sort_values("imf")
            x = group["imf"].to_numpy(float)
            y = group["mean"].to_numpy(float)
            sem = group["sem"].to_numpy(float)
            color = colors.get(float(freq), "0.35")
            ax.plot(x, y, marker="o", linewidth=1.5, markersize=3.5, color=color, label=f"{freq:g} Hz")
            ok = np.isfinite(sem)
            ax.fill_between(x[ok], y[ok] - sem[ok], y[ok] + sem[ok], color=color, alpha=0.16, linewidth=0)
        ax.axhline(0, color="0.25", linewidth=0.7)
        ax.set_title(family)
        ax.set_xticks(range(1, 7))
        ax.set_xticklabels(range(1, 7))
        ax.set_xlabel("IMF")
        ax.set_ylabel(metric.replace("_", " "))
    for ax in axes[len(top_families) :]:
        ax.axis("off")
    axes[0].legend(frameon=False)
    fig.suptitle("File-level IMF connectivity profiles by target region")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_imf4_coupling(joined: pd.DataFrame, out_path: Path) -> Path:
    set_publication_style()
    df = joined.loc[
        joined["analysis_pair_include"].fillna(False)
        & joined["imf"].eq(4)
        & joined["target_region_family"].isin(["hippocampus", "temporal", "frontal", "insula"])
    ].copy()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.8), constrained_layout=True)
    specs = [
        ("delta_imcoh", "delta_mean_if"),
        ("delta_coh", "delta_mean_if"),
        ("delta_imcoh", "delta_peak2trough"),
        ("delta_coh", "delta_peak2trough"),
    ]
    colors = {1.0: "#2a9d8f", 50.0: "#d62828"}
    for ax, (x, y) in zip(axes.ravel(), specs):
        for freq, group in df.groupby("stim_frequency_hz"):
            ax.scatter(
                group[x],
                group[y],
                s=14,
                alpha=0.45,
                color=colors.get(float(freq), "0.4"),
                edgecolor="none",
                label=f"{freq:g} Hz",
            )
        ax.axhline(0, color="0.25", linewidth=0.7)
        ax.axvline(0, color="0.25", linewidth=0.7)
        ax.set_xlabel(f"IMF4 {x.replace('_', ' ')}")
        ax.set_ylabel(y.replace("_", " "))
    axes[0, 0].legend(frameon=False)
    fig.suptitle("IMF4 connectivity change vs target waveform change")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path
