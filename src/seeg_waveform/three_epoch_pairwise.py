from __future__ import annotations

from itertools import combinations
import hashlib
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import signal, stats
from statsmodels.stats.multitest import multipletests

from .all_pairs_waveform_connectivity import (
    FEATURES,
    HYPOTHESIS_SHORT_LABELS,
    ROOT_CONFIGS,
    _bootstrap_ci,
    _pair_dynamics,
    _permuted_run_rhos,
    _read_cached_csv,
)
from .config import load_config
from .io import iter_mat_files
from .preprocess import window_mask
from .primary_bipolar_mne import (
    _decompose_window,
    _metric_mean,
    _mne_values,
    _selected_imf,
    _share_constituent_contact,
    analysis_windows,
    make_bipolar_run,
)
from .primary_car_mne import make_car_run
from .region_waveform import subject_number
from .spatial_analysis import load_brainstorm_atlas_csv


CACHE_VERSION = "three_epoch_pairwise_v1"
EPOCH_ORDER = ("pre", "stimulation", "post")
EPOCH_LABELS = {
    "pre": "Pre-stimulation",
    "stimulation": "During stimulation (0-4 s)",
    "post": "Post-stimulation",
}
CONTRASTS = {
    "stimulation_minus_pre": ("stimulation", "pre"),
    "post_minus_pre": ("post", "pre"),
    "post_minus_stimulation": ("post", "stimulation"),
}
HYPOTHESIS_AXES = {
    "if_matching_imcoh": (
        "mean_if_absolute_difference",
        "abs_imcoh",
        "|IF A - IF B| (Hz)",
        "Absolute imaginary coherence",
    ),
    "ascdesc_direction_pdc": (
        "asc2desc_log_ratio",
        "gpdc_log_ratio",
        "Log ascent/descent ratio (A/B)",
        "Log gPDC ratio (A->B / B->A)",
    ),
    "peaktrough_mismatch_tra": (
        "peak2trough_absolute_difference",
        "tra_unsigned",
        "|Peak/trough ratio A - B|",
        "Temporal irreversibility",
    ),
}


def _save_figure(fig: plt.Figure, stem: Path, dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _prepare_three_epoch_run(path: Path, montage: str, base_cfg: dict, cfg: dict):
    if montage == "car":
        run, pre, post = make_car_run(path, base_cfg)
    else:
        run = make_bipolar_run(path, base_cfg)
        if run.status != "ok":
            return run, None
        pre_range, post_range = analysis_windows(run.recording, base_cfg)
        pre = _decompose_window(
            run.recording, window_mask(run.recording.time, *pre_range), base_cfg
        )
        post = _decompose_window(
            run.recording, window_mask(run.recording.time, *post_range), base_cfg
        )
    if run.status != "ok" or pre is None or post is None:
        return run, None
    stimulation_range = tuple(float(value) for value in cfg["analysis"]["stimulation_epoch_sec"])
    stimulation = _decompose_window(
        run.recording,
        window_mask(run.recording.time, *stimulation_range),
        base_cfg,
    )
    return run, {"pre": pre, "stimulation": stimulation, "post": post}


def _stimulation_segments(
    windows, valid: list[int], run, base_cfg: dict, cfg: dict
) -> np.ndarray:
    imf_number = int(cfg["analysis"]["imf"])
    signals = np.column_stack([
        _selected_imf(windows[index].result, imf_number) for index in valid
    ])
    samples = int(round(
        float(cfg["analysis"]["stimulation_connectivity_segment_sec"])
        * run.recording.fs
    ))
    count = min(
        int(cfg["analysis"]["stimulation_connectivity_segments"]),
        signals.shape[0] // samples,
    )
    if count < 2:
        raise ValueError("The stimulation epoch requires two complete subepochs.")
    return np.stack([
        signals[start * samples:(start + 1) * samples].T
        for start in range(count)
    ])


def analyze_stimulation_run(
    path: Path, montage: str, base_cfg: dict, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run, epochs = _prepare_three_epoch_run(path, montage, base_cfg, cfg)
    if epochs is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame([{
            "file": path.stem, "montage": montage, "status": run.status
        }])
    eligible = [
        index
        for index, node in run.nodes.iterrows()
        if bool(node["primary_region_include"])
        and all(
            not epochs[name][index].noise_flag
            and epochs[name][index].result.status == "ok"
            and _selected_imf(
                epochs[name][index].result, int(cfg["analysis"]["imf"])
            ) is not None
            for name in EPOCH_ORDER
        )
    ]
    pairs = [
        (left, right)
        for left, right in combinations(eligible, 2)
        if montage == "car"
        or not _share_constituent_contact(
            run.nodes.iloc[left], run.nodes.iloc[right]
        )
    ]
    if len(eligible) < 2 or not pairs:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame([{
            "file": path.stem,
            "montage": montage,
            "status": "too_few_three_epoch_nodes",
        }])
    stimulation = epochs["stimulation"]
    local = {node: position for position, node in enumerate(eligible)}
    mne_pairs = [(local[left], local[right]) for left, right in pairs]
    segmented = _stimulation_segments(stimulation, eligible, run, base_cfg, cfg)
    conn_cfg = {
        **base_cfg,
        "connectivity": {
            **base_cfg["connectivity"],
            "segment_length_sec": cfg["analysis"]["stimulation_connectivity_segment_sec"],
            "n_segments": cfg["analysis"]["stimulation_connectivity_segments"],
        },
    }
    connectivity = _mne_values(
        segmented, mne_pairs, run.recording.fs, conn_cfg
    )
    signals = {
        index: np.asarray(stimulation[index].result.imf[:, 3], float)
        for index in eligible
    }
    envelopes = {
        index: np.abs(signal.hilbert(values))
        for index, values in signals.items()
    }
    dynamics = Parallel(
        n_jobs=int(cfg["analysis"]["n_jobs"]),
        prefer="processes",
        batch_size=1,
        max_nbytes="2M",
    )(
        delayed(_pair_dynamics)(
            signals[left],
            signals[right],
            envelopes[left],
            envelopes[right],
            run.recording.fs,
            cfg["analysis"],
        )
        for left, right in pairs
    )
    feature_rows = []
    for index in eligible:
        node = run.nodes.iloc[index]
        feature_rows.append({
            "file": run.recording.file_stem,
            "subject": run.recording.subject,
            "stim_frequency_hz": run.recording.stim_frequency_hz,
            "montage": montage,
            "epoch": "stimulation",
            "channel": node["channel"],
            "region_family": node["region_family"],
            "is_stimulated_channel": bool(node["is_stimulated_channel"]),
            "x_mni": node["x_mni"],
            "y_mni": node["y_mni"],
            "z_mni": node["z_mni"],
            "n_cycles": stimulation[index].result.n_cycles,
            **{
                metric: _metric_mean(stimulation[index].result, metric)
                for metric in FEATURES
            },
        })
    pair_rows, qc_rows = [], []
    epsilon = float(cfg["analysis"]["pdc_epsilon"])
    for pair_index, (left, right) in enumerate(pairs):
        node_a, node_b = run.nodes.iloc[left], run.nodes.iloc[right]
        pdc, tra, tra_status = dynamics[pair_index]
        row = {
            "file": run.recording.file_stem,
            "subject": run.recording.subject,
            "stim_frequency_hz": run.recording.stim_frequency_hz,
            "montage": montage,
            "epoch": "stimulation",
            "channel_a": node_a["channel"],
            "channel_b": node_b["channel"],
            "pair_name": f"{node_a['channel']}--{node_b['channel']}",
            "region_a": node_a["region_family"],
            "region_b": node_b["region_family"],
            "x_a": node_a["x_mni"], "y_a": node_a["y_mni"], "z_a": node_a["z_mni"],
            "x_b": node_b["x_mni"], "y_b": node_b["y_mni"], "z_b": node_b["z_mni"],
            "pair_type": (
                "seed_to_target"
                if bool(node_a["is_stimulated_channel"])
                or bool(node_b["is_stimulated_channel"])
                else "nonstimulated_to_nonstimulated"
            ),
            "imcoh_signed": float(connectivity["imcoh"][pair_index]),
            "abs_imcoh": abs(float(connectivity["imcoh"][pair_index])),
            "coh": float(connectivity["coh"][pair_index]),
            "tra_unsigned": tra,
            "tra_status": tra_status,
            **pdc,
        }
        for metric in FEATURES:
            value_a = _metric_mean(stimulation[left].result, metric)
            value_b = _metric_mean(stimulation[right].result, metric)
            row[f"{metric}_a"] = value_a
            row[f"{metric}_b"] = value_b
            row[f"{metric}_absolute_difference"] = abs(value_a - value_b)
            row[f"{metric}_log_ratio"] = np.log(
                (value_a + epsilon) / (value_b + epsilon)
            )
        pair_rows.append(row)
        qc_rows.append({
            "file": run.recording.file_stem,
            "subject": run.recording.subject,
            "montage": montage,
            "channel_a": node_a["channel"],
            "channel_b": node_b["channel"],
            "pdc_valid": pdc.get("pdc_valid", False),
            "pdc_reason": pdc.get("pdc_reason", ""),
            "pdc_order": pdc.get("pdc_order", np.nan),
            "tra_status": tra_status,
        })
    return pd.DataFrame(feature_rows), pd.DataFrame(pair_rows), pd.DataFrame(qc_rows)


def _coordinate_table(
    source_features: pd.DataFrame, cfg: dict
) -> pd.DataFrame:
    rows = []
    atlas_dir = Path(load_config(ROOT_CONFIGS["bipolar"])["project"]["atlas_dir"])
    requested = source_features[[
        "file", "subject", "montage", "channel", "region_family"
    ]].drop_duplicates()
    for (subject, montage), group in requested.groupby(["subject", "montage"]):
        number = subject_number(subject)
        if number is None:
            continue
        atlas_type = "bipolar2" if montage == "bipolar" else "unipolar_filtered"
        hits = sorted(
            atlas_dir.glob(
                f"*Subject{number:03d}_{atlas_type}_Coordinates_and_Atlas.csv"
            )
        )
        if not hits:
            continue
        atlas = load_brainstorm_atlas_csv(hits[0], "AAL")
        lookup = atlas.drop_duplicates("channel").set_index("channel")
        for item in group.itertuples():
            if item.channel not in lookup.index:
                continue
            atlas_row = lookup.loc[item.channel]
            rows.append({
                "file": item.file,
                "montage": montage,
                "channel": item.channel,
                "x_mni": atlas_row["x_mni"],
                "y_mni": atlas_row["y_mni"],
                "z_mni": atlas_row["z_mni"],
                "region_family": item.region_family,
            })
    return pd.DataFrame(rows).drop_duplicates(["file", "montage", "channel"])


def enrich_existing_pairs(
    source_pairs: pd.DataFrame, coordinates: pd.DataFrame
) -> pd.DataFrame:
    pairs = source_pairs.copy()
    if "pair_name" not in pairs:
        pairs["pair_name"] = pairs["channel_a"] + "--" + pairs["channel_b"]
    for side in ("a", "b"):
        lookup = coordinates.rename(columns={
            "channel": f"channel_{side}",
            "x_mni": f"x_{side}",
            "y_mni": f"y_{side}",
            "z_mni": f"z_{side}",
            "region_family": f"coordinate_region_{side}",
        })
        pairs = pairs.merge(
            lookup,
            on=["file", "montage", f"channel_{side}"],
            how="left",
        )
    return pairs


def run_correlations(pairs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for keys, group in pairs.groupby(
        ["montage", "file", "subject", "stim_frequency_hz", "epoch"]
    ):
        for hypothesis in cfg["statistics"]["hypotheses"]:
            x_name = f"{hypothesis['waveform_metric']}_{hypothesis['transform']}"
            y_name = hypothesis["connectivity_metric"]
            selected = group
            if y_name == "gpdc_log_ratio":
                selected = selected[
                    selected["pdc_valid"].astype(str).str.lower().eq("true")
                ]
            elif y_name == "tra_unsigned":
                selected = selected[selected["tra_status"].eq("ok")]
            clean = selected[[x_name, y_name]].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            rho, p_value = (
                stats.spearmanr(clean[x_name], clean[y_name])
                if len(clean) >= 4
                and clean[x_name].nunique() > 1
                and clean[y_name].nunique() > 1
                else (np.nan, np.nan)
            )
            rows.append({
                "montage": keys[0], "file": keys[1], "subject": keys[2],
                "stim_frequency_hz": keys[3], "epoch": keys[4],
                "hypothesis": hypothesis["name"], "family": hypothesis["family"],
                "n_pairs": len(clean), "spearman_rho": rho,
                "pair_level_p_descriptive": p_value,
            })
    return pd.DataFrame(rows)


def subject_correlations(run: pd.DataFrame) -> pd.DataFrame:
    return run.groupby(
        ["montage", "subject", "stim_frequency_hz", "epoch", "hypothesis", "family"],
        as_index=False,
    ).agg(mean_run_rho=("spearman_rho", "mean"), n_runs=("file", "nunique"))


def _contrast_permutation_null(
    pairs: pd.DataFrame,
    montage: str,
    protocol: float,
    contrast: str,
    hypothesis: dict,
    cfg: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    newer, older = CONTRASTS[contrast]
    selected = pairs[
        (pairs["montage"] == montage)
        & (pairs["stim_frequency_hz"] == protocol)
    ]
    groups = list(selected.groupby(["subject", "file"]))
    count = int(cfg["analysis"]["n_permutations"])
    seeds = rng.integers(0, np.iinfo(np.uint32).max, len(groups), dtype=np.uint32)

    def one_run(item, seed):
        (subject, file_name), group = item
        local_rng = np.random.default_rng(int(seed))
        first = _permuted_run_rhos(
            group[group["epoch"] == newer], hypothesis, count, local_rng
        )
        second = _permuted_run_rhos(
            group[group["epoch"] == older], hypothesis, count, local_rng
        )
        return subject, file_name, first - second

    computed = Parallel(
        n_jobs=min(int(cfg["analysis"]["n_jobs"]), max(1, len(groups))),
        prefer="processes",
        batch_size=1,
    )(
        delayed(one_run)(item, seed) for item, seed in zip(groups, seeds)
    )
    subject_vectors = []
    for subject in sorted({item[0] for item in computed}):
        values = np.vstack([
            vector for active_subject, _, vector in computed
            if active_subject == subject
        ])
        valid = np.isfinite(values)
        denominator = valid.sum(axis=0)
        subject_vectors.append(np.divide(
            np.nansum(values, axis=0),
            denominator,
            out=np.full(count, np.nan),
            where=denominator > 0,
        ))
    values = np.vstack(subject_vectors)
    valid = np.isfinite(values)
    denominator = valid.sum(axis=0)
    return np.divide(
        np.nansum(values, axis=0),
        denominator,
        out=np.full(count, np.nan),
        where=denominator > 0,
    )


def correlation_change_inference(
    subject: pd.DataFrame, pairs: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = subject.pivot_table(
        index=["montage", "subject", "stim_frequency_hz", "hypothesis", "family"],
        columns="epoch",
        values="mean_run_rho",
        aggfunc="first",
    ).reset_index()
    change_rows = []
    for contrast, (newer, older) in CONTRASTS.items():
        if newer not in wide or older not in wide:
            continue
        frame = wide[[
            "montage", "subject", "stim_frequency_hz", "hypothesis", "family"
        ]].copy()
        frame["contrast"] = contrast
        frame["rho_change"] = wide[newer] - wide[older]
        change_rows.append(frame)
    changes = pd.concat(change_rows, ignore_index=True)
    rng = np.random.default_rng(int(cfg["analysis"]["random_seed"]))
    rows = []
    for keys, group in changes.groupby(
        ["montage", "stim_frequency_hz", "contrast", "hypothesis", "family"]
    ):
        values = group["rho_change"].dropna().to_numpy(float)
        if not len(values):
            continue
        low, high = _bootstrap_ci(
            values, int(cfg["analysis"]["n_bootstraps"]), rng
        )
        hypothesis = next(
            item for item in cfg["statistics"]["hypotheses"]
            if item["name"] == keys[3]
        )
        null = _contrast_permutation_null(
            pairs, keys[0], keys[1], keys[2], hypothesis, cfg, rng
        )
        null = null[np.isfinite(null)]
        p_value = (
            (1 + np.sum(np.abs(null) >= abs(values.mean()))) / (len(null) + 1)
            if len(null) else np.nan
        )
        rows.append({
            "montage": keys[0], "stim_frequency_hz": keys[1],
            "contrast": keys[2], "hypothesis": keys[3], "family": keys[4],
            "n_subjects": len(values), "mean_rho_change": values.mean(),
            "median_rho_change": np.median(values),
            "bootstrap_ci_low": low, "bootstrap_ci_high": high,
            "channel_label_permutation_p": p_value,
        })
    inference = pd.DataFrame(rows)
    inference["p_fdr_within_family"] = np.nan
    for _, indices in inference.groupby(
        ["montage", "contrast", "family"]
    ).groups.items():
        inference.loc[indices, "p_fdr_within_family"] = multipletests(
            inference.loc[indices, "channel_label_permutation_p"],
            method="fdr_bh",
        )[1]
    return changes, inference


def make_timing_figure(output_dir: Path, dpi: int) -> list[Path]:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 3.8), sharex=True)
    for ax, protocol, stim_end in zip(axes, ("1 Hz protocol", "50 Hz protocol"), (30, 5)):
        ax.axvspan(-30, -2, color="#457B9D", alpha=0.28, label="Guarded pre")
        ax.axvspan(0, 4, color="#E76F51", alpha=0.55, label="Exploratory stimulation")
        ax.axvspan(stim_end + 1, stim_end + 29, color="#2A9D8F", alpha=0.28, label="Guarded post")
        ax.axvspan(0, stim_end, color="#E9C46A", alpha=0.22, label="Stimulation train")
        ax.axvline(0, color="0.2", lw=1)
        ax.set_yticks([])
        ax.set_ylabel(protocol, rotation=0, ha="right", va="center")
        ax.text(2, 0.86, "0-4 s exploratory epoch", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
    axes[-1].set_xlabel("Time from stimulation onset (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Protocol-specific timing and analysis epochs")
    fig.tight_layout(rect=(0, 0.16, 1, 0.95))
    stem = output_dir / "figure_01_protocol_timing"
    _save_figure(fig, stem, dpi)
    return [stem.with_suffix(".png"), stem.with_suffix(".pdf")]


def _rank_trend(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_rank, y_rank = stats.rankdata(x), stats.rankdata(y)
    slope, intercept = np.polyfit(x_rank, y_rank, 1)
    order = np.argsort(x)
    y_sorted = np.sort(y)
    fitted_rank = intercept + slope * x_rank[order]
    return x[order], np.interp(
        fitted_rank, np.arange(1, len(y_sorted) + 1), y_sorted
    )


def make_pair_scatter_figures(
    pairs: pd.DataFrame, cfg: dict, output_dir: Path
) -> list[Path]:
    paths = []
    top_n = int(cfg["analysis"]["static_label_top_n"])
    dpi = int(cfg["analysis"]["figure_dpi"])
    for (montage, file_name), group in pairs.groupby(["montage", "file"]):
        subject = str(group["subject"].iloc[0])
        protocol = float(group["stim_frequency_hz"].iloc[0])
        fig, axes = plt.subplots(3, 3, figsize=(13, 11))
        for row, hypothesis in enumerate(HYPOTHESIS_AXES):
            x_name, y_name, x_label, y_label = HYPOTHESIS_AXES[hypothesis]
            all_clean = group[[x_name, y_name]].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            x_limits = (
                all_clean[x_name].quantile(0.005),
                all_clean[x_name].quantile(0.995),
            ) if len(all_clean) else None
            y_limits = (
                all_clean[y_name].quantile(0.005),
                all_clean[y_name].quantile(0.995),
            ) if len(all_clean) else None
            for column, epoch in enumerate(EPOCH_ORDER):
                ax = axes[row, column]
                selected = group[group["epoch"] == epoch].copy()
                if y_name == "gpdc_log_ratio":
                    selected = selected[
                        selected["pdc_valid"].astype(str).str.lower().eq("true")
                    ]
                elif y_name == "tra_unsigned":
                    selected = selected[selected["tra_status"].eq("ok")]
                clean = selected.replace([np.inf, -np.inf], np.nan).dropna(
                    subset=[x_name, y_name]
                )
                ax.scatter(
                    clean[x_name], clean[y_name], s=8, alpha=0.22,
                    color="#31688E", rasterized=True
                )
                if len(clean) >= 4:
                    rho, p_value = stats.spearmanr(clean[x_name], clean[y_name])
                    trend_x, trend_y = _rank_trend(
                        clean[x_name].to_numpy(), clean[y_name].to_numpy()
                    )
                    ax.plot(trend_x, trend_y, color="#D1495B", lw=1.2)
                    influence = (
                        stats.rankdata(abs(clean[x_name] - clean[x_name].median()))
                        + stats.rankdata(abs(clean[y_name] - clean[y_name].median()))
                    )
                    labels = clean.assign(_influence=influence).nlargest(
                        min(top_n, len(clean)), "_influence"
                    )
                    for _, item in labels.iterrows():
                        ax.annotate(
                            item["pair_name"], (item[x_name], item[y_name]),
                            fontsize=4.5, alpha=0.75, xytext=(2, 2),
                            textcoords="offset points"
                        )
                    ax.text(
                        0.02, 0.98,
                        f"rho={rho:.2f}; pair-level p={p_value:.2g}; n={len(clean):,}",
                        transform=ax.transAxes, va="top", fontsize=7
                    )
                if x_limits and np.isfinite(x_limits).all():
                    ax.set_xlim(*x_limits)
                if y_limits and np.isfinite(y_limits).all():
                    ax.set_ylim(*y_limits)
                if row == 0:
                    ax.set_title(EPOCH_LABELS[epoch], fontsize=10)
                if row == 2:
                    ax.set_xlabel(x_label, fontsize=8)
                if column == 0:
                    ax.set_ylabel(y_label, fontsize=8)
        fig.suptitle(
            f"{subject}, {protocol:g} Hz, "
            f"{'adjacent bipolar' if montage == 'bipolar' else 'common-average'} reference\n"
            "Each point is one electrode pair; labels identify the most influential pairs",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        stem = output_dir / montage / f"{file_name}_pair_scatter"
        _save_figure(fig, stem, dpi)
        paths.extend([stem.with_suffix(".png"), stem.with_suffix(".pdf")])
    return paths


def _interactive_html(group: pd.DataFrame, title: str) -> str:
    payload = []
    for hypothesis, (x_name, y_name, x_label, y_label) in HYPOTHESIS_AXES.items():
        for epoch in EPOCH_ORDER:
            selected = group[group["epoch"] == epoch].copy()
            if y_name == "gpdc_log_ratio":
                selected = selected[
                    selected["pdc_valid"].astype(str).str.lower().eq("true")
                ]
            elif y_name == "tra_unsigned":
                selected = selected[selected["tra_status"].eq("ok")]
            selected = selected.replace([np.inf, -np.inf], np.nan).dropna(
                subset=[x_name, y_name]
            )
            for row in selected.itertuples():
                payload.append({
                    "hypothesis": hypothesis, "epoch": epoch,
                    "x": float(getattr(row, x_name)), "y": float(getattr(row, y_name)),
                    "x_label": x_label, "y_label": y_label,
                    "pair": row.pair_name, "a": row.channel_a, "b": row.channel_b,
                    "region_a": row.region_a, "region_b": row.region_b,
                    "subject": row.subject, "file": row.file,
                    "montage": row.montage,
                })
    data_json = json.dumps(payload, separators=(",", ":"))
    options = "".join(
        f'<option value="{key}">{html.escape(HYPOTHESIS_SHORT_LABELS[key])}</option>'
        for key in HYPOTHESIS_AXES
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font-family:Arial,sans-serif;margin:18px;color:#222}} select{{margin:4px 12px 12px 0}}
#chart{{border:1px solid #bbb;background:white}} .tip{{position:absolute;display:none;
background:#fff;border:1px solid #555;padding:7px;font-size:12px;pointer-events:none;max-width:360px}}
</style></head><body>
<h2>{html.escape(title)}</h2>
<p>Each point is one electrode pair. Hover to identify the pair and regions.
Pair-level p-values are descriptive; group inference uses participant summaries and channel-label permutations.</p>
<label>Relationship <select id="hyp">{options}</select></label>
<label>Epoch <select id="epoch"><option value="pre">Pre</option>
<option value="stimulation">Stimulation 0-4 s</option><option value="post">Post</option></select></label>
<br><svg id="chart" width="1000" height="650"></svg><div id="tip" class="tip"></div>
<script>
const data={data_json}; const svg=document.getElementById('chart'),tip=document.getElementById('tip');
const NS='http://www.w3.org/2000/svg';
function el(n,a){{let e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);return e}}
function draw(){{
 svg.innerHTML=''; const h=document.getElementById('hyp').value,ep=document.getElementById('epoch').value;
 const d=data.filter(x=>x.hypothesis===h&&x.epoch===ep); if(!d.length)return;
 const W=1000,H=650,m={{l:90,r:30,t:35,b:75}};
 let xs=d.map(x=>x.x),ys=d.map(x=>x.y),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
 let xp=x=>m.l+(x-xmin)/(xmax-xmin||1)*(W-m.l-m.r),yp=y=>H-m.b-(y-ymin)/(ymax-ymin||1)*(H-m.t-m.b);
 svg.append(el('line',{{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,stroke:'#222'}}));
 svg.append(el('line',{{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,stroke:'#222'}}));
 let xt=el('text',{{x:W/2,y:H-22,'text-anchor':'middle','font-size':16}});xt.textContent=d[0].x_label;svg.append(xt);
 let yt=el('text',{{x:20,y:H/2,'text-anchor':'middle','font-size':16,transform:`rotate(-90 20 ${{H/2}})`}});yt.textContent=d[0].y_label;svg.append(yt);
 d.forEach(p=>{{let c=el('circle',{{cx:xp(p.x),cy:yp(p.y),r:4,fill:'#31688E','fill-opacity':.45}});
 c.onmousemove=e=>{{tip.style.display='block';tip.style.left=(e.pageX+12)+'px';tip.style.top=(e.pageY+12)+'px';
 tip.innerHTML=`<b>${{p.pair}}</b><br>${{p.a}} (${{p.region_a}}) - ${{p.b}} (${{p.region_b}})<br>${{p.x_label}}: ${{p.x.toPrecision(5)}}<br>${{p.y_label}}: ${{p.y.toPrecision(5)}}<br>${{p.subject}}; ${{p.epoch}}; ${{p.montage}}`;}};c.onmouseout=()=>tip.style.display='none';svg.append(c)}})
}}
document.getElementById('hyp').onchange=draw;document.getElementById('epoch').onchange=draw;draw();
</script></body></html>"""


def make_interactive_scatter(
    pairs: pd.DataFrame, output_dir: Path
) -> list[Path]:
    paths = []
    for (montage, file_name), group in pairs.groupby(["montage", "file"]):
        title = (
            f"{group['subject'].iloc[0]}, {group['stim_frequency_hz'].iloc[0]:g} Hz, "
            f"{montage} pair-level waveform-connectivity explorer"
        )
        path = output_dir / montage / f"{file_name}_interactive.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_interactive_html(group, title), encoding="utf-8")
        paths.append(path)
    return paths


def make_subject_rho_figures(
    subject: pd.DataFrame, inference: pd.DataFrame, cfg: dict, output_dir: Path
) -> list[Path]:
    paths = []
    dpi = int(cfg["analysis"]["figure_dpi"])
    colors = {"pre": "#457B9D", "stimulation": "#E76F51", "post": "#2A9D8F"}
    for montage in ("bipolar", "car"):
        for hypothesis in HYPOTHESIS_AXES:
            selected = subject[
                (subject["montage"] == montage)
                & (subject["hypothesis"] == hypothesis)
            ]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
            for ax, protocol in zip(axes, (1.0, 50.0)):
                protocol_data = selected[selected["stim_frequency_hz"] == protocol]
                for subject_name, group in protocol_data.groupby("subject"):
                    values = group.set_index("epoch")["mean_run_rho"].reindex(EPOCH_ORDER)
                    ax.plot(
                        range(3), values, color="0.65", lw=0.8, alpha=0.7
                    )
                    ax.scatter(
                        range(3), values,
                        c=[colors[epoch] for epoch in EPOCH_ORDER],
                        s=24, zorder=3
                    )
                ax.axhline(0, color="0.5", lw=0.8, ls="--")
                ax.set_xticks(range(3), ["Pre", "Stim.", "Post"])
                ax.set_title(f"{protocol:g} Hz")
                if ax is axes[0]:
                    ax.set_ylabel("Participant Spearman rho")
                result = inference[
                    (inference["montage"] == montage)
                    & (inference["stim_frequency_hz"] == protocol)
                    & (inference["hypothesis"] == hypothesis)
                ]
                text = "\n".join(
                    f"{row.contrast.replace('_', ' ')}: p={row.channel_label_permutation_p:.3g}"
                    for row in result.itertuples()
                )
                ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=6, va="bottom")
            fig.suptitle(
                f"{HYPOTHESIS_SHORT_LABELS[hypothesis]}\n"
                f"{'Adjacent bipolar' if montage == 'bipolar' else 'Common-average'} reference"
            )
            fig.tight_layout(rect=(0, 0, 1, 0.9))
            stem = output_dir / f"{montage}_{hypothesis}_subject_rho"
            _save_figure(fig, stem, dpi)
            paths.extend([stem.with_suffix(".png"), stem.with_suffix(".pdf")])
    return paths


def _network_panel(
    ax: plt.Axes,
    pair_group: pd.DataFrame,
    feature_group: pd.DataFrame,
    edge_value: str,
    title: str,
    percentile: float,
    *,
    signed_edges: bool = False,
):
    nodes = feature_group.drop_duplicates("channel").set_index("channel")
    valid = pair_group.dropna(subset=[edge_value, "x_a", "y_a", "x_b", "y_b"])
    if valid.empty or nodes.empty:
        ax.text(0.5, 0.5, "No eligible network", transform=ax.transAxes, ha="center")
        return
    threshold = np.nanpercentile(np.abs(valid[edge_value]), percentile)
    edges = valid[np.abs(valid[edge_value]) >= threshold]
    maximum = max(np.abs(edges[edge_value]).max(), 1e-12)
    for row in edges.itertuples():
        value = getattr(row, edge_value)
        edge_color = (
            ("#D1495B" if value >= 0 else "#31688E")
            if signed_edges else "#777777"
        )
        ax.plot(
            [row.x_a, row.x_b], [row.y_a, row.y_b],
            color=edge_color,
            alpha=0.35, lw=0.4 + 2.2 * abs(value) / maximum,
        )
    node_scatter = ax.scatter(
        nodes["x_mni"], nodes["y_mni"], c=nodes["mean_if"],
        cmap="viridis", s=18, edgecolor="white", linewidth=0.25, zorder=3
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("MNI x (mm)")
    ax.set_ylabel("MNI y (mm)")
    ax.set_aspect("equal", adjustable="datalim")
    return node_scatter


def make_network_figures(
    pairs: pd.DataFrame, features: pd.DataFrame, cfg: dict, output_dir: Path
) -> tuple[list[Path], pd.DataFrame]:
    paths, manifest = [], []
    dpi = int(cfg["analysis"]["figure_dpi"])
    percentile = float(cfg["analysis"]["network_edge_percentile"])
    run_counts = (
        features.groupby(["montage", "stim_frequency_hz", "file"], as_index=False)
        ["channel"].nunique().rename(columns={"channel": "n_nodes"})
    )
    representatives = []
    for _, group in run_counts.groupby(["montage", "stim_frequency_hz"]):
        median = group["n_nodes"].median()
        representatives.append(
            group.iloc[(group["n_nodes"] - median).abs().argsort().iloc[0]]
        )
    for representative in representatives:
        montage, protocol, file_name = (
            representative["montage"],
            representative["stim_frequency_hz"],
            representative["file"],
        )
        pair_run = pairs[
            (pairs["montage"] == montage) & (pairs["file"] == file_name)
        ]
        feature_run = features[
            (features["montage"] == montage) & (features["file"] == file_name)
        ]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
        node_scatter = None
        for ax, epoch in zip(axes, EPOCH_ORDER):
            node_scatter = _network_panel(
                ax,
                pair_run[pair_run["epoch"] == epoch],
                feature_run[feature_run["epoch"] == epoch],
                "abs_imcoh",
                EPOCH_LABELS[epoch],
                percentile,
                signed_edges=False,
            )
        fig.suptitle(
            f"Representative implanted-network state: {file_name}\n"
            f"Top {100 - percentile:g}% of absolute imaginary-coherence edges; "
            "gray edge width indicates connection strength"
        )
        if node_scatter is not None:
            fig.colorbar(
                node_scatter, ax=axes.tolist(), shrink=0.72, pad=0.02,
                label="IMF4 mean instantaneous frequency (Hz)"
            )
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        stem = output_dir / f"{montage}_{protocol:g}hz_network_state"
        _save_figure(fig, stem, dpi)
        paths.extend([stem.with_suffix(".png"), stem.with_suffix(".pdf")])

        pivot = pair_run.pivot_table(
            index=[
                "channel_a", "channel_b", "pair_name", "x_a", "y_a", "z_a",
                "x_b", "y_b", "z_b",
            ],
            columns="epoch", values="abs_imcoh", aggfunc="first"
        ).reset_index()
        pivot["stimulation_minus_pre"] = pivot["stimulation"] - pivot["pre"]
        pivot["post_minus_pre"] = pivot["post"] - pivot["pre"]
        node_post = feature_run[feature_run["epoch"] == "post"]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
        node_scatter = None
        for ax, contrast in zip(axes, ("stimulation_minus_pre", "post_minus_pre")):
            node_scatter = _network_panel(
                ax, pivot, node_post, contrast,
                contrast.replace("_", " ").title(), percentile,
                signed_edges=True,
            )
        fig.suptitle(
            f"Representative implanted-network change: {file_name}\n"
            "Red indicates increased and blue decreased absolute imaginary coherence; "
            "edge width indicates absolute change"
        )
        if node_scatter is not None:
            fig.colorbar(
                node_scatter, ax=axes.tolist(), shrink=0.72, pad=0.02,
                label="Post-stimulation IMF4 mean instantaneous frequency (Hz)"
            )
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        stem = output_dir / f"{montage}_{protocol:g}hz_network_change"
        _save_figure(fig, stem, dpi)
        paths.extend([stem.with_suffix(".png"), stem.with_suffix(".pdf")])
        manifest.append({
            "montage": montage, "stim_frequency_hz": protocol,
            "file": file_name, "n_nodes": representative["n_nodes"],
            "selection_rule": "run nearest montage-by-protocol median eligible node count",
            "edge_display_rule": f"top {100 - percentile:g}% by absolute edge value",
        })
    return paths, pd.DataFrame(manifest)


def generate_reports(
    cfg: dict, features: pd.DataFrame, pairs: pd.DataFrame,
    subject: pd.DataFrame, inference: pd.DataFrame
) -> dict[str, Path]:
    root = Path(cfg["project"]["output_dir"])
    static = root / "figures" / "static"
    interactive = root / "figures" / "interactive"
    outputs = {}
    make_timing_figure(static, int(cfg["analysis"]["figure_dpi"]))
    make_pair_scatter_figures(pairs, cfg, static / "pair_scatter")
    make_interactive_scatter(pairs, interactive)
    make_subject_rho_figures(subject, inference, cfg, static / "subject_rho")
    _, manifest = make_network_figures(
        pairs, features, cfg, static / "networks"
    )
    manifest_path = root / "tables" / "network_figure_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    outputs["static_figures"] = static
    outputs["interactive_figures"] = interactive
    outputs["network_manifest"] = manifest_path
    return outputs


def run_three_epoch_extension(
    config_path: str | Path,
    montage: str | None = None,
    file_path: str | Path | None = None,
    max_files: int | None = None,
    report_only: bool = False,
) -> dict[str, Path]:
    config_path = Path(config_path)
    cfg = load_config(config_path)
    root = Path(cfg["project"]["output_dir"])
    tables, cache = root / "tables", root / "run_cache"
    tables.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    if report_only:
        features = pd.read_csv(tables / "three_epoch_channel_features.csv")
        pairs = pd.read_csv(tables / "three_epoch_pair_table.csv", low_memory=False)
        subject = pd.read_csv(tables / "three_epoch_subject_correlations.csv")
        inference = pd.read_csv(tables / "three_epoch_correlation_change_inference.csv")
        return generate_reports(cfg, features, pairs, subject, inference)

    source_root = Path(cfg["project"]["source_output_dir"])
    source_features = pd.read_csv(
        source_root / "tables" / "channel_epoch_imf4_features.csv"
    )
    source_pairs = pd.read_csv(
        source_root / "tables" / "all_pairs_connectivity.csv", low_memory=False
    )
    files = [Path(file_path)] if file_path else list(
        iter_mat_files(cfg["project"]["data_dir"])
    )
    excluded = set(cfg["analysis"].get("excluded_files", []))
    files = [path for path in files if path.name not in excluded]
    files.sort(key=lambda path: (path.stat().st_size, path.name))
    if max_files is not None:
        files = files[:max_files]
    active_montages = [montage] if montage else cfg["analysis"]["montages"]
    selected_stems = {path.stem for path in files}
    source_features = source_features[
        source_features["file"].isin(selected_stems)
        & source_features["montage"].isin(active_montages)
    ].copy()
    source_pairs = source_pairs[
        source_pairs["file"].isin(selected_stems)
        & source_pairs["montage"].isin(active_montages)
    ].copy()
    coordinates = _coordinate_table(source_features, cfg)
    source_pairs = enrich_existing_pairs(source_pairs, coordinates)
    source_features = source_features.merge(
        coordinates,
        on=["file", "montage", "channel"],
        how="left",
        suffixes=("", "_coordinate"),
    )
    feature_frames, pair_frames, qc_frames = [], [], []
    for active_montage in active_montages:
        base_cfg = load_config(ROOT_CONFIGS[active_montage])
        for path in files:
            key = hashlib.sha1(
                f"{CACHE_VERSION}|{active_montage}|{path.resolve()}".encode()
            ).hexdigest()[:12]
            cache_paths = {
                name: cache / f"{path.stem}_{active_montage}_{key}_{name}.csv"
                for name in ("features", "pairs", "qc")
            }
            if all(path.exists() for path in cache_paths.values()):
                features, pairs, qc = (
                    _read_cached_csv(cache_paths[name])
                    for name in ("features", "pairs", "qc")
                )
            else:
                features, pairs, qc = analyze_stimulation_run(
                    path, active_montage, base_cfg, cfg
                )
                for name, frame in (
                    ("features", features), ("pairs", pairs), ("qc", qc)
                ):
                    frame.to_csv(cache_paths[name], index=False)
            feature_frames.append(features)
            pair_frames.append(pairs)
            qc_frames.append(qc)
            print(
                f"{active_montage}: {path.name}: "
                f"stimulation_nodes={len(features)}, pair_rows={len(pairs)}",
                flush=True,
            )
    stimulation_features = pd.concat(feature_frames, ignore_index=True)
    stimulation_pairs = pd.concat(pair_frames, ignore_index=True)
    stimulation_qc = pd.concat(qc_frames, ignore_index=True)
    pair_keys = ["montage", "file", "channel_a", "channel_b"]
    eligible_pairs = stimulation_pairs[pair_keys].drop_duplicates()
    source_pairs = source_pairs.merge(eligible_pairs, on=pair_keys, how="inner")

    feature_keys = ["montage", "file", "channel"]
    eligible_features = stimulation_features[feature_keys].drop_duplicates()
    source_features = source_features.merge(eligible_features, on=feature_keys, how="inner")

    features = pd.concat(
        [source_features, stimulation_features], ignore_index=True, sort=False
    )
    pairs = pd.concat(
        [source_pairs, stimulation_pairs], ignore_index=True, sort=False
    )
    run = run_correlations(pairs, cfg)
    subject = subject_correlations(run)
    changes, inference = correlation_change_inference(
        subject, pairs, cfg
    )
    outputs = {
        "three_epoch_channel_features": features,
        "three_epoch_pair_table": pairs,
        "stimulation_epoch_qc": stimulation_qc,
        "three_epoch_run_correlations": run,
        "three_epoch_subject_correlations": subject,
        "three_epoch_subject_correlation_changes": changes,
        "three_epoch_correlation_change_inference": inference,
    }
    paths = {}
    for name, frame in outputs.items():
        path = tables / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    paths.update(generate_reports(cfg, features, pairs, subject, inference))
    return paths
