from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    "imf4_mean_if": {
        "label": "IMF4 mean instantaneous frequency",
        "unit": "Hz",
        "delta_label": "Post - pre IMF4 mean IF (Hz)",
    },
    "imf4_asc_desc_ratio": {
        "label": "IMF4 ascent/descent duration ratio",
        "unit": "ascent/descent ratio",
        "delta_label": "Post - pre ascent/descent ratio",
    },
    "imf4_peak_trough_ratio": {
        "label": "IMF4 peak/trough duration ratio",
        "unit": "peak/trough ratio",
        "delta_label": "Post - pre peak/trough ratio",
    },
    "imf4_abs_imcoh": {
        "label": "IMF4 seed-to-target absolute imaginary coherence",
        "unit": "absolute ImCoh",
        "delta_label": "Post - pre seed-to-target absolute ImCoh",
    },
    "imf4_remote_abs_imcoh": {
        "label": "IMF4 non-stimulated network absolute imaginary coherence",
        "unit": "absolute ImCoh",
        "delta_label": "Post - pre non-stimulated network absolute ImCoh",
    },
    "imf4_waveform_imcoh_coupling": {
        "label": "Waveform-connectivity coupling",
        "unit": "Spearman rho",
        "delta_label": "Within-run Spearman rho",
    },
}
COLORS = {1.0: "#2A9D8F", 50.0: "#D62828"}
MARKERS = {"Primary adjacent bipolar": "o", "CAR sensitivity": "D"}


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


def _load(reference: str, root: Path) -> dict[str, pd.DataFrame]:
    tables = root / "tables"
    return {
        "waveform": pd.read_csv(tables / "waveform_nodes.csv", low_memory=False),
        "connectivity": pd.read_csv(tables / "connectivity_edges.csv", low_memory=False),
        "coupling": pd.read_csv(tables / "waveform_connectivity_coupling.csv", low_memory=False),
    }


def _subject_prepost_waveform_metric(reference: str, waveform: pd.DataFrame, endpoint: str, pre_col: str, post_col: str) -> pd.DataFrame:
    clean = waveform.loc[waveform["analysis_include"].fillna(False).astype(bool)].copy()
    run = (
        clean.groupby(["file", "subject", "stim_frequency_hz"], as_index=False)
        .agg(pre_value=(pre_col, "mean"), post_value=(post_col, "mean"), n_sites=("channel", "nunique"))
    )
    subject = (
        run.groupby(["subject", "stim_frequency_hz"], as_index=False)
        .agg(pre_value=("pre_value", "mean"), post_value=("post_value", "mean"), n_runs=("file", "nunique"), n_sites=("n_sites", "sum"))
    )
    subject["delta"] = subject["post_value"] - subject["pre_value"]
    subject["analysis_reference"] = reference
    subject["endpoint"] = endpoint
    return subject


def _subject_prepost_waveform(reference: str, waveform: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            _subject_prepost_waveform_metric(reference, waveform, "imf4_mean_if", "pre_mean_if", "post_mean_if"),
            _subject_prepost_waveform_metric(reference, waveform, "imf4_asc_desc_ratio", "pre_asc2desc", "post_asc2desc"),
            _subject_prepost_waveform_metric(
                reference, waveform, "imf4_peak_trough_ratio", "pre_peak2trough", "post_peak2trough"
            ),
        ],
        ignore_index=True,
    )


def _subject_prepost_connectivity(reference: str, connectivity: pd.DataFrame, edge_type: str, endpoint: str) -> pd.DataFrame:
    clean = connectivity.loc[
        connectivity["imf"].eq(4)
        & connectivity["edge_type"].eq(edge_type)
        & connectivity["analysis_include"].fillna(False).astype(bool)
    ].copy()
    run = (
        clean.groupby(["file", "subject", "stim_frequency_hz"], as_index=False)
        .agg(pre_value=("pre_abs_imcoh", "mean"), post_value=("post_abs_imcoh", "mean"), n_edges=("delta_abs_imcoh", "size"))
    )
    subject = (
        run.groupby(["subject", "stim_frequency_hz"], as_index=False)
        .agg(pre_value=("pre_value", "mean"), post_value=("post_value", "mean"), n_runs=("file", "nunique"), n_edges=("n_edges", "sum"))
    )
    subject["delta"] = subject["post_value"] - subject["pre_value"]
    subject["analysis_reference"] = reference
    subject["endpoint"] = endpoint
    return subject


def _subject_coupling(reference: str, coupling: pd.DataFrame) -> pd.DataFrame:
    subject = (
        coupling.dropna(subset=["spearman_rho"])
        .groupby(["subject", "stim_frequency_hz"], as_index=False)
        .agg(delta=("spearman_rho", "mean"), n_runs=("file", "nunique"), n_edges=("n_edges", "sum"))
    )
    subject["pre_value"] = np.nan
    subject["post_value"] = np.nan
    subject["analysis_reference"] = reference
    subject["endpoint"] = "imf4_waveform_imcoh_coupling"
    return subject


def build_subject_values() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for reference, root in ANALYSES.items():
        tables = _load(reference, root)
        frames.append(_subject_prepost_waveform(reference, tables["waveform"]))
        frames.append(_subject_prepost_connectivity(reference, tables["connectivity"], "seed_to_target", "imf4_abs_imcoh"))
        frames.append(_subject_prepost_connectivity(reference, tables["connectivity"], "remote_to_remote", "imf4_remote_abs_imcoh"))
        frames.append(_subject_coupling(reference, tables["coupling"]))
    return pd.concat(frames, ignore_index=True, sort=False)


def prepost_stats(values: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (reference, endpoint, freq), group in values.groupby(["analysis_reference", "endpoint", "stim_frequency_hz"]):
        delta = group["delta"].dropna()
        rows.append(
            {
                "analysis_reference": reference,
                "endpoint": endpoint,
                "endpoint_label": ENDPOINTS[endpoint]["label"],
                "comparison": f"{freq:g} Hz post minus pre",
                "stim_frequency_hz": float(freq),
                "n_subjects": int(group["subject"].nunique()),
                "mean_delta": float(delta.mean()) if len(delta) else np.nan,
                "median_delta": float(delta.median()) if len(delta) else np.nan,
                "sem_delta": _sem(delta),
                "exact_signflip_p": _exact_signflip(delta),
            }
        )
    return pd.DataFrame(rows)


def protocol_contrasts(values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    contrast_rows: list[dict] = []
    stats_rows: list[dict] = []
    for (reference, endpoint), group in values.groupby(["analysis_reference", "endpoint"]):
        pivot = group.pivot_table(index="subject", columns="stim_frequency_hz", values="delta", aggfunc="mean")
        if 1.0 not in pivot.columns or 50.0 not in pivot.columns:
            diffs = pd.Series(dtype=float)
        else:
            diffs = (pivot[50.0] - pivot[1.0]).dropna()
            for subject, value in diffs.items():
                contrast_rows.append(
                    {
                        "analysis_reference": reference,
                        "endpoint": endpoint,
                        "endpoint_label": ENDPOINTS[endpoint]["label"],
                        "subject": subject,
                        "contrast": "50 Hz minus 1 Hz",
                        "difference": float(value),
                        "value_1hz": float(pivot.loc[subject, 1.0]),
                        "value_50hz": float(pivot.loc[subject, 50.0]),
                    }
                )
        stats_rows.append(
            {
                "analysis_reference": reference,
                "endpoint": endpoint,
                "endpoint_label": ENDPOINTS[endpoint]["label"],
                "contrast": "50 Hz minus 1 Hz",
                "n_paired_subjects": int(len(diffs)),
                "mean_difference": float(diffs.mean()) if len(diffs) else np.nan,
                "median_difference": float(diffs.median()) if len(diffs) else np.nan,
                "sem_difference": _sem(diffs),
                "exact_signflip_p": _exact_signflip(diffs),
            }
        )
    return pd.DataFrame(contrast_rows), pd.DataFrame(stats_rows)


def _panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")


def _paired_prepost_panel(ax: plt.Axes, values: pd.DataFrame, reference: str, endpoint: str) -> None:
    data = values.loc[values["analysis_reference"].eq(reference) & values["endpoint"].eq(endpoint)].copy()
    rng = np.random.default_rng(307)
    for freq, x0, x1 in [(1.0, 0.0, 0.65), (50.0, 1.35, 2.0)]:
        sub = data.loc[data["stim_frequency_hz"].eq(freq)].dropna(subset=["pre_value", "post_value"])
        color = COLORS[freq]
        for xpos, col in [(x0, "pre_value"), (x1, "post_value")]:
            vals = sub[col].dropna().astype(float)
            if len(vals):
                parts = ax.violinplot([vals.to_numpy()], positions=[xpos], widths=0.42, showextrema=False)
                for body in parts["bodies"]:
                    body.set_facecolor(color)
                    body.set_edgecolor(color)
                    body.set_alpha(0.20)
                jitter = rng.normal(0, 0.018, len(vals))
                ax.scatter(
                    np.repeat(xpos, len(vals)) + jitter,
                    vals,
                    color=color,
                    s=24,
                    alpha=0.78,
                    edgecolor="white",
                    lw=0.35,
                    zorder=3,
                )
                ax.scatter([xpos], [vals.mean()], color=color, marker="D", s=58, edgecolor="white", zorder=4)
    ax.set_xticks([0.0, 0.65, 1.35, 2.0], ["1 Hz\npre", "1 Hz\npost", "50 Hz\npre", "50 Hz\npost"])
    ax.set_title(reference)
    ax.set_ylabel(ENDPOINTS[endpoint]["unit"])
    ax.grid(axis="y", color="0.9", lw=0.45)


def _protocol_contrast_panel(ax: plt.Axes, values: pd.DataFrame, endpoint: str) -> None:
    subset = values.loc[values["endpoint"].eq(endpoint)].copy()
    offsets = {"Primary adjacent bipolar": -0.06, "CAR sensitivity": 0.06}
    for reference, group in subset.groupby("analysis_reference"):
        marker = MARKERS[reference]
        for freq in (1.0, 50.0):
            vals = group.loc[group["stim_frequency_hz"].eq(freq), "delta"].dropna()
            xpos = freq + offsets[reference]
            if len(vals):
                parts = ax.violinplot([vals.to_numpy()], positions=[xpos], widths=0.38, showextrema=False)
                for body in parts["bodies"]:
                    body.set_facecolor(COLORS[freq])
                    body.set_edgecolor(COLORS[freq])
                    body.set_alpha(0.13)
            jitter = np.linspace(-0.018, 0.018, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(np.repeat(xpos, len(vals)) + jitter, vals, color=COLORS[freq], marker=marker, s=34, edgecolor="white", lw=0.35, alpha=0.82, label=reference if freq == 1.0 else None)
            if len(vals):
                ax.scatter([xpos], [vals.mean()], color=COLORS[freq], marker="D", s=64, edgecolor="white", zorder=4)
                ax.text(
                    xpos,
                    0.97,
                    f"n={len(vals)}\np={_exact_signflip(vals):.3f}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=5.8,
                    color=COLORS[freq],
                )
    ax.axhline(0, color="0.35", lw=0.8, ls="--")
    ax.set_xticks([1.0, 50.0], ["1 Hz", "50 Hz"])
    ax.set_ylabel(ENDPOINTS[endpoint]["delta_label"])
    ax.set_title(ENDPOINTS[endpoint]["label"])
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", color="0.9", lw=0.45)


def _delta_violin_panel(ax: plt.Axes, values: pd.DataFrame, endpoint: str, reference: str | None = None) -> None:
    data = values.loc[values["endpoint"].eq(endpoint)].copy()
    if reference is not None:
        data = data.loc[data["analysis_reference"].eq(reference)].copy()
    offsets = {"Primary adjacent bipolar": -0.07, "CAR sensitivity": 0.07}
    rng = np.random.default_rng(29)
    for ref, group in data.groupby("analysis_reference"):
        marker = MARKERS[ref]
        for freq in (1.0, 50.0):
            vals = group.loc[group["stim_frequency_hz"].eq(freq), "delta"].dropna().astype(float)
            xpos = freq if reference is not None else freq + offsets[ref]
            if len(vals):
                parts = ax.violinplot([vals.to_numpy()], positions=[xpos], widths=0.62, showextrema=False)
                for body in parts["bodies"]:
                    body.set_facecolor(COLORS[freq])
                    body.set_edgecolor(COLORS[freq])
                    body.set_alpha(0.14 if reference is None else 0.22)
                jitter = rng.normal(0, 0.025, len(vals))
                ax.scatter(
                    np.repeat(xpos, len(vals)) + jitter,
                    vals,
                    color=COLORS[freq],
                    marker=marker,
                    s=30,
                    edgecolor="white",
                    lw=0.35,
                    alpha=0.82,
                    label=ref if freq == 1.0 and reference is None else None,
                )
                ax.scatter([xpos], [vals.mean()], marker="D", s=58, color=COLORS[freq], edgecolor="white", zorder=4)
                ax.text(
                    xpos,
                    0.97,
                    f"n={len(vals)}\np={_exact_signflip(vals):.3f}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=5.8,
                    color=COLORS[freq],
                )
    ax.axhline(0, color="0.35", lw=0.8, ls="--")
    ax.set_xticks([1.0, 50.0], ["1 Hz", "50 Hz"])
    ax.set_ylabel(ENDPOINTS[endpoint]["delta_label"])
    ax.set_title(ENDPOINTS[endpoint]["label"] if reference is None else reference)
    ax.grid(axis="y", color="0.9", lw=0.45)
    if reference is None:
        ax.legend(frameon=False, fontsize=7)


def _reference_comparison_panel(ax: plt.Axes, values: pd.DataFrame, endpoint: str, freq: float) -> None:
    data = values.loc[values["endpoint"].eq(endpoint) & values["stim_frequency_hz"].eq(freq)].copy()
    paired = data.pivot_table(index="subject", columns="analysis_reference", values="delta", aggfunc="mean")
    labels = list(ANALYSES)
    x = np.arange(len(labels), dtype=float)
    rng = np.random.default_rng(511)
    for xpos, label in zip(x, labels):
        vals = paired[label].dropna().astype(float) if label in paired else pd.Series(dtype=float)
        if len(vals):
            parts = ax.violinplot([vals.to_numpy()], positions=[xpos], widths=0.55, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(COLORS[freq])
                body.set_edgecolor(COLORS[freq])
                body.set_alpha(0.20)
            jitter = rng.normal(0, 0.025, len(vals))
            ax.scatter(
                np.repeat(xpos, len(vals)) + jitter,
                vals,
                color=COLORS[freq],
                s=28,
                edgecolor="white",
                lw=0.35,
                alpha=0.82,
            )
            ax.text(
                xpos,
                0.97,
                f"n={len(vals)}\np={_exact_signflip(vals):.3f}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=6,
                color=COLORS[freq],
            )
    means = [paired[label].mean() if label in paired else np.nan for label in labels]
    ax.scatter(x, means, marker="D", s=70, color=COLORS[freq], edgecolor="white", zorder=4)
    ax.axhline(0, color="0.35", lw=0.8, ls="--")
    ax.set_xticks(x, ["Bipolar", "CAR"])
    ax.set_ylabel(ENDPOINTS[endpoint]["delta_label"])
    ax.set_title(f"{ENDPOINTS[endpoint]['label']}\n{freq:g} Hz")
    ax.grid(axis="y", color="0.9", lw=0.45)


def make_figures(values: pd.DataFrame, out_dir: Path) -> list[Path]:
    set_publication_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.0), constrained_layout=True)
    for col, endpoint in enumerate(["imf4_mean_if", "imf4_asc_desc_ratio", "imf4_peak_trough_ratio"]):
        for row, reference in enumerate(ANALYSES):
            _paired_prepost_panel(axes[row, col], values, reference, endpoint)
            if row == 0:
                axes[row, col].set_title(f"{ENDPOINTS[endpoint]['label']}\n{reference}")
            else:
                axes[row, col].set_title(reference)
    for ax, letter in zip(axes.ravel(), "abcdef"):
        _panel(ax, letter)
    fig.suptitle("Within-subject pre/post waveform dynamics by stimulation protocol", y=1.02)
    path = out_dir / "figure_01_within_subject_prepost_waveform_endpoints.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.0), constrained_layout=True)
    for col, endpoint in enumerate(["imf4_abs_imcoh", "imf4_remote_abs_imcoh"]):
        for row, reference in enumerate(ANALYSES):
            _paired_prepost_panel(axes[row, col], values, reference, endpoint)
            if row == 0:
                axes[row, col].set_title(f"{ENDPOINTS[endpoint]['label']}\n{reference}")
            else:
                axes[row, col].set_title(reference)
    for ax, letter in zip(axes.ravel(), "abcd"):
        _panel(ax, letter)
    fig.suptitle("Within-subject pre/post imaginary-coherence endpoints", y=1.02)
    path = out_dir / "figure_02_within_subject_prepost_connectivity_endpoints.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.0), constrained_layout=True)
    for ax, endpoint, letter in zip(
        axes.ravel(),
        [
            "imf4_mean_if",
            "imf4_asc_desc_ratio",
            "imf4_peak_trough_ratio",
            "imf4_abs_imcoh",
            "imf4_remote_abs_imcoh",
            "imf4_waveform_imcoh_coupling",
        ],
        "abcdef",
    ):
        _protocol_contrast_panel(ax, values, endpoint)
        _panel(ax, letter)
    fig.suptitle("Within-subject post-minus-pre changes by protocol and reference", y=1.02)
    path = out_dir / "figure_03_within_subject_delta_summaries_all_endpoints.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    for reference, safe_name in [
        ("Primary adjacent bipolar", "bipolar"),
        ("CAR sensitivity", "car"),
    ]:
        fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.0), constrained_layout=True)
        for ax, endpoint, letter in zip(
            axes.ravel(),
            [
                "imf4_mean_if",
                "imf4_asc_desc_ratio",
                "imf4_peak_trough_ratio",
                "imf4_abs_imcoh",
                "imf4_remote_abs_imcoh",
                "imf4_waveform_imcoh_coupling",
            ],
            "abcdef",
        ):
            _delta_violin_panel(ax, values, endpoint, reference=reference)
            _panel(ax, letter)
        fig.suptitle(f"Within-subject post-minus-pre changes: {reference}", y=1.02)
        path = out_dir / f"figure_03_{safe_name}_within_subject_delta_summaries_all_endpoints.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.0), constrained_layout=True)
    for ax, endpoint, letter in zip(
        axes.ravel(),
        [
            "imf4_mean_if",
            "imf4_asc_desc_ratio",
            "imf4_peak_trough_ratio",
            "imf4_abs_imcoh",
            "imf4_remote_abs_imcoh",
            "imf4_waveform_imcoh_coupling",
        ],
        "abcdef",
    ):
        _delta_violin_panel(ax, values, endpoint)
        _panel(ax, letter)
    fig.suptitle("Subject-level post-minus-pre changes: primary reference and CAR sensitivity", y=1.02)
    path = out_dir / "figure_04_within_subject_violin_summary_all_endpoints.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    for reference, safe_name in [
        ("Primary adjacent bipolar", "bipolar"),
        ("CAR sensitivity", "car"),
    ]:
        fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.0), constrained_layout=True)
        for ax, endpoint, letter in zip(
            axes.ravel(),
            [
                "imf4_mean_if",
                "imf4_asc_desc_ratio",
                "imf4_peak_trough_ratio",
                "imf4_abs_imcoh",
                "imf4_remote_abs_imcoh",
                "imf4_waveform_imcoh_coupling",
            ],
            "abcdef",
        ):
            _delta_violin_panel(ax, values, endpoint, reference=reference)
            _panel(ax, letter)
        fig.suptitle(f"Subject-level post-minus-pre changes: {reference}", y=1.02)
        path = out_dir / f"figure_04_{safe_name}_within_subject_violin_summary_all_endpoints.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.0), constrained_layout=True)
    for ax, endpoint, letter in zip(
        axes.ravel(),
        [
            "imf4_mean_if",
            "imf4_asc_desc_ratio",
            "imf4_peak_trough_ratio",
            "imf4_abs_imcoh",
            "imf4_remote_abs_imcoh",
            "imf4_waveform_imcoh_coupling",
        ],
        "abcdef",
    ):
        _reference_comparison_panel(ax, values, endpoint, 1.0)
        _panel(ax, letter)
    fig.suptitle("Within-subject reference comparison for 1 Hz post-minus-pre changes", y=1.02)
    path = out_dir / "figure_05_within_subject_reference_comparison_1hz.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.0), constrained_layout=True)
    for ax, endpoint, letter in zip(
        axes.ravel(),
        [
            "imf4_mean_if",
            "imf4_asc_desc_ratio",
            "imf4_peak_trough_ratio",
            "imf4_abs_imcoh",
            "imf4_remote_abs_imcoh",
            "imf4_waveform_imcoh_coupling",
        ],
        "abcdef",
    ):
        _reference_comparison_panel(ax, values, endpoint, 50.0)
        _panel(ax, letter)
    fig.suptitle("Within-subject reference comparison for 50 Hz post-minus-pre changes", y=1.02)
    path = out_dir / "figure_06_within_subject_reference_comparison_50hz.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def write_summary(stats_prepost: pd.DataFrame, stats_contrast: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "Within-subject SEEG stimulation analysis",
        "========================================",
        "",
        "Design:",
        "- Each endpoint was summarized within run first, then averaged within subject x protocol.",
        "- Pre/post tests use subject-level post-minus-pre change scores.",
        "- 50 Hz minus 1 Hz contrasts use only subjects contributing both protocols.",
        "- Exact sign-flip tests are reported; contacts and edges are not treated as independent observations.",
        "",
        "Pre/post subject-level tests:",
    ]
    for row in stats_prepost.itertuples():
        lines.append(
            f"- {row.analysis_reference}; {row.endpoint_label}; {row.comparison}: "
            f"n={row.n_subjects}, mean={row.mean_delta:.4g}, p={row.exact_signflip_p:.4g}"
        )
    lines.append("")
    lines.append("Paired protocol contrasts:")
    for row in stats_contrast.itertuples():
        lines.append(
            f"- {row.analysis_reference}; {row.endpoint_label}; {row.contrast}: "
            f"n={row.n_paired_subjects}, mean difference={row.mean_difference:.4g}, p={row.exact_signflip_p:.4g}"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out_root = ROOT / "outputs" / "publication_comparison" / "within_subject_analysis"
    table_dir = out_root / "tables"
    figure_dir = out_root / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    values = build_subject_values()
    stats_prepost = prepost_stats(values)
    contrast_values, contrast_stats = protocol_contrasts(values)
    values.to_csv(table_dir / "within_subject_subject_protocol_values.csv", index=False)
    stats_prepost.to_csv(table_dir / "within_subject_prepost_stats.csv", index=False)
    contrast_values.to_csv(table_dir / "within_subject_protocol_contrast_values.csv", index=False)
    contrast_stats.to_csv(table_dir / "within_subject_protocol_contrast_stats.csv", index=False)
    write_summary(stats_prepost, contrast_stats, out_root / "WITHIN_SUBJECT_SUMMARY.txt")
    paths = make_figures(values, figure_dir)
    print(f"subject_values: {table_dir / 'within_subject_subject_protocol_values.csv'}")
    print(f"prepost_stats: {table_dir / 'within_subject_prepost_stats.csv'}")
    print(f"protocol_contrast_values: {table_dir / 'within_subject_protocol_contrast_values.csv'}")
    print(f"protocol_contrast_stats: {table_dir / 'within_subject_protocol_contrast_stats.csv'}")
    print(f"summary: {out_root / 'WITHIN_SUBJECT_SUMMARY.txt'}")
    for path in paths:
        print(f"figure: {path}")


if __name__ == "__main__":
    main()
