from __future__ import annotations

from itertools import combinations
from pathlib import Path
import hashlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import signal, stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.api import VAR

from .config import load_config
from .io import iter_mat_files
from .preprocess import window_mask
from .primary_bipolar_mne import (
    _decompose_window,
    _metric_mean,
    _mne_values,
    _segmented_imfs,
    _share_constituent_contact,
    analysis_windows,
    make_bipolar_run,
)
from .primary_car_mne import make_car_run


FEATURES = ("mean_if", "asc2desc", "peak2trough")
CACHE_VERSION = "all_pairs_imf4_v8"


def _read_cached_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def temporal_irreversibility_unsigned(
    x: np.ndarray, y: np.ndarray, max_lag: int, covariance_epsilon: float = 1e-14
) -> tuple[float, str]:
    """Exact Python translation of invariant_features_bivariate_v2.m, S=0."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    if x.size != y.size or x.size <= max_lag + 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return np.nan, "invalid_input"
    x_sd, y_sd = float(np.std(x, ddof=1)), float(np.std(y, ddof=1))
    if x_sd <= np.finfo(float).eps or y_sd <= np.finfo(float).eps:
        return np.nan, "degenerate_covariance"
    # TRA is invariant to separate positive scaling of A and B. Scale by each
    # SD for numerical conditioning, but do not center: MATLAB xcorr with the
    # supplied implementation operates on the raw Hilbert-envelope levels.
    x = x / x_sd
    y = y / y_sd
    covariance_det = float(np.linalg.det(np.cov(np.column_stack([x, y]), rowvar=False)))
    if not np.isfinite(covariance_det) or covariance_det <= covariance_epsilon:
        return np.nan, "degenerate_covariance"
    # Positive-lag full correlations reproduce the two MATLAB dot products
    # exactly, while the FFT implementation avoids one O(n) loop per lag.
    y_after_x = signal.correlate(y, x, mode="full", method="fft")
    x_after_y = signal.correlate(x, y, mode="full", method="fft")
    center = x.size - 1
    lag_index = np.arange(1, max_lag + 1)
    values = (
        y_after_x[center + lag_index] - x_after_y[center + lag_index]
    ) / (x.size - lag_index)
    return float(np.sqrt(np.mean(values**2) / covariance_det)), "ok"


def _gpdc_from_fit(fit, fs: float, band: tuple[float, float], epsilon: float) -> tuple[float, float]:
    freqs = np.linspace(band[0], band[1], 64)
    variances = np.maximum(np.diag(fit.sigma_u), epsilon)
    a_to_b = np.empty(len(freqs), float)
    b_to_a = np.empty(len(freqs), float)
    for fi, freq in enumerate(freqs):
        af = np.eye(2, dtype=complex)
        for lag, coef in enumerate(fit.coefs, start=1):
            af -= coef * np.exp(-2j * np.pi * freq * lag / fs)
        from_a = np.abs(af[:, 0]) ** 2 / variances
        from_b = np.abs(af[:, 1]) ** 2 / variances
        a_to_b[fi] = from_a[1] / max(from_a.sum(), epsilon)
        b_to_a[fi] = from_b[0] / max(from_b.sum(), epsilon)
    # Matrix convention: target row, source column.
    return float(a_to_b.mean()), float(b_to_a.mean())


def generalized_pdc_pair(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    band: tuple[float, float] = (8.0, 11.0),
    target_fs: float = 128.0,
    max_order: int = 30,
    whiteness_alpha: float = 0.01,
    epsilon: float = 1e-12,
) -> dict:
    """Estimate squared bivariate gPDC A->B and B->A with guarded VAR diagnostics."""
    down = int(round(target_fs))
    up = int(round(fs))
    divisor = np.gcd(down, up)
    data = np.column_stack(
        [
            signal.resample_poly(np.asarray(x, float), down // divisor, up // divisor),
            signal.resample_poly(np.asarray(y, float), down // divisor, up // divisor),
        ]
    )
    data = signal.detrend(data, axis=0)
    data -= data.mean(axis=0)
    data /= np.maximum(data.std(axis=0, ddof=1), epsilon)
    allowed = min(max_order, max(1, data.shape[0] // 20))
    try:
        selected = VAR(data).select_order(maxlags=allowed)
        order = selected.selected_orders.get("aic") or 1
        fit = VAR(data).fit(max(1, min(int(order), allowed)))
    except Exception as exc:
        return {"pdc_valid": False, "pdc_reason": f"var_failed:{exc}"[:120]}
    if not fit.is_stable(verbose=False):
        return {"pdc_valid": False, "pdc_reason": "unstable_var", "pdc_order": fit.k_ar}
    lag = min(20, max(1, len(fit.resid) // 5))
    pvalues = [
        float(acorr_ljungbox(fit.resid[:, column], lags=[lag], return_df=True)["lb_pvalue"].iloc[0])
        for column in range(2)
    ]
    a_to_b, b_to_a = _gpdc_from_fit(fit, target_fs, band, epsilon)
    valid = min(pvalues) > whiteness_alpha
    return {
        "pdc_valid": valid,
        "pdc_reason": "ok" if valid else "residual_autocorrelation",
        "pdc_order": fit.k_ar,
        "pdc_min_whiteness_p": min(pvalues),
        "gpdc_a_to_b": a_to_b,
        "gpdc_b_to_a": b_to_a,
        "gpdc_ratio_a_over_b": (a_to_b + epsilon) / (b_to_a + epsilon),
        "gpdc_log_ratio": np.log((a_to_b + epsilon) / (b_to_a + epsilon)),
    }


def _pair_type(a: pd.Series, b: pd.Series) -> str:
    left, right = bool(a["is_stimulated_channel"]), bool(b["is_stimulated_channel"])
    if left or right:
        return "seed_to_target"
    return "nonstimulated_to_nonstimulated"


def _pair_dynamics(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    envelope_a: np.ndarray,
    envelope_b: np.ndarray,
    fs: float,
    settings: dict,
) -> tuple[dict, float, str]:
    pdc = generalized_pdc_pair(
        signal_a,
        signal_b,
        fs,
        target_fs=settings["pdc_downsample_hz"],
        max_order=settings["pdc_max_order"],
        whiteness_alpha=settings["pdc_residual_whiteness_alpha"],
        epsilon=settings["pdc_epsilon"],
    )
    tra, tra_status = temporal_irreversibility_unsigned(
        envelope_a,
        envelope_b,
        int(round(settings["tra_max_lag_sec"] * fs)),
        settings["tra_covariance_epsilon"],
    )
    return pdc, tra, tra_status


def _prepare_run(path: Path, montage: str, base_cfg: dict):
    if montage == "car":
        run, pre, post = make_car_run(path, base_cfg)
        return run, pre, post
    run = make_bipolar_run(path, base_cfg)
    if run.status != "ok":
        return run, None, None
    pre_range, post_range = analysis_windows(run.recording, base_cfg)
    pre = _decompose_window(run.recording, window_mask(run.recording.time, *pre_range), base_cfg)
    post = _decompose_window(run.recording, window_mask(run.recording.time, *post_range), base_cfg)
    return run, pre, post


def analyze_run(path: Path, montage: str, base_cfg: dict, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run, pre, post = _prepare_run(path, montage, base_cfg)
    if run.status != "ok" or pre is None or post is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame([{"file": path.stem, "montage": montage, "status": run.status}])
    eligible = [
        i for i, node in run.nodes.iterrows()
        if bool(node["primary_region_include"])
        and all(not windows[i].noise_flag and windows[i].result.status == "ok" for windows in (pre, post))
    ]
    pairs = [
        (a, b) for a, b in combinations(eligible, 2)
        if montage == "car" or not _share_constituent_contact(run.nodes.iloc[a], run.nodes.iloc[b])
    ]
    if len(eligible) < 2 or not pairs:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame([{"file": path.stem, "montage": montage, "status": "too_few_eligible_nodes"}])

    feature_rows, pair_rows, qc_rows = [], [], []
    local = {index: pos for pos, index in enumerate(eligible)}
    mne_pairs = [(local[a], local[b]) for a, b in pairs]
    settings = cfg["analysis"]
    for epoch, windows in (("pre", pre), ("post", post)):
        segmented = _segmented_imfs(windows, eligible, run.recording, base_cfg, 4)
        conn = _mne_values(segmented, mne_pairs, run.recording.fs, base_cfg)
        signals = {i: np.asarray(windows[i].result.imf[:, 3], float) for i in eligible}
        envelopes = {i: np.abs(signal.hilbert(values)) for i, values in signals.items()}
        dynamics = Parallel(
            n_jobs=int(settings.get("n_jobs", cfg["analysis"].get("n_jobs", 1))),
            prefer="processes",
            batch_size=1,
            max_nbytes="2M",
        )(
            delayed(_pair_dynamics)(
                signals[a],
                signals[b],
                envelopes[a],
                envelopes[b],
                run.recording.fs,
                settings,
            )
            for a, b in pairs
        )
        for i in eligible:
            node = run.nodes.iloc[i]
            feature_rows.append({
                "file": run.recording.file_stem, "subject": run.recording.subject,
                "stim_frequency_hz": run.recording.stim_frequency_hz, "montage": montage,
                "epoch": epoch, "channel": node["channel"], "region_family": node["region_family"],
                "is_stimulated_channel": bool(node["is_stimulated_channel"]),
                "n_cycles": windows[i].result.n_cycles,
                **{metric: _metric_mean(windows[i].result, metric) for metric in FEATURES},
            })
        for pair_index, (a, b) in enumerate(pairs):
            node_a, node_b = run.nodes.iloc[a], run.nodes.iloc[b]
            pdc, tra, tra_status = dynamics[pair_index]
            row = {
                "file": run.recording.file_stem, "subject": run.recording.subject,
                "stim_frequency_hz": run.recording.stim_frequency_hz, "montage": montage, "epoch": epoch,
                "channel_a": node_a["channel"], "channel_b": node_b["channel"],
                "region_a": node_a["region_family"], "region_b": node_b["region_family"],
                "pair_type": _pair_type(node_a, node_b),
                "imcoh_signed": float(conn["imcoh"][pair_index]),
                "abs_imcoh": abs(float(conn["imcoh"][pair_index])),
                "coh": float(conn["coh"][pair_index]), "tra_unsigned": tra, "tra_status": tra_status,
                **pdc,
            }
            for metric in FEATURES:
                va = _metric_mean(windows[a].result, metric)
                vb = _metric_mean(windows[b].result, metric)
                row[f"{metric}_a"] = va
                row[f"{metric}_b"] = vb
                row[f"{metric}_absolute_difference"] = abs(va - vb)
                row[f"{metric}_log_ratio"] = np.log((va + settings["pdc_epsilon"]) / (vb + settings["pdc_epsilon"]))
            pair_rows.append(row)
            qc_rows.append({
                "file": run.recording.file_stem, "montage": montage, "epoch": epoch,
                "channel_a": node_a["channel"], "channel_b": node_b["channel"],
                "pdc_valid": pdc.get("pdc_valid", False), "pdc_reason": pdc.get("pdc_reason", ""),
                "pdc_order": pdc.get("pdc_order", np.nan), "tra_status": tra_status,
            })
    return pd.DataFrame(feature_rows), pd.DataFrame(pair_rows), pd.DataFrame(qc_rows)


def correlation_tables(pairs: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for keys, group in pairs.groupby(["montage", "file", "subject", "stim_frequency_hz", "epoch"]):
        for hypothesis in cfg["statistics"]["hypotheses"]:
            x = f"{hypothesis['waveform_metric']}_{hypothesis['transform']}"
            y = hypothesis["connectivity_metric"]
            eligible_group = group
            if y == "gpdc_log_ratio":
                eligible_group = eligible_group[eligible_group["pdc_valid"].fillna(False)]
            elif y == "tra_unsigned":
                eligible_group = eligible_group[eligible_group["tra_status"].eq("ok")]
            clean = eligible_group[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
            rho, p = stats.spearmanr(clean[x], clean[y]) if len(clean) >= 4 else (np.nan, np.nan)
            rows.append({
                "montage": keys[0], "file": keys[1], "subject": keys[2],
                "stim_frequency_hz": keys[3], "epoch": keys[4],
                "hypothesis": hypothesis["name"], "family": hypothesis["family"],
                "n_pairs": len(clean), "spearman_rho": rho, "pooled_pair_p_descriptive": p,
            })
    run = pd.DataFrame(rows)
    if not run.empty:
        index = ["montage", "file", "subject", "stim_frequency_hz", "hypothesis", "family"]
        wide = run.pivot_table(index=index, columns="epoch", values=["spearman_rho", "n_pairs"], aggfunc="first")
        if ("spearman_rho", "pre") in wide and ("spearman_rho", "post") in wide:
            change = wide.reset_index()
            change_rows = pd.DataFrame({
                column: change[(column, "")] for column in index
            })
            change_rows["epoch"] = "post_minus_pre"
            change_rows["n_pairs"] = np.minimum(
                change[("n_pairs", "pre")], change[("n_pairs", "post")]
            )
            change_rows["spearman_rho"] = (
                change[("spearman_rho", "post")] - change[("spearman_rho", "pre")]
            )
            change_rows["pooled_pair_p_descriptive"] = np.nan
            run = pd.concat([run, change_rows[run.columns]], ignore_index=True)
    subject = run.groupby(["montage", "subject", "stim_frequency_hz", "epoch", "hypothesis", "family"], as_index=False).agg(
        mean_run_rho=("spearman_rho", "mean"), n_runs=("file", "nunique")
    )
    return run, subject


def _bootstrap_ci(values: np.ndarray, n_bootstraps: int, rng: np.random.Generator) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(n_bootstraps, len(values)))
    boot = values[indices].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _permuted_run_rhos(
    group: pd.DataFrame,
    hypothesis: dict,
    n_permutations: int,
    rng: np.random.Generator,
    batch_size: int = 250,
) -> np.ndarray:
    """Permute channel feature labels and preserve the fixed connectivity matrix."""
    metric = hypothesis["waveform_metric"]
    transform = hypothesis["transform"]
    y_name = hypothesis["connectivity_metric"]
    columns = ["channel_a", "channel_b", f"{metric}_a", f"{metric}_b", y_name]
    eligible_group = group
    if y_name == "gpdc_log_ratio":
        eligible_group = eligible_group[eligible_group["pdc_valid"].fillna(False)]
    elif y_name == "tra_unsigned":
        eligible_group = eligible_group[eligible_group["tra_status"].eq("ok")]
    clean = eligible_group[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 4:
        return np.full(n_permutations, np.nan)
    channels = sorted(set(clean["channel_a"]) | set(clean["channel_b"]))
    positions = {channel: index for index, channel in enumerate(channels)}
    values = {}
    for side in ("a", "b"):
        for channel, value in clean[[f"channel_{side}", f"{metric}_{side}"]].itertuples(index=False):
            values[channel] = float(value)
    if len(values) != len(channels):
        return np.full(n_permutations, np.nan)
    feature = np.asarray([values[channel] for channel in channels], float)
    a_index = clean["channel_a"].map(positions).to_numpy(int)
    b_index = clean["channel_b"].map(positions).to_numpy(int)
    y_rank = stats.rankdata(clean[y_name].to_numpy(float))
    y_rank -= y_rank.mean()
    y_norm = np.sqrt(np.sum(y_rank**2))
    if y_norm <= np.finfo(float).eps:
        return np.full(n_permutations, np.nan)
    result = np.empty(n_permutations, float)
    epsilon = 1e-12
    for start in range(0, n_permutations, batch_size):
        stop = min(start + batch_size, n_permutations)
        permutations = np.asarray(
            [rng.permutation(len(channels)) for _ in range(stop - start)], dtype=int
        )
        permuted = feature[permutations]
        if transform == "absolute_difference":
            x = np.abs(permuted[:, a_index] - permuted[:, b_index])
        elif transform == "log_ratio":
            x = np.log(
                (permuted[:, a_index] + epsilon) / (permuted[:, b_index] + epsilon)
            )
        else:
            raise ValueError(f"Unsupported waveform transform: {transform}")
        x_rank = stats.rankdata(x, axis=1)
        x_rank -= x_rank.mean(axis=1, keepdims=True)
        denominator = np.sqrt(np.sum(x_rank**2, axis=1)) * y_norm
        result[start:stop] = np.divide(
            x_rank @ y_rank,
            denominator,
            out=np.full(stop - start, np.nan),
            where=denominator > np.finfo(float).eps,
        )
    return result


def _hierarchical_label_null(
    pairs: pd.DataFrame,
    montage: str,
    protocol: float,
    epoch: str,
    hypothesis: dict,
    cfg: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    selected = pairs[
        (pairs["montage"] == montage) & (pairs["stim_frequency_hz"] == protocol)
    ]
    n_permutations = int(cfg["analysis"]["n_permutations"])
    epochs = ("pre", "post") if epoch == "post_minus_pre" else (epoch,)
    groups = list(selected.groupby(["subject", "file"]))
    seeds = rng.integers(0, np.iinfo(np.uint32).max, size=len(groups), dtype=np.uint32)

    def compute_run_vector(item, seed):
        (subject, file_name), run_group = item
        run_rng = np.random.default_rng(int(seed))
        vectors = {}
        for active_epoch in epochs:
            vectors[active_epoch] = _permuted_run_rhos(
                run_group[run_group["epoch"] == active_epoch],
                hypothesis,
                n_permutations,
                run_rng,
            )
        vector = (
            vectors["post"] - vectors["pre"]
            if epoch == "post_minus_pre"
            else vectors[epoch]
        )
        return (subject, file_name), vector

    computed = Parallel(
        n_jobs=min(int(cfg["analysis"].get("n_jobs", 1)), max(1, len(groups))),
        prefer="processes",
        batch_size=1,
    )(
        delayed(compute_run_vector)(item, seed)
        for item, seed in zip(groups, seeds)
    )
    run_vectors = dict(computed)
    subject_vectors = []
    for subject in sorted({key[0] for key in run_vectors}):
        values = np.vstack([value for (sub, _), value in run_vectors.items() if sub == subject])
        valid = np.isfinite(values)
        count = valid.sum(axis=0)
        subject_vectors.append(np.divide(
            np.nansum(values, axis=0),
            count,
            out=np.full(values.shape[1], np.nan),
            where=count > 0,
        ))
    if not subject_vectors:
        return np.array([])
    values = np.vstack(subject_vectors)
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    return np.divide(
        np.nansum(values, axis=0),
        count,
        out=np.full(values.shape[1], np.nan),
        where=count > 0,
    )


def summarize_inference(subject: pd.DataFrame, pairs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Equal-participant inference using within-run channel-label permutations."""
    rng = np.random.default_rng(cfg["analysis"]["random_seed"])
    rows = []
    for keys, group in subject.groupby(["montage", "stim_frequency_hz", "epoch", "hypothesis", "family"]):
        values = group["mean_run_rho"].dropna().to_numpy(float)
        if not len(values):
            continue
        ci_low, ci_high = _bootstrap_ci(
            values, int(cfg["analysis"]["n_bootstraps"]), rng
        )
        hypothesis = next(
            item for item in cfg["statistics"]["hypotheses"] if item["name"] == keys[3]
        )
        # Pre is retained as baseline context. Formal permutation inference is
        # prespecified for post and for the post-minus-pre association change.
        if keys[2] == "pre":
            p = np.nan
        else:
            null = _hierarchical_label_null(
                pairs, keys[0], keys[1], keys[2], hypothesis, cfg, rng
            )
            null = null[np.isfinite(null)]
            p = (
                (1 + np.sum(np.abs(null) >= abs(values.mean()))) / (len(null) + 1)
                if len(null)
                else np.nan
            )
        rows.append({
            "montage": keys[0], "stim_frequency_hz": keys[1], "epoch": keys[2],
            "hypothesis": keys[3], "family": keys[4], "n_subjects": len(values),
            "mean_subject_rho": values.mean(), "median_subject_rho": np.median(values),
            "bootstrap_ci_low": ci_low, "bootstrap_ci_high": ci_high,
            "channel_label_permutation_p": p,
        })
    columns = [
        "montage", "stim_frequency_hz", "epoch", "hypothesis", "family",
        "n_subjects", "mean_subject_rho", "median_subject_rho",
        "bootstrap_ci_low", "bootstrap_ci_high", "channel_label_permutation_p",
        "p_fdr_within_family",
    ]
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=columns)
    out["p_fdr_within_family"] = np.nan
    for _, idx in out.groupby(["montage", "epoch", "family"]).groups.items():
        valid = out.loc[idx, "channel_label_permutation_p"].notna()
        valid_idx = out.loc[idx].index[valid]
        if len(valid_idx):
            out.loc[valid_idx, "p_fdr_within_family"] = multipletests(
                out.loc[valid_idx, "channel_label_permutation_p"], method="fdr_bh"
            )[1]
    return out[columns]


HYPOTHESIS_LABELS = {
    "if_matching_imcoh": "|Instantaneous frequency difference| vs |imaginary coherence|",
    "ascdesc_direction_pdc": "Log ascent/descent ratio vs log generalized PDC ratio",
    "ascdesc_mismatch_tra": "|Ascent/descent difference| vs temporal irreversibility",
    "ascdesc_matching_imcoh": "|Ascent/descent difference| vs |imaginary coherence|",
    "peaktrough_matching_imcoh": "|Peak/trough difference| vs |imaginary coherence|",
    "if_direction_pdc": "Log instantaneous frequency ratio vs log generalized PDC ratio",
    "peaktrough_direction_pdc": "Log peak/trough ratio vs log generalized PDC ratio",
    "if_mismatch_tra": "|Instantaneous frequency difference| vs temporal irreversibility",
    "peaktrough_mismatch_tra": "|Peak/trough difference| vs temporal irreversibility",
}
HYPOTHESIS_SHORT_LABELS = {
    "if_matching_imcoh": "IF difference vs absolute ImCoh",
    "ascdesc_direction_pdc": "Ascent/descent direction vs gPDC direction",
    "ascdesc_mismatch_tra": "Ascent/descent difference vs TRA",
    "ascdesc_matching_imcoh": "Ascent/descent difference vs absolute ImCoh",
    "peaktrough_matching_imcoh": "Peak/trough difference vs absolute ImCoh",
    "if_direction_pdc": "IF direction vs gPDC direction",
    "peaktrough_direction_pdc": "Peak/trough direction vs gPDC direction",
    "if_mismatch_tra": "IF difference vs TRA",
    "peaktrough_mismatch_tra": "Peak/trough difference vs TRA",
}


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_summary_figures(
    pairs: pd.DataFrame,
    subject: pd.DataFrame,
    inference: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    paths = []
    primary = ["if_matching_imcoh", "ascdesc_direction_pdc", "ascdesc_mismatch_tra"]
    montage_labels = {"bipolar": "Adjacent bipolar reference", "car": "Common-average reference"}

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.8), sharex=True)
    panel_letter = iter("abcdef")
    for row, montage in enumerate(("bipolar", "car")):
        for column, hypothesis in enumerate(primary):
            ax = axes[row, column]
            selected = inference[
                (inference["montage"] == montage)
                & (inference["epoch"] == "post")
                & (inference["hypothesis"] == hypothesis)
            ].sort_values("stim_frequency_hz")
            for _, item in selected.iterrows():
                x = 0 if item["stim_frequency_hz"] == 1 else 1
                ax.errorbar(
                    x,
                    item["mean_subject_rho"],
                    yerr=np.array([[
                        item["mean_subject_rho"] - item["bootstrap_ci_low"]
                    ], [
                        item["bootstrap_ci_high"] - item["mean_subject_rho"]
                    ]]),
                    fmt="o",
                    color="#276FBF" if x == 0 else "#C44536",
                    capsize=3,
                    markersize=6,
                )
                ax.text(
                    x,
                    0.96,
                    f"n={int(item['n_subjects'])}\np={item['channel_label_permutation_p']:.3g}",
                    ha="center",
                    va="top",
                    fontsize=7,
                    transform=ax.get_xaxis_transform(),
                )
            ax.axhline(0, color="0.5", lw=0.8, ls="--")
            ax.set_xticks([0, 1], ["1 Hz", "50 Hz"])
            ax.set_title(HYPOTHESIS_SHORT_LABELS[hypothesis], fontsize=9, pad=8)
            ax.text(
                -0.08, 1.04, next(panel_letter), transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom"
            )
            if column == 0:
                ax.set_ylabel(f"{montage_labels[montage]}\nParticipant-mean Spearman rho")
    fig.suptitle("Post-stimulation waveform-connectivity associations", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    stem = output_dir / "figure_01_main_hypothesis_summary"
    _save_figure(fig, stem); paths.extend([stem.with_suffix(".png"), stem.with_suffix(".pdf")])

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.7), sharey=False)
    for row, montage in enumerate(("bipolar", "car")):
        for column, epoch in enumerate(("pre", "post", "post_minus_pre")):
            ax = axes[row, column]
            selected = subject[
                (subject["montage"] == montage)
                & (subject["epoch"] == epoch)
                & subject["hypothesis"].isin(primary)
            ]
            offsets = np.linspace(-0.22, 0.22, 3)
            for offset, hypothesis, color in zip(
                offsets, primary, ("#276FBF", "#6A4C93", "#D17A22")
            ):
                values = selected[selected["hypothesis"] == hypothesis]
                for protocol, marker_x in ((1.0, 0), (50.0, 1)):
                    points = values[values["stim_frequency_hz"] == protocol]["mean_run_rho"]
                    ax.scatter(
                        np.full(len(points), marker_x + offset),
                        points,
                        s=20,
                        alpha=0.75,
                        color=color,
                        label=HYPOTHESIS_SHORT_LABELS[hypothesis]
                        if row == 0 and column == 0 and protocol == 1.0
                        else None,
                    )
            ax.axhline(0, color="0.5", lw=0.8, ls="--")
            ax.set_xticks([0, 1], ["1 Hz", "50 Hz"])
            ax.set_title({"pre": "Pre-stimulation", "post": "Post-stimulation",
                          "post_minus_pre": "Post minus pre association"}[epoch])
            if column == 0:
                ax.set_ylabel(f"{montage_labels[montage]}\nParticipant Spearman rho")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    stem = output_dir / "figure_02_pre_post_and_association_change"
    _save_figure(fig, stem); paths.extend([stem.with_suffix(".png"), stem.with_suffix(".pdf")])

    hypothesis_order = [item["hypothesis"] for _, item in inference[
        inference["epoch"] == "post"
    ].drop_duplicates("hypothesis").sort_values(["family", "hypothesis"]).iterrows()]
    row_order = [
        (protocol, hypothesis)
        for protocol in (1.0, 50.0)
        for hypothesis in hypothesis_order
    ]
    row_labels = [
        f"{protocol:g} Hz: {HYPOTHESIS_SHORT_LABELS[hypothesis]}"
        for protocol, hypothesis in row_order
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.2), sharey=True)
    for ax, montage in zip(axes, ("bipolar", "car")):
        selected = inference[
            (inference["montage"] == montage) & (inference["epoch"] == "post")
        ].copy()
        selected = (
            selected.set_index(["stim_frequency_hz", "hypothesis"])
            .reindex(row_order)
            .reset_index()
        )
        y = np.arange(len(selected))
        colors = np.where(selected["family"] == "primary", "#C44536", "#7A8B99")
        ax.scatter(selected["mean_subject_rho"], y, c=colors, s=35)
        for yi, (_, item) in zip(y, selected.iterrows()):
            ax.plot(
                [item["bootstrap_ci_low"], item["bootstrap_ci_high"]],
                [yi, yi],
                color="0.35",
                lw=1,
            )
            if pd.notna(item["p_fdr_within_family"]) and item["p_fdr_within_family"] < 0.05:
                ax.text(item["bootstrap_ci_high"] + 0.02, yi, "FDR", va="center", fontsize=7)
        ax.axvline(0, color="0.5", lw=0.8, ls="--")
        ax.set_yticks(y, row_labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(montage_labels[montage])
        ax.set_xlabel("Participant-mean Spearman rho")
    fig.suptitle("Post-stimulation prespecified and exploratory associations", fontsize=12)
    fig.tight_layout()
    stem = output_dir / "figure_03_exploratory_metric_comparison"
    _save_figure(fig, stem); paths.extend([stem.with_suffix(".png"), stem.with_suffix(".pdf")])

    participant_dir = output_dir / "participant_scatter"
    participant_dir.mkdir(exist_ok=True)
    for montage in ("bipolar", "car"):
        for hypothesis_name in primary:
            x_name, y_name = {
                "if_matching_imcoh": ("mean_if_absolute_difference", "abs_imcoh"),
                "ascdesc_direction_pdc": ("asc2desc_log_ratio", "gpdc_log_ratio"),
                "ascdesc_mismatch_tra": ("asc2desc_absolute_difference", "tra_unsigned"),
            }[hypothesis_name]
            selected = pairs[(pairs["montage"] == montage) & (pairs["epoch"] == "post")]
            subjects = sorted(selected["subject"].dropna().unique())
            if not subjects:
                continue
            ncols = 4
            nrows = int(np.ceil(len(subjects) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(12.2, 2.7 * nrows), squeeze=False)
            for ax, subject_name in zip(axes.flat, subjects):
                group = selected[selected["subject"] == subject_name][[x_name, y_name]].dropna()
                ax.scatter(group[x_name], group[y_name], s=7, alpha=0.25, color="#31572C")
                if len(group) >= 4:
                    rho = stats.spearmanr(group[x_name], group[y_name]).statistic
                    x_rank = stats.rankdata(group[x_name])
                    y_rank = stats.rankdata(group[y_name])
                    slope, intercept = np.polyfit(x_rank, y_rank, 1)
                    order = np.argsort(group[x_name].to_numpy())
                    fitted_rank = intercept + slope * x_rank[order]
                    y_sorted = np.sort(group[y_name].to_numpy())
                    fitted = np.interp(fitted_rank, np.arange(1, len(y_sorted) + 1), y_sorted)
                    ax.plot(group[x_name].to_numpy()[order], fitted, color="#BC4749", lw=1.2)
                    ax.set_title(f"{subject_name}; rho={rho:.2f}", fontsize=8)
                else:
                    ax.set_title(subject_name, fontsize=8)
            for ax in axes.flat[len(subjects):]:
                ax.axis("off")
            fig.supxlabel(HYPOTHESIS_LABELS[hypothesis_name].split(" vs ")[0])
            fig.supylabel(HYPOTHESIS_LABELS[hypothesis_name].split(" vs ")[-1])
            fig.suptitle(
                f"Post-stimulation participant-level channel-pair distributions: "
                f"{montage_labels[montage]}",
                fontsize=11,
            )
            fig.tight_layout()
            stem = participant_dir / f"{montage}_{hypothesis_name}_participant_scatter"
            _save_figure(fig, stem); paths.extend([stem.with_suffix(".png"), stem.with_suffix(".pdf")])
    return paths


def write_supervisor_summary(
    features: pd.DataFrame,
    pairs: pd.DataFrame,
    qc: pd.DataFrame,
    inference: pd.DataFrame,
    path: Path,
) -> None:
    pair_qc = qc[qc.get("channel_a", pd.Series(index=qc.index)).notna()].copy()
    pdc_valid = (
        pair_qc.get("pdc_valid", pd.Series(dtype=object))
        .astype(str)
        .str.lower()
        .eq("true")
    )
    tra_ok = pair_qc.get("tra_status", pd.Series(dtype=str)).eq("ok")
    tested = inference["channel_label_permutation_p"].notna()
    significant = inference.loc[tested & (inference["channel_label_permutation_p"] < 0.05)]
    fdr = inference.loc[tested & (inference["p_fdr_within_family"] < 0.05)]
    lines = [
        "ALL-PAIRS IMF4 WAVEFORM, CONNECTIVITY, AND IRREVERSIBILITY ANALYSIS",
        "",
        f"Runs represented: {pairs['file'].nunique() if not pairs.empty else 0}",
        f"Participants represented: {pairs['subject'].nunique() if not pairs.empty else 0}",
        f"Channel-epoch feature rows: {len(features):,}",
        f"Pair-epoch rows: {len(pairs):,}",
        f"Valid bivariate gPDC fits: {int(pdc_valid.sum()):,} / {len(pair_qc):,}",
        f"Valid temporal-irreversibility estimates: {int(tra_ok.sum()):,} / {len(pair_qc):,}",
        f"Formal post/change tests: {int(tested.sum())}",
        f"Uncorrected permutation p < 0.05: {len(significant)}",
        f"BH-FDR within family < 0.05: {len(fdr)}",
        "",
        "PRESPECIFIED INTERPRETATION",
        "Post-stimulation is the primary epoch. Pre-stimulation is baseline context, and",
        "post-minus-pre association change is a key secondary endpoint. The biological",
        "sample size is the number of participants, never the number of channel pairs.",
        "Bivariate gPDC is exploratory because it does not condition on all recorded nodes.",
        "",
        "PRESPECIFIED POST-STIMULATION RESULTS",
    ]
    selected = inference[
        (inference["epoch"] == "post") & (inference["family"] == "primary")
    ].sort_values(["montage", "hypothesis", "stim_frequency_hz"])
    for _, item in selected.iterrows():
        lines.append(
            f"- {item['montage']}, {item['stim_frequency_hz']:g} Hz, "
            f"{HYPOTHESIS_LABELS[item['hypothesis']]}: rho={item['mean_subject_rho']:.3f}, "
            f"95% CI [{item['bootstrap_ci_low']:.3f}, {item['bootstrap_ci_high']:.3f}], "
            f"permutation p={item['channel_label_permutation_p']:.4g}, "
            f"family FDR p={item['p_fdr_within_family']:.4g}, n={int(item['n_subjects'])}."
        )
    lines.extend(["", "ALL FDR-SURVIVING POST OR CHANGE RESULTS"])
    surviving = inference[
        inference["p_fdr_within_family"].lt(0.05)
        & inference["epoch"].isin(["post", "post_minus_pre"])
    ].sort_values("p_fdr_within_family")
    if surviving.empty:
        lines.append("- None.")
    for _, item in surviving.iterrows():
        lines.append(
            f"- {item['montage']}, {item['stim_frequency_hz']:g} Hz, {item['epoch']}, "
            f"{HYPOTHESIS_LABELS[item['hypothesis']]}: rho={item['mean_subject_rho']:.3f}, "
            f"permutation p={item['channel_label_permutation_p']:.4g}, "
            f"family FDR p={item['p_fdr_within_family']:.4g}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_reports_from_tables(config_path: str | Path) -> dict[str, Path]:
    cfg = load_config(config_path)
    root = Path(cfg["project"]["output_dir"])
    tables = root / "tables"
    features = pd.read_csv(tables / "channel_epoch_imf4_features.csv")
    pairs = pd.read_csv(tables / "all_pairs_connectivity.csv")
    qc = pd.read_csv(tables / "all_pairs_qc.csv", low_memory=False)
    subject = pd.read_csv(tables / "subject_correlations.csv")
    inference = pd.read_csv(tables / "hierarchical_inference.csv")
    figure_dir = root / "figures" / "full_dataset"
    make_summary_figures(pairs, subject, inference, figure_dir)
    summary_path = root / "SUPERVISOR_SUMMARY.txt"
    write_supervisor_summary(features, pairs, qc, inference, summary_path)
    return {"supervisor_summary": summary_path, "figures": figure_dir}


def combine_montage_outputs(config_path: str | Path) -> dict[str, Path]:
    cfg = load_config(config_path)
    root = Path(cfg["project"]["output_dir"])
    tables = root / "tables"
    table_names = (
        "channel_epoch_imf4_features",
        "all_pairs_connectivity",
        "all_pairs_qc",
        "var_pdc_qc",
        "temporal_irreversibility_qc",
        "run_correlations",
        "subject_correlations",
        "hierarchical_inference",
    )
    outputs = {}
    for name in table_names:
        frames = [
            pd.read_csv(tables / montage / f"{name}.csv", low_memory=False)
            for montage in ("bipolar", "car")
        ]
        path = tables / f"{name}.csv"
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        outputs[name] = path
    outputs.update(generate_reports_from_tables(config_path))
    return outputs


def run_all_pairs(
    config_path: str | Path,
    max_files: int | None = None,
    file_path: str | Path | None = None,
    montage: str | None = None,
) -> dict[str, Path]:
    cfg = load_config(config_path)
    root = Path(cfg["project"]["output_dir"])
    tables = root / "tables" / montage if montage else root / "tables"
    cache = root / "run_cache"
    tables.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    files = [Path(file_path)] if file_path else list(iter_mat_files(cfg["project"]["data_dir"]))
    excluded_names = set(cfg["analysis"].get("excluded_files", []))
    files = [path for path in files if path.name not in excluded_names]
    files.sort(key=lambda path: (path.stat().st_size, path.name))
    if max_files is not None:
        files = files[:max_files]
    feature_frames, pair_frames, qc_frames = [], [], []
    montages = [montage] if montage else cfg["analysis"]["montages"]
    for active_montage in montages:
        base_config = ROOT_CONFIGS[active_montage]
        base_cfg = load_config(base_config)
        for path in files:
            key = hashlib.sha1(
                f"{CACHE_VERSION}|{active_montage}|{path.resolve()}".encode()
            ).hexdigest()[:12]
            paths = {name: cache / f"{path.stem}_{active_montage}_{key}_{name}.csv" for name in ("features", "pairs", "qc")}
            if all(p.exists() for p in paths.values()):
                features, pairs, qc = (
                    _read_cached_csv(paths[name]) for name in ("features", "pairs", "qc")
                )
            else:
                features, pairs, qc = analyze_run(path, active_montage, base_cfg, cfg)
                for name, frame in (("features", features), ("pairs", pairs), ("qc", qc)):
                    frame.to_csv(paths[name], index=False)
            feature_frames.append(features); pair_frames.append(pairs); qc_frames.append(qc)
            print(f"{active_montage}: {path.name}: nodes={len(features)//2}, pair_epochs={len(pairs)}", flush=True)
    features = pd.concat(feature_frames, ignore_index=True)
    pairs = pd.concat(pair_frames, ignore_index=True)
    qc = pd.concat(qc_frames, ignore_index=True)
    run, subject = correlation_tables(pairs, cfg)
    inference = summarize_inference(subject, pairs, cfg)
    outputs = {
        "channel_epoch_imf4_features": features, "all_pairs_connectivity": pairs,
        "all_pairs_qc": qc,
        "var_pdc_qc": qc[[
            column for column in (
                "file", "montage", "epoch", "channel_a", "channel_b",
                "pdc_valid", "pdc_reason", "pdc_order"
            ) if column in qc
        ]],
        "temporal_irreversibility_qc": qc[[
            column for column in (
                "file", "montage", "epoch", "channel_a", "channel_b", "tra_status"
            ) if column in qc
        ]],
        "run_correlations": run,
        "subject_correlations": subject, "hierarchical_inference": inference,
    }
    paths = {}
    for name, frame in outputs.items():
        paths[name] = tables / f"{name}.csv"; frame.to_csv(paths[name], index=False)
    figure_paths = make_summary_figures(
        pairs, subject, inference, root / "figures" / "full_dataset"
    )
    summary_path = root / "SUPERVISOR_SUMMARY.txt"
    write_supervisor_summary(features, pairs, qc, inference, summary_path)
    paths["supervisor_summary"] = summary_path
    if figure_paths:
        paths["figures"] = root / "figures" / "full_dataset"
    return paths


ROOT_CONFIGS = {
    "bipolar": Path(__file__).resolve().parents[2] / "configs" / "primary_bipolar_mne_config.yaml",
    "car": Path(__file__).resolve().parents[2] / "configs" / "primary_car_mne_config.yaml",
}
