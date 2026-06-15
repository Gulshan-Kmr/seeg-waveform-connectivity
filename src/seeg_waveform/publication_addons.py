from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from .config import load_config
from .io import load_recording
from .pipeline import channel_diagnostics, safe_name
from .quinn_figures import set_publication_style


def _default_config() -> dict:
    return load_config(Path(__file__).resolve().parents[2] / "configs" / "analysis_config.yaml")


def _summary_path(cfg: dict) -> Path:
    return Path(cfg["project"]["output_dir"]) / "MASTER_all_channels_summary.csv"


def _fig_dir(cfg: dict, name: str) -> Path:
    out = Path(cfg["project"]["output_dir"]) / "figures" / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _ok_summary(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(_summary_path(cfg))
    ok = df[df["status"].astype(str).str.startswith("ok")].copy()
    for col in ["pre_noise_flag", "post_noise_flag"]:
        if col in ok:
            ok = ok[~ok[col].fillna(False).astype(bool)]
    return ok


def _load_npz_for_row(row: pd.Series, cfg: dict) -> dict[str, np.ndarray]:
    stem = safe_name(row["channel"]) + f"_IMF{int(cfg['emd'].get('imf_index', 3)) + 1}_outputs.npz"
    path = Path(cfg["project"]["output_dir"]) / str(row["file"]) / stem
    return dict(np.load(path, allow_pickle=True))


def _best_row(cfg: dict) -> pd.Series:
    ok = _ok_summary(cfg)
    min_cycles = int(cfg["emd"].get("publication_min_cycles", 5))
    ok = ok[(ok["pre_n_cycles"] >= min_cycles) & (ok["post_n_cycles"] >= min_cycles)]
    sort_col = "pa_if_change_score" if "pa_if_change_score" in ok else "delta_mean_if"
    ok = ok[np.isfinite(ok[sort_col])]
    return ok.sort_values(sort_col, ascending=False).iloc[0]


def _mean_sem(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=1)
    n = np.sum(np.isfinite(x), axis=1)
    sem = np.nanstd(x, axis=1, ddof=1) / np.sqrt(np.maximum(n, 1))
    return mean, sem


def make_eight_cycle_figure(cfg: dict | None = None, row: pd.Series | None = None) -> Path:
    """Create a Quinn Fig. 7-style panel from saved SEEG cycle profiles."""
    cfg = cfg or _default_config()
    row = _best_row(cfg) if row is None else row
    z = _load_npz_for_row(row, cfg)
    pre_if = z["pre_pa_if"]
    post_if = z["post_pa_if"]
    pre_w = z["pre_norm_waveform"]
    post_w = z["post_norm_waveform"]

    n_each = 4
    pre_idx = np.linspace(0, pre_if.shape[1] - 1, min(n_each, pre_if.shape[1]), dtype=int)
    post_idx = np.linspace(0, post_if.shape[1] - 1, min(n_each, post_if.shape[1]), dtype=int)
    examples = [("Pre", i, pre_if[:, i], pre_w[:, i], "tab:blue") for i in pre_idx]
    examples += [("Post", i, post_if[:, i], post_w[:, i], "tab:red") for i in post_idx]

    set_publication_style()
    fig, axes = plt.subplots(len(examples), 3, figsize=(7.2, 8.8), constrained_layout=True)
    phase_if = np.linspace(-np.pi, np.pi, pre_if.shape[0])
    phase_w = np.linspace(0, 1, pre_w.shape[0])
    sine = np.sin(np.linspace(0, 2 * np.pi, pre_w.shape[0]))

    for r, (epoch, idx, if_prof, wave, color) in enumerate(examples):
        axes[r, 0].plot(phase_w, wave, color=color, linewidth=1.1)
        axes[r, 0].plot(phase_w, sine, color="0.65", linestyle="--", linewidth=0.8)
        axes[r, 0].set_ylabel(f"{epoch}\ncycle {idx + 1}")
        axes[r, 0].set_ylim(-1.2, 1.2)

        axes[r, 1].plot(phase_if, if_prof, color=color, linewidth=1.1)
        axes[r, 1].set_ylabel("IF (Hz)")

        axes[r, 2].plot(phase_w, wave - sine, color=color, linewidth=1.1)
        axes[r, 2].axhline(0, color="0.65", linewidth=0.8)
        axes[r, 2].set_ylabel("Residual")

        for c in range(3):
            axes[r, c].tick_params(length=2)

    axes[0, 0].set_title("Normalized waveform")
    axes[0, 1].set_title("Phase-aligned IF")
    axes[0, 2].set_title("Waveform minus sine")
    for ax in axes[-1, :]:
        ax.set_xlabel("Cycle phase")

    out = _fig_dir(cfg, "quinn_examples") / f"{safe_name(row['file'])}__{safe_name(row['channel'])}__eight_cycles.png"
    fig.savefig(out, dpi=cfg["outputs"].get("figure_dpi", 300), bbox_inches="tight")
    plt.close(fig)
    return out


def _morlet_scalogram(x: np.ndarray, fs: float, freqs: np.ndarray, n_cycles: float = 7.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    power = np.empty((len(freqs), len(x)), dtype=float)
    for i, f in enumerate(freqs):
        sigma_t = n_cycles / (2 * np.pi * f)
        half = int(np.ceil(4 * sigma_t * fs))
        t = np.arange(-half, half + 1) / fs
        wavelet = np.exp(2j * np.pi * f * t) * np.exp(-(t**2) / (2 * sigma_t**2))
        wavelet /= np.sqrt(np.sum(np.abs(wavelet) ** 2))
        conv = signal.fftconvolve(x, wavelet, mode="same")
        power[i] = np.abs(conv) ** 2
    return power


def make_validation_figure(cfg: dict | None = None, row: pd.Series | None = None) -> Path:
    """Create PSD, Morlet time-frequency, and EMD/IF validation panels."""
    cfg = cfg or _default_config()
    row = _best_row(cfg) if row is None else row
    rec = load_recording(row["path"])
    diag = channel_diagnostics(rec, row["channel"], cfg)
    imf_index = diag["imf_index"]

    set_publication_style()
    fig, axes = plt.subplots(3, 2, figsize=(8.0, 7.2), constrained_layout=True)
    colors = {"Pre": "tab:blue", "Post": "tab:red"}

    for col, epoch in enumerate(["Pre", "Post"]):
        seg = diag[epoch.lower()]
        x = seg["processed"]
        t = seg["time"] - seg["time"][0]
        res = seg["result"]
        imf = res.imf[:, imf_index] if res.imf is not None and res.imf.shape[1] > imf_index else None
        ifreq = res.ifreq[:, imf_index] if res.ifreq is not None and res.ifreq.shape[1] > imf_index else None

        axes[0, col].plot(t, x, color="0.35", linewidth=0.6, label="Processed SEEG")
        if imf is not None:
            axes[0, col].plot(t[: imf.size], imf, color=colors[epoch], linewidth=0.9, label=f"IMF{imf_index + 1}")
        axes[0, col].set_title(f"{epoch}: processed SEEG and selected IMF")
        axes[0, col].set_xlabel("Time (s)")
        axes[0, col].set_ylabel("Amplitude")
        axes[0, col].legend(frameon=False, loc="upper right")

        f, pxx = signal.welch(x, fs=rec.fs, nperseg=min(int(rec.fs * 2), len(x)))
        axes[1, col].semilogy(f, pxx, color="0.35", linewidth=0.9, label="Processed SEEG")
        if imf is not None:
            fi, pxxi = signal.welch(imf, fs=rec.fs, nperseg=min(int(rec.fs * 2), len(imf)))
            axes[1, col].semilogy(fi, pxxi, color=colors[epoch], linewidth=1.0, label=f"IMF{imf_index + 1}")
        axes[1, col].set_xlim(0, 30)
        axes[1, col].set_title(f"{epoch}: Welch PSD")
        axes[1, col].set_xlabel("Frequency (Hz)")
        axes[1, col].set_ylabel("Power")
        axes[1, col].legend(frameon=False)

        freqs = np.linspace(2, 30, 60)
        power = _morlet_scalogram(x, rec.fs, freqs)
        im = axes[2, col].imshow(
            np.log10(power + np.finfo(float).eps),
            aspect="auto",
            origin="lower",
            extent=[t[0], t[-1], freqs[0], freqs[-1]],
            cmap="magma",
        )
        if ifreq is not None:
            if_plot = np.asarray(ifreq, dtype=float).copy()
            if_plot[(if_plot < freqs[0]) | (if_plot > freqs[-1])] = np.nan
            axes[2, col].plot(t[: if_plot.size], if_plot, color="cyan", linewidth=0.8, alpha=0.85, label="EMD IF")
            axes[2, col].legend(frameon=False, loc="upper right")
        axes[2, col].set_title(f"{epoch}: Morlet power with EMD IF")
        axes[2, col].set_xlabel("Time (s)")
        axes[2, col].set_ylabel("Frequency (Hz)")
        axes[2, col].set_ylim(freqs[0], freqs[-1])
        fig.colorbar(im, ax=axes[2, col], shrink=0.82, label="log power")

    out = _fig_dir(cfg, "validation") / f"{safe_name(row['file'])}__{safe_name(row['channel'])}__psd_morlet_emd.png"
    fig.savefig(out, dpi=cfg["outputs"].get("figure_dpi", 300), bbox_inches="tight")
    plt.close(fig)
    return out


def make_qc_flow_figure(cfg: dict | None = None) -> tuple[Path, Path]:
    """Create publication-style QC count figure and table."""
    cfg = cfg or _default_config()
    df = pd.read_csv(_summary_path(cfg))
    out_dir = _fig_dir(cfg, "qc")
    status_counts = df["status"].value_counts(dropna=False)
    ok = df[df["status"].astype(str).str.startswith("ok")]

    rows = [
        ("MAT files", df["file"].nunique()),
        ("Rows/channels", len(df)),
        ("OK or OK with noise flag", len(ok)),
        ("Excluded: few electrodes", int((df["status"] == "excluded_few_channels").sum())),
        ("Partial/failed channels", int((df["status"] == "partial_or_failed").sum())),
        ("Pre drift flags", int(ok.get("pre_drift_flag", pd.Series(False, index=ok.index)).fillna(False).sum())),
        ("Post drift flags", int(ok.get("post_drift_flag", pd.Series(False, index=ok.index)).fillna(False).sum())),
        ("Pre noise flags", int(ok.get("pre_noise_flag", pd.Series(False, index=ok.index)).fillna(False).sum())),
        ("Post noise flags", int(ok.get("post_noise_flag", pd.Series(False, index=ok.index)).fillna(False).sum())),
        ("Stimulated contacts", int(ok.get("is_stimulated_channel", pd.Series(False, index=ok.index)).fillna(False).sum())),
        ("Non-stimulated contacts", int((~ok.get("is_stimulated_channel", pd.Series(False, index=ok.index)).fillna(False).astype(bool)).sum())),
    ]
    table = pd.DataFrame(rows, columns=["qc_item", "count"])
    table_path = out_dir / "qc_flow_counts.csv"
    table.to_csv(table_path, index=False)

    set_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2), constrained_layout=True)

    axes[0].barh(table["qc_item"].iloc[:5][::-1], table["count"].iloc[:5][::-1], color="0.35")
    axes[0].set_title("Inclusion flow")
    axes[0].set_xlabel("Count")
    axes[0].bar_label(axes[0].containers[0], padding=2, fontsize=7)

    drift_noise = table.iloc[5:9]
    axes[1].bar(drift_noise["qc_item"], drift_noise["count"], color=["#4C78A8", "#F58518", "#72B7B2", "#E45756"])
    axes[1].set_title("QC flags")
    axes[1].set_ylabel("Channel windows")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].bar_label(axes[1].containers[0], padding=2, fontsize=7)

    stim_counts = table.iloc[9:11]
    axes[2].bar(stim_counts["qc_item"], stim_counts["count"], color=["#54A24B", "#B279A2"])
    axes[2].set_title("Contact class")
    axes[2].set_ylabel("Rows")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].bar_label(axes[2].containers[0], padding=2, fontsize=7)

    fig_path = out_dir / "qc_flow_counts.png"
    fig.savefig(fig_path, dpi=cfg["outputs"].get("figure_dpi", 300), bbox_inches="tight")
    plt.close(fig)
    return fig_path, table_path


def run_mixed_effects(cfg: dict | None = None) -> Path:
    """Fit channel-random-intercept mixed models on pre/post channel metrics."""
    cfg = cfg or _default_config()
    import statsmodels.formula.api as smf

    df = _ok_summary(cfg)
    df = df[(df["pre_n_cycles"] >= 5) & (df["post_n_cycles"] >= 5)].copy()
    df["stim_frequency_hz"] = df["stim_frequency_hz"].astype("category")
    df["is_stimulated_channel"] = df["is_stimulated_channel"].fillna(False).astype(int)
    df["subject"] = df["subject"].fillna(df["file"]).astype(str)
    df["channel_uid"] = df["file"].astype(str) + "|" + df["channel"].astype(str)

    metrics = {
        "mean_if": ("pre_mean_if", "post_mean_if"),
        "asc2desc": ("pre_asc2desc", "post_asc2desc"),
        "peak2trough": ("pre_peak2trough", "post_peak2trough"),
    }
    rows = []
    out_dir = Path(cfg["project"]["output_dir"]) / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric, (pre_col, post_col) in metrics.items():
        base = df[np.isfinite(df[pre_col]) & np.isfinite(df[post_col])].copy()
        pre = base[
            ["channel_uid", "subject", "file", "stim_frequency_hz", "stim_amplitude_ma", "is_stimulated_channel", pre_col]
        ].rename(columns={pre_col: "value"})
        pre["epoch"] = 0
        post = base[
            ["channel_uid", "subject", "file", "stim_frequency_hz", "stim_amplitude_ma", "is_stimulated_channel", post_col]
        ].rename(columns={post_col: "value"})
        post["epoch"] = 1
        data = pd.concat([pre, post], ignore_index=True)
        data["stim_frequency_hz"] = data["stim_frequency_hz"].astype("category")
        formula = "value ~ epoch * C(stim_frequency_hz) * is_stimulated_channel + stim_amplitude_ma"
        row_base = {
            "metric": metric,
            "outcome": "pre_post_value",
            "n_channels": base["channel_uid"].nunique(),
            "n_subjects": base["subject"].nunique(),
        }
        try:
            model = smf.mixedlm(formula, data, groups=data["channel_uid"])
            fit = model.fit(method="lbfgs", reml=False, maxiter=300, disp=False)
            for term, coef in fit.params.items():
                if term == "Group Var":
                    continue
                rows.append(
                    {
                        **row_base,
                        "model": "mixedlm_channel_random_intercept",
                        "term": term,
                        "estimate": coef,
                        "se": fit.bse.get(term, np.nan),
                        "z_or_t": fit.tvalues.get(term, np.nan),
                        "p": fit.pvalues.get(term, np.nan),
                        "converged": bool(fit.converged),
                        "note": "",
                    }
                )
        except Exception as exc:
            ols = smf.ols(formula, data).fit(cov_type="cluster", cov_kwds={"groups": data["channel_uid"]})
            for term, coef in ols.params.items():
                rows.append(
                    {
                        **row_base,
                        "model": "ols_clustered_by_channel_fallback",
                        "term": term,
                        "estimate": coef,
                        "se": ols.bse.get(term, np.nan),
                        "z_or_t": ols.tvalues.get(term, np.nan),
                        "p": ols.pvalues.get(term, np.nan),
                        "converged": False,
                        "note": str(exc)[:240],
                    }
                )

    out = pd.DataFrame(rows)
    out_path = out_dir / "mixed_effects_prepost_models.csv"
    out.to_csv(out_path, index=False)
    return out_path
