from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .config import load_config
from .quinn_figures import set_publication_style


def _eligible_channels(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    min_cycles = int(cfg["emd"].get("publication_min_cycles", 5))
    out = df[df["status"].eq("ok")].copy()
    out = out[(out["pre_n_cycles"] >= min_cycles) & (out["post_n_cycles"] >= min_cycles)]
    for col in ["pre_noise_flag", "post_noise_flag"]:
        if col in out:
            out = out[~out[col].fillna(False).astype(bool)]
    return out


def _violin(
    ax,
    groups: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    title: str,
    ylabel: str,
    reference: float | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    groups = [g[np.isfinite(g)] for g in groups]
    keep = [i for i, g in enumerate(groups) if g.size > 0]
    groups = [groups[i] for i in keep]
    labels = [labels[i] for i in keep]
    colors = [colors[i] for i in keep]
    if not groups:
        ax.text(0.5, 0.5, "No eligible data", ha="center", va="center", transform=ax.transAxes)
        return
    parts = ax.violinplot(groups, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("0.25")
        body.set_alpha(0.72)
    if "cmedians" in parts:
        parts["cmedians"].set_color("0.05")
        parts["cmedians"].set_linewidth(1.2)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if reference is not None:
        ax.axhline(reference, color="0.25", linewidth=0.8, linestyle="--")
    if ylim is not None:
        ax.set_ylim(*ylim)


def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan)
    valid = np.isfinite(pvals)
    if not valid.any():
        return out
    pv = pvals[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    tmp = np.empty_like(adjusted)
    tmp[order] = np.clip(adjusted, 0, 1)
    out[valid] = tmp
    return out


def _p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _stats_p(stats_df: pd.DataFrame, metric: str, level: str, group: str, p_col: str = "wilcoxon_p_fdr") -> float:
    rows = stats_df[
        stats_df["metric"].eq(metric)
        & stats_df["level"].eq(level)
        & stats_df["group"].eq(group)
    ]
    if rows.empty or p_col not in rows:
        return np.nan
    return float(rows.iloc[0][p_col])


def _annotate_pair(ax, x1: float, x2: float, p: float, y_pad: float = 0.04) -> None:
    label = _p_to_stars(p)
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    y = ymax - yrange * y_pad
    h = yrange * 0.025
    ax.plot([x1, x1, x2, x2], [y - h, y, y, y - h], color="0.15", linewidth=0.8, clip_on=False)
    ax.text((x1 + x2) / 2, y + h * 0.25, label, ha="center", va="bottom", fontsize=8)


def _annotate_single(ax, x: float, p: float, y_pad: float = 0.06) -> None:
    label = _p_to_stars(p)
    if label == "n.s.":
        return
    ymin, ymax = ax.get_ylim()
    y = ymax - (ymax - ymin) * y_pad
    ax.text(x, y, label, ha="center", va="center", fontsize=9, fontweight="bold")


def _paired_stats(values_pre: np.ndarray, values_post: np.ndarray) -> dict:
    pre = np.asarray(values_pre, dtype=float)
    post = np.asarray(values_post, dtype=float)
    mask = np.isfinite(pre) & np.isfinite(post)
    pre = pre[mask]
    post = post[mask]
    delta = post - pre
    if delta.size < 2:
        return {"n": int(delta.size)}
    ttest = stats.ttest_rel(post, pre, nan_policy="omit")
    try:
        wilcoxon = stats.wilcoxon(delta)
        p_wilcoxon = float(wilcoxon.pvalue)
        w_stat = float(wilcoxon.statistic)
    except ValueError:
        p_wilcoxon = np.nan
        w_stat = np.nan
    sd = float(np.nanstd(delta, ddof=1))
    mean = float(np.nanmean(delta))
    sem = sd / np.sqrt(delta.size)
    ci = stats.t.interval(0.95, delta.size - 1, loc=mean, scale=sem)
    return {
        "n": int(delta.size),
        "pre_mean": float(np.nanmean(pre)),
        "post_mean": float(np.nanmean(post)),
        "mean_delta": mean,
        "median_delta": float(np.nanmedian(delta)),
        "sd_delta": sd,
        "cohens_dz": mean / sd if sd > 0 else np.nan,
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "paired_t": float(ttest.statistic),
        "paired_t_p": float(ttest.pvalue),
        "wilcoxon_w": w_stat,
        "wilcoxon_p": p_wilcoxon,
    }


def write_control_point_stats(df: pd.DataFrame, out_dir: Path) -> Path:
    rows = []
    metrics = ["asc2desc", "peak2trough"]
    group_specs = [("all", df)]
    for freq in sorted(df["stim_frequency_hz"].dropna().unique()):
        group_specs.append((f"{freq:g}Hz", df[df["stim_frequency_hz"].eq(freq)]))
    group_specs.extend(
        [
            ("stimulated", df[df["is_stimulated_channel"].astype(bool)]),
            ("nonstimulated", df[~df["is_stimulated_channel"].astype(bool)]),
        ]
    )
    for freq in sorted(df["stim_frequency_hz"].dropna().unique()):
        freq_df = df[df["stim_frequency_hz"].eq(freq)]
        group_specs.append((f"{freq:g}Hz_stimulated", freq_df[freq_df["is_stimulated_channel"].astype(bool)]))
        group_specs.append((f"{freq:g}Hz_nonstimulated", freq_df[~freq_df["is_stimulated_channel"].astype(bool)]))

    for metric in metrics:
        for level, data in [("channel", df), ("file_mean", df.groupby(["file", "stim_frequency_hz"], as_index=False).mean(numeric_only=True))]:
            local_group_specs = group_specs if level == "channel" else [("all", data)] + [
                (f"{freq:g}Hz", data[data["stim_frequency_hz"].eq(freq)])
                for freq in sorted(data["stim_frequency_hz"].dropna().unique())
            ]
            for group_name, group in local_group_specs:
                if f"pre_{metric}" not in group or f"post_{metric}" not in group:
                    continue
                row = {"metric": metric, "level": level, "group": group_name}
                row.update(_paired_stats(group[f"pre_{metric}"].to_numpy(), group[f"post_{metric}"].to_numpy()))
                rows.append(row)

    stats_df = pd.DataFrame(rows)
    if not stats_df.empty:
        stats_df["paired_t_p_fdr"] = _fdr_bh(stats_df["paired_t_p"].to_numpy())
        stats_df["wilcoxon_p_fdr"] = _fdr_bh(stats_df["wilcoxon_p"].to_numpy())
    out_path = out_dir / "control_point_pre_post_stats.csv"
    stats_df.to_csv(out_path, index=False)
    return out_path


def plot_control_point_violins(config_path: str | Path, master_csv: str | Path | None = None) -> list[Path]:
    set_publication_style()
    cfg = load_config(config_path)
    out_root = Path(cfg["project"]["output_dir"])
    master_csv = Path(master_csv) if master_csv else out_root / "MASTER_all_channels_summary.csv"
    df = _eligible_channels(pd.read_csv(master_csv), cfg)
    out_dir = out_root / "figures" / "control_point_violins"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = write_control_point_stats(df, out_dir)
    stats_df = pd.read_csv(stats_path)

    paths: list[Path] = [stats_path]
    metrics = [
        ("asc2desc", "Ascent-to-descent ratio"),
        ("peak2trough", "Peak-to-trough ratio"),
    ]
    for metric, ylabel in metrics:
        fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.0), constrained_layout=True)
        fig.suptitle(f"{ylabel} | eligible channels n={len(df)}")

        pre = df[f"pre_{metric}"].to_numpy(dtype=float)
        post = df[f"post_{metric}"].to_numpy(dtype=float)
        delta = df[f"delta_{metric}"].to_numpy(dtype=float)
        _violin(axes[0, 0], [pre, post], ["Pre", "Post"], ["tab:blue", "tab:red"], "Pre vs post", ylabel, reference=0.5, ylim=(0.35, 0.65))
        _annotate_pair(axes[0, 0], 1, 2, _stats_p(stats_df, metric, "channel", "all"))
        _violin(axes[1, 0], [delta], ["Post - pre"], ["0.5"], "Delta", f"Delta {ylabel}", reference=0.0)
        _annotate_single(axes[1, 0], 1, _stats_p(stats_df, metric, "channel", "all"))

        groups = []
        labels = []
        colors = []
        for freq, color in [(1.0, "tab:green"), (50.0, "tab:purple")]:
            sub = df[df["stim_frequency_hz"].eq(freq)]
            groups.append(sub[f"delta_{metric}"].to_numpy(dtype=float))
            labels.append(f"{freq:g} Hz")
            colors.append(color)
        _violin(axes[0, 1], groups, labels, colors, "Delta by frequency", f"Delta {ylabel}", reference=0.0)
        for xpos, group_name in enumerate(["1Hz", "50Hz"], start=1):
            _annotate_single(axes[0, 1], xpos, _stats_p(stats_df, metric, "channel", group_name))

        groups = []
        labels = []
        colors = []
        for is_stim, label, color in [(True, "Stim", "tab:orange"), (False, "Non-stim", "0.55")]:
            sub = df[df["is_stimulated_channel"].astype(bool).eq(is_stim)]
            groups.append(sub[f"delta_{metric}"].to_numpy(dtype=float))
            labels.append(label)
            colors.append(color)
        _violin(axes[1, 1], groups, labels, colors, "Delta by contact type", f"Delta {ylabel}", reference=0.0)
        for xpos, group_name in enumerate(["stimulated", "nonstimulated"], start=1):
            _annotate_single(axes[1, 1], xpos, _stats_p(stats_df, metric, "channel", group_name))

        groups = []
        labels = []
        colors = []
        for freq, is_stim, label, color in [
            (1.0, True, "1 Hz\nstim", "tab:orange"),
            (1.0, False, "1 Hz\nnon-stim", "tab:green"),
            (50.0, True, "50 Hz\nstim", "tab:red"),
            (50.0, False, "50 Hz\nnon-stim", "tab:purple"),
        ]:
            sub = df[df["stim_frequency_hz"].eq(freq) & df["is_stimulated_channel"].astype(bool).eq(is_stim)]
            groups.append(sub[f"delta_{metric}"].to_numpy(dtype=float))
            labels.append(label)
            colors.append(color)
        _violin(axes[0, 2], groups, labels, colors, "Delta by frequency/contact", f"Delta {ylabel}", reference=0.0)
        for xpos, group_name in enumerate(
            ["1Hz_stimulated", "1Hz_nonstimulated", "50Hz_stimulated", "50Hz_nonstimulated"],
            start=1,
        ):
            _annotate_single(axes[0, 2], xpos, _stats_p(stats_df, metric, "channel", group_name))

        # Participant/file-level means reduce over-weighting dense electrode implantations.
        file_means = df.groupby(["file", "stim_frequency_hz"], as_index=False)[f"delta_{metric}"].mean()
        groups = []
        labels = []
        colors = []
        for freq, color in [(1.0, "tab:green"), (50.0, "tab:purple")]:
            sub = file_means[file_means["stim_frequency_hz"].eq(freq)]
            groups.append(sub[f"delta_{metric}"].to_numpy(dtype=float))
            labels.append(f"{freq:g} Hz")
            colors.append(color)
        _violin(axes[1, 2], groups, labels, colors, "File-level delta means", f"Delta {ylabel}", reference=0.0)
        for xpos, group_name in enumerate(["1Hz", "50Hz"], start=1):
            _annotate_single(axes[1, 2], xpos, _stats_p(stats_df, metric, "file_mean", group_name))

        out_path = out_dir / f"{metric}_violins.png"
        fig.savefig(out_path, dpi=cfg["outputs"].get("figure_dpi", 300))
        plt.close(fig)
        paths.append(out_path)

        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
        fig.suptitle(f"{ylabel}: pre vs post by frequency/contact group")
        group_specs = [
            (1.0, True, "1 Hz stimulated"),
            (1.0, False, "1 Hz non-stimulated"),
            (50.0, True, "50 Hz stimulated"),
            (50.0, False, "50 Hz non-stimulated"),
        ]
        for ax, (freq, is_stim, title) in zip(axes.ravel(), group_specs):
            sub = df[df["stim_frequency_hz"].eq(freq) & df["is_stimulated_channel"].astype(bool).eq(is_stim)]
            group_name = f"{freq:g}Hz_{'stimulated' if is_stim else 'nonstimulated'}"
            _violin(
                ax,
                [sub[f"pre_{metric}"].to_numpy(dtype=float), sub[f"post_{metric}"].to_numpy(dtype=float)],
                ["Pre", "Post"],
                ["tab:blue", "tab:red"],
                f"{title} | n={len(sub)}",
                ylabel,
                reference=0.5,
                ylim=(0.35, 0.65),
            )
            _annotate_pair(ax, 1, 2, _stats_p(stats_df, metric, "channel", group_name))
        out_path = out_dir / f"{metric}_pre_post_by_frequency_contact.png"
        fig.savefig(out_path, dpi=cfg["outputs"].get("figure_dpi", 300))
        plt.close(fig)
        paths.append(out_path)

    return paths
