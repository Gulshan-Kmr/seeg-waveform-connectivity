from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def paired_summary(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    valid = df[df["status"].eq("ok")].copy()
    for metric in metrics:
        delta_col = f"delta_{metric}"
        if delta_col not in valid:
            continue
        values = valid[delta_col].dropna().to_numpy(dtype=float)
        if values.size < 2:
            continue
        test = stats.ttest_1samp(values, 0.0, nan_policy="omit")
        mean = float(np.nanmean(values))
        sd = float(np.nanstd(values, ddof=1))
        sem = sd / np.sqrt(values.size)
        ci = stats.t.interval(0.95, values.size - 1, loc=mean, scale=sem)
        rows.append(
            {
                "metric": metric,
                "n": values.size,
                "mean_delta": mean,
                "sd_delta": sd,
                "cohens_dz": mean / sd if sd > 0 else np.nan,
                "t": float(test.statistic),
                "p_uncorrected": float(test.pvalue),
                "ci95_low": float(ci[0]),
                "ci95_high": float(ci[1]),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr_bh"] = fdr_bh(out["p_uncorrected"].to_numpy())
    return out


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    ranked = pvals[order]
    n = len(pvals)
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def write_stats(master_csv: str | Path, output_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(master_csv)
    summary = paired_summary(df, ["mean_if", "asc2desc", "peak2trough"])
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    return summary
