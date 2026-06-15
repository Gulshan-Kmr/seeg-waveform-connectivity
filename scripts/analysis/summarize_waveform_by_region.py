from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seeg_waveform.localization import normalize_channel_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize waveform results by anatomical region.")
    parser.add_argument("--subject-id", default="Subject13")
    parser.add_argument(
        "--waveform-summary",
        type=Path,
        default=Path(r"C:\Data_Science\seeg\outputs\waveform_relaxed\MASTER_all_channels_summary.csv"),
    )
    parser.add_argument(
        "--region-labels",
        type=Path,
        default=Path(r"C:\Data_Science\seeg\outputs\localization\sub_13\regions\subject13_electrode_region_labels.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\Data_Science\seeg\outputs\localization\sub_13\regions"),
    )
    return parser.parse_args()


def mean_ci(values: pd.Series) -> str:
    arr = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if len(arr) == 0:
        return ""
    mean = float(np.mean(arr))
    if len(arr) < 2:
        return f"{mean:.4g}"
    se = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
    return f"{mean:.4g} [{mean - 1.96 * se:.4g}, {mean + 1.96 * se:.4g}]"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.region_labels.exists():
        raise FileNotFoundError(
            f"Region labels not found: {args.region_labels}. Run label_subject13_regions.py with --atlas-volume or --label-table first."
        )
    wf = pd.read_csv(args.waveform_summary)
    wf = wf.loc[wf["subject"].astype(str).str.contains(args.subject_id, case=False, na=False)].copy()
    wf["channel_norm"] = wf["channel"].map(normalize_channel_name)
    labels = pd.read_csv(args.region_labels)
    merged = wf.merge(labels, on="channel_norm", how="left", suffixes=("", "_region"))
    merged_path = args.output_dir / "subject13_waveform_with_regions.csv"
    merged.to_csv(merged_path, index=False)

    metrics = ["delta_mean_if", "delta_asc2desc", "delta_peak2trough"]
    rows = []
    for keys, group in merged.groupby(["stim_frequency_hz", "atlas_hemisphere", "atlas_lobe", "atlas_region"], dropna=False):
        row = {
            "stim_frequency_hz": keys[0],
            "hemisphere": keys[1],
            "lobe": keys[2],
            "region": keys[3],
            "n_channels": group["channel_norm"].nunique(),
            "n_stimulated": int(group["is_stimulated_channel"].astype(bool).sum()),
        }
        for metric in metrics:
            row[f"{metric}_mean_ci"] = mean_ci(group[metric])
            row[f"{metric}_mean"] = group[metric].replace([np.inf, -np.inf], np.nan).mean()
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["stim_frequency_hz", "hemisphere", "lobe", "region"])
    summary_path = args.output_dir / "subject13_waveform_region_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"merged waveform labels: {merged_path}")
    print(f"region waveform summary: {summary_path}")


if __name__ == "__main__":
    main()
