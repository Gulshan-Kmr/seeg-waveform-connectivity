from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seeg_waveform.config import load_config
from seeg_waveform.pipeline import analyze_dataset
from seeg_waveform.stats import paired_summary


PROFILES = {
    "default": {
        "amp_percentile": 25.0,
        "max_if_range_hz": 10.0,
        "publication_min_cycles": 5,
    },
    "relaxed": {
        "amp_percentile": 10.0,
        "max_if_range_hz": 20.0,
        "publication_min_cycles": 3,
    },
    "strict": {
        "amp_percentile": 40.0,
        "max_if_range_hz": 8.0,
        "publication_min_cycles": 10,
    },
}


def apply_profile(base_cfg: dict, profile_name: str, output_root: Path) -> dict:
    cfg = deepcopy(base_cfg)
    profile = PROFILES[profile_name]
    cfg["emd"]["amp_percentile"] = profile["amp_percentile"]
    cfg["emd"]["max_if_range_hz"] = profile["max_if_range_hz"]
    cfg["emd"]["publication_min_cycles"] = profile["publication_min_cycles"]
    cfg["project"]["output_dir"] = str((output_root / profile_name).resolve())
    # Sensitivity runs compare master metrics; cycle-level npz/CSV outputs are not needed.
    cfg["outputs"]["save_channel_npz"] = False
    cfg["outputs"]["save_cycle_csv"] = False
    return cfg


def eligible_for_publication(master: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    min_cycles = int(cfg["emd"].get("publication_min_cycles", 5))
    out = master[master["status"].eq("ok")].copy()
    if {"pre_n_cycles", "post_n_cycles"}.issubset(out.columns):
        out = out[(out["pre_n_cycles"] >= min_cycles) & (out["post_n_cycles"] >= min_cycles)]
    for col in ["pre_noise_flag", "post_noise_flag"]:
        if col in out:
            out = out[~out[col].fillna(False).astype(bool)]
    return out


def summarize_profile(name: str, cfg: dict, master: pd.DataFrame) -> dict:
    eligible = eligible_for_publication(master, cfg)
    row = {
        "profile": name,
        "amp_percentile": cfg["emd"]["amp_percentile"],
        "max_if_range_hz": cfg["emd"]["max_if_range_hz"],
        "publication_min_cycles": cfg["emd"]["publication_min_cycles"],
        "total_rows": len(master),
        "ok_rows": int(master["status"].eq("ok").sum()) if "status" in master else 0,
        "eligible_rows": len(eligible),
        "excluded_few_channels": int(master["status"].eq("excluded_few_channels").sum()) if "status" in master else 0,
        "noise_flag_rows": int(
            master[[c for c in ["pre_noise_flag", "post_noise_flag"] if c in master]]
            .fillna(False)
            .astype(bool)
            .any(axis=1)
            .sum()
        )
        if any(c in master for c in ["pre_noise_flag", "post_noise_flag"])
        else 0,
    }

    for metric in ["mean_if", "asc2desc", "peak2trough"]:
        delta_col = f"delta_{metric}"
        if delta_col in eligible:
            row[f"{delta_col}_mean"] = eligible[delta_col].mean()
            row[f"{delta_col}_median"] = eligible[delta_col].median()
            row[f"{delta_col}_sd"] = eligible[delta_col].std()

    stats_df = paired_summary(eligible, ["mean_if", "asc2desc", "peak2trough"])
    for _, stat in stats_df.iterrows():
        metric = stat["metric"]
        row[f"{metric}_paired_t_p"] = stat["p_uncorrected"]
        row[f"{metric}_paired_t_p_fdr"] = stat["p_fdr_bh"]
        row[f"{metric}_cohens_dz"] = stat["cohens_dz"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run waveform criteria sensitivity analyses.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "analysis_config.yaml"))
    parser.add_argument("--profiles", nargs="+", default=list(PROFILES.keys()), choices=list(PROFILES.keys()))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-channels", type=int, default=None)
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    output_root = Path(base_cfg["project"]["root_dir"]) / "outputs" / "sensitivity"
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for profile_name in args.profiles:
        cfg = apply_profile(base_cfg, profile_name, output_root)
        print(f"Running sensitivity profile: {profile_name}")
        master = analyze_dataset(cfg, max_files=args.max_files, max_channels=args.max_channels)
        out_dir = Path(cfg["project"]["output_dir"])
        paired = paired_summary(eligible_for_publication(master, cfg), ["mean_if", "asc2desc", "peak2trough"])
        paired.to_csv(out_dir / "eligible_paired_stats.csv", index=False)
        rows.append(summarize_profile(profile_name, cfg, master))

    summary = pd.DataFrame(rows)
    summary_path = output_root / "sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Sensitivity summary: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
