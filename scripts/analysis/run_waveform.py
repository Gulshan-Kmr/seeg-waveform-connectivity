from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seeg_waveform.config import load_config
from seeg_waveform.pipeline import analyze_dataset, analyze_file
from seeg_waveform.plotting import plot_group_summary, plot_top_changes
from seeg_waveform.stats import write_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Quinn-style SEEG waveform analysis.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "analysis_config.yaml"))
    parser.add_argument("--file", default=None, help="Optional single .mat file to process.")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-channels", type=int, default=None)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["project"]["output_dir"])

    if args.file:
        master = analyze_file(args.file, cfg, max_channels=args.max_channels)
        master_csv = out_dir / "single_file_summary.csv"
        master.to_csv(master_csv, index=False)
    else:
        master = analyze_dataset(cfg, max_files=args.max_files, max_channels=args.max_channels)
        master_csv = out_dir / "MASTER_all_channels_summary.csv"

    stats_csv = out_dir / "stats" / "paired_pre_post_stats.csv"
    write_stats(master_csv, stats_csv)

    if not args.skip_figures:
        fig_dir = out_dir / "figures"
        plot_top_changes(master_csv, fig_dir / "top_channels", cfg["outputs"].get("top_n_channels", 30), cfg["outputs"].get("figure_dpi", 300))
        plot_group_summary(master_csv, fig_dir / "group_summary", cfg["outputs"].get("figure_dpi", 300))

    print(f"rows: {len(master)}")
    print(f"summary: {master_csv}")
    print(f"stats: {stats_csv}")
    print(f"output dir: {out_dir}")


if __name__ == "__main__":
    main()
