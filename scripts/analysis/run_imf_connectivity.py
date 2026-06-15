from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seeg_waveform.imf_connectivity import analyze_imf_connectivity_dataset, summarize_existing_imf_connectivity


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IMF-wise stimulated-seed connectivity.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "imf_connectivity_config.yaml"))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Recompute stats from the existing IMF connectivity CSV without rerunning EMD.",
    )
    args = parser.parse_args()

    if args.stats_only:
        stats = summarize_existing_imf_connectivity(args.config)
        print(f"stats rows: {len(stats)}")
        print(stats.to_string(index=False))
        return

    pairs, freqs, stats = analyze_imf_connectivity_dataset(args.config, max_files=args.max_files)
    print(f"pair rows: {len(pairs)}")
    print(f"frequency QC rows: {len(freqs)}")
    print(f"stats rows: {len(stats)}")
    if not stats.empty:
        print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
