from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seeg_waveform.quinn_figures import make_selected_quinn_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Quinn-style diagnostic figures for selected SEEG channels.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "analysis_config.yaml"))
    parser.add_argument("--master-csv", default=None)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--no-noise-flagged", action="store_true")
    args = parser.parse_args()

    paths = make_selected_quinn_figures(
        args.config,
        master_csv=args.master_csv,
        top_n=args.top_n,
        include_noise_flagged=not args.no_noise_flagged,
    )
    print(f"created {len(paths)} Quinn-style diagnostic figures")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
