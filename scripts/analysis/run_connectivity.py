from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seeg_waveform.connectivity import analyze_connectivity_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stimulated-seed to non-stimulated-target connectivity.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "connectivity_config.yaml"))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--include-pdc",
        action="store_true",
        help="Also run guarded bivariate VAR/PDC. This is slower and may reject pairs failing assumptions.",
    )
    args = parser.parse_args()

    master, summary = analyze_connectivity_dataset(
        args.config,
        max_files=args.max_files,
        include_pdc=args.include_pdc,
    )
    print(f"pairs/rows: {len(master)}")
    print(f"stats rows: {len(summary)}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
