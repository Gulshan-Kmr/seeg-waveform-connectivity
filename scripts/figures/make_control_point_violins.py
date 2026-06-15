from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seeg_waveform.control_point_figures import plot_control_point_violins


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate asc2desc and peak2trough violin plots.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "analysis_config.yaml"))
    parser.add_argument("--master-csv", default=None)
    args = parser.parse_args()
    paths = plot_control_point_violins(args.config, args.master_csv)
    print(f"created {len(paths)} control-point violin figures")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
