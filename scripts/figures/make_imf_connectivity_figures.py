from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seeg_waveform.imf_connectivity import plot_imf_connectivity_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Make IMF connectivity QC and summary figures.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "imf_connectivity_config.yaml"))
    args = parser.parse_args()
    paths = plot_imf_connectivity_outputs(args.config)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
