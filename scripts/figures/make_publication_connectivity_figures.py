from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seeg_waveform.connectivity_figures import make_publication_connectivity_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Make publication-style standard connectivity figures.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "connectivity_config.yaml"))
    args = parser.parse_args()
    paths = make_publication_connectivity_figures(args.config)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
