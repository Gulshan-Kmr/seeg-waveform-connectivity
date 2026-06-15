from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seeg_waveform.spectral import run_spectral_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pre/post SEEG spectral band-power analysis.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "spectral_config.yaml")
    parser.add_argument(
        "--waveform-regions",
        type=Path,
        default=ROOT / "outputs" / "waveform_relaxed" / "region_quinn" / "MASTER_waveform_with_brainstorm_regions.csv",
    )
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()
    paths = run_spectral_analysis(args.config, waveform_regions_csv=args.waveform_regions, max_files=args.max_files)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
