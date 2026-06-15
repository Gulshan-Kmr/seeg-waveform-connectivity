from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Keep scientific-library caches local to ignored outputs and avoid user-profile writes.
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MNE_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "outputs" / ".numba_cache"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from seeg_waveform.primary_car_mne import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run common-average-reference unipolar IMF waveform and MNE connectivity sensitivity analysis."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "primary_car_mne_config.yaml")
    parser.add_argument("--file", type=Path, default=None, help="Run one recording for validation.")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--early-sensitivity", action="store_true", help="Use the prespecified original 10-second early window.")
    args = parser.parse_args()
    paths = run_analysis(args.config, max_files=args.max_files, early=args.early_sensitivity, file_path=args.file)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
