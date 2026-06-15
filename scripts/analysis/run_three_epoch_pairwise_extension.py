from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MNE_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "outputs" / ".numba_cache"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from seeg_waveform.three_epoch_pairwise import run_three_epoch_extension


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the exploratory pre/stimulation/post all-pairs extension."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "three_epoch_pairwise_config.yaml",
    )
    parser.add_argument("--montage", choices=["bipolar", "car"])
    parser.add_argument("--file", type=Path)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    outputs = run_three_epoch_extension(
        args.config,
        montage=args.montage,
        file_path=args.file,
        max_files=args.max_files,
        report_only=args.report_only,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
