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

from seeg_waveform.all_pairs_waveform_connectivity import (
    combine_montage_outputs,
    generate_reports_from_tables,
    run_all_pairs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all-pairs IMF4 waveform, gPDC, ImCoh, and irreversibility analysis.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "all_pairs_waveform_connectivity_config.yaml")
    parser.add_argument("--file", type=Path)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--montage", choices=["bipolar", "car"])
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate figures and supervisor summary from completed tables.",
    )
    parser.add_argument(
        "--combine-montages",
        action="store_true",
        help="Combine completed bipolar and CAR tables and regenerate final reports.",
    )
    args = parser.parse_args()
    outputs = (
        combine_montage_outputs(args.config)
        if args.combine_montages
        else generate_reports_from_tables(args.config)
        if args.report_only
        else run_all_pairs(
            args.config, max_files=args.max_files, file_path=args.file, montage=args.montage
        )
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
