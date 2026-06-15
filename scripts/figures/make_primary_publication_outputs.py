from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seeg_waveform.primary_publication import run_primary_publication_outputs


def main() -> None:
    """Build final primary tables, manuscript figures, and Extended Data copies."""
    paths = run_primary_publication_outputs(ROOT)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
