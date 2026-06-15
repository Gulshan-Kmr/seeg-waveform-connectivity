from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seeg_waveform.config import load_config
from seeg_waveform.publication_addons import run_mixed_effects


def main() -> None:
    cfg = load_config(ROOT / "configs" / "analysis_config.yaml")
    out = run_mixed_effects(cfg)
    print(f"mixed-effects stats: {out}")


if __name__ == "__main__":
    main()
