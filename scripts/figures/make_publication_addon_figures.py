from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seeg_waveform.config import load_config
from seeg_waveform.publication_addons import make_eight_cycle_figure, make_qc_flow_figure, make_validation_figure


def main() -> None:
    cfg = load_config(ROOT / "configs" / "analysis_config.yaml")
    paths = []
    paths.append(make_eight_cycle_figure(cfg))
    paths.append(make_validation_figure(cfg))
    paths.extend(make_qc_flow_figure(cfg))
    print("created publication add-on outputs")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
