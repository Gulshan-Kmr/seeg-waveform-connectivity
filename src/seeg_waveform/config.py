from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load an analysis YAML file and resolve project paths."""
    cfg_path = Path(path).resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = cfg_path.parents[1]
    project = cfg.setdefault("project", {})
    project["root_dir"] = str(root)
    for key in ("data_dir", "output_dir", "atlas_dir", "freesurfer_dir"):
        if key in project:
            project[key] = str((root / project[key]).resolve())
    return cfg


def ensure_output_dirs(cfg: dict[str, Any]) -> Path:
    out = Path(cfg["project"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    return out
