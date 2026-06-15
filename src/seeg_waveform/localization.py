from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


@dataclass(frozen=True)
class LocalizationPaths:
    subject_id: str
    channel_mat: Path
    waveform_summary: Path
    output_dir: Path


def normalize_channel_name(name: object) -> str:
    """Normalize contact labels across Brainstorm and stimulation MAT files."""
    text = str(name).strip()
    text = re.sub(r"\s+", "", text)
    return text.upper()


def contact_stem(name: str) -> str:
    match = re.match(r"([A-Za-z]+)", str(name).strip())
    return match.group(1).upper() if match else ""


def load_brainstorm_channels(channel_mat: Path) -> pd.DataFrame:
    mat = sio.loadmat(channel_mat, squeeze_me=True, struct_as_record=False)
    channels = np.atleast_1d(mat["Channel"])
    rows: list[dict[str, object]] = []
    for idx, ch in enumerate(channels):
        ch_type = str(getattr(ch, "Type", ""))
        if ch_type.upper() != "SEEG":
            continue
        loc = np.asarray(getattr(ch, "Loc", []), dtype=float).reshape(-1)
        if loc.size < 3 or not np.all(np.isfinite(loc[:3])):
            continue
        label = str(getattr(ch, "Name", f"ch{idx + 1}")).strip()
        xyz_m = loc[:3]
        rows.append(
            {
                "channel": label,
                "channel_norm": normalize_channel_name(label),
                "shaft": contact_stem(label),
                "x_m": xyz_m[0],
                "y_m": xyz_m[1],
                "z_m": xyz_m[2],
                "x_mm": xyz_m[0] * 1000.0,
                "y_mm": xyz_m[1] * 1000.0,
                "z_mm": xyz_m[2] * 1000.0,
                "coordinate_source": "Brainstorm channel.mat SCS",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Keep the first coordinate if Brainstorm contains duplicate labels.
    return out.drop_duplicates("channel_norm", keep="first").reset_index(drop=True)


def load_subject_waveform(summary_csv: Path, subject_id: str) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    subject_mask = df["subject"].astype(str).str.contains(subject_id, case=False, na=False)
    out = df.loc[subject_mask].copy()
    if out.empty:
        return out
    out["channel_norm"] = out["channel"].map(normalize_channel_name)
    out["stimulated_norms"] = out["stimulated_channels"].fillna("").map(
        lambda text: ";".join(normalize_channel_name(x) for x in str(text).split(";") if str(x).strip())
    )
    return out


def build_localization_table(paths: LocalizationPaths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coords = load_brainstorm_channels(paths.channel_mat)
    waveform = load_subject_waveform(paths.waveform_summary, paths.subject_id)
    if waveform.empty:
        merged = coords.copy()
        merged["waveform_match"] = False
        return coords, waveform, merged
    merged = waveform.merge(coords, on="channel_norm", how="left", suffixes=("_waveform", "_loc"))
    merged["waveform_match"] = merged["x_mm"].notna()
    return coords, waveform, merged


def summarize_match(coords: pd.DataFrame, waveform: pd.DataFrame, merged: pd.DataFrame) -> dict[str, object]:
    channel_col = "channel_waveform" if "channel_waveform" in merged.columns else "channel"
    return {
        "n_location_contacts": int(coords["channel_norm"].nunique()) if not coords.empty else 0,
        "n_waveform_rows": int(len(waveform)),
        "n_waveform_channels": int(waveform["channel_norm"].nunique()) if not waveform.empty else 0,
        "n_matched_rows": int(merged["waveform_match"].sum()) if "waveform_match" in merged else 0,
        "n_matched_channels": int(merged.loc[merged["waveform_match"], "channel_norm"].nunique())
        if "waveform_match" in merged
        else 0,
        "unmatched_waveform_channels": ";".join(
            sorted(merged.loc[~merged["waveform_match"], channel_col].dropna().astype(str).unique())
        )
        if "waveform_match" in merged
        else "",
    }


def make_region_template(coords: pd.DataFrame) -> pd.DataFrame:
    cols = ["channel", "channel_norm", "shaft", "region_label", "hemisphere", "notes"]
    template = coords[["channel", "channel_norm", "shaft", "x_mm"]].copy()
    template["region_label"] = ""
    template["hemisphere"] = np.where(template["x_mm"] < 0, "left", "right")
    template["notes"] = ""
    return template[cols]
