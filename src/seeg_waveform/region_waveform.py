from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .localization import normalize_channel_name
from .spatial_analysis import ATLAS_COLUMNS, is_non_gray_label, load_brainstorm_atlas_csv


def subject_number(value: object) -> int | None:
    match = re.search(r"Subject0*([0-9]+)", str(value), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"sub[_-]?0*([0-9]+)", str(value), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def find_unipolar_atlas_files(atlas_root: Path) -> dict[int, Path]:
    files = sorted(atlas_root.rglob("*_unipolar_filtered_Coordinates_and_Atlas.csv"))
    out: dict[int, Path] = {}
    for path in files:
        num = subject_number(path.name)
        if num is not None:
            out[num] = path
    return out


def load_all_unipolar_atlases(atlas_root: Path, primary_atlas: str = "AAL") -> pd.DataFrame:
    rows = []
    for num, path in find_unipolar_atlas_files(atlas_root).items():
        atlas = load_brainstorm_atlas_csv(path, primary_atlas)
        atlas["subject_number"] = num
        atlas["subject_key"] = f"Subject{num:02d}"
        atlas["atlas_csv"] = str(path)
        rows.append(atlas)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def join_waveform_to_atlas(
    waveform_csv: Path,
    atlas_root: Path,
    primary_atlas: str = "AAL",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wf = pd.read_csv(waveform_csv)
    wf["subject_number"] = wf["subject"].map(subject_number)
    wf["channel_norm"] = wf["channel"].map(normalize_channel_name)
    atlas = load_all_unipolar_atlases(atlas_root, primary_atlas)
    keep = [
        "subject_number",
        "subject_key",
        "channel_norm",
        "channel",
        "x_mni",
        "y_mni",
        "z_mni",
        "x_world",
        "y_world",
        "z_world",
        "primary_atlas",
        "primary_region",
        "primary_non_gray_flag",
        "any_out_of_atlas_flag",
        "any_white_matter_flag",
        "analysis_gray_matter_flag",
        "hemisphere",
        *[c for c in ATLAS_COLUMNS if c in atlas.columns],
    ]
    merged = wf.merge(
        atlas[keep].drop_duplicates(["subject_number", "channel_norm"]),
        on=["subject_number", "channel_norm"],
        how="left",
        suffixes=("", "_atlas"),
    )
    merged["has_atlas_label"] = merged["primary_region"].notna()
    merged["region_analysis_include"] = (
        merged["has_atlas_label"]
        & merged["analysis_gray_matter_flag"].fillna(False)
        & merged["status"].eq("ok")
        & ~merged["pre_noise_flag"].fillna(False).astype(bool)
        & ~merged["post_noise_flag"].fillna(False).astype(bool)
    )
    return merged, atlas


def common_region_summary(merged: pd.DataFrame) -> pd.DataFrame:
    valid = merged.loc[merged["has_atlas_label"]].copy()
    rows = []
    for stim_state, group in [
        ("stimulated", valid.loc[valid["is_stimulated_channel"].fillna(False).astype(bool)]),
        ("nonstimulated", valid.loc[~valid["is_stimulated_channel"].fillna(False).astype(bool)]),
        ("all", valid),
    ]:
        if group.empty:
            continue
        agg = (
            group.groupby(["primary_region", "analysis_gray_matter_flag"], dropna=False)
            .agg(
                n_rows=("channel_norm", "size"),
                n_contacts=("channel_norm", "nunique"),
                n_subjects=("subject_number", "nunique"),
                n_files=("file", "nunique"),
                n_1hz=("stim_frequency_hz", lambda x: int((x == 1).sum())),
                n_50hz=("stim_frequency_hz", lambda x: int((x == 50).sum())),
                mean_delta_mean_if=("delta_mean_if", "mean"),
                mean_delta_asc2desc=("delta_asc2desc", "mean"),
                mean_delta_peak2trough=("delta_peak2trough", "mean"),
            )
            .reset_index()
        )
        agg["contact_group"] = stim_state
        rows.append(agg)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out.sort_values(["contact_group", "n_subjects", "n_contacts"], ascending=[True, False, False])


def select_common_regions(summary: pd.DataFrame, min_subjects: int = 2, max_regions: int = 12) -> pd.DataFrame:
    candidates = summary.loc[
        summary["analysis_gray_matter_flag"].fillna(False)
        & summary["contact_group"].isin(["stimulated", "nonstimulated"])
        & (summary["n_subjects"] >= min_subjects)
    ].copy()
    return (
        candidates.sort_values(["contact_group", "n_subjects", "n_contacts"], ascending=[True, False, False])
        .groupby("contact_group", group_keys=False)
        .head(max_regions)
    )


def atlas_region_family(region: object) -> str:
    text = str(region)
    low = text.lower()
    if "hippocampus" in low:
        return "hippocampus"
    if "amygdala" in low:
        return "amygdala"
    if "insula" in low:
        return "insula"
    if "temporal" in low or "fusiform" in low or "parahippocampal" in low:
        return "temporal"
    if "frontal" in low or "acc" in low or "supp_motor" in low or "precentral" in low:
        return "frontal"
    if "parietal" in low or "postcentral" in low or "precuneus" in low:
        return "parietal"
    if "occipital" in low or "calcarine" in low or "cuneus" in low or "lingual" in low:
        return "occipital"
    if is_non_gray_label(text):
        return "non_gray"
    return "other"


def add_region_family(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["region_family"] = out["primary_region"].map(atlas_region_family)
    return out
