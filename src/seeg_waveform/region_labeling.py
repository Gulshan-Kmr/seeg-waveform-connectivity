from __future__ import annotations

import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def load_freesurfer_lut(path: Path | None = None) -> dict[int, str]:
    """Load a FreeSurferColorLUT-style table.

    If no LUT is supplied, a compact fallback covers common aseg/aparc+aseg labels.
    """
    fallback = {
        0: "Unknown",
        2: "Left-Cerebral-White-Matter",
        3: "Left-Cerebral-Cortex",
        4: "Left-Lateral-Ventricle",
        7: "Left-Cerebellum-White-Matter",
        8: "Left-Cerebellum-Cortex",
        10: "Left-Thalamus",
        11: "Left-Caudate",
        12: "Left-Putamen",
        13: "Left-Pallidum",
        16: "Brain-Stem",
        17: "Left-Hippocampus",
        18: "Left-Amygdala",
        26: "Left-Accumbens-area",
        28: "Left-VentralDC",
        41: "Right-Cerebral-White-Matter",
        42: "Right-Cerebral-Cortex",
        43: "Right-Lateral-Ventricle",
        46: "Right-Cerebellum-White-Matter",
        47: "Right-Cerebellum-Cortex",
        49: "Right-Thalamus",
        50: "Right-Caudate",
        51: "Right-Putamen",
        52: "Right-Pallidum",
        53: "Right-Hippocampus",
        54: "Right-Amygdala",
        58: "Right-Accumbens-area",
        60: "Right-VentralDC",
    }
    if path is None or not path.exists():
        return fallback
    lut: dict[int, str] = {}
    for line in path.read_text(errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2 or not parts[0].lstrip("-").isdigit():
            continue
        lut[int(parts[0])] = parts[1]
    fallback.update(lut)
    return fallback


def infer_lobe(region: str) -> str:
    text = region.lower()
    if any(k in text for k in ["frontal", "precentral", "pars", "orbitofrontal", "rostralmiddlefrontal", "caudalmiddlefrontal"]):
        return "frontal"
    if any(k in text for k in ["temporal", "fusiform", "entorhinal", "parahippocampal", "bankssts"]):
        return "temporal"
    if any(k in text for k in ["parietal", "postcentral", "precuneus", "supramarginal", "inferiorparietal", "superiorparietal"]):
        return "parietal"
    if any(k in text for k in ["occipital", "cuneus", "lingual", "pericalcarine", "lateraloccipital"]):
        return "occipital"
    if any(k in text for k in ["insula"]):
        return "insula"
    if any(k in text for k in ["hippocampus", "amygdala"]):
        return "mesial_temporal"
    if any(k in text for k in ["thalamus", "caudate", "putamen", "pallidum", "accumbens", "ventraldc"]):
        return "subcortical"
    if any(k in text for k in ["cerebellum"]):
        return "cerebellum"
    if "unknown" in text:
        return "unknown"
    return "other"


def infer_hemisphere(region: str, x_mm: float | None = None) -> str:
    lower = region.lower()
    if lower.startswith("left-") or lower.startswith("ctx-lh"):
        return "left"
    if lower.startswith("right-") or lower.startswith("ctx-rh"):
        return "right"
    if x_mm is None or not np.isfinite(x_mm):
        return "unknown"
    return "left" if x_mm < 0 else "right"


def voxel_to_world(vox: np.ndarray, affine: np.ndarray) -> np.ndarray:
    hom = np.c_[vox, np.ones(len(vox))]
    return (hom @ affine.T)[:, :3]


def world_to_voxel(xyz_mm: np.ndarray, affine: np.ndarray) -> np.ndarray:
    hom = np.c_[xyz_mm, np.ones(len(xyz_mm))]
    return (hom @ np.linalg.inv(affine).T)[:, :3]


def label_from_volume(
    coords: pd.DataFrame,
    atlas_volume: Path,
    lut_path: Path | None = None,
    search_radius_mm: float = 2.0,
) -> pd.DataFrame:
    img = nib.load(atlas_volume)
    data = np.asanyarray(img.dataobj)
    lut = load_freesurfer_lut(lut_path)
    xyz = coords[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=float)
    vox = world_to_voxel(xyz, img.affine)
    ijk = np.round(vox).astype(int)
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
    radius_vox = np.maximum(1, np.ceil(search_radius_mm / zooms).astype(int))
    shape = np.asarray(data.shape)
    rows = []
    for contact, v, world in zip(coords.itertuples(index=False), ijk, xyz):
        inside = bool(np.all((v >= 0) & (v < shape)))
        nearest_label = int(data[tuple(v)]) if inside else 0
        label_value = nearest_label
        distance_mm = 0.0 if label_value != 0 and inside else np.nan
        if inside and label_value == 0:
            lo = np.maximum(v - radius_vox, 0)
            hi = np.minimum(v + radius_vox + 1, shape)
            block = data[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
            nz = np.argwhere(block != 0)
            if len(nz):
                candidates = nz + lo
                candidate_world = voxel_to_world(candidates, img.affine)
                dists = np.linalg.norm(candidate_world - world[None, :], axis=1)
                best = int(np.argmin(dists))
                label_value = int(data[tuple(candidates[best])])
                distance_mm = float(dists[best])
        region = lut.get(label_value, f"Label-{label_value}")
        rows.append(
            {
                "channel": contact.channel,
                "channel_norm": contact.channel_norm,
                "shaft": contact.shaft,
                "x_mm": contact.x_mm,
                "y_mm": contact.y_mm,
                "z_mm": contact.z_mm,
                "atlas_label_value": label_value,
                "atlas_region": region,
                "atlas_lobe": infer_lobe(region),
                "atlas_hemisphere": infer_hemisphere(region, contact.x_mm),
                "atlas_distance_mm": distance_mm,
                "atlas_source": str(atlas_volume),
                "atlas_inside_volume": inside,
            }
        )
    return pd.DataFrame(rows)


def normalize_label_channel(name: object) -> str:
    return re.sub(r"\s+", "", str(name).strip()).upper()


def label_from_table(coords: pd.DataFrame, label_table: Path) -> pd.DataFrame:
    labels = pd.read_csv(label_table)
    if "channel_norm" not in labels.columns:
        if "channel" not in labels.columns:
            raise ValueError("Label table must contain either 'channel_norm' or 'channel'.")
        labels = labels.copy()
        labels["channel_norm"] = labels["channel"].map(normalize_label_channel)
    region_col = "region_label" if "region_label" in labels.columns else "atlas_region"
    if region_col not in labels.columns:
        raise ValueError("Label table must contain 'region_label' or 'atlas_region'.")
    keep_cols = ["channel_norm", region_col]
    optional = [c for c in ["hemisphere", "lobe", "notes"] if c in labels.columns]
    merged = coords.merge(labels[keep_cols + optional], on="channel_norm", how="left")
    merged["atlas_region"] = merged[region_col].fillna("unlabeled")
    merged["atlas_lobe"] = merged["lobe"] if "lobe" in merged else merged["atlas_region"].map(infer_lobe)
    merged["atlas_hemisphere"] = (
        merged["hemisphere"]
        if "hemisphere" in merged
        else [infer_hemisphere(r, x) for r, x in zip(merged["atlas_region"], merged["x_mm"])]
    )
    merged["atlas_source"] = str(label_table)
    return merged


def summarize_regions(labels: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["atlas_hemisphere", "atlas_lobe", "atlas_region"]
    return (
        labels.groupby(group_cols, dropna=False)
        .agg(n_contacts=("channel_norm", "nunique"), shafts=("shaft", lambda x: ";".join(sorted(set(map(str, x))))))
        .reset_index()
        .sort_values(["atlas_hemisphere", "atlas_lobe", "atlas_region"])
    )
