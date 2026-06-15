from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .io import SeegRecording, iter_mat_files, load_recording
from .preprocess import preprocess_segment, window_mask
from .region_waveform import atlas_region_family, subject_number
from .spatial_analysis import load_brainstorm_atlas_csv
from .primary_bipolar_mne import (
    NodeWindow,
    _decompose_window,
    _mne_values,
    _profile_rows,
    _selected_imf,
    _segmented_imfs,
    analysis_windows,
    duplicate_recording_exclusions,
    filename_stim_amplitude_ma,
    normalize_contact,
    spectral_rows,
    summarize_imcoh_change,
    summarize_outputs,
    waveform_rows,
    make_figures,
)


@dataclass
class CarRun:
    """A common-average-referenced unipolar run with atlas-localized contacts."""

    recording: SeegRecording
    nodes: pd.DataFrame
    source_recording: SeegRecording
    status: str
    message: str = ""
    reference_pool_channels: tuple[str, ...] = ()


@dataclass
class QcWindow:
    noise_flag: bool
    status: str


def _unipolar_atlas_path(atlas_dir: Path, subject: str) -> Path | None:
    number = subject_number(subject)
    if number is None:
        return None
    hits = sorted(atlas_dir.glob(f"*Subject{number:03d}_unipolar_filtered_Coordinates_and_Atlas.csv"))
    return hits[0] if hits else None


def _make_unreferenced_run(path: str | Path, cfg: dict) -> CarRun:
    """Create atlas-matched unipolar nodes before deriving the CAR signal."""
    source = load_recording(path)
    minimum = int(cfg["preprocessing"].get("min_channels_per_file", 5))
    if source.signal.shape[1] < minimum:
        return CarRun(source, pd.DataFrame(), source, "excluded_few_channels", f"n_channels < {minimum}")

    atlas_dir = Path(cfg["project"]["atlas_dir"])
    atlas_path = _unipolar_atlas_path(atlas_dir, source.subject)
    if atlas_path is None:
        return CarRun(source, pd.DataFrame(), source, "missing_unipolar_atlas", "No matching unipolar atlas CSV")

    atlas = load_brainstorm_atlas_csv(atlas_path, str(cfg["design"].get("primary_atlas", "AAL")))
    atlas["contact_norm"] = atlas["channel"].map(normalize_contact)
    atlas["region_family"] = atlas["primary_region"].map(atlas_region_family)
    atlas_lookup = atlas.drop_duplicates("contact_norm").set_index("contact_norm")
    stim_contacts = {normalize_contact(name) for name in source.stimulated_channel_names}

    rows: list[dict] = []
    keep_indices: list[int] = []
    for source_index, channel in enumerate(source.channel_names):
        norm = normalize_contact(channel)
        if norm not in atlas_lookup.index:
            continue
        atlas_row = atlas_lookup.loc[norm]
        is_stim = norm in stim_contacts
        rows.append(
            {
                "channel": channel,
                "channel_norm": norm,
                "contact_a_norm": norm,
                "contact_b_norm": norm,
                "source_index": source_index,
                "source_index_a": source_index,
                "source_index_b": source_index,
                "is_stimulated_channel": is_stim,
                "shares_stim_contact": is_stim,
                "overlaps_seed": False,
                "primary_region": atlas_row["primary_region"],
                "region_family": atlas_row["region_family"],
                "analysis_gray_matter_flag": bool(atlas_row["analysis_gray_matter_flag"]),
                "region_unambiguous_flag": bool(atlas_row["analysis_gray_matter_flag"]),
                "constituent_region_families": atlas_row["region_family"],
                "x_mni": atlas_row["x_mni"],
                "y_mni": atlas_row["y_mni"],
                "z_mni": atlas_row["z_mni"],
                "atlas_csv": str(atlas_path),
            }
        )
        keep_indices.append(source_index)

    nodes = pd.DataFrame(rows)
    if nodes.empty:
        return CarRun(source, nodes, source, "no_matched_unipolar_nodes", "No atlas contacts matched recorded channels")
    if not nodes["is_stimulated_channel"].any():
        return CarRun(source, nodes, source, "missing_seed_contacts", "No stimulated contact matched the unipolar atlas")

    nodes["primary_region_include"] = (
        nodes["analysis_gray_matter_flag"].astype(bool)
        & nodes["region_unambiguous_flag"].astype(bool)
    )
    matched = replace(
        source,
        signal=source.signal[:, keep_indices],
        channel_names=nodes["channel"].tolist(),
        stimulated_channel_names=nodes.loc[nodes["is_stimulated_channel"], "channel"].tolist(),
    )
    return CarRun(matched, nodes, source, "ok")


def _qc_only_window(rec: SeegRecording, mask: np.ndarray, cfg: dict) -> list[QcWindow]:
    """Run preprocessing/QC without EMD, used only to define the CAR pool."""
    outputs: list[QcWindow] = []
    for index in range(rec.signal.shape[1]):
        _, _, noise = preprocess_segment(rec.signal[mask, index], rec.time[mask], rec.fs, cfg["preprocessing"])
        outputs.append(QcWindow(bool(noise.flagged), "ok"))
    return outputs


def _car_reference_pool(run: CarRun, pre: list[QcWindow], post: list[QcWindow], cfg: dict) -> list[int]:
    """Use clean non-stimulated contacts as the average-reference pool."""
    min_reference = int(cfg["preprocessing"].get("min_car_reference_channels", 5))
    pool = [
        index
        for index, node in run.nodes.iterrows()
        if not bool(node["is_stimulated_channel"])
        and not pre[index].noise_flag
        and not post[index].noise_flag
        and pre[index].status == "ok"
        and post[index].status == "ok"
    ]
    if len(pool) < min_reference:
        return []
    return pool


def _apply_car_reference(run: CarRun, reference_pool: list[int]) -> CarRun:
    reference = np.nanmean(run.recording.signal[:, reference_pool], axis=1, keepdims=True)
    car_signal = run.recording.signal - reference
    car_recording = replace(run.recording, signal=car_signal)
    return replace(
        run,
        recording=car_recording,
        reference_pool_channels=tuple(run.nodes.iloc[reference_pool]["channel"].astype(str).tolist()),
    )


def make_car_run(path: str | Path, cfg: dict, early: bool = False) -> tuple[CarRun, list[NodeWindow] | None, list[NodeWindow] | None]:
    """Build a CAR run using QC-clean non-stimulated contacts as the reference."""
    base = _make_unreferenced_run(path, cfg)
    if base.status != "ok":
        return base, None, None

    pre_range, post_range = analysis_windows(base.recording, cfg, early=early)
    pre_raw = _qc_only_window(base.recording, window_mask(base.recording.time, *pre_range), cfg)
    post_raw = _qc_only_window(base.recording, window_mask(base.recording.time, *post_range), cfg)
    reference_pool = _car_reference_pool(base, pre_raw, post_raw, cfg)
    if not reference_pool:
        minimum = int(cfg["preprocessing"].get("min_car_reference_channels", 5))
        return replace(
            base,
            status="insufficient_clean_car_reference_pool",
            message=f"Fewer than {minimum} clean non-stimulated contacts available for CAR.",
        ), None, None

    car = _apply_car_reference(base, reference_pool)
    pre = _decompose_window(car.recording, window_mask(car.recording.time, *pre_range), cfg)
    post = _decompose_window(car.recording, window_mask(car.recording.time, *post_range), cfg)
    return car, pre, post


def _share_car_contact(left_node: pd.Series, right_node: pd.Series) -> bool:
    return str(left_node["channel_norm"]) == str(right_node["channel_norm"])


def connectivity_rows(run: CarRun, pre: list[NodeWindow], post: list[NodeWindow], cfg: dict) -> pd.DataFrame:
    """Compute CAR unipolar MNE connectivity while allowing multiple stimulated seeds."""
    nodes = run.nodes
    rows: list[dict] = []
    imf_numbers = [int(value) for value in cfg["design"].get("connectivity_imfs", [cfg["design"]["primary_imf"]])]
    primary_imf = int(cfg["design"]["primary_imf"])
    for imf_number in imf_numbers:
        valid = [
            index
            for index in range(len(nodes))
            if not pre[index].noise_flag
            and not post[index].noise_flag
            and _selected_imf(pre[index].result, imf_number) is not None
            and _selected_imf(post[index].result, imf_number) is not None
        ]
        seed_hits = [index for index in valid if bool(nodes.iloc[index]["is_stimulated_channel"])]
        targets = [index for index in valid if index not in seed_hits and not bool(nodes.iloc[index]["is_stimulated_channel"])]
        if not seed_hits or not targets:
            continue

        remotes = targets
        edge_pairs = [(seed, target, "seed_to_target") for seed in seed_hits for target in targets]
        edge_pairs += [(left, right, "remote_to_remote") for left, right in combinations(remotes, 2)]
        local_index = {node_index: i for i, node_index in enumerate(valid)}
        pairs = [(local_index[left], local_index[right]) for left, right, _ in edge_pairs]
        pre_vals = _mne_values(_segmented_imfs(pre, valid, run.recording, cfg, imf_number), pairs, run.recording.fs, cfg)
        post_vals = _mne_values(_segmented_imfs(post, valid, run.recording, cfg, imf_number), pairs, run.recording.fs, cfg)
        samples = int(round(float(cfg["connectivity"]["segment_length_sec"]) * run.recording.fs))
        n_segments_used = min(int(cfg["connectivity"]["n_segments"]), len(pre[valid[0]].processed) // samples)
        seed_xyzs = nodes.iloc[seed_hits][["x_mni", "y_mni", "z_mni"]].to_numpy(float)

        for edge_index, (left, right, edge_type) in enumerate(edge_pairs):
            left_node, right_node = nodes.iloc[left], nodes.iloc[right]
            left_xyz = left_node[["x_mni", "y_mni", "z_mni"]].to_numpy(float)
            right_xyz = right_node[["x_mni", "y_mni", "z_mni"]].to_numpy(float)
            if edge_type == "seed_to_target":
                seed_xyz = left_xyz
                edge_distance_to_seed = float(np.linalg.norm(right_xyz - seed_xyz))
            else:
                edge_distance_to_seed = float(
                    min(
                        np.nanmin(np.linalg.norm(seed_xyzs - left_xyz, axis=1)),
                        np.nanmin(np.linalg.norm(seed_xyzs - right_xyz, axis=1)),
                    )
                )
            pre_signed = float(pre_vals["imcoh"][edge_index])
            post_signed = float(post_vals["imcoh"][edge_index])
            analysis_include = (
                bool(right_node["primary_region_include"])
                if edge_type == "seed_to_target"
                else bool(left_node["primary_region_include"] and right_node["primary_region_include"])
            )
            row = {
                "file": run.recording.file_stem,
                "subject": run.recording.subject,
                "stim_frequency_hz": run.recording.stim_frequency_hz,
                "stim_amplitude_ma": run.recording.stim_amplitude_ma,
                "reference": "clean_nonstim_common_average",
                "n_car_reference_channels": len(run.reference_pool_channels),
                "imf": imf_number,
                "imf_role": "primary" if imf_number == primary_imf else ("secondary" if imf_number <= 3 else "exploratory"),
                "edge_type": edge_type,
                "n_segments_used": n_segments_used,
                "seed_channel": left_node["channel"] if edge_type == "seed_to_target" else ";".join(nodes.iloc[seed_hits]["channel"].astype(str)),
                "node_a": left_node["channel"],
                "node_b": right_node["channel"],
                "target_channel": right_node["channel"] if edge_type == "seed_to_target" else "",
                "node_a_contact_a": left_node["contact_a_norm"],
                "node_a_contact_b": left_node["contact_b_norm"],
                "node_b_contact_a": right_node["contact_a_norm"],
                "node_b_contact_b": right_node["contact_b_norm"],
                "shared_constituent_contact": _share_car_contact(left_node, right_node),
                "node_a_region_family": left_node["region_family"],
                "node_b_region_family": right_node["region_family"],
                "node_a_primary_include": bool(left_node["primary_region_include"]),
                "node_b_primary_include": bool(right_node["primary_region_include"]),
                "distance_mm": float(np.linalg.norm(left_xyz - right_xyz)),
                "node_a_distance_to_seed_mm": float(np.nanmin(np.linalg.norm(seed_xyzs - left_xyz, axis=1))),
                "node_b_distance_to_seed_mm": float(np.nanmin(np.linalg.norm(seed_xyzs - right_xyz, axis=1))),
                "edge_distance_to_seed_mm": edge_distance_to_seed,
                "pre_coh": float(pre_vals["coh"][edge_index]),
                "post_coh": float(post_vals["coh"][edge_index]),
                "delta_coh": float(post_vals["coh"][edge_index] - pre_vals["coh"][edge_index]),
                "analysis_include": analysis_include,
            }
            row.update(summarize_imcoh_change(pre_signed, post_signed))
            rows.append(row)
    return pd.DataFrame(rows)


def run_analysis(
    config_path: str | Path,
    max_files: int | None = None,
    early: bool = False,
    file_path: str | Path | None = None,
) -> dict[str, Path]:
    cfg = load_config(config_path)
    root = Path(cfg["project"]["output_dir"]) / ("early_10s_guarded_sensitivity" if early else "primary_28s_guarded")
    tables = root / "tables"
    figures = root / "figures"
    tables.mkdir(parents=True, exist_ok=True)

    waveform_frames: list[pd.DataFrame] = []
    spectral_frames: list[pd.DataFrame] = []
    connectivity_frames: list[pd.DataFrame] = []
    profile_rows: list[dict] = []
    qc_rows: list[dict] = []
    connectivity_qc_rows: list[dict] = []

    files = [Path(file_path)] if file_path is not None else list(iter_mat_files(cfg["project"]["data_dir"]))
    if file_path is None and max_files is not None:
        files = files[:max_files]
    duplicate_exclusions = duplicate_recording_exclusions(files) if file_path is None else {}
    primary_imf = int(cfg["design"]["primary_imf"])

    for path in files:
        run, pre, post = make_car_run(path, cfg, early=early)
        filename_current = filename_stim_amplitude_ma(path)
        metadata_match = bool(np.isclose(filename_current, run.source_recording.stim_amplitude_ma, equal_nan=False))
        status = "excluded_duplicate_recording" if path in duplicate_exclusions else run.status
        message = duplicate_exclusions[path] if path in duplicate_exclusions else run.message
        seed_channels = run.nodes.loc[run.nodes.get("is_stimulated_channel", pd.Series(dtype=bool)).fillna(False), "channel"].astype(str).tolist() if not run.nodes.empty else []
        qc = {
            "file": run.source_recording.file_stem,
            "subject": run.source_recording.subject,
            "stim_frequency_hz": run.source_recording.stim_frequency_hz,
            "stim_amplitude_ma": run.source_recording.stim_amplitude_ma,
            "filename_stim_amplitude_ma": filename_current,
            "filename_current_matches_metadata": metadata_match,
            "reference": "clean_nonstim_common_average",
            "status": status,
            "message": message,
            "n_unipolar_channels": run.source_recording.signal.shape[1],
            "n_atlas_matched_unipolar_nodes": len(run.nodes),
            "n_car_reference_channels": len(run.reference_pool_channels),
            "car_reference_channels": ";".join(run.reference_pool_channels),
            "stimulated_seed_channels": ";".join(seed_channels),
        }
        qc_rows.append(qc)
        if status != "ok" or pre is None or post is None:
            connectivity_qc_rows.append({**qc, "connectivity_status": status, "n_edges": 0, "n_primary_edges": 0})
            continue

        waveform_frames.append(waveform_rows(run, pre, post))
        spectral_frames.append(spectral_rows(run, pre, post, cfg))
        connectivity = connectivity_rows(run, pre, post, cfg)
        connectivity_frames.append(connectivity)

        seed_indices = [index for index, node in run.nodes.iterrows() if bool(node["is_stimulated_channel"])]
        seed_imf_available = any(
            _selected_imf(pre[index].result, primary_imf) is not None and _selected_imf(post[index].result, primary_imf) is not None
            for index in seed_indices
        )
        seed_noise = any(pre[index].noise_flag or post[index].noise_flag for index in seed_indices)
        if seed_noise:
            conn_status = "excluded_all_or_some_seed_noise"
        elif not seed_imf_available:
            conn_status = "excluded_seed_missing_imf"
        elif connectivity.empty:
            conn_status = "excluded_no_valid_edges"
        else:
            conn_status = "ok"
        connectivity_qc_rows.append(
            {
                **qc,
                "connectivity_status": conn_status,
                "seed_channel": ";".join(seed_channels),
                "seed_imf_available": seed_imf_available,
                "n_edges": len(connectivity),
                "n_primary_edges": int(connectivity.loc[connectivity["imf"].eq(primary_imf), "analysis_include"].sum())
                if not connectivity.empty
                else 0,
            }
        )
        profile_rows.extend(_profile_rows(run, pre, post))

    qc = pd.DataFrame(qc_rows)
    waveform = pd.concat(waveform_frames, ignore_index=True) if waveform_frames else pd.DataFrame()
    spectral = pd.concat(spectral_frames, ignore_index=True) if spectral_frames else pd.DataFrame()
    connectivity = pd.concat(connectivity_frames, ignore_index=True) if connectivity_frames else pd.DataFrame()
    profiles = pd.DataFrame(profile_rows)
    outputs = {
        "run_qc": qc,
        "connectivity_qc": pd.DataFrame(connectivity_qc_rows),
        "waveform_nodes": waveform,
        "spectral_nodes": spectral,
        "connectivity_edges": connectivity,
        "waveform_profiles": profiles,
    }
    paths: dict[str, Path] = {}
    for name, table in outputs.items():
        path = tables / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path

    if not waveform.empty and not connectivity.empty and not spectral.empty:
        summaries = summarize_outputs(waveform, connectivity, spectral)
        for name, table in summaries.items():
            path = tables / f"{name}.csv"
            table.to_csv(path, index=False)
            paths[name] = path
        for index, path in enumerate(
            make_figures(qc, waveform, connectivity, spectral, profiles, summaries, figures, int(cfg["outputs"].get("figure_dpi", 300))),
            start=1,
        ):
            paths[f"figure_{index:02d}"] = path
    return paths
