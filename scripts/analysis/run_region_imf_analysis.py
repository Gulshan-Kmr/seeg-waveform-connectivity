from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seeg_waveform.region_imf_analysis import (
    compare_frequency_region_imf,
    distance_bin_region_imf_summary,
    imf_waveform_coupling,
    join_imf_connectivity_to_regions,
    leave_one_subject_out_frequency_effect,
    plot_distance_bin_profiles,
    plot_imf4_coupling,
    plot_leave_one_subject_out,
    plot_frequency_contrast_heatmap,
    plot_region_imf_file_profiles,
    plot_region_imf_heatmap,
    run_region_imf_mixed_models,
    subject_level_region_imf_summary,
    summarize_region_imf,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Region-wise IMF connectivity and waveform coupling.")
    parser.add_argument(
        "--imf-connectivity",
        type=Path,
        default=Path(r"C:\Data_Science\seeg\outputs\imf_connectivity\imf_seed_to_nonstim_target_connectivity.csv"),
    )
    parser.add_argument(
        "--waveform-regions",
        type=Path,
        default=Path(r"C:\Data_Science\seeg\outputs\waveform_relaxed\region_quinn\MASTER_waveform_with_brainstorm_regions.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\Data_Science\seeg\outputs\imf_connectivity\region_imf"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    joined = join_imf_connectivity_to_regions(args.imf_connectivity, args.waveform_regions)
    joined_path = args.output_dir / "imf_connectivity_waveform_regions_join.csv"
    joined.to_csv(joined_path, index=False)

    pair_summary = summarize_region_imf(joined, level="pair")
    file_summary = summarize_region_imf(joined, level="file_mean")
    summary = pd.concat([pair_summary, file_summary], ignore_index=True)
    summary_path = args.output_dir / "region_family_imf_connectivity_summary.csv"
    summary.to_csv(summary_path, index=False)

    pair_coupling = imf_waveform_coupling(joined, level="pair")
    target_coupling = imf_waveform_coupling(joined, level="file_target_mean")
    coupling = pd.concat([pair_coupling, target_coupling], ignore_index=True)
    coupling_path = args.output_dir / "region_family_imf_waveform_coupling.csv"
    coupling.to_csv(coupling_path, index=False)

    contrast = compare_frequency_region_imf(joined)
    contrast_path = args.output_dir / "region_family_imf_50hz_vs_1hz_contrast.csv"
    contrast.to_csv(contrast_path, index=False)

    subject_summary = subject_level_region_imf_summary(joined)
    subject_summary_path = args.output_dir / "subject_level_region_imf_summary.csv"
    subject_summary.to_csv(subject_summary_path, index=False)

    sensitivity = leave_one_subject_out_frequency_effect(joined)
    sensitivity_path = args.output_dir / "leave_one_subject_out_frequency_effect.csv"
    sensitivity.to_csv(sensitivity_path, index=False)

    distance_summary = distance_bin_region_imf_summary(joined)
    distance_summary_path = args.output_dir / "distance_bin_region_imf_summary.csv"
    distance_summary.to_csv(distance_summary_path, index=False)

    mixed = run_region_imf_mixed_models(joined)
    mixed_path = args.output_dir / "region_family_imf_mixed_models.csv"
    mixed.to_csv(mixed_path, index=False)

    heat_imcoh = plot_region_imf_heatmap(summary, fig_dir / "region_family_imf_imcoh_heatmap.png", metric="delta_imcoh_mean")
    heat_coh = plot_region_imf_heatmap(summary, fig_dir / "region_family_imf_coh_heatmap.png", metric="delta_coh_mean")
    profile_imcoh = plot_region_imf_file_profiles(joined, fig_dir / "region_family_imf_imcoh_profiles.png", metric="delta_imcoh")
    profile_coh = plot_region_imf_file_profiles(joined, fig_dir / "region_family_imf_coh_profiles.png", metric="delta_coh")
    contrast_imcoh = plot_frequency_contrast_heatmap(
        contrast, fig_dir / "region_family_imf_50hz_vs_1hz_imcoh_contrast.png", metric="delta_imcoh"
    )
    contrast_coh = plot_frequency_contrast_heatmap(
        contrast, fig_dir / "region_family_imf_50hz_vs_1hz_coh_contrast.png", metric="delta_coh"
    )
    sensitivity_imcoh = plot_leave_one_subject_out(
        sensitivity, fig_dir / "leave_one_subject_out_imcoh_frequency_effect.png", metric="delta_imcoh"
    )
    distance_imcoh = plot_distance_bin_profiles(
        distance_summary, fig_dir / "distance_bin_imf_imcoh_profiles.png", metric="delta_imcoh"
    )
    distance_coh = plot_distance_bin_profiles(
        distance_summary, fig_dir / "distance_bin_imf_coh_profiles.png", metric="delta_coh"
    )
    imf4 = plot_imf4_coupling(joined, fig_dir / "imf4_connectivity_waveform_coupling.png")

    print(f"joined rows: {len(joined)}")
    print(f"included rows: {int(joined['analysis_pair_include'].sum())}")
    print(f"joined table: {joined_path}")
    print(f"summary: {summary_path}")
    print(f"coupling: {coupling_path}")
    print(f"frequency contrast: {contrast_path}")
    print(f"subject-level summary: {subject_summary_path}")
    print(f"leave-one-subject-out: {sensitivity_path}")
    print(f"distance-bin summary: {distance_summary_path}")
    print(f"mixed models: {mixed_path}")
    print(f"figure: {heat_imcoh}")
    print(f"figure: {heat_coh}")
    print(f"figure: {profile_imcoh}")
    print(f"figure: {profile_coh}")
    print(f"figure: {contrast_imcoh}")
    print(f"figure: {contrast_coh}")
    print(f"figure: {sensitivity_imcoh}")
    print(f"figure: {distance_imcoh}")
    print(f"figure: {distance_coh}")
    print(f"figure: {imf4}")


if __name__ == "__main__":
    main()
