from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seeg_waveform.region_quinn_figures import make_region_quinn_figures
from seeg_waveform.region_waveform import (
    add_region_family,
    common_region_summary,
    join_waveform_to_atlas,
    select_common_regions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Quinn-style waveform figures by anatomical region.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "analysis_config.yaml"))
    parser.add_argument(
        "--waveform-summary",
        type=Path,
        default=Path(r"C:\Data_Science\seeg\outputs\waveform_relaxed\MASTER_all_channels_summary.csv"),
    )
    parser.add_argument(
        "--atlas-root",
        type=Path,
        default=Path(r"C:\Data_Science\seeg\data\SEEG_Coordinates"),
    )
    parser.add_argument("--primary-atlas", default="AAL")
    parser.add_argument("--min-subjects", type=int, default=2)
    parser.add_argument("--max-regions-per-group", type=int, default=10)
    parser.add_argument("--min-contacts-for-figure", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\Data_Science\seeg\outputs\waveform_relaxed\region_quinn"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged, atlas = join_waveform_to_atlas(args.waveform_summary, args.atlas_root, args.primary_atlas)
    merged = add_region_family(merged)
    merged_csv = args.output_dir / "MASTER_waveform_with_brainstorm_regions.csv"
    atlas_csv = args.output_dir / "MASTER_brainstorm_atlas_contacts.csv"
    summary_csv = args.output_dir / "common_region_summary.csv"
    family_summary_csv = args.output_dir / "common_region_family_summary.csv"
    selected_csv = args.output_dir / "selected_common_regions.csv"

    merged.to_csv(merged_csv, index=False)
    atlas.to_csv(atlas_csv, index=False)
    summary = common_region_summary(merged)
    summary.to_csv(summary_csv, index=False)
    family_summary = (
        merged.loc[merged["has_atlas_label"].fillna(False)]
        .groupby(["region_family", "analysis_gray_matter_flag", "is_stimulated_channel"], dropna=False)
        .agg(
            n_rows=("channel_norm", "size"),
            n_contacts=("channel_norm", "nunique"),
            n_subjects=("subject_number", "nunique"),
            n_files=("file", "nunique"),
            mean_delta_mean_if=("delta_mean_if", "mean"),
            mean_delta_asc2desc=("delta_asc2desc", "mean"),
            mean_delta_peak2trough=("delta_peak2trough", "mean"),
        )
        .reset_index()
        .sort_values(["is_stimulated_channel", "n_subjects", "n_contacts"], ascending=[False, False, False])
    )
    family_summary.to_csv(family_summary_csv, index=False)
    selected = select_common_regions(summary, min_subjects=args.min_subjects, max_regions=args.max_regions_per_group)
    selected.to_csv(selected_csv, index=False)

    paths = make_region_quinn_figures(
        args.config,
        merged_csv,
        selected_csv,
        output_dir=args.output_dir / "figures",
        min_contacts=args.min_contacts_for_figure,
    )

    matched = merged["has_atlas_label"].fillna(False)
    print(f"waveform rows: {len(merged)}")
    print(f"atlas-matched rows: {int(matched.sum())}/{len(merged)}")
    print(f"atlas subjects: {atlas['subject_number'].nunique()}")
    print(f"waveform subjects with atlas: {merged.loc[matched, 'subject_number'].nunique()}")
    print(f"common region summary: {summary_csv}")
    print(f"common family summary: {family_summary_csv}")
    print(f"selected regions: {selected_csv}")
    print(f"created {len(paths)} region Quinn figures")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
