from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seeg_waveform.config import load_config
from seeg_waveform.io import iter_mat_files, load_recording
from seeg_waveform.pipeline import analyze_channel


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small SEEG waveform smoke test.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "analysis_config.yaml"))
    parser.add_argument("--file", default=None, help="Optional .mat file. Defaults to the first data file.")
    parser.add_argument("--channel-index", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    path = Path(args.file) if args.file else next(iter_mat_files(cfg["project"]["data_dir"]))
    rec = load_recording(path)

    print(f"file: {rec.path}")
    print(f"entry: {rec.entry_index}")
    print(f"signal shape: {rec.signal.shape}")
    print(f"time range: {rec.time.min():.3f} to {rec.time.max():.3f} s")
    print(f"fs: {rec.fs:.3f} Hz")
    print(f"stim: {rec.stim_frequency_hz:g} Hz, {rec.stim_amplitude_ma:g} mA")
    print(f"channels: {len(rec.channel_names)}")
    print(f"stimulated channels: {rec.stimulated_channel_names}")

    row = analyze_channel(rec, args.channel_index, cfg, file_out_dir=None)
    print("\nchannel smoke result:")
    for key in [
        "channel",
        "status",
        "pre_status",
        "post_status",
        "pre_n_cycles",
        "post_n_cycles",
        "pre_drift_flag",
        "post_drift_flag",
        "delta_mean_if",
        "delta_asc2desc",
        "delta_peak2trough",
    ]:
        if key in row:
            print(f"{key}: {row[key]}")


if __name__ == "__main__":
    main()
