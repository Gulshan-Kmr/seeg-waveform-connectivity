from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np

from seeg_waveform.config import load_config
from seeg_waveform.io import iter_mat_files, load_recording
from seeg_waveform.preprocess import post_window_for_frequency, preprocess_segment, window_mask
from seeg_waveform.quinn import analyze_segment
from seeg_waveform.quinn_figures import set_publication_style


def _pick_recording(cfg: dict, file_contains: str | None):
    files = list(iter_mat_files(cfg["project"]["data_dir"]))
    if file_contains:
        matches = [p for p in files if file_contains.lower() in p.stem.lower()]
        if not matches:
            raise ValueError(f"No .mat file matched: {file_contains}")
        return load_recording(matches[0])
    return load_recording(files[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot raw SEEG and IMF1-IMF6 decomposition for one channel/window.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "imf_connectivity_config.yaml"))
    parser.add_argument("--file-contains", default=None, help="Optional substring to choose a specific recording file.")
    parser.add_argument("--channel", default=None, help="Optional channel label. Defaults to first stimulated channel, else channel 0.")
    parser.add_argument("--epoch", choices=["pre", "post"], default="pre")
    parser.add_argument("--seconds", type=float, default=5.0, help="Seconds to display from the selected analysis window.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rec = _pick_recording(cfg, args.file_contains)

    if args.channel:
        if args.channel not in rec.channel_names:
            raise ValueError(f"Channel {args.channel!r} not found in {rec.file_stem}")
        ch_idx = rec.channel_names.index(args.channel)
    elif rec.stimulated_channel_names:
        candidates = [ch for ch in rec.stimulated_channel_names if ch in rec.channel_names]
        ch_idx = rec.channel_names.index(candidates[0]) if candidates else 0
    else:
        ch_idx = 0

    if args.epoch == "pre":
        start, stop = cfg["windows"]["pre"]
    else:
        start, stop = post_window_for_frequency(
            rec.stim_frequency_hz,
            cfg["windows"]["stim_end_by_frequency"],
            cfg["windows"]["post_duration_sec"],
            cfg["windows"]["post_artifact_buffer_sec"],
        )
    mask = window_mask(rec.time, start, stop)
    raw = rec.signal[mask, ch_idx]
    t_abs = rec.time[mask]
    processed, drift, noise = preprocess_segment(raw, t_abs, rec.fs, cfg["preprocessing"])
    result = analyze_segment(processed, rec.fs, cfg["emd"])
    if result.imf is None:
        raise RuntimeError(f"EMD failed for {rec.file_stem} {rec.channel_names[ch_idx]}: {result.status} {result.message}")

    n_show = min(int(args.seconds * rec.fs), len(processed), result.imf.shape[0])
    t = t_abs[:n_show] - t_abs[0]
    imfs = result.imf[:n_show, :6]

    set_publication_style()
    n_rows = 2 + imfs.shape[1]
    fig, axes = plt.subplots(n_rows, 1, figsize=(8.0, 1.15 * n_rows), sharex=True, constrained_layout=True)
    axes[0].plot(t, raw[:n_show], color="0.25", linewidth=0.8)
    axes[0].set_ylabel("Raw")
    axes[0].set_title(
        f"{rec.file_stem} | {rec.channel_names[ch_idx]} | {args.epoch} window | "
        f"{rec.stim_frequency_hz:g} Hz stimulation"
    )
    axes[1].plot(t, processed[:n_show], color="0.15", linewidth=0.8)
    axes[1].set_ylabel("Processed")
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2"]
    for i in range(imfs.shape[1]):
        axes[i + 2].plot(t, imfs[:, i], color=colors[i % len(colors)], linewidth=0.8)
        axes[i + 2].set_ylabel(f"IMF{i + 1}")
    axes[-1].set_xlabel("Time from window start (s)")
    for ax in axes:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    out_dir = Path(cfg["project"]["output_dir"]) / "figures" / "imf_decomposition"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{rec.file_stem}__{rec.channel_names[ch_idx]}__{args.epoch}_IMF1-IMF6.png"
    safe = "".join(c if c.isalnum() or c in "._- " else "_" for c in out_path.name)
    out_path = out_dir / safe
    fig.savefig(out_path, dpi=cfg["outputs"].get("figure_dpi", 300), bbox_inches="tight")
    plt.close(fig)
    print(out_path)
    print(f"status={result.status}, cycles={result.n_cycles}, drift_flag={drift.flagged}, noise_flag={noise.flagged}")


if __name__ == "__main__":
    main()
