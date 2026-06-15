from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seeg_waveform.quinn_figures import set_publication_style


def normalized_signal(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x / np.nanmax(np.abs(x))


def _phase_from_control_fractions(fractions: list[float], n_samples: int) -> np.ndarray:
    target_phase = np.asarray([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], dtype=float)
    target_sample = np.asarray(fractions, dtype=float) * (n_samples - 1)
    interpolator = PchipInterpolator(target_sample, target_phase)
    samples = np.arange(n_samples)
    phase = interpolator(samples)
    return np.clip(phase, 0, 2 * np.pi)


def simulated_cycles(n_samples: int = 512) -> tuple[np.ndarray, list[tuple[str, np.ndarray, np.ndarray]]]:
    phase = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    sinusoid_phase = _phase_from_control_fractions([0.0, 0.25, 0.50, 0.75, 1.0], n_samples)
    sharp_peak_phase = _phase_from_control_fractions([0.0, 0.18, 0.50, 0.82, 1.0], n_samples)
    broad_peak_phase = _phase_from_control_fractions([0.0, 0.34, 0.50, 0.66, 1.0], n_samples)
    asymmetric_phase = _phase_from_control_fractions([0.0, 0.20, 0.43, 0.76, 1.0], n_samples)
    return phase, [
        ("Sinusoid", normalized_signal(np.sin(sinusoid_phase)), sinusoid_phase),
        ("Sharp peak", normalized_signal(np.sin(sharp_peak_phase)), sharp_peak_phase),
        ("Broad peak", normalized_signal(np.sin(broad_peak_phase)), broad_peak_phase),
        ("Asymmetric cycle", normalized_signal(np.sin(asymmetric_phase)), asymmetric_phase),
    ]


def instantaneous_frequency(inst_phase: np.ndarray) -> np.ndarray:
    ifreq = np.gradient(np.unwrap(inst_phase))
    return ifreq / np.nanmean(ifreq)


def control_points(signal: np.ndarray) -> dict[str, int]:
    n = len(signal)
    peak = int(np.nanargmax(signal))
    trough = int(np.nanargmin(signal))
    rising = []
    falling = []
    for idx in range(n - 1):
        if signal[idx] <= 0 < signal[idx + 1]:
            rising.append(idx)
        if signal[idx] >= 0 > signal[idx + 1]:
            falling.append(idx)
    return {
        "rise_zero": rising[0] if rising else 0,
        "peak": peak,
        "fall_zero": falling[0] if falling else n // 2,
        "trough": trough,
    }


def control_durations(points: dict[str, int], n_samples: int) -> tuple[list[str], np.ndarray]:
    names = ["rise-to-peak", "peak-to-fall", "fall-to-trough", "trough-to-rise"]
    ordered = [points["rise_zero"], points["peak"], points["fall_zero"], points["trough"], points["rise_zero"] + n_samples]
    ordered = np.asarray(ordered, dtype=float)
    durations = np.diff(ordered) / float(n_samples)
    return names, durations


def make_schematic(out_path: Path) -> Path:
    set_publication_style()
    phase, cycles = simulated_cycles()
    sample_rate = len(phase)
    fig, axes = plt.subplots(4, 4, figsize=(10.6, 6.8), sharex="col", constrained_layout=True)
    sinusoid = cycles[0][1]
    control_colors = ["#4361ee", "#2a9d8f", "#e76f51", "#9d4edd"]
    for col, (title, signal, inst_phase) in enumerate(cycles):
        ifreq = instantaneous_frequency(inst_phase)
        points = control_points(signal)
        names, durations = control_durations(points, len(signal))

        axes[0, col].plot(phase, signal, color="0.1", lw=1.7)
        if col > 0:
            axes[0, col].plot(phase, sinusoid, color="0.35", lw=1.0, ls="--")
        for point in points.values():
            axes[0, col].axvline(phase[point % len(phase)], color="0.85", lw=0.65)
        axes[0, col].set_title(title)

        axes[1, col].plot(phase, np.mod(inst_phase, 2 * np.pi), color="#5a189a", lw=1.5)
        axes[1, col].set_ylim(-0.2, 2 * np.pi + 0.2)
        axes[1, col].set_yticks([0, np.pi, 2 * np.pi], ["0", r"$\pi$", r"$2\pi$"] if col == 0 else [])

        axes[2, col].plot(phase, ifreq, color="#d1495b", lw=1.5)
        axes[2, col].axhline(1.0, color="0.45", lw=0.7, ls="--")
        axes[2, col].set_ylim(0.0, 2.05)

        x = np.arange(len(names))
        axes[3, col].bar(x, durations, color=control_colors, alpha=0.75, width=0.72)
        axes[3, col].axhline(0.25, color="0.35", lw=0.8, ls="--")
        axes[3, col].set_ylim(0, max(0.5, float(np.nanmax(durations)) * 1.2))
        axes[3, col].set_xticks(x, ["R-P", "P-F", "F-T", "T-R"], rotation=30, ha="right")

        for row in range(4):
            axes[row, col].spines["top"].set_visible(False)
            axes[row, col].spines["right"].set_visible(False)
            axes[row, col].grid(axis="y", color="0.92", lw=0.4)
            if row != 1:
                axes[row, col].yaxis.get_major_formatter().set_useOffset(False)

    axes[0, 0].set_ylabel("Waveform\n(a.u.)")
    axes[1, 0].set_ylabel("Inst. phase")
    axes[2, 0].set_ylabel("Inst. frequency\n(a.u.)")
    axes[3, 0].set_ylabel("Cycle fraction")
    for col in range(4):
        axes[3, col].set_xlabel("Control-point interval")
        axes[2, col].set_xticks([0, np.pi, 2 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
    fig.suptitle("Conceptual link between waveform shape and within-cycle instantaneous frequency", y=1.02)
    fig.text(
        0.5,
        -0.015,
        "Dotted line in waveform panels shows a sinusoidal reference; dotted line in duration panels shows equal quarter-cycle timing.",
        ha="center",
        va="top",
        fontsize=8,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create conceptual within-cycle waveform schematic for manuscript Figure 2.")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs"
        / "publication_bipolar_mne"
        / "primary_28s_guarded"
        / "figures"
        / "corrected_addons"
        / "figure_02a_within_cycle_waveform_schematic.png",
    )
    args = parser.parse_args()
    print(make_schematic(args.out))


if __name__ == "__main__":
    main()
