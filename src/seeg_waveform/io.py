from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterator

import h5py
import numpy as np


@dataclass
class SeegRecording:
    path: Path
    entry_index: int
    file_stem: str
    signal: np.ndarray
    time: np.ndarray
    fs: float
    channel_names: list[str]
    stimulated_channel_names: list[str]
    stim_frequency_hz: float
    stim_amplitude_ma: float
    subject: str


def iter_mat_files(data_dir: str | Path) -> Iterator[Path]:
    yield from sorted(Path(data_dir).glob("F*/*.mat"))


def parse_subject_from_name(path: str | Path) -> str:
    name = Path(path).stem
    match = re.search(r"(HEJ_)?Subject\d+", name)
    return match.group(0) if match else "unknown"


def h5_string_from_any(h5: h5py.File, obj) -> str:
    """Decode MATLAB v7.3 strings stored as refs, char codes, bytes, or arrays."""
    if isinstance(obj, h5py.Reference):
        if not obj:
            return ""
        obj = h5[obj]

    if hasattr(obj, "__getitem__") and not isinstance(obj, np.ndarray):
        try:
            obj = obj[()]
        except Exception:
            return str(obj)

    arr = np.asarray(obj)
    if arr.size == 0:
        return ""

    if arr.dtype == object:
        parts = [h5_string_from_any(h5, x).strip() for x in arr.ravel()]
        return " ".join([p for p in parts if p]).strip()

    if np.issubdtype(arr.dtype, np.integer):
        return "".join(chr(int(x)) for x in arr.ravel() if int(x) != 0).strip()

    parts: list[str] = []
    for x in arr.ravel():
        if isinstance(x, bytes):
            parts.append(x.decode("utf-8", errors="ignore"))
        else:
            parts.append(str(x))
    return "".join(parts).strip()


def strings_from_dataset(h5: h5py.File, dataset) -> list[str]:
    out = []
    arr = np.asarray(dataset[()])
    for item in arr.ravel():
        s = h5_string_from_any(h5, item).strip()
        if s:
            out.append(s)
    return out


def strings_from_ref_dataset(h5: h5py.File, dataset, entry_index: int) -> list[str]:
    ref = dataset[entry_index, 0]
    if not ref:
        return []
    return strings_from_dataset(h5, h5[ref])


def extract_channel_names_from_group(h5: h5py.File, channel_file_group) -> list[str]:
    chan_group = channel_file_group["Channel"]
    name_ds = chan_group["Name"]
    names = []
    for item in name_ds[:, 0]:
        names.append(h5_string_from_any(h5, item).strip())
    return names


def extract_channel_names(h5: h5py.File, stim_group, entry_index: int) -> list[str]:
    chan_obj = stim_group["channel_file"]
    if isinstance(chan_obj, h5py.Group):
        return extract_channel_names_from_group(h5, chan_obj)
    chan_ref = chan_obj[entry_index, 0]
    return extract_channel_names_from_group(h5, h5[chan_ref])


def _read_scalar_ref(h5: h5py.File, ref) -> float:
    return float(np.asarray(h5[ref][()]).squeeze())


def load_recording(path: str | Path, entry_index: int = 0) -> SeegRecording:
    path = Path(path)
    with h5py.File(path, "r") as h5:
        stim = h5["stimulations"]
        trial_obj = stim["trial_file"]
        direct_single_entry = isinstance(trial_obj, h5py.Group)
        trial = trial_obj if direct_single_entry else h5[trial_obj[entry_index, 0]]
        signal = np.asarray(trial["F"][()], dtype=float)
        time = np.asarray(trial["Time"][()], dtype=float).squeeze()
        fs = float(1.0 / np.median(np.diff(time)))

        channel_names = extract_channel_names(h5, stim, entry_index)
        if direct_single_entry:
            stimulated = strings_from_dataset(h5, stim["stimulated_channel_names"])
            stim_frequency = float(np.asarray(stim["frequence"][()]).squeeze())
            stim_amplitude = float(np.asarray(stim["amperage"][()]).squeeze())
        else:
            stimulated = strings_from_ref_dataset(h5, stim["stimulated_channel_names"], entry_index)
            stim_frequency = _read_scalar_ref(h5, stim["frequence"][entry_index, 0])
            stim_amplitude = _read_scalar_ref(h5, stim["amperage"][entry_index, 0])

    if signal.shape[0] != time.size and signal.shape[1] == time.size:
        signal = signal.T
    if signal.shape[0] != time.size:
        raise ValueError(f"Signal/time shape mismatch in {path}: {signal.shape}, {time.shape}")

    if len(channel_names) != signal.shape[1]:
        channel_names = [f"channel_{i + 1:03d}" for i in range(signal.shape[1])]

    return SeegRecording(
        path=path,
        entry_index=entry_index,
        file_stem=path.stem,
        signal=signal,
        time=time,
        fs=fs,
        channel_names=channel_names,
        stimulated_channel_names=stimulated,
        stim_frequency_hz=stim_frequency,
        stim_amplitude_ma=stim_amplitude,
        subject=parse_subject_from_name(path),
    )
