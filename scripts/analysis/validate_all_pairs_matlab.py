from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy import signal
from scipy.io import loadmat, savemat
from statsmodels.tsa.api import VAR

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MNE_LOGGING_LEVEL", "WARNING")

from seeg_waveform.all_pairs_waveform_connectivity import (
    _gpdc_from_fit,
    _prepare_run,
    temporal_irreversibility_unsigned,
)
from seeg_waveform.config import load_config


def fit_validation_model(x: np.ndarray, y: np.ndarray, fs: float) -> dict:
    target_fs = 128.0
    down, up = int(target_fs), int(round(fs))
    divisor = np.gcd(down, up)
    data = np.column_stack([
        signal.resample_poly(x, down // divisor, up // divisor),
        signal.resample_poly(y, down // divisor, up // divisor),
    ])
    data = signal.detrend(data, axis=0)
    data = (data - data.mean(axis=0)) / data.std(axis=0, ddof=1)
    selected = VAR(data).select_order(maxlags=30)
    order = selected.selected_orders.get("aic") or 1
    fit = VAR(data).fit(int(order))
    a_to_b, b_to_a = _gpdc_from_fit(fit, target_fs, (8.0, 11.0), 1e-12)
    return {
        "u": data.T,
        "A": np.transpose(fit.coefs, (1, 2, 0)),
        "pf": fit.sigma_u,
        "python_a_to_b": a_to_b,
        "python_b_to_a": b_to_a,
    }


def real_imf4_pair(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    cfg = load_config(ROOT / "configs" / "primary_bipolar_mne_config.yaml")
    run, pre, _ = _prepare_run(path, "bipolar", cfg)
    eligible = [
        index
        for index, node in run.nodes.iterrows()
        if bool(node["primary_region_include"])
        and not pre[index].noise_flag
        and pre[index].result.status == "ok"
    ]
    if len(eligible) < 2:
        raise RuntimeError("The validation recording has fewer than two eligible IMF4 channels.")
    return (
        np.asarray(pre[eligible[0]].result.imf[:, 3], float),
        np.asarray(pre[eligible[1]].result.imf[:, 3], float),
        float(run.recording.fs),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-file",
        type=Path,
        default=ROOT / "data" / "F50" / "stims_HEJ_Subject14_F=50_I=2.mat",
    )
    args = parser.parse_args()
    output = ROOT / "outputs" / "publication_all_pairs_imf4" / "validation"
    output.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260612)
    n = 7000
    synthetic_a = np.zeros(n)
    synthetic_b = np.zeros(n)
    for index in range(2, n):
        synthetic_a[index] = (
            0.72 * synthetic_a[index - 1]
            - 0.18 * synthetic_a[index - 2]
            + rng.normal()
        )
        synthetic_b[index] = (
            0.48 * synthetic_b[index - 1]
            + 0.42 * synthetic_a[index - 1]
            + rng.normal()
        )
    real_a, real_b, real_fs = real_imf4_pair(args.real_file)
    datasets = {
        "synthetic": (synthetic_a, synthetic_b, 512.0),
        "real_imf4": (real_a, real_b, real_fs),
    }
    payload = {}
    expected = {}
    for name, (x, y, fs) in datasets.items():
        model = fit_validation_model(x, y, fs)
        envelope_x = np.abs(signal.hilbert(x))
        envelope_y = np.abs(signal.hilbert(y))
        tra, status = temporal_irreversibility_unsigned(
            envelope_x, envelope_y, int(round(fs))
        )
        payload[f"{name}_u"] = model["u"]
        payload[f"{name}_A"] = model["A"]
        payload[f"{name}_pf"] = model["pf"]
        payload[f"{name}_env_a"] = envelope_x
        payload[f"{name}_env_b"] = envelope_y
        payload[f"{name}_mlag"] = int(round(fs))
        expected[name] = {
            "python_gpdc_a_to_b": model["python_a_to_b"],
            "python_gpdc_b_to_a": model["python_b_to_a"],
            "python_tra": tra,
            "python_tra_status": status,
        }
    input_mat = output / "matlab_validation_input.mat"
    result_mat = output / "matlab_validation_output.mat"
    savemat(input_mat, payload)

    asymp = (ROOT / "documentation" / "asympPDC-main").as_posix()
    grassmann = (ROOT / "documentation" / "Grassmann_manifold-main").as_posix()
    matlab_script = output / "run_matlab_validation.m"
    matlab_script.write_text(
        "\n".join([
            f"addpath(genpath('{asymp}'));",
            f"addpath('{grassmann}');",
            f"load('{input_mat.as_posix()}');",
            "names = {'synthetic','real_imf4'};",
            "for k = 1:numel(names)",
            "  name = names{k};",
            "  u = eval([name '_u']); A = eval([name '_A']); pf = eval([name '_pf']);",
            "  c = asymp_pdc(u,A,pf,4096,'diag',0);",
            "  freq = (0:4095) / (2*4096) * 128;",
            "  idx = freq >= 8 & freq <= 11;",
            "  eval([name '_gpdc_a_to_b = mean(squeeze(c.pdc2(2,1,idx)));']);",
            "  eval([name '_gpdc_b_to_a = mean(squeeze(c.pdc2(1,2,idx)));']);",
            "  env_a = eval([name '_env_a']); env_b = eval([name '_env_b']);",
            "  mlag = eval([name '_mlag']);",
            "  eval([name '_tra = invariant_features_bivariate_v2(env_a,env_b,mlag,0);']);",
            "end",
            f"save('{result_mat.as_posix()}','synthetic_gpdc_a_to_b','synthetic_gpdc_b_to_a',"
            "'synthetic_tra','real_imf4_gpdc_a_to_b','real_imf4_gpdc_b_to_a','real_imf4_tra');",
        ]) + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["matlab", "-batch", f"run('{matlab_script.as_posix()}')"],
        cwd=ROOT,
        check=True,
        timeout=900,
    )
    matlab = loadmat(result_mat, squeeze_me=True)
    report = {}
    for name in datasets:
        report[name] = expected[name] | {
            "matlab_gpdc_a_to_b": float(matlab[f"{name}_gpdc_a_to_b"]),
            "matlab_gpdc_b_to_a": float(matlab[f"{name}_gpdc_b_to_a"]),
            "matlab_tra": float(matlab[f"{name}_tra"]),
        }
        report[name]["gpdc_a_to_b_abs_error"] = abs(
            report[name]["python_gpdc_a_to_b"] - report[name]["matlab_gpdc_a_to_b"]
        )
        report[name]["gpdc_b_to_a_abs_error"] = abs(
            report[name]["python_gpdc_b_to_a"] - report[name]["matlab_gpdc_b_to_a"]
        )
        report[name]["tra_abs_error"] = abs(
            report[name]["python_tra"] - report[name]["matlab_tra"]
        )
    report_path = output / "python_matlab_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Validation report: {report_path}")


if __name__ == "__main__":
    main()
