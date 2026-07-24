"""Injected balance calibration: Streamlined reduces forces from the cal
MATRIX freestream stores in the run file (/meta/devices/<balance>), with no
.vol — and a loaded .vol still overrides."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

h5py = pytest.importorskip("h5py")

from utils.windtunnel.calibration import balance_cal_from_matrix
from utils.windtunnel.transforms import get_distance_values, calc_brf_forces, Geometry
from utils.windtunnel.data_io import read_hdf5_file


def test_balance_cal_from_matrix_shape_and_distances():
    m = np.arange(36, dtype=float).reshape(6, 6)
    cal = balance_cal_from_matrix(m, "Linear", [1.1, 1.2, 1.3, 1.4], "50 lb")
    assert cal.coeffs.shape == (6, 6)
    assert cal.cal_type == "Linear"
    assert cal.description.serial_number == "50 lb"
    d = get_distance_values(cal)
    assert (d["dx1"], d["dx2"], d["dy1"], d["dy2"]) == (1.1, 1.2, 1.3, 1.4)


def _write_h5_with_injected_cal(path, matrix, n=64):
    with h5py.File(path, "w") as f:
        f.attrs["alpha"] = 0.0
        f.attrs["beta"] = 0.0
        f.attrs["balance_type"] = "internal"
        f.attrs["balance_group"] = "StrainBook_0"
        grp = f.create_group("StrainBook_0")
        rng = np.random.default_rng(1)
        for ch in ("N1", "N2", "Y1", "Y2", "Axial", "Roll", "Excitation"):
            d = grp.create_dataset(
                ch, data=(np.full(n, 10.0) if ch == "Excitation"
                          else rng.standard_normal(n) * 0.001 + 0.01))
            d.attrs["wf_increment"] = 1.0 / 1000
            d.attrs["wf_samples"] = n
        meta = f.create_group("meta")
        dev = meta.create_group("devices").create_group("strainbook")
        dev.attrs["cal_matrix"] = matrix
        dev.attrs["cal_type"] = "Linear"
        dev.attrs["cal_distances"] = np.array([1.28, 1.27, 1.28, 1.27])
        dev.attrs["balance_serial"] = "50 lb"
        dev.attrs["balance_type"] = "internal"
        meta.create_dataset("config", data="{}")


def test_injected_cal_captured_on_read(tmp_path):
    matrix = (np.eye(6) * 1e5 + 1.0)
    p = tmp_path / "run_0001_alpha_0.0_beta_0.0_mach_0.30.h5"
    _write_h5_with_injected_cal(str(p), matrix)
    raw, _ = read_hdf5_file(str(p))
    inj = raw.properties.get("injected_balance_cal")
    assert inj is not None
    np.testing.assert_allclose(inj["matrix"], matrix)
    assert inj["cal_type"] == "Linear"
    np.testing.assert_allclose(inj["distances"], [1.28, 1.27, 1.28, 1.27])
    assert inj["serial"] == "50 lb"


def test_forces_reduce_from_injected_matrix(tmp_path):
    matrix = (np.eye(6) * 1e5 + np.arange(36).reshape(6, 6))
    p = tmp_path / "run_0001_alpha_0.0_beta_0.0_mach_0.30.h5"
    _write_h5_with_injected_cal(str(p), matrix)
    raw, _ = read_hdf5_file(str(p))
    inj = raw.properties["injected_balance_cal"]
    cal = balance_cal_from_matrix(inj["matrix"], inj["cal_type"],
                                  inj["distances"], inj["serial"])
    # reduces without error and matches a directly-built reference cal
    ref = balance_cal_from_matrix(matrix, "Linear", [1.28, 1.27, 1.28, 1.27])
    geo = Geometry(C=2.0, S=10.0)
    a = calc_brf_forces(dict(raw.data), cal, geo, "Force")
    b = calc_brf_forces(dict(raw.data), ref, geo, "Force")
    np.testing.assert_allclose(a.Fx, b.Fx)
    np.testing.assert_allclose(a.Mz, b.Mz)
    assert np.isfinite(np.mean(a.Fx))
