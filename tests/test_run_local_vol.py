"""Run-local balance calibration hand-off (freestream -> Streamlined).

Freestream copies the active .vol into the run directory at run start and
records it in manifest.json as a top-level 'balance_cal' object.  The
reduction resolves that run-local .vol automatically; an explicitly loaded
.vol still overrides (the balance_cal / balance_cal_file branches in
ProcessingWorker._process_with_backend are checked first).
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.windtunnel.data_io import (  # noqa: E402
    MANIFEST_FILENAME, MANIFEST_SCHEMA_VERSION, find_run_balance_cal,
)

VOL_NAME = "50lb 2026_07_24.vol"

# Load, then 6 bridge volts, then excitation volts (mirrors
# test_vol_parsing's minimal one-channel fixture).
_LOADS = [
    [0.0, 0.0, 8.6e-05, 8.0e-05, 3.7e-05, -7.8e-05, -2.4e-04, 5.0015],
    [6.435, 2.4e-04, 8.8e-05, 7.8e-05, 3.7e-05, -8.0e-05, -2.4e-04, 5.0015],
    [12.87, 4.8e-04, 9.2e-05, 7.7e-05, 3.6e-05, -8.1e-05, -2.5e-04, 5.0015],
]


def write_vol(directory: Path, name: str = VOL_NAME) -> Path:
    """A minimal parseable one-channel .vol in `directory`."""
    lines = [
        "Voltage Calibration File",
        "Date-->28-Feb-2023",
        "",
        "[Balance Description]",
        "Balance Type-->1-Force / 5-Moment",
        "Balance Serial Number-->94017",
        "",
        "[Maximal Balance Loads]",
        "Aft_Pitch-->75 in-lb",
        "",
        "[Distances]",
        "Distance from Aft_Pitch to center of Balance x1 in inches--> 1.287",
        "",
        "[Aft_Pitch pos]",
        f"Number of Loads --> {len(_LOADS)}",
        ",".join(["Load[in-lb]"] + [f"ch{i}[Volt]" for i in range(7)]),
    ]
    for row in _LOADS:
        lines.append(",".join(f"{v:.8f}" for v in row))
    path = directory / name
    path.write_bytes(("\n".join(lines) + "\n").encode("latin-1"))
    return path


def write_manifest(directory: Path, balance_cal=None, config=None,
                   points=()):
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "config_name": "F16_Val",
        "output_format": "mat",
        "created": "2026-08-05T22:10:36",
        "updated": "2026-08-05T22:15:57",
        "config": config or {},
        "points": list(points),
    }
    if balance_cal is not None:
        manifest["balance_cal"] = balance_cal
    (directory / MANIFEST_FILENAME).write_text(json.dumps(manifest),
                                               encoding="utf-8")


def f16_point(i):
    """One F16_Val-style manifest point entry."""
    name = f"run_{i:04d}_alpha_{2.0 * i - 4:.1f}_beta_0.0_Hz_60.0.mat"
    return {"run_number": i, "filename": name,
            "timestamp": "2026-08-05T22:14:02", "alpha": 2.0 * i - 4,
            "beta": 0.0, "mach": 0.095, "speed_value": 60.0,
            "speed_unit": "hz", "air_state": "AirOn", "sweep_dir": ""}


class TestFindRunBalanceCal:

    def test_manifest_entry_resolves_run_local_vol(self, tmp_path):
        # the F16_Val layout: manifest + run_*.mat files + the staged .vol
        vol = write_vol(tmp_path)
        points = [f16_point(i) for i in range(1, 4)]
        for p in points:
            (tmp_path / p["filename"]).write_bytes(b"")
        write_manifest(tmp_path, balance_cal={
            "vol_file": VOL_NAME, "vol_source": "C:/gone/CalFiles/x.vol",
            "cal_type": "Linear", "balance_config": "Moment",
            "balance_type": "internal", "balance_serial": "50 lb"},
            config={"cal_type": "Linear"}, points=points)
        resolved = find_run_balance_cal(str(tmp_path))
        assert resolved["vol_path"] == str(vol)
        assert resolved["cal_type"] == "Linear"
        assert resolved["balance_config"] == "Moment"
        assert resolved["balance_serial"] == "50 lb"
        assert "vol_file" not in resolved       # replaced by vol_path

    def test_manifest_entry_with_missing_file_falls_back(self, tmp_path):
        vol = write_vol(tmp_path)
        write_manifest(tmp_path,
                       balance_cal={"vol_file": "not_there.vol",
                                    "cal_type": "Cubic"},
                       config={"cal_type": "Quadratic"})
        resolved = find_run_balance_cal(str(tmp_path))
        # falls back to the single *.vol; cal_type from the config block
        assert resolved == {"vol_path": str(vol), "cal_type": "Quadratic"}

    def test_single_vol_without_manifest(self, tmp_path):
        vol = write_vol(tmp_path)
        assert find_run_balance_cal(str(tmp_path)) == {
            "vol_path": str(vol)}

    def test_two_vols_without_manifest_entry_is_ambiguous(self, tmp_path):
        write_vol(tmp_path, "a.vol")
        write_vol(tmp_path, "b.vol")
        assert find_run_balance_cal(str(tmp_path)) == {}

    def test_manifest_entry_disambiguates_multiple_vols(self, tmp_path):
        write_vol(tmp_path, "old.vol")
        vol = write_vol(tmp_path)
        write_manifest(tmp_path, balance_cal={"vol_file": VOL_NAME,
                                              "cal_type": "Linear"})
        resolved = find_run_balance_cal(str(tmp_path))
        assert resolved["vol_path"] == str(vol)

    def test_no_vol_resolves_to_nothing(self, tmp_path):
        write_manifest(tmp_path)
        assert find_run_balance_cal(str(tmp_path)) == {}


class TestWorkerPicksUpRunLocalVol:
    """ProcessingWorker builds a usable calibration from the run-local
    .vol (offscreen; needs PyQt6 + the windtunnel backend)."""

    @pytest.fixture
    def worker(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        pytest.importorskip("PyQt6")
        from utils.gui.controllers.data_controller import (
            BACKEND_AVAILABLE, ProcessingWorker)
        if not BACKEND_AVAILABLE:
            pytest.skip("windtunnel backend not importable")
        return ProcessingWorker(
            directories=[], balance_cal=None, pressure_cal=None,
            balance_cal_file=None, pressure_cal_file=None,
            geometry={}, settings={"cal_type": "Linear"})

    def _info(self, directory):
        from utils.gui.controllers.data_controller import FileInfoSimple
        return FileInfoSimple(
            filepath=directory / "run_0001_alpha_0.0_beta_0.0_Hz_60.0.mat",
            configuration="F16_Val", air_state="AirOn", alpha=0.0, beta=0.0)

    def test_run_local_vol_becomes_a_calibration(self, worker, tmp_path):
        write_vol(tmp_path)
        write_manifest(tmp_path, balance_cal={"vol_file": VOL_NAME,
                                              "cal_type": "Linear"})
        cal = worker._run_local_balance_cal([self._info(tmp_path)])
        assert cal is not None
        assert cal.cal_type == "Linear"
        assert cal.coeffs.size > 0              # fitted, reduction-ready

    def test_no_run_local_vol_returns_none(self, worker, tmp_path):
        assert worker._run_local_balance_cal([self._info(tmp_path)]) is None
        assert worker._run_local_balance_cal([]) is None
