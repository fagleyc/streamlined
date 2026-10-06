"""External balance needs no .vol, in the backend probe and in the GUI.

An external balance (the ATE) streams resolved loads in engineering
units, so there is nothing for a bridge-volts calibration to do.
Demanding one blocked a perfectly reducible run directory. Two things
have to hold: the directory probe has to recognise the run as external,
and the calibration panel has to grey its .vol input out rather than
offering a control that cannot help.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.windtunnel.data_io import (MANIFEST_FILENAME,   # noqa: E402
                                      MANIFEST_SCHEMA_VERSION,
                                      run_balance_type)

scipy_io = pytest.importorskip("scipy.io")

CHANNELS = ("Lift", "Drag", "Side", "Roll", "Pitch", "Yaw")


def _write_run(directory: Path, name: str, balance_type: str,
               n: int = 32) -> Path:
    """Minimal freestream-shaped .mat run file with a balance marker."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    group = {c: rng.normal(size=n) for c in CHANNELS}
    group["Pdiff"] = np.full(n, 0.8)
    path = directory / name
    scipy_io.savemat(str(path), {
        "ATE_Balance": group,
        "Time": {"Time": np.arange(n) / 50.0},
        "meta": {"run": {"balance_type": balance_type,
                         "air_state": "AirOn",
                         "alpha": 0.0, "beta": 0.0},
                 "config_json": json.dumps({})},
    }, long_field_names=True)
    return path


def _manifest(directory: Path, **balance_cal) -> None:
    payload = {"schema_version": MANIFEST_SCHEMA_VERSION,
               "config_name": directory.name, "output_format": "mat",
               "points": []}
    if balance_cal:
        payload["balance_cal"] = balance_cal
    (directory / MANIFEST_FILENAME).write_text(json.dumps(payload),
                                               encoding="utf-8")


# ── directory probe ─────────────────────────────────────────────────────
def test_probe_reads_external_from_run_metadata(tmp_path):
    d = tmp_path / "ext"
    _write_run(d, "run_0001_alpha_0.0.mat", "external")
    _manifest(d)
    assert run_balance_type(str(d)) == "external"


def test_probe_reads_internal_from_run_metadata(tmp_path):
    d = tmp_path / "int"
    _write_run(d, "run_0001_alpha_0.0.mat", "internal")
    _manifest(d)
    assert run_balance_type(str(d)) == "internal"


def test_probe_prefers_the_manifest_when_it_says(tmp_path):
    d = tmp_path / "manifest_wins"
    _write_run(d, "run_0001_alpha_0.0.mat", "internal")
    _manifest(d, balance_type="external", vol_file="")
    assert run_balance_type(str(d)) == "external"


def test_probe_is_quiet_when_nothing_says(tmp_path):
    d = tmp_path / "silent"
    _write_run(d, "run_0001_alpha_0.0.mat", "")
    _manifest(d)
    assert run_balance_type(str(d)) == ""


def test_probe_survives_an_empty_directory(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    assert run_balance_type(str(d)) == ""


# ── GUI: the .vol input greys out ───────────────────────────────────────
@pytest.fixture(scope="module")
def app():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([sys.argv[0]])


def test_calibration_panel_greys_out_for_external(app):
    from utils.gui.models.settings import AppSettings
    from utils.gui.views.data_panel import CalibrationSection

    section = CalibrationSection(AppSettings())
    try:
        assert section.btn_balance.isEnabled()

        section.set_external_mode(True)
        assert not section.btn_balance.isEnabled()
        assert not section.txt_balance.isEnabled()
        assert "external" in section.txt_balance.placeholderText().lower()
        assert "external" in section.lbl_status.text().lower()

        section.set_external_mode(False)          # internal restores it
        assert section.btn_balance.isEnabled()
        assert section.txt_balance.isEnabled()
        assert ".vol" in section.btn_balance.toolTip().lower()
    finally:
        section.deleteLater()


def test_data_panel_routes_the_detected_type(app):
    from utils.gui.models.data_model import DataModel
    from utils.gui.models.settings import AppSettings
    from utils.gui.views.data_panel import DataPanel

    panel = DataPanel(DataModel(), AppSettings())
    try:
        panel.set_balance_type("external")
        assert not panel.cal_section.btn_balance.isEnabled()
        panel.set_balance_type("internal")
        assert panel.cal_section.btn_balance.isEnabled()
        panel.set_balance_type("")                # unknown: leave usable
        assert panel.cal_section.btn_balance.isEnabled()
    finally:
        panel.deleteLater()


# ── controller: the processing gate ─────────────────────────────────────
def test_controller_gate_skips_the_vol_for_external(tmp_path, app):
    """The 'No Balance Calibration' refusal must not fire for a run
    directory whose balance is external."""
    from utils.gui.controllers.data_controller import DataController
    from utils.gui.models.data_model import DataModel
    from utils.gui.models.settings import AppSettings

    d = tmp_path / "extrun"
    _write_run(d, "run_0001_alpha_0.0.mat", "external")
    _manifest(d)

    ctrl = DataController(DataModel(), AppSettings())
    errors = []
    ctrl.error_occurred.connect(lambda t, m: errors.append((t, m)))
    ctrl._seed_config_from_run_files([str(d)])
    assert ctrl._run_balance_type == "external"
    assert not errors

    ctrl_int = DataController(DataModel(), AppSettings())
    d2 = tmp_path / "intrun"
    _write_run(d2, "run_0001_alpha_0.0.mat", "internal")
    _manifest(d2)
    ctrl_int._seed_config_from_run_files([str(d2)])
    assert ctrl_int._run_balance_type == "internal"
