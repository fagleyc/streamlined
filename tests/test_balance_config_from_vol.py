"""The .vol's declared balance type decides Force vs Moment reduction.

Regression (2026-10-06): Freestream recorded balance_config 'Moment' for
the 100 lb FORCE balance ('5 Force/1 Moment'); reducing its N1/N2/Y1/Y2
force elements with the moment equations read CL ~2.75x low (0.48 at
alpha 20 for the F16 check model instead of ~1.3).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.windtunnel.calibration import (  # noqa: E402
    balance_config_from_type, read_vol_file, resolve_balance_config)


@pytest.mark.parametrize("text,expected", [
    ("5 Force/1 Moment", "Force"),
    ("5 Moment/1 Force", "Moment"),
    ("1-Force / 5-Moment", "Moment"),
    ("1 Force/5 Moment", "Moment"),
    ("6 Force", "Force"),
    ("", None),
    ("internal", None),
])
def test_declared_type(text, expected):
    assert balance_config_from_type(text) == expected


def _cal(balance_type):
    return SimpleNamespace(description=SimpleNamespace(
        balance_type=balance_type, serial_number="SN1"))


def test_declaration_overrides_a_conflicting_request():
    cfg, note = resolve_balance_config(_cal("5 Force/1 Moment"), "Moment")
    assert cfg == "Force" and "overridden" in note


def test_matching_request_is_silent():
    assert resolve_balance_config(_cal("5 Moment/1 Force"), "Moment") == (
        "Moment", None)


def test_undeclared_calibration_keeps_the_request():
    assert resolve_balance_config(_cal(""), "Moment") == ("Moment", None)
    assert resolve_balance_config(None, "Moment") == ("Moment", None)


@pytest.mark.parametrize("vol,expected", [
    ("2025_06_06_2 100 lb.vol", "Force"),
    ("50lb Mom Bal_Calibration_V2L.vol", "Moment"),
    ("200009_26MAR2012.vol", "Moment"),
    ("SN29850_E99_July2022.vol", "Force"),
])
def test_shipped_calibrations_declare_their_type(vol, expected):
    path = ROOT / "CalFiles" / vol
    if not path.exists():
        pytest.skip(f"{vol} not present")
    assert balance_config_from_type(
        read_vol_file(str(path)).description.balance_type) == expected
