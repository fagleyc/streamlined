"""COE export: one .COE file is one alpha sweep at ONE Mach step and ONE beta.

The legacy Excel tool and coe_postprocess.py take alpha derivatives (Cma,
CLa, static margin) along each file, so a file that interleaves two tunnel
speeds gives nonsense. The May 26 F16 set steps M0.2 and M0.3 at betas
-5, 0 and 5: it must export as six files, keyed on the COMMANDED Mach and
beta the GUI filters on.

Uses the F16 check-model data (data/F16_CheckModel); skips without it.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import test_f16_check_model as f16  # noqa: E402
from utils.windtunnel.coe_reader import read_coe_file  # noqa: E402
from utils.windtunnel.coe_writer import (  # noqa: E402
    _geometry_ips, _num_tag, write_coe_files)

pytestmark = f16.pytestmark


@pytest.fixture(scope="module")
def may26():
    cases, errors = f16._reduce(f16.SETS["May26"])
    assert not f16._problems(errors)
    return cases[0]


def test_one_file_per_mach_step_and_beta(may26, tmp_path):
    paths = write_coe_files(may26, str(tmp_path), dict(f16.GEOMETRY))
    names = sorted(Path(p).name for p in paths)
    assert names == sorted(
        f"{may26.name}_M{m}_b{b}.COE"
        for m in ("0p20", "0p30") for b in ("-5", "0", "5"))

    total = 0
    for p in paths:
        coe = read_coe_file(p)
        mach = 0.2 if "_M0p20_" in p else 0.3
        # one speed, one beta, one monotonic sweep per file
        assert np.all(np.abs(coe["Ma"] - mach) < 0.02), p
        assert np.ptp(coe["Beta"]) < 0.5, p
        assert np.all(np.diff(coe["Alpha"]) > 0), p
        total += coe.n_rows
    # the one flagged point (M3 alpha 26 / beta 5: Mach 0.04, positioner
    # at 0/0) is left out of the rows and named in that file's [Comment]
    assert total == may26.n_points - 1
    flagged = next(p for p in paths if p.endswith("_M0p30_b5.COE"))
    text = Path(flagged).read_text(encoding="utf-8")
    assert "Point left out (alpha 26:" in text


def test_rows_match_the_case(may26, tmp_path):
    """Every exported CL is the case's own CL at that point."""
    paths = write_coe_files(may26, str(tmp_path), dict(f16.GEOMETRY))
    exported = np.sort(np.concatenate(
        [read_coe_file(p)["CLift"] for p in paths]))
    cl = np.asarray(may26.Cl, float).ravel()
    valid = np.asarray(may26.machs, float).ravel() > 0.1   # flagged point
    case_cl = np.sort(np.round(cl[valid], 4))
    assert np.allclose(exported, case_cl, atol=1e-4)


def test_single_speed_case_keeps_its_old_name(tmp_path):
    cases, _ = f16._reduce(f16.SETS["Oct26"], vol=False)
    paths = write_coe_files(cases[0], str(tmp_path), dict(f16.GEOMETRY))
    assert [Path(p).name for p in paths] == [f"{cases[0].name}_b0.COE"]


def test_metric_geometry_is_written_in_inches():
    geo = _geometry_ips(dict(mac=0.0254 * 2.86, ref_area=0.0254 ** 2 * 18.75,
                             span=0.0254, mrc=[0.0254 * 1.6, 0, 0],
                             units="MKS"))
    assert geo["mac"] == pytest.approx(2.86)
    assert geo["ref_area"] == pytest.approx(18.75)
    assert geo["mrc"][0] == pytest.approx(1.6)


def test_fractional_beta_gets_its_own_tag():
    assert _num_tag(2.5) == "2p5"
    assert _num_tag(2.0) == "2"
    assert _num_tag(-5) == "-5"
