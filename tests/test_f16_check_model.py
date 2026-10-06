"""F16 check-model regression suite: the reference datasets every change
must keep reducing the same way.

``data/F16_CheckModel`` holds the same F16 model on the same 100 lb
internal balance, recorded three ways:

* ``Run01``      - LabVIEW TDMS, legacy names, no speed token
                   (``AirOn_F16check_no_beta_Alpha_-2.0_Beta_0.0.tdms``)
* ``May 26``     - LabVIEW TDMS, legacy names with a bare tenths Mach token
                   and a run counter (``AirOn_F16_100lb_M3_Run_1_Alpha_..``);
                   the M2 session was recorded WITHOUT the DaqBook, so its
                   tunnel conditions are rebuilt from the commanded Mach
* ``October 26`` - Freestream .mat runs with a manifest and a run-local .vol

The suite checks three things:

1. each set loads and groups as one case with the right Mach steps;
2. the reductions agree WITH EACH OTHER (same model -> same polar), which
   is what makes this a check model rather than a snapshot;
3. every reduced coefficient matches the golden record in
   ``tests/golden/f16_check_model.json`` - so any software change that
   moves a number is caught. When a change is MEANT to move them, review
   the diff and regenerate with::

       python tests/test_f16_check_model.py --update-golden

The data is not in the repository (``data/`` is ignored); the suite skips
where it is absent.
"""
import contextlib
import io
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "F16_CheckModel"
VOL = ROOT / "CalFiles" / "2025_06_06_2 100 lb.vol"
GOLDEN = Path(__file__).resolve().parent / "golden" / "f16_check_model.json"
SETS = {"Run01": "Run01", "May26": "May 26",
        "Oct26": os.path.join("October 26", "F16_Comparison")}

pytestmark = pytest.mark.skipif(
    not all((DATA / sub).is_dir() for sub in SETS.values()),
    reason="F16 check-model data not present (data/F16_CheckModel)")

# The reduction setup recorded with the October set (manifest.json):
# moment-configuration balance, cubic calibration, F16 reference geometry.
GEOMETRY = dict(mac=2.86, ref_area=18.75, span=1.0, mrc=[1.6, 0.0, 0.0],
                units="IPS", output_units="IPS")
SETTINGS = dict(cal_type="Cubic", facility="SWT", output_units="IPS",
                balance_config="Moment")
COEFFS = ("Cl", "Cd", "Cs", "CRoll", "CPitch", "CYaw")


def _reduce(sub, vol=True):
    """Reduce one set through the GUI's own ProcessingWorker."""
    from PyQt6.QtWidgets import QApplication
    QApplication.instance() or QApplication([sys.argv[0]])
    from utils.gui.controllers.data_controller import ProcessingWorker

    worker = ProcessingWorker(
        [str(DATA / sub)], None, None, str(VOL) if vol else None, None,
        geometry=dict(GEOMETRY), settings=dict(SETTINGS))
    cases, errors = [], []
    worker.signals.case_ready.connect(cases.append)
    worker.signals.error.connect(lambda t, m: errors.append((t, m)))
    with warnings.catch_warnings(), \
            contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        worker.run()
    return cases, errors


def _snapshot(case):
    """Per point: nominal alpha/beta, commanded Mach step, measured Mach,
    coefficients - sorted speed -> beta -> alpha (the run order)."""
    an = np.asarray(case.alpha_nominal, float).ravel()
    bn = np.asarray(case.beta_nominal, float).ravel()
    pm = np.asarray(case.point_machs, float).ravel()
    order = np.lexsort((an, bn, np.round(pm, 3)))
    out = {"alpha": an[order].round(2).tolist(),
           "beta": bn[order].round(2).tolist(),
           "mach_step": np.round(pm[order], 3).tolist(),
           "mach": np.asarray(case.machs, float).ravel()[order]
           .round(4).tolist()
           if np.asarray(case.machs).size == an.size else []}
    for name in COEFFS:
        out[name] = np.asarray(getattr(case, name), float).ravel()[order] \
            .round(5).tolist()
    return out


@pytest.fixture(scope="module")
def reduced():
    out = {}
    for key, sub in SETS.items():
        # the October set carries its own .vol: load it with NONE given
        cases, errors = _reduce(sub, vol=(key != "Oct26"))
        out[key] = (cases, errors)
    return out


def _polar(case, step, beta=0.0):
    """(alpha, CL) of the beta polar at one commanded Mach step, valid
    points only (the flagged tunnel-off point is excluded by its note)."""
    an = np.asarray(case.alpha_nominal, float).ravel()
    bn = np.asarray(case.beta_nominal, float).ravel()
    pm = np.asarray(case.point_machs, float).ravel()
    m = np.asarray(case.machs, float).ravel()
    cl = np.asarray(case.Cl, float).ravel()
    sel = (np.isclose(pm, step, atol=0.006) & np.isclose(bn, beta)
           & (m > 0.5 * step))
    i = np.argsort(an[sel])
    return an[sel][i], cl[sel][i]


# ── 1. loading and grouping ─────────────────────────────────────────────
class TestLoading:
    def test_run01_is_one_case_at_one_mach(self, reduced):
        cases, errors = reduced["Run01"]
        assert not errors and len(cases) == 1
        case = cases[0]
        assert np.asarray(case.alphas).size == 14
        # no speed token: measured Machs cluster to ONE filter entry
        steps = np.unique(np.asarray(case.point_machs, float))
        assert steps.size == 1 and steps[0] == pytest.approx(0.29, abs=0.01)

    def test_may26_m2_and_m3_are_one_case_two_mach_steps(self, reduced):
        cases, _ = reduced["May26"]
        assert len(cases) == 1, [c.name for c in cases]
        case = cases[0]
        assert case.name.endswith("F16_100lb_Run_1")
        assert np.asarray(case.alphas).size == 102
        assert sorted(np.unique(np.asarray(case.point_machs, float))) == [
            pytest.approx(0.2), pytest.approx(0.3)]

    def test_may26_m2_tunnel_conditions_are_rebuilt_and_reported(
            self, reduced):
        cases, errors = reduced["May26"]
        notes = cases[0].metadata.get("reduction_notes", [])
        assert any("COMMANDED Mach 0.2" in n for n in notes)
        # ... and surfaced to the operator, not silently applied
        assert any(t == "Reduction Notes" for t, _m in errors)
        m = np.asarray(cases[0].machs, float).ravel()
        pm = np.asarray(cases[0].point_machs, float).ravel()
        assert np.allclose(m[np.isclose(pm, 0.2)], 0.2, atol=1e-6)

    def test_may26_tunnel_off_point_is_flagged_not_dropped(self, reduced):
        case = reduced["May26"][0][0]
        notes = case.metadata.get("reduction_notes", [])
        assert any("alpha 26.0 / beta 5.0" in n and "tunnel off" in n
                   for n in notes)
        assert np.asarray(case.alphas).size == 102           # still there

    def test_october_loads_with_its_run_local_vol(self, reduced):
        cases, errors = reduced["Oct26"]
        assert len(cases) == 1 and not errors
        assert np.unique(np.asarray(cases[0].point_machs, float)).tolist() \
            == [pytest.approx(0.3)]

    def test_october_reduces_to_mach_0_3_not_the_recorded_0_72(
            self, reduced):
        """The Oct 2026 files carry Freestream's bogus Tunnel/Mach_meas
        (0.72, computed from raw volts); the reduction must not use it."""
        m = np.asarray(reduced["Oct26"][0][0].machs, float)
        assert np.all((m > 0.27) & (m < 0.31))

    def test_gui_preflight_accepts_the_run_local_vol(self):
        from utils.gui.controllers.data_controller import DataController
        assert DataController._directories_carry_balance_cal(
            [str(DATA / SETS["Oct26"])])
        assert not DataController._directories_carry_balance_cal(
            [str(DATA / "Run01")])


# ── 2. the sets agree with each other ───────────────────────────────────
class TestCrossDatasetConsistency:
    """Same model, same balance: every set's beta=0 polar must agree."""

    ALPHAS = (-4.0, 0.0, 4.0, 8.0, 12.0)

    def _curves(self, reduced):
        may = reduced["May26"][0][0]
        return {
            "Run01 M0.29": _polar(reduced["Run01"][0][0], 0.291),
            "May M0.2": _polar(may, 0.2),
            "May M0.3": _polar(may, 0.3),
            "Oct M0.3": _polar(reduced["Oct26"][0][0], 0.3),
        }

    def test_the_labview_era_sets_agree_tightly(self, reduced):
        """Run01, May M0.2 (rebuilt q) and May M0.3 agree to < 0.01 CL -
        which is also what validates reading 'M2' as Mach 0.2."""
        curves = self._curves(reduced)
        legacy = np.vstack([np.interp(self.ALPHAS, *curves[k]) for k in
                            ("Run01 M0.29", "May M0.2", "May M0.3")])
        spread = legacy.max(axis=0) - legacy.min(axis=0)
        assert np.all(spread < 0.01), dict(zip(self.ALPHAS, spread))

    def test_october_stays_near_the_labview_era_polar(self, reduced):
        """KNOWN OFFSET (2026-10-06): the Freestream-recorded October set
        reads CL about +0.01 at alpha 0 growing to +0.025 at alpha 12
        (~7 % more lift-curve slope) at the same Mach - a force, not a q,
        difference. Bounded here so a gross error (a doubled q, a swapped
        channel) still fails; the golden record pins the exact values."""
        curves = self._curves(reduced)
        legacy = np.mean([np.interp(self.ALPHAS, *curves[k]) for k in
                          ("Run01 M0.29", "May M0.2", "May M0.3")], axis=0)
        oct_ = np.interp(self.ALPHAS, *curves["Oct M0.3"])
        assert np.all(np.abs(oct_ - legacy) < 0.04), dict(
            zip(self.ALPHAS, oct_ - legacy))

    def test_lift_curve_slopes_agree(self, reduced):
        slopes = {}
        for k, (a, cl) in self._curves(reduced).items():
            lin = (a >= -4) & (a <= 8)
            slopes[k] = np.polyfit(a[lin], cl[lin], 1)[0]
        vals = np.array(list(slopes.values()))
        assert np.all((vals > 0.024) & (vals < 0.029)), slopes
        assert vals.max() - vals.min() < 0.002, slopes


# ── 3. golden numbers ───────────────────────────────────────────────────
TOL = {"mach": 1e-3, "Cl": 1e-3, "Cd": 1e-3, "Cs": 1e-3, "CRoll": 1e-3,
       "CPitch": 1e-3, "CYaw": 1e-3}


@pytest.mark.skipif(not GOLDEN.exists(), reason="no golden record yet")
@pytest.mark.parametrize("key", list(SETS))
def test_matches_golden(reduced, key):
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))[key]
    snap = _snapshot(reduced[key][0][0])
    for field in ("alpha", "beta", "mach_step"):
        assert snap[field] == pytest.approx(golden[field], abs=1e-6), field
    for field, tol in TOL.items():
        got, want = np.asarray(snap[field]), np.asarray(golden[field])
        assert got.shape == want.shape, field
        bad = np.flatnonzero(~np.isclose(got, want, atol=tol,
                                         equal_nan=True))
        assert bad.size == 0, (
            f"{key}.{field} moved at {bad.size} point(s), first "
            f"alpha={snap['alpha'][bad[0]]} beta={snap['beta'][bad[0]]} "
            f"mach={snap['mach_step'][bad[0]]}: {got[bad[0]]:.5f} vs "
            f"golden {want[bad[0]]:.5f}")


def _update_golden():
    record = {}
    for key, sub in SETS.items():
        cases, _ = _reduce(sub, vol=(key != "Oct26"))
        record[key] = _snapshot(cases[0])
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {GOLDEN}")


if __name__ == "__main__" and "--update-golden" in sys.argv:
    _update_golden()
