"""Alpha/beta attitude offsets in the geometry (bent-sting rectification).

A bent or drooped sting puts the model at a different attitude from the
one the positioner recorded: the encoder reports the sting root, the
model sits at root + offset. The geometry therefore carries an alpha and
a beta offset [deg] that are ADDED to every point's recorded attitude,
air-on and air-off alike, before anything reads the attitude. Reducing a
point recorded at 0 deg with a 12 deg offset must be indistinguishable
from reducing one recorded at 12 deg with no offset.

The raw Alpha/Beta channels are NOT touched: they are exported as
recorded, and only the reduced attitude carries the correction.
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

from utils.windtunnel.data_io import (  # noqa: E402
    MANIFEST_FILENAME, MANIFEST_SCHEMA_VERSION)
from utils.windtunnel.daq import DAQ  # noqa: E402
from utils.windtunnel.reduction import reduce_single_point  # noqa: E402
from utils.windtunnel.transforms import Geometry  # noqa: E402

scipy_io = pytest.importorskip("scipy.io")

CHANNELS = ("Lift", "Drag", "Side", "Roll", "Pitch", "Yaw")


def _point(alpha=0.0, beta=0.0, n=16, span="half"):
    """One external-balance point with a tare, all channels constant.

    Half span is used because there the wind-axis resolution rotates by
    alpha, so a wrong attitude shows up in the reduced loads and not
    only in the reported angle.
    """
    base = {"Lift": 0.5, "Drag": 2.0, "Side": 20.0,
            "Roll": 1.0, "Pitch": 3.0, "Yaw": 7.0}
    on = {k: np.full(n, v, dtype=float) for k, v in base.items()}
    on.update({"Alpha": np.full(n, float(alpha)),
               "Beta": np.full(n, float(beta)),
               "Pdiff": np.full(n, 0.8), "Ptot": np.full(n, 12.2),
               "Temp": np.full(n, 295.0),
               "balance_type": "external", "span_config": span})
    off = {k: (np.zeros(n) if k in CHANNELS else v) for k, v in on.items()}
    return on, off


def _reduce(on, off, alpha_offset=0.0, beta_offset=0.0):
    geo = Geometry(C=2.86, S=18.75, b=6.0, mshift=np.zeros(3),
                   alpha_offset=alpha_offset, beta_offset=beta_offset)
    return reduce_single_point(on, off, cal=None, geo=geo,
                               pressure_cal={}, facility='SWT')


class TestReductionAppliesTheOffset:
    def test_reduced_alpha_and_beta_carry_the_offset(self):
        on, off = _point(alpha=4.0, beta=1.0)
        red = _reduce(on, off, alpha_offset=0.5, beta_offset=-0.25)
        assert np.allclose(red.alpha, 4.5)
        assert np.allclose(red.beta, 0.75)

    def test_zero_offset_changes_nothing(self):
        on, off = _point(alpha=4.0)
        plain = _reduce(on, off)
        zero = _reduce(on, off, alpha_offset=0.0, beta_offset=0.0)
        assert np.allclose(plain.alpha, zero.alpha)
        assert np.allclose(np.mean(plain.wrf_aero.Lift),
                           np.mean(zero.wrf_aero.Lift))

    def test_offset_point_reduces_like_a_point_recorded_at_that_angle(
            self):
        # The whole purpose: (recorded 0, offset 12) == (recorded 12, no
        # offset), in the reduced loads and not just the reported angle.
        on0, off0 = _point(alpha=0.0)
        on12, off12 = _point(alpha=12.0)
        corrected = _reduce(on0, off0, alpha_offset=12.0)
        reference = _reduce(on12, off12)
        for ch in ("Lift", "Drag", "Side"):
            assert np.allclose(np.mean(getattr(corrected.wrf_aero, ch)),
                               np.mean(getattr(reference.wrf_aero, ch))), ch
        assert np.allclose(np.mean(corrected.coeffs.Cl),
                           np.mean(reference.coeffs.Cl))

    def test_offset_actually_moves_the_loads(self):
        # Guard against the previous test passing trivially: at half span
        # the resolved loads depend on alpha, so 12 deg must differ from 0.
        on, off = _point(alpha=0.0)
        plain = _reduce(on, off)
        bent = _reduce(on, off, alpha_offset=12.0)
        assert not np.allclose(np.mean(plain.wrf_aero.Lift),
                               np.mean(bent.wrf_aero.Lift))

    def test_the_tare_attitude_is_offset_too(self):
        # A tare taken at a different recorded attitude is rotated by ITS
        # alpha; the sting is just as bent then, so it gets the offset.
        n = 16
        on, _ = _point(alpha=12.0, n=n)
        off = {k: (np.zeros(n) if k in CHANNELS else v)
               for k, v in on.items()}
        off["Alpha"] = np.zeros(n)
        # Give the tare a real load so its rotation matters
        off["Side"] = np.full(n, 20.0)

        corrected = _reduce(on, off, alpha_offset=3.0)

        # Same thing with the offset baked into the recorded angles
        on_ref, _ = _point(alpha=15.0, n=n)
        off_ref = dict(off)
        off_ref["Alpha"] = np.full(n, 3.0)
        reference = _reduce(on_ref, off_ref)

        assert np.allclose(np.mean(corrected.wrf_aero.Lift),
                           np.mean(reference.wrf_aero.Lift))

    def test_raw_channels_stay_as_recorded(self):
        on, off = _point(alpha=4.0, beta=1.0)
        red = _reduce(on, off, alpha_offset=0.5, beta_offset=0.5)
        assert np.allclose(red.air_on["Alpha"], 4.0)
        assert np.allclose(red.air_on["Beta"], 1.0)
        assert np.allclose(red.air_off["Alpha"], 4.0)

    def test_a_geometry_without_the_fields_still_reduces(self):
        # Old Geometry objects (or a caller passing a bare namespace)
        # have no offset attributes; that must read as zero.
        on, off = _point(alpha=4.0)
        geo = Geometry(C=2.86, S=18.75, b=6.0, mshift=np.zeros(3))
        del geo.alpha_offset
        del geo.beta_offset
        red = reduce_single_point(on, off, cal=None, geo=geo,
                                  pressure_cal={}, facility='SWT')
        assert np.allclose(red.alpha, 4.0)


class TestDaqSetGeometry:
    def test_offsets_reach_the_geometry(self):
        daq = DAQ()
        daq.set_geometry(MAC=4.0, S=96.0, MRC=[0, 0, 0], span=24.0,
                         alpha_offset=0.75, beta_offset=-0.5)
        assert daq.geo[0].alpha_offset == pytest.approx(0.75)
        assert daq.geo[0].beta_offset == pytest.approx(-0.5)

    def test_offsets_default_to_zero(self):
        daq = DAQ()
        daq.set_geometry(MAC=4.0, S=96.0, MRC=[0, 0, 0], span=24.0)
        assert daq.geo[0].alpha_offset == 0.0
        assert daq.geo[0].beta_offset == 0.0

    def test_offsets_are_degrees_whatever_the_length_units(self):
        # The unit system scales lengths and areas; an angle is an angle.
        for units in ("IPS", "FPS", "MKS", "CGS"):
            daq = DAQ()
            daq.set_geometry(MAC=1.0, S=1.0, MRC=[0, 0, 0], units=units,
                             alpha_offset=2.0)
            assert daq.geo[0].alpha_offset == pytest.approx(2.0), units


class TestModelAndDialog:
    @pytest.fixture(scope="class")
    def qapp(self):
        from PyQt6.QtWidgets import QApplication
        return QApplication.instance() or QApplication([sys.argv[0]])

    def test_model_default_geometry_has_zero_offsets(self):
        from utils.gui.models.data_model import DataModel
        model = DataModel()
        assert model.alpha_offset == 0.0
        assert model.beta_offset == 0.0
        assert model.geometries['Default']['alpha_offset'] == 0.0

    def test_model_reads_offsets_from_the_default_geometry(self):
        from utils.gui.models.data_model import DataModel
        model = DataModel()
        model.geometries['Default']['alpha_offset'] = 0.3
        model.geometries['Default']['beta_offset'] = -0.1
        assert model.alpha_offset == pytest.approx(0.3)
        assert model.beta_offset == pytest.approx(-0.1)

    def test_model_tolerates_a_geometry_saved_without_the_keys(self):
        from utils.gui.models.data_model import DataModel
        model = DataModel()
        model.geometries['Default'] = {
            'mac': 1.0, 'ref_area': 1.0, 'span': 1.0,
            'mrc': [0.0, 0.0, 0.0], 'units': 'IPS'}
        assert model.alpha_offset == 0.0

    def test_dialog_round_trips_the_offsets(self, qapp):
        from utils.gui.views.dialogs import GeometryDialog
        dialog = GeometryDialog(geometries={
            'Wing A': {'mac': 4.0, 'ref_area': 96.0, 'span': 24.0,
                       'mrc': [0.0, 0.0, 0.0], 'units': 'IPS',
                       'alpha_offset': 0.4, 'beta_offset': -0.2}})
        assert dialog.spn_alpha_offset.value() == pytest.approx(0.4)
        assert dialog.spn_beta_offset.value() == pytest.approx(-0.2)
        dialog.spn_alpha_offset.setValue(1.25)
        dialog._on_accept()
        geo = dialog.get_geometries()['Wing A']
        assert geo['alpha_offset'] == pytest.approx(1.25)
        assert geo['beta_offset'] == pytest.approx(-0.2)

    def test_dialog_loads_a_legacy_geometry_as_zero_offset(self, qapp):
        from utils.gui.views.dialogs import GeometryDialog
        dialog = GeometryDialog(geometries={
            'Old': {'mac': 4.0, 'ref_area': 96.0, 'span': 24.0,
                    'mrc': [0.0, 0.0, 0.0], 'units': 'IPS'}})
        assert dialog.spn_alpha_offset.value() == 0.0
        dialog._on_accept()
        assert dialog.get_geometries()['Old']['alpha_offset'] == 0.0

    def test_a_new_geometry_starts_with_zero_offsets(self, qapp):
        from utils.gui.views.dialogs import GeometryDialog
        dialog = GeometryDialog()
        dialog._add_geometry()
        dialog._on_accept()
        for geo in dialog.get_geometries().values():
            assert geo['alpha_offset'] == 0.0
            assert geo['beta_offset'] == 0.0


# ── end to end through the processing worker ────────────────────────────
def _write_run(directory: Path, name: str, alpha: float, air_state: str,
               n: int = 16) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    group = {c: np.ones(n) for c in CHANNELS}
    group["Pdiff"] = np.full(n, 0.8)
    group["Ptot"] = np.full(n, 12.2)
    group["Temp"] = np.full(n, 295.0)
    group["Alpha"] = np.full(n, alpha)
    group["Beta"] = np.zeros(n)
    scipy_io.savemat(str(directory / name), {
        "ATE_Balance": group,
        "Time": {"Time": np.arange(n) / 50.0},
        "meta": {"run": {"balance_type": "external", "span_config": "half",
                         "air_state": air_state, "alpha": alpha,
                         "beta": 0.0},
                 "config_json": json.dumps({})},
    }, long_field_names=True)


def _run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "bent_sting"
    points = [(1, 0.0, "AirOff"), (2, 4.0, "AirOff"),
              (3, 0.0, "AirOn"), (4, 4.0, "AirOn")]
    manifest = []
    for run, alpha, state in points:
        name = "run_%04d_alpha_%.1f_beta_0.0.mat" % (run, alpha)
        _write_run(d, name, alpha, state)
        manifest.append({"run_number": run, "filename": name,
                         "alpha": alpha, "beta": 0.0, "mach": 0.05,
                         "air_state": state})
    (d / MANIFEST_FILENAME).write_text(json.dumps(
        {"schema_version": MANIFEST_SCHEMA_VERSION, "config_name": d.name,
         "output_format": "mat", "points": manifest}), encoding="utf-8")
    return d


def _process(directory: Path, **geometry_extra):
    from utils.gui.controllers.data_controller import ProcessingWorker
    geometry = {'mac': 4.0, 'ref_area': 96.0, 'span': 24.0,
                'mrc': [0.0, 0.0, 0.0], 'units': 'IPS'}
    geometry.update(geometry_extra)
    worker = ProcessingWorker(
        directories=[str(directory)], balance_cal=None, pressure_cal=None,
        balance_cal_file=None, pressure_cal_file=None,
        geometry=geometry,
        settings={'facility': 'SWT', 'balance_config': 'Force',
                  'cal_type': 'Linear'})
    cases, errors = [], []
    worker.signals.case_ready.connect(cases.append)
    worker.signals.error.connect(lambda t, m: errors.append((t, m)))
    worker.run()
    assert not errors, errors
    assert len(cases) == 1
    return cases[0]


class TestWorkerCarriesTheOffset:
    @pytest.fixture(scope="class")
    def qapp(self):
        from PyQt6.QtWidgets import QApplication
        return QApplication.instance() or QApplication([sys.argv[0]])

    def test_case_alphas_are_the_corrected_attitude(self, qapp, tmp_path):
        case = _process(_run_dir(tmp_path), alpha_offset=0.5)
        np.testing.assert_allclose(np.sort(np.ravel(case.alphas)),
                                   [0.5, 4.5])

    def test_no_offset_leaves_the_recorded_attitude(self, qapp, tmp_path):
        case = _process(_run_dir(tmp_path))
        np.testing.assert_allclose(np.sort(np.ravel(case.alphas)),
                                   [0.0, 4.0])

    def test_the_daq_geometry_carries_it_for_reprocessing(self, qapp,
                                                          tmp_path):
        # reprocess_case re-reduces from case.daq, so the offset has to be
        # on the DAQ geometry and not only in the worker's dict.
        case = _process(_run_dir(tmp_path), alpha_offset=0.5,
                        beta_offset=0.1)
        assert case.daq.geo[0].alpha_offset == pytest.approx(0.5)
        assert case.daq.geo[0].beta_offset == pytest.approx(0.1)
