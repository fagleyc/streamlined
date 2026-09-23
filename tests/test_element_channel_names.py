"""The six element channels are named, and filled, by their own balance.

Every balance produces six channels into the same slots (columns 0..5 of
BRFForces.elements), but which six depends on the balance. Exports named
them N1/N2/Y1/Y2/Axial/Roll unconditionally - the internal FORCE
balance's channels - so an external run exported six columns of zeros
under names its balance never had.

Two things were wrong and both are fixed here: the NAMES now come from
the balance that recorded the run, and for an external balance the slots
are FILLED with its own six channels instead of being left empty. The
external set is also three forces and three moments, so the unit label
varies per slot.
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

pytest.importorskip("PyQt6")

from utils.gui.models.case import TestCase as Case  # noqa: E402
from utils.windtunnel.external_balance import (  # noqa: E402
    EXTERNAL_CHANNEL_ORDER)
from utils.windtunnel.reduction import reduce_single_point  # noqa: E402
from utils.windtunnel.transforms import (  # noqa: E402
    Geometry, element_channel_names, element_channels)

scipy_io = pytest.importorskip("scipy.io")

SLOTS = ('elem_N1', 'elem_N2', 'elem_Y1', 'elem_Y2', 'elem_Ax', 'elem_Roll')


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([sys.argv[0]])


class TestTheNameTable:
    def test_external(self):
        assert element_channel_names('external') == (
            'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz')

    def test_internal_force(self):
        assert element_channel_names('internal', 'Force') == (
            'N1', 'N2', 'Y1', 'Y2', 'Axial', 'Roll')

    def test_internal_moment(self):
        assert element_channel_names('internal', 'Moment') == (
            'AftPitch', 'AftYaw', 'FwdPitch', 'FwdYaw', 'Axial', 'Roll')

    def test_the_external_names_match_the_balance_module(self):
        # transforms names the slots and external_balance stacks them;
        # if the two ever disagree the export mislabels every column.
        assert element_channel_names('external') == EXTERNAL_CHANNEL_ORDER

    def test_an_external_balance_ignores_the_config(self):
        # It has only one channel set; Force/Moment is an internal
        # calibration distinction.
        assert (element_channel_names('external', 'Moment')
                == element_channel_names('external', 'Force'))

    def test_an_unknown_balance_reads_as_internal_force(self):
        # The historical default, and what a run recording neither
        # should still produce.
        assert element_channel_names('') == (
            'N1', 'N2', 'Y1', 'Y2', 'Axial', 'Roll')
        assert element_channel_names('', 'nonsense') == (
            'N1', 'N2', 'Y1', 'Y2', 'Axial', 'Roll')

    def test_case_is_ignored_on_the_markers(self):
        assert element_channel_names('EXTERNAL') == \
            element_channel_names('external')
        assert element_channel_names('internal', 'moment') == \
            element_channel_names('internal', 'Moment')

    def test_an_external_balance_reads_three_moments(self):
        kinds = [kind for _, kind in element_channels('external')]
        assert kinds == ['force', 'force', 'force',
                         'moment', 'moment', 'moment']

    def test_an_internal_balance_reads_six_forces(self):
        for config in ('Force', 'Moment'):
            kinds = [k for _, k in element_channels('internal', config)]
            assert kinds == ['force'] * 6, config

    def test_every_balance_offers_exactly_six(self):
        for bt, bc in (('external', 'Force'), ('internal', 'Force'),
                       ('internal', 'Moment'), ('', '')):
            assert len(element_channels(bt, bc or 'Force')) == 6


# ── the reduction fills the slots for an external balance ───────────────
CHANNELS = ('Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz')
_VALUES = {'Fx': 2.0, 'Fy': 20.0, 'Fz': 0.5,
           'Mx': 1.0, 'My': 3.0, 'Mz': 7.0}


def _external_point(n=8, load_units='lb'):
    on = {k: np.full(n, v, dtype=float) for k, v in _VALUES.items()}
    on.update({'Alpha': np.full(n, 4.0), 'Beta': np.zeros(n),
               'Pdiff': np.full(n, 0.8), 'Ptot': np.full(n, 12.2),
               'Temp': np.full(n, 295.0),
               'balance_type': 'external', 'span_config': 'half',
               'load_units': load_units})
    off = {k: (np.zeros(n) if k in CHANNELS else v) for k, v in on.items()}
    return on, off


def _reduce(on, off):
    geo = Geometry(C=2.86, S=18.75, b=6.0, mshift=np.zeros(3))
    return reduce_single_point(on, off, cal=None, geo=geo,
                               pressure_cal={}, facility='SWT')


class TestExternalSlotsAreFilled:
    def test_the_elements_are_no_longer_empty(self):
        red = _reduce(*_external_point())
        elements = np.asarray(red.brf_on.elements)
        assert elements.ndim == 2 and elements.shape[1] == 6
        assert elements.size > 0

    def test_they_hold_the_balance_frame_channels_in_slot_order(self):
        red = _reduce(*_external_point())
        elements = np.asarray(red.brf_on.elements)
        for col, name in enumerate(EXTERNAL_CHANNEL_ORDER):
            assert np.allclose(elements[:, col], _VALUES[name]), name

    def test_they_are_unit_converted(self):
        # lbft pairs lbf with lbf*FT, so the moments scale by 12.
        red = _reduce(*_external_point(load_units='lbft'))
        elements = np.asarray(red.brf_on.elements)
        assert np.allclose(elements[:, 0], _VALUES['Fx'])        # force
        assert np.allclose(elements[:, 3], _VALUES['Mx'] * 12.0)  # moment

    def test_the_tare_is_filled_too(self):
        red = _reduce(*_external_point())
        assert np.asarray(red.brf_off.elements).size > 0

    def test_the_body_axis_forces_stay_empty(self):
        # There is no body-axis reduction on the external path; only
        # the element matrix is meaningful.
        red = _reduce(*_external_point())
        assert np.asarray(red.brf_on.Fx).size == 0


# ── end to end: the case, the table and the exports ─────────────────────
def _write_run(directory, name, alpha, mach, air_state, n=8):
    directory.mkdir(parents=True, exist_ok=True)
    # The tare carries no load, so air-on minus tare is nonzero and a
    # zero-filled slot really means the slot was never populated.
    group = {k: np.full(n, 0.0 if air_state == 'AirOff' else v)
             for k, v in _VALUES.items()}
    group['Pdiff'] = np.full(n, 0.8)
    group['Ptot'] = np.full(n, 12.2)
    group['Temp'] = np.full(n, 295.0)
    group['Alpha'] = np.full(n, alpha)
    group['Beta'] = np.zeros(n)
    scipy_io.savemat(str(directory / name), {
        'ATE_Balance': group,
        'Time': {'Time': np.arange(n) / 50.0},
        'meta': {'run': {'balance_type': 'external',
                         'span_config': 'half', 'air_state': air_state,
                         'alpha': alpha, 'beta': 0.0},
                 'config_json': json.dumps({})},
    }, long_field_names=True)


def _external_case(tmp_path):
    from utils.gui.controllers.data_controller import ProcessingWorker
    from utils.windtunnel.data_io import (MANIFEST_FILENAME,
                                          MANIFEST_SCHEMA_VERSION)
    d = tmp_path / "ExternalRun"
    points, run = [], 0
    for state, mach in (("AirOff", 0.0), ("AirOn", 0.2)):
        for alpha in (-2.0, 0.0, 2.0):
            run += 1
            name = "run_%04d_alpha_%.1f_beta_0.0_mach_%.2f.mat" % (
                run, alpha, mach)
            _write_run(d, name, alpha, mach, state)
            points.append({'run_number': run, 'filename': name,
                           'alpha': alpha, 'beta': 0.0, 'mach': mach,
                           'air_state': state})
    (d / MANIFEST_FILENAME).write_text(json.dumps({
        'schema_version': MANIFEST_SCHEMA_VERSION, 'config_name': d.name,
        'output_format': 'mat', 'points': points}), encoding='utf-8')

    worker = ProcessingWorker(
        directories=[str(d)], balance_cal=None, pressure_cal=None,
        balance_cal_file=None, pressure_cal_file=None,
        geometry={'mac': 4.0, 'ref_area': 96.0, 'span': 24.0,
                  'mrc': [0.0, 0.0, 0.0], 'units': 'IPS'},
        settings={'facility': 'SWT', 'balance_config': 'Force',
                  'cal_type': 'Linear'})
    cases, errors = [], []
    worker.signals.case_ready.connect(cases.append)
    worker.signals.error.connect(lambda t, m: errors.append((t, m)))
    worker.run()
    assert not errors, errors
    return cases[0]


class TestTheCaseCarriesItsBalance:
    def test_an_external_run_is_recorded_as_external(self, qapp,
                                                      tmp_path):
        case = _external_case(tmp_path)
        assert case.balance_type == 'external'

    def test_the_case_names_its_own_channels(self, qapp, tmp_path):
        case = _external_case(tmp_path)
        assert [n for n, _ in case.element_channels] == [
            'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']

    def test_the_slots_carry_values_not_zeros(self, qapp, tmp_path):
        case = _external_case(tmp_path)
        for slot in SLOTS:
            values = np.ravel(getattr(case, slot))
            assert values.size > 0, slot
            assert np.any(values != 0.0), slot

    def test_a_case_that_says_nothing_falls_back_to_internal(self):
        case = Case(id="x", name="x")
        assert [n for n, _ in case.element_channels] == [
            'N1', 'N2', 'Y1', 'Y2', 'Axial', 'Roll']


class TestTableAndExportsUseThem:
    def _panel(self, qapp, case):
        from utils.gui.models.data_model import DataModel
        from utils.gui.models.settings import AppSettings
        from utils.gui.views.table_panel import TablePanel
        model = DataModel()
        model.cases.add(case)
        return TablePanel(model, AppSettings())

    def test_the_table_columns_are_named_for_the_balance(self, qapp,
                                                          tmp_path):
        panel = self._panel(qapp, _external_case(tmp_path))
        headers = [panel.table.horizontalHeaderItem(i).text()
                   for i in range(panel.table.columnCount())
                   if panel.table.horizontalHeaderItem(i)]
        for name in ('Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'):
            assert any(h.startswith(name + ' ') for h in headers), name
        for stale in ('N1', 'N2', 'Y1', 'Y2'):
            assert not any(h.startswith(stale + ' ') for h in headers), \
                stale

    def test_the_moment_slots_carry_a_moment_unit(self, qapp, tmp_path):
        panel = self._panel(qapp, _external_case(tmp_path))
        headers = [panel.table.horizontalHeaderItem(i).text()
                   for i in range(panel.table.columnCount())
                   if panel.table.horizontalHeaderItem(i)]
        fx = next(h for h in headers if h.startswith('Fx '))
        mx = next(h for h in headers if h.startswith('Mx '))
        assert 'lbf' in fx, fx
        assert 'lb-in' in mx, mx

    def test_a_csv_export_uses_them(self, qapp, tmp_path):
        import csv
        case = _external_case(tmp_path)
        panel = self._panel(qapp, case)
        path = tmp_path / "out.csv"
        panel._write_export(str(path), 'csv',
                            {'format': 'csv', 'case_scope': 'all',
                             'filepath': str(path),
                             'include_unsteady': False})
        with open(path, newline='') as fh:
            header = next(csv.reader(fh))
        for name in ('Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'):
            assert any(h.startswith(name + ' ') for h in header), name
        assert not [h for h in header
                    if h.split()[0] in ('N1', 'N2', 'Y1', 'Y2')]

    def test_the_unsteady_export_uses_them(self, qapp, tmp_path):
        from utils.gui.views.table_panel import TablePanel
        case = _external_case(tmp_path)
        panel = self._panel(qapp, case)
        data = panel._extract_unsteady_point(case.daq.red[0])
        for name in ('Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'):
            assert 'element_%s' % name in data, name
        for stale in ('N1', 'N2', 'Y1', 'Y2'):
            assert 'element_%s' % stale not in data, stale


class TestInternalIsUnchanged:
    def test_a_force_balance_still_reads_N1(self, qapp):
        from utils.gui.models.data_model import DataModel
        from utils.gui.models.settings import AppSettings
        from utils.gui.views.table_panel import TablePanel
        case = Case(id="i", name="Internal")
        case.alphas = np.array([0.0, 2.0])
        case.betas = np.zeros(2)
        case.Cl = np.array([0.1, 0.2])
        case.Cd = np.array([0.02, 0.03])
        case.balance_type = 'internal'
        case.balance_config = 'Force'
        case.elem_N1 = np.array([1.0, 2.0])
        model = DataModel()
        model.cases.add(case)
        panel = TablePanel(model, AppSettings())
        headers = [panel.table.horizontalHeaderItem(i).text()
                   for i in range(panel.table.columnCount())
                   if panel.table.horizontalHeaderItem(i)]
        assert any(h.startswith('N1 ') for h in headers), headers
        assert not any(h.startswith('Fx ') for h in headers)

    def test_a_moment_balance_reads_its_own_names(self, qapp):
        from utils.gui.models.data_model import DataModel
        from utils.gui.models.settings import AppSettings
        from utils.gui.views.table_panel import TablePanel
        case = Case(id="m", name="Moment")
        case.alphas = np.array([0.0, 2.0])
        case.betas = np.zeros(2)
        case.Cl = np.array([0.1, 0.2])
        case.Cd = np.array([0.02, 0.03])
        case.balance_type = 'internal'
        case.balance_config = 'Moment'
        case.elem_N1 = np.array([1.0, 2.0])
        model = DataModel()
        model.cases.add(case)
        panel = TablePanel(model, AppSettings())
        headers = [panel.table.horizontalHeaderItem(i).text()
                   for i in range(panel.table.columnCount())
                   if panel.table.horizontalHeaderItem(i)]
        assert any(h.startswith('AftPitch ') for h in headers), headers
        assert any(h.startswith('FwdYaw ') for h in headers), headers
