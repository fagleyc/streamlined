"""The calibration travels with the raw volts it converts.

Freestream records the tunnel channels as RAW VOLTS with their
calibration attached (cal_slope / cal_offset / cal_unit / cal_type), and
the reduction applies it - that half already worked. What the EXPORT did
was carry the volts out and leave the calibration behind: Raw.Pdiff came
out as 0.836 with nothing in the file saying it was volts, let alone
that 0.386949 psid/V turns it into pressure.

MAT gets a Channel_Cal group beside Raw. HDF5 gets the same four
attributes freestream writes, on the Raw datasets themselves, so an
exported file reads back into Streamlined with its calibration intact.
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
scipy_io = pytest.importorskip("scipy.io")

from utils.windtunnel.coefficients import (  # noqa: E402
    calc_tunnel_conditions)
from utils.windtunnel.data_io import (  # noqa: E402
    MANIFEST_FILENAME, MANIFEST_SCHEMA_VERSION, read_run_file)

CHANNELS = ('Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz')

#: Volts, and the calibration that turns them into engineering units.
CAL = {
    'Pdiff': {'slope': 0.386949, 'offset': 0.0, 'unit': 'psid',
              'type': 'linear'},
    'Ptot': {'slope': 1.92604, 'offset': 0.0, 'unit': 'psia',
             'type': 'linear'},
    'Temp': {'slope': 10.0, 'offset': 0.0, 'unit': 'degC',
             'type': 'linear'},
}
VOLTS = {'Pdiff': 0.836, 'Ptot': 5.9094, 'Temp': 2.5683}


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([sys.argv[0]])


def _write_run(directory, name, alpha, mach, air_state, n=8):
    directory.mkdir(parents=True, exist_ok=True)
    group = {c: np.full(n, 0.0 if air_state == 'AirOff' else 2.0)
             for c in CHANNELS}
    for ch, volts in VOLTS.items():
        group[ch] = np.full(n, volts)
    group['Alpha'] = np.full(n, alpha)
    group['Beta'] = np.zeros(n)
    # freestream writes the cal into meta.channels.<group>.<channel>
    chan_meta = {ch: {'cal_slope': CAL[ch]['slope'],
                      'cal_offset': CAL[ch]['offset'],
                      'cal_unit': CAL[ch]['unit'],
                      'cal_type': CAL[ch]['type']}
                 for ch in CAL}
    scipy_io.savemat(str(directory / name), {
        'ATE_Balance': group,
        'Time': {'Time': np.arange(n) / 50.0},
        'meta': {'run': {'balance_type': 'external',
                         'span_config': 'half', 'air_state': air_state,
                         'alpha': alpha, 'beta': 0.0},
                 'channels': {'ATE_Balance': chan_meta},
                 'config_json': json.dumps({})},
    }, long_field_names=True)


def _run_dir(tmp_path):
    d = tmp_path / "CalRun"
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
    return d


def _panel(tmp_path):
    from utils.gui.controllers.data_controller import ProcessingWorker
    from utils.gui.models.data_model import DataModel
    from utils.gui.models.settings import AppSettings
    from utils.gui.views.table_panel import TablePanel

    worker = ProcessingWorker(
        directories=[str(_run_dir(tmp_path))], balance_cal=None,
        pressure_cal=None, balance_cal_file=None, pressure_cal_file=None,
        geometry={'mac': 4.0, 'ref_area': 96.0, 'span': 24.0,
                  'mrc': [0.0, 0.0, 0.0], 'units': 'IPS'},
        settings={'facility': 'SWT', 'balance_config': 'Force',
                  'cal_type': 'Linear'})
    cases, errors = [], []
    worker.signals.case_ready.connect(cases.append)
    worker.signals.error.connect(lambda t, m: errors.append((t, m)))
    worker.run()
    assert not errors, errors
    model = DataModel()
    model.cases.add(cases[0])
    return TablePanel(model, AppSettings()), cases[0]


class TestTheRunFileAlreadyCarriesIt:
    """The acquisition half, which already worked - pinned so it stays."""

    def test_the_channels_are_volts_not_engineering_units(self, tmp_path):
        run = sorted(_run_dir(tmp_path).glob('*mach_0.20*.mat'))[0]
        raw, _ = read_run_file(str(run))
        assert float(np.mean(raw.data['Pdiff'])) == pytest.approx(0.836)

    def test_the_calibration_rides_along(self, tmp_path):
        run = sorted(_run_dir(tmp_path).glob('*mach_0.20*.mat'))[0]
        raw, _ = read_run_file(str(run))
        cal = raw.properties.get('channel_cal') or {}
        assert set(cal) >= {'Pdiff', 'Ptot', 'Temp'}
        assert cal['Pdiff']['slope'] == pytest.approx(0.386949)
        assert cal['Pdiff']['type'] == 'linear'

    def test_the_reduction_applies_it(self, tmp_path):
        run = sorted(_run_dir(tmp_path).glob('*mach_0.20*.mat'))[0]
        raw, _ = read_run_file(str(run))
        channels = dict(raw.data)
        channels['channel_cal'] = raw.properties['channel_cal']
        tunnel = calc_tunnel_conditions(channels, {}, 'SWT', '220', '690')
        q = float(np.mean(tunnel.Q))
        # Converted, not the raw volts: 0.836 V -> ~0.32 psi
        assert 0.25 < q < 0.40, q
        assert abs(q - 0.836) > 0.1, "the raw volts were used as pressure"


class TestTheMatExportCarriesIt:
    def test_there_is_a_channel_cal_group(self, qapp, tmp_path):
        panel, _ = _panel(tmp_path)
        path = tmp_path / "out.mat"
        panel._write_export(str(path), 'mat',
                            {'format': 'mat', 'case_scope': 'all',
                             'filepath': str(path),
                             'include_unsteady': False})
        case = scipy_io.loadmat(str(path), struct_as_record=False,
                                squeeze_me=True)['case_001']
        assert 'Channel_Cal' in case._fieldnames
        assert set(case.Channel_Cal._fieldnames) >= {'Pdiff', 'Ptot',
                                                     'Temp'}

    def test_the_entries_convert_the_exported_volts(self, qapp,
                                                     tmp_path):
        panel, _ = _panel(tmp_path)
        path = tmp_path / "out.mat"
        panel._write_export(str(path), 'mat',
                            {'format': 'mat', 'case_scope': 'all',
                             'filepath': str(path),
                             'include_unsteady': False})
        case = scipy_io.loadmat(str(path), struct_as_record=False,
                                squeeze_me=True)['case_001']
        volts = float(np.ravel(case.Raw.Pdiff)[0])
        entry = case.Channel_Cal.Pdiff
        assert volts == pytest.approx(0.836, abs=1e-3)
        assert float(entry.slope) == pytest.approx(0.386949)
        assert str(entry.unit) == 'psid'
        assert str(entry.type) == 'linear'
        # The pair is what makes the file self-describing
        assert volts * float(entry.slope) + float(entry.offset) == \
            pytest.approx(0.3235, abs=1e-3)

    def test_an_uncalibrated_run_gets_no_group(self, qapp, tmp_path):
        # Nothing channel-specific to record: the reduction used its
        # built-in defaults, and a bogus empty group would imply
        # otherwise.
        panel, case = _panel(tmp_path)
        for point in case.daq.red:
            point.air_on.pop('channel_cal', None)
        struct = panel._build_categorized_struct(case)
        assert 'Channel_Cal' not in struct


class TestTheHdf5ExportCarriesIt:
    def test_the_raw_datasets_carry_the_cal_attributes(self, qapp,
                                                        tmp_path):
        h5py = pytest.importorskip("h5py")
        panel, _ = _panel(tmp_path)
        path = tmp_path / "out.h5"
        panel._write_export(str(path), 'hdf5',
                            {'format': 'hdf5', 'case_scope': 'all',
                             'filepath': str(path),
                             'include_unsteady': False})
        with h5py.File(str(path), 'r') as handle:
            case_key = next(k for k in handle
                            if k not in ('calibration', 'geometry'))
            raw = handle[case_key]['Raw']
            attrs = dict(raw['Pdiff'].attrs)
            assert float(attrs['cal_slope']) == pytest.approx(0.386949)
            assert float(attrs['cal_offset']) == pytest.approx(0.0)
            unit = attrs['cal_unit']
            assert (unit.decode() if isinstance(unit, bytes)
                    else unit) == 'psid'

    def test_an_uncalibrated_channel_gets_no_attributes(self, qapp,
                                                         tmp_path):
        h5py = pytest.importorskip("h5py")
        panel, _ = _panel(tmp_path)
        path = tmp_path / "out.h5"
        panel._write_export(str(path), 'hdf5',
                            {'format': 'hdf5', 'case_scope': 'all',
                             'filepath': str(path),
                             'include_unsteady': False})
        with h5py.File(str(path), 'r') as handle:
            case_key = next(k for k in handle
                            if k not in ('calibration', 'geometry'))
            raw = handle[case_key]['Raw']
            # The balance channels are engineering units already; only
            # the calibrated tunnel channels carry a cal.
            assert 'cal_slope' not in dict(raw['Fx'].attrs)

    def test_channel_cal_is_not_written_as_a_dataset_group(self, qapp,
                                                            tmp_path):
        # It is metadata; writing dicts as datasets would silently drop
        # it (the writer swallows per-dataset failures).
        h5py = pytest.importorskip("h5py")
        panel, _ = _panel(tmp_path)
        path = tmp_path / "out.h5"
        panel._write_export(str(path), 'hdf5',
                            {'format': 'hdf5', 'case_scope': 'all',
                             'filepath': str(path),
                             'include_unsteady': False})
        with h5py.File(str(path), 'r') as handle:
            case_key = next(k for k in handle
                            if k not in ('calibration', 'geometry'))
            assert 'Channel_Cal' not in handle[case_key]


class TestTheAttributeNamesAreTheContract:
    def test_they_are_the_four_names_freestream_writes(self, qapp,
                                                        tmp_path):
        """The names are not arbitrary: they are the ones this project's
        own HDF5 reader looks for on a run file, so a consumer that
        knows Streamlined data knows how to read these.

        The export nests a case group above Raw, so the file is not
        itself a run file and does not read back as one.
        """
        h5py = pytest.importorskip("h5py")
        panel, _ = _panel(tmp_path)
        path = tmp_path / "out.h5"
        panel._write_export(str(path), 'hdf5',
                            {'format': 'hdf5', 'case_scope': 'all',
                             'filepath': str(path),
                             'include_unsteady': False})
        with h5py.File(str(path), 'r') as handle:
            case_key = next(k for k in handle
                            if k not in ('calibration', 'geometry'))
            attrs = set(handle[case_key]['Raw']['Pdiff'].attrs)
        assert attrs == {'cal_slope', 'cal_offset', 'cal_unit',
                         'cal_type'}, attrs
