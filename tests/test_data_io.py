"""
Tests for utils.windtunnel.data_io run-file readers.

Covers both wind tunnel run-file flavors written by Freestream/Conductor:

* Legacy / Mode-1 (internal sting balance): balance group StrainBook_0
  with bridge-volt channels N1, N2, Y1, Y2, Axial, Roll (needs .vol
  calibration via calc_brf_forces).
* Mode-2 (ATE external balance): balance group ATE_Balance with
  resolved wind-axis loads Lift, Pitch, Drag, Side, Yaw, Roll in
  N / N*m (NO calibration needed), self-described by balance_group /
  balance_type markers in the root attrs and/or /meta.

Run with:
    cd Streamlined
    python -m pytest tests/test_data_io.py -q
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

h5py = pytest.importorskip('h5py')
from scipy import io as scipy_io  # noqa: E402

from utils.windtunnel.data_io import (  # noqa: E402
    RawData,
    read_hdf5_file, read_mat_file, read_run_file,
    classify_files_by_condition, extract_mach_from_filename,
    extract_speed_from_filename, extract_sort_key_from_filename,
    speed_condition_key, copy_balance_markers,
    extract_air_state_from_filename, extract_configuration_from_filename,
    group_files_by_configuration,
    BALANCE_GROUP_INTERNAL, BALANCE_GROUP_EXTERNAL,
)
from utils.windtunnel.transforms import (  # noqa: E402
    calc_brf_forces, is_external_balance_data, wrf_from_resolved_loads,
)
from utils.windtunnel.reduction import (  # noqa: E402
    reduce_single_point, reduce_steady_state, to_dataframe,
)
from utils.windtunnel.transforms import Geometry  # noqa: E402


N_FAST = 100          # fast (balance) channel samples
DT_FAST = 0.001
N_SLOW = 10           # slow (positioner) channel samples
DT_SLOW = 0.01

INTERNAL_CHANNELS = ('N1', 'N2', 'Y1', 'Y2', 'Axial', 'Roll')
EXTERNAL_CHANNELS = ('Lift', 'Pitch', 'Drag', 'Side', 'Yaw', 'Roll')


def _channel_wave(i, n=N_FAST):
    """Deterministic per-channel test signal."""
    return np.linspace(0.0, 1.0, n) + i


def _write_channel(group, name, data, dt):
    dset = group.create_dataset(name, data=data)
    dset.attrs['wf_increment'] = dt
    dset.attrs['wf_samples'] = len(data)


def _write_common(h5, alpha=2.0):
    """Positioner + Time groups shared by both flavors."""
    pos = h5.create_group('Positioner')
    _write_channel(pos, 'Alpha',
                   np.full(N_SLOW, alpha), DT_SLOW)
    time_grp = h5.create_group('Time')
    time_grp.create_dataset('Time', data=np.arange(N_FAST) * DT_FAST)


def write_legacy_h5(path, alpha=2.0):
    """Old-style file: StrainBook_0 bridge volts, no balance markers."""
    with h5py.File(path, 'w') as h5:
        sb = h5.create_group(BALANCE_GROUP_INTERNAL)
        for i, name in enumerate(INTERNAL_CHANNELS):
            _write_channel(sb, name, _channel_wave(i), DT_FAST)
        _write_common(h5, alpha)
        h5.attrs['run_number'] = 1
        h5.attrs['Alpha'] = alpha
    return path


def write_external_h5(path, alpha=2.0, root_markers=True,
                      meta_markers=True):
    """Mode-2 file: ATE_Balance resolved loads + self-describing markers."""
    with h5py.File(path, 'w') as h5:
        ate = h5.create_group(BALANCE_GROUP_EXTERNAL)
        for i, name in enumerate(EXTERNAL_CHANNELS):
            _write_channel(ate, name, _channel_wave(i), DT_FAST)
        _write_common(h5, alpha)
        h5.attrs['run_number'] = 4
        if root_markers:
            h5.attrs['balance_group'] = BALANCE_GROUP_EXTERNAL
            h5.attrs['balance_type'] = 'external'
            h5.attrs['positions_source'] = 'positioner'
        if meta_markers:
            meta = h5.create_group('meta')
            meta.attrs['balance_group'] = BALANCE_GROUP_EXTERNAL
            meta.attrs['balance_type'] = 'external'
            ate_dev = meta.create_group('devices/ate')
            ate_dev.attrs['span_config'] = 'full'
            ate_dev.attrs['balance_type'] = 'external'
    return path


def write_speed_h5(path, alpha=0.0, speed_value=30.0, speed_unit='hz',
                   mach=0.0, setpoints=(0.0, 10.0, 20.0, 30.0)):
    """
    Non-Mach velocity-sweep file: ATE loads + speed root attrs matching
    the freestream CONTRACT (speed_unit/speed_value/mach/speed_setpoints).
    """
    with h5py.File(path, 'w') as h5:
        ate = h5.create_group(BALANCE_GROUP_EXTERNAL)
        for i, name in enumerate(EXTERNAL_CHANNELS):
            _write_channel(ate, name, _channel_wave(i), DT_FAST)
        _write_common(h5, alpha)
        h5.attrs['run_number'] = 7
        h5.attrs['balance_group'] = BALANCE_GROUP_EXTERNAL
        h5.attrs['balance_type'] = 'external'
        h5.attrs['speed_unit'] = speed_unit
        h5.attrs['speed_value'] = speed_value
        h5.attrs['mach'] = mach
        h5.attrs['speed_setpoints'] = np.array(setpoints, dtype=float)
    return path


# ---------------------------------------------------------------------------
# read_hdf5_file — legacy (internal balance) files
# ---------------------------------------------------------------------------

class TestReadHdf5Legacy:

    def test_channels_and_resampling(self, tmp_path):
        path = write_legacy_h5(tmp_path / 'run_0001_alpha_2.0_beta_0.0_mach_0.30.h5')
        raw, properties = read_hdf5_file(str(path))

        for i, name in enumerate(INTERNAL_CHANNELS):
            assert name in raw.data
            np.testing.assert_allclose(raw.data[name], _channel_wave(i))
        # Slow Alpha channel gets resampled onto the fast time base
        assert len(raw.time) == N_FAST
        assert len(raw.data['Alpha']) == N_FAST
        assert properties['Alpha'] == 2.0

    def test_defaults_to_internal_balance(self, tmp_path):
        path = write_legacy_h5(tmp_path / 'legacy.h5')
        raw, _ = read_hdf5_file(str(path))

        assert raw.properties['balance_type'] == 'internal'
        assert raw.properties['balance_group'] == BALANCE_GROUP_INTERNAL
        assert raw.balance_type == 'internal'
        assert raw.balance_group == BALANCE_GROUP_INTERNAL

    def test_root_attrs_still_carried(self, tmp_path):
        path = write_legacy_h5(tmp_path / 'legacy.h5')
        raw, _ = read_hdf5_file(str(path))
        assert raw.properties['run_number'] == 1


# ---------------------------------------------------------------------------
# read_hdf5_file — Mode-2 (ATE external balance) files
# ---------------------------------------------------------------------------

class TestReadHdf5External:

    def test_loads_under_in_file_names(self, tmp_path):
        path = write_external_h5(tmp_path / 'run_0004_alpha_2.0_beta_0.0_mach_0.30.h5')
        raw, _ = read_hdf5_file(str(path))

        for i, name in enumerate(EXTERNAL_CHANNELS):
            assert name in raw.data, f'missing resolved load channel {name}'
            np.testing.assert_allclose(raw.data[name], _channel_wave(i))
        # No bridge channels appear
        for name in ('N1', 'N2', 'Y1', 'Y2', 'Axial'):
            assert name not in raw.data

    def test_external_marker_carried(self, tmp_path):
        path = write_external_h5(tmp_path / 'ext.h5')
        raw, _ = read_hdf5_file(str(path))

        assert raw.properties['balance_type'] == 'external'
        assert raw.properties['balance_group'] == BALANCE_GROUP_EXTERNAL
        assert raw.properties['positions_source'] == 'positioner'
        assert raw.balance_type == 'external'
        assert raw.balance_group == BALANCE_GROUP_EXTERNAL

    def test_markers_from_meta_only(self, tmp_path):
        """Root attrs absent -> /meta markers still resolve the flavor."""
        path = write_external_h5(tmp_path / 'ext_meta.h5',
                                 root_markers=False, meta_markers=True)
        raw, _ = read_hdf5_file(str(path))
        assert raw.balance_type == 'external'
        assert raw.balance_group == BALANCE_GROUP_EXTERNAL

    def test_fallback_group_presence(self, tmp_path):
        """No markers at all -> ATE_Balance group presence wins."""
        path = write_external_h5(tmp_path / 'ext_bare.h5',
                                 root_markers=False, meta_markers=False)
        raw, _ = read_hdf5_file(str(path))
        assert raw.balance_type == 'external'
        assert raw.balance_group == BALANCE_GROUP_EXTERNAL

    def test_alpha_from_positioner(self, tmp_path):
        path = write_external_h5(tmp_path / 'ext.h5', alpha=3.0)
        raw, _ = read_hdf5_file(str(path))
        np.testing.assert_allclose(raw.data['Alpha'],
                                   np.full(N_FAST, 3.0), atol=1e-8)


# ---------------------------------------------------------------------------
# read_mat_file
# ---------------------------------------------------------------------------

class TestReadMat:

    def _write_mat(self, path, external):
        n = N_FAST
        if external:
            balance = {name: _channel_wave(i, n)
                       for i, name in enumerate(EXTERNAL_CHANNELS)}
            contents = {
                BALANCE_GROUP_EXTERNAL: balance,
                'Positioner': {'Alpha': np.full(n, 2.0)},
                'Time': {'Time': np.arange(n) * DT_FAST},
                'meta': {
                    'run': {'run_number': 4,
                            'balance_group': BALANCE_GROUP_EXTERNAL,
                            'balance_type': 'external'},
                    'devices': {'ate': {'span_config': 'full',
                                        'balance_type': 'external'}},
                },
            }
        else:
            balance = {name: _channel_wave(i, n)
                       for i, name in enumerate(INTERNAL_CHANNELS)}
            contents = {
                BALANCE_GROUP_INTERNAL: balance,
                'Positioner': {'Alpha': np.full(n, 2.0)},
                'Time': {'Time': np.arange(n) * DT_FAST},
                'meta': {'run': {'run_number': 1}},
            }
        scipy_io.savemat(str(path), contents)
        return path

    def test_external_mat(self, tmp_path):
        path = self._write_mat(tmp_path / 'run_0004_alpha_2.0_mach_0.30.mat',
                               external=True)
        raw, _ = read_mat_file(str(path))

        for i, name in enumerate(EXTERNAL_CHANNELS):
            assert name in raw.data
            np.testing.assert_allclose(raw.data[name], _channel_wave(i))
        assert raw.balance_type == 'external'
        assert raw.balance_group == BALANCE_GROUP_EXTERNAL

    def test_legacy_mat_defaults_internal(self, tmp_path):
        path = self._write_mat(tmp_path / 'run_0001.mat', external=False)
        raw, _ = read_mat_file(str(path))

        for name in INTERNAL_CHANNELS:
            assert name in raw.data
        assert raw.balance_type == 'internal'
        assert raw.balance_group == BALANCE_GROUP_INTERNAL


# ---------------------------------------------------------------------------
# read_run_file dispatch
# ---------------------------------------------------------------------------

class TestReadRunFile:

    def test_dispatches_h5(self, tmp_path):
        path = write_external_h5(tmp_path / 'run_0004_mach_0.30.h5')
        raw, _ = read_run_file(str(path))
        assert 'Lift' in raw.data
        assert raw.balance_type == 'external'


# ---------------------------------------------------------------------------
# Filename classification (mode-2 names must keep working)
# ---------------------------------------------------------------------------

class TestClassification:

    def test_mode2_filenames_classify(self):
        files = [
            'run_0001_alpha_2.0_beta_0.0_mach_0.00.h5',
            'run_0002_alpha_2.0_beta_0.0_mach_0.30.h5',
            'run_0003_alpha_4.0_beta_0.0_mach_0.30.h5',
        ]
        classified = classify_files_by_condition(files)
        assert classified['AirOff'] == [files[0]]
        assert classified['AirOn'] == files[1:]

    def test_extract_mach(self):
        assert extract_mach_from_filename(
            'run_0004_alpha_2.0_beta_0.0_mach_0.30.h5') == pytest.approx(0.30)
        assert extract_mach_from_filename(
            'run_0004_alpha_2.0_beta_0.0_mach_0.00.h5') == pytest.approx(0.0)
        assert extract_mach_from_filename('AirOn_F16_Alpha_2.0.tdms') is None


# ---------------------------------------------------------------------------
# Downstream guard: external loads must not enter the volts->forces path
# ---------------------------------------------------------------------------

def _external_raw_dict(n=N_FAST):
    raw = {name: _channel_wave(i, n)
           for i, name in enumerate(EXTERNAL_CHANNELS)}
    raw['Alpha'] = np.full(n, 2.0)
    raw['Beta'] = np.zeros(n)
    raw['Time'] = np.arange(n) * DT_FAST
    return raw


def _internal_raw_dict(n=N_FAST):
    raw = {name: _channel_wave(i, n)
           for i, name in enumerate(INTERNAL_CHANNELS)}
    raw['Alpha'] = np.full(n, 2.0)
    raw['Beta'] = np.zeros(n)
    raw['Time'] = np.arange(n) * DT_FAST
    return raw


class TestExternalBalanceGuard:

    def test_detection_structural(self):
        assert is_external_balance_data(_external_raw_dict())
        assert not is_external_balance_data(_internal_raw_dict())

    def test_detection_explicit_marker_wins(self):
        raw = _external_raw_dict()
        raw['balance_type'] = 'external'
        assert is_external_balance_data(raw)
        # Explicit internal marker overrides structure
        raw['balance_type'] = 'internal'
        assert not is_external_balance_data(raw)

    def test_calc_brf_forces_rejects_external(self):
        with pytest.raises(ValueError, match='external-balance'):
            calc_brf_forces(_external_raw_dict(), cal=None,
                            geo=Geometry())

    def test_wrf_passthrough(self):
        wrf = wrf_from_resolved_loads(_external_raw_dict())
        for i, name in enumerate(EXTERNAL_CHANNELS):
            np.testing.assert_allclose(getattr(wrf, name), _channel_wave(i))

    def test_wrf_passthrough_zero_fills_missing(self):
        wrf = wrf_from_resolved_loads({'Lift': np.ones(5)})
        np.testing.assert_allclose(wrf.Lift, np.ones(5))
        np.testing.assert_allclose(wrf.Yaw, np.zeros(5))

    def test_reduce_single_point_passthrough(self):
        """External data reduces without any balance calibration."""
        raw_on = _external_raw_dict()
        raw_off = _external_raw_dict()
        # Tunnel channels so SWT conditions can be computed
        for raw in (raw_on, raw_off):
            raw['Pdiff'] = np.full(N_FAST, 0.5)
            raw['Ptot'] = np.full(N_FAST, 2.0)
            raw['Temp'] = np.full(N_FAST, 2.0)  # ~20 degC (new cal)
        pressure_cal = {'P220': SimpleNamespace(slope=1.0),
                        'P690': SimpleNamespace(slope=7.0)}

        result = reduce_single_point(
            raw_on, raw_off, cal=None, geo=Geometry(C=2.0, S=10.0),
            pressure_cal=pressure_cal, facility='SWT')

        # Loads passed straight through (no volts->forces reduction)
        np.testing.assert_allclose(result.wrf_on.Lift, raw_on['Lift'])
        np.testing.assert_allclose(result.wrf_on.Drag, raw_on['Drag'])
        # BRF stays empty: nothing to reduce
        assert len(result.brf_on.Fx) == 0
        # Tare subtraction still applies (identical on/off -> mean removed)
        np.testing.assert_allclose(
            result.wrf_aero.Lift,
            raw_on['Lift'] - np.mean(raw_off['Lift']))
        # Coefficients come out finite
        assert np.all(np.isfinite(result.coeffs.Cl))


# ---------------------------------------------------------------------------
# Speed / velocity sweep dimension (Hz/ftps/mps/RPM/mach)
# ---------------------------------------------------------------------------

class TestExtractSpeedFromFilename:

    def test_hz_token(self):
        v, u = extract_speed_from_filename(
            'run_0001_alpha_0.0_beta_0.0_Hz_30.0.h5')
        assert v == pytest.approx(30.0)
        assert u == 'hz'

    def test_ftps_token(self):
        v, u = extract_speed_from_filename(
            'run_0002_alpha_2.0_beta_0.0_ftps_50.0.h5')
        assert v == pytest.approx(50.0)
        assert u == 'ft/s'

    def test_mps_and_rpm_tokens(self):
        v, u = extract_speed_from_filename('run_alpha_0.0_beta_0.0_mps_25.0.h5')
        assert (v, u) == (pytest.approx(25.0), 'm/s')
        v, u = extract_speed_from_filename('run_alpha_0.0_beta_0.0_RPM_1200.h5')
        assert (v, u) == (pytest.approx(1200.0), 'rpm')

    def test_legacy_mach_token(self):
        v, u = extract_speed_from_filename(
            'run_0004_alpha_2.0_beta_0.0_mach_0.30.h5')
        assert v == pytest.approx(0.30)
        assert u == 'mach'

    def test_no_token(self):
        assert extract_speed_from_filename('AirOn_F16_Alpha_2.0.tdms') == (
            None, None)


class TestSpeedOnRawData:

    def test_hz_file_exposes_speed(self, tmp_path):
        path = write_speed_h5(
            tmp_path / 'run_0007_alpha_0.0_beta_0.0_Hz_30.0.h5',
            speed_value=30.0, speed_unit='hz')
        raw, _ = read_hdf5_file(str(path))

        assert raw.properties['speed_value'] == pytest.approx(30.0)
        assert raw.properties['speed_unit'] == 'hz'
        assert 'Speed' in raw.data
        np.testing.assert_allclose(raw.data['Speed'],
                                   np.full(N_FAST, 30.0))

    def test_root_attr_wins_over_filename(self, tmp_path):
        # Root attr says 20 Hz even though filename token says 30
        path = write_speed_h5(
            tmp_path / 'run_alpha_0.0_beta_0.0_Hz_30.0.h5',
            speed_value=20.0, speed_unit='hz')
        raw, _ = read_hdf5_file(str(path))
        assert raw.properties['speed_value'] == pytest.approx(20.0)

    def test_legacy_mach_file_speed_unit_mach(self, tmp_path):
        path = write_external_h5(
            tmp_path / 'run_0004_alpha_2.0_beta_0.0_mach_0.30.h5')
        raw, _ = read_hdf5_file(str(path))
        assert raw.properties['speed_unit'] == 'mach'
        assert raw.properties['speed_value'] == pytest.approx(0.30)
        assert 'Speed' in raw.data

    def test_legacy_internal_file_degrades(self, tmp_path):
        # No speed/mach token in name, no speed attrs, no mach attr:
        # speed simply not exposed (graceful degrade).
        path = write_legacy_h5(tmp_path / 'legacy.h5')
        raw, _ = read_hdf5_file(str(path))
        assert 'speed_value' not in raw.properties
        assert 'Speed' not in raw.data


class TestSpeedClassification:

    def test_hz_sweep_distinct_conditions(self):
        files = [
            'run_0001_alpha_0.0_beta_0.0_Hz_0.0.h5',
            'run_0002_alpha_0.0_beta_0.0_Hz_10.0.h5',
            'run_0003_alpha_0.0_beta_0.0_Hz_20.0.h5',
            'run_0004_alpha_0.0_beta_0.0_Hz_30.0.h5',
        ]
        classified = classify_files_by_condition(files)

        # Speed 0 -> air off; nonzero speeds -> air on
        assert classified['AirOff'] == [files[0]]
        assert classified['AirOn'] == files[1:]

        # NOT collapsed: each nonzero speed is its own condition key
        speed_keys = [k for k in classified
                      if k not in ('AirOn', 'AirOff')]
        assert len(speed_keys) == 3
        assert classified[speed_condition_key(30.0, 'hz')] == [files[3]]
        assert classified[speed_condition_key(10.0, 'hz')] == [files[1]]

    def test_legacy_mach_classification_unchanged(self):
        files = [
            'run_0001_alpha_2.0_beta_0.0_mach_0.00.h5',
            'run_0002_alpha_2.0_beta_0.0_mach_0.30.h5',
            'run_0003_alpha_4.0_beta_0.0_mach_0.30.h5',
        ]
        classified = classify_files_by_condition(files)
        assert classified['AirOff'] == [files[0]]
        assert classified['AirOn'] == files[1:]


class TestSpeedOrdering:

    def test_sort_key_alpha_beta_speed(self):
        assert extract_sort_key_from_filename(
            'run_alpha_2.0_beta_1.0_Hz_30.0.h5') == (
            pytest.approx(2.0), pytest.approx(1.0), pytest.approx(30.0))

    def test_directory_orders_alpha_then_beta_then_speed(self):
        files = [
            'run_a_alpha_2.0_beta_0.0_Hz_30.0.h5',
            'run_b_alpha_0.0_beta_0.0_Hz_30.0.h5',
            'run_c_alpha_0.0_beta_0.0_Hz_10.0.h5',
            'run_d_alpha_0.0_beta_5.0_Hz_10.0.h5',
        ]
        ordered = sorted(files, key=extract_sort_key_from_filename)
        assert ordered == [
            'run_c_alpha_0.0_beta_0.0_Hz_10.0.h5',   # a0 b0 s10
            'run_b_alpha_0.0_beta_0.0_Hz_30.0.h5',   # a0 b0 s30
            'run_d_alpha_0.0_beta_5.0_Hz_10.0.h5',   # a0 b5 s10
            'run_a_alpha_2.0_beta_0.0_Hz_30.0.h5',   # a2 b0 s30
        ]


class TestSpeedAirStateAndConfig:
    """The GUI groups a directory by (configuration, air_state) before it
    reduces. Freestream files carry NO AirOn/AirOff token and (for a
    non-Mach sweep) NO mach token — the air state must come from the SPEED
    token, else every file lands 'Unknown', the reducer finds zero AirOn
    files, and NO configuration is detected (the bug this guards)."""

    def test_hz_air_state_from_speed_token(self):
        assert extract_air_state_from_filename(
            'run_0001_alpha_0.0_beta_0.0_Hz_0.0.h5') == 'AirOff'
        assert extract_air_state_from_filename(
            'run_0002_alpha_0.0_beta_0.0_Hz_30.0.h5') == 'AirOn'

    def test_air_state_every_speed_unit(self):
        for tag, on, off in (('ftps', '50.0', '0.0'), ('mps', '15.0', '0.0'),
                             ('RPM', '600', '0'), ('mach', '0.30', '0.00')):
            assert extract_air_state_from_filename(
                f'run_0001_alpha_0.0_beta_0.0_{tag}_{off}.h5') == 'AirOff'
            assert extract_air_state_from_filename(
                f'run_0002_alpha_0.0_beta_0.0_{tag}_{on}.h5') == 'AirOn'

    def test_explicit_airon_airoff_still_wins(self):
        assert extract_air_state_from_filename(
            'AirOff_clean_Alpha_0.0_Beta_0.0.tdms') == 'AirOff'
        assert extract_air_state_from_filename(
            'AirOn_clean_Alpha_0.0_Beta_0.0.tdms') == 'AirOn'

    def test_run_counter_not_a_configuration(self):
        # run_NNNN is a counter, not a model config — all runs group together
        assert extract_configuration_from_filename(
            'run_0007_alpha_2.0_beta_0.0_Hz_30.0.h5') == 'Unknown'

    def test_hz_sweep_groups_into_one_config_with_airon(self):
        files = [
            'run_0001_alpha_-2.0_beta_0.0_Hz_0.0.h5',
            'run_0002_alpha_-2.0_beta_0.0_Hz_20.0.h5',
            'run_0003_alpha_-2.0_beta_0.0_Hz_40.0.h5',
            'run_0004_alpha_0.0_beta_0.0_Hz_0.0.h5',
            'run_0005_alpha_0.0_beta_0.0_Hz_20.0.h5',
            'run_0006_alpha_0.0_beta_0.0_Hz_40.0.h5',
        ]
        grouped = group_files_by_configuration(files)
        # exactly one detected configuration, and it HAS air-on files
        assert list(grouped.keys()) == ['Unknown']
        states = grouped['Unknown']
        assert len(states['AirOn']) == 4
        assert len(states['AirOff']) == 2
        assert len(states['Unknown']) == 0


def _external_raw_dict_speed(value, unit='hz', setpoints=None, n=N_FAST):
    raw = _external_raw_dict(n)
    raw['Speed'] = np.full(n, value)
    raw['speed_value'] = value
    raw['speed_unit'] = unit
    if setpoints is not None:
        raw['speed_setpoints'] = np.array(setpoints, dtype=float)
    # Tunnel channels so SWT conditions can be computed
    raw['Pdiff'] = np.full(n, 0.5)
    raw['Ptot'] = np.full(n, 2.0)
    raw['Temp'] = np.full(n, 2.0)
    return raw


class TestSpeedReductionMetadata:

    def _reduce_sweep(self, speeds, setpoints=None):
        pressure_cal = {'P220': SimpleNamespace(slope=1.0),
                        'P690': SimpleNamespace(slope=7.0)}
        reduced = []
        for s in speeds:
            raw = _external_raw_dict_speed(s, setpoints=setpoints)
            rd = reduce_single_point(
                raw, raw, cal=None, geo=Geometry(C=2.0, S=10.0),
                pressure_cal=pressure_cal, facility='SWT')
            reduced.append(rd)
        return reduced

    def test_reduced_point_carries_speed(self):
        reduced = self._reduce_sweep([30.0])
        assert reduced[0].speed_value == pytest.approx(30.0)
        assert reduced[0].speed_unit == 'hz'

    def test_steady_state_lists_multiple_speeds(self):
        # Velocity sweep at fixed alpha/beta -> speed is the axis
        reduced = self._reduce_sweep([30.0, 10.0, 20.0])
        ss = reduce_steady_state(reduced)

        # Metadata surfaces the distinct swept speeds
        np.testing.assert_allclose(ss.speed_setpoints, [10.0, 20.0, 30.0])
        assert ss.speed_unit == 'hz'
        # Points are organized in ascending speed order
        np.testing.assert_allclose(np.ravel(ss.speeds), [10.0, 20.0, 30.0])
        # Speed column surfaces into the exported dataframe
        df = to_dataframe(ss)
        assert 'Speed' in df.columns

    def test_run_level_setpoints_win(self):
        # Only two air-on points reduced, but the run-level setpoints
        # record all four speeds swept (incl. the air-off 0).
        reduced = self._reduce_sweep(
            [10.0, 20.0], setpoints=[0.0, 10.0, 20.0, 30.0])
        ss = reduce_steady_state(reduced)
        np.testing.assert_allclose(ss.speed_setpoints,
                                   [0.0, 10.0, 20.0, 30.0])


# ---------------------------------------------------------------------------
# Package smoke test
# ---------------------------------------------------------------------------

def test_package_imports():
    import utils.windtunnel as wt
    assert callable(wt.read_hdf5_file)
    assert callable(wt.read_run_file)
    assert callable(wt.is_external_balance_data)
    assert callable(wt.wrf_from_resolved_loads)
    assert callable(wt.extract_speed_from_filename)
    assert callable(wt.extract_sort_key_from_filename)
    assert isinstance(RawData().balance_type, str)
    assert RawData().balance_type == 'internal'
