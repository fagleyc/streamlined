"""Metadata-first ingest tests.

Streamlined used to classify run files by REGEX OVER THE FILENAME even
though the file itself records the same facts authoritatively.  Three
things went wrong because of it:

* air state was INFERRED from speed == 0, so a tare taken with the fan
  running was silently filed as an air-on point;
* configuration always collapsed to 'Unknown' for Freestream files
  (``run_0006`` is a counter, not a config name), so two model
  configurations recorded into one folder could not be separated;
* the run number and the hysteresis leg were thrown away entirely.

These tests build synthetic .mat run files that mirror the real
Freestream layout (meta.run + meta.config_json) and pin the metadata
-> filename precedence, the manifest.json index, the config seeding and
the per-point case arrays that downstream sorting rides on.

Run with:
    cd Streamlined
    python -m pytest tests/test_metadata_ingest.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.windtunnel import data_io  # noqa: E402
from utils.windtunnel.data_io import (  # noqa: E402
    MANIFEST_FILENAME, MANIFEST_SCHEMA_VERSION,
    classify_files_by_condition, extract_configuration_from_filename,
    extract_run_number_from_filename, extract_sweep_dir_from_filename,
    find_run_files, group_files_by_configuration, parse_run_file,
    parse_tdms_filename, read_run_config, read_run_manifest,
    read_run_metadata, reference_geometry_from_config, scan_run_directory,
)

# A channel long enough that materializing it would be obvious.
SAMPLES = 20000

# The 1.0-placeholder / 0.0 reference sets a real LSWT_Test5 config
# carries (one set holds the defaults, the other holds zeros).
PLACEHOLDER_CONFIG = {
    'Sref': 0.0, 'cref': 0.0, 'bref': 0.0,
    'ref_area': 1.0, 'ref_chord': 1.0, 'ref_span': 1.0,
    'MRC_x': 0.0, 'MRC_y': 0.0, 'MRC_z': 0.0,
}

_RUN_FIELDS = ('run_number', 'config_name', 'air_state', 'alpha', 'beta',
               'mach', 'speed_value', 'speed_unit', 'sweep_dir', 'timestamp')


def write_run_mat(path, config=None, n_samples=SAMPLES, **run_attrs):
    """Write a synthetic Freestream-style .mat run file.

    Mirrors the real layout: one top-level struct per device group
    holding the channel arrays, plus a 'meta' struct carrying meta.run
    (the run attrs) and meta.config_json (the measurement config).  Only
    the run attrs actually passed are recorded, so a caller can leave
    meta.run silent about a field and exercise the filename fallback.
    """
    path = Path(path)
    run = {key: run_attrs[key] for key in _RUN_FIELDS if key in run_attrs}
    meta = {'run': run}
    if config is not None:
        meta['config_json'] = json.dumps(config)

    sio.savemat(str(path), {
        'NI_USB_6351': {'N1': np.zeros(n_samples),
                        'N2': np.zeros(n_samples)},
        'Positioner': {'Alpha': np.zeros(n_samples)},
        'Time': {'t': np.arange(n_samples) / 1000.0},
        'meta': meta,
    })
    return path


def write_manifest(directory, points, config_name='LSWT_Test5',
                   config=None, schema_version=MANIFEST_SCHEMA_VERSION):
    """Write a schema_version-1 manifest.json into a run directory."""
    manifest = {
        'schema_version': schema_version,
        'config_name': config_name,
        'output_format': 'mat',
        'created': '2026-07-24T22:08:11.123456',
        'updated': '2026-07-24T22:10:02.987654',
        'config': config if config is not None else {},
        'points': points,
    }
    path = Path(directory) / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest), encoding='utf-8')
    return path


def manifest_point(filename, run_number, alpha=0.0, beta=0.0, mach=0.0,
                   speed_value=0.0, speed_unit='hz', air_state='AirOff',
                   sweep_dir=''):
    """One manifest 'points' entry (every key the schema requires)."""
    return {
        'run_number': run_number,
        'filename': filename,
        'timestamp': '2026-07-24T22:08:11.123456',
        'alpha': alpha,
        'beta': beta,
        'mach': mach,
        'speed_value': speed_value,
        'speed_unit': speed_unit,
        'air_state': air_state,
        'sweep_dir': sweep_dir,
    }


def write_junk_run(path):
    """A file named like a run file whose contents cannot be read."""
    path = Path(path)
    path.write_bytes(b'not a MATLAB file')
    return path


class LoadmatSpy:
    """Records the kwargs every scipy.io.loadmat call is made with."""

    def __init__(self, real):
        self._real = real
        self.calls = []

    def loadmat(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self._real.loadmat(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture
def loadmat_spy(monkeypatch):
    spy = LoadmatSpy(data_io.scipy_io)
    monkeypatch.setattr(data_io, 'scipy_io', spy)
    return spy


# ---------------------------------------------------------------------------
# The metadata-only reader
# ---------------------------------------------------------------------------

class TestReadRunMetadata:

    def test_reads_run_attrs_and_config(self, tmp_path):
        path = write_run_mat(
            tmp_path / 'run_0006_alpha_2.0_beta_0.0_Hz_20.0.mat',
            run_number=6, config_name='LSWT_Test5', air_state='AirOn',
            alpha=2.0, beta=0.0, speed_value=20.0, speed_unit='hz',
            config={'Sref': 18.75, 'balance_config': 'Moment'})

        meta = read_run_metadata(str(path))

        assert meta['run_number'] == 6
        assert meta['config_name'] == 'LSWT_Test5'
        assert meta['air_state'] == 'AirOn'
        assert meta['alpha'] == pytest.approx(2.0)
        assert meta['speed_unit'] == 'hz'
        assert meta['config']['Sref'] == pytest.approx(18.75)

    def test_does_not_materialise_channel_arrays(self, tmp_path,
                                                 loadmat_spy):
        """The whole point: a directory scan must not read the channels."""
        path = write_run_mat(tmp_path / 'run_0001_alpha_0.0_beta_0.0_Hz_0.0.mat',
                             run_number=1, air_state='AirOff')

        meta = read_run_metadata(str(path))

        # Only the meta struct was requested from the file...
        assert loadmat_spy.calls, 'loadmat was never called'
        assert all(call.get('variable_names') == ['meta']
                   for call in loadmat_spy.calls)
        # ...and nothing channel-sized came back.
        assert 'NI_USB_6351' not in meta
        for value in meta.values():
            assert np.asarray(value).size < SAMPLES

    def test_memoised_per_file(self, tmp_path, loadmat_spy):
        path = write_run_mat(tmp_path / 'run_0002_alpha_0.0_beta_0.0_Hz_0.0.mat',
                             run_number=2, air_state='AirOff')

        for _ in range(4):
            read_run_metadata(str(path))

        assert len(loadmat_spy.calls) == 1

    def test_corrupt_file_returns_empty_without_raising(self, tmp_path):
        path = write_junk_run(
            tmp_path / 'run_0003_alpha_0.0_beta_0.0_Hz_0.0.mat')

        with pytest.warns(UserWarning):
            assert read_run_metadata(str(path)) == {}

    def test_missing_file_returns_empty(self, tmp_path):
        assert read_run_metadata(
            str(tmp_path / 'nope_alpha_0.0_beta_0.0_Hz_0.0.mat')) == {}

    def test_tdms_returns_empty(self, tmp_path):
        path = tmp_path / 'AirOn_F16check_Alpha_2.0_Beta_0.0.tdms'
        path.write_bytes(b'')
        assert read_run_metadata(str(path)) == {}


# ---------------------------------------------------------------------------
# parse_run_file: metadata first, filename per field
# ---------------------------------------------------------------------------

class TestParseRunFile:

    def test_recorded_air_state_beats_filename(self, tmp_path):
        """A tare taken with the fan RUNNING is the case that used to
        misclassify: the filename speed token says air-off, the file says
        otherwise, and the file wins."""
        path = write_run_mat(
            tmp_path / 'run_0011_alpha_0.0_beta_0.0_Hz_0.0.mat',
            run_number=11, air_state='AirOn', speed_value=40.0,
            speed_unit='hz')

        info = parse_run_file(str(path))

        assert info.air_state == 'AirOn'
        assert info.speed == pytest.approx(40.0)
        # The filename on its own would have said the opposite
        assert parse_tdms_filename(str(path)).air_state == 'AirOff'

    def test_fan_running_tare_stays_air_off(self, tmp_path):
        path = write_run_mat(
            tmp_path / 'run_0012_alpha_0.0_beta_0.0_Hz_40.0.mat',
            run_number=12, air_state='AirOff', speed_value=40.0,
            speed_unit='hz')

        assert parse_run_file(str(path)).air_state == 'AirOff'
        assert parse_tdms_filename(str(path)).air_state == 'AirOn'

    def test_config_name_yields_a_real_configuration(self, tmp_path):
        path = write_run_mat(
            tmp_path / 'run_0006_alpha_2.0_beta_0.0_Hz_20.0.mat',
            run_number=6, config_name='LSWT_Test5', air_state='AirOn')

        assert parse_run_file(str(path)).configuration == 'LSWT_Test5'
        # The run counter alone collapses to 'Unknown'
        assert extract_configuration_from_filename(str(path)) == 'Unknown'

    def test_run_number_and_sweep_dir_recorded(self, tmp_path):
        path = write_run_mat(
            tmp_path / 'run_0021_alpha_4.0_beta_0.0_Hz_20.0.mat',
            run_number=21, air_state='AirOn', sweep_dir='dn')

        info = parse_run_file(str(path))
        assert info.run_number == 21
        assert info.sweep_dir == 'dn'

    def test_angles_and_speed_recorded(self, tmp_path):
        """Recorded angles beat the filename token (which rounds)."""
        path = write_run_mat(
            tmp_path / 'run_0022_alpha_2.0_beta_0.0_Hz_20.0.mat',
            run_number=22, air_state='AirOn', alpha=2.03, beta=-0.02,
            speed_value=20.5, speed_unit='hz')

        info = parse_run_file(str(path))
        assert info.alpha == pytest.approx(2.03)
        assert info.beta == pytest.approx(-0.02)
        assert info.speed == pytest.approx(20.5)
        assert info.speed_unit == 'hz'

    def test_missing_metadata_falls_back_per_field(self, tmp_path):
        """meta.run records only the run number; every other field comes
        from the filename, exactly as before."""
        path = write_run_mat(
            tmp_path / 'run_0031_alpha_6.0_beta_3.0_Hz_40.0.mat',
            run_number=31)

        info = parse_run_file(str(path))
        assert info.run_number == 31
        assert info.air_state == 'AirOn'          # from the Hz token
        assert info.configuration == 'Unknown'     # from the run counter
        assert info.alpha == pytest.approx(6.0)
        assert info.beta == pytest.approx(3.0)
        assert info.speed == pytest.approx(40.0)
        assert info.sweep_dir == ''

    def test_corrupt_file_falls_back_to_filename(self, tmp_path):
        path = write_junk_run(
            tmp_path / 'run_0032_alpha_2.0_beta_0.0_Hz_20.0.mat')

        with pytest.warns(UserWarning):
            info = parse_run_file(str(path))

        assert info.air_state == 'AirOn'
        assert info.alpha == pytest.approx(2.0)
        assert info.run_number == 32

    def test_legacy_tdms_name_unchanged(self):
        """No file on disk at all: pure filename parsing, as before."""
        info = parse_run_file(
            'AirOff_F16check_no_beta_Alpha_-2.0_Beta_0.0.tdms')
        assert info.configuration == 'F16check_no_beta'
        assert info.air_state == 'AirOff'
        assert info.alpha == pytest.approx(-2.0)
        assert info.run_number is None
        assert info.sweep_dir == ''

    def test_unrecognized_air_state_falls_back(self, tmp_path):
        """An air state nobody understands must not become a third
        grouping key."""
        path = write_run_mat(
            tmp_path / 'run_0033_alpha_0.0_beta_0.0_Hz_20.0.mat',
            run_number=33, air_state='whoKnows')

        assert parse_run_file(str(path)).air_state == 'AirOn'


class TestFilenameExtractors:
    """The legacy extractors stay working — they are the fallback."""

    def test_run_number_token(self):
        assert extract_run_number_from_filename(
            'run_0006_alpha_2.0_beta_0.0_Hz_20.0.mat') == 6
        assert extract_run_number_from_filename(
            'AirOn_F16_Alpha_2.0.tdms') is None

    def test_sweep_dir_token(self):
        assert extract_sweep_dir_from_filename(
            'run_0009_alpha_2.0_beta_0.0_Hz_20.0_dn.mat') == 'dn'
        assert extract_sweep_dir_from_filename(
            'run_0009_alpha_2.0_beta_0.0_Hz_20.0_up.mat') == 'up'
        assert extract_sweep_dir_from_filename(
            'run_0009_alpha_2.0_beta_0.0_Hz_20.0.mat') == ''


# ---------------------------------------------------------------------------
# Classification and grouping now ride on the recorded metadata
# ---------------------------------------------------------------------------

class TestClassificationUsesMetadata:

    def test_fan_running_tare_classifies_air_off(self, tmp_path):
        tare = write_run_mat(
            tmp_path / 'run_0001_alpha_0.0_beta_0.0_Hz_40.0.mat',
            run_number=1, air_state='AirOff', speed_value=40.0,
            speed_unit='hz')
        point = write_run_mat(
            tmp_path / 'run_0002_alpha_0.0_beta_0.0_Hz_40.0.mat',
            run_number=2, air_state='AirOn', speed_value=40.0,
            speed_unit='hz')

        classified = classify_files_by_condition([tare, point])

        assert classified['AirOff'] == [tare]
        assert classified['AirOn'] == [point]

    def test_speed_conditions_still_split(self, tmp_path):
        files = [
            write_run_mat(tmp_path / f'run_000{i}_alpha_0.0_beta_0.0'
                                     f'_Hz_{speed}.mat',
                          run_number=i, air_state='AirOn',
                          speed_value=speed, speed_unit='hz')
            for i, speed in enumerate((20.0, 40.0), start=1)]

        classified = classify_files_by_condition(files)

        assert classified['AirOn'] == files
        assert classified[data_io.speed_condition_key(20.0, 'hz')] == [files[0]]
        assert classified[data_io.speed_condition_key(40.0, 'hz')] == [files[1]]


class TestGroupingSeparatesConfigurations:

    def test_two_configurations_in_one_directory(self, tmp_path):
        """The bug: two model configurations recorded into one folder
        collapsed into a single 'Unknown' group and could not be
        separated.  meta.run.config_name separates them."""
        for run_number, config_name in ((1, 'ClipDelta'), (2, 'ClipDelta'),
                                        (3, 'ClipDeltaLEX'),
                                        (4, 'ClipDeltaLEX')):
            write_run_mat(
                tmp_path / f'run_000{run_number}_alpha_2.0_beta_0.0'
                           f'_Hz_20.0.mat',
                run_number=run_number, config_name=config_name,
                air_state='AirOn', alpha=2.0, beta=0.0,
                speed_value=20.0, speed_unit='hz')

        grouped = group_files_by_configuration(find_run_files(tmp_path))

        assert sorted(grouped) == ['ClipDelta', 'ClipDeltaLEX']
        assert len(grouped['ClipDelta']['AirOn']) == 2
        assert len(grouped['ClipDeltaLEX']['AirOn']) == 2

    def test_accepts_preparsed_file_infos(self, tmp_path):
        write_run_mat(tmp_path / 'run_0001_alpha_0.0_beta_0.0_Hz_20.0.mat',
                      run_number=1, config_name='CfgA', air_state='AirOn')
        infos, _ = scan_run_directory(tmp_path)

        grouped = group_files_by_configuration(infos)

        assert list(grouped) == ['CfgA']


# ---------------------------------------------------------------------------
# manifest.json
# ---------------------------------------------------------------------------

def _three_runs(directory):
    """Three run files whose meta.run records only the run number."""
    names = []
    for run_number, alpha in ((1, 0.0), (2, 2.0), (3, 4.0)):
        name = f'run_000{run_number}_alpha_{alpha}_beta_0.0_Hz_0.0.mat'
        write_run_mat(Path(directory) / name, run_number=run_number)
        names.append(name)
    return names


class TestManifest:

    def test_honoured_and_fixes_acquisition_order(self, tmp_path):
        names = _three_runs(tmp_path)
        # Acquisition order deliberately NOT the sorted path order
        write_manifest(tmp_path, [
            manifest_point(names[2], 3, alpha=4.0),
            manifest_point(names[0], 1, alpha=0.0),
            manifest_point(names[1], 2, alpha=2.0),
        ])

        infos, mismatches = scan_run_directory(tmp_path)

        assert mismatches == []
        assert [i.filepath.name for i in infos] == [names[2], names[0],
                                                    names[1]]
        assert [i.run_number for i in infos] == [3, 1, 2]

    def test_fills_fields_the_file_does_not_carry(self, tmp_path):
        names = _three_runs(tmp_path)
        write_manifest(tmp_path, [
            manifest_point(names[0], 1, alpha=0.0, air_state='AirOn',
                           speed_value=40.0, speed_unit='hz',
                           sweep_dir='dn'),
        ] + [manifest_point(n, i) for i, n in enumerate(names[1:], start=2)],
            config_name='LSWT_Test5')

        infos, _ = scan_run_directory(tmp_path)
        first = infos[0]

        # The filename says Hz_0.0 -> AirOff; the manifest knows better,
        # and the file itself records neither.
        assert first.air_state == 'AirOn'
        assert first.speed == pytest.approx(40.0)
        assert first.sweep_dir == 'dn'
        assert first.configuration == 'LSWT_Test5'

    def test_file_metadata_still_beats_the_manifest(self, tmp_path):
        name = 'run_0001_alpha_0.0_beta_0.0_Hz_0.0.mat'
        write_run_mat(tmp_path / name, run_number=1, config_name='CfgA',
                      air_state='AirOff', alpha=0.0)
        write_manifest(tmp_path, [manifest_point(name, 1, alpha=9.0,
                                                 air_state='AirOn')],
                       config_name='CfgB')

        infos, _ = scan_run_directory(tmp_path)

        assert infos[0].air_state == 'AirOff'
        assert infos[0].configuration == 'CfgA'
        assert infos[0].alpha == pytest.approx(0.0)

    def test_count_mismatch_is_reported(self, tmp_path):
        names = _three_runs(tmp_path)
        write_manifest(tmp_path, [manifest_point(names[0], 1),
                                  manifest_point(names[1], 2)])

        with pytest.warns(UserWarning):
            infos, mismatches = scan_run_directory(tmp_path)

        assert any('lists 2 point(s) but 3 run file(s)' in m
                   for m in mismatches)
        assert any('not listed in' in m for m in mismatches)
        # Nothing is dropped: the unlisted file is still indexed
        assert len(infos) == 3

    def test_manifest_point_without_a_file_is_reported(self, tmp_path):
        names = _three_runs(tmp_path)
        write_manifest(tmp_path, [manifest_point(n, i) for i, n
                                  in enumerate(names, start=1)]
                       + [manifest_point('run_0009_alpha_8.0_beta_0.0'
                                         '_Hz_0.0.mat', 9)])

        with pytest.warns(UserWarning):
            infos, mismatches = scan_run_directory(tmp_path)

        assert any('no run file on disk' in m for m in mismatches)
        assert len(infos) == 3

    def test_missing_manifest_is_not_an_error(self, tmp_path):
        names = _three_runs(tmp_path)

        infos, mismatches = scan_run_directory(tmp_path)

        assert mismatches == []
        assert sorted(i.filepath.name for i in infos) == sorted(names)
        assert read_run_manifest(tmp_path) == {}

    def test_malformed_manifest_falls_back(self, tmp_path):
        names = _three_runs(tmp_path)
        (tmp_path / MANIFEST_FILENAME).write_text('{not json',
                                                  encoding='utf-8')

        with pytest.warns(UserWarning):
            infos, mismatches = scan_run_directory(tmp_path)

        assert mismatches == []
        assert len(infos) == len(names)

    def test_wrong_schema_version_falls_back(self, tmp_path):
        names = _three_runs(tmp_path)
        write_manifest(tmp_path, [manifest_point(names[0], 1)],
                       schema_version=99)

        with pytest.warns(UserWarning):
            assert read_run_manifest(tmp_path) == {}

    def test_manifest_is_not_a_run_file(self, tmp_path):
        _three_runs(tmp_path)
        write_manifest(tmp_path, [])

        found = find_run_files(tmp_path)

        assert MANIFEST_FILENAME not in [f.name for f in found]
        assert len(found) == 3


# ---------------------------------------------------------------------------
# Seeding the measurement config from the file
# ---------------------------------------------------------------------------

class TestReferenceGeometryFromConfig:

    def test_prefers_the_non_default_reference_set(self):
        seed = reference_geometry_from_config(
            {'Sref': 18.75, 'cref': 2.86, 'bref': 9.0,
             'ref_area': 1.0, 'ref_chord': 1.0, 'ref_span': 1.0})

        assert seed['ref_area'] == pytest.approx(18.75)
        assert seed['mac'] == pytest.approx(2.86)
        assert seed['span'] == pytest.approx(9.0)

    def test_takes_the_other_set_when_it_holds_the_real_values(self):
        seed = reference_geometry_from_config(
            {'Sref': 0.0, 'cref': 0.0, 'bref': 0.0,
             'ref_area': 18.75, 'ref_chord': 2.86, 'ref_span': 9.0})

        assert seed['ref_area'] == pytest.approx(18.75)
        assert seed['mac'] == pytest.approx(2.86)
        assert seed['span'] == pytest.approx(9.0)

    def test_placeholder_config_seeds_no_geometry(self):
        """Both sets are absent/zero/1.0-default: seed NOTHING rather than
        pass a placeholder off as a reference dimension."""
        seed = reference_geometry_from_config(PLACEHOLDER_CONFIG)

        assert 'ref_area' not in seed
        assert 'mac' not in seed
        assert 'span' not in seed
        assert 'mrc' not in seed

    def test_mrc_only_when_nonzero(self):
        assert 'mrc' not in reference_geometry_from_config(
            {'MRC_x': 0.0, 'MRC_y': 0.0, 'MRC_z': 0.0})
        seed = reference_geometry_from_config(
            {'MRC_x': 1.5, 'MRC_y': 0.0, 'MRC_z': -0.25})
        assert seed['mrc'] == pytest.approx([1.5, 0.0, -0.25])

    def test_balance_config(self):
        assert reference_geometry_from_config(
            {'balance_config': 'Moment'})['balance_config'] == 'Moment'
        assert reference_geometry_from_config(
            {'balance_config': 'force'})['balance_config'] == 'Force'
        assert 'balance_config' not in reference_geometry_from_config(
            {'balance_config': ''})

    def test_junk_config_seeds_nothing(self):
        assert reference_geometry_from_config(None) == {}
        assert reference_geometry_from_config({}) == {}


class TestReadRunConfig:

    def test_from_manifest(self, tmp_path):
        _three_runs(tmp_path)
        write_manifest(tmp_path, [], config={'Sref': 18.75})

        assert read_run_config(tmp_path)['Sref'] == pytest.approx(18.75)

    def test_from_the_first_run_file(self, tmp_path):
        write_run_mat(tmp_path / 'run_0001_alpha_0.0_beta_0.0_Hz_0.0.mat',
                      run_number=1, config={'Sref': 18.75})

        assert read_run_config(tmp_path)['Sref'] == pytest.approx(18.75)

    def test_legacy_directory_records_none(self, tmp_path):
        _three_runs(tmp_path)
        assert read_run_config(tmp_path) == {}


class TestControllerSeeding:
    """Seeding is an INITIAL value: it fills the untouched defaults and
    leaves anything the operator set alone."""

    @staticmethod
    def _controller():
        pytest.importorskip('PyQt6')
        from utils.gui.controllers.data_controller import DataController
        from utils.gui.models.data_model import DataModel
        return DataController(DataModel(), None)

    def test_seeds_untouched_defaults(self, tmp_path):
        write_run_mat(tmp_path / 'run_0001_alpha_0.0_beta_0.0_Hz_0.0.mat',
                      run_number=1,
                      config={'Sref': 18.75, 'cref': 2.86, 'bref': 9.0,
                              'MRC_x': 1.5, 'MRC_y': 0.0, 'MRC_z': 0.0,
                              'balance_config': 'Moment'})
        controller = self._controller()

        controller._seed_config_from_run_files([str(tmp_path)])
        geo = controller.model.get_geometry(controller.model.default_geometry)

        assert geo['ref_area'] == pytest.approx(18.75)
        assert geo['mac'] == pytest.approx(2.86)
        assert geo['span'] == pytest.approx(9.0)
        assert geo['mrc'] == pytest.approx([1.5, 0.0, 0.0])
        assert controller.model.balance_config == 'Moment'
        assert controller.last_run_config['Sref'] == pytest.approx(18.75)

    def test_never_overwrites_an_operator_value(self, tmp_path):
        write_run_mat(tmp_path / 'run_0001_alpha_0.0_beta_0.0_Hz_0.0.mat',
                      run_number=1,
                      config={'Sref': 18.75, 'cref': 2.86, 'bref': 9.0})
        controller = self._controller()
        controller.model.set_geometry(mac=3.5, ref_area=25.0,
                                      mrc=[0.0, 0.0, 0.0], span=1.0)

        controller._seed_config_from_run_files([str(tmp_path)])
        geo = controller.model.get_geometry(controller.model.default_geometry)

        assert geo['mac'] == pytest.approx(3.5)
        assert geo['ref_area'] == pytest.approx(25.0)
        # span was still at its untouched 1.0 default, so it is seeded
        assert geo['span'] == pytest.approx(9.0)

    def test_placeholder_config_leaves_geometry_alone(self, tmp_path):
        write_run_mat(tmp_path / 'run_0001_alpha_0.0_beta_0.0_Hz_0.0.mat',
                      run_number=1, config=dict(PLACEHOLDER_CONFIG))
        controller = self._controller()

        controller._seed_config_from_run_files([str(tmp_path)])
        geo = controller.model.get_geometry(controller.model.default_geometry)

        assert geo['mac'] == pytest.approx(1.0)
        assert geo['ref_area'] == pytest.approx(1.0)
        assert geo['span'] == pytest.approx(1.0)
        assert geo['mrc'] == pytest.approx([0.0, 0.0, 0.0])

    def test_seeds_once_per_session(self, tmp_path):
        write_run_mat(tmp_path / 'run_0001_alpha_0.0_beta_0.0_Hz_0.0.mat',
                      run_number=1, config={'Sref': 18.75})
        controller = self._controller()

        controller._seed_config_from_run_files([str(tmp_path)])
        controller.model.ref_area = 25.0
        controller._seed_config_from_run_files([str(tmp_path)])

        assert controller.model.ref_area == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# The per-point case arrays other views sort on
# ---------------------------------------------------------------------------

class FakeSteadyState:
    """Stand-in for SteadyStateData carrying just the sort permutation."""

    def __init__(self, indices):
        self.indices = np.asarray(indices)


def _case_and_infos(shape=(2, 2), sweep_dirs=('', 'up', 'dn', 'up')):
    from utils.gui.models.case import TestCase
    from utils.windtunnel.data_io import FileInfo

    case = TestCase(name='case')
    case.alphas = np.zeros(shape)
    infos = [
        FileInfo(filepath=Path(f'run_000{i + 1}.mat'), configuration='CfgA',
                 air_state='AirOn', alpha=0.0, beta=0.0,
                 run_number=i + 1, sweep_dir=sweep_dirs[i])
        for i in range(int(np.prod(shape)))]
    return case, infos


class TestCasePointMetadata:

    @staticmethod
    def _attach(case, infos, ss):
        pytest.importorskip('PyQt6')
        from utils.gui.controllers.data_controller import ProcessingWorker
        ProcessingWorker._attach_point_metadata(case, infos, ss)

    def test_defaults_are_empty(self):
        from utils.gui.models.case import TestCase
        case = TestCase(name='empty')
        assert case.run_numbers.size == 0
        assert case.sweep_dirs.size == 0

    def test_ordered_by_the_reduction_permutation(self):
        case, infos = _case_and_infos()
        self._attach(case, infos, FakeSteadyState([2, 0, 3, 1]))

        assert case.run_numbers.shape == case.alphas.shape
        assert case.sweep_dirs.shape == case.alphas.shape
        np.testing.assert_allclose(case.run_numbers.flatten(),
                                   [3.0, 1.0, 4.0, 2.0])
        assert list(case.sweep_dirs.flatten()) == ['dn', '', 'up', 'up']

    def test_point_count_mismatch_leaves_arrays_empty(self):
        """Misaligned per-point metadata is worse than none: a consumer
        would silently sort on the wrong points."""
        case, infos = _case_and_infos()
        self._attach(case, infos[:3], FakeSteadyState([2, 0, 1]))

        assert case.run_numbers.size == 0
        assert case.sweep_dirs.size == 0

    def test_no_recorded_values_leaves_arrays_empty(self):
        case, infos = _case_and_infos(sweep_dirs=('', '', '', ''))
        for info in infos:
            info.run_number = None
        self._attach(case, infos, FakeSteadyState([0, 1, 2, 3]))

        assert case.run_numbers.size == 0
        assert case.sweep_dirs.size == 0
