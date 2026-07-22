"""
Smoke tests against the two REAL Freestream-recorded datasets
(real hardware, recorded 2026-07-17, output_format 'mat'):

* freestream/runs/default_Mode1 — mode1, internal StrainBook balance
  (raw bridge VOLTS, 2000 samples @ 200 Hz) + DaqBook2005 + Positioner
  + Tunnel groups. All runs are mach 0.00 (AirOff / tare points).
* freestream/runs/default_ext — mode2, external ATE balance
  (RESOLVED wind-axis loads in N / N*m, 5000 samples) with
  positions_source='ate', balance_group='ATE_Balance',
  balance_type='external', span_config='full'.

These tests skip cleanly when the dataset directories are not present
(they live in the sibling freestream project).

Run with:
    cd Streamlined
    python -m pytest tests/test_real_datasets.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.windtunnel.data_io import (  # noqa: E402
    read_run_file, classify_files_by_condition, copy_balance_markers,
    extract_alpha_beta_from_filename,
    BALANCE_GROUP_INTERNAL, BALANCE_GROUP_EXTERNAL,
)
from utils.windtunnel.calibration import (  # noqa: E402
    read_vol_file, read_pcf_file, calc_coeffs,
)
from utils.windtunnel.transforms import Geometry  # noqa: E402
from utils.windtunnel.reduction import (  # noqa: E402
    reduce_raw, reduce_single_point, reduce_steady_state,
)
from utils.windtunnel.external_balance import (  # noqa: E402
    N_TO_LBF, NM_TO_INLB, build_load_matrix,
    calc_uncertainty_ext_balance, calc_precision_uncertainty_cases,
)

# ---------------------------------------------------------------------------
# Dataset locations (freestream is a sibling project of Streamlined)
# ---------------------------------------------------------------------------

_STREAMLINED = Path(__file__).resolve().parents[1]
_PROJECTS = _STREAMLINED.parent
MODE1_DIR = _PROJECTS / 'freestream' / 'runs' / 'default_Mode1'
EXT_DIR = _PROJECTS / 'freestream' / 'runs' / 'default_ext'

VOL_FILE = _STREAMLINED / 'CalFiles' / '2025_06_06_2 100 lb.vol'
PCF_FILE = _STREAMLINED / 'CalFiles' / 'PRESSLOPvxi18.PCF'

pytestmark = pytest.mark.skipif(
    not (MODE1_DIR.is_dir() and EXT_DIR.is_dir()),
    reason='real Freestream datasets not present')

# Reference geometry from deprecated/Process_Runs.m (MAC, S, MRC 'IPS')
GEO = dict(C=2.8600, S=18.75)

MODE1_CHANNELS = {'N1', 'N2', 'Y1', 'Y2', 'Axial', 'Roll',      # StrainBook_0
                  'Pdiff', 'Ptot', 'Temp',                       # DaqBook2005
                  'Alpha', 'Beta',                               # Positioner
                  'RPM_meas', 'RPM_cmd', 'Mach_cmd', 'Mach_meas',
                  'q_meas'}                                      # Tunnel
EXT_CHANNELS = {'Lift', 'Pitch', 'Drag', 'Side', 'Yaw', 'Roll',  # ATE_Balance
                'Pdiff', 'Ptot', 'Temp',
                'Alpha', 'Beta',
                'RPM_meas', 'RPM_cmd', 'Mach_cmd', 'Mach_meas', 'q_meas'}

MODE1_SAMPLES = 2000
EXT_SAMPLES = 5000


def _mat_files(directory):
    files = sorted(directory.glob('*.mat'))
    assert files, f'no .mat run files found in {directory}'
    return files


@pytest.fixture(scope='module')
def pressure_cal():
    if not PCF_FILE.is_file():
        pytest.skip(f'pressure calibration missing: {PCF_FILE}')
    cal = read_pcf_file(str(PCF_FILE))
    # SWT default channels (deprecated/scripts/DAQ.m set_SWT_defaults)
    assert 'P220' in cal and 'P690' in cal
    return cal


def _self_tared_entries(directory):
    """
    Build reduce_raw() input from an air-off-only directory.

    Both real datasets contain ONLY mach 0.00 (air-off / tare) points,
    so there is no air-on data to tare: each point is paired with its
    own alpha-matched tare (itself), which exercises the full tare
    subtraction path (wrf_on/wrf_off -> subtract_wrf_forces) exactly as
    an air-on point would. A full air-on reduction additionally needs
    runs recorded at mach > 0 (classified AirOn by
    classify_files_by_condition); those pair against these air-off
    points by alpha/beta in DAQ.load_data_directory and produce
    physically meaningful Q and coefficients.
    """
    entries = []
    for f in _mat_files(directory):
        raw, _ = read_run_file(str(f))
        chan = dict(raw.data)
        chan['Time'] = raw.time
        # Ride the self-describing markers (balance_type / load_units)
        # into the flat channel dicts, as the directory loaders do.
        copy_balance_markers(raw, chan)
        entries.append({'AirOn': dict(chan), 'AirOff': dict(chan)})
    return entries


# ---------------------------------------------------------------------------
# 1) Reading: every file loads with the correct structure
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def mode1_loaded():
    return [(f, *read_run_file(str(f))) for f in _mat_files(MODE1_DIR)]


@pytest.fixture(scope='module')
def ext_loaded():
    return [(f, *read_run_file(str(f))) for f in _mat_files(EXT_DIR)]


class TestMode1Files:
    """
    default_Mode1 holds two recorded sessions, distinguished by each
    file's own meta.run attrs: runs 1-6 are the mode1 sweep described
    in the run sheet (200 Hz StrainBook with Excitation, real
    positioner alpha), runs 7-12 a later session (mode='mode2' config
    at 1 kHz, no Excitation) that still used the internal StrainBook
    balance. Every file must read cleanly; the mode1-specific
    assertions key off the per-file 'mode' attr.
    """

    @pytest.fixture()
    def loaded(self, mode1_loaded):
        return mode1_loaded

    def test_channels_present_and_resampled(self, loaded):
        for f, raw, _ in loaded:
            missing = MODE1_CHANNELS - set(raw.data)
            assert not missing, f'{f.name}: missing channels {missing}'
            # Fast (StrainBook/DaqBook) base is 2000 samples; the slower
            # Positioner/Tunnel channels must be resampled onto it
            # (existing resample-to-fastest convention)
            assert len(raw.time) == MODE1_SAMPLES, f.name
            for name in MODE1_CHANNELS:
                assert len(raw.data[name]) == MODE1_SAMPLES, \
                    f'{f.name}: {name} not on the fast time base'
            if raw.properties.get('mode') == 'mode1':
                # mode1 session: 200 Hz StrainBook with Excitation
                assert raw.time[1] - raw.time[0] == pytest.approx(0.005), \
                    f.name
                assert 'Excitation' in raw.data, f.name

    def test_balance_markers_internal(self, loaded):
        for f, raw, _ in loaded:
            assert raw.balance_type == 'internal', f.name
            assert raw.balance_group == BALANCE_GROUP_INTERNAL, f.name

    def test_run_attrs_in_properties(self, loaded):
        n_mode1 = sum(1 for _, raw, _ in loaded
                      if raw.properties.get('mode') == 'mode1')
        assert n_mode1 >= 6, 'expected the 6-run mode1 sweep'
        for f, raw, _ in loaded:
            props = raw.properties
            assert props['air_state'] == 'AirOff', f.name
            assert props['mach'] == pytest.approx(0.0), f.name
            alpha_file, beta_file = extract_alpha_beta_from_filename(f.name)
            assert props['alpha'] == pytest.approx(alpha_file), f.name
            assert props['beta'] == pytest.approx(beta_file), f.name

    def test_bridge_volts_look_raw(self, loaded):
        # StrainBook channels are raw bridge VOLTS: sub-volt magnitudes
        for f, raw, _ in loaded:
            for name in ('N1', 'N2', 'Y1', 'Y2', 'Axial', 'Roll'):
                assert np.max(np.abs(raw.data[name])) < 1.0, \
                    f'{f.name}: {name} does not look like bridge volts'


class TestExtFiles:

    @pytest.fixture()
    def loaded(self, ext_loaded):
        return ext_loaded

    def test_channels_present_and_resampled(self, loaded):
        for f, raw, _ in loaded:
            missing = EXT_CHANNELS - set(raw.data)
            assert not missing, f'{f.name}: missing channels {missing}'
            assert len(raw.time) == EXT_SAMPLES, f.name
            for name in EXT_CHANNELS:
                assert len(raw.data[name]) == EXT_SAMPLES, \
                    f'{f.name}: {name} not on the fast time base'
            # No internal bridge channels in a mode-2 file
            assert 'N1' not in raw.data and 'Axial' not in raw.data, f.name

    def test_balance_markers_external(self, loaded):
        for f, raw, _ in loaded:
            assert raw.balance_type == 'external', f.name
            assert raw.balance_group == BALANCE_GROUP_EXTERNAL, f.name
            props = raw.properties
            assert props['positions_source'] == 'ate', f.name
            assert props['span_config'] == 'full', f.name
            # ATE streams N / N*m -> load_units marker drives the
            # lb / in-lb conversion in the reduction chain
            assert props['load_units'] == 'N', f.name

    def test_run_attrs_in_properties(self, loaded):
        for f, raw, _ in loaded:
            props = raw.properties
            assert props['mode'] == 'mode2', f.name
            assert props['air_state'] == 'AirOff', f.name
            alpha_file, _ = extract_alpha_beta_from_filename(f.name)
            assert props['alpha'] == pytest.approx(alpha_file), f.name

    def test_positions_nonzero(self, loaded):
        # positions_source='ate': the Positioner group carries the
        # ATE's own (nonzero) positions on the full 5000-sample base
        for f, raw, _ in loaded:
            assert np.any(raw.data['Alpha'] != 0.0), f.name
            assert np.any(raw.data['Beta'] != 0.0), f.name


# ---------------------------------------------------------------------------
# 2) Classification: all runs are mach 0.00 -> AirOff
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('directory', [MODE1_DIR, EXT_DIR],
                         ids=['mode1', 'ext'])
def test_all_runs_classify_air_off(directory):
    files = _mat_files(directory)
    classified = classify_files_by_condition(files)
    assert classified['AirOff'] == files
    assert classified['AirOn'] == []


# ---------------------------------------------------------------------------
# 3) Reduction — external (ATE) directory
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def ext_reduced(pressure_cal):
    geo = Geometry(C=GEO['C'], S=GEO['S'],
                   mshift=np.array([0.0, 0.0, 0.0]))
    entries = _self_tared_entries(EXT_DIR)
    # External data needs NO .vol balance calibration: cal=None must
    # work because the resolved loads bypass calc_brf_forces.
    red = reduce_raw(entries, cal=None, geo=geo,
                     pressure_cal=pressure_cal, facility='SWT')
    return entries, red


class TestExtReduction:

    @pytest.fixture()
    def reduced(self, ext_reduced):
        return ext_reduced

    def test_external_path_never_touches_brf(self, pressure_cal,
                                             monkeypatch):
        import utils.windtunnel.reduction as reduction_mod

        def _forbidden(*args, **kwargs):  # pragma: no cover
            raise AssertionError(
                'external-balance path must not call calc_brf_forces')

        monkeypatch.setattr(reduction_mod, 'calc_brf_forces', _forbidden)
        geo = Geometry(C=GEO['C'], S=GEO['S'])
        entries = _self_tared_entries(EXT_DIR)
        red = reduce_raw(entries, cal=None, geo=geo,
                         pressure_cal=pressure_cal, facility='SWT')
        assert len(red) == len(entries)

    def test_loads_converted_n_to_ips(self, reduced):
        # ATE streams N / N*m; the chain works in lb / in-lb
        # (deprecated/scripts/calc_coeffs.m 'External' Units field)
        entries, red = reduced
        for entry, rd in zip(entries, red):
            np.testing.assert_allclose(
                rd.wrf_on.Lift, entry['AirOn']['Lift'] * N_TO_LBF)
            np.testing.assert_allclose(
                rd.wrf_on.Drag, entry['AirOn']['Drag'] * N_TO_LBF)
            np.testing.assert_allclose(
                rd.wrf_on.Pitch, entry['AirOn']['Pitch'] * NM_TO_INLB)
            # BRF stays empty: no bridge data was reduced
            assert len(rd.brf_on.Fx) == 0

    def test_tare_subtraction_applied(self, reduced):
        # Self-tare: aero loads = on - mean(off) -> zero-mean
        _, red = reduced
        for rd in red:
            assert np.mean(rd.wrf_aero.Lift) == pytest.approx(0.0, abs=1e-12)
            assert np.mean(rd.wrf_aero.Yaw) == pytest.approx(0.0, abs=1e-12)

    def test_coefficients_and_steady_state(self, reduced):
        _, red = reduced
        assert len(red) == len(_mat_files(EXT_DIR))
        for rd in red:
            for name in ('Cl', 'Cd', 'Cs', 'CRoll', 'CPitch', 'CYaw'):
                coeff = getattr(rd.coeffs, name)
                assert len(coeff) == EXT_SAMPLES
                assert np.all(np.isfinite(coeff)), name
            assert np.all(np.isfinite(rd.tunnel.Q))
        ss = reduce_steady_state(red)
        assert np.ravel(ss.Cl).size == len(red)

    def test_uncertainty_port(self, reduced):
        # Port of calc_uncertainty_Extbalance.m + calc_uncertainty.m
        _, red = reduced
        rd = red[0]
        coeff_names = ('Cl', 'Cd', 'Cs', 'CRoll', 'CPitch', 'CYaw')
        prec = {name: calc_precision_uncertainty_cases(
                    [getattr(r.coeffs, name) for r in red])
                for name in coeff_names}
        coeffs = {name: getattr(rd.coeffs, name) for name in coeff_names}
        unc = calc_uncertainty_ext_balance(
            coeffs, build_load_matrix(rd.wrf_aero),
            np.atleast_1d(rd.alpha), rd.tunnel.Q,
            S=GEO['S'], C=GEO['C'], prec=prec)

        assert set(unc) == {'InfCoeffs', 'bias', 'total'}
        q_pos = rd.tunnel.Q > 0
        assert np.any(q_pos)
        for name in coeff_names:
            total = np.atleast_1d(unc['total'][name])
            assert total.shape == (EXT_SAMPLES,)
            # Air-off points contain genuine Q == 0 samples where the
            # MATLAB divisions produce Inf; totals must be finite
            # wherever Q > 0.
            assert np.all(np.isfinite(total[q_pos])), name
            assert 'total' in unc['bias'][name]


# ---------------------------------------------------------------------------
# 4) Reduction — internal (mode1 / StrainBook) directory
# ---------------------------------------------------------------------------

class TestMode1Reduction:

    def test_missing_vol_complains_controlled(self, pressure_cal):
        # Internal bridge-volt data without a .vol calibration must fail
        # with a clear ValueError, not an obscure AttributeError.
        entries = _self_tared_entries(MODE1_DIR)
        geo = Geometry(C=GEO['C'], S=GEO['S'])
        with pytest.raises(ValueError, match=r'\.vol'):
            reduce_single_point(entries[0]['AirOn'], entries[0]['AirOff'],
                                cal=None, geo=geo,
                                pressure_cal=pressure_cal, facility='SWT')

    def test_full_reduction_with_vol(self, pressure_cal):
        if not VOL_FILE.is_file():
            pytest.skip(f'balance calibration missing: {VOL_FILE}')
        # Linear cal + Force config, as in deprecated/Process_Runs.m
        cal = calc_coeffs(read_vol_file(str(VOL_FILE)), 'Linear')
        geo = Geometry(C=GEO['C'], S=GEO['S'],
                       mshift=np.array([0.0, 0.0, 0.0]))
        entries = _self_tared_entries(MODE1_DIR)
        red = reduce_raw(entries, cal=cal, geo=geo,
                         pressure_cal=pressure_cal, facility='SWT',
                         balance_config='Force')
        assert len(red) == len(entries)
        for rd in red:
            # volts -> balance elements -> BRF -> WRF chain ran
            assert rd.brf_on.elements.shape == (MODE1_SAMPLES, 6)
            assert np.all(np.isfinite(rd.brf_on.Fz))
            assert len(rd.wrf_on.Lift) == MODE1_SAMPLES
            for name in ('Cl', 'Cd', 'Cs', 'CRoll', 'CPitch', 'CYaw'):
                assert np.all(np.isfinite(getattr(rd.coeffs, name))), name
        ss = reduce_steady_state(red)
        assert np.ravel(ss.Cl).size == len(red)
