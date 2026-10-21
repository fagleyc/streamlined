"""File > Export writes every quantity in ONE unit system, picked per file.

The exports used to carry whatever unit each quantity happened to be held
in: the .mat file had Q in psi beside P_tot and P_static in Pa, U_inf in
m/s, rho in kg/m^3, T in degC, loads in lbf / lb-in and the geometry as
typed, with nothing saying which was which. The Excel sheet converted its
table but not its geometry, and had no P_static at all.

Now the export dialog offers Metric (MKS, the default) or English (IPS),
every dimensional series is converted to it, each field's unit rides along
(case.Units.<group>.<field> in a .mat, a 'units' attribute in HDF5, the
column header in Excel / CSV), and raw channels stay as recorded.
"""
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
pd = pytest.importorskip("pandas")

from utils.windtunnel.units import (  # noqa: E402
    PA_PER_PSI, UnitSystem, convert_quantity, geometry_to_ips,
    quantity_label)

from test_channel_cal_export import _panel, qapp  # noqa: E402,F401

IN = 0.0254
SQ_IN = IN ** 2
LBF = 4.4482216152605
SLUG_FT3_PER_KG_M3 = 0.00194032033


def _export(tmp_path, fmt, units=None, ext=None):
    panel, case = _panel(tmp_path)
    # the geometry the run was reduced with, as the main window holds it
    panel.model.set_geometry(4.0, 96.0, [0.0, 0.0, 0.0], 'IPS', span=24.0)
    path = tmp_path / f"out.{ext or fmt}"
    config = {'format': fmt, 'case_scope': 'all', 'filepath': str(path),
              'include_unsteady': False}
    if units:
        config['units'] = units
    panel._write_export(str(path), fmt, config)
    return path, panel, case


def _mat_case(path):
    return scipy_io.loadmat(str(path), struct_as_record=False,
                            squeeze_me=True)['case_001']


def _first(x):
    return float(np.ravel(x)[0])


class TestTheUnitsModule:
    def test_pa_and_psi_pressures_land_on_the_same_scale(self):
        psi = 1.25
        for system in (UnitSystem.MKS, UnitSystem.IPS):
            a = convert_quantity(psi, 'pressure_psi', system)
            b = convert_quantity(psi * PA_PER_PSI, 'pressure_pa', system)
            assert a == pytest.approx(b, rel=1e-12)

    def test_metric_pressure_is_the_reductions_own_factor(self):
        assert convert_quantity(1.0, 'pressure_psi', UnitSystem.MKS) == \
            pytest.approx(PA_PER_PSI, rel=1e-12)

    def test_geometry_comes_out_of_its_input_units(self):
        assert geometry_to_ips(1.0, 'length', 'FPS') == pytest.approx(12.0)
        assert geometry_to_ips(1.0, 'area', 'FPS') == pytest.approx(144.0)
        assert geometry_to_ips(0.0254, 'length', 'MKS') == pytest.approx(1.0)
        assert geometry_to_ips(5.0, 'length', 'IPS') == 5.0

    def test_labels(self):
        assert quantity_label('pressure_pa', UnitSystem.IPS) == 'psi'
        assert quantity_label('moment', UnitSystem.MKS) == 'N-m'
        assert quantity_label('angle', UnitSystem.MKS) == 'deg'


class TestTheDialog:
    def test_metric_is_the_default_and_rides_in_the_config(self, qapp):
        from utils.gui.widgets.export_dialog import ExportDialog
        dlg = ExportDialog(case_names=['a'])
        assert dlg.get_export_config()['units'] == 'MKS'
        dlg.cmb_units.setCurrentIndex(dlg.cmb_units.findData('IPS'))
        assert dlg.get_export_config()['units'] == 'IPS'

    def test_the_choice_does_not_apply_to_coe(self, qapp):
        from utils.gui.widgets.export_dialog import ExportDialog
        dlg = ExportDialog(case_names=['a'])
        dlg.cmb_format.setCurrentIndex(dlg.cmb_format.findData('coe'))
        assert not dlg.cmb_units.isEnabled()


class TestTheExcelSheet:
    def _sheet(self, tmp_path, units=None):
        path, panel, case = _export(tmp_path, 'excel', units, ext='xlsx')
        raw = pd.read_excel(path, header=None)
        # the table starts after the header block and one blank row
        start = raw.index[raw[0] == 'Alpha [deg]'][0]
        table = pd.read_excel(path, header=start)
        header = dict(zip(raw[0][:start].astype(str), raw[1][:start]))
        return table, header, case

    def test_static_pressure_is_there(self, qapp, tmp_path):
        table, header, case = self._sheet(tmp_path)
        assert 'P_static [Pa]' in table.columns
        assert any(k.startswith('Static Pressure (P_static)')
                   for k in header)
        expected = np.sort(case.static_pressures) * PA_PER_PSI
        assert np.sort(table['P_static [Pa]'].to_numpy()) == \
            pytest.approx(expected, rel=1e-9)

    def test_metric_by_default(self, qapp, tmp_path):
        table, header, case = self._sheet(tmp_path)
        for col in ('Lift [N]', 'M_Pitch [N-m]', 'Q [Pa]', 'U_inf [m/s]',
                    'rho [kg/m^3]', 'T [degC]', 'P_tot [Pa]'):
            assert col in table.columns, col
        assert header['Output Units'] == 'MKS'
        assert float(header['MAC [m]']) == pytest.approx(4.0 * IN, rel=1e-5)
        assert float(header['Ref Area [m^2]']) == \
            pytest.approx(96.0 * SQ_IN, rel=1e-5)

    def test_english_on_request(self, qapp, tmp_path):
        table, header, case = self._sheet(tmp_path, 'IPS')
        for col in ('Lift [lbf]', 'Q [psi]', 'P_static [psi]',
                    'U_inf [ft/s]', 'T [degF]'):
            assert col in table.columns, col
        assert float(header['MAC [in]']) == pytest.approx(4.0)
        assert np.sort(table['P_static [psi]'].to_numpy()) == \
            pytest.approx(np.sort(case.static_pressures), rel=1e-9)

    def test_external_moment_elements_convert_as_moments(self, qapp,
                                                         tmp_path):
        metric, _, case = self._sheet(tmp_path)
        english, _, _ = self._sheet(tmp_path / 'e', 'IPS')
        # the test balance is the external ATE: slots 4-6 are Mx..Mz
        assert 'Mx [N-m]' in metric.columns
        ratio = metric['Mx [N-m]'].to_numpy() / english['Mx [lb-in]'].to_numpy()
        assert np.allclose(ratio[np.isfinite(ratio)], LBF * IN, rtol=1e-9)


class TestTheMatFile:
    def test_metric_by_default_and_self_describing(self, qapp, tmp_path):
        path, _, _ = _export(tmp_path, 'mat')
        case = _mat_case(path)
        assert case.Units.system == 'MKS'
        tc, tu = case.Tunnel_Conditions, case.Units.Tunnel_Conditions
        assert tu.Q == 'Pa' and tu.P_static == 'Pa' and tu.P_tot == 'Pa'
        assert tu.U_inf == 'm/s' and tu.rho == 'kg/m^3' and tu.T0 == 'degC'
        # Q is in Pa now, so Q_mks would only repeat it
        assert 'Q_mks' not in tc._fieldnames
        assert _first(tc.Q) == pytest.approx(_first(tc.q_inf))
        assert case.meta.units == 'MKS'

    def test_every_pressure_is_on_one_scale(self, qapp, tmp_path):
        metric = _mat_case(_export(tmp_path, 'mat')[0]).Tunnel_Conditions
        english = _mat_case(
            _export(tmp_path / 'e', 'mat', 'IPS')[0]).Tunnel_Conditions
        for name in ('Q', 'P_tot', 'P_static'):
            assert _first(metric.__dict__[name]) == pytest.approx(
                _first(english.__dict__[name]) * PA_PER_PSI, rel=1e-9), name
        # static sits below total
        assert _first(english.P_static) < _first(english.P_tot)

    def test_english_conversions(self, qapp, tmp_path):
        metric = _mat_case(_export(tmp_path, 'mat')[0])
        english = _mat_case(_export(tmp_path / 'e', 'mat', 'IPS')[0])
        m, e = metric.Tunnel_Conditions, english.Tunnel_Conditions
        assert _first(e.U_inf) == pytest.approx(_first(m.U_inf) / 0.3048)
        assert _first(e.rho) == pytest.approx(
            _first(m.rho) * SLUG_FT3_PER_KG_M3)
        assert _first(e.T0) == pytest.approx(_first(m.T0) * 1.8 + 32.0)
        assert english.Units.Tunnel_Conditions.U_inf == 'ft/s'

    def test_geometry_is_converted(self, qapp, tmp_path):
        metric = _mat_case(_export(tmp_path, 'mat')[0])
        assert _first(metric.Geometry.MAC) == pytest.approx(4.0 * IN)
        assert _first(metric.Geometry.ref_area) == pytest.approx(96.0 * SQ_IN)
        assert metric.Units.Geometry.MAC == 'm'
        assert metric.meta.geometry.mac == pytest.approx(4.0 * IN)
        assert metric.meta.geometry.output_units == 'MKS'

    def test_raw_channels_stay_as_recorded(self, qapp, tmp_path):
        metric = _mat_case(_export(tmp_path, 'mat')[0])
        english = _mat_case(_export(tmp_path / 'e', 'mat', 'IPS')[0])
        assert _first(metric.Raw.Pdiff) == pytest.approx(
            _first(english.Raw.Pdiff))
        assert metric.Units.Raw.Pdiff == 'V'
        assert metric.Units.Raw.Alpha == 'deg'

    def test_a_raw_channel_named_like_a_load_is_not_converted(self, qapp,
                                                              tmp_path):
        """The external ATE records Fx..Mz as volts; that is not a force."""
        metric = _mat_case(_export(tmp_path, 'mat')[0])
        assert _first(metric.BRF_Forces.Fx) == pytest.approx(
            _first(metric.Raw.Fx))
        assert metric.Units.BRF_Forces.Fx == 'V'

    def test_wind_axis_loads_are_converted(self, qapp, tmp_path):
        metric = _mat_case(_export(tmp_path, 'mat')[0])
        english = _mat_case(_export(tmp_path / 'e', 'mat', 'IPS')[0])
        assert _first(metric.WRF_Forces.Lift) == pytest.approx(
            _first(english.WRF_Forces.Lift) * LBF)
        assert metric.Units.WRF_Forces.Pitch_W == 'N-m'


class TestHdf5AndCsv:
    def test_hdf5_datasets_carry_their_units(self, qapp, tmp_path):
        h5py = pytest.importorskip("h5py")
        path, _, _ = _export(tmp_path, 'hdf5', ext='h5')
        with h5py.File(path, 'r') as hf:
            assert hf.attrs['output_units'] == 'MKS'
            grp = hf[next(k for k in hf if k not in ('calibration',
                                                      'geometry'))]
            assert grp['Tunnel_Conditions/P_static'].attrs['units'] == 'Pa'
            assert hf['geometry'].attrs['length_unit'] == 'm'
            assert 'Units' not in grp

    def test_csv_is_in_the_export_units(self, qapp, tmp_path):
        path, _, case = _export(tmp_path, 'csv', 'IPS')
        df = pd.read_csv(path)
        assert 'P_static [psi]' in df.columns and 'Lift [lbf]' in df.columns
        assert len(df) == case.n_points
