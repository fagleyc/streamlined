"""
Synthetic unit tests for utils.windtunnel.external_balance — the port
of the MATLAB external-balance chain (deprecated/scripts/calc_coeffs.m,
calc_uncertainty_Extbalance.m, calc_uncertainty.m, plus the
DPM_calc_BRF_forces.m MRC-shift convention).

Run with:
    cd Streamlined
    python -m pytest tests/test_external_balance.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.windtunnel.transforms import WRFForces  # noqa: E402
from utils.windtunnel.external_balance import (  # noqa: E402
    EXTERNAL_CHANNEL_ORDER, EXTERNAL_CAL_BIAS, EXTERNAL_CAL_UNITS,
    N_TO_LBF, NM_TO_INLB,
    external_loads_in_si, external_loads_to_ips,
    transfer_external_loads_to_mrc, build_load_matrix,
    calc_precision_uncertainty_cases, calc_uncertainty_ext_balance,
)


def _raw(n=8, load_units=None):
    raw = {name: np.full(n, float(i + 1))
           for i, name in enumerate(('Lift', 'Drag', 'Side',
                                     'Roll', 'Pitch', 'Yaw'))}
    if load_units is not None:
        raw['load_units'] = load_units
    return raw


class TestUnitConversion:

    def test_calibration_constants_from_matlab(self):
        # calc_coeffs.m 'External': Forcechan / Bias / Units
        assert EXTERNAL_CHANNEL_ORDER == ('Drag', 'Side', 'Lift',
                                          'Roll', 'Pitch', 'Yaw')
        np.testing.assert_allclose(
            EXTERNAL_CAL_BIAS,
            [0.0164, 0.0368, 0.0238, 0.0201, 0.0158, 0.0081])
        assert EXTERNAL_CAL_UNITS == ('lb', 'lb', 'lb',
                                      'in-lb', 'in-lb', 'in-lb')

    def test_marker_detection(self):
        assert not external_loads_in_si(_raw())
        assert not external_loads_in_si(_raw(load_units='lb'))
        assert external_loads_in_si(_raw(load_units='N'))
        assert external_loads_in_si(_raw(load_units='si'))

    def test_unmarked_dict_passes_through_unchanged(self):
        raw = _raw()
        out = external_loads_to_ips(raw)
        assert out is raw  # legacy lb / in-lb behavior: no copy, no math

    def test_si_marked_dict_converted(self):
        raw = _raw(load_units='N')
        out = external_loads_to_ips(raw)
        assert out is not raw
        np.testing.assert_allclose(out['Lift'], raw['Lift'] * N_TO_LBF)
        np.testing.assert_allclose(out['Drag'], raw['Drag'] * N_TO_LBF)
        np.testing.assert_allclose(out['Side'], raw['Side'] * N_TO_LBF)
        np.testing.assert_allclose(out['Roll'], raw['Roll'] * NM_TO_INLB)
        np.testing.assert_allclose(out['Pitch'], raw['Pitch'] * NM_TO_INLB)
        np.testing.assert_allclose(out['Yaw'], raw['Yaw'] * NM_TO_INLB)
        assert out['load_units'] == 'lb'
        # source dict untouched
        np.testing.assert_allclose(raw['Lift'], np.full(8, 1.0))


class TestMrcTransfer:

    def _wrf(self, n=4):
        wrf = WRFForces()
        for i, name in enumerate(('Lift', 'Drag', 'Side',
                                  'Roll', 'Pitch', 'Yaw')):
            setattr(wrf, name, np.full(n, float(i + 1)))
        return wrf

    def test_zero_shift_is_noop(self):
        # MATLAB never re-references the external ATE loads
        # (DPM_calc_BRF_forces.m applies mshift on the internal path
        # only), so a zero shift must pass the loads through untouched.
        wrf = self._wrf()
        out = transfer_external_loads_to_mrc(
            wrf, np.zeros(4), np.zeros(4), [0.0, 0.0, 0.0])
        assert out is wrf

    def test_shift_at_zero_attitude(self):
        # At alpha = beta = 0: Fx = Drag, Fy = Side, Fz = Lift, so the
        # DPM_calc_BRF_forces.m arm terms become:
        #   Roll  -> Roll  - Side*mz
        #   Pitch -> Pitch - Lift*mx - Drag*mz
        #   Yaw   -> Yaw   + Side*my - Side*mx
        wrf = self._wrf()
        mx, my, mz = 1.6, 0.25, -0.5
        out = transfer_external_loads_to_mrc(
            wrf, np.zeros(4), np.zeros(4), [mx, my, mz])
        np.testing.assert_allclose(out.Lift, wrf.Lift)
        np.testing.assert_allclose(out.Drag, wrf.Drag)
        np.testing.assert_allclose(out.Side, wrf.Side)
        np.testing.assert_allclose(out.Roll, wrf.Roll - wrf.Side * mz)
        np.testing.assert_allclose(
            out.Pitch, wrf.Pitch - wrf.Lift * mx - wrf.Drag * mz)
        np.testing.assert_allclose(
            out.Yaw, wrf.Yaw + wrf.Side * my - wrf.Side * mx)


class TestUncertainty:

    def test_precision_matches_matlab_formula(self):
        # calc_uncertainty.m: std(temp,[],2)*tinv(.975,ncase)/sqrt(ncase)
        from scipy.stats import t as t_dist
        cases = [np.array([1.0, 2.0]), np.array([2.0, 4.0]),
                 np.array([3.0, 6.0])]
        prec = calc_precision_uncertainty_cases(cases)
        expected = (np.std(np.column_stack(cases), axis=1, ddof=1)
                    * t_dist.ppf(0.975, 3) / np.sqrt(3))
        np.testing.assert_allclose(prec, expected)

    def test_single_case_returns_zero(self):
        prec = calc_precision_uncertainty_cases([np.arange(5.0)])
        np.testing.assert_allclose(prec, np.zeros(5))

    def test_bias_and_total_structure(self):
        n = 16
        wrf = WRFForces()
        for i, name in enumerate(('Lift', 'Drag', 'Side',
                                  'Roll', 'Pitch', 'Yaw')):
            setattr(wrf, name, np.linspace(0.1, 1.0, n) * (i + 1))
        loads = build_load_matrix(wrf)
        assert loads.shape == (n, 6)
        # column order is the cal channel order (Drag, Side, Lift, ...)
        np.testing.assert_allclose(loads[:, 0], wrf.Drag)
        np.testing.assert_allclose(loads[:, 2], wrf.Lift)

        Q = np.full(n, 0.05)
        alpha = np.full(n, 4.0)
        coeffs = {name: np.ones(n)
                  for name in ('Cl', 'Cd', 'Cs', 'CRoll', 'CPitch', 'CYaw')}
        prec = {name: np.full(n, 1e-4) for name in coeffs}
        unc = calc_uncertainty_ext_balance(
            coeffs, loads, alpha, Q, S=18.75, C=2.86, prec=prec)

        assert set(unc) == {'InfCoeffs', 'bias', 'total'}
        for name in coeffs:
            bias_total = np.atleast_1d(unc['bias'][name]['total'])
            total = np.atleast_1d(unc['total'][name])
            assert np.all(np.isfinite(bias_total)), name
            assert np.all(np.isfinite(total)), name
            # total = sqrt(bias^2 + prec^2) >= bias
            assert np.all(total >= bias_total), name

        # Spot-check the Cs bias against the .m formulas:
        # Cs = Fz/QS with fb(:,3) = Lift channel (cal order)
        S = 18.75
        bQ = 0.0005 * np.mean(Q)
        bS = 0.005 ** 2
        pCpFz = 1.0 / Q / S
        pCpQ = -0.5 * loads[:, 2] / Q ** 2 / S
        pCpS = -0.5 * loads[:, 2] / S ** 2 / Q
        expected = np.sqrt((pCpFz * EXTERNAL_CAL_BIAS[2]) ** 2
                           + (pCpQ * bQ) ** 2 + (pCpS * bS) ** 2)
        np.testing.assert_allclose(unc['bias']['Cs']['total'], expected)

    def test_horizontal_config_not_implemented(self):
        with pytest.raises(NotImplementedError):
            calc_uncertainty_ext_balance(
                {'Cl': np.ones(2)}, np.ones((2, 6)), np.zeros(2),
                np.ones(2), S=1.0, C=1.0, config='Horizontal')
