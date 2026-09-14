"""The Mach filter offers one entry per COMMANDED Mach, like alpha/beta.

Two conditions were showing up as four entries (M = 0.205, 0.207, 0.297,
0.298). Two separate faults produced that:

1. The legacy TDMS naming convention writes Mach as a bare ``M0p25``
   token, which no parser recognized. Those runs carried no speed
   setpoint at all, so the step key fell back to the measured Mach.
2. Even with a setpoint, each step was LABELLED with its own mean
   measured Mach. Two runs commanded to the same Mach that held it a
   thousandth apart are one condition, but offered two filter entries.

A Mach setpoint is now reported as-is, exactly as the alpha filter
reports the commanded angle. Runs commanded in Hz or RPM have no Mach
setpoint to report and keep the mean-measured label, one per step.
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

from utils.gui.models.case import (  # noqa: E402
    CaseCollection, TestCase as Case)
from utils.windtunnel.data_io import (  # noqa: E402
    extract_air_state_from_filename, extract_mach_from_filename,
    extract_speed_from_filename, speed_condition_key)


# ── the legacy filename token ───────────────────────────────────────────
class TestLegacyMachToken:
    """A bare M0p25 token is a Mach setpoint."""

    @pytest.mark.parametrize("name,expected", [
        ("AirOn_DrpPd3_NB1_TF1_R01_M0p25_Sp26_Alpha_-10.0_Beta_0.0.tdms",
         0.25),
        ("AirOff_M0p25_Alpha_0.0_Beta_0.0.tdms", 0.25),
        ("Model_M0.30_Alpha_2.0.tdms", 0.3),
        ("Run_M0p3_Alpha_2.0.tdms", 0.3),
        ("M0p25_Alpha_0.0.tdms", 0.25),
    ])
    def test_parsed(self, name, expected):
        assert extract_mach_from_filename(name) == pytest.approx(expected)
        assert extract_speed_from_filename(name) == (
            pytest.approx(expected), 'mach')

    @pytest.mark.parametrize("name", [
        # M not preceded by a delimiter: a configuration code, not Mach
        "AirOff_WPO-15_WPM0_WPI0_WSI0_WSM0_WSO0_R0_GOff_Alpha_-2.5.tdms",
        "Test_CM0p5_Alpha_1.0.tdms",
        # No decimal separator: 'M0' would read as Mach 0 and turn the
        # run into a tare; 'M3' is far likelier to be a model number.
        "Run_M0_Alpha_2.0.tdms",
        "Config_M3_Alpha_2.0.tdms",
        # Nothing Mach-like at all
        "AirOn_F16check_no_beta_Alpha_-2.0_Beta_0.0.tdms",
    ])
    def test_rejected(self, name):
        assert extract_mach_from_filename(name) is None
        assert extract_speed_from_filename(name) == (None, None)

    def test_the_tagged_form_still_wins(self):
        # A Freestream name carries its own token; the legacy pattern is
        # only a fallback and must not pre-empt it.
        assert extract_speed_from_filename(
            "run_0001_alpha_0.0_beta_0.0_Hz_30.0.h5") == (30.0, 'hz')
        assert extract_speed_from_filename(
            "run_0009_alpha_-4.0_beta_0.0_mach_0.05.mat") == (0.05, 'mach')

    def test_a_tare_with_a_mach_token_is_still_a_tare(self):
        # Air state comes from the filename prefix, not the speed, so
        # giving air-off files a nonzero setpoint must not reclassify
        # them - that would silently destroy every tare subtraction.
        name = "AirOff_DrpPd3_M0p25_Alpha_0.0_Beta_0.0.tdms"
        assert extract_air_state_from_filename(name) == 'AirOff'
        assert extract_speed_from_filename(name)[0] == pytest.approx(0.25)

    def test_a_nonzero_setpoint_is_its_own_condition(self):
        assert speed_condition_key(0.25, 'mach') == 'mach_0.25'
        assert speed_condition_key(0.0, 'mach') == 'AirOff'


# ── the step key ────────────────────────────────────────────────────────
def _case(name, measured, commanded, unit='mach', n_alpha=4):
    """One case: n_alpha angles at each commanded speed step."""
    case = Case(id=name, name=name)
    alphas, machs, speeds = [], [], []
    for cmd, meas in zip(commanded, measured):
        for i in range(n_alpha):
            alphas.append(float(i * 2))
            # The tunnel drifts across the sweep, as it really does
            machs.append(meas + 0.0004 * i)
            speeds.append(cmd)
    case.alphas = np.array(alphas)
    case.betas = np.zeros(len(alphas))
    case.Cl = np.arange(len(alphas), dtype=float)
    case.machs = np.array(machs)
    case.speeds = np.array(speeds, dtype=float)
    case.speed_unit = unit
    case.alpha_nominal = np.array(alphas)
    case.beta_nominal = np.zeros(len(alphas))
    return case


class TestCommandedMachIsTheKey:
    def test_the_setpoint_is_reported_as_is(self):
        case = _case("A", [0.2051, 0.2971], [0.2, 0.3])
        assert sorted(set(case.point_machs.tolist())) == [0.2, 0.3]

    def test_drift_within_a_step_does_not_split_it(self):
        case = _case("A", [0.2051, 0.2971], [0.2, 0.3], n_alpha=8)
        for step in (0.2, 0.3):
            assert np.isclose(case.point_machs, step,
                              atol=5e-4).sum() == 8

    def test_two_runs_at_the_same_command_offer_one_entry(self):
        # The reported bug: two configurations commanded to M0.2 and
        # M0.3 that held them a thousandth apart gave four entries.
        cases = CaseCollection()
        cases.add(_case("run A", [0.205, 0.297], [0.2, 0.3]))
        cases.add(_case("run B", [0.207, 0.298], [0.2, 0.3]))
        assert cases.all_mach_numbers == [0.2, 0.3]

    def test_measured_means_alone_would_give_four_entries(self):
        # The guard: the same two runs without a Mach setpoint.
        cases = CaseCollection()
        for nm, meas in (("run A", [0.205, 0.297]),
                         ("run B", [0.207, 0.298])):
            case = _case(nm, meas, [0.2, 0.3], unit='hz')
            cases.add(case)
        assert len(cases.all_mach_numbers) == 4

    def test_distinct_commands_stay_distinct(self):
        case = _case("A", [0.199, 0.2005], [0.19, 0.20])
        assert sorted(set(case.point_machs.tolist())) == [0.19, 0.2]


class TestNonMachSetpoints:
    """A speed commanded in Hz says nothing about Mach."""

    def test_each_step_keeps_its_mean_measured_mach(self):
        case = _case("A", [0.0305, 0.0605], [20.0, 40.0], unit='hz')
        steps = sorted(set(case.point_machs.tolist()))
        assert len(steps) == 2
        # Labelled by what was measured, not by 20 and 40
        assert all(0.0 < s < 0.1 for s in steps)

    def test_the_step_is_still_whole(self):
        case = _case("A", [0.0305, 0.0605], [20.0, 40.0], unit='hz',
                     n_alpha=6)
        for step in set(case.point_machs.tolist()):
            assert np.isclose(case.point_machs, step, atol=5e-4).sum() == 6

    def test_an_rpm_sweep_is_not_read_as_a_mach(self):
        case = _case("A", [0.05, 0.09], [1500.0, 3000.0], unit='rpm')
        assert max(case.point_machs) < 1.0


class TestFallbacks:
    def test_no_setpoints_falls_back_to_the_measured_mach(self):
        case = _case("A", [0.205, 0.297], [0.2, 0.3])
        case.speeds = np.array([])
        np.testing.assert_allclose(case.point_machs, case.machs)

    def test_misaligned_setpoints_are_ignored(self):
        case = _case("A", [0.205, 0.297], [0.2, 0.3])
        case.speeds = np.array([0.2, 0.3])
        np.testing.assert_allclose(case.point_machs, case.machs)

    def test_an_unrecorded_unit_keeps_the_measured_label(self):
        case = _case("A", [0.205, 0.297], [0.2, 0.3], unit='')
        steps = sorted(set(case.point_machs.tolist()))
        assert steps != [0.2, 0.3]
        assert len(steps) == 2

    def test_a_point_without_a_setpoint_keeps_its_measured_value(self):
        case = _case("A", [0.205, 0.297], [0.2, 0.3])
        case.speeds = case.speeds.copy()
        case.speeds[0] = np.nan
        out = case.point_machs
        assert np.isfinite(out).all(), "a NaN key drops the point entirely"
        assert out[0] == pytest.approx(case.machs[0])

    def test_empty_without_machs(self):
        case = _case("A", [0.205, 0.297], [0.2, 0.3])
        case.machs = np.array([])
        assert case.point_machs.size == 0


class TestSpeedUnitReachesTheCase:
    def test_the_controller_attaches_it(self):
        from utils.gui.controllers.data_controller import ProcessingWorker

        class _SS:
            speeds = np.array([0.2, 0.2, 0.3, 0.3])
            speed_unit = 'mach'

        case = Case(id="c", name="c")
        case.alphas = np.array([0.0, 2.0, 0.0, 2.0])
        ProcessingWorker._attach_speed_setpoints(case, _SS())
        assert case.speed_unit == 'mach'
        np.testing.assert_allclose(np.ravel(case.speeds),
                                   [0.2, 0.2, 0.3, 0.3])

    def test_a_missing_unit_is_an_empty_string_not_none(self):
        from utils.gui.controllers.data_controller import ProcessingWorker

        class _SS:
            speeds = np.array([0.2, 0.2])
            speed_unit = None

        case = Case(id="c", name="c")
        case.alphas = np.array([0.0, 2.0])
        ProcessingWorker._attach_speed_setpoints(case, _SS())
        assert case.speed_unit == ''

    def test_the_injector_carries_the_unit_in(self):
        from utils.gui.controllers.data_controller import ProcessingWorker

        class _Info:
            alpha, beta, speed, speed_unit = 0.0, 0.0, 0.25, 'mach'

        channels = {}
        ProcessingWorker._inject_nominal_setpoints(channels, _Info())
        assert channels['speed_value'] == pytest.approx(0.25)
        assert channels['speed_unit'] == 'mach'

    def test_an_existing_unit_marker_is_not_overwritten(self):
        from utils.gui.controllers.data_controller import ProcessingWorker

        class _Info:
            alpha, beta, speed, speed_unit = 0.0, 0.0, 30.0, 'mach'

        channels = {'speed_value': 30.0, 'speed_unit': 'hz'}
        ProcessingWorker._inject_nominal_setpoints(channels, _Info())
        assert channels['speed_unit'] == 'hz'
