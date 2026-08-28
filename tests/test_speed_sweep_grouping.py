"""A speed sweep is grouped by its SETPOINT, not by the Mach it measured.

A Mach sweep records every speed step into one run directory, and one
alpha sweep is run at each step. The tunnel does not hold an exact Mach
across a sweep: a step commanded at M0.05 was measured at 0.0436 to
0.0521 on the Check_NACA_08_26_b run. Grouping on that measured value
shatters three 8-point curves into eleven fragments, five of them a
single point, both on the plot and in the MATLAB export's Mach axis.

The setpoint the operator commanded is what identifies a step, so these
tests pin the setpoint as the grouping key and the step's MEAN measured
Mach as its label.

Also covered: a run directory's 'processed' output subdirectory must
never be indexed as run data (it makes the folder disagree with
manifest.json), and the self-describing balance markers that ride inside
a raw channel dict must never be exported as if they were channels.
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

# Aliased: pytest would try to COLLECT a class named TestCase.
from utils.gui.models.case import (  # noqa: E402
    CaseCollection, TestCase as Case)
from utils.windtunnel.data_io import (  # noqa: E402
    BALANCE_MARKER_KEYS, find_run_files, scan_run_directory)


# Measured Machs from the Check_NACA_08_26_b Mach sweep: 8 alphas at each
# of three commanded steps (0.05, 0.07 and a step the acquisition
# labelled 1.00 that the tunnel actually ran near 0.09).
_MEASURED = [
    0.0509, 0.0654, 0.0899, 0.0509, 0.0654, 0.0903,
    0.0436, 0.0670, 0.0904, 0.0513, 0.0676, 0.0910,
    0.0516, 0.0708, 0.0916, 0.0519, 0.0712, 0.0921,
    0.0521, 0.0715, 0.0923, 0.0521, 0.0716, 0.0924,
]
_SETPOINTS = [0.05, 0.07, 1.00] * 8
_SWEEP_ALPHAS = (-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0)
_ALPHAS = [a for a in _SWEEP_ALPHAS for _ in range(3)]


def _sweep_case() -> Case:
    case = Case(id="c1", name="Check_NACA_08_26_b")
    case.alphas = np.array(_ALPHAS, dtype=float)
    case.betas = np.zeros(24)
    case.Cl = np.linspace(-0.5, 1.4, 24)
    case.machs = np.array(_MEASURED, dtype=float)
    case.speeds = np.array(_SETPOINTS, dtype=float)
    return case


class TestStepGrouping:
    """point_machs collapses each speed step to a single Mach."""

    def test_three_setpoints_give_three_groups(self):
        case = _sweep_case()
        assert sorted(set(case.point_machs.tolist())) == [0.051, 0.069, 0.091]

    def test_measured_mach_alone_would_fragment_the_sweep(self):
        # The guard on the fix: without the setpoints, the same data
        # rounds into eleven groups instead of three.
        case = _sweep_case()
        case.speeds = np.array([])
        assert len(set(np.round(case.point_machs, 3).tolist())) == 11

    def test_every_step_keeps_all_eight_alphas(self):
        case = _sweep_case()
        grouped = case.point_machs
        for step in sorted(set(grouped.tolist())):
            mask = np.isclose(grouped, step, atol=5e-4)
            assert mask.sum() == 8, "step {0} lost points".format(step)
            assert sorted(case.alphas[mask].tolist()) == sorted(_SWEEP_ALPHAS)

    def test_label_is_the_steps_mean_measured_mach(self):
        # Not the commanded setpoint: the third step was commanded 1.00
        # and ran near 0.091, and reporting M=1.000 would be a lie.
        case = _sweep_case()
        third = case.point_machs[2]
        assert third == pytest.approx(0.091)
        assert round(float(np.mean(_MEASURED[2::3])), 3) == third

    def test_mach_filter_offers_one_entry_per_step(self):
        cases = CaseCollection()
        cases.add(_sweep_case())
        assert cases.all_mach_numbers == [0.051, 0.069, 0.091]

    def test_filter_value_matches_a_step_exactly(self):
        # The combo stores the rounded value; the plot mask compares it
        # with np.isclose(atol=5e-4). Rounding both sides identically is
        # what keeps that comparison off the tolerance edge.
        cases = CaseCollection()
        case = _sweep_case()
        cases.add(case)
        for offered in cases.all_mach_numbers:
            mask = np.isclose(case.point_machs, offered, atol=5e-4)
            assert mask.sum() == 8

    def test_falls_back_to_measured_mach_without_setpoints(self):
        case = _sweep_case()
        case.speeds = np.array([])
        np.testing.assert_allclose(case.point_machs, _MEASURED)

    def test_misaligned_setpoints_are_ignored(self):
        case = _sweep_case()
        case.speeds = np.array([0.05, 0.07])
        np.testing.assert_allclose(case.point_machs, _MEASURED)

    def test_empty_without_machs(self):
        case = _sweep_case()
        case.machs = np.array([])
        assert case.point_machs.size == 0


class TestMatGridUsesSteps:
    """The MATLAB export's Mach axis is the step axis."""

    def test_grid_is_dense_over_alpha_and_step(self):
        from utils.gui.views.table_panel import TablePanel

        to3d, axes = TablePanel._case_3d_grid(_sweep_case())
        assert to3d is not None
        np.testing.assert_allclose(axes['mach'], [0.051, 0.069, 0.091])
        np.testing.assert_allclose(axes['shape'], [8, 1, 3])

        grid = to3d(_sweep_case().Cl)
        assert grid.shape == (8, 3)
        assert not np.isnan(grid).any(), "grid should have no empty cells"


class TestProcessedOutputIsNotRunData:
    """The 'processed' folder holds results OF a run, never a run."""

    @staticmethod
    def _make_run_dir(tmp_path: Path) -> Path:
        import json
        import scipy.io

        data = {'Pdiff': np.zeros(4), 'Ptot': np.zeros(4)}
        points = []
        for n in (1, 2):
            name = "run_{0:04d}_alpha_0.0_beta_0.0_mach_0.05.mat".format(n)
            scipy.io.savemat(str(tmp_path / name), data)
            points.append({'run_number': n, 'filename': name,
                           'alpha': 0.0, 'beta': 0.0, 'mach': 0.05,
                           'air_state': 'AirOn'})
        (tmp_path / 'processed').mkdir()
        scipy.io.savemat(str(tmp_path / 'processed' / 'summary.mat'), data)
        (tmp_path / 'manifest.json').write_text(json.dumps({
            'schema_version': 1, 'config_name': 'X', 'points': points}))
        return tmp_path

    def test_recursive_scan_skips_the_output_folder(self, tmp_path):
        run_dir = self._make_run_dir(tmp_path)
        names = [f.name for f in find_run_files(run_dir, recursive=True)]
        assert names == ['run_0001_alpha_0.0_beta_0.0_mach_0.05.mat',
                         'run_0002_alpha_0.0_beta_0.0_mach_0.05.mat']

    def test_no_manifest_mismatch_is_reported(self, tmp_path):
        run_dir = self._make_run_dir(tmp_path)
        infos, mismatches = scan_run_directory(run_dir, recursive=True)
        assert mismatches == []
        assert len(infos) == 2

    def test_a_stray_run_file_is_still_reported(self, tmp_path):
        # The exclusion must not blind the check to a genuine
        # manifest-vs-disk disagreement.
        import scipy.io
        run_dir = self._make_run_dir(tmp_path)
        scipy.io.savemat(str(run_dir / 'run_0003_alpha_2.0_beta_0.0.mat'),
                         {'Pdiff': np.zeros(4)})
        _, mismatches = scan_run_directory(run_dir, recursive=True)
        assert any('not listed' in m for m in mismatches)


class TestMarkersAreNotChannels:
    """The balance markers share the channel dict but are metadata."""

    class _Point:
        def __init__(self):
            self.air_on = {
                'Pdiff': np.zeros(8),
                'balance_type': 'external',
                'speed_value': 0.05,
                'speed_unit': 'mach',
                'channel_cal': {'Pdiff': {'slope': 1.0}},
            }

    def test_unsteady_extract_survives_scalar_markers(self):
        # len() on a 0-d marker raised TypeError and aborted the whole
        # unsteady MAT/HDF5 export for any external-balance run.
        from utils.gui.views.table_panel import TablePanel

        data = TablePanel._extract_unsteady_point(TablePanel, self._Point())
        assert 'raw_Pdiff' in data
        for marker in BALANCE_MARKER_KEYS:
            assert 'raw_{0}'.format(marker) not in data
