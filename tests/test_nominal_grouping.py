"""Groups come from what the run was COMMANDED, not what it measured.

The positioner never lands exactly on the commanded angle, and it misses
differently on every pass: a commanded alpha of 4.0 records as 3.949 on
one speed step and 3.951 on the next. Both are the same point of the
sweep, but rounding the measured value to a tenth puts them in different
groups - two traces, two filter entries, two export columns, each
holding half the data.

The run's own record (meta.run, the directory manifest, the filename)
carries the commanded value, and that is the grouping key. The measured
attitude stays what gets plotted and exported as the angle the model
actually flew at.
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

from utils.gui.models.case import (  # noqa: E402
    CaseCollection, TestCase as Case)
from utils.gui.models.data_model import DataModel  # noqa: E402
from utils.gui.views.plot_panel import PlotPanel  # noqa: E402
from utils.gui.views.table_panel import TablePanel  # noqa: E402
from utils.windtunnel.data_io import (  # noqa: E402
    BALANCE_MARKER_KEYS, MANIFEST_FILENAME, MANIFEST_SCHEMA_VERSION)
from utils.windtunnel.reduction import (  # noqa: E402
    reduce_single_point, reduce_steady_state)
from utils.windtunnel.transforms import Geometry  # noqa: E402

scipy_io = pytest.importorskip("scipy.io")

CHANNELS = ("Lift", "Drag", "Side", "Roll", "Pitch", "Yaw")

# Two speed steps at each of three commanded alphas. Every measured
# value sits within 0.002 deg of its command, but two of the three
# commands straddle a rounding boundary at one tenth: 3.949 rounds to
# 3.9 and 3.951 to 4.0, and likewise 5.949 / 5.951.
_COMMANDED = [0.0, 0.0, 4.0, 4.0, 6.0, 6.0]
_MEASURED = [-0.001, 0.001, 3.949, 3.951, 5.949, 5.951]
_SETPOINTS = [0.05, 0.07] * 3


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([sys.argv[0]])


def _straddling_case(nominal=True) -> Case:
    case = Case(id="c1", name="Sweep")
    case.alphas = np.array(_MEASURED, dtype=float)
    case.betas = np.zeros(6)
    case.Cl = np.arange(6, dtype=float)
    case.Cd = np.full(6, 0.05)
    case.machs = np.array([0.051, 0.069] * 3)
    case.speeds = np.array(_SETPOINTS, dtype=float)
    if nominal:
        case.alpha_nominal = np.array(_COMMANDED, dtype=float)
        case.beta_nominal = np.zeros(6)
    return case


class _Harness:
    """A PlotPanel whose canvas plot() calls are captured."""

    def __init__(self, case):
        self.model = DataModel()
        self.model.cases.add(case)
        self.panel = PlotPanel(self.model)
        self.traces = []
        self.panel.plot_canvas.plot = self._capture

    def _capture(self, x, y, **kw):
        self.traces.append({'label': kw.get('label'),
                            'x': np.asarray(x, dtype=float)})

    def draw(self, x_var=""):
        self.panel.plot_controls.plot_selector.set_x_var(x_var)
        self.traces.clear()
        self.panel._update_plot()
        return self.traces


class TestGroupingKeys:
    def test_point_alphas_are_the_commanded_angles(self):
        np.testing.assert_allclose(_straddling_case().point_alphas,
                                   _COMMANDED)

    def test_measured_alphas_are_left_alone(self):
        np.testing.assert_allclose(_straddling_case().alphas, _MEASURED)

    def test_measured_values_alone_would_split_the_sweep(self):
        # The guard on the fix: three commanded angles, five groups.
        case = _straddling_case(nominal=False)
        assert len(set(np.round(case.point_alphas, 1).tolist())) == 5

    def test_commanded_values_give_one_group_per_angle(self):
        case = _straddling_case()
        assert len(set(np.round(case.point_alphas, 1).tolist())) == 3

    def test_falls_back_to_measured_without_a_commanded_record(self):
        case = _straddling_case(nominal=False)
        np.testing.assert_allclose(case.point_alphas, _MEASURED)
        np.testing.assert_allclose(case.point_betas, np.zeros(6))

    def test_a_misaligned_record_is_ignored(self):
        case = _straddling_case()
        case.alpha_nominal = np.array([0.0, 4.0])      # wrong length
        np.testing.assert_allclose(case.point_alphas, _MEASURED)

    def test_a_record_with_gaps_is_ignored(self):
        case = _straddling_case()
        case.alpha_nominal = np.array(_COMMANDED, dtype=float)
        case.alpha_nominal[2] = np.nan
        np.testing.assert_allclose(case.point_alphas, _MEASURED)

    def test_the_key_keeps_the_shape_of_a_grid(self):
        case = Case(id="g", name="Grid")
        case.alphas = np.array([[-0.01, 0.01], [3.99, 4.01]])
        case.betas = np.array([[-5.0, 5.0], [-5.0, 5.0]])
        case.alpha_nominal = np.array([0.0, 0.0, 4.0, 4.0])
        assert case.point_alphas.shape == (2, 2)
        np.testing.assert_allclose(case.point_alphas,
                                   [[0.0, 0.0], [4.0, 4.0]])


class TestFilterEnumeration:
    def test_filter_offers_one_entry_per_commanded_angle(self):
        cases = CaseCollection()
        cases.add(_straddling_case())
        assert cases.all_alpha_values == [0.0, 4.0, 6.0]

    def test_measured_values_alone_would_overfill_the_filter(self):
        cases = CaseCollection()
        cases.add(_straddling_case(nominal=False))
        assert len(cases.all_alpha_values) == 5

    def test_beta_filter_uses_the_commanded_angle(self):
        case = _straddling_case()
        case.betas = np.array([-5.049, -4.951, -5.049, -4.951,
                               -5.049, -4.951])
        case.beta_nominal = np.full(6, -5.0)
        cases = CaseCollection()
        cases.add(case)
        assert cases.all_beta_values == [-5.0]


class TestPlotTraces:
    def test_one_trace_per_commanded_angle_against_mach(self, qapp):
        traces = _Harness(_straddling_case()).draw("Mach")
        assert [t['label'] for t in traces] == [
            "Sweep α=0.0°",
            "Sweep α=4.0°",
            "Sweep α=6.0°"]
        assert all(len(t['x']) == 2 for t in traces)

    def test_measured_values_alone_would_fragment_the_traces(self, qapp):
        # Three commanded angles, five traces. The point masks are
        # generous (atol=0.15), so a straddled angle is not torn in half
        # - it is drawn TWICE under two labels, one per rounded reading,
        # which is what fills the legend with phantom angles.
        traces = _Harness(_straddling_case(nominal=False)).draw("Mach")
        labels = [t['label'] for t in traces]
        assert len(traces) == 5
        assert "Sweep α=3.9°" in labels and "Sweep α=4.0°" in labels
        assert "Sweep α=5.9°" in labels and "Sweep α=6.0°" in labels

    def test_the_alpha_filter_selects_the_whole_angle(self, qapp):
        harness = _Harness(_straddling_case())
        harness.panel.filter_toolbar.set_alpha_values(
            harness.model.get_available_alphas())
        traces = harness.draw("Mach")
        # Both speed steps of the commanded 4.0 stay together
        four = [t for t in traces if "4.0" in t['label']]
        assert len(four) == 1 and len(four[0]['x']) == 2

    def test_x_axis_still_reports_the_measured_attitude(self, qapp):
        # Grouping is commanded; the plotted angle is what was measured.
        traces = _Harness(_straddling_case()).draw("Alpha")
        x = np.concatenate([t['x'] for t in traces])
        assert np.isclose(np.sort(x), sorted(_MEASURED)).all()


class TestExportAxes:
    def test_axes_are_the_commanded_grid(self):
        to3d, axes = TablePanel._case_3d_grid(_straddling_case())
        np.testing.assert_allclose(axes['alpha'], [0.0, 4.0, 6.0])
        np.testing.assert_allclose(axes['shape'], [3, 1, 2])
        grid = to3d(_straddling_case().Cl)
        assert grid.shape == (3, 2)
        assert not np.isnan(grid).any()

    def test_measured_values_alone_would_scatter_the_grid(self):
        _, axes = TablePanel._case_3d_grid(_straddling_case(nominal=False))
        assert len(np.ravel(axes['alpha'])) == 5


# ── the reduction carries the commanded values ──────────────────────────
def _raw(alpha_measured, alpha_cmd=None, beta_cmd=None, n=8):
    base = {"Lift": 0.5, "Drag": 2.0, "Side": 20.0,
            "Roll": 1.0, "Pitch": 3.0, "Yaw": 7.0}
    on = {k: np.full(n, v, dtype=float) for k, v in base.items()}
    on.update({"Alpha": np.full(n, alpha_measured), "Beta": np.zeros(n),
               "Pdiff": np.full(n, 0.8), "Ptot": np.full(n, 12.2),
               "Temp": np.full(n, 295.0),
               "balance_type": "external", "span_config": "full"})
    if alpha_cmd is not None:
        on["alpha_nominal"] = alpha_cmd
    if beta_cmd is not None:
        on["beta_nominal"] = beta_cmd
    off = {k: (np.zeros(n) if k in CHANNELS else v) for k, v in on.items()}
    return on, off


def _reduce(on, off):
    geo = Geometry(C=2.86, S=18.75, b=6.0, mshift=np.zeros(3))
    return reduce_single_point(on, off, cal=None, geo=geo,
                               pressure_cal={}, facility='SWT')


class TestReductionCarriesTheCommandedAttitude:
    def test_markers_reach_the_reduced_point(self):
        red = _reduce(*_raw(3.949, alpha_cmd=4.0, beta_cmd=-5.0))
        assert red.alpha_nominal == pytest.approx(4.0)
        assert red.beta_nominal == pytest.approx(-5.0)

    def test_the_measured_attitude_is_untouched(self):
        red = _reduce(*_raw(3.949, alpha_cmd=4.0))
        assert np.allclose(red.alpha, 3.949)

    def test_absent_markers_leave_it_unknown(self):
        red = _reduce(*_raw(3.949))
        assert red.alpha_nominal is None
        assert red.beta_nominal is None

    def test_the_markers_are_not_exported_as_channels(self):
        # They ride in the channel dict; the export has to skip them.
        assert 'alpha_nominal' in BALANCE_MARKER_KEYS
        assert 'beta_nominal' in BALANCE_MARKER_KEYS

    def test_steady_state_exposes_the_commanded_arrays(self):
        points = [_reduce(*_raw(m, alpha_cmd=c))
                  for m, c in zip(_MEASURED, _COMMANDED)]
        ss = reduce_steady_state(points)
        np.testing.assert_allclose(np.ravel(ss.alpha_nominal),
                                   sorted(_COMMANDED))

    def test_steady_state_leaves_it_empty_without_markers(self):
        points = [_reduce(*_raw(m)) for m in _MEASURED]
        ss = reduce_steady_state(points)
        assert np.asarray(ss.alpha_nominal).size == 0

    def test_a_partial_record_is_refused_wholesale(self):
        # One point missing its command makes the key array untrustworthy.
        points = [_reduce(*_raw(m, alpha_cmd=c))
                  for m, c in zip(_MEASURED[:-1], _COMMANDED[:-1])]
        points.append(_reduce(*_raw(_MEASURED[-1])))
        ss = reduce_steady_state(points)
        assert np.asarray(ss.alpha_nominal).size == 0

    def test_the_grid_is_detected_on_the_commanded_angles(self):
        # Three commanded alphas x one beta, measured straddling a half
        # degree: 3.749 rounds to 3.5 and 3.751 to 4.0, which would make
        # the reduction see four alphas and refuse to build the grid.
        measured = [0.0, 3.749, 3.751]
        commanded = [0.0, 4.0, 4.0]
        with_cmd = reduce_steady_state(
            [_reduce(*_raw(m, alpha_cmd=c))
             for m, c in zip(measured, commanded)])
        assert np.asarray(with_cmd.Cl).ndim == 1   # 3 points, 2 alphas

        n_unique = len({round(m * 2) / 2 for m in measured})
        assert n_unique == 3, "the measured values really do straddle"


# ── end to end through the processing worker ────────────────────────────
def _write_run(directory: Path, name: str, measured: float,
               commanded: float, mach: float, air_state: str,
               n: int = 8) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    group = {c: np.ones(n) for c in CHANNELS}
    # Dynamic pressure follows the speed step, so the two steps really
    # do measure different Machs.  (point_machs labels each step with its
    # mean measured Mach, and deliberately merges steps that agree to
    # within 0.001 - constant tunnel conditions would look like one step.)
    group["Pdiff"] = np.full(n, 0.8 * (max(mach, 0.05) / 0.05) ** 2)
    group["Ptot"] = np.full(n, 12.2)
    group["Temp"] = np.full(n, 295.0)
    group["Alpha"] = np.full(n, measured)
    group["Beta"] = np.zeros(n)
    # A measured speed channel that jitters, and NO speed_value marker:
    # the filename setpoint has to be what groups the steps.
    group["Speed"] = np.full(n, mach + 0.0007 * commanded)
    scipy_io.savemat(str(directory / name), {
        "ATE_Balance": group,
        "Time": {"Time": np.arange(n) / 50.0},
        "meta": {"run": {"balance_type": "external", "span_config": "full",
                         "air_state": air_state, "alpha": commanded,
                         "beta": 0.0},
                 "config_json": json.dumps({})},
    }, long_field_names=True)
    return name


def _run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "jitter"
    points = []
    run = 0
    # Commanded 4.0 and 6.0 both straddle a tenth boundary
    for commanded, measured in ((4.0, 3.949), (6.0, 5.949)):
        run += 1
        points.append((run, _write_run(
            d, "run_%04d_alpha_%.1f_beta_0.0_mach_0.00.mat" % (run,
                                                               commanded),
            measured, commanded, 0.0, "AirOff"), commanded, 0.0))
    for mach in (0.05, 0.07):
        for commanded, measured in ((4.0, 3.949), (6.0, 5.951)):
            run += 1
            points.append((run, _write_run(
                d, "run_%04d_alpha_%.1f_beta_0.0_mach_%.2f.mat" % (
                    run, commanded, mach),
                measured + (0.002 if mach > 0.05 else 0.0), commanded,
                mach, "AirOn"), commanded, mach))
    (d / MANIFEST_FILENAME).write_text(json.dumps({
        "schema_version": MANIFEST_SCHEMA_VERSION, "config_name": d.name,
        "output_format": "mat",
        "points": [{"run_number": r, "filename": f, "alpha": a,
                    "beta": 0.0, "mach": m,
                    "air_state": ("AirOff" if m == 0.0 else "AirOn")}
                   for r, f, a, m in points]}), encoding="utf-8")
    return d


def _process(directory: Path):
    from utils.gui.controllers.data_controller import ProcessingWorker
    worker = ProcessingWorker(
        directories=[str(directory)], balance_cal=None, pressure_cal=None,
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
    assert len(cases) == 1
    return cases[0]


class TestInjector:
    """What the controller rides in with each point's channels."""

    @staticmethod
    def _inject(channels, **info_fields):
        from utils.gui.controllers.data_controller import ProcessingWorker

        class _Info:
            pass
        info = _Info()
        for k, v in info_fields.items():
            setattr(info, k, v)
        ProcessingWorker._inject_nominal_setpoints(channels, info)
        return channels

    def test_commanded_attitude_is_injected(self):
        out = self._inject({}, alpha=4.0, beta=-5.0, speed=0.05)
        assert out['alpha_nominal'] == pytest.approx(4.0)
        assert out['beta_nominal'] == pytest.approx(-5.0)

    def test_the_speed_setpoint_fills_in_when_absent(self):
        # read_mat_file usually supplies this marker already; the
        # fallback covers a reader that does not.
        out = self._inject({}, alpha=0.0, beta=0.0, speed=0.07)
        assert out['speed_value'] == pytest.approx(0.07)

    def test_an_existing_speed_marker_is_not_overwritten(self):
        out = self._inject({'speed_value': 0.05}, alpha=0.0, beta=0.0,
                           speed=99.0)
        assert out['speed_value'] == pytest.approx(0.05)

    def test_unknown_fields_are_left_out_entirely(self):
        out = self._inject({}, alpha=None, beta=None, speed=None)
        assert 'alpha_nominal' not in out
        assert 'beta_nominal' not in out
        assert 'speed_value' not in out

    def test_real_channels_are_untouched(self):
        channels = {'Lift': np.ones(4)}
        out = self._inject(channels, alpha=4.0, beta=0.0, speed=0.05)
        np.testing.assert_allclose(out['Lift'], np.ones(4))


class TestWorkerEndToEnd:
    def test_the_case_carries_the_commanded_attitude(self, qapp, tmp_path):
        case = _process(_run_dir(tmp_path))
        np.testing.assert_allclose(np.sort(np.ravel(case.point_alphas)),
                                   [4.0, 4.0, 6.0, 6.0])

    def test_the_measured_attitude_is_still_available(self, qapp,
                                                      tmp_path):
        case = _process(_run_dir(tmp_path))
        measured = np.ravel(case.alphas)
        assert not np.allclose(measured, np.ravel(case.point_alphas))

    def test_the_filter_offers_the_commanded_angles(self, qapp, tmp_path):
        case = _process(_run_dir(tmp_path))
        cases = CaseCollection()
        cases.add(case)
        assert cases.all_alpha_values == [4.0, 6.0]

    def test_the_setpoint_beats_the_measured_speed_channel(self, qapp,
                                                            tmp_path):
        # The runs carry a Speed channel that reads differently at every
        # point (0.0528, 0.0542, ...). Grouping on it would give one step
        # per point; the recorded setpoint gives two.
        case = _process(_run_dir(tmp_path))
        assert sorted(set(np.ravel(case.speeds).tolist())) == [0.05, 0.07]

    def test_two_speed_steps_give_two_mach_groups(self, qapp, tmp_path):
        case = _process(_run_dir(tmp_path))
        assert len(set(np.ravel(case.point_machs).tolist())) == 2

    def test_the_markers_stay_out_of_the_raw_export(self, qapp, tmp_path):
        case = _process(_run_dir(tmp_path))
        point = case.daq.red[0]
        data = TablePanel._extract_unsteady_point(TablePanel, point)
        assert 'raw_alpha_nominal' not in data
        assert 'raw_beta_nominal' not in data
