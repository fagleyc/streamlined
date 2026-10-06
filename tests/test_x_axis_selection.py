"""The X Axis dropdown, and what putting a speed variable on x means.

The plot type used to fix the x variable, with one checkbox to swap alpha
for beta. The X Axis combo generalizes that: any built-in quantity or
calculator output can go on x.

Putting a SPEED variable (Mach, Re, q, U_inf) there inverts the trace
grouping. With alpha on x a speed sweep draws one alpha sweep per speed
step; with Mach on x it has to draw one speed sweep per angle, or a Mach
sweep would appear as a column of unconnected points.
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

from PyQt6.QtWidgets import QApplication  # noqa: E402

# Aliased: pytest would try to COLLECT a class named TestCase.
from utils.gui.models.case import TestCase as Case  # noqa: E402
from utils.gui.models.data_model import DataModel  # noqa: E402
from utils.gui.views.plot_panel import PlotPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# 3 alphas x 1 beta x 3 speed steps, in reduced point order
# (alpha, beta, speed). The measured Machs drift within a step, which is
# exactly why the setpoint is the grouping key.
_ALPHAS = [-2.0, -2.0, -2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0]
_MEASURED = [0.051, 0.065, 0.090, 0.044, 0.067, 0.090,
             0.052, 0.071, 0.092]
_SETPOINTS = [0.05, 0.07, 1.00] * 3


def _sweep_case() -> Case:
    case = Case(id="c1", name="Sweep")
    case.alphas = np.array(_ALPHAS)
    case.betas = np.zeros(9)
    case.machs = np.array(_MEASURED)
    case.speeds = np.array(_SETPOINTS)
    case.Cl = np.arange(9, dtype=float)
    case.Cd = np.full(9, 0.05)
    # q in psi, U_inf in m/s - the internal storage units
    case.dynamic_pressures = np.array(_MEASURED) ** 2 * 1000.0
    case.velocities = np.array(_MEASURED) * 340.0
    case.reynolds = np.array(_MEASURED) * 2.0e6
    return case


def _grid_case() -> Case:
    """A 2-D (3 alpha x 2 beta) grid whose per-point Machs are stored FLAT.

    This is the shape the controller produces: alphas/betas come back from
    the reduction as a grid, while the tunnel conditions are collected one
    entry per point into a flat array.
    """
    case = Case(id="c2", name="Grid")
    case.alphas = np.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
    case.betas = np.array([[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]])
    case.Cl = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    case.Cd = np.full((3, 2), 0.05)
    # Flat, one per point, in row-major order matching the grid
    case.machs = np.array([0.10, 0.11, 0.12, 0.13, 0.14, 0.15])
    return case


class _Harness:
    """A PlotPanel with its canvas plot() calls captured."""

    def __init__(self, case):
        self.model = DataModel()
        self.model.cases.add(case)
        self.panel = PlotPanel(self.model)
        self.traces = []
        self.panel.plot_canvas.plot = self._capture

    def _capture(self, x, y, **kw):
        self.traces.append({
            'label': kw.get('label'),
            'x': np.asarray(x, dtype=float),
            'y': np.asarray(y, dtype=float),
        })

    def draw(self, x_var):
        self.panel.plot_controls.plot_selector.set_x_var(x_var)
        self.traces.clear()
        self.panel._update_plot()
        return self.traces


class TestComboContents:
    def test_defaults_to_the_plot_types_own_x(self, qapp):
        panel = PlotPanel(DataModel())
        assert panel.plot_controls.get_x_var() == ""

    def test_offers_the_built_in_variables(self, qapp):
        panel = PlotPanel(DataModel())
        combo = panel.plot_controls.plot_selector.cmb_x_axis
        offered = {combo.itemData(i) for i in range(combo.count())}
        for var in ("", "Alpha", "Beta", "Mach", "Re", "Q", "U_inf",
                    "Cl", "Cd", "Cs", "CRoll", "CPitch", "CYaw", "L/D"):
            assert var in offered

    def test_round_trips_a_selection(self, qapp):
        panel = PlotPanel(DataModel())
        panel.plot_controls.plot_selector.set_x_var("Mach")
        assert panel.plot_controls.get_x_var() == "Mach"

    def test_unknown_name_falls_back_to_default(self, qapp):
        panel = PlotPanel(DataModel())
        panel.plot_controls.plot_selector.set_x_var("Mach")
        panel.plot_controls.plot_selector.set_x_var("NotAVariable")
        assert panel.plot_controls.get_x_var() == ""

    def test_calculator_outputs_are_offered_on_both_axes(self, qapp):
        panel = PlotPanel(DataModel())
        sel = panel.plot_controls.plot_selector
        sel.populate_custom_vars(["Cp1", "Cp2"])
        x_offered = {sel.cmb_x_axis.itemData(i)
                     for i in range(sel.cmb_x_axis.count())}
        y_offered = {sel.cmb_custom_y.itemData(i)
                     for i in range(sel.cmb_custom_y.count())}
        assert {"Cp1", "Cp2"} <= x_offered
        assert {"Cp1", "Cp2"} <= y_offered
        # The built-ins survive alongside them
        assert "Mach" in x_offered

    def test_a_built_in_x_selection_survives_a_calculator_edit(self, qapp):
        panel = PlotPanel(DataModel())
        sel = panel.plot_controls.plot_selector
        sel.set_x_var("Mach")
        sel.populate_custom_vars(["Cp1"])
        assert sel.get_x_var() == "Mach"

    def test_a_custom_x_selection_survives_a_repopulate(self, qapp):
        panel = PlotPanel(DataModel())
        sel = panel.plot_controls.plot_selector
        sel.populate_custom_vars(["Cp1", "Cp2"])
        sel.set_x_var("Cp2")
        sel.populate_custom_vars(["Cp1", "Cp2", "Cp3"])
        assert sel.get_x_var() == "Cp2"


class TestSpeedVariableInvertsGrouping:
    def test_alpha_on_x_gives_one_trace_per_speed_step(self, qapp):
        traces = _Harness(_sweep_case()).draw("Alpha")
        assert len(traces) == 3
        for t in traces:
            np.testing.assert_allclose(t['x'], [-2.0, 0.0, 2.0])

    def test_mach_on_x_gives_one_trace_per_angle(self, qapp):
        traces = _Harness(_sweep_case()).draw("Mach")
        assert len(traces) == 3
        for t in traces:
            assert len(t['x']) == 3, "a speed sweep must be a curve"
        # Each trace walks the three speed steps at one alpha
        np.testing.assert_allclose(traces[0]['x'], [0.051, 0.065, 0.090])
        np.testing.assert_allclose(traces[1]['x'], [0.044, 0.067, 0.090])
        np.testing.assert_allclose(traces[2]['x'], [0.052, 0.071, 0.092])

    def test_traces_are_labelled_by_angle(self, qapp):
        traces = _Harness(_sweep_case()).draw("Mach")
        labels = [t['label'] for t in traces]
        assert labels == ["Sweep α=-2.0°",
                          "Sweep α=0.0°",
                          "Sweep α=2.0°"]

    def test_beta_is_omitted_from_the_label_when_there_is_only_one(self,
                                                                  qapp):
        traces = _Harness(_sweep_case()).draw("Mach")
        assert all("β" not in t['label'] for t in traces)

    def test_y_values_follow_their_points(self, qapp):
        traces = _Harness(_sweep_case()).draw("Mach")
        # Cl is 0..8 in (alpha, speed) order, so alpha=-2 keeps 0,1,2
        np.testing.assert_allclose(traces[0]['y'], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(traces[2]['y'], [6.0, 7.0, 8.0])

    def test_points_are_ordered_by_x_not_by_acquisition(self, qapp):
        # A run where the tunnel overshot: the middle step measured the
        # highest Mach. The line must still be monotonic in x.
        case = _sweep_case()
        case.machs = np.array([0.051, 0.090, 0.065,
                               0.044, 0.090, 0.067,
                               0.052, 0.092, 0.071])
        traces = _Harness(case).draw("Mach")
        for t in traces:
            assert np.all(np.diff(t['x']) > 0)
        np.testing.assert_allclose(traces[0]['y'], [0.0, 2.0, 1.0])

    def test_q_and_velocity_also_invert_the_grouping(self, qapp):
        for var in ("Q", "U_inf", "Re"):
            traces = _Harness(_sweep_case()).draw(var)
            assert len(traces) == 3, var
            assert all(len(t['x']) == 3 for t in traces), var

    def test_a_coefficient_on_x_keeps_the_alpha_sweep_grouping(self, qapp):
        # Cd is not a sweep dimension, so a drag polar still draws one
        # trace per speed step along the alpha sweep.
        traces = _Harness(_sweep_case()).draw("Cd")
        assert len(traces) == 3
        assert all(len(t['x']) == 3 for t in traces)


class TestDimensionalAxesUseTheOutputUnits:
    def test_q_is_converted_and_labelled(self, qapp):
        harness = _Harness(_sweep_case())
        harness.model.output_units = "MKS"
        traces = harness.draw("Q")
        # psi -> Pa
        expected = _sweep_case().dynamic_pressures[0] * 6894.757
        assert traces[0]['x'][0] == pytest.approx(expected, rel=1e-3)
        label = harness.panel._x_axis_label("Q", harness.model.plot_config)
        assert "Pa" in label

    def test_q_in_ips_stays_in_psi(self, qapp):
        harness = _Harness(_sweep_case())
        traces = harness.draw("Q")
        assert traces[0]['x'][0] == pytest.approx(
            _sweep_case().dynamic_pressures[0])
        assert "psi" in harness.panel._x_axis_label(
            "Q", harness.model.plot_config)

    def test_mach_is_never_converted(self, qapp):
        harness = _Harness(_sweep_case())
        harness.model.output_units = "MKS"
        traces = harness.draw("Mach")
        np.testing.assert_allclose(traces[0]['x'], [0.051, 0.065, 0.090])

    def test_a_custom_variable_is_labelled_by_its_name(self, qapp):
        harness = _Harness(_sweep_case())
        harness.panel.plot_controls.plot_selector.populate_custom_vars(
            ["Cp1"])
        harness.panel.plot_controls.plot_selector.set_x_var("Cp1")
        assert harness.panel._x_axis_label(
            "Cp1", harness.model.plot_config) == "Cp1"

    def test_the_plot_type_label_is_kept_when_no_override(self, qapp):
        harness = _Harness(_sweep_case())
        harness.panel.plot_controls.plot_selector.set_x_var("")
        config = harness.model.plot_config
        assert harness.panel._x_axis_label(
            config.x_var, config) == config.x_label


class TestFlatConditionsOnAGrid:
    """Per-point tunnel conditions are flat; alphas/betas may be a grid."""

    def test_a_column_takes_the_right_points(self, qapp):
        # Without conforming, a flat 6-element array indexed by the 3 alpha
        # rows would silently return points 0,1,2 for BOTH beta columns.
        panel = PlotPanel(DataModel())
        case = _grid_case()
        col0 = panel._get_col_data(case, "Mach", 0)
        col1 = panel._get_col_data(case, "Mach", 1)
        np.testing.assert_allclose(col0, [0.10, 0.12, 0.14])
        np.testing.assert_allclose(col1, [0.11, 0.13, 0.15])

    def test_a_row_takes_the_right_points(self, qapp):
        panel = PlotPanel(DataModel())
        case = _grid_case()
        np.testing.assert_allclose(
            panel._get_row_data(case, "Mach", 1), [0.12, 0.13])

    def test_a_grid_keeps_its_own_shaped_arrays(self, qapp):
        panel = PlotPanel(DataModel())
        case = _grid_case()
        np.testing.assert_allclose(
            panel._get_col_data(case, "Cl", 1), [0.2, 0.4, 0.6])

    def test_a_grid_case_does_not_take_the_speed_sweep_path(self, qapp):
        # A 2-D grid exists only when the run held ONE speed, so there is
        # no speed sweep to draw; it keeps the ordinary per-beta grouping.
        traces = _Harness(_grid_case()).draw("Mach")
        assert len(traces) == 2
        np.testing.assert_allclose(traces[0]['x'], [0.10, 0.12, 0.14])
