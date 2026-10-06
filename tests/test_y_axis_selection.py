"""The Y Axis selector: any coefficient, wind- or body-axis load, balance
element, tunnel condition or derivative, plotted in the output units."""
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

from utils.gui import plot_variables as pv                    # noqa: E402
from utils.gui.models.case import TestCase as Case            # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([sys.argv[0]])


def _case():
    from utils.windtunnel.transforms import BRFForces, calc_wrf_forces
    case = Case(id="c", name="F16")
    case.alphas = np.array([-4.0, 0.0, 4.0, 8.0])
    case.betas = np.array([0.0, 0.0, 5.0, 5.0])
    case.Cl = np.array([-0.19, 0.09, 0.38, 0.65])
    case.Cd = np.array([0.05, 0.04, 0.05, 0.07])
    brf = BRFForces()
    brf.Fx = np.array([0.5, 0.4, -0.2, -1.0])
    brf.Fy = np.array([0.1, 0.0, 0.3, 0.4])
    brf.Fz = np.array([-2.0, 1.0, 4.0, 7.5])
    brf.Mx = brf.My = brf.Mz = np.zeros(4)
    wrf = calc_wrf_forces(brf, case.alphas, case.betas)
    case.lift_forces, case.drag_forces = wrf.Lift, wrf.Drag
    case.side_forces = wrf.Side
    case.roll_moments = np.array([0.1, 0.2, 0.3, 0.4])
    case.pitch_moments = np.array([1.0, 2.0, 3.0, 4.0])
    case.yaw_moments = np.array([-0.1, 0.0, 0.1, 0.2])
    case.dynamic_pressures = np.full(4, 0.66)
    case._brf = brf
    return case


class TestRegistry:
    def test_every_group_is_offered(self):
        groups = [g for g, _v in pv.grouped(["Cp1"])]
        assert groups == list(pv.GROUP_ORDER)

    def test_element_entries_take_the_balance_names(self):
        names = ["N1", "N2", "Y1", "Y2", "Axial", "Roll"]
        elems = dict(pv.grouped([], names))[pv.ELEMENTS]
        assert [v.label for v in elems] == names
        assert pv.axis_label("elem_Y1", "IPS", names) == "Y1 [lbf]"

    @pytest.mark.parametrize("key,units,expected", [
        ("Lift", "IPS", "Lift [lbf]"),
        ("Lift", "MKS", "Lift [N]"),
        ("PitchMoment", "IPS", "Pitching moment [lb-in]"),
        ("Fz", "MKS", "$F_z$ [N]"),
        ("Q", "IPS", "q [psi]"),
        ("Cl", "MKS", r"$C_L$"),
        ("Cp1", "MKS", "Cp1"),
    ])
    def test_axis_labels_carry_the_output_unit(self, key, units, expected):
        assert pv.axis_label(key, units) == expected

    def test_forces_convert_and_coefficients_do_not(self):
        lift = np.array([10.0])
        assert pv.convert(lift, "Lift", "MKS")[0] == pytest.approx(44.4822)
        assert pv.convert(lift, "Cl", "MKS")[0] == 10.0

    def test_legacy_plot_types_map_to_axes(self):
        assert pv.LEGACY_PLOT_TYPES["CL_VS_CD"] == ("Cd", "Cl")


class TestBodyAxisLoads:
    def test_body_loads_invert_the_wind_transform(self):
        case = _case()
        np.testing.assert_allclose(case.get_coefficient("Fx"), case._brf.Fx)
        np.testing.assert_allclose(case.get_coefficient("Fy"), case._brf.Fy)
        np.testing.assert_allclose(case.get_coefficient("Fz"), case._brf.Fz)
        np.testing.assert_allclose(case.get_coefficient("My"),
                                   case.pitch_moments)

    def test_wind_loads_and_tunnel_resolve(self):
        case = _case()
        np.testing.assert_allclose(case.get_coefficient("Lift"),
                                   case.lift_forces)
        assert case.get_coefficient("Q").size == 4

    def test_misaligned_loads_give_empty_not_wrong(self):
        case = _case()
        case.lift_forces = case.lift_forces[:2]
        assert case.get_coefficient("Fz").size == 0


class TestSelector:
    def test_y_axis_offers_loads_and_skips_headers(self, qapp):
        from utils.gui.widgets.filter_widgets import PlotTypeSelector
        sel = PlotTypeSelector()
        keys = [sel.cmb_y_axis.itemData(i)
                for i in range(sel.cmb_y_axis.count())]
        for key in ("Cl", "Lift", "Fz", "elem_N1", "Q", "CLa"):
            assert key in keys
        headers = [i for i, k in enumerate(keys) if k is None]
        assert headers and all(
            not sel.cmb_y_axis.model().item(i).isEnabled() for i in headers)
        assert sel.get_y_var() == "Cl"                     # default

    def test_legacy_plot_type_selects_both_axes(self, qapp):
        from utils.gui.widgets.filter_widgets import PlotTypeSelector
        sel = PlotTypeSelector()
        sel.set_plot_type("CL_VS_CD")
        assert (sel.get_x_var(), sel.get_y_var()) == ("Cd", "Cl")

    def test_y_selection_survives_a_calculator_repopulate(self, qapp):
        from utils.gui.widgets.filter_widgets import PlotTypeSelector
        sel = PlotTypeSelector()
        sel.set_y_var("Drag")
        sel.populate_custom_vars(["Cp1"])
        assert sel.get_y_var() == "Drag"
        sel.set_y_var("Cp1")
        assert sel.get_custom_y_var() == "Cp1"

    def test_panel_plots_lift_in_newtons_mks(self, qapp):
        from utils.gui.models.data_model import DataModel
        from utils.gui.views.plot_panel import PlotPanel
        model = DataModel()
        model.output_units = "MKS"
        case = _case()
        case.machs = np.full(4, 0.29)
        model.cases.add(case)
        panel = PlotPanel(model)
        panel.plot_controls.plot_selector.set_y_var("Lift")
        assert panel._get_axis_vars() == ("Alpha", "Lift")
        converted = panel._convert_y(case.lift_forces, "Lift")
        np.testing.assert_allclose(converted, case.lift_forces * 4.44822)
