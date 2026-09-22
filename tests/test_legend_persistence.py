"""A moved legend survives a redraw, and Clear All empties the case list.

The canvas legend is draggable, but every redraw destroys and rebuilds
it - and adding data is a redraw. The rebuilt legend was created at the
fixed default offset, so a legend the user had moved reappeared in the
corner. That reads as a second legend arriving rather than the original
one moving back.

The position is now remembered across the rebuild through autoAnchor,
the same public call a drag makes, so it also survives a resize.
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
pg = pytest.importorskip("pyqtgraph")

from utils.gui.models.case import TestCase as Case  # noqa: E402
from utils.gui.models.data_model import DataModel  # noqa: E402
from utils.gui.views.plot_panel import PlotPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([sys.argv[0]])


def _case(k, name):
    case = Case(id="c%d" % k, name=name)
    case.alphas = np.arange(-4.0, 5.0, 1.0)
    case.betas = np.zeros(9)
    case.Cl = (0.086 + 0.002 * k) * case.alphas + 0.35
    case.Cd = 0.022 + 0.05 * case.Cl ** 2
    case.machs = np.full(9, 0.2)
    return case


def _drawn_panel(qapp):
    model = DataModel()
    model.cases.add(_case(0, "B52_Halfspan_Baseline"))
    panel = PlotPanel(model)
    panel.resize(1000, 700)
    panel.show()
    panel._update_plot()
    qapp.processEvents()
    if not hasattr(panel.plot_canvas, 'plot_item'):
        pytest.skip("pyqtgraph canvas not in use")
    return model, panel


def _legend_count(canvas):
    return sum(1 for item in canvas.plot_widget.scene().items()
               if isinstance(item, pg.LegendItem))


class TestMovedLegendSurvivesARedraw:
    def test_adding_a_case_leaves_the_legend_where_it_was(self, qapp):
        model, panel = _drawn_panel(qapp)
        canvas = panel.plot_canvas
        assert canvas._legend is not None

        canvas._legend.autoAnchor(pg.Point(600.0, 470.0))
        qapp.processEvents()
        moved = pg.Point(canvas._legend.pos())

        model.cases.add(_case(1, "B52_Halfspan_C5_T0"))
        panel._update_plot()
        qapp.processEvents()

        now = canvas._legend.pos()
        assert abs(now.x() - moved.x()) < 1.0
        assert abs(now.y() - moved.y()) < 1.0

    def test_only_one_legend_exists_afterwards(self, qapp):
        model, panel = _drawn_panel(qapp)
        canvas = panel.plot_canvas
        canvas._legend.autoAnchor(pg.Point(600.0, 470.0))
        qapp.processEvents()

        model.cases.add(_case(1, "B52_Halfspan_C5_T0"))
        panel._update_plot()
        qapp.processEvents()

        assert _legend_count(canvas) == 1

    def test_the_rebuilt_legend_lists_every_trace(self, qapp):
        model, panel = _drawn_panel(qapp)
        canvas = panel.plot_canvas
        canvas._legend.autoAnchor(pg.Point(600.0, 470.0))
        qapp.processEvents()

        model.cases.add(_case(1, "B52_Halfspan_C5_T0"))
        panel._update_plot()
        qapp.processEvents()

        assert len(canvas._legend.items) == 2

    def test_an_untouched_legend_still_starts_in_the_corner(self, qapp):
        # Nothing was dragged, so nothing is remembered: the default
        # placement has to be unchanged.
        model, panel = _drawn_panel(qapp)
        canvas = panel.plot_canvas
        first = canvas._legend.pos()
        model.cases.add(_case(1, "B52_Halfspan_C5_T0"))
        panel._update_plot()
        qapp.processEvents()
        now = canvas._legend.pos()
        assert abs(now.x() - first.x()) < 1.0
        assert abs(now.y() - first.y()) < 1.0

    def test_the_position_survives_toggling_the_legend_off_and_on(self,
                                                                  qapp):
        model, panel = _drawn_panel(qapp)
        canvas = panel.plot_canvas
        canvas._legend.autoAnchor(pg.Point(600.0, 470.0))
        qapp.processEvents()
        moved = pg.Point(canvas._legend.pos())

        canvas.remove_legend()
        canvas.add_legend()
        qapp.processEvents()

        now = canvas._legend.pos()
        assert abs(now.x() - moved.x()) < 1.0
        assert abs(now.y() - moved.y()) < 1.0


class TestClearAll:
    def test_the_button_exists_and_emits(self, qapp):
        from utils.gui.widgets.case_list import CaseListWidget
        widget = CaseListWidget()
        fired = []
        widget.clear_all_requested.connect(lambda: fired.append(True))
        widget.btn_clear_all.click()
        assert fired == [True]

    def test_clearing_empties_the_model(self, qapp):
        model = DataModel()
        model.cases.add(_case(0, "A"))
        model.cases.add(_case(1, "B"))
        assert len(list(model.cases)) == 2
        model.clear_all()
        assert len(list(model.cases)) == 0

    def test_the_panel_confirms_before_clearing(self, qapp, monkeypatch):
        from PyQt6.QtWidgets import QMessageBox
        from utils.gui.views.data_panel import DataPanel
        from utils.gui.models.settings import AppSettings

        model = DataModel()
        model.cases.add(_case(0, "A"))
        panel = DataPanel(model, AppSettings())

        asked = []
        monkeypatch.setattr(
            QMessageBox, "question",
            lambda *a, **k: (asked.append(True),
                             QMessageBox.StandardButton.No)[1])
        panel._on_clear_all_requested()
        assert asked == [True], "Clear All must ask first"
        assert len(list(model.cases)) == 1, "declining must keep the cases"

    def test_confirming_clears(self, qapp, monkeypatch):
        from PyQt6.QtWidgets import QMessageBox
        from utils.gui.views.data_panel import DataPanel
        from utils.gui.models.settings import AppSettings

        model = DataModel()
        model.cases.add(_case(0, "A"))
        model.cases.add(_case(1, "B"))
        panel = DataPanel(model, AppSettings())

        monkeypatch.setattr(QMessageBox, "question",
                            lambda *a, **k: QMessageBox.StandardButton.Yes)
        panel._on_clear_all_requested()
        assert len(list(model.cases)) == 0

    def test_it_does_not_prompt_when_there_is_nothing_to_clear(
            self, qapp, monkeypatch):
        from PyQt6.QtWidgets import QMessageBox
        from utils.gui.views.data_panel import DataPanel
        from utils.gui.models.settings import AppSettings

        model = DataModel()
        panel = DataPanel(model, AppSettings())
        asked = []
        monkeypatch.setattr(
            QMessageBox, "question",
            lambda *a, **k: (asked.append(True),
                             QMessageBox.StandardButton.Yes)[1])
        panel._on_clear_all_requested()
        assert asked == []
