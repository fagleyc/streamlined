"""Re-reducing a loaded case, and the geometry of a raster export.

Two faults the suite could not see because nothing exercised either path.

Changing a geometry re-reduces every case assigned to it, through
DataController.reprocess_case. That method reached for helpers that were
staticmethods on ProcessingWorker, a different class, so the whole
re-reduce died with AttributeError - which is what an alpha offset added
AFTER loading runs into. Loading with the offset already set worked,
because that path is the worker's.

And QGraphicsScene.itemsBoundingRect() measures HIDDEN items. The canvas
keeps a hidden crosshair readout whose stale geometry runs to tens of
thousands of pixels, so a raster export measured an aspect ratio of ~14
instead of ~0.8 and wrote a tall, nearly empty image with the figure as a
thin band across the top.
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

from utils.gui.models.case import TestCase as Case  # noqa: E402
from utils.gui.models.data_model import DataModel  # noqa: E402
from utils.gui.models.settings import AppSettings  # noqa: E402
from utils.gui.controllers.data_controller import (  # noqa: E402
    DataController, ProcessingWorker)
from utils.windtunnel.data_io import (  # noqa: E402
    MANIFEST_FILENAME, MANIFEST_SCHEMA_VERSION)

scipy_io = pytest.importorskip("scipy.io")

CHANNELS = ("Lift", "Drag", "Side", "Roll", "Pitch", "Yaw")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([sys.argv[0]])


# ── a small two-speed run directory ─────────────────────────────────────
def _write_run(directory: Path, name: str, alpha: float, mach: float,
               air_state: str, n: int = 8) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    group = {c: np.ones(n) for c in CHANNELS}
    group["Pdiff"] = np.full(n, 0.8 * (max(mach, 0.2) / 0.2) ** 2)
    group["Ptot"] = np.full(n, 12.2)
    group["Temp"] = np.full(n, 295.0)
    # Measured attitude sits slightly off the command, as it really does
    group["Alpha"] = np.full(n, alpha - 0.02)
    group["Beta"] = np.zeros(n)
    scipy_io.savemat(str(directory / name), {
        "ATE_Balance": group,
        "Time": {"Time": np.arange(n) / 50.0},
        "meta": {"run": {"balance_type": "external", "span_config": "half",
                         "air_state": air_state, "alpha": alpha,
                         "beta": 0.0},
                 "config_json": json.dumps({})},
    }, long_field_names=True)


def _run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "B52_Halfspan_C5_T0"
    points, run = [], 0
    for alpha in (-4.0, -2.0, 0.0, 2.0):
        run += 1
        name = "run_%04d_alpha_%.1f_beta_0.0_mach_0.00.mat" % (run, alpha)
        _write_run(d, name, alpha, 0.0, "AirOff")
        points.append((run, name, alpha, 0.0, "AirOff"))
    for mach in (0.2, 0.3):
        for alpha in (-4.0, -2.0, 0.0, 2.0):
            run += 1
            name = "run_%04d_alpha_%.1f_beta_0.0_mach_%.2f.mat" % (
                run, alpha, mach)
            _write_run(d, name, alpha, mach, "AirOn")
            points.append((run, name, alpha, mach, "AirOn"))
    (d / MANIFEST_FILENAME).write_text(json.dumps({
        "schema_version": MANIFEST_SCHEMA_VERSION, "config_name": d.name,
        "output_format": "mat",
        "points": [{"run_number": r, "filename": f, "alpha": a,
                    "beta": 0.0, "mach": m, "air_state": s}
                   for r, f, a, m, s in points]}), encoding="utf-8")
    return d


def _loaded_case(tmp_path: Path) -> Case:
    worker = ProcessingWorker(
        directories=[str(_run_dir(tmp_path))], balance_cal=None,
        pressure_cal=None, balance_cal_file=None, pressure_cal_file=None,
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


def _controller(case: Case, **geometry_extra):
    model = DataModel()
    model.cases.add(case)
    model.geometries['Default'].update(
        {'mac': 4.0, 'ref_area': 96.0, 'span': 24.0,
         'mrc': [0.0, 0.0, 0.0], 'units': 'IPS'})
    model.geometries['Default'].update(geometry_extra)
    controller = DataController(model, AppSettings())
    errors = []
    controller.error_occurred.connect(lambda t, m: errors.append((t, m)))
    return controller, errors


class TestReprocessAfterAGeometryChange:
    """The path an alpha offset added AFTER loading has to take."""

    def test_reprocess_does_not_raise(self, qapp, tmp_path):
        case = _loaded_case(tmp_path)
        controller, errors = _controller(case, alpha_offset=0.5)
        controller.reprocess_case(case.id)
        assert errors == [], errors

    def test_the_offset_reaches_the_reduced_attitude(self, qapp, tmp_path):
        case = _loaded_case(tmp_path)
        before = np.sort(np.ravel(case.alphas).copy())
        controller, errors = _controller(case, alpha_offset=0.5)
        controller.reprocess_case(case.id)
        assert errors == [], errors
        after = np.sort(np.ravel(case.alphas))
        np.testing.assert_allclose(after, before + 0.5, atol=1e-9)

    def test_loading_with_the_offset_matches_reprocessing_into_it(
            self, qapp, tmp_path):
        # The two routes to the same state have to agree; only the
        # loading one worked before.
        reprocessed = _loaded_case(tmp_path / "a")
        controller, errors = _controller(reprocessed, alpha_offset=0.75)
        controller.reprocess_case(reprocessed.id)
        assert errors == [], errors

        worker = ProcessingWorker(
            directories=[str(_run_dir(tmp_path / "b"))], balance_cal=None,
            pressure_cal=None, balance_cal_file=None,
            pressure_cal_file=None,
            geometry={'mac': 4.0, 'ref_area': 96.0, 'span': 24.0,
                      'mrc': [0.0, 0.0, 0.0], 'units': 'IPS',
                      'alpha_offset': 0.75},
            settings={'facility': 'SWT', 'balance_config': 'Force',
                      'cal_type': 'Linear'})
        loaded = []
        worker.signals.case_ready.connect(loaded.append)
        worker.run()

        np.testing.assert_allclose(np.sort(np.ravel(reprocessed.alphas)),
                                   np.sort(np.ravel(loaded[0].alphas)))

    def test_the_speed_setpoints_survive(self, qapp, tmp_path):
        case = _loaded_case(tmp_path)
        controller, errors = _controller(case, alpha_offset=0.5)
        controller.reprocess_case(case.id)
        assert errors == [], errors
        assert sorted(set(np.ravel(case.speeds).tolist())) == [0.2, 0.3]
        assert case.speed_unit == 'mach'
        assert sorted(set(np.ravel(case.point_machs).tolist())) == [0.2, 0.3]

    def test_the_commanded_attitude_survives(self, qapp, tmp_path):
        case = _loaded_case(tmp_path)
        controller, errors = _controller(case, alpha_offset=0.5)
        controller.reprocess_case(case.id)
        assert errors == [], errors
        assert sorted(set(np.ravel(case.point_alphas).tolist())) == [
            -4.0, -2.0, 0.0, 2.0]

    def test_a_plain_reprocess_changes_nothing(self, qapp, tmp_path):
        case = _loaded_case(tmp_path)
        before = np.sort(np.ravel(case.alphas).copy())
        controller, errors = _controller(case)
        controller.reprocess_case(case.id)
        assert errors == [], errors
        np.testing.assert_allclose(np.sort(np.ravel(case.alphas)), before)


class TestSharedHelpersAreReachable:
    """Both reducing paths reach the same helpers."""

    def test_they_are_module_level_functions(self):
        from utils.gui.controllers import data_controller as dc
        assert callable(dc.attach_speed_setpoints)
        assert callable(dc.attach_nominal_attitude)

    def test_the_controller_is_not_missing_them(self, qapp, tmp_path):
        # The exact failure: reprocess_case reached through self for a
        # helper that lived on the worker.
        import inspect
        from utils.gui.controllers import data_controller as dc
        source = inspect.getsource(dc.DataController.reprocess_case)
        assert 'self._attach_speed_setpoints' not in source
        assert 'self._attach_nominal_attitude' not in source


# ── export geometry ─────────────────────────────────────────────────────
def _panel_with_a_plot(qapp):
    from utils.gui.views.plot_panel import PlotPanel

    model = DataModel()
    case = Case(id="c1", name="B52_Halfspan_Baseline")
    case.alphas = np.arange(-4.0, 5.0, 1.0)
    case.betas = np.zeros(9)
    case.Cl = 0.1 * case.alphas + 0.3
    case.Cd = 0.02 + 0.04 * case.Cl ** 2
    case.machs = np.full(9, 0.2)
    model.cases.add(case)
    panel = PlotPanel(model)
    panel.resize(1100, 750)
    panel.show()
    panel._update_plot()
    qapp.processEvents()
    return panel


class TestExportRectIgnoresHiddenItems:
    def test_a_hidden_giant_does_not_widen_the_rect(self, qapp):
        from PyQt6.QtWidgets import QGraphicsScene, QGraphicsRectItem
        from PyQt6.QtCore import QRectF
        from utils.gui.widgets.save_image_dialog import SaveImageDialog

        scene = QGraphicsScene()
        shown = QGraphicsRectItem(QRectF(0, 0, 800, 600))
        scene.addItem(shown)
        hidden = QGraphicsRectItem(QRectF(0, 0, 700, 16000))
        hidden.setVisible(False)
        scene.addItem(hidden)

        # What the old code measured, and what it should measure
        assert scene.itemsBoundingRect().height() > 15000
        rect = SaveImageDialog._visible_items_rect(scene)
        assert rect.height() == pytest.approx(600, abs=2)
        assert rect.width() == pytest.approx(800, abs=2)

    def test_an_empty_scene_falls_back(self, qapp):
        from PyQt6.QtWidgets import QGraphicsScene
        from utils.gui.widgets.save_image_dialog import SaveImageDialog

        scene = QGraphicsScene()
        rect = SaveImageDialog._visible_items_rect(scene)
        assert rect == scene.itemsBoundingRect()

    def test_visible_axis_labels_are_kept(self, qapp):
        # The labels are siblings of the plot item and sit outside its
        # rect; measuring the plot item alone would crop them.
        panel = _panel_with_a_plot(qapp)
        canvas = panel.plot_canvas
        if not hasattr(canvas, 'plot_item'):
            pytest.skip("pyqtgraph canvas not in use")
        from utils.gui.widgets.save_image_dialog import SaveImageDialog

        scene = canvas.plot_widget.scene()
        rect = SaveImageDialog._visible_items_rect(scene)
        plot_rect = canvas.plot_item.sceneBoundingRect()
        assert rect.contains(plot_rect.adjusted(1, 1, -1, -1))
        assert rect.width() >= plot_rect.width()


class TestExportedImageFillsItsFrame:
    def test_the_aspect_is_sane_and_the_ink_fills_it(self, qapp, tmp_path):
        from PyQt6.QtGui import QImage
        from utils.gui.widgets.save_image_dialog import SaveImageDialog

        panel = _panel_with_a_plot(qapp)
        canvas = panel.plot_canvas
        if not hasattr(canvas, 'plot_item'):
            pytest.skip("pyqtgraph canvas not in use")

        dialog = SaveImageDialog(
            plot_items=list(canvas._plot_items),
            plot_data=list(canvas._plot_data),
            plot_item=canvas.plot_item, plot_widget=canvas.plot_widget,
            show_grid=True, show_legend=True, parent=panel)

        path = tmp_path / "fig.png"
        dialog._export_to_file(str(path))
        image = QImage(str(path))
        assert not image.isNull()

        aspect = image.height() / image.width()
        assert 0.3 < aspect < 3.0, (
            "a plot is not 14x taller than it is wide: got %dx%d"
            % (image.width(), image.height()))

        # The figure has to actually FILL the frame it was given
        background = image.pixelColor(0, 0)
        step = max(1, min(image.width(), image.height()) // 200)
        ys = [y for y in range(0, image.height(), step)
              for x in range(0, image.width(), step)
              if image.pixelColor(x, y) != background]
        assert ys, "exported image is blank"
        covered = (max(ys) - min(ys)) / image.height()
        assert covered > 0.8, (
            "figure fills only %.0f%% of the image height" % (covered * 100))
