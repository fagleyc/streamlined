"""The application has to survive being built.

A launch-crash regression (``self.data_controller`` where the attribute
is ``self.controller``) shipped despite a green suite, because nothing
exercised :meth:`WindTunnelApp.create_components` - the method that
constructs the model, controller and window and wires every signal
between them. A typo in any of those connect() lines raises
AttributeError before the window is ever shown.

These tests build the app the way ``streamlined.py`` does and then
prove the wiring is live, not merely that the import succeeded.
"""
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("PyQt6")

from utils.gui.main import WindTunnelApp  # noqa: E402


@pytest.fixture(scope="module")
def app():
    """A fully constructed application, as produced at launch.

    Module-scoped: building several full widget trees (each with its own
    matplotlib canvases) in one process trips a GC-time access violation
    in the Qt/matplotlib teardown path on Windows. One app is also
    exactly what the launch path produces.
    """
    application = WindTunnelApp()
    application.initialize()
    application.create_components()
    return application


class TestStartupWiring:
    def test_create_components_succeeds(self, app):
        """The launch path builds without raising."""
        assert app.controller is not None
        assert app.main_window is not None

    def test_controller_reachable_from_window(self, app):
        """The window keeps the controller the reprocess paths use."""
        assert app.main_window.data_controller is app.controller

    def test_balance_type_signal_reaches_the_panel(self, app):
        """External runs have to be able to grey out the .vol input.

        Emitting is the point: a mis-typed connect() only fails when the
        signal actually fires.
        """
        cal = app.main_window.data_panel.cal_section
        app.controller.balance_type_detected.emit("external")
        assert cal.btn_balance.isEnabled() is False

        app.controller.balance_type_detected.emit("internal")
        assert cal.btn_balance.isEnabled() is True


class TestPanelSignalsConnectedOnce:
    """`set_balance_type` was pasted into the middle of
    `_connect_signals`, so the panel's model wiring only happened when a
    balance type arrived - and re-connected on every later detection.
    """

    def test_model_signals_survive_repeated_detection(self, app):
        model = app.model
        before = model.receivers(model.cases_changed)

        for _ in range(3):
            app.controller.balance_type_detected.emit("external")

        assert model.receivers(model.cases_changed) == before
