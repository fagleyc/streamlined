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


class TestThemeWiring:
    """Live theming: View menu tools, header toolbar, theme chip, and the
    legacy ``DarkTheme`` names following the active palette."""

    @pytest.fixture(autouse=True)
    def _no_persist(self, app):
        from utils.gui.utils import themekit
        m = themekit.manager()
        saved, m._settings = m._settings, None   # never touch the registry
        yield
        m.apply(themekit.DEFAULT_THEME, scale=1.0, density="comfortable",
                persist=False, force=True)
        m._settings = saved

    def test_view_menu_carries_theme_tools(self, app):
        from utils.gui.utils import themekit
        paths = [p for p, _a in themekit.collect_actions(app.main_window)]
        assert any(p.startswith("View ▸ Theme ▸ USAFA Night") for p in paths)
        assert "View ▸ Appearance" in paths
        assert "View ▸ Command Palette" in paths
        assert "View ▸ Plot View" in paths            # the originals stay

    def test_dark_theme_names_follow_the_active_palette(self, app):
        from utils.gui.utils import themekit
        from utils.gui.utils.themes import DarkTheme, get_plot_style
        themekit.manager().apply("usafa_day", persist=False)
        day = themekit.PALETTES["usafa_day"]
        assert DarkTheme.BACKGROUND == day["BG"]
        assert DarkTheme.TEXT_SECONDARY == day["TEXT_DIM"]
        assert DarkTheme.PLOT_AXES == day["PLOT_BG"]
        assert get_plot_style()["axes.facecolor"] == day["PLOT_BG"]
        assert app.main_window.theme_toggle.text().endswith("USAFA Day")

    def test_live_switch_restyles_open_panels(self, app):
        from utils.gui.utils import themekit
        themekit.manager().apply("usafa_night", persist=False, force=True)
        frame = app.main_window.data_panel.cal_section
        themekit.manager().apply("classic_light", persist=False)
        light = themekit.PALETTES["classic_light"]
        assert light["BG_LIGHT"] in frame.styleSheet()
        fig = app.main_window.plot_panel.findChildren(
            __import__("matplotlib.backends.backend_qtagg",
                       fromlist=["FigureCanvasQTAgg"]).FigureCanvasQTAgg)
        for canvas in fig:
            from matplotlib.colors import to_hex
            assert to_hex(canvas.figure.get_facecolor()) == light["PLOT_FIG"]

    def test_icons_are_theme_tokens_unless_pinned(self, app):
        from utils.gui.utils import themekit
        from utils.gui.utils.icons import Icons
        themekit.manager().apply("usafa_day", persist=False)
        img = Icons.add().pixmap(24, 24).toImage()
        inked = {img.pixelColor(x, y).name() for x in range(24)
                 for y in range(24) if img.pixelColor(x, y).alpha() == 255}
        assert themekit.PALETTES["usafa_day"]["TEXT"] in inked
        pinned = Icons.add("#ff0000").pixmap(24, 24).toImage()
        assert any(pinned.pixelColor(x, y).name() == "#ff0000"
                   for x in range(24) for y in range(24))
