"""
Help menu / About dialog regression tests.

The About dialog previously raised ``NameError: name 'i' is not defined``
when triggered: the changelog HTML was embedded in an f-string containing
literal ``{i}`` placeholders (v1.2.7 custom-calculator notes), which
Python evaluated as format expressions. These tests guard against any
regression of that class of bug by actually constructing the dialog and
firing the menu action offscreen.
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

from PyQt6.QtWidgets import QApplication  # noqa: E402

from utils.gui import (  # noqa: E402
    __version__, APP_NAME, AUTHOR, CONTACT, VERSION_HISTORY,
)
from utils.gui.views.dialogs import AboutDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_about_metadata():
    assert APP_NAME == "Streamlined"
    assert AUTHOR == "C. Fagley"
    assert CONTACT == "casey.fagley@afacademy.af.edu"
    assert VERSION_HISTORY[0][0] == __version__  # newest first
    for entry in VERSION_HISTORY:
        version, iso_date, note = entry
        assert version and iso_date and note


def test_about_dialog_constructs(qapp):
    # Regression: this used to raise NameError from f-string placeholders.
    dialog = AboutDialog()
    assert APP_NAME in dialog.windowTitle()
    dialog.deleteLater()


def test_about_action_does_not_raise(qapp, monkeypatch):
    """Triggering Help -> About on the real main window must not raise."""
    from utils.gui.models.data_model import DataModel
    from utils.gui.models.settings import AppSettings
    from utils.gui.views.main_window import MainWindow

    shown = []
    monkeypatch.setattr(AboutDialog, "exec",
                        lambda self: shown.append(self) or 0)

    window = MainWindow(DataModel(), AppSettings())
    try:
        window.action_about.trigger()
        assert len(shown) == 1
        assert isinstance(shown[0], AboutDialog)
    finally:
        window.close()
        window.deleteLater()


def test_help_menu_has_documentation_action(qapp):
    from utils.gui.models.data_model import DataModel
    from utils.gui.models.settings import AppSettings
    from utils.gui.views.main_window import MainWindow

    window = MainWindow(DataModel(), AppSettings())
    try:
        assert window.action_documentation.text() == "&Documentation"
        assert APP_NAME in window.action_about.text()
    finally:
        window.close()
        window.deleteLater()


def test_documentation_file_exists():
    docs = ROOT / "docs" / "index.html"
    assert docs.is_file()
    html = docs.read_text(encoding="utf-8")
    assert APP_NAME in html
    assert CONTACT in html
