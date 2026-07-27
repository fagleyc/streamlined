"""MultiDirectoryDialog (root + file-system tree) regression tests.

The dialog was reworked from a plain list + "Browse (Multi-select)" popup into
a persistent root directory plus a live QFileSystemModel tree.  These tests
cover the new tree wiring and pin the public API data_panel.py depends on:
constructor signature, get_directories() and the `recursive` property.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QDir  # noqa: E402
from PyQt6.QtWidgets import QApplication, QAbstractItemView  # noqa: E402
from PyQt6.QtCore import QItemSelectionModel  # noqa: E402

from utils.gui.views import MultiDirectoryDialog, get_multiple_directories  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _FakeSettings:
    """Stand-in for AppSettings holding only the persisted root."""

    def __init__(self, root: str = ""):
        self.data_root_directory = root


@pytest.fixture
def tree_root(tmp_path):
    """A small directory tree: root/{runA,runB,runA/sweep1}."""
    (tmp_path / "runA" / "sweep1").mkdir(parents=True)
    (tmp_path / "runB").mkdir()
    (tmp_path / "runA" / "notes.txt").write_text("not a directory")
    return tmp_path


def _norm(path) -> str:
    return str(Path(path))


# ---------------------------------------------------------------------------
# Tree wiring
# ---------------------------------------------------------------------------

class TestDirectoryTree:
    def test_model_roots_at_initial_dir(self, qapp, tree_root):
        dlg = MultiDirectoryDialog(None, str(tree_root))
        assert _norm(dlg.fs_model.rootPath()) == _norm(tree_root)
        assert _norm(dlg.fs_model.filePath(dlg.tree.rootIndex())) == _norm(tree_root)
        assert dlg.root_directory == _norm(tree_root)
        assert _norm(dlg.txt_root.text()) == _norm(tree_root)

    def test_persisted_root_wins_over_initial_dir(self, qapp, tree_root, tmp_path_factory):
        other = tmp_path_factory.mktemp("elsewhere")
        dlg = MultiDirectoryDialog(None, str(other), _FakeSettings(str(tree_root)))
        assert dlg.root_directory == _norm(tree_root)

    def test_missing_persisted_root_falls_back_to_initial_dir(self, qapp, tree_root):
        settings = _FakeSettings(str(tree_root / "does_not_exist"))
        dlg = MultiDirectoryDialog(None, str(tree_root), settings)
        assert dlg.root_directory == _norm(tree_root)

    def test_directories_only_filter(self, qapp, tree_root):
        dlg = MultiDirectoryDialog(None, str(tree_root))
        flt = dlg.fs_model.filter()
        assert flt & QDir.Filter.Dirs
        assert flt & QDir.Filter.NoDotAndDotDot
        assert not (flt & QDir.Filter.Files)

    def test_size_type_date_columns_hidden(self, qapp, tree_root):
        dlg = MultiDirectoryDialog(None, str(tree_root))
        assert not dlg.tree.isColumnHidden(0)
        for column in (1, 2, 3):
            assert dlg.tree.isColumnHidden(column)

    def test_extended_selection_mode(self, qapp, tree_root):
        dlg = MultiDirectoryDialog(None, str(tree_root))
        assert dlg.tree.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection
        assert dlg.dir_list.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection
        assert dlg.tree.expandsOnDoubleClick()

    def test_changing_root_rerootes_in_place(self, qapp, tree_root):
        dlg = MultiDirectoryDialog(None, str(tree_root))
        model, tree = dlg.fs_model, dlg.tree

        dlg._set_root(str(tree_root / "runA"))

        assert dlg.fs_model is model  # same model/view, just re-rooted
        assert dlg.tree is tree
        assert _norm(dlg.fs_model.filePath(tree.rootIndex())) == _norm(tree_root / "runA")
        assert dlg.root_directory == _norm(tree_root / "runA")

    def test_invalid_root_is_ignored(self, qapp, tree_root):
        dlg = MultiDirectoryDialog(None, str(tree_root))
        dlg._set_root(str(tree_root / "nope"))
        assert dlg.root_directory == _norm(tree_root)


# ---------------------------------------------------------------------------
# Adding from the tree
# ---------------------------------------------------------------------------

class TestAddSelected:
    def _select(self, dlg, *paths):
        selection = dlg.tree.selectionModel()
        selection.clearSelection()
        for path in paths:
            index = dlg.fs_model.index(str(path))
            assert index.isValid(), path
            selection.select(index, QItemSelectionModel.SelectionFlag.Select
                             | QItemSelectionModel.SelectionFlag.Rows)

    def test_add_selected_adds_every_selected_directory(self, qapp, tree_root):
        dlg = MultiDirectoryDialog(None, str(tree_root))
        self._select(dlg, tree_root / "runA", tree_root / "runB")
        dlg._add_selected_from_tree()

        assert sorted(dlg.get_directories()) == sorted(
            [_norm(tree_root / "runA"), _norm(tree_root / "runB")]
        )
        assert dlg.dir_list.count() == 2

    def test_add_selected_deduplicates(self, qapp, tree_root):
        dlg = MultiDirectoryDialog(None, str(tree_root))
        self._select(dlg, tree_root / "runA")
        dlg._add_selected_from_tree()
        dlg._add_selected_from_tree()

        assert dlg.get_directories() == [_norm(tree_root / "runA")]
        assert dlg.dir_list.count() == 1

    def test_nested_directory_can_be_added_directly(self, qapp, tree_root):
        dlg = MultiDirectoryDialog(None, str(tree_root))
        self._select(dlg, tree_root / "runA" / "sweep1")
        dlg._add_selected_from_tree()
        assert dlg.get_directories() == [_norm(tree_root / "runA" / "sweep1")]


# ---------------------------------------------------------------------------
# List management + public API
# ---------------------------------------------------------------------------

class TestListManagement:
    def test_add_remove_clear(self, qapp, tree_root):
        dlg = MultiDirectoryDialog(None, str(tree_root))

        assert dlg._add_directory(str(tree_root / "runA")) is True
        assert dlg._add_directory(str(tree_root / "runB")) is True
        assert dlg._add_directory(str(tree_root / "runA")) is False  # de-duped
        dlg._update_status()
        assert dlg.dir_list.count() == 2

        dlg.dir_list.item(0).setSelected(True)
        dlg._remove_selected()
        assert dlg.get_directories() == [_norm(tree_root / "runB")]
        assert dlg.dir_list.count() == 1

        dlg._clear_all()
        assert dlg.get_directories() == []
        assert dlg.dir_list.count() == 0

    def test_get_directories_returns_a_copy(self, qapp, tree_root):
        dlg = MultiDirectoryDialog(None, str(tree_root))
        dlg._add_directory(str(tree_root / "runA"))
        dirs = dlg.get_directories()
        dirs.append("mutated")
        assert dlg.get_directories() == [_norm(tree_root / "runA")]

    def test_ok_button_enable_logic(self, qapp, tree_root):
        dlg = MultiDirectoryDialog(None, str(tree_root))
        assert not dlg.btn_ok.isEnabled()
        assert dlg.lbl_status.text() == "No directories selected"

        dlg._add_directory(str(tree_root / "runA"))
        dlg._update_status()
        assert dlg.btn_ok.isEnabled()
        assert dlg.lbl_status.text() == "1 directory selected"

        dlg._add_directory(str(tree_root / "runB"))
        dlg._update_status()
        assert dlg.lbl_status.text() == "2 directories selected"

        dlg._clear_all()
        assert not dlg.btn_ok.isEnabled()
        assert dlg.lbl_status.text() == "No directories selected"

    def test_recursive_property_and_checkbox_kept(self, qapp, tree_root):
        dlg = MultiDirectoryDialog(None, str(tree_root))
        assert dlg.chk_recursive.isChecked()
        assert dlg.recursive is True
        dlg.chk_recursive.setChecked(False)
        assert dlg.recursive is False

    def test_public_surface_intact(self, qapp, tree_root):
        # data_panel.py calls MultiDirectoryDialog(self, last_dir) with two args
        dlg = MultiDirectoryDialog(None, str(tree_root))
        assert callable(dlg.get_directories)
        assert isinstance(dlg.recursive, bool)
        assert callable(get_multiple_directories)
