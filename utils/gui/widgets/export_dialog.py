"""
Export Dialog
=============

Consolidated export dialog for all data export formats.
"""

from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QCheckBox,
    QComboBox, QLabel, QPushButton, QLineEdit, QFileDialog,
    QDialogButtonBox
)
from PyQt6.QtCore import Qt

from ..utils.themes import DarkTheme


# Format definitions: (display name, key, file filter, supports extended)
# Order chosen for usability: most common scientific formats first,
# legacy COE last.
_FORMATS = [
    ("Excel", "excel", "Excel Files (*.xlsx);;All Files (*.*)", False),
    ("MATLAB (.mat)", "mat", "MAT Files (*.mat);;All Files (*.*)", True),
    ("HDF5", "hdf5", "HDF5 Files (*.h5 *.hdf5);;All Files (*.*)", True),
    ("CSV", "csv", "CSV Files (*.csv);;All Files (*.*)", False),
    ("COE (legacy Reduce2)", "coe", "", False),
]


class ExportDialog(QDialog):
    """Dialog for configuring and executing data export."""

    def __init__(self, case_names=None, current_case_name=None,
                 last_directory="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Data")
        self.setMinimumWidth(500)
        self._last_directory = last_directory
        self._case_names = case_names or []
        self._current_case_name = current_case_name
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # --- Format selection ---
        fmt_layout = QHBoxLayout()
        fmt_layout.addWidget(QLabel("Format:"))
        self.cmb_format = QComboBox()
        for display, key, filt, ext in _FORMATS:
            self.cmb_format.addItem(display, key)
        self.cmb_format.currentIndexChanged.connect(self._on_format_changed)
        fmt_layout.addWidget(self.cmb_format, stretch=1)
        layout.addLayout(fmt_layout)

        # --- Case selection ---
        case_layout = QHBoxLayout()
        case_layout.addWidget(QLabel("Cases:"))
        self.cmb_cases = QComboBox()
        self.cmb_cases.addItem("All Cases", "all")
        for name in self._case_names:
            self.cmb_cases.addItem(name, name)
        if self._current_case_name:
            idx = self.cmb_cases.findText(self._current_case_name)
            if idx >= 0:
                self.cmb_cases.setCurrentIndex(idx)
        case_layout.addWidget(self.cmb_cases, stretch=1)
        layout.addLayout(case_layout)

        # --- Unsteady data option (HDF5 / MAT only) ---
        # A single toggle: include the full per-sample time history for
        # every channel, or just the averaged (steady-state) values.
        # The averaged coefficients, forces, tunnel conditions, raw
        # channels, geometry, and metadata are ALWAYS written.  This
        # only adds the time-series ("unsteady") data on top.
        self.chk_unsteady = QCheckBox(
            "Include all unsteady (time-series) data")
        self.chk_unsteady.setChecked(False)
        self.chk_unsteady.setToolTip(
            "Save the full per-sample time history for every channel "
            "(raw signals, coefficients, tunnel conditions, forces).\n"
            "Averaged/steady-state values and metadata are always "
            "included regardless of this setting.\n"
            "Produces substantially larger files.")
        layout.addWidget(self.chk_unsteady)

        # --- File path ---
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("File:"))
        self.txt_filepath = QLineEdit()
        self.txt_filepath.setPlaceholderText("Select output file...")
        path_layout.addWidget(self.txt_filepath, stretch=1)
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._browse)
        path_layout.addWidget(self.btn_browse)
        layout.addLayout(path_layout)

        # --- Buttons ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Export")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Initial state
        self._on_format_changed()

    def _on_format_changed(self):
        """Enable/disable the unsteady option based on format."""
        idx = self.cmb_format.currentIndex()
        supports_extended = _FORMATS[idx][3]

        # The unsteady time-series option is only meaningful for the
        # container formats (HDF5 / MATLAB).
        self.chk_unsteady.setEnabled(supports_extended)
        if not supports_extended:
            self.chk_unsteady.setChecked(False)

        # Update placeholder to clarify file-vs-directory for COE
        fmt_key = self.cmb_format.currentData()
        if fmt_key == 'coe':
            self.txt_filepath.setPlaceholderText(
                "Select output directory (one .COE file per case/beta)...")
        else:
            self.txt_filepath.setPlaceholderText("Select output file...")

        # Clear filepath when format changes
        self.txt_filepath.clear()

    def _browse(self):
        """Open file save dialog (or directory picker for COE)."""
        idx = self.cmb_format.currentIndex()
        fmt_name = _FORMATS[idx][0]
        fmt_key = self.cmb_format.currentData()

        if fmt_key == 'coe':
            # COE is multi-file; choose a directory instead
            out_dir = QFileDialog.getExistingDirectory(
                self, "Export COE Files - choose output directory",
                self._last_directory)
            if out_dir:
                self.txt_filepath.setText(out_dir)
                self._last_directory = out_dir
            return

        file_filter = _FORMATS[idx][2]
        filepath, _ = QFileDialog.getSaveFileName(
            self, f"Export to {fmt_name}",
            self._last_directory, file_filter
        )
        if filepath:
            self.txt_filepath.setText(filepath)
            self._last_directory = str(Path(filepath).parent)

    def get_export_config(self) -> dict:
        """Return the export configuration dict."""
        idx = self.cmb_format.currentIndex()
        supports_extended = _FORMATS[idx][3]
        return {
            'format': self.cmb_format.currentData(),
            'case_scope': self.cmb_cases.currentData(),
            'filepath': self.txt_filepath.text(),
            # Single unsteady flag: include the full time-series for
            # every channel.  Averaged data + metadata are always saved.
            'include_unsteady': (self.chk_unsteady.isChecked()
                                 if supports_extended else False),
        }
