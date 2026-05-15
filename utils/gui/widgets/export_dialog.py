"""
Export Dialog
=============

Consolidated export dialog for all data export formats.
"""

from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox,
    QComboBox, QLabel, QPushButton, QLineEdit, QFileDialog,
    QGridLayout, QDialogButtonBox, QToolButton, QFrame, QWidget
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

        # --- Advanced (collapsible): Data to Include (HDF5/MAT only) ---
        # Toggle button shows / hides the inner widget.  Collapsed by
        # default so the dialog stays compact for everyday export.
        self.btn_advanced = QToolButton()
        self.btn_advanced.setText("Advanced - Data to Include")
        self.btn_advanced.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.btn_advanced.setArrowType(Qt.ArrowType.RightArrow)
        self.btn_advanced.setCheckable(True)
        self.btn_advanced.setChecked(False)
        self.btn_advanced.setStyleSheet(
            'QToolButton { border: none; font-weight: bold; }')
        self.btn_advanced.toggled.connect(self._on_advanced_toggled)
        layout.addWidget(self.btn_advanced)

        self.advanced_panel = QFrame()
        self.advanced_panel.setFrameShape(QFrame.Shape.StyledPanel)
        data_layout = QGridLayout(self.advanced_panel)
        data_layout.setContentsMargins(12, 8, 12, 8)

        self.chk_averaged = QCheckBox("Averaged Coefficients / Forces")
        self.chk_averaged.setChecked(True)
        self.chk_averaged.setEnabled(False)  # always included
        data_layout.addWidget(self.chk_averaged, 0, 0)

        self.chk_metadata = QCheckBox("Metadata / Calibration / Geometry")
        self.chk_metadata.setChecked(True)
        data_layout.addWidget(self.chk_metadata, 0, 1)

        self.chk_air_on = QCheckBox("Air-On Raw Data")
        self.chk_air_on.setChecked(False)
        data_layout.addWidget(self.chk_air_on, 1, 0)

        self.chk_air_off = QCheckBox("Air-Off Raw Data")
        self.chk_air_off.setChecked(False)
        data_layout.addWidget(self.chk_air_off, 1, 1)

        self.chk_reduced = QCheckBox("Reduced BRF/WRF Data")
        self.chk_reduced.setChecked(False)
        data_layout.addWidget(self.chk_reduced, 2, 0)

        self.chk_coefficients_ts = QCheckBox("Coefficient Time-Series")
        self.chk_coefficients_ts.setChecked(False)
        data_layout.addWidget(self.chk_coefficients_ts, 2, 1)

        self.chk_tunnel_ts = QCheckBox("Tunnel Conditions Time-Series")
        self.chk_tunnel_ts.setChecked(False)
        data_layout.addWidget(self.chk_tunnel_ts, 3, 0)

        self.chk_elements = QCheckBox("Balance Elements (tared)")
        self.chk_elements.setChecked(False)
        data_layout.addWidget(self.chk_elements, 3, 1)

        self.advanced_panel.setVisible(False)
        layout.addWidget(self.advanced_panel)

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

    def _on_advanced_toggled(self, checked: bool):
        """Show / hide the Advanced data-to-include panel."""
        self.advanced_panel.setVisible(checked)
        self.btn_advanced.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        # Force the dialog to resize to fit the new content
        self.adjustSize()

    def _on_format_changed(self):
        """Enable/disable extended data options based on format."""
        idx = self.cmb_format.currentIndex()
        supports_extended = _FORMATS[idx][3]

        # Advanced toggle and panel only meaningful for formats that
        # support extended data (HDF5 / MATLAB).
        self.btn_advanced.setEnabled(supports_extended)
        self.advanced_panel.setEnabled(supports_extended)
        if not supports_extended and self.btn_advanced.isChecked():
            self.btn_advanced.setChecked(False)

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
            'include_metadata': (self.chk_metadata.isChecked()
                                 if supports_extended else False),
            'include_air_on': (self.chk_air_on.isChecked()
                               if supports_extended else False),
            'include_air_off': (self.chk_air_off.isChecked()
                                if supports_extended else False),
            'include_reduced': (self.chk_reduced.isChecked()
                                if supports_extended else False),
            'include_coefficients_ts': (
                self.chk_coefficients_ts.isChecked()
                if supports_extended else False),
            'include_tunnel_ts': (self.chk_tunnel_ts.isChecked()
                                  if supports_extended else False),
            'include_elements': (self.chk_elements.isChecked()
                                 if supports_extended else False),
        }
