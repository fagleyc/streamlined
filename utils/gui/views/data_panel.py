"""
Data Panel
==========

Panel for data management, file loading, and calibration.
"""

from pathlib import Path
from typing import Optional, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFrame, QGroupBox, QLineEdit, QFileDialog, QMessageBox,
    QProgressBar
)
from PyQt6.QtCore import pyqtSignal, Qt

from ..models.data_model import DataModel
from ..models.settings import AppSettings
from ..widgets.case_list import CaseListWidget
from ..utils.themes import DarkTheme
from ..utils.icons import Icons
from .dialogs import MultiDirectoryDialog


class CalibrationSection(QFrame):
    """Section for calibration file management."""

    balance_loaded = pyqtSignal(str)
    pressure_loaded = pyqtSignal(str)

    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        # Title
        title = QLabel("Calibration Files")
        title.setStyleSheet("font-weight: bold; font-size: 11pt;")
        layout.addWidget(title)

        # Balance calibration
        balance_layout = QHBoxLayout()
        balance_label = QLabel("Balance:")
        balance_label.setFixedWidth(60)
        balance_layout.addWidget(balance_label)

        self.txt_balance = QLineEdit()
        self.txt_balance.setPlaceholderText("No file loaded")
        self.txt_balance.setReadOnly(True)
        balance_layout.addWidget(self.txt_balance)

        self.btn_balance = QPushButton()
        self.btn_balance.setIcon(Icons.folder_open())
        self.btn_balance.setToolTip("Load balance calibration (.vol)")
        self.btn_balance.setFixedSize(28, 28)
        self.btn_balance.clicked.connect(self._load_balance)
        balance_layout.addWidget(self.btn_balance)

        layout.addLayout(balance_layout)

        # Pressure calibration
        pressure_layout = QHBoxLayout()
        pressure_label = QLabel("Pressure:")
        pressure_label.setFixedWidth(60)
        pressure_layout.addWidget(pressure_label)

        self.txt_pressure = QLineEdit()
        self.txt_pressure.setPlaceholderText("No file loaded")
        self.txt_pressure.setReadOnly(True)
        pressure_layout.addWidget(self.txt_pressure)

        self.btn_pressure = QPushButton()
        self.btn_pressure.setIcon(Icons.folder_open())
        self.btn_pressure.setToolTip("Load pressure calibration (.PCF)")
        self.btn_pressure.setFixedSize(28, 28)
        self.btn_pressure.clicked.connect(self._load_pressure)
        pressure_layout.addWidget(self.btn_pressure)

        layout.addLayout(pressure_layout)

        # Status
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet(f"color: {DarkTheme.TEXT_SECONDARY}; font-size: 9pt;")
        layout.addWidget(self.lbl_status)

    def _load_balance(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Balance Calibration",
            self.settings.last_calibration_directory,
            "VOL Files (*.vol);;All Files (*.*)"
        )
        if filepath:
            self.settings.last_calibration_directory = str(Path(filepath).parent)
            self.settings.add_recent_balance_file(filepath)
            self.txt_balance.setText(Path(filepath).name)
            self.balance_loaded.emit(filepath)

    def _load_pressure(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Pressure Calibration",
            self.settings.last_calibration_directory,
            "PCF Files (*.PCF *.pcf);;All Files (*.*)"
        )
        if filepath:
            self.settings.last_calibration_directory = str(Path(filepath).parent)
            self.settings.add_recent_pressure_file(filepath)
            self.txt_pressure.setText(Path(filepath).name)
            self.pressure_loaded.emit(filepath)

    def set_balance_file(self, filepath: str):
        """Set the balance file display."""
        self.txt_balance.setText(Path(filepath).name if filepath else "")

    def set_pressure_file(self, filepath: str):
        """Set the pressure file display."""
        self.txt_pressure.setText(Path(filepath).name if filepath else "")

    def set_status(self, message: str, is_error: bool = False):
        """Set status message."""
        color = DarkTheme.ERROR if is_error else DarkTheme.TEXT_SECONDARY
        self.lbl_status.setStyleSheet(f"color: {color}; font-size: 9pt;")
        self.lbl_status.setText(message)


class DataPanel(QWidget):
    """
    Panel for data management.

    Contains calibration loading, case management, and data loading controls.

    Signals
    -------
    load_data_requested : pyqtSignal(str)
        Emitted when user wants to load data from a directory
    case_delete_requested : pyqtSignal(str)
        Emitted when user wants to delete a case
    """

    load_data_requested = pyqtSignal(list, bool)  # List of directories, recursive flag
    append_data_requested = pyqtSignal(str)  # Append single directory (don't clear existing)
    balance_cal_requested = pyqtSignal(str)
    pressure_cal_requested = pyqtSignal(str)
    geometry_requested = pyqtSignal()
    process_requested = pyqtSignal()
    case_delete_requested = pyqtSignal(str)  # case_id
    case_visibility_changed = pyqtSignal(str, bool)  # case_id, visible
    case_color_changed = pyqtSignal(str, str)  # case_id, color

    def __init__(self, model: DataModel, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.model = model
        self.settings = settings
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Calibration section
        self.cal_section = CalibrationSection(self.settings)
        self.cal_section.setStyleSheet(f"""
            QFrame {{
                background-color: {DarkTheme.BACKGROUND_LIGHT};
                border: none;
                border-bottom: 1px solid {DarkTheme.BORDER};
            }}
        """)
        layout.addWidget(self.cal_section)

        # Geometry button
        geo_frame = QFrame()
        geo_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {DarkTheme.BACKGROUND_LIGHT};
                border: none;
                border-bottom: 1px solid {DarkTheme.BORDER};
            }}
        """)
        geo_layout = QHBoxLayout(geo_frame)
        geo_layout.setContentsMargins(12, 8, 12, 8)

        self.btn_geometry = QPushButton("Set Geometry...")
        self.btn_geometry.setIcon(Icons.cube())
        self.btn_geometry.clicked.connect(self.geometry_requested.emit)
        geo_layout.addWidget(self.btn_geometry)

        geo_layout.addStretch()

        self.lbl_geometry = QLabel("No geometry set")
        self.lbl_geometry.setStyleSheet(f"color: {DarkTheme.TEXT_SECONDARY};")
        geo_layout.addWidget(self.lbl_geometry)

        layout.addWidget(geo_frame)

        # Data loading section
        load_frame = QFrame()
        load_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {DarkTheme.BACKGROUND_LIGHT};
                border: none;
                border-bottom: 1px solid {DarkTheme.BORDER};
            }}
        """)
        load_layout = QVBoxLayout(load_frame)
        load_layout.setContentsMargins(12, 8, 12, 8)
        load_layout.setSpacing(8)

        title = QLabel("Data Loading")
        title.setStyleSheet("font-weight: bold; font-size: 11pt;")
        load_layout.addWidget(title)

        btn_layout = QHBoxLayout()

        self.btn_load_dir = QPushButton("Load Directory...")
        self.btn_load_dir.setIcon(Icons.folder_open())
        self.btn_load_dir.setToolTip("Load all TDMS files from a directory")
        self.btn_load_dir.clicked.connect(self._load_directory)
        btn_layout.addWidget(self.btn_load_dir)

        self.btn_process = QPushButton("Process Data")
        self.btn_process.setIcon(Icons.play())
        self.btn_process.setToolTip("Process loaded data")
        self.btn_process.setProperty("primary", True)
        self.btn_process.clicked.connect(self.process_requested.emit)
        btn_layout.addWidget(self.btn_process)

        load_layout.addLayout(btn_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        load_layout.addWidget(self.progress_bar)

        layout.addWidget(load_frame)

        # Case list
        self.case_list = CaseListWidget()
        layout.addWidget(self.case_list, stretch=1)

    def _connect_signals(self):
        """Connect internal signals."""
        self.cal_section.balance_loaded.connect(self.balance_cal_requested.emit)
        self.cal_section.pressure_loaded.connect(self.pressure_cal_requested.emit)

        # Model signals
        self.model.cases_changed.connect(self._update_case_list)
        self.model.processing_progress.connect(self._update_progress)

        # Case list signals - forward to parent
        self.case_list.case_deleted.connect(self._on_case_delete)
        self.case_list.case_visibility_changed.connect(self._on_case_visibility_changed)
        self.case_list.case_color_changed.connect(self._on_case_color_changed)
        self.case_list.add_requested.connect(self._on_add_case_requested)

    def _on_case_delete(self, case_id: str):
        """Handle case delete request."""
        self.case_delete_requested.emit(case_id)

    def _on_case_visibility_changed(self, case_id: str, visible: bool):
        """Handle case visibility change."""
        self.model.set_case_visibility(case_id, visible)

    def _on_case_color_changed(self, case_id: str, color: str):
        """Handle case color change."""
        self.model.update_case(case_id, color=color)

    def _on_add_case_requested(self):
        """Handle add case request - open directory picker to append data."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Data Directory to Add",
            self.settings.last_data_directory
        )
        if directory:
            self.settings.last_data_directory = directory
            self.append_data_requested.emit(directory)

    def _load_directory(self):
        """Open multi-directory dialog for loading data."""
        dialog = MultiDirectoryDialog(self, self.settings.last_data_directory)
        if dialog.exec():
            directories = dialog.get_directories()
            if directories:
                # Update last directory setting
                self.settings.last_data_directory = str(Path(directories[-1]).parent)
                self.load_data_requested.emit(directories, dialog.recursive)

    def _update_case_list(self):
        """Update case list from model."""
        self.case_list.clear()
        for case in self.model.cases:
            self.case_list.add_case(case)

    def _update_progress(self, current: int, total: int):
        """Update progress bar."""
        if total > 0:
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
            self.progress_bar.setFormat(f"Processing... {current}/{total}")
        else:
            self.progress_bar.setVisible(False)

    def set_geometry_display(self, mac: float, area: float, units: str):
        """Update geometry display."""
        self.lbl_geometry.setText(f"MAC={mac:.2f}, S={area:.2f} ({units})")

    def show_processing(self, show: bool):
        """Show/hide processing indicator."""
        self.progress_bar.setVisible(show)
        if show:
            self.progress_bar.setMaximum(0)  # Indeterminate
        self.btn_process.setEnabled(not show)
        self.btn_load_dir.setEnabled(not show)
