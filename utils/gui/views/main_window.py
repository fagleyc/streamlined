"""
Main Window
===========

Main application window with all panels.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTabWidget, QMenuBar, QMenu, QToolBar, QStatusBar, QLabel,
    QMessageBox, QProgressBar, QFileDialog
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from pathlib import Path

from ..models.data_model import DataModel
from ..models.settings import AppSettings
from .data_panel import DataPanel
from .plot_panel import PlotPanel
from .table_panel import TablePanel
from .time_history_panel import TimeHistoryPanel
from .dialogs import (
    GeometryDialog, CalibrationDialog, AboutDialog, OutputUnitsDialog,
    TunnelCorrectionsDialog,
)
from ..widgets.export_dialog import ExportDialog
from ..utils.themes import DarkTheme
from ..utils.icons import Icons
from .. import __app_name__, __version__


class MainWindow(QMainWindow):
    """
    Main application window.
    """

    # Signals for configuration save/load
    save_config_requested = pyqtSignal(str)  # filepath
    load_config_requested = pyqtSignal(str)  # filepath

    def __init__(self, model: DataModel, settings: AppSettings):
        super().__init__()

        self.model = model
        self.settings = settings

        self._setup_window()
        self._setup_ui()
        self._setup_menubar()
        self._setup_toolbar()
        self._setup_statusbar()
        self._connect_signals()
        self._restore_state()

    def _setup_window(self):
        """Configure the main window."""
        self.setWindowTitle(f"{__app_name__} v{__version__}")
        self.setMinimumSize(1000, 700)

        # Restore window geometry
        if self.settings.window_maximized:
            self.showMaximized()
        else:
            self.resize(self.settings.window_size)
            self.move(self.settings.window_position)

    def _setup_ui(self):
        """Set up the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Main splitter
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Data management
        self.data_panel = DataPanel(self.model, self.settings)
        self.data_panel.setMinimumWidth(280)
        self.data_panel.setMaximumWidth(400)
        self.main_splitter.addWidget(self.data_panel)

        # Right area - Tabs for plot and table
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)

        # Plot tab
        self.plot_panel = PlotPanel(self.model)
        self.tab_widget.addTab(self.plot_panel, Icons.chart(), "Plot")

        # Table tab
        self.table_panel = TablePanel(self.model, self.settings)
        self.tab_widget.addTab(self.table_panel, Icons.table(), "Data Table")

        # Time History tab
        self.time_history_panel = TimeHistoryPanel(self.model)
        self.tab_widget.addTab(self.time_history_panel, Icons.chart(), "Time History")

        right_layout.addWidget(self.tab_widget)
        self.main_splitter.addWidget(right_widget)

        # Set splitter sizes
        self.main_splitter.setSizes([300, 1100])

        layout.addWidget(self.main_splitter)

    def _setup_menubar(self):
        """Set up the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        self.action_save_config = QAction(Icons.save(), "&Save Configuration...", self)
        self.action_save_config.setShortcut(QKeySequence.StandardKey.Save)
        self.action_save_config.setStatusTip("Save current configuration to file")
        file_menu.addAction(self.action_save_config)

        self.action_load_config = QAction(Icons.folder_open(), "L&oad Configuration...", self)
        self.action_load_config.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self.action_load_config.setStatusTip("Load configuration from file")
        file_menu.addAction(self.action_load_config)

        file_menu.addSeparator()

        self.action_load_data = QAction(Icons.folder_open(), "&Load Data Directory...", self)
        self.action_load_data.setShortcut(QKeySequence.StandardKey.Open)
        self.action_load_data.setStatusTip("Load data from directory")
        file_menu.addAction(self.action_load_data)

        file_menu.addSeparator()

        self.action_load_balance = QAction("Load &Balance Calibration...", self)
        self.action_load_balance.setStatusTip("Load balance calibration file")
        file_menu.addAction(self.action_load_balance)

        self.action_add_balance = QAction("Add Balance Calibration...", self)
        self.action_add_balance.setStatusTip(
            "Load an additional balance calibration for per-case assignment")
        file_menu.addAction(self.action_add_balance)

        self.action_load_pressure = QAction("Load &Pressure Calibration...", self)
        self.action_load_pressure.setStatusTip("Load pressure calibration file")
        file_menu.addAction(self.action_load_pressure)

        file_menu.addSeparator()

        self.action_export = QAction(Icons.export(), "&Export...", self)
        self.action_export.setShortcut(QKeySequence("Ctrl+Shift+E"))
        self.action_export.setStatusTip("Export data to CSV, Excel, HDF5, or MATLAB")
        file_menu.addAction(self.action_export)

        file_menu.addSeparator()

        self.action_exit = QAction("E&xit", self)
        self.action_exit.setShortcut(QKeySequence.StandardKey.Quit)
        self.action_exit.triggered.connect(self.close)
        file_menu.addAction(self.action_exit)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")

        self.action_geometry = QAction(Icons.cube(), "Model &Geometry...", self)
        self.action_geometry.setShortcut(QKeySequence("Ctrl+G"))
        self.action_geometry.setStatusTip("Set model geometry parameters")
        edit_menu.addAction(self.action_geometry)

        self.action_calibration = QAction(Icons.tune(), "&Calibration Settings...", self)
        self.action_calibration.setStatusTip("Configure calibration settings")
        edit_menu.addAction(self.action_calibration)

        self.action_output_units = QAction("&Output Units...", self)
        self.action_output_units.setStatusTip("Set output unit system for display and export")
        self.action_output_units.triggered.connect(self._show_output_units_dialog)
        edit_menu.addAction(self.action_output_units)

        self.action_tunnel_corrections = QAction(
            "&Tunnel Corrections...", self)
        self.action_tunnel_corrections.setStatusTip(
            "Configure blockage / wall-effect corrections")
        self.action_tunnel_corrections.triggered.connect(
            self._show_tunnel_corrections_dialog)
        edit_menu.addAction(self.action_tunnel_corrections)

        edit_menu.addSeparator()

        self.action_clear = QAction(Icons.delete(), "&Clear All Data", self)
        self.action_clear.setStatusTip("Clear all loaded data")
        edit_menu.addAction(self.action_clear)

        # View menu
        view_menu = menubar.addMenu("&View")

        self.action_plot_tab = QAction("&Plot View", self)
        self.action_plot_tab.setShortcut(QKeySequence("Ctrl+1"))
        self.action_plot_tab.triggered.connect(lambda: self.tab_widget.setCurrentIndex(0))
        view_menu.addAction(self.action_plot_tab)

        self.action_table_tab = QAction("&Table View", self)
        self.action_table_tab.setShortcut(QKeySequence("Ctrl+2"))
        self.action_table_tab.triggered.connect(lambda: self.tab_widget.setCurrentIndex(1))
        view_menu.addAction(self.action_table_tab)

        self.action_time_history_tab = QAction("Time &History View", self)
        self.action_time_history_tab.setShortcut(QKeySequence("Ctrl+3"))
        self.action_time_history_tab.triggered.connect(lambda: self.tab_widget.setCurrentIndex(2))
        view_menu.addAction(self.action_time_history_tab)

        view_menu.addSeparator()

        self.action_refresh = QAction(Icons.refresh(), "&Refresh Plot", self)
        self.action_refresh.setShortcut(QKeySequence.StandardKey.Refresh)
        view_menu.addAction(self.action_refresh)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        self.action_readme = QAction("&Documentation (README)", self)
        self.action_readme.triggered.connect(self._open_readme)
        help_menu.addAction(self.action_readme)

        help_menu.addSeparator()

        self.action_about = QAction("&About", self)
        self.action_about.triggered.connect(self._show_about)
        help_menu.addAction(self.action_about)

    def _setup_toolbar(self):
        """Set up the main toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        toolbar.addAction(self.action_load_data)
        toolbar.addSeparator()
        toolbar.addAction(self.action_geometry)
        toolbar.addAction(self.action_calibration)
        toolbar.addSeparator()
        toolbar.addAction(self.action_refresh)

    def _setup_statusbar(self):
        """Set up the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        # Status message
        self.status_label = QLabel("Ready")
        self.statusbar.addWidget(self.status_label, stretch=1)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.statusbar.addPermanentWidget(self.progress_bar)

        # Case count
        self.case_count_label = QLabel("0 cases")
        self.case_count_label.setStyleSheet(f"color: {DarkTheme.TEXT_SECONDARY};")
        self.statusbar.addPermanentWidget(self.case_count_label)

    def _connect_signals(self):
        """Connect signals to slots."""
        # Menu actions
        self.action_load_data.triggered.connect(
            lambda: self.data_panel._load_directory()
        )
        self.action_load_balance.triggered.connect(
            lambda: self.data_panel.cal_section._load_balance()
        )
        self.action_add_balance.triggered.connect(self._add_balance_calibration)
        self.action_load_pressure.triggered.connect(
            lambda: self.data_panel.cal_section._load_pressure()
        )
        self.action_export.triggered.connect(self._show_export_dialog)
        self.action_geometry.triggered.connect(self._show_geometry_dialog)
        self.action_calibration.triggered.connect(self._show_calibration_dialog)
        self.action_clear.triggered.connect(self._clear_data)
        self.action_refresh.triggered.connect(
            lambda: self.plot_panel._update_plot()
        )
        self.action_save_config.triggered.connect(self._save_configuration)
        self.action_load_config.triggered.connect(self._load_configuration)

        # Data panel signals
        self.data_panel.geometry_requested.connect(self._show_geometry_dialog)

        # Case list geometry assignment
        self.data_panel.case_list.geometry_assigned.connect(
            self._on_geometry_assigned
        )

        # Case list calibration assignment
        self.data_panel.case_list.calibration_assigned.connect(
            self._on_calibration_assigned
        )

        # Plot panel right-click signals
        self.plot_panel.view_time_history_requested.connect(
            self._switch_to_time_history
        )
        self.plot_panel.view_fft_requested.connect(
            self._switch_to_fft
        )

        # Model signals
        self.model.cases_changed.connect(self._update_case_count)
        self.model.processing_started.connect(self._on_processing_started)
        self.model.processing_finished.connect(self._on_processing_finished)
        self.model.processing_progress.connect(self._on_progress)
        self.model.error_occurred.connect(self._show_error)

    def _restore_state(self):
        """Restore saved state."""
        # Restore splitter state
        splitter_state = self.settings.load_splitter_state("main_splitter")
        if splitter_state:
            self.main_splitter.restoreState(splitter_state)

    def _save_state(self):
        """Save current state."""
        # Save window geometry
        if self.isMaximized():
            self.settings.window_maximized = True
        else:
            self.settings.window_maximized = False
            self.settings.window_size = self.size()
            self.settings.window_position = self.pos()

        # Save splitter state
        self.settings.save_splitter_state("main_splitter", self.main_splitter.saveState())

        self.settings.sync()

    def closeEvent(self, event):
        """Handle window close."""
        self._save_state()
        event.accept()

    def _show_export_dialog(self):
        """Show the consolidated export dialog."""
        case_names = [c.name for c in self.model.cases if c.has_data]
        if not case_names:
            QMessageBox.information(
                self, "Nothing to Export",
                "No reduced cases are available to export.\n\n"
                "Load a data directory and process the data before exporting.")
            return

        current_case = None
        case_id = self.table_panel.cmb_case.currentData()
        if case_id:
            c = self.model.cases.get(case_id)
            if c:
                current_case = c.name

        dialog = ExportDialog(
            case_names=case_names,
            current_case_name=current_case,
            last_directory=self.settings.last_export_directory,
            parent=self
        )
        if dialog.exec() == ExportDialog.DialogCode.Accepted:
            config = dialog.get_export_config()
            if not config.get('filepath'):
                QMessageBox.warning(
                    self, "Export Cancelled",
                    "No file path was selected. Export aborted.")
                return
            self.settings.last_export_directory = str(
                Path(config['filepath']).parent)
            try:
                self.table_panel.do_export(config)
            except Exception as e:
                import traceback
                traceback.print_exc()
                QMessageBox.critical(
                    self, "Export Failed",
                    f"Failed to write {Path(config['filepath']).name}:\n\n"
                    f"{type(e).__name__}: {e}")

    def _show_geometry_dialog(self):
        """Show multi-geometry settings dialog."""
        dialog = GeometryDialog(self, geometries=self.model.geometries)

        if dialog.exec():
            new_geos = dialog.get_geometries()
            self.model.geometries = new_geos
            # Ensure default geometry still exists
            if self.model.default_geometry not in new_geos:
                self.model.default_geometry = next(iter(new_geos))
            # Update case list with new geometry names
            self.data_panel.case_list.set_geometry_names(list(new_geos.keys()))
            # Update display with default geometry info
            default_geo = new_geos.get(self.model.default_geometry, {})
            self.data_panel.set_geometry_display(
                default_geo.get('mac', 1.0),
                default_geo.get('ref_area', 1.0),
                default_geo.get('units', 'IPS')
            )
            self.set_status(
                f"Geometries updated: {len(new_geos)} definition(s)")
            self.plot_panel._update_plot()

    def _show_output_units_dialog(self):
        """Show output units selection dialog."""
        dialog = OutputUnitsDialog(self, current_units=self.model.output_units)

        if dialog.exec():
            units = dialog.get_units()
            self.model.set_output_units(units)
            self.set_status(f"Output units set to {units}")

    def _show_tunnel_corrections_dialog(self):
        """Show tunnel-blockage / wall-effect correction settings."""
        current = getattr(self.model, 'blockage_config', {'method': 'none'})
        dialog = TunnelCorrectionsDialog(self, current_config=current)
        if dialog.exec():
            cfg = dialog.get_config()
            self.model.blockage_config = cfg
            method = cfg.get('method', 'none')
            try:
                from utils.windtunnel.blockage import METHOD_LABELS
                label = METHOD_LABELS.get(method, method)
            except Exception:
                label = method
            self.set_status(f"Tunnel corrections: {label}")

    def _on_geometry_assigned(self, case_id: str, geometry_name: str):
        """Handle geometry assignment to a case -- auto re-processes."""
        self.model.assign_geometry(case_id, geometry_name)
        self.data_panel.case_list.update_case_geometry_label(case_id, geometry_name)
        self.set_status(f"Assigned geometry '{geometry_name}' to case -- reprocessing...")
        if hasattr(self, 'data_controller'):
            self.data_controller.reprocess_case(case_id)

    def _on_calibration_assigned(self, case_id: str, cal_name: str):
        """Handle calibration assignment to a case -- auto re-processes."""
        self.model.assign_calibration(case_id, cal_name)
        self.data_panel.case_list.update_case_calibration_label(case_id, cal_name)
        self.set_status(f"Assigned calibration '{cal_name}' to case -- reprocessing...")
        if hasattr(self, 'data_controller'):
            self.data_controller.reprocess_case(case_id)

    def _show_calibration_dialog(self):
        """Show calibration settings dialog."""
        dialog = CalibrationDialog(
            self,
            cal_type=self.model.cal_type,
            facility=self.model.facility,
            balance_config=self.model.balance_config,
            pdiff_channel=getattr(self.model, 'pdiff_channel', '220'),
            p0_channel=getattr(self.model, 'p0_channel', '690'),
        )

        if dialog.exec():
            values = dialog.get_values()
            self.model.cal_type = values['cal_type']
            self.model.facility = values['facility']
            self.model.balance_config = values['balance_config']
            self.model.pdiff_channel = values['pdiff_channel']
            self.model.p0_channel = values['p0_channel']
            self.set_status("Calibration settings updated - Use 'Process Data' to apply changes")

    def _add_balance_calibration(self):
        """Load an additional balance calibration file for per-case assignment."""
        from PyQt6.QtWidgets import (
            QFileDialog, QDialog, QVBoxLayout, QFormLayout,
            QComboBox, QLineEdit, QDialogButtonBox, QGroupBox,
        )
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Add Balance Calibration",
            self.settings.last_calibration_directory,
            "VOL Files (*.vol);;All Files (*.*)"
        )
        if not filepath:
            return
        self.settings.last_calibration_directory = str(Path(filepath).parent)

        # Prompt for calibration settings
        dlg = QDialog(self)
        dlg.setWindowTitle("Calibration Settings")
        dlg.setMinimumWidth(320)
        layout = QVBoxLayout(dlg)

        group = QGroupBox(Path(filepath).name)
        form = QFormLayout(group)

        txt_name = QLineEdit(Path(filepath).stem)
        form.addRow("Name:", txt_name)

        cmb_type = QComboBox()
        cmb_type.addItems(["Linear", "Quadratic", "Cubic"])
        cmb_type.setCurrentText(self.model.cal_type)
        form.addRow("Fit Type:", cmb_type)

        cmb_config = QComboBox()
        cmb_config.addItems(["Force", "Moment"])
        cmb_config.setCurrentText(self.model.balance_config)
        form.addRow("Balance Config:", cmb_config)

        layout.addWidget(group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        cal_name = txt_name.text().strip() or Path(filepath).stem
        cal_type = cmb_type.currentText()
        balance_config = cmb_config.currentText()

        if hasattr(self, 'data_controller'):
            if self.data_controller.add_balance_calibration(
                    filepath, cal_type, balance_config, name=cal_name):
                self.data_panel.case_list.set_calibration_names(
                    self.model.calibration_names)
                self.set_status(
                    f"Added calibration '{cal_name}' "
                    f"({cal_type}, {balance_config})")

    def _open_readme(self):
        """Open the README.md file with the system default application."""
        import os, subprocess, sys
        readme_path = Path(__file__).resolve().parents[3] / "README.md"
        if not readme_path.exists():
            QMessageBox.warning(self, "Not Found",
                                f"README.md not found at:\n{readme_path}")
            return
        if sys.platform == 'win32':
            os.startfile(str(readme_path))
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', str(readme_path)])
        else:
            subprocess.Popen(['xdg-open', str(readme_path)])

    def _show_about(self):
        """Show about dialog."""
        dialog = AboutDialog(self)
        dialog.exec()

    def _clear_data(self):
        """Clear all data after confirmation."""
        reply = QMessageBox.question(
            self, "Clear Data",
            "Are you sure you want to clear all loaded data?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.model.clear_all()
            self.set_status("All data cleared")

    def _save_configuration(self):
        """Save current configuration to file."""
        project_root = str(Path(__file__).parent.parent.parent.parent)
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration",
            project_root,
            "Configuration Files (*.json);;All Files (*.*)"
        )
        if filepath:
            # Ensure .json extension
            if not filepath.endswith('.json'):
                filepath += '.json'
            self.save_config_requested.emit(filepath)

    def _load_configuration(self):
        """Load configuration from file."""
        project_root = str(Path(__file__).parent.parent.parent.parent)
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration",
            project_root,
            "Configuration Files (*.json);;All Files (*.*)"
        )
        if filepath:
            self.load_config_requested.emit(filepath)

    def update_geometry_display(self):
        """Update the geometry display from model."""
        self.data_panel.set_geometry_display(
            self.model.mac, self.model.ref_area, self.model.units
        )
        # Update case list with available geometry names
        self.data_panel.case_list.set_geometry_names(self.model.geometry_names)

    def update_calibration_display(self):
        """Update the calibration file display from model."""
        if self.model.balance_cal_file:
            self.data_panel.cal_section.set_balance_file(str(self.model.balance_cal_file))
        if self.model.pressure_cal_file:
            self.data_panel.cal_section.set_pressure_file(str(self.model.pressure_cal_file))
        # Sync calibration names to case list context menu
        self.data_panel.case_list.set_calibration_names(
            self.model.calibration_names)

    def _update_case_count(self):
        """Update case count display."""
        count = len(self.model.cases)
        self.case_count_label.setText(f"{count} case{'s' if count != 1 else ''}")

        # Update plot filters
        self.plot_panel.update_filters()

    def _on_processing_started(self, message: str):
        """Handle processing started."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(0)  # Indeterminate
        self.set_status(message)

    def _on_processing_finished(self, message: str):
        """Handle processing finished."""
        self.progress_bar.setVisible(False)
        self.set_status(message)

    def _on_progress(self, current: int, total: int):
        """Update progress bar."""
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
        else:
            self.progress_bar.setMaximum(0)

    def _show_error(self, title: str, message: str):
        """Show error message."""
        QMessageBox.critical(self, title, message)
        self.set_status(f"Error: {title}")

    def _switch_to_time_history(self, case_id: str, alpha: float, beta: float, y_var: str):
        """Switch to Time History tab, select the case, point, channel, and set time domain."""
        self.tab_widget.setCurrentIndex(2)
        self._select_time_history_case(case_id)
        self._select_time_history_point(alpha, beta)
        self._select_time_history_channel(y_var)
        # Set domain to time
        for i in range(self.time_history_panel.cmb_domain.count()):
            if self.time_history_panel.cmb_domain.itemData(i) == "time":
                self.time_history_panel.cmb_domain.setCurrentIndex(i)
                break

    def _switch_to_fft(self, case_id: str, alpha: float, beta: float, y_var: str):
        """Switch to Time History tab, select the case/point/channel, and set FFT domain."""
        self.tab_widget.setCurrentIndex(2)
        self._select_time_history_case(case_id)
        self._select_time_history_point(alpha, beta)
        self._select_time_history_channel(y_var)
        # Set domain to FFT
        for i in range(self.time_history_panel.cmb_domain.count()):
            if self.time_history_panel.cmb_domain.itemData(i) == "fft":
                self.time_history_panel.cmb_domain.setCurrentIndex(i)
                break

    def _select_time_history_case(self, case_id: str):
        """Select a case in the time history panel by case_id."""
        cmb = self.time_history_panel.cmb_case
        for i in range(cmb.count()):
            if cmb.itemData(i) == case_id:
                cmb.setCurrentIndex(i)
                break

    def _select_time_history_point(self, alpha: float, beta: float):
        """Select the closest point in the time history panel by alpha/beta."""
        import re
        cmb = self.time_history_panel.cmb_point
        best_idx = 0
        best_dist = float('inf')

        for i in range(cmb.count()):
            text = cmb.itemText(i)
            # Parse "Point N: α=X.X°, β=Y.Y°"
            m = re.search(r'\u03b1=([-\d.]+)\u00b0.*\u03b2=([-\d.]+)\u00b0', text)
            if m:
                pt_alpha = float(m.group(1))
                pt_beta = float(m.group(2))
                dist = abs(pt_alpha - alpha) + abs(pt_beta - beta)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

        if best_idx < cmb.count():
            cmb.setCurrentIndex(best_idx)

    def _select_time_history_channel(self, y_var: str):
        """Select the time history channel matching the plot's y-variable."""
        cmb = self.time_history_panel.cmb_channel
        for i in range(cmb.count()):
            data = cmb.itemData(i)
            if data and len(data) == 2 and data[1] == y_var:
                cmb.setCurrentIndex(i)
                return

    def set_status(self, message: str):
        """Set status bar message."""
        self.status_label.setText(message)
        self.statusbar.showMessage(message, 5000)
