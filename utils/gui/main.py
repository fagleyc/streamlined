"""
Wind Tunnel Data Reduction GUI
==============================

Main entry point for the application.
"""

import sys
import os
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QFont, QPainter, QColor

from .models.data_model import DataModel
from .models.settings import AppSettings
from .views.main_window import MainWindow
from .controllers.data_controller import DataController
from .utils.themes import get_stylesheet, DarkTheme
from . import __app_name__, __version__


def create_splash_pixmap() -> QPixmap:
    """Create a splash screen pixmap."""
    pixmap = QPixmap(400, 250)
    pixmap.fill(QColor(DarkTheme.BACKGROUND))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Draw border
    painter.setPen(QColor(DarkTheme.BORDER))
    painter.drawRect(0, 0, 399, 249)

    # Draw title
    painter.setPen(QColor(DarkTheme.ACCENT))
    font = QFont("Segoe UI", 20, QFont.Weight.Bold)
    painter.setFont(font)
    painter.drawText(pixmap.rect().adjusted(0, 60, 0, 0),
                     Qt.AlignmentFlag.AlignHCenter, __app_name__)

    # Draw version
    painter.setPen(QColor(DarkTheme.TEXT_SECONDARY))
    font = QFont("Segoe UI", 12)
    painter.setFont(font)
    painter.drawText(pixmap.rect().adjusted(0, 100, 0, 0),
                     Qt.AlignmentFlag.AlignHCenter, f"Version {__version__}")

    # Draw loading text
    painter.setPen(QColor(DarkTheme.TEXT_PRIMARY))
    font = QFont("Segoe UI", 10)
    painter.setFont(font)
    painter.drawText(pixmap.rect().adjusted(0, 180, 0, 0),
                     Qt.AlignmentFlag.AlignHCenter, "Loading...")

    painter.end()
    return pixmap


class WindTunnelApp:
    """
    Main application class.

    Initializes the application, creates the model-view-controller structure,
    and manages the application lifecycle.
    """

    def __init__(self):
        self.app = None
        self.model = None
        self.settings = None
        self.controller = None
        self.main_window = None

    def initialize(self):
        """Initialize the application."""
        # Create QApplication
        self.app = QApplication(sys.argv)
        self.app.setApplicationName(__app_name__)
        self.app.setApplicationVersion(__version__)
        self.app.setOrganizationName("WindTunnel")
        self.app.setOrganizationDomain("windtunnel.local")

        # Apply stylesheet
        self.app.setStyleSheet(get_stylesheet())

        # Set default font
        font = QFont("Segoe UI", 10)
        self.app.setFont(font)

    def create_components(self):
        """Create the MVC components."""
        # Create model
        self.model = DataModel()

        # Create settings
        self.settings = AppSettings()

        # Create controller
        self.controller = DataController(self.model, self.settings)

        # Create main window
        self.main_window = MainWindow(self.model, self.settings)

        # Make controller accessible from main window for reprocessing
        self.main_window.data_controller = self.controller

        # Connect controller to window
        self._connect_controller()

    def _connect_controller(self):
        """Connect controller signals to the main window."""
        # Controller status updates
        self.controller.status_changed.connect(self.main_window.set_status)
        self.controller.error_occurred.connect(self.main_window._show_error)

        # Data panel requests
        self.main_window.data_panel.load_data_requested.connect(
            lambda dirs, recursive: self.controller.load_data_directories(dirs, recursive=recursive)
        )
        self.main_window.data_panel.balance_cal_requested.connect(
            self._on_balance_cal_requested
        )
        self.main_window.data_panel.process_requested.connect(
            self.controller.process_data
        )

        # Case management
        self.main_window.data_panel.case_delete_requested.connect(
            self.controller.remove_case
        )
        self.main_window.data_panel.append_data_requested.connect(
            self.controller.append_data_directory
        )

        # Configuration save/load
        self.main_window.save_config_requested.connect(
            self.controller.save_configuration
        )
        self.main_window.load_config_requested.connect(
            self._load_config_and_update
        )

        # Show backend status
        status = self.controller.get_backend_status()
        self.main_window.set_status(status)

    def _load_config_and_update(self, filepath: str):
        """Load configuration and update UI."""
        if self.controller.load_configuration(filepath):
            # Update geometry display
            self.main_window.update_geometry_display()
            # Update calibration display
            self.main_window.update_calibration_display()

    def _on_balance_cal_requested(self, filepath: str):
        """Load balance calibration and sync calibration names to case list."""
        if self.controller.load_balance_calibration(filepath):
            self.main_window.data_panel.case_list.set_calibration_names(
                self.main_window.model.calibration_names)

    def run(self) -> int:
        """Run the application."""
        self.initialize()

        # Show splash screen
        splash_pixmap = create_splash_pixmap()
        splash = QSplashScreen(splash_pixmap)
        splash.show()
        self.app.processEvents()

        # Create components (simulated loading delay)
        QTimer.singleShot(500, lambda: self._finish_loading(splash))

        return self.app.exec()

    def _finish_loading(self, splash: QSplashScreen):
        """Finish loading and show main window."""
        self.create_components()

        # Show main window
        self.main_window.show()

        # Close splash
        splash.finish(self.main_window)


def main():
    """Main entry point."""
    # Enable high DPI scaling
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"

    # Create and run application
    app = WindTunnelApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
