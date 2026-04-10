"""
Application Settings
====================

Persistent settings using QSettings.
"""

from pathlib import Path
from typing import Optional, List, Any
from PyQt6.QtCore import QSettings, QSize, QPoint


class AppSettings:
    """
    Application settings manager using QSettings for persistence.
    """

    def __init__(self):
        self._settings = QSettings("WindTunnelLab", "DataAnalyzer")

    # Window geometry
    @property
    def window_size(self) -> QSize:
        return self._settings.value("window/size", QSize(1400, 900))

    @window_size.setter
    def window_size(self, size: QSize):
        self._settings.setValue("window/size", size)

    @property
    def window_position(self) -> QPoint:
        return self._settings.value("window/position", QPoint(100, 100))

    @window_position.setter
    def window_position(self, pos: QPoint):
        self._settings.setValue("window/position", pos)

    @property
    def window_maximized(self) -> bool:
        return self._settings.value("window/maximized", False, type=bool)

    @window_maximized.setter
    def window_maximized(self, maximized: bool):
        self._settings.setValue("window/maximized", maximized)

    # Last used directories
    @property
    def last_data_directory(self) -> str:
        return self._settings.value("paths/last_data_dir", "")

    @last_data_directory.setter
    def last_data_directory(self, path: str):
        self._settings.setValue("paths/last_data_dir", path)

    @property
    def last_calibration_directory(self) -> str:
        return self._settings.value("paths/last_cal_dir", "")

    @last_calibration_directory.setter
    def last_calibration_directory(self, path: str):
        self._settings.setValue("paths/last_cal_dir", path)

    @property
    def last_export_directory(self) -> str:
        return self._settings.value("paths/last_export_dir", "")

    @last_export_directory.setter
    def last_export_directory(self, path: str):
        self._settings.setValue("paths/last_export_dir", path)

    # Recent files
    @property
    def recent_balance_files(self) -> List[str]:
        return self._settings.value("recent/balance_files", [])

    @recent_balance_files.setter
    def recent_balance_files(self, files: List[str]):
        self._settings.setValue("recent/balance_files", files[:10])

    @property
    def recent_pressure_files(self) -> List[str]:
        return self._settings.value("recent/pressure_files", [])

    @recent_pressure_files.setter
    def recent_pressure_files(self, files: List[str]):
        self._settings.setValue("recent/pressure_files", files[:10])

    def add_recent_balance_file(self, filepath: str):
        """Add a file to recent balance calibrations."""
        files = self.recent_balance_files
        if filepath in files:
            files.remove(filepath)
        files.insert(0, filepath)
        self.recent_balance_files = files

    def add_recent_pressure_file(self, filepath: str):
        """Add a file to recent pressure calibrations."""
        files = self.recent_pressure_files
        if filepath in files:
            files.remove(filepath)
        files.insert(0, filepath)
        self.recent_pressure_files = files

    # Default geometry settings
    @property
    def default_mac(self) -> float:
        return self._settings.value("geometry/mac", 1.0, type=float)

    @default_mac.setter
    def default_mac(self, value: float):
        self._settings.setValue("geometry/mac", value)

    @property
    def default_ref_area(self) -> float:
        return self._settings.value("geometry/ref_area", 1.0, type=float)

    @default_ref_area.setter
    def default_ref_area(self, value: float):
        self._settings.setValue("geometry/ref_area", value)

    @property
    def default_units(self) -> str:
        return self._settings.value("geometry/units", "IPS")

    @default_units.setter
    def default_units(self, value: str):
        self._settings.setValue("geometry/units", value)

    # Plot settings
    @property
    def plot_use_latex(self) -> bool:
        return self._settings.value("plot/use_latex", False, type=bool)

    @plot_use_latex.setter
    def plot_use_latex(self, value: bool):
        self._settings.setValue("plot/use_latex", value)

    @property
    def plot_show_grid(self) -> bool:
        return self._settings.value("plot/show_grid", True, type=bool)

    @plot_show_grid.setter
    def plot_show_grid(self, value: bool):
        self._settings.setValue("plot/show_grid", value)

    @property
    def plot_show_legend(self) -> bool:
        return self._settings.value("plot/show_legend", True, type=bool)

    @plot_show_legend.setter
    def plot_show_legend(self, value: bool):
        self._settings.setValue("plot/show_legend", value)

    # Theme
    @property
    def theme(self) -> str:
        return self._settings.value("appearance/theme", "dark")

    @theme.setter
    def theme(self, value: str):
        self._settings.setValue("appearance/theme", value)

    # Splitter states
    def save_splitter_state(self, name: str, state: bytes):
        self._settings.setValue(f"splitters/{name}", state)

    def load_splitter_state(self, name: str) -> Optional[bytes]:
        return self._settings.value(f"splitters/{name}")

    # Generic value access
    def value(self, key: str, default: Any = None, value_type: type = None) -> Any:
        if value_type:
            return self._settings.value(key, default, type=value_type)
        return self._settings.value(key, default)

    def setValue(self, key: str, value: Any):
        self._settings.setValue(key, value)

    def sync(self):
        """Force sync settings to storage."""
        self._settings.sync()
