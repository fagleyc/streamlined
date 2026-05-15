"""
Data Model
==========

Central data model that the GUI observes. Implements observer pattern
using Qt signals for reactivity.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto

from PyQt6.QtCore import QObject, pyqtSignal

from .case import TestCase, CaseCollection


class PlotType(Enum):
    """Available plot types."""
    CL_VS_ALPHA = auto()
    CD_VS_ALPHA = auto()
    CL_VS_CD = auto()
    CM_VS_ALPHA = auto()
    CM_VS_CL = auto()
    LD_VS_ALPHA = auto()
    LATERAL_VS_BETA = auto()
    CY_VS_ALPHA = auto()       # Side force vs alpha
    CROLL_VS_ALPHA = auto()    # Roll moment vs alpha
    CYAW_VS_ALPHA = auto()     # Yaw moment vs alpha
    CUSTOM = auto()


@dataclass
class PlotConfig:
    """Configuration for a plot."""
    plot_type: PlotType = PlotType.CL_VS_ALPHA
    x_var: str = "Alpha"
    y_var: str = "Cl"
    show_grid: bool = True
    show_legend: bool = True
    show_std_dev: bool = False
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    title: str = ""
    x_label: str = r"$\alpha$ [deg]"
    y_label: str = r"$C_L$"


@dataclass
class FilterState:
    """Current filter state."""
    selected_betas: List[float] = field(default_factory=list)
    selected_alphas: List[float] = field(default_factory=list)
    selected_machs: List[float] = field(default_factory=list)
    alpha_range: tuple = (-20.0, 30.0)
    show_all_betas: bool = True
    show_all_alphas: bool = True


class DataModel(QObject):
    """
    Central data model for the application.

    Emits signals when data changes to notify GUI components.
    """

    # Signals for data changes
    cases_changed = pyqtSignal()  # Cases added/removed
    case_updated = pyqtSignal(str)  # Single case updated (by ID)
    case_visibility_changed = pyqtSignal(str, bool)  # Case visibility toggled
    filters_changed = pyqtSignal()  # Filter settings changed
    plot_config_changed = pyqtSignal()  # Plot configuration changed
    calibration_loaded = pyqtSignal(str)  # Calibration file loaded
    processing_started = pyqtSignal(str)  # Processing operation started
    processing_finished = pyqtSignal(str)  # Processing operation finished
    processing_progress = pyqtSignal(int, int)  # Progress update (current, total)
    error_occurred = pyqtSignal(str, str)  # Error (title, message)
    output_units_changed = pyqtSignal(str)  # Output unit system changed

    def __init__(self, parent=None):
        super().__init__(parent)

        # Test cases
        self.cases = CaseCollection()

        # Calibration data
        self.balance_calibration = None
        self.pressure_calibration = None
        self.balance_cal_file: Optional[Path] = None
        self.pressure_cal_file: Optional[Path] = None
        self.cal_type: str = "Cubic"

        # Multi-calibration definitions: name -> {bal_file, press_file, cal_type}
        self.calibrations: Dict[str, dict] = {}
        self.default_calibration: str = ''
        self.case_calibration_map: Dict[str, str] = {}  # case_id -> cal name

        # Multi-geometry definitions: name → {mac, ref_area, span, mrc, units}
        self.geometries: Dict[str, dict] = {
            'Default': {
                'mac': 1.0, 'ref_area': 1.0, 'span': 1.0,
                'mrc': [0.0, 0.0, 0.0], 'units': 'IPS'
            }
        }
        self.default_geometry: str = 'Default'
        self.case_geometry_map: Dict[str, str] = {}  # case_id → geometry name

        # Output units (independent of geometry)
        self.output_units: str = "IPS"

        # Facility settings
        self.facility: str = "SWT"
        self.balance_config: str = "Force"
        self.pdiff_channel: str = "220"
        self.p0_channel: str = "690"

        # Tunnel blockage / wall-effect correction configuration.
        # Stored as a plain dict for trivial JSON round-tripping; the
        # blockage module reconstructs a BlockageConfig dataclass from
        # it on demand.  Default 'none' preserves backward-compatible
        # behavior.
        self.blockage_config: dict = {'method': 'none'}

        # Filter state
        self.filters = FilterState()

        # Plot configuration
        self.plot_config = PlotConfig()

        # Color palette for cases
        self._color_palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
        ]
        self._color_index = 0

        # Marker palette
        self._marker_palette = ["o", "s", "^", "D", "v", "<", ">", "p", "h"]
        self._marker_index = 0

    def _next_color(self) -> str:
        """Get next color from palette."""
        color = self._color_palette[self._color_index % len(self._color_palette)]
        self._color_index += 1
        return color

    def _next_marker(self) -> str:
        """Get next marker from palette."""
        marker = self._marker_palette[self._marker_index % len(self._marker_palette)]
        self._marker_index += 1
        return marker

    def add_case(self, case: TestCase) -> None:
        """Add a test case."""
        if not case.color or case.color == "#1f77b4":
            case.color = self._next_color()
        if not case.marker:
            case.marker = "o"

        self.cases.add(case)
        self.cases_changed.emit()

    def remove_case(self, case_id: str) -> None:
        """Remove a test case."""
        self.cases.remove(case_id)
        self.cases_changed.emit()

    def update_case(self, case_id: str, **kwargs) -> None:
        """Update case properties."""
        case = self.cases.get(case_id)
        if case:
            for key, value in kwargs.items():
                if hasattr(case, key):
                    setattr(case, key, value)
            self.case_updated.emit(case_id)

    def set_case_visibility(self, case_id: str, visible: bool) -> None:
        """Set case visibility."""
        case = self.cases.get(case_id)
        if case:
            case.visible = visible
            self.case_visibility_changed.emit(case_id, visible)

    def toggle_case_visibility(self, case_id: str) -> None:
        """Toggle case visibility."""
        case = self.cases.get(case_id)
        if case:
            case.visible = not case.visible
            self.case_visibility_changed.emit(case_id, case.visible)

    def set_filters(self, **kwargs) -> None:
        """Update filter settings."""
        for key, value in kwargs.items():
            if hasattr(self.filters, key):
                setattr(self.filters, key, value)
        self.filters_changed.emit()

    def set_plot_config(self, **kwargs) -> None:
        """Update plot configuration."""
        for key, value in kwargs.items():
            if hasattr(self.plot_config, key):
                setattr(self.plot_config, key, value)
        self.plot_config_changed.emit()

    def set_plot_type(self, plot_type: PlotType) -> None:
        """Set the current plot type and update config."""
        self.plot_config.plot_type = plot_type

        # Set default axis labels based on plot type
        type_configs = {
            PlotType.CL_VS_ALPHA: ("Alpha", "Cl", r"$\alpha$ [deg]", r"$C_L$"),
            PlotType.CD_VS_ALPHA: ("Alpha", "Cd", r"$\alpha$ [deg]", r"$C_D$"),
            PlotType.CL_VS_CD: ("Cd", "Cl", r"$C_D$", r"$C_L$"),
            PlotType.CM_VS_ALPHA: ("Alpha", "CPitch", r"$\alpha$ [deg]", r"$C_m$"),
            PlotType.CM_VS_CL: ("Cl", "CPitch", r"$C_L$", r"$C_m$"),
            PlotType.LD_VS_ALPHA: ("Alpha", "L/D", r"$\alpha$ [deg]", r"$L/D$"),
            PlotType.LATERAL_VS_BETA: ("Beta", "Lateral", r"$\beta$ [deg]", "Coeff"),
            PlotType.CY_VS_ALPHA: ("Alpha", "Cs", r"$\alpha$ [deg]", r"$C_Y$"),
            PlotType.CROLL_VS_ALPHA: ("Alpha", "CRoll", r"$\alpha$ [deg]", r"$C_l$"),
            PlotType.CYAW_VS_ALPHA: ("Alpha", "CYaw", r"$\alpha$ [deg]", r"$C_n$"),
        }

        if plot_type in type_configs:
            x_var, y_var, x_label, y_label = type_configs[plot_type]
            self.plot_config.x_var = x_var
            self.plot_config.y_var = y_var
            self.plot_config.x_label = x_label
            self.plot_config.y_label = y_label

        self.plot_config_changed.emit()

    # --- Backward-compatible properties (read from default geometry) ---

    @property
    def mac(self) -> float:
        return self.geometries.get(self.default_geometry, {}).get('mac', 1.0)

    @mac.setter
    def mac(self, value: float):
        if self.default_geometry in self.geometries:
            self.geometries[self.default_geometry]['mac'] = value

    @property
    def ref_area(self) -> float:
        return self.geometries.get(self.default_geometry, {}).get('ref_area', 1.0)

    @ref_area.setter
    def ref_area(self, value: float):
        if self.default_geometry in self.geometries:
            self.geometries[self.default_geometry]['ref_area'] = value

    @property
    def span(self) -> float:
        return self.geometries.get(self.default_geometry, {}).get('span', 1.0)

    @span.setter
    def span(self, value: float):
        if self.default_geometry in self.geometries:
            self.geometries[self.default_geometry]['span'] = value

    @property
    def mrc(self) -> List[float]:
        return self.geometries.get(self.default_geometry, {}).get('mrc', [0.0, 0.0, 0.0])

    @mrc.setter
    def mrc(self, value: List[float]):
        if self.default_geometry in self.geometries:
            self.geometries[self.default_geometry]['mrc'] = value

    @property
    def units(self) -> str:
        return self.geometries.get(self.default_geometry, {}).get('units', 'IPS')

    @units.setter
    def units(self, value: str):
        if self.default_geometry in self.geometries:
            self.geometries[self.default_geometry]['units'] = value

    # --- Multi-geometry methods ---

    @property
    def geometry_names(self) -> List[str]:
        """Get list of all defined geometry names."""
        return list(self.geometries.keys())

    def add_geometry(self, name: str, params: dict = None) -> None:
        """Add a new geometry definition."""
        if params is None:
            params = {
                'mac': 1.0, 'ref_area': 1.0, 'span': 1.0,
                'mrc': [0.0, 0.0, 0.0], 'units': 'IPS'
            }
        self.geometries[name] = params

    def remove_geometry(self, name: str) -> bool:
        """Remove a geometry definition. Cannot remove the last one."""
        if len(self.geometries) <= 1:
            return False
        if name not in self.geometries:
            return False
        del self.geometries[name]
        # Update default if removed
        if self.default_geometry == name:
            self.default_geometry = next(iter(self.geometries))
        # Unmap any cases using the removed geometry
        for case_id, geo_name in list(self.case_geometry_map.items()):
            if geo_name == name:
                del self.case_geometry_map[case_id]
        return True

    def rename_geometry(self, old_name: str, new_name: str) -> bool:
        """Rename a geometry definition."""
        if old_name not in self.geometries or new_name in self.geometries:
            return False
        self.geometries[new_name] = self.geometries.pop(old_name)
        if self.default_geometry == old_name:
            self.default_geometry = new_name
        for case_id, geo_name in self.case_geometry_map.items():
            if geo_name == old_name:
                self.case_geometry_map[case_id] = new_name
        return True

    def get_geometry(self, name: str) -> dict:
        """Get a geometry definition by name."""
        return self.geometries.get(name, self.geometries.get(self.default_geometry, {}))

    def get_geometry_for_case(self, case_id: str) -> dict:
        """Get the geometry assigned to a specific case."""
        geo_name = self.case_geometry_map.get(case_id, self.default_geometry)
        return self.get_geometry(geo_name)

    def get_geometry_name_for_case(self, case_id: str) -> str:
        """Get the geometry name assigned to a specific case."""
        return self.case_geometry_map.get(case_id, self.default_geometry)

    def assign_geometry(self, case_id: str, geometry_name: str) -> None:
        """Assign a geometry to a case."""
        if geometry_name in self.geometries:
            self.case_geometry_map[case_id] = geometry_name
            # Update the case object too
            case = self.cases.get(case_id)
            if case:
                case.geometry_name = geometry_name

    def set_geometry(self, mac: float, ref_area: float, mrc: List[float],
                     units: str = "IPS", output_units: str = None,
                     span: float = None) -> None:
        """Set default geometry (backward-compatible)."""
        geo = self.geometries.setdefault(self.default_geometry, {})
        geo['mac'] = mac
        geo['ref_area'] = ref_area
        geo['mrc'] = mrc
        geo['units'] = units
        if span is not None:
            geo['span'] = span
        if output_units is not None:
            self.set_output_units(output_units)

    def set_output_units(self, units: str) -> None:
        """
        Set the output unit system for display and export.

        Parameters
        ----------
        units : str
            Output unit system: 'IPS', 'FPS', 'MKS', or 'CGS'
        """
        if units.upper() in ['IPS', 'FPS', 'MKS', 'CGS']:
            self.output_units = units.upper()
            self.output_units_changed.emit(self.output_units)

    # --- Multi-calibration methods ---

    @property
    def calibration_names(self) -> List[str]:
        """Get list of all defined calibration names."""
        return list(self.calibrations.keys())

    def add_calibration(self, name: str, params: dict) -> None:
        """Add a named calibration definition."""
        self.calibrations[name] = params
        if not self.default_calibration:
            self.default_calibration = name

    def remove_calibration(self, name: str) -> bool:
        """Remove a calibration definition. Cannot remove the last one."""
        if len(self.calibrations) <= 1:
            return False
        if name not in self.calibrations:
            return False
        del self.calibrations[name]
        if self.default_calibration == name:
            self.default_calibration = next(iter(self.calibrations))
        for case_id, cal_name in list(self.case_calibration_map.items()):
            if cal_name == name:
                del self.case_calibration_map[case_id]
        return True

    def rename_calibration(self, old_name: str, new_name: str) -> bool:
        """Rename a calibration definition."""
        if old_name not in self.calibrations or new_name in self.calibrations:
            return False
        self.calibrations[new_name] = self.calibrations.pop(old_name)
        if self.default_calibration == old_name:
            self.default_calibration = new_name
        for case_id, cal_name in self.case_calibration_map.items():
            if cal_name == old_name:
                self.case_calibration_map[case_id] = new_name
        return True

    def get_calibration(self, name: str) -> dict:
        """Get a calibration definition by name."""
        return self.calibrations.get(
            name, self.calibrations.get(self.default_calibration, {}))

    def get_calibration_for_case(self, case_id: str) -> dict:
        """Get the calibration assigned to a specific case."""
        cal_name = self.case_calibration_map.get(
            case_id, self.default_calibration)
        return self.get_calibration(cal_name)

    def get_calibration_name_for_case(self, case_id: str) -> str:
        """Get the calibration name assigned to a specific case."""
        return self.case_calibration_map.get(
            case_id, self.default_calibration)

    def assign_calibration(self, case_id: str, calibration_name: str) -> None:
        """Assign a calibration to a case."""
        if calibration_name in self.calibrations:
            self.case_calibration_map[case_id] = calibration_name
            case = self.cases.get(case_id)
            if case:
                case.calibration_name = calibration_name

    def set_calibration_files(self, balance_file: Optional[Path] = None,
                              pressure_file: Optional[Path] = None) -> None:
        """Set calibration file paths."""
        if balance_file:
            self.balance_cal_file = Path(balance_file)
        if pressure_file:
            self.pressure_cal_file = Path(pressure_file)

    def get_visible_cases(self) -> List[TestCase]:
        """Get all currently visible cases."""
        return self.cases.visible_cases

    def get_filtered_data(self, case: TestCase) -> Dict[str, np.ndarray]:
        """Get case data filtered by current filter settings."""
        if not case.has_data:
            return {}

        # Apply beta filter
        if self.filters.show_all_betas:
            return {
                'alphas': case.alphas,
                'betas': case.betas,
                'Cl': case.Cl,
                'Cd': case.Cd,
                'Cs': case.Cs,
                'CRoll': case.CRoll,
                'CPitch': case.CPitch,
                'CYaw': case.CYaw,
            }
        elif self.filters.selected_betas:
            # Return data for selected beta only
            beta = self.filters.selected_betas[0]
            return case.get_sweep_at_beta(beta)

        return {}

    def get_available_betas(self) -> List[float]:
        """Get all available beta values across all cases."""
        return self.cases.all_beta_values

    def get_available_alphas(self) -> List[float]:
        """Get all available alpha values across all cases."""
        return self.cases.all_alpha_values

    def clear_all(self) -> None:
        """Clear all data."""
        self.cases.clear()
        self._color_index = 0
        self._marker_index = 0
        self.cases_changed.emit()

    def emit_error(self, title: str, message: str) -> None:
        """Emit an error signal."""
        self.error_occurred.emit(title, message)

    def emit_progress(self, current: int, total: int) -> None:
        """Emit progress update."""
        self.processing_progress.emit(current, total)
