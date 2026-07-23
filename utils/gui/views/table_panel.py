"""
Table Panel
===========

Panel for displaying coefficient data in tabular format with tunnel conditions.
"""

import numpy as np
from pathlib import Path
from typing import Optional, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QComboBox, QFrame, QHeaderView, QFileDialog,
    QAbstractItemView, QMessageBox, QGroupBox, QGridLayout, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

from ..models.data_model import DataModel
from ..models.case import TestCase
from ..models.settings import AppSettings
from ..utils.themes import DarkTheme
from ..utils.icons import Icons
from .. import __version__

# Import units module for conversion
try:
    from utils.windtunnel.units import UnitSystem, UnitConverter, UNIT_LABELS
    UNITS_AVAILABLE = True
except ImportError:
    UNITS_AVAILABLE = False


class NumericTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically instead of lexicographically."""

    def __lt__(self, other):
        try:
            return (float(self.data(Qt.ItemDataRole.UserRole))
                    < float(other.data(Qt.ItemDataRole.UserRole)))
        except (TypeError, ValueError):
            return super().__lt__(other)


class TunnelConditionsPanel(QWidget):
    """Panel displaying tunnel conditions for the selected case."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._output_units = "IPS"
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Title
        title = QLabel("Tunnel Conditions")
        title.setStyleSheet(f"font-weight: bold; color: {DarkTheme.TEXT_PRIMARY};")
        layout.addWidget(title)

        # Conditions grid
        self._grid = QGridLayout()
        self._grid.setSpacing(8)

        # Create labels for each condition (will be updated based on units)
        self._condition_labels = {}
        self._value_labels = {}

        conditions = [
            ("Q", "lbl_Q"),
            ("U_inf", "lbl_velocity"),
            ("Mach", "lbl_mach"),
            ("Re", "lbl_reynolds"),
            ("rho", "lbl_density"),
            ("T", "lbl_temperature"),
        ]

        for i, (key, attr_name) in enumerate(conditions):
            row = i // 2
            col = (i % 2) * 2

            label = QLabel(self._get_label_text(key))
            label.setStyleSheet(f"color: {DarkTheme.TEXT_SECONDARY};")
            self._grid.addWidget(label, row, col)
            self._condition_labels[key] = label

            value_label = QLabel("--")
            value_label.setStyleSheet(f"""
                color: {DarkTheme.TEXT_PRIMARY};
                font-family: monospace;
                font-weight: bold;
            """)
            setattr(self, attr_name, value_label)
            self._value_labels[key] = value_label
            self._grid.addWidget(value_label, row, col + 1)

        layout.addLayout(self._grid)
        layout.addStretch()

    def _get_label_text(self, key: str) -> str:
        """Get label text with appropriate unit based on output_units."""
        if UNITS_AVAILABLE:
            try:
                labels = UNIT_LABELS[UnitSystem[self._output_units]]
                if key == "Q":
                    return f"Q ({labels.pressure}):"
                elif key == "U_inf":
                    return f"U_inf ({labels.velocity}):"
                elif key == "rho":
                    return f"rho ({labels.density}):"
                elif key == "T":
                    return f"T ({labels.temperature}):"
            except (KeyError, AttributeError):
                pass

        # Fallback labels
        fallback = {
            "Q": "Q (psi):",
            "U_inf": "U_inf (m/s):",
            "Mach": "Mach:",
            "Re": "Re:",
            "rho": "rho (kg/m^3):",
            "T": "T (degC):",
        }
        return fallback.get(key, f"{key}:")

    def set_output_units(self, units: str):
        """Update output units and refresh labels."""
        self._output_units = units
        self._update_labels()

    def _update_labels(self):
        """Update all condition labels with current units."""
        for key, label in self._condition_labels.items():
            label.setText(self._get_label_text(key))

    def update_conditions(self, case: Optional[TestCase]):
        """Update displayed conditions from a test case."""
        if case is None or not case.has_data:
            self.lbl_Q.setText("--")
            self.lbl_velocity.setText("--")
            self.lbl_mach.setText("--")
            self.lbl_reynolds.setText("--")
            self.lbl_density.setText("--")
            self.lbl_temperature.setText("--")
            return

        # Get converter if available
        converter = None
        if UNITS_AVAILABLE:
            try:
                converter = UnitConverter(UnitSystem[self._output_units])
            except (KeyError, AttributeError):
                pass

        # Get mean values and convert if needed
        # Dynamic pressure (stored in psi)
        Q_raw = None
        if case.pressure is not None:
            Q_raw = case.pressure
        elif len(case.dynamic_pressures) > 0:
            Q_raw = float(np.mean(case.dynamic_pressures))

        if Q_raw is not None:
            Q_display = converter.convert_pressure(Q_raw) if converter else Q_raw
            self.lbl_Q.setText(f"{Q_display:.4f}")
        else:
            self.lbl_Q.setText("--")

        # Velocity (stored in m/s)
        V_raw = None
        if case.velocity is not None:
            V_raw = case.velocity
        elif len(case.velocities) > 0:
            V_raw = float(np.mean(case.velocities))

        if V_raw is not None:
            V_display = converter.convert_velocity(V_raw) if converter else V_raw
            self.lbl_velocity.setText(f"{V_display:.2f}")
        else:
            self.lbl_velocity.setText("--")

        # Mach (dimensionless - no conversion)
        if case.mach_number is not None:
            self.lbl_mach.setText(f"{case.mach_number:.4f}")
        elif len(case.machs) > 0:
            self.lbl_mach.setText(f"{np.mean(case.machs):.4f}")
        else:
            self.lbl_mach.setText("--")

        # Reynolds number (dimensionless - no conversion)
        if case.reynolds_number is not None:
            self.lbl_reynolds.setText(f"{case.reynolds_number:.2e}")
        elif len(case.reynolds) > 0:
            self.lbl_reynolds.setText(f"{np.mean(case.reynolds):.2e}")
        else:
            self.lbl_reynolds.setText("--")

        # Density (stored in kg/m^3)
        rho_raw = None
        if case.density is not None:
            rho_raw = case.density
        elif len(case.densities) > 0:
            rho_raw = float(np.mean(case.densities))

        if rho_raw is not None:
            rho_display = converter.convert_density(rho_raw) if converter else rho_raw
            self.lbl_density.setText(f"{rho_display:.6f}")
        else:
            self.lbl_density.setText("--")

        # Temperature (stored in Celsius)
        T_raw = None
        if case.temperature is not None:
            T_raw = case.temperature
        elif len(case.temperatures) > 0:
            T_raw = float(np.mean(case.temperatures))

        if T_raw is not None:
            T_display = converter.convert_temperature(T_raw) if converter else T_raw
            self.lbl_temperature.setText(f"{T_display:.1f}")
        else:
            self.lbl_temperature.setText("--")


class TablePanel(QWidget):
    """
    Panel for displaying aerodynamic coefficient data in a table.
    """

    def __init__(self, model: DataModel, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.model = model
        self.settings = settings
        self._output_units = "IPS"
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QFrame()
        toolbar.setStyleSheet(f"""
            QFrame {{
                background-color: {DarkTheme.BACKGROUND_LIGHT};
                border-bottom: 1px solid {DarkTheme.BORDER};
            }}
        """)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 4, 8, 4)

        # Case selector
        toolbar_layout.addWidget(QLabel("Case:"))

        self.cmb_case = QComboBox()
        self.cmb_case.setMinimumWidth(200)
        self.cmb_case.currentIndexChanged.connect(self._update_table)
        toolbar_layout.addWidget(self.cmb_case)

        toolbar_layout.addStretch()

        # Include tunnel conditions checkbox
        self.chk_tunnel_conditions = QCheckBox("Include Tunnel Conditions")
        self.chk_tunnel_conditions.setChecked(True)
        self.chk_tunnel_conditions.stateChanged.connect(self._update_table)
        toolbar_layout.addWidget(self.chk_tunnel_conditions)

        # (Export buttons removed - use File > Export... instead.)

        toolbar_layout.addSpacing(16)

        # Include unsteady (time-series) data option for HDF5/MAT exports
        self.chk_include_unsteady = QCheckBox("Include Unsteady")
        self.chk_include_unsteady.setChecked(False)
        self.chk_include_unsteady.setToolTip(
            "Include full time-series data in HDF5/MAT exports\n"
            "(in addition to averaged values)"
        )
        toolbar_layout.addWidget(self.chk_include_unsteady)

        layout.addWidget(toolbar)

        # Content area with table and conditions panel
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Table
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setSortingEnabled(True)

        # Style the table
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {DarkTheme.BACKGROUND_LIGHT};
                gridline-color: {DarkTheme.BORDER};
                border: none;
            }}
            QTableWidget::item {{
                padding: 6px;
            }}
            QTableWidget::item:selected {{
                background-color: {DarkTheme.SELECTION};
            }}
            QTableWidget::item:alternate {{
                background-color: {DarkTheme.BACKGROUND_LIGHTER};
            }}
            QHeaderView::section {{
                background-color: {DarkTheme.SURFACE};
                color: {DarkTheme.TEXT_PRIMARY};
                padding: 8px;
                border: none;
                border-right: 1px solid {DarkTheme.BORDER};
                border-bottom: 1px solid {DarkTheme.BORDER};
                font-weight: bold;
            }}
        """)

        content_layout.addWidget(self.table, stretch=1)

        # Tunnel conditions panel (sidebar)
        self.conditions_panel = TunnelConditionsPanel()
        self.conditions_panel.setFixedWidth(200)
        self.conditions_panel.setStyleSheet(f"""
            QWidget {{
                background-color: {DarkTheme.BACKGROUND_LIGHTER};
                border-left: 1px solid {DarkTheme.BORDER};
            }}
        """)
        content_layout.addWidget(self.conditions_panel)

        layout.addWidget(content)

        # Status bar
        self.status_bar = QFrame()
        self.status_bar.setStyleSheet(f"""
            QFrame {{
                background-color: {DarkTheme.BACKGROUND_LIGHTER};
                border-top: 1px solid {DarkTheme.BORDER};
            }}
        """)
        status_layout = QHBoxLayout(self.status_bar)
        status_layout.setContentsMargins(8, 4, 8, 4)

        self.lbl_status = QLabel("No data")
        self.lbl_status.setStyleSheet(f"color: {DarkTheme.TEXT_SECONDARY};")
        status_layout.addWidget(self.lbl_status)

        status_layout.addStretch()

        self.lbl_selection = QLabel("")
        self.lbl_selection.setStyleSheet(f"color: {DarkTheme.TEXT_SECONDARY};")
        status_layout.addWidget(self.lbl_selection)

        layout.addWidget(self.status_bar)

        # Initialize columns
        self._update_columns()

    def _get_unit_labels(self):
        """Get unit labels for the current output system."""
        if UNITS_AVAILABLE:
            try:
                return UNIT_LABELS[UnitSystem[self._output_units]]
            except (KeyError, AttributeError):
                pass
        # Fallback to default IPS labels
        from collections import namedtuple
        Labels = namedtuple('Labels', ['pressure', 'velocity', 'density', 'temperature', 'force', 'moment'])
        return Labels('psi', 'm/s', 'kg/m^3', 'degC', 'lbf', 'lb-in')

    def _update_columns(self):
        """Update table columns based on checkbox state and output units."""
        labels = self._get_unit_labels()

        base_columns = [
            ("Alpha", "Alpha [deg]"),
            ("Beta", "Beta [deg]"),
            ("Cl", "CL"),
            ("Cd", "CD"),
            ("Cs", "CY"),
            ("CRoll", "Cl (roll)"),
            ("CPitch", "Cm"),
            ("CYaw", "Cn"),
            ("L/D", "L/D"),
        ]

        # Force columns with dynamic unit labels
        force_columns = [
            ("Lift", f"Lift [{labels.force}]"),
            ("Drag", f"Drag [{labels.force}]"),
            ("Side", f"Side [{labels.force}]"),
        ]

        # Moment columns with dynamic unit labels
        moment_columns = [
            ("M_Roll", f"M_Roll [{labels.moment}]"),
            ("M_Pitch", f"M_Pitch [{labels.moment}]"),
            ("M_Yaw", f"M_Yaw [{labels.moment}]"),
        ]

        # Balance element force columns with dynamic unit labels
        # Use moment-balance names if balance_config is 'Moment'
        bal_cfg = getattr(self.model, 'balance_config', 'Force')
        if bal_cfg == 'Moment':
            e_names = ['AftPitch', 'AftYaw', 'FwdPitch', 'FwdYaw',
                       'Axial', 'Roll']
        else:
            e_names = ['N1', 'N2', 'Y1', 'Y2', 'Axial', 'Roll']
        element_columns = [
            ("elem_N1", f"{e_names[0]} [{labels.force}]"),
            ("elem_N2", f"{e_names[1]} [{labels.force}]"),
            ("elem_Y1", f"{e_names[2]} [{labels.force}]"),
            ("elem_Y2", f"{e_names[3]} [{labels.force}]"),
            ("elem_Ax", f"{e_names[4]} [{labels.force}]"),
            ("elem_Roll", f"{e_names[5]} [{labels.force}]"),
        ]

        # Tunnel condition columns with dynamic unit labels
        tunnel_columns = [
            ("Q", f"Q [{labels.pressure}]"),
            ("Mach", "Mach"),
            ("Re", "Re"),
            ("U_inf", f"U_inf [{labels.velocity}]"),
            ("rho", f"rho [{labels.density}]"),
            ("T", f"T [{labels.temperature}]"),
            ("P_tot", f"P_tot [{labels.pressure}]"),
        ]

        # Custom calculator columns - one per unique custom variable
        # name across all loaded cases.  Custom variables are
        # dimensionless / user-defined so we don't add a unit suffix.
        custom_columns = []
        seen_custom = set()
        for c in self.model.cases:
            cv = getattr(c, 'custom_vars', None) or {}
            for name in cv.keys():
                if name not in seen_custom:
                    seen_custom.add(name)
                    custom_columns.append((f'custom:{name}', name))

        # Build column list based on checkbox
        self._columns = base_columns + force_columns + moment_columns + element_columns
        if self.chk_tunnel_conditions.isChecked():
            self._columns = self._columns + tunnel_columns
        # Append custom columns last so they don't shift built-ins
        self._columns = self._columns + custom_columns

        self.table.setColumnCount(len(self._columns))
        self.table.setHorizontalHeaderLabels([col[1] for col in self._columns])

        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        for i in range(len(self._columns)):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)

    def set_output_units(self, units: str):
        """Set output units and refresh the display."""
        self._output_units = units
        self.conditions_panel.set_output_units(units)
        self._update_table()

    def _connect_signals(self):
        """Connect signals."""
        self.model.cases_changed.connect(self._update_case_list)
        self.table.itemSelectionChanged.connect(self._update_selection_info)
        self.model.output_units_changed.connect(self.set_output_units)

    def _update_case_list(self):
        """Update the case dropdown."""
        self.cmb_case.blockSignals(True)
        self.cmb_case.clear()

        self.cmb_case.addItem("All Cases", None)

        for case in self.model.cases:
            self.cmb_case.addItem(case.name, case.id)

        self.cmb_case.blockSignals(False)
        self._update_table()

    def _update_table(self):
        """Update table contents."""
        self._update_columns()
        self.table.setRowCount(0)

        case_id = self.cmb_case.currentData()

        rows = []

        if case_id is None:
            # Show all cases
            for case in self.model.cases:
                if case.has_data:
                    rows.extend(self._get_case_rows(case))
            # Update conditions panel with first case
            if len(self.model.cases) > 0:
                first_case = next(iter(self.model.cases), None)
                self.conditions_panel.update_conditions(first_case)
            else:
                self.conditions_panel.update_conditions(None)
        else:
            # Show single case
            case = self.model.cases.get(case_id)
            if case and case.has_data:
                rows = self._get_case_rows(case)
            self.conditions_panel.update_conditions(case)

        # Disable sorting while populating to prevent Qt from re-sorting
        # as items are inserted (which scrambles our lexsort order)
        self.table.setSortingEnabled(False)

        # Populate table
        self.table.setRowCount(len(rows))

        for row_idx, row_data in enumerate(rows):
            for col_idx, value in enumerate(row_data):
                if isinstance(value, float):
                    if abs(value) > 1e6:
                        text = f"{value:.2e}"
                    elif abs(value) < 0.001 and value != 0:
                        text = f"{value:.6f}"
                    elif abs(value) < 0.01:
                        text = f"{value:.5f}"
                    else:
                        text = f"{value:.4f}"
                else:
                    text = str(value)

                item = NumericTableWidgetItem(text)
                if isinstance(value, (int, float)):
                    item.setData(Qt.ItemDataRole.UserRole,
                                 float(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

                # Color-code L/D column (index 8 in base columns)
                if col_idx == 8 and isinstance(value, float):
                    if value > 10:
                        item.setForeground(QColor(DarkTheme.SUCCESS))
                    elif value < 0:
                        item.setForeground(QColor(DarkTheme.ERROR))

                self.table.setItem(row_idx, col_idx, item)

        # Re-enable sorting after population is complete
        self.table.setSortingEnabled(True)

        self._update_status(len(rows))

    def _get_case_rows(self, case: TestCase) -> List[List]:
        """Extract rows from a test case with unit conversions applied."""
        rows = []

        # Get converter for unit conversions
        converter = None
        if UNITS_AVAILABLE:
            try:
                converter = UnitConverter(UnitSystem[self._output_units])
            except (KeyError, AttributeError):
                pass

        alphas = case.alphas.flatten()
        betas = case.betas.flatten()

        # Sort by beta first, then alpha within each beta group.
        # Round betas to nearest integer for grouping (handles measurement
        # noise — values within ~0.5° are treated as the same beta).
        sort_order = np.lexsort((alphas, np.round(betas)))
        alphas = alphas[sort_order]
        betas = betas[sort_order]

        Cl = case.Cl.flatten()[sort_order]
        Cd = case.Cd.flatten()[sort_order]
        Cs = case.Cs.flatten()[sort_order]
        CRoll = case.CRoll.flatten()[sort_order]
        CPitch = case.CPitch.flatten()[sort_order]
        CYaw = case.CYaw.flatten()[sort_order]

        # Helper to sort optional arrays by the same order
        def _sort(arr):
            flat = arr.flatten()
            return flat[sort_order] if len(flat) >= len(sort_order) else None

        # Get force/moment arrays (may be empty)
        Lift = _sort(case.lift_forces) if len(case.lift_forces) > 0 else None
        Drag = _sort(case.drag_forces) if len(case.drag_forces) > 0 else None
        Side = _sort(case.side_forces) if len(case.side_forces) > 0 else None
        M_Roll = _sort(case.roll_moments) if len(case.roll_moments) > 0 else None
        M_Pitch = _sort(case.pitch_moments) if len(case.pitch_moments) > 0 else None
        M_Yaw = _sort(case.yaw_moments) if len(case.yaw_moments) > 0 else None

        # Get balance element force arrays (may be empty)
        eN1 = _sort(case.elem_N1) if len(case.elem_N1) > 0 else None
        eN2 = _sort(case.elem_N2) if len(case.elem_N2) > 0 else None
        eY1 = _sort(case.elem_Y1) if len(case.elem_Y1) > 0 else None
        eY2 = _sort(case.elem_Y2) if len(case.elem_Y2) > 0 else None
        eAx = _sort(case.elem_Ax) if len(case.elem_Ax) > 0 else None
        eRl = _sort(case.elem_Roll) if len(case.elem_Roll) > 0 else None

        # Get tunnel conditions arrays (may be shorter or empty)
        Q = _sort(case.dynamic_pressures) if len(case.dynamic_pressures) > 0 else None
        Mach = _sort(case.machs) if len(case.machs) > 0 else None
        Re = _sort(case.reynolds) if len(case.reynolds) > 0 else None
        U_inf = _sort(case.velocities) if len(case.velocities) > 0 else None
        rho = _sort(case.densities) if len(case.densities) > 0 else None
        T = _sort(case.temperatures) if len(case.temperatures) > 0 else None
        Ptot = _sort(case.total_pressures) if len(case.total_pressures) > 0 else None

        # Pre-sort custom variables to the same row order so columns
        # line up.  Missing values for a case yield NaN per row.
        custom_sorted: dict = {}
        case_custom = getattr(case, 'custom_vars', None) or {}
        for name, arr in case_custom.items():
            try:
                custom_sorted[name] = _sort(arr)
            except Exception:
                custom_sorted[name] = None

        include_tunnel = self.chk_tunnel_conditions.isChecked()

        for i in range(len(alphas)):
            ld = Cl[i] / Cd[i] if Cd[i] != 0 else float('inf')

            # Base columns (coefficients are dimensionless - no conversion)
            row = [
                alphas[i], betas[i],
                Cl[i], Cd[i], Cs[i],
                CRoll[i], CPitch[i], CYaw[i],
                ld
            ]

            # Force columns (convert from lbf)
            lift_val = Lift[i] if Lift is not None and i < len(Lift) else 0.0
            drag_val = Drag[i] if Drag is not None and i < len(Drag) else 0.0
            side_val = Side[i] if Side is not None and i < len(Side) else 0.0

            if converter:
                lift_val = converter.convert_force(lift_val)
                drag_val = converter.convert_force(drag_val)
                side_val = converter.convert_force(side_val)

            row.extend([lift_val, drag_val, side_val])

            # Moment columns (convert from lb-in)
            m_roll_val = M_Roll[i] if M_Roll is not None and i < len(M_Roll) else 0.0
            m_pitch_val = M_Pitch[i] if M_Pitch is not None and i < len(M_Pitch) else 0.0
            m_yaw_val = M_Yaw[i] if M_Yaw is not None and i < len(M_Yaw) else 0.0

            if converter:
                m_roll_val = converter.convert_moment(m_roll_val)
                m_pitch_val = converter.convert_moment(m_pitch_val)
                m_yaw_val = converter.convert_moment(m_yaw_val)

            row.extend([m_roll_val, m_pitch_val, m_yaw_val])

            # Balance element force columns (convert from lbf)
            n1_val = eN1[i] if eN1 is not None and i < len(eN1) else 0.0
            n2_val = eN2[i] if eN2 is not None and i < len(eN2) else 0.0
            y1_val = eY1[i] if eY1 is not None and i < len(eY1) else 0.0
            y2_val = eY2[i] if eY2 is not None and i < len(eY2) else 0.0
            ax_val = eAx[i] if eAx is not None and i < len(eAx) else 0.0
            rl_val = eRl[i] if eRl is not None and i < len(eRl) else 0.0

            if converter:
                n1_val = converter.convert_force(n1_val)
                n2_val = converter.convert_force(n2_val)
                y1_val = converter.convert_force(y1_val)
                y2_val = converter.convert_force(y2_val)
                ax_val = converter.convert_force(ax_val)
                rl_val = converter.convert_force(rl_val)

            row.extend([n1_val, n2_val, y1_val, y2_val, ax_val, rl_val])

            # Tunnel condition columns
            if include_tunnel:
                # Q (convert from psi)
                q_val = Q[i] if Q is not None and i < len(Q) else 0.0
                if converter:
                    q_val = converter.convert_pressure(q_val)
                row.append(q_val)

                # Mach (dimensionless - no conversion)
                row.append(Mach[i] if Mach is not None and i < len(Mach) else 0.0)

                # Re (dimensionless - no conversion)
                row.append(Re[i] if Re is not None and i < len(Re) else 0.0)

                # U_inf (convert from m/s)
                u_val = U_inf[i] if U_inf is not None and i < len(U_inf) else 0.0
                if converter:
                    u_val = converter.convert_velocity(u_val)
                row.append(u_val)

                # rho (convert from kg/m^3)
                rho_val = rho[i] if rho is not None and i < len(rho) else 0.0
                if converter:
                    rho_val = converter.convert_density(rho_val)
                row.append(rho_val)

                # T (convert from Celsius)
                t_val = T[i] if T is not None and i < len(T) else 0.0
                if converter:
                    t_val = converter.convert_temperature(t_val)
                row.append(t_val)

                # P_tot (convert from psi)
                ptot_val = Ptot[i] if Ptot is not None and i < len(Ptot) else 0.0
                if converter:
                    ptot_val = converter.convert_pressure(ptot_val)
                row.append(ptot_val)

            # Custom calculator columns (no unit conversion - they are
            # user-defined and the user expression dictates the units)
            for col_key, _label in self._columns:
                if not col_key.startswith('custom:'):
                    continue
                name = col_key[len('custom:'):]
                arr = custom_sorted.get(name)
                if arr is None or i >= len(arr):
                    row.append(float('nan'))
                else:
                    row.append(float(arr[i]))

            rows.append(row)

        return rows

    def _update_status(self, row_count: int):
        """Update status bar."""
        self.lbl_status.setText(f"{row_count} data point{'s' if row_count != 1 else ''}")

    def _update_selection_info(self):
        """Update selection info."""
        n_cols = len(self._columns)
        selected = len(self.table.selectedItems()) // n_cols if n_cols > 0 else 0
        if selected > 0:
            self.lbl_selection.setText(f"{selected} row{'s' if selected != 1 else ''} selected")
        else:
            self.lbl_selection.setText("")

    def _export_csv(self):
        """Export table to CSV."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export to CSV",
            self.settings.last_export_directory,
            "CSV Files (*.csv);;All Files (*.*)"
        )
        if filepath:
            self.settings.last_export_directory = str(Path(filepath).parent)
            self._do_export(filepath, 'csv')

    def _export_excel(self):
        """Export table to Excel."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export to Excel",
            self.settings.last_export_directory,
            "Excel Files (*.xlsx);;All Files (*.*)"
        )
        if filepath:
            self.settings.last_export_directory = str(Path(filepath).parent)
            self._do_export(filepath, 'excel')

    def _export_hdf5(self):
        """Export table to HDF5."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export to HDF5",
            self.settings.last_export_directory,
            "HDF5 Files (*.h5 *.hdf5);;All Files (*.*)"
        )
        if filepath:
            self.settings.last_export_directory = str(Path(filepath).parent)
            self._do_export(filepath, 'hdf5')

    def _export_mat(self):
        """Export table to MATLAB .mat file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export to MAT",
            self.settings.last_export_directory,
            "MAT Files (*.mat);;All Files (*.*)"
        )
        if filepath:
            self.settings.last_export_directory = str(Path(filepath).parent)
            self._do_export(filepath, 'mat')

    def _sanitize_sheet_name(self, name: str) -> str:
        """Sanitize a string for use as an Excel sheet name.

        Excel sheet-name rules: max 31 chars; cannot contain
        : \\ / ? * [ ]; cannot be blank; cannot start or end with an
        apostrophe.  Returns '' when nothing valid remains so the
        caller can substitute a generic 'Case_NNN' name.
        """
        for ch in ['/', '\\', '[', ']', '*', '?', ':']:
            name = name.replace(ch, '_')
        name = name.strip().strip("'").strip()
        name = name[:31]
        return name

    def _build_case_header(self, case: TestCase) -> list:
        """Build header key-value pairs summarizing a case for Excel export."""
        header = [("Case Name", case.name)]

        # Alpha range
        if case.has_data:
            alphas = case.alphas.flatten()
            header.append(("Alpha Range",
                           f"{np.min(alphas):.1f} to {np.max(alphas):.1f} deg"))
            # Beta values
            betas = sorted(set(round(float(b), 1) for b in case.betas.flatten()))
            header.append(("Beta Values",
                           ", ".join(f"{b:.1f}" for b in betas) + " deg"))
            header.append(("Data Points", str(case.n_points)))

        # Set up converter and unit labels so header values match output units
        converter = None
        labels = None
        if UNITS_AVAILABLE:
            try:
                converter = UnitConverter(UnitSystem[self._output_units])
                labels = UNIT_LABELS[UnitSystem[self._output_units]]
            except (KeyError, AttributeError):
                pass

        # Tunnel conditions
        if case.mach_number is not None:
            header.append(("Mach", f"{case.mach_number:.4f}"))
        elif len(case.machs) > 0:
            header.append(("Mach", f"{np.mean(case.machs):.4f}"))

        if case.reynolds_number is not None:
            header.append(("Reynolds Number", f"{case.reynolds_number:.2e}"))
        elif len(case.reynolds) > 0:
            header.append(("Reynolds Number", f"{np.mean(case.reynolds):.2e}"))

        # Dynamic pressure (stored in psi, convert to output)
        q_val = None
        if case.pressure is not None:
            q_val = case.pressure
        elif len(case.dynamic_pressures) > 0:
            q_val = float(np.mean(case.dynamic_pressures))
        if q_val is not None:
            if converter:
                q_val = converter.convert_pressure(q_val)
            unit = f" [{labels.pressure}]" if labels else ""
            header.append((f"Dynamic Pressure (Q){unit}", f"{q_val:.4f}"))

        # Total pressure (stored in psi, convert to output)
        if len(case.total_pressures) > 0:
            ptot_val = float(np.mean(case.total_pressures))
            if converter:
                ptot_val = converter.convert_pressure(ptot_val)
            unit = f" [{labels.pressure}]" if labels else ""
            header.append((f"Total Pressure (P_tot){unit}", f"{ptot_val:.4f}"))

        # Velocity (stored in m/s, convert to output)
        u_val = None
        if case.velocity is not None:
            u_val = case.velocity
        elif len(case.velocities) > 0:
            u_val = float(np.mean(case.velocities))
        if u_val is not None:
            if converter:
                u_val = converter.convert_velocity(u_val)
            unit = f" [{labels.velocity}]" if labels else ""
            header.append((f"Velocity (U_inf){unit}", f"{u_val:.2f}"))

        # Density (stored in kg/m^3, convert to output)
        rho_val = None
        if case.density is not None:
            rho_val = case.density
        elif len(case.densities) > 0:
            rho_val = float(np.mean(case.densities))
        if rho_val is not None:
            if converter:
                rho_val = converter.convert_density(rho_val)
            unit = f" [{labels.density}]" if labels else ""
            header.append((f"Density (rho){unit}", f"{rho_val:.6f}"))

        # Temperature (stored in Celsius, convert to output)
        t_val = None
        if case.temperature is not None:
            t_val = case.temperature
        elif len(case.temperatures) > 0:
            t_val = float(np.mean(case.temperatures))
        if t_val is not None:
            if converter:
                t_val = converter.convert_temperature(t_val)
            unit = f" [{labels.temperature}]" if labels else ""
            header.append((f"Temperature{unit}", f"{t_val:.1f}"))

        # Calibration information
        header.extend(self._build_calibration_header())

        # Geometry
        header.append(("MAC", f"{self.model.mac}"))
        header.append(("Ref Area", f"{self.model.ref_area}"))
        header.append(("Span", f"{self.model.span}"))
        header.append(("MRC", f"{self.model.mrc}"))
        header.append(("Input Units", self.model.units))
        header.append(("Output Units", self.model.output_units))

        # Metadata
        for key, val in case.metadata.items():
            header.append((str(key), str(val)))

        return header

    # MATLAB's namelengthmax is 63; struct field / variable names longer
    # than this are rejected by MATLAB on load even though scipy will
    # happily write them.
    _MATLAB_NAME_MAX = 63

    def _sanitize_matlab_name(self, name: str) -> str:
        """Sanitize a string into a valid MATLAB variable/field name.

        Guarantees the result is a legal MATLAB identifier: starts with
        a letter, contains only [A-Za-z0-9_], is non-empty, and is no
        longer than 63 characters (MATLAB's namelengthmax).  scipy will
        write over-length or hyphenated names, but real MATLAB refuses
        them on load, so we must enforce this here.
        """
        import re
        # Replace common symbols with underscore / strip brackets
        name = name.replace(' ', '_').replace('[', '').replace(']', '')
        name = name.replace('/', '_').replace('(', '').replace(')', '')
        name = name.replace('^', '').replace('.', '_')
        name = name.replace('-', '_')  # hyphens are illegal in identifiers
        # Remove any remaining non-alphanumeric/underscore chars
        name = re.sub(r'[^a-zA-Z0-9_]', '', name)
        # Ensure starts with a letter
        if name and not name[0].isalpha():
            name = 'x_' + name
        # Ensure not empty
        if not name:
            name = 'unnamed'
        # Enforce MATLAB's 63-character identifier limit
        if len(name) > self._MATLAB_NAME_MAX:
            name = name[:self._MATLAB_NAME_MAX]
        return name

    def _build_case_meta(self, case: TestCase) -> dict:
        """Build metadata dict for MAT export (becomes a MATLAB struct)."""
        meta = {}
        meta['name'] = case.name
        meta['run_number'] = np.float64(case.run_number if case.run_number else 0)

        if case.has_data:
            alphas = case.alphas.flatten()
            meta['alpha_min'] = np.float64(np.min(alphas))
            meta['alpha_max'] = np.float64(np.max(alphas))
            betas = sorted(set(round(float(b), 1)
                               for b in case.betas.flatten()))
            meta['beta_values'] = np.array(betas) if betas else np.array([])
            meta['n_points'] = np.float64(case.n_points)

        # Set up converter so meta values are in output units
        converter = None
        if UNITS_AVAILABLE:
            try:
                converter = UnitConverter(UnitSystem[self._output_units])
            except (KeyError, AttributeError):
                pass

        # Tunnel conditions (mean values, converted to output units)
        if case.mach_number is not None:
            meta['Mach'] = np.float64(case.mach_number)
        elif len(case.machs) > 0:
            meta['Mach'] = np.float64(np.mean(case.machs))

        if case.reynolds_number is not None:
            meta['Reynolds'] = np.float64(case.reynolds_number)
        elif len(case.reynolds) > 0:
            meta['Reynolds'] = np.float64(np.mean(case.reynolds))

        q_val = None
        if case.pressure is not None:
            q_val = case.pressure
        elif len(case.dynamic_pressures) > 0:
            q_val = float(np.mean(case.dynamic_pressures))
        if q_val is not None:
            meta['Q'] = np.float64(
                converter.convert_pressure(q_val) if converter else q_val)

        if len(case.total_pressures) > 0:
            ptot_val = float(np.mean(case.total_pressures))
            meta['P_tot'] = np.float64(
                converter.convert_pressure(ptot_val) if converter else ptot_val)

        u_val = None
        if case.velocity is not None:
            u_val = case.velocity
        elif len(case.velocities) > 0:
            u_val = float(np.mean(case.velocities))
        if u_val is not None:
            meta['U_inf'] = np.float64(
                converter.convert_velocity(u_val) if converter else u_val)

        rho_val = None
        if case.density is not None:
            rho_val = case.density
        elif len(case.densities) > 0:
            rho_val = float(np.mean(case.densities))
        if rho_val is not None:
            meta['rho'] = np.float64(
                converter.convert_density(rho_val) if converter else rho_val)

        t_val = None
        if case.temperature is not None:
            t_val = case.temperature
        elif len(case.temperatures) > 0:
            t_val = float(np.mean(case.temperatures))
        if t_val is not None:
            meta['temperature'] = np.float64(
                converter.convert_temperature(t_val) if converter else t_val)

        # Date
        if case.date:
            meta['date'] = case.date.strftime('%Y-%m-%d %H:%M:%S')

        # Calibration information
        meta['calibration'] = self._build_calibration_meta_dict()

        # Geometry
        meta['geometry'] = {
            'mac': np.float64(self.model.mac),
            'ref_area': np.float64(self.model.ref_area),
            'span': np.float64(self.model.span),
            'mrc': np.array(self.model.mrc, dtype=np.float64),
            'input_units': self.model.units,
            'output_units': self.model.output_units,
        }

        # Additional metadata from case
        for key, val in case.metadata.items():
            safe_key = self._sanitize_matlab_name(str(key))
            if isinstance(val, (int, float)):
                meta[safe_key] = np.float64(val)
            else:
                meta[safe_key] = str(val)

        return meta

    def _build_raw_dict(self, case: TestCase) -> dict:
        """
        Build a 'raw' sub-struct of mean values per test point for export.

        Includes calibrated tunnel quantities (pdiff, ptot, ttot) and
        body-frame balance forces/moments (Fx, Fy, Fz, Mx, My, Mz),
        plus the selected output unit system label. All values are
        converted to the user's selected output units, NaN-filled if
        the per-point series is missing.
        """
        # Set up converter
        converter = None
        if UNITS_AVAILABLE:
            try:
                converter = UnitConverter(UnitSystem[self._output_units])
            except (KeyError, AttributeError):
                pass

        raw = {'units': self._output_units}

        daq = getattr(case, 'daq', None)
        red = getattr(daq, 'red', None) if daq is not None else None
        if not red:
            return raw

        n = len(red)
        # Pre-allocate arrays
        pdiff = np.full(n, np.nan)   # Differential pressure (Q in psi)
        ptot = np.full(n, np.nan)    # Total pressure (Pa internally)
        ttot = np.full(n, np.nan)    # Total temperature (C internally)
        Fx = np.full(n, np.nan)
        Fy = np.full(n, np.nan)
        Fz = np.full(n, np.nan)
        Mx = np.full(n, np.nan)
        My = np.full(n, np.nan)
        Mz = np.full(n, np.nan)
        alpha = np.full(n, np.nan)
        beta = np.full(n, np.nan)

        for i, pt in enumerate(red):
            # Tunnel conditions (means)
            tc = getattr(pt, 'tunnel', None)
            if tc is not None:
                if hasattr(tc, 'Q') and len(tc.Q) > 0:
                    pdiff[i] = float(np.mean(tc.Q))  # psi
                if hasattr(tc, 'P_tot') and len(tc.P_tot) > 0:
                    # P_tot is in Pa internally; convert to psi for the
                    # uniform 'pressure' converter
                    ptot[i] = float(np.mean(tc.P_tot)) / 6894.75729
                if hasattr(tc, 'T0') and len(tc.T0) > 0:
                    ttot[i] = float(np.mean(tc.T0))  # Celsius
                elif hasattr(tc, 'T') and len(tc.T) > 0:
                    ttot[i] = float(np.mean(tc.T))

            # BRF forces (air-on minus mean of air-off, in IPS lbf / lb-in)
            brf_on = getattr(pt, 'brf_on', None)
            brf_off = getattr(pt, 'brf_off', None)

            def _aero(attr):
                if brf_on is None:
                    return np.nan
                on_arr = getattr(brf_on, attr, None)
                if on_arr is None or len(on_arr) == 0:
                    return np.nan
                if brf_off is not None:
                    off_arr = getattr(brf_off, attr, None)
                    if off_arr is not None and len(off_arr) > 0:
                        return float(np.mean(on_arr) - np.mean(off_arr))
                return float(np.mean(on_arr))

            Fx[i] = _aero('Fx')
            Fy[i] = _aero('Fy')
            Fz[i] = _aero('Fz')
            Mx[i] = _aero('Mx')
            My[i] = _aero('My')
            Mz[i] = _aero('Mz')

            # Alpha / Beta (means)
            if hasattr(pt, 'alpha') and len(pt.alpha) > 0:
                alpha[i] = float(np.mean(pt.alpha))
            if hasattr(pt, 'beta') and len(pt.beta) > 0:
                beta[i] = float(np.mean(pt.beta))

        # Apply unit conversions
        if converter:
            pdiff = converter.convert_pressure(pdiff)
            ptot = converter.convert_pressure(ptot)
            ttot = converter.convert_temperature(ttot)
            Fx = converter.convert_force(Fx)
            Fy = converter.convert_force(Fy)
            Fz = converter.convert_force(Fz)
            Mx = converter.convert_moment(Mx)
            My = converter.convert_moment(My)
            Mz = converter.convert_moment(Mz)

        raw['alpha'] = alpha
        raw['beta'] = beta
        raw['pdiff'] = pdiff
        raw['ptot'] = ptot
        raw['ttot'] = ttot
        raw['Fx'] = Fx
        raw['Fy'] = Fy
        raw['Fz'] = Fz
        raw['Mx'] = Mx
        raw['My'] = My
        raw['Mz'] = Mz
        return raw

    # ------------------------------------------------------------------
    # Categorized export structure (MAT / HDF5)
    # ------------------------------------------------------------------

    def _case_extra_scalars(self, case: TestCase) -> dict:
        try:
            from utils.windtunnel.calculator import geometry_scalars
            geo = self.model.get_geometry_for_case(case.id)
            return geometry_scalars(geo)
        except Exception:
            return {}

    def _per_point_means_grouped(self, case: TestCase):
        """
        Iterate case.daq.red and build:
          - means: dict[var_name -> array shaped like case.alphas]
          - raw_means: dict[raw_channel_name -> array shaped like alphas]
        Returns (means, raw_means).
        """
        daq = getattr(case, 'daq', None)
        red = getattr(daq, 'red', None) if daq is not None else None
        if not red:
            return {}, {}

        try:
            from utils.windtunnel.calculator import build_namespace
        except Exception:
            return {}, {}

        extra = self._case_extra_scalars(case)

        per_var: dict = {}
        per_raw: dict = {}
        for pt in red:
            try:
                ns = build_namespace(pt, extra_scalars=extra)
            except Exception:
                continue
            for name, val in ns.items():
                try:
                    arr = np.asarray(val, dtype=float)
                    if arr.ndim == 0:
                        m = float(arr)
                    elif arr.size == 0:
                        m = float('nan')
                    else:
                        m = float(np.mean(arr))
                except Exception:
                    m = float('nan')
                per_var.setdefault(name, []).append(m)
            air_on = getattr(pt, 'air_on', None) or {}
            for name, val in air_on.items():
                try:
                    arr = np.asarray(val, dtype=float)
                    m = float(np.mean(arr)) if arr.size > 0 else float('nan')
                except Exception:
                    m = float('nan')
                per_raw.setdefault(name, []).append(m)

        # Reshape via sort_idx + alphas.shape
        sort_idx = None
        ss = getattr(daq, 'ss', None)
        if ss is not None and hasattr(ss, 'indices'):
            try:
                sort_idx = np.asarray(ss.indices)
            except Exception:
                sort_idx = None
        target = case.alphas.shape

        def _shape(lst):
            flat = np.array(lst, dtype=float)
            if (sort_idx is not None and sort_idx.size > 0
                    and len(flat) >= sort_idx.size):
                try:
                    flat = flat[sort_idx]
                except Exception:
                    pass
            try:
                if target != flat.shape:
                    flat = flat.reshape(target)
            except ValueError:
                pass
            return flat

        means = {k: _shape(v) for k, v in per_var.items()}
        raw_means = {k: _shape(v) for k, v in per_raw.items()}
        return means, raw_means

    def _build_categorized_struct(self, case: TestCase) -> dict:
        """
        Build a nested dict for MAT/HDF5 export organized by the
        calculator's variable categories:
            case.Tunnel_Conditions.Q, ...
            case.BRF_Forces.Fx, ...
            case.WRF_Forces.Lift, ...
            case.Pressure_Ports.p0, ...
            case.Balance_Channels.N1, ...
            case.Coefficients.Cl, ...
            case.Geometry.MAC, ...
            case.Position.alpha, ...
            case.Raw.<every raw channel>
        Plus case.Custom and case.Custom_std if any rules are active.
        Returns a dict ready to merge into the case_struct.
        """
        means, raw_means = self._per_point_means_grouped(case)
        try:
            from utils.windtunnel.calculator import categorize_variables
        except Exception:
            return {}

        out: dict = {}
        cats = categorize_variables(sorted(means.keys()))
        # Map category display name -> safe struct field name
        category_keys = {
            'Pressure Ports': 'Pressure_Ports',
            'Balance Channels': 'Balance_Channels',
            'Tunnel Conditions': 'Tunnel_Conditions',
            'BRF Forces': 'BRF_Forces',
            'WRF Forces': 'WRF_Forces',
            'Coefficients': 'Coefficients',
            'Geometry': 'Geometry',
            'Position / Time': 'Position',
            'Constants': 'Constants',
            'Other': 'Other',
        }
        for cat_name, names in cats.items():
            if not names:
                continue
            key = category_keys.get(cat_name,
                                    self._sanitize_matlab_name(cat_name))
            sub = {}
            for n in names:
                arr = means.get(n)
                if arr is None:
                    continue
                safe = self._sanitize_matlab_name(n)
                sub[safe] = np.asarray(arr)
            if sub:
                out[key] = sub

        # All raw signals
        if raw_means:
            raw_sub = {}
            for n, arr in raw_means.items():
                safe = self._sanitize_matlab_name(n)
                raw_sub[safe] = np.asarray(arr)
            out['Raw'] = raw_sub

        return out

    def _build_calibration_header(self) -> list:
        """Build calibration key-value pairs for Excel/header export."""
        header = []
        if self.model.balance_cal_file:
            header.append(("Balance Cal File",
                           str(self.model.balance_cal_file)))
        if self.model.pressure_cal_file:
            header.append(("Pressure Cal File",
                           str(self.model.pressure_cal_file)))
        header.append(("Cal Type", self.model.cal_type))
        header.append(("Facility", self.model.facility))
        header.append(("Balance Config", self.model.balance_config))
        return header

    def _build_calibration_meta_dict(self) -> dict:
        """Build calibration metadata dict for MAT/HDF5 export."""
        cal = {}
        if self.model.balance_cal_file:
            cal['balance_cal_file'] = str(self.model.balance_cal_file)
        if self.model.pressure_cal_file:
            cal['pressure_cal_file'] = str(self.model.pressure_cal_file)
        cal['cal_type'] = self.model.cal_type
        cal['facility'] = self.model.facility
        cal['balance_config'] = self.model.balance_config
        return cal

    def _extract_unsteady_point(self, pt) -> dict:
        """
        Extract all time-series data from a single ReducedDataPoint.

        Returns a dict of numpy arrays keyed by channel name,
        suitable for writing to HDF5 datasets or MAT struct fields.
        """
        data = {}

        # Time vector
        if hasattr(pt, 'time') and len(pt.time) > 0:
            data['time'] = np.asarray(pt.time)

        # Angles
        if hasattr(pt, 'alpha') and len(pt.alpha) > 0:
            data['alpha'] = np.asarray(pt.alpha)
        if hasattr(pt, 'beta') and len(pt.beta) > 0:
            data['beta'] = np.asarray(pt.beta)

        # Aerodynamic coefficients
        if hasattr(pt, 'coeffs') and pt.coeffs is not None:
            for attr in ('Cl', 'Cd', 'Cs', 'CRoll', 'CPitch', 'CYaw'):
                val = getattr(pt.coeffs, attr, None)
                if val is not None and len(val) > 0:
                    data[f'coeff_{attr}'] = np.asarray(val)

        # WRF forces
        if hasattr(pt, 'wrf_aero') and pt.wrf_aero is not None:
            for attr in ('Lift', 'Drag', 'Side', 'Roll', 'Pitch', 'Yaw'):
                val = getattr(pt.wrf_aero, attr, None)
                if val is not None and len(val) > 0:
                    data[f'force_{attr}'] = np.asarray(val)

        # Tunnel conditions
        if hasattr(pt, 'tunnel') and pt.tunnel is not None:
            for attr in ('Q', 'U_inf', 'Mach', 'Re', 'rho', 'T', 'P_tot'):
                val = getattr(pt.tunnel, attr, None)
                if val is not None and len(val) > 0:
                    data[f'tunnel_{attr}'] = np.asarray(val)

        # Balance element forces (air-on minus air-off)
        if hasattr(pt, 'brf_on') and pt.brf_on is not None:
            elems_on = getattr(pt.brf_on, 'elements', None)
            elems_off = (getattr(pt.brf_off, 'elements', None)
                         if hasattr(pt, 'brf_off')
                         and pt.brf_off is not None else None)
            if (elems_on is not None and hasattr(elems_on, 'ndim')
                    and elems_on.ndim == 2 and elems_on.shape[1] >= 6):
                if (elems_off is not None
                        and hasattr(elems_off, 'ndim')
                        and elems_off.ndim == 2
                        and elems_off.shape[1] >= 6):
                    elems = elems_on - np.mean(elems_off, axis=0)
                else:
                    elems = elems_on
                bal_cfg = getattr(self.model, 'balance_config', 'Force')
                if bal_cfg == 'Moment':
                    elem_names = ['AftPitch', 'AftYaw', 'FwdPitch',
                                  'FwdYaw', 'Axial', 'Roll']
                else:
                    elem_names = ['N1', 'N2', 'Y1', 'Y2', 'Axial', 'Roll']
                for col_idx, name in enumerate(elem_names):
                    data[f'element_{name}'] = elems[:, col_idx]

        # Raw channels (air-on)
        if hasattr(pt, 'air_on') and pt.air_on is not None:
            skip = {'Time', 'Alpha', 'Beta'}
            for key, val in pt.air_on.items():
                if key not in skip:
                    arr = np.asarray(val)
                    if len(arr) > 0:
                        data[f'raw_{key}'] = arr

        return data

    def do_export(self, config: dict):
        """Public entry point for export from ExportDialog config."""
        filepath = config.get('filepath', '')
        fmt = config.get('format', 'csv')
        if not filepath:
            return
        self._do_export(filepath, fmt)

    def _export_coe(self, output_dir: str):
        """Export reduced cases to legacy Reduce2 .COE files.

        One .COE file is written per case per unique beta. The output
        directory is used as the destination; filenames are derived from
        each case name and beta value.
        """
        from pathlib import Path as _Path

        try:
            from utils.windtunnel.coe_writer import write_coe_files
        except Exception as e:
            QMessageBox.critical(
                self, "COE Export Failed",
                f"Could not import COE writer module:\n{e}")
            return

        # Resolve output directory: if the user typed a file path, use
        # its parent. If it's already a directory, use as-is.
        p = _Path(output_dir)
        if p.exists() and p.is_file():
            out_dir = p.parent
        else:
            out_dir = p
        out_dir.mkdir(parents=True, exist_ok=True)

        # Determine which cases to export
        case_id = self.cmb_case.currentData()
        cases_to_export = []
        if case_id is None:
            for case in self.model.cases:
                if case.has_data:
                    cases_to_export.append(case)
        else:
            case = self.model.cases.get(case_id)
            if case and case.has_data:
                cases_to_export.append(case)

        if not cases_to_export:
            QMessageBox.warning(
                self, "COE Export",
                "No reduced cases available to export.")
            return

        # Cases without raw DAQ data attached cannot be exported to COE
        # because the format requires per-point body-frame elements and
        # tunnel conditions; only the in-memory case object has those.
        bal_file = (str(self.model.balance_cal_file)
                    if self.model.balance_cal_file else '')

        all_written = []
        skipped = []
        for case in cases_to_export:
            if not getattr(case, 'daq', None):
                skipped.append(case.name)
                continue
            geo = self.model.get_geometry_for_case(case.id)
            try:
                paths = write_coe_files(
                    case=case,
                    output_dir=str(out_dir),
                    case_geometry=geo,
                    balance_cal_file=bal_file)
                all_written.extend(paths)
            except Exception as e:
                import traceback
                traceback.print_exc()
                QMessageBox.critical(
                    self, "COE Export Failed",
                    f"Failed to write COE for case '{case.name}':\n\n"
                    f"{type(e).__name__}: {e}")
                return

        # Summary dialog
        msg_lines = [f"Wrote {len(all_written)} .COE file(s) to:",
                     str(out_dir), ""]
        msg_lines.extend(f"  - {_Path(p).name}" for p in all_written[:10])
        if len(all_written) > 10:
            msg_lines.append(f"  ... and {len(all_written) - 10} more")
        if skipped:
            msg_lines.append("")
            msg_lines.append(
                f"Skipped (no raw DAQ data attached): {', '.join(skipped)}")
        QMessageBox.information(
            self, "COE Export Complete", '\n'.join(msg_lines))

    def _do_export(self, filepath: str, format: str):
        """Perform the export."""
        try:
            import pandas as pd

            if format == 'coe':
                self._export_coe(filepath)
                return

            if format == 'excel':
                # Export each case to its own sheet
                case_id = self.cmb_case.currentData()
                cases_to_export = []

                if case_id is None:
                    # All cases
                    for case in self.model.cases:
                        if case.has_data:
                            cases_to_export.append(case)
                else:
                    case = self.model.cases.get(case_id)
                    if case and case.has_data:
                        cases_to_export.append(case)

                if not cases_to_export:
                    QMessageBox.warning(self, "Export Error", "No data to export.")
                    return

                col_headers = [col[1] for col in self._columns]
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    used_names = set()
                    n_cases = len(cases_to_export)
                    key_width = max(2, len(str(n_cases)))
                    for case_i, case in enumerate(cases_to_export, start=1):
                        rows = self._get_case_rows(case)
                        df = pd.DataFrame(rows, columns=col_headers)

                        # Readable sheet name when possible, else a
                        # generic Case_NN.  The full configuration name
                        # always appears in cell A1 ("Case Name" header)
                        # so nothing is lost regardless of the tab label.
                        base = self._sanitize_sheet_name(case.name)
                        if not base:
                            base = f"Case_{case_i:0{key_width}d}"
                        sheet_name = base
                        dup = 1
                        while sheet_name.lower() in used_names:
                            suffix = f"_{dup}"
                            sheet_name = base[:31 - len(suffix)] + suffix
                            dup += 1
                        used_names.add(sheet_name.lower())

                        # Write header summary then data below it
                        header_rows = self._build_case_header(case)
                        startrow = len(header_rows) + 1  # +1 blank row
                        df.to_excel(writer, sheet_name=sheet_name,
                                    index=False, startrow=startrow)

                        # Write header info into the sheet
                        ws = writer.sheets[sheet_name]
                        for r, (key, val) in enumerate(header_rows, start=1):
                            ws.cell(row=r, column=1, value=key)
                            ws.cell(row=r, column=2, value=val)
            elif format == 'mat':
                import scipy.io

                # Export each case as a named MATLAB struct
                case_id = self.cmb_case.currentData()
                cases_to_export = []
                if case_id is None:
                    for case in self.model.cases:
                        if case.has_data:
                            cases_to_export.append(case)
                else:
                    case = self.model.cases.get(case_id)
                    if case and case.has_data:
                        cases_to_export.append(case)

                if not cases_to_export:
                    QMessageBox.warning(self, "Export Error",
                                        "No data to export.")
                    return

                col_headers = [col[1] for col in self._columns]
                mat_dict = {}

                # Top-level variable names use a generic, always-valid
                # convention (case_001, case_002, ...) so long or
                # hyphenated configuration names never break MATLAB's
                # 63-char identifier limit.  The real configuration name
                # is preserved inside each struct (.name) and in a
                # top-level 'case_index' manifest, so no data is lost.
                n_cases = len(cases_to_export)
                key_width = max(3, len(str(n_cases)))
                idx_keys = []
                idx_names = []
                idx_runs = []

                for case_i, case in enumerate(cases_to_export, start=1):
                    # Categorized struct: case.Tunnel_Conditions.Q,
                    # case.BRF_Forces.Fx, case.Pressure_Ports.p0, etc.,
                    # plus case.Raw.<every raw channel>.
                    case_struct = self._build_categorized_struct(case)

                    # Metadata sub-struct (calibration, geometry, etc.)
                    case_struct['meta'] = self._build_case_meta(case)

                    # Name pointer fields so the generic key can be
                    # mapped back to the real configuration name.
                    case_key = f"case_{case_i:0{key_width}d}"
                    case_struct['name'] = str(case.name)
                    case_struct['key'] = case_key
                    case_struct['run_number'] = np.float64(
                        case.run_number if case.run_number else 0)

                    # Custom calculator outputs: per-point mean + std
                    custom_means = getattr(case, 'custom_vars', None) or {}
                    custom_stds = getattr(case, 'custom_vars_std', None) or {}
                    if custom_means:
                        case_struct['Custom'] = {
                            self._sanitize_matlab_name(name): np.asarray(arr)
                            for name, arr in custom_means.items()
                        }
                    if custom_stds:
                        case_struct['Custom_std'] = {
                            self._sanitize_matlab_name(name): np.asarray(arr)
                            for name, arr in custom_stds.items()
                        }

                    # Add unsteady (time-series) data if requested
                    if self.chk_include_unsteady.isChecked():
                        daq = getattr(case, 'daq', None)
                        red = (getattr(daq, 'red', None)
                               if daq is not None else None)
                        if red:
                            unsteady = {}
                            for pt_idx, pt in enumerate(red):
                                pt_data = (
                                    self._extract_unsteady_point(pt))
                                if pt_data:
                                    pt_struct = {}
                                    for ch_name, arr in pt_data.items():
                                        safe_ch = (
                                            self._sanitize_matlab_name(
                                                ch_name))
                                        pt_struct[safe_ch] = arr
                                    # Store mean alpha/beta
                                    if 'alpha' in pt_data:
                                        pt_struct['mean_alpha'] = (
                                            np.float64(np.mean(
                                                pt_data['alpha'])))
                                    if 'beta' in pt_data:
                                        pt_struct['mean_beta'] = (
                                            np.float64(np.mean(
                                                pt_data['beta'])))
                                    unsteady[f'point_{pt_idx}'] = (
                                        pt_struct)
                            if unsteady:
                                case_struct['unsteady'] = unsteady

                    mat_dict[case_key] = case_struct
                    idx_keys.append(case_key)
                    idx_names.append(str(case.name))
                    idx_runs.append(
                        float(case.run_number) if case.run_number else 0.0)

                # Top-level manifest mapping generic keys -> real names.
                # In MATLAB:  case_index.names{2} gives case_002's name.
                mat_dict['case_index'] = {
                    'keys': np.array(idx_keys, dtype=object),
                    'names': np.array(idx_names, dtype=object),
                    'run_numbers': np.array(idx_runs, dtype=float),
                    'count': np.float64(n_cases),
                }

                scipy.io.savemat(filepath, mat_dict, long_field_names=False)

            elif format == 'csv':
                # CSV - collect from the table widget
                data = {col[1]: [] for col in self._columns}

                for row in range(self.table.rowCount()):
                    for col, col_info in enumerate(self._columns):
                        item = self.table.item(row, col)
                        try:
                            value = float(item.text())
                        except ValueError:
                            value = item.text()
                        data[col_info[1]].append(value)

                df = pd.DataFrame(data)
                df.to_csv(filepath, index=False)

            elif format == 'hdf5':
                import h5py

                # Build per-case structure with averaged data,
                # optional unsteady data, and metadata
                case_id = self.cmb_case.currentData()
                cases_to_export = []
                if case_id is None:
                    for case in self.model.cases:
                        if case.has_data:
                            cases_to_export.append(case)
                else:
                    case = self.model.cases.get(case_id)
                    if case and case.has_data:
                        cases_to_export.append(case)

                if not cases_to_export:
                    QMessageBox.warning(self, "Export Error",
                                        "No data to export.")
                    return

                include_unsteady = self.chk_include_unsteady.isChecked()
                col_headers = [col[1] for col in self._columns]
                used_names = {}

                with h5py.File(filepath, 'w') as hf:
                    # Global metadata
                    hf.attrs['streamlined_version'] = __version__
                    hf.attrs['output_units'] = self.model.output_units
                    hf.attrs['n_cases'] = len(cases_to_export)

                    # Global calibration group
                    cal_grp = hf.create_group('calibration')
                    cal_meta = self._build_calibration_meta_dict()
                    for k, v in cal_meta.items():
                        cal_grp.attrs[k] = v

                    # Global geometry group
                    geo_grp = hf.create_group('geometry')
                    geo_grp.attrs['mac'] = self.model.mac
                    geo_grp.attrs['ref_area'] = self.model.ref_area
                    geo_grp.attrs['span'] = self.model.span
                    geo_grp.create_dataset('mrc',
                                           data=np.array(self.model.mrc))
                    geo_grp.attrs['input_units'] = self.model.units
                    geo_grp.attrs['output_units'] = self.model.output_units

                    for case in cases_to_export:
                        # Sanitize case name for HDF5 group
                        safe_name = case.name.replace('/', '_')

                        # Deduplicate
                        if safe_name in used_names:
                            used_names[safe_name] += 1
                            safe_name = (f"{safe_name}"
                                         f"_{used_names[safe_name]}")
                        else:
                            used_names[safe_name] = 0

                        case_grp = hf.create_group(safe_name)

                        # Case metadata as attributes
                        case_grp.attrs['name'] = case.name
                        case_grp.attrs['run_number'] = (
                            case.run_number if case.run_number else 0)
                        if case.date:
                            case_grp.attrs['date'] = (
                                case.date.strftime('%Y-%m-%d %H:%M:%S'))
                        if case.mach_number is not None:
                            case_grp.attrs['Mach'] = case.mach_number
                        if case.reynolds_number is not None:
                            case_grp.attrs['Reynolds'] = (
                                case.reynolds_number)

                        # Categorized data: per-point means grouped
                        # by Tunnel_Conditions / BRF_Forces / etc.,
                        # plus a Raw group containing every raw DAQ
                        # channel's per-point mean.
                        categorized = self._build_categorized_struct(
                            case)
                        for cat_key, sub in categorized.items():
                            sub_grp = case_grp.create_group(cat_key)
                            for name, arr in sub.items():
                                try:
                                    sub_grp.create_dataset(
                                        name, data=np.asarray(arr))
                                except Exception:
                                    pass

                        # Custom calculator outputs: means and stds
                        custom_means = getattr(
                            case, 'custom_vars', None) or {}
                        custom_stds = getattr(
                            case, 'custom_vars_std', None) or {}
                        if custom_means:
                            cust_grp = case_grp.create_group('Custom')
                            for name, arr in custom_means.items():
                                try:
                                    cust_grp.create_dataset(
                                        name, data=np.asarray(arr))
                                except Exception:
                                    pass
                        if custom_stds:
                            cust_std_grp = case_grp.create_group(
                                'Custom_std')
                            for name, arr in custom_stds.items():
                                try:
                                    cust_std_grp.create_dataset(
                                        name, data=np.asarray(arr))
                                except Exception:
                                    pass

                        # Unsteady (time-series) data sub-group
                        if include_unsteady:
                            daq = getattr(case, 'daq', None)
                            red = (getattr(daq, 'red', None)
                                   if daq is not None else None)
                            if red:
                                unst_grp = case_grp.create_group(
                                    'unsteady')
                                for pt_idx, pt in enumerate(red):
                                    pt_data = (
                                        self._extract_unsteady_point(pt))
                                    if pt_data:
                                        pt_grp = unst_grp.create_group(
                                            f'point_{pt_idx}')
                                        # Store mean alpha/beta as attrs
                                        if 'alpha' in pt_data:
                                            pt_grp.attrs['mean_alpha'] = (
                                                float(np.mean(
                                                    pt_data['alpha'])))
                                        if 'beta' in pt_data:
                                            pt_grp.attrs['mean_beta'] = (
                                                float(np.mean(
                                                    pt_data['beta'])))
                                        for ch_name, arr in (
                                                pt_data.items()):
                                            pt_grp.create_dataset(
                                                ch_name, data=arr)

            QMessageBox.information(
                self, "Export Complete",
                f"Data exported successfully to:\n{filepath}"
            )

        except ImportError:
            QMessageBox.warning(
                self, "Export Error",
                "pandas is required for export. Install with: pip install pandas"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Export Error",
                f"Failed to export data:\n{str(e)}"
            )
