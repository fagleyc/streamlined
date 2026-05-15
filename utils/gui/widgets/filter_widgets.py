"""
Filter Widgets
==============

Widgets for filtering data by alpha, beta, Mach number, etc.
"""

from typing import List, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QFrame, QSlider, QDoubleSpinBox, QCheckBox,
    QButtonGroup, QToolButton, QGroupBox, QDialog, QScrollArea,
    QDialogButtonBox
)
from PyQt6.QtCore import pyqtSignal, Qt

from ..utils.themes import DarkTheme
from ..utils.icons import Icons


class MultiSelectFilter(QWidget):
    """
    Compact multi-select filter widget.

    Displays a QPushButton showing a selection summary (e.g. "Beta: All"
    or "Beta: 0.0, 5.0"). Clicking the button opens a popup dialog with
    a scrollable list of checkboxes and Select All / Apply / Cancel.

    Signals
    -------
    selection_changed : pyqtSignal(list)
        Emitted when the selection is applied. Payload is the list of
        selected float values.
    """

    selection_changed = pyqtSignal(list)

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self._label = label
        self._values: List[float] = []
        self._selected: List[float] = []
        self._all_selected = True
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._btn = QPushButton(f"{self._label}: All")
        self._btn.setMinimumWidth(100)
        self._btn.clicked.connect(self._open_popup)
        layout.addWidget(self._btn)

    def _update_button_text(self):
        """Update button label to reflect current selection."""
        if self._all_selected or len(self._selected) == len(self._values):
            self._btn.setText(f"{self._label}: All")
        elif len(self._selected) == 0:
            self._btn.setText(f"{self._label}: None")
        elif len(self._selected) <= 3:
            vals = ", ".join(f"{v:.1f}" for v in self._selected)
            self._btn.setText(f"{self._label}: {vals}")
        else:
            self._btn.setText(f"{self._label}: {len(self._selected)} sel")

    def _open_popup(self):
        """Open the multi-select dialog."""
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Select {self._label}")
        dlg.setMinimumWidth(200)

        layout = QVBoxLayout(dlg)

        # Select All checkbox
        chk_all = QCheckBox("Select All")
        chk_all.setChecked(self._all_selected)
        layout.addWidget(chk_all)

        # Scrollable area for value checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(4, 4, 4, 4)
        scroll_layout.setSpacing(2)

        checkboxes: List[QCheckBox] = []
        for val in self._values:
            chk = QCheckBox(f"{val:.1f}")
            chk.setChecked(self._all_selected or val in self._selected)
            checkboxes.append(chk)
            scroll_layout.addWidget(chk)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        # Wire Select All to toggle individual checkboxes
        def on_select_all(state):
            checked = state == Qt.CheckState.Checked.value
            for cb in checkboxes:
                cb.setChecked(checked)

        chk_all.stateChanged.connect(on_select_all)

        # Apply / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_selected = []
            for i, chk in enumerate(checkboxes):
                if chk.isChecked():
                    new_selected.append(self._values[i])

            all_checked = len(new_selected) == len(self._values)
            self._all_selected = all_checked
            self._selected = sorted(new_selected) if not all_checked else list(self._values)
            self._update_button_text()
            self.selection_changed.emit(list(self._selected))

    # ---- public API ----

    def set_values(self, values: List[float]):
        """Set available values and select all by default."""
        self._values = sorted(values)
        self._selected = list(self._values)
        self._all_selected = True
        self._update_button_text()

    def get_selected(self) -> List[float]:
        """Return the currently selected values."""
        if self._all_selected:
            return list(self._values)
        return list(self._selected)

    def is_all_selected(self) -> bool:
        """True when all values are selected (i.e. no filtering)."""
        return self._all_selected

    def select_all(self):
        """Programmatically select all values."""
        self._all_selected = True
        self._selected = list(self._values)
        self._update_button_text()


class FilterToolbar(QWidget):
    """
    Toolbar with filter controls.

    Signals
    -------
    filters_changed : pyqtSignal
        Emitted when any filter changes
    """

    filters_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(16)

        # Filter icon
        filter_icon = QLabel()
        filter_icon.setPixmap(Icons.filter().pixmap(20, 20))
        layout.addWidget(filter_icon)

        # Alpha multi-select filter
        self.alpha_filter = MultiSelectFilter("Alpha")
        self.alpha_filter.selection_changed.connect(lambda _: self.filters_changed.emit())
        layout.addWidget(self.alpha_filter)

        # Beta multi-select filter
        self.beta_filter = MultiSelectFilter("Beta")
        self.beta_filter.selection_changed.connect(lambda _: self.filters_changed.emit())
        layout.addWidget(self.beta_filter)

        layout.addSpacing(16)

        # Mach number filter
        mach_group = QWidget()
        mach_layout = QHBoxLayout(mach_group)
        mach_layout.setContentsMargins(0, 0, 0, 0)
        mach_layout.setSpacing(4)

        mach_label = QLabel("Mach:")
        mach_layout.addWidget(mach_label)

        self.cmb_mach = QComboBox()
        self.cmb_mach.addItem("All", None)
        self.cmb_mach.setMinimumWidth(80)
        self.cmb_mach.currentIndexChanged.connect(lambda: self.filters_changed.emit())
        mach_layout.addWidget(self.cmb_mach)

        layout.addWidget(mach_group)

        # Reynolds number filter
        re_group = QWidget()
        re_layout = QHBoxLayout(re_group)
        re_layout.setContentsMargins(0, 0, 0, 0)
        re_layout.setSpacing(4)

        re_label = QLabel("Re:")
        re_layout.addWidget(re_label)

        self.cmb_reynolds = QComboBox()
        self.cmb_reynolds.addItem("All", None)
        self.cmb_reynolds.setMinimumWidth(100)
        self.cmb_reynolds.currentIndexChanged.connect(lambda: self.filters_changed.emit())
        re_layout.addWidget(self.cmb_reynolds)

        layout.addWidget(re_group)

        # Velocity filter
        vel_group = QWidget()
        vel_layout = QHBoxLayout(vel_group)
        vel_layout.setContentsMargins(0, 0, 0, 0)
        vel_layout.setSpacing(4)

        vel_label = QLabel("U_inf:")
        vel_layout.addWidget(vel_label)

        self.cmb_velocity = QComboBox()
        self.cmb_velocity.addItem("All", None)
        self.cmb_velocity.setMinimumWidth(90)
        self.cmb_velocity.currentIndexChanged.connect(lambda: self.filters_changed.emit())
        vel_layout.addWidget(self.cmb_velocity)

        layout.addWidget(vel_group)

        layout.addStretch()

        # Reset button
        self.btn_reset = QPushButton("Reset Filters")
        self.btn_reset.setIcon(Icons.refresh())
        self.btn_reset.clicked.connect(self.reset_filters)
        layout.addWidget(self.btn_reset)

    # ---- Alpha / Beta accessors ----

    def set_alpha_values(self, alphas: List[float]):
        """Set available alpha values."""
        self.alpha_filter.set_values(alphas)

    def set_beta_values(self, betas: List[float]):
        """Set available beta values."""
        self.beta_filter.set_values(betas)

    def get_selected_alphas(self) -> Optional[List[float]]:
        """Selected alpha values, or None when all selected (no filter)."""
        if self.alpha_filter.is_all_selected():
            return None
        return self.alpha_filter.get_selected()

    def get_selected_betas(self) -> Optional[List[float]]:
        """Selected beta values, or None when all selected (no filter)."""
        if self.beta_filter.is_all_selected():
            return None
        return self.beta_filter.get_selected()

    # ---- Other filter accessors ----

    def set_mach_values(self, machs: List[float]):
        """Set available Mach numbers."""
        self.cmb_mach.blockSignals(True)
        self.cmb_mach.clear()
        self.cmb_mach.addItem("All", None)

        for mach in machs:
            self.cmb_mach.addItem(f"M = {mach:.3f}", mach)

        self.cmb_mach.blockSignals(False)

    def set_reynolds_values(self, reynolds: List[float]):
        """Set available Reynolds numbers."""
        self.cmb_reynolds.blockSignals(True)
        self.cmb_reynolds.clear()
        self.cmb_reynolds.addItem("All", None)

        for re in reynolds:
            if re >= 1e6:
                self.cmb_reynolds.addItem(f"{re/1e6:.2f}M", re)
            else:
                self.cmb_reynolds.addItem(f"{re:.0f}", re)

        self.cmb_reynolds.blockSignals(False)

    def set_velocity_values(self, velocities: List[float]):
        """Set available velocities (m/s)."""
        self.cmb_velocity.blockSignals(True)
        self.cmb_velocity.clear()
        self.cmb_velocity.addItem("All", None)

        for vel in velocities:
            self.cmb_velocity.addItem(f"{vel:.1f} m/s", vel)

        self.cmb_velocity.blockSignals(False)

    def get_selected_mach(self) -> Optional[float]:
        """Get selected Mach filter."""
        return self.cmb_mach.currentData()

    def get_selected_reynolds(self) -> Optional[float]:
        """Get selected Reynolds number filter."""
        return self.cmb_reynolds.currentData()

    def get_selected_velocity(self) -> Optional[float]:
        """Get selected velocity filter."""
        return self.cmb_velocity.currentData()

    def reset_filters(self):
        """Reset all filters to default (select all)."""
        self.alpha_filter.select_all()
        self.beta_filter.select_all()
        self.cmb_mach.setCurrentIndex(0)
        self.cmb_reynolds.setCurrentIndex(0)
        self.cmb_velocity.setCurrentIndex(0)
        self.filters_changed.emit()


class PlotTypeSelector(QWidget):
    """
    Widget for selecting plot type.

    Signals
    -------
    plot_type_changed : pyqtSignal(str)
        Emitted when plot type changes
    """

    plot_type_changed = pyqtSignal(str)

    PLOT_TYPES = [
        ("CL vs Alpha", "CL_VS_ALPHA", "Lift coefficient vs angle of attack"),
        ("CD vs Alpha", "CD_VS_ALPHA", "Drag coefficient vs angle of attack"),
        ("Drag Polar", "CL_VS_CD", "Lift vs drag (drag polar)"),
        ("Cm vs Alpha", "CM_VS_ALPHA", "Pitching moment vs angle of attack"),
        ("Cm vs CL", "CM_VS_CL", "Pitching moment vs lift coefficient"),
        ("L/D vs Alpha", "LD_VS_ALPHA", "Lift-to-drag ratio vs angle of attack"),
        ("CY vs Alpha", "CY_VS_ALPHA", "Side force coefficient vs angle of attack"),
        ("Cl (roll) vs Alpha", "CROLL_VS_ALPHA", "Rolling moment vs angle of attack"),
        ("Cn (yaw) vs Alpha", "CYAW_VS_ALPHA", "Yawing moment vs angle of attack"),
        ("Lateral", "LATERAL_VS_BETA", "Lateral-directional coefficients"),
        # Stability derivatives (central-difference)
        ("Cma vs Alpha", "CMA_VS_ALPHA",
         "Longitudinal stability slope dCm/dalpha (central diff)"),
        ("CLa vs Alpha", "CLA_VS_ALPHA",
         "Lift curve slope dCL/dalpha (central diff)"),
        ("Static Margin vs Alpha", "SM_VS_ALPHA",
         "Static margin -Cma/CLa"),
        ("CYb vs Alpha", "CYB_VS_ALPHA",
         "Side-force derivative dCY/dbeta (requires >= 2 betas)"),
        ("Cnb vs Alpha", "CNB_VS_ALPHA",
         "Directional stability dCn/dbeta (requires >= 2 betas)"),
        ("Clb vs Alpha", "CLB_VS_ALPHA",
         "Lateral stability dCl/dbeta (requires >= 2 betas)"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        label = QLabel("Plot Type:")
        layout.addWidget(label)

        self.cmb_plot_type = QComboBox()
        self.cmb_plot_type.setMinimumWidth(150)

        for display_name, value, tooltip in self.PLOT_TYPES:
            self.cmb_plot_type.addItem(display_name, value)
            idx = self.cmb_plot_type.count() - 1
            self.cmb_plot_type.setItemData(idx, tooltip, Qt.ItemDataRole.ToolTipRole)

        self.cmb_plot_type.currentIndexChanged.connect(self._on_changed)
        layout.addWidget(self.cmb_plot_type)

    def _on_changed(self, index):
        value = self.cmb_plot_type.currentData()
        self.plot_type_changed.emit(value)

    def get_plot_type(self) -> str:
        """Get current plot type."""
        return self.cmb_plot_type.currentData()

    def set_plot_type(self, plot_type: str):
        """Set current plot type."""
        for i in range(self.cmb_plot_type.count()):
            if self.cmb_plot_type.itemData(i) == plot_type:
                self.cmb_plot_type.setCurrentIndex(i)
                break
