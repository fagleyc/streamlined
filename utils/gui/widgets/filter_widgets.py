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

    def get_selected_mach(self) -> Optional[float]:
        """Get selected Mach filter."""
        return self.cmb_mach.currentData()

    def reset_filters(self):
        """Reset all filters to default (select all)."""
        self.alpha_filter.select_all()
        self.beta_filter.select_all()
        self.cmb_mach.setCurrentIndex(0)
        self.filters_changed.emit()


class PlotTypeSelector(QWidget):
    """
    The plot's axis selector: a Y Axis list of every available quantity,
    grouped (coefficients, wind-axis loads, body-axis loads, balance
    elements, tunnel conditions, attitude, stability derivatives,
    calculator outputs), and an X Axis list.

    The groups and labels come from :mod:`utils.gui.plot_variables`, so a
    quantity added there appears here. Group headers are disabled rows.

    Signals
    -------
    plot_type_changed : pyqtSignal(str)
        Emitted with the Y variable key whenever either axis changes
        (name kept for the panel wiring).
    """

    plot_type_changed = pyqtSignal(str)

    # X-axis variables. The empty value means alpha (the historical
    # default). Mach / Re / q / U_inf are SPEED-sweep variables: picking
    # one makes each alpha/beta point its own trace across the speed
    # steps, instead of each speed step its own alpha sweep.
    X_AXIS_VARS = [
        ("α  alpha (default)", "",
         "Angle of attack [deg]; one trace per sideslip and speed step"),
        ("α  alpha", "Alpha",
         "Angle of attack [deg]; one trace per sideslip and speed step"),
        ("β  beta", "Beta",
         "Sideslip angle [deg]; one trace per angle of attack"),
        ("Mach", "Mach",
         "Measured freestream Mach; one trace per alpha/beta, so a "
         "speed sweep reads as a curve"),
        ("Re", "Re",
         "Reynolds number; one trace per alpha/beta"),
        ("q", "Q",
         "Dynamic pressure, in the output unit system; "
         "one trace per alpha/beta"),
        ("U∞", "U_inf",
         "Freestream velocity, in the output unit system; "
         "one trace per alpha/beta"),
        ("CL", "Cl", "Lift coefficient on the x axis (drag polar: CD)"),
        ("CD", "Cd", "Drag coefficient on the x axis"),
        ("CY", "Cs", "Side-force coefficient on the x axis"),
        ("Cl (roll)", "CRoll", "Rolling-moment coefficient on the x axis"),
        ("Cm", "CPitch", "Pitching-moment coefficient on the x axis"),
        ("Cn", "CYaw", "Yawing-moment coefficient on the x axis"),
        ("L/D", "L/D", "Lift-to-drag ratio on the x axis"),
    ]

    DEFAULT_Y = "Cl"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._custom: list = []
        self._element_names: list = []
        self._setup_ui()

    def _setup_ui(self):
        from PyQt6.QtWidgets import QGridLayout
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)

        layout.addWidget(QLabel("Y Axis:"), 0, 0)
        self.cmb_y_axis = QComboBox()
        self.cmb_y_axis.setMinimumWidth(200)
        self.cmb_y_axis.setMaxVisibleItems(30)
        self.cmb_y_axis.setToolTip(
            "Quantity to plot: coefficients, wind- or body-axis forces and "
            "moments, balance elements, tunnel conditions, stability "
            "derivatives or calculator outputs")
        self._fill_y()
        self.cmb_y_axis.currentIndexChanged.connect(self._on_changed)
        layout.addWidget(self.cmb_y_axis, 0, 1)

        layout.addWidget(QLabel("X Axis:"), 1, 0)
        self.cmb_x_axis = QComboBox()
        self.cmb_x_axis.setMinimumWidth(200)
        for display_name, value, tooltip in self.X_AXIS_VARS:
            self.cmb_x_axis.addItem(display_name, value)
            idx = self.cmb_x_axis.count() - 1
            self.cmb_x_axis.setItemData(
                idx, tooltip, Qt.ItemDataRole.ToolTipRole)
        self._n_builtin_x = self.cmb_x_axis.count()
        self.cmb_x_axis.currentIndexChanged.connect(self._on_changed)
        layout.addWidget(self.cmb_x_axis, 1, 1)
        layout.setColumnStretch(1, 1)

    # ── Y list ──────────────────────────────────────────────────────────
    def _fill_y(self, keep: str = ""):
        from PyQt6.QtGui import QFont
        from ..plot_variables import grouped
        current = keep or (self.get_y_var() if self.cmb_y_axis.count()
                           else self.DEFAULT_Y)
        self.cmb_y_axis.blockSignals(True)
        self.cmb_y_axis.clear()
        model = self.cmb_y_axis.model()
        for group, variables in grouped(self._custom, self._element_names):
            self.cmb_y_axis.addItem(group.upper(), None)
            header = model.item(self.cmb_y_axis.count() - 1)
            header.setEnabled(False)
            font = QFont(header.font())
            font.setBold(True)
            font.setPointSizeF(max(font.pointSizeF() - 1, 7))
            header.setFont(font)
            for var in variables:
                self.cmb_y_axis.addItem("    " + var.label, var.key)
                if var.tip:
                    self.cmb_y_axis.setItemData(
                        self.cmb_y_axis.count() - 1, var.tip,
                        Qt.ItemDataRole.ToolTipRole)
        self.cmb_y_axis.blockSignals(False)
        self.set_y_var(current or self.DEFAULT_Y, emit=False)

    def set_element_names(self, names):
        """Relabel the six balance elements with the recording balance's
        channel names (N1.. / AftPitch.. / Fx..)."""
        names = list(names or [])
        if names != self._element_names:
            self._element_names = names
            self._fill_y(self.get_y_var())

    # ── signals / accessors ─────────────────────────────────────────────
    def _on_changed(self, index):
        self.plot_type_changed.emit(self.get_y_var())

    def get_y_var(self) -> str:
        """Key of the quantity on the Y axis."""
        return self.cmb_y_axis.currentData() or self.DEFAULT_Y

    def set_y_var(self, key: str, emit: bool = True) -> bool:
        """Select a Y quantity by key; False (selection unchanged) when the
        list does not offer it."""
        for i in range(self.cmb_y_axis.count()):
            if self.cmb_y_axis.itemData(i) == key:
                if not emit:
                    self.cmb_y_axis.blockSignals(True)
                self.cmb_y_axis.setCurrentIndex(i)
                self.cmb_y_axis.blockSignals(False)
                return True
        return False

    def get_plot_type(self) -> str:
        """Back-compat alias: the Y variable key."""
        return self.get_y_var()

    def set_plot_type(self, plot_type: str):
        """Back-compat: an old PlotType name (CL_VS_CD, ...) selects the
        equivalent X/Y pair; a variable key selects that Y quantity."""
        from ..plot_variables import LEGACY_PLOT_TYPES
        if plot_type in LEGACY_PLOT_TYPES:
            x_var, y_var = LEGACY_PLOT_TYPES[plot_type]
            self.set_x_var("" if x_var == "Alpha" else x_var)
            self.set_y_var(y_var)
        else:
            self.set_y_var(plot_type)

    def get_custom_y_var(self) -> str:
        """Back-compat: the Y quantity when it is a calculator output."""
        key = self.get_y_var()
        return key if key in self._custom else ""

    def get_x_var(self) -> str:
        """Return the selected x variable, or "" for the alpha default."""
        return self.cmb_x_axis.currentData() or ""

    def set_x_var(self, var: str):
        """Select an x variable by name; an unknown name selects the default."""
        for i in range(self.cmb_x_axis.count()):
            if self.cmb_x_axis.itemData(i) == (var or ""):
                self.cmb_x_axis.setCurrentIndex(i)
                return
        self.cmb_x_axis.setCurrentIndex(0)

    def populate_custom_vars(self, names: list):
        """Offer calculator outputs on BOTH axes: in the Y list's
        'Calculator outputs' group, and appended after the built-in x
        variables. Each combo keeps its selection when it still exists."""
        self._custom = list(names)
        self._fill_y(self.get_y_var())

        current_x = self.cmb_x_axis.currentData()
        self.cmb_x_axis.blockSignals(True)
        while self.cmb_x_axis.count() > self._n_builtin_x:
            self.cmb_x_axis.removeItem(self.cmb_x_axis.count() - 1)
        for n in names:
            self.cmb_x_axis.addItem(n, n)
        if current_x:
            for i in range(self.cmb_x_axis.count()):
                if self.cmb_x_axis.itemData(i) == current_x:
                    self.cmb_x_axis.setCurrentIndex(i)
                    break
        self.cmb_x_axis.blockSignals(False)
