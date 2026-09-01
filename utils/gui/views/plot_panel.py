"""
Plot Panel
==========

Panel containing the plot canvas and plot controls.
"""

import numpy as np
from typing import Optional, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QPushButton, QCheckBox, QGroupBox, QLabel, QDoubleSpinBox, QMenu
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QCursor

from ..models.data_model import DataModel, PlotType
from ..models.case import TestCase
from ..widgets.filter_widgets import MultiSelectFilter, PlotTypeSelector, FilterToolbar
from ..utils.themes import DarkTheme
from ..utils.icons import Icons

# Try to use fast plotting (pyqtgraph) first, fall back to matplotlib
try:
    from ..widgets.fast_plot_canvas import FastPlotCanvas, is_available as fast_plot_available
    USE_FAST_PLOT = fast_plot_available()
except ImportError:
    USE_FAST_PLOT = False

if not USE_FAST_PLOT:
    from ..widgets.plot_canvas import PlotCanvas


class PlotControlsWidget(QWidget):
    """Widget containing plot controls."""

    plot_type_changed = pyqtSignal(str)
    autoscale_requested = pyqtSignal()
    options_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Plot type selector
        type_group = QGroupBox("Plot Type")
        type_layout = QVBoxLayout(type_group)

        self.plot_selector = PlotTypeSelector()
        self.plot_selector.plot_type_changed.connect(self.plot_type_changed.emit)
        type_layout.addWidget(self.plot_selector)

        layout.addWidget(type_group)

        # Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)

        self.chk_grid = QCheckBox("Show Grid")
        self.chk_grid.setChecked(True)
        self.chk_grid.stateChanged.connect(lambda: self.options_changed.emit())
        options_layout.addWidget(self.chk_grid)

        self.chk_legend = QCheckBox("Show Legend")
        self.chk_legend.setChecked(True)
        self.chk_legend.stateChanged.connect(lambda: self.options_changed.emit())
        options_layout.addWidget(self.chk_legend)

        self.chk_std_dev = QCheckBox("Show Std Dev")
        self.chk_std_dev.setChecked(False)
        self.chk_std_dev.setToolTip(
            "Show \u00b11\u03c3 shading around each trace"
        )
        self.chk_std_dev.stateChanged.connect(
            lambda: self.options_changed.emit())
        options_layout.addWidget(self.chk_std_dev)

        # Line weight control
        lw_layout = QHBoxLayout()
        lw_label = QLabel("Line Width:")
        lw_layout.addWidget(lw_label)
        self.spn_linewidth = QDoubleSpinBox()
        self.spn_linewidth.setRange(0.5, 5.0)
        self.spn_linewidth.setSingleStep(0.5)
        self.spn_linewidth.setValue(1.5)
        self.spn_linewidth.setDecimals(1)
        self.spn_linewidth.valueChanged.connect(lambda: self.options_changed.emit())
        lw_layout.addWidget(self.spn_linewidth)
        options_layout.addLayout(lw_layout)

        # Marker size control
        ms_layout = QHBoxLayout()
        ms_label = QLabel("Marker Size:")
        ms_layout.addWidget(ms_label)
        self.spn_markersize = QDoubleSpinBox()
        self.spn_markersize.setRange(1.0, 15.0)
        self.spn_markersize.setSingleStep(1.0)
        self.spn_markersize.setValue(6.0)
        self.spn_markersize.setDecimals(1)
        self.spn_markersize.valueChanged.connect(lambda: self.options_changed.emit())
        ms_layout.addWidget(self.spn_markersize)
        options_layout.addLayout(ms_layout)

        # Auto scale button
        self.btn_auto_scale = QPushButton("Auto Scale")
        self.btn_auto_scale.clicked.connect(self.autoscale_requested.emit)
        options_layout.addWidget(self.btn_auto_scale)

        layout.addWidget(options_group)

        layout.addStretch()


    def get_linewidth(self) -> float:
        """Get the current line width setting."""
        return self.spn_linewidth.value()

    def get_markersize(self) -> float:
        """Get the current marker size setting."""
        return self.spn_markersize.value()

    def get_x_var(self) -> str:
        """The user-selected x variable, or "" for the plot-type default.

        Replaces the old "Plot vs beta" checkbox, which is now just the
        sideslip entry of the X Axis combo.
        """
        return self.plot_selector.get_x_var()


_BETA_LINESTYLES = ['-', '--', '-.', ':']
_BETA_MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']

# Variables that vary along the SPEED sweep rather than along an alpha or
# beta sweep.  Putting one of these on the x axis means each alpha/beta
# point becomes its own trace, walking across the speed steps; otherwise a
# speed sweep would draw as a zigzag connecting unrelated conditions.
_X_SPEED_VARS = frozenset({'Mach', 'Re', 'Q', 'U_inf', 'Velocity'})

# Axis labels for the built-in x variables.  Entries with a {} placeholder
# are dimensional and are filled from the active output unit system, which
# is also what their data is converted into (see PlotPanel._convert_x).
_X_AXIS_LABELS = {
    'Alpha': r"$\alpha$ [deg]",
    'Beta': r"$\beta$ [deg]",
    'Mach': "Mach",
    'Re': "Re",
    'Q': "q [{}]",
    'U_inf': r"$U_\infty$ [{}]",
    'Cl': r"$C_L$",
    'Cd': r"$C_D$",
    'Cs': r"$C_Y$",
    'CRoll': r"$C_l$",
    'CPitch': r"$C_m$",
    'CYaw': r"$C_n$",
    'L/D': r"$L/D$",
}


class PlotPanel(QWidget):
    """
    Main plotting panel with canvas and controls.
    """

    view_time_history_requested = pyqtSignal(str, float, float, str)  # case_id, alpha, beta, y_var
    view_fft_requested = pyqtSignal(str, float, float, str)  # case_id, alpha, beta, y_var

    def __init__(self, model: DataModel, parent=None):
        super().__init__(parent)
        self.model = model
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Filter toolbar
        self.filter_toolbar = FilterToolbar()
        self.filter_toolbar.setStyleSheet(f"""
            QWidget {{
                background-color: {DarkTheme.BACKGROUND_LIGHT};
                border-bottom: 1px solid {DarkTheme.BORDER};
            }}
        """)
        layout.addWidget(self.filter_toolbar)

        # Main content
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Plot canvas (main area) - use fast plot if available
        if USE_FAST_PLOT:
            self.plot_canvas = FastPlotCanvas()
        else:
            self.plot_canvas = PlotCanvas()
        content_layout.addWidget(self.plot_canvas, stretch=1)

        # Connect right-click context menu signal (works for both canvas types)
        self.plot_canvas.context_menu_requested.connect(self._on_context_menu)

        # Controls sidebar
        controls_frame = QFrame()
        controls_frame.setFixedWidth(250)
        controls_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {DarkTheme.BACKGROUND_LIGHT};
                border-left: 1px solid {DarkTheme.BORDER};
            }}
        """)

        self.plot_controls = PlotControlsWidget()
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(self.plot_controls)

        content_layout.addWidget(controls_frame)

        layout.addLayout(content_layout)

    def _connect_signals(self):
        """Connect signals for reactive updates."""
        # Model signals
        self.model.cases_changed.connect(self._update_plot)
        self.model.case_visibility_changed.connect(lambda: self._update_plot())
        self.model.filters_changed.connect(self._update_plot)
        self.model.plot_config_changed.connect(self._update_plot)

        # Filter toolbar
        self.filter_toolbar.filters_changed.connect(self._on_filter_changed)

        # Plot controls
        self.plot_controls.plot_type_changed.connect(self._on_plot_type_changed)
        self.plot_controls.autoscale_requested.connect(self._on_autoscale)
        self.plot_controls.options_changed.connect(self._on_options_changed)

    def _on_filter_changed(self):
        """Handle filter change."""
        sel_alphas = self.filter_toolbar.get_selected_alphas()
        sel_betas = self.filter_toolbar.get_selected_betas()

        self.model.set_filters(
            selected_alphas=sel_alphas if sel_alphas is not None else [],
            show_all_alphas=sel_alphas is None,
            selected_betas=sel_betas if sel_betas is not None else [],
            show_all_betas=sel_betas is None,
        )

    def _on_plot_type_changed(self, plot_type_str: str):
        """Handle plot type change."""
        try:
            plot_type = PlotType[plot_type_str]
            self.model.set_plot_type(plot_type)
        except KeyError:
            pass

    def _on_options_changed(self):
        """Handle options change (grid, legend, linewidth)."""
        self.model.plot_config.show_grid = self.plot_controls.chk_grid.isChecked()
        self.model.plot_config.show_legend = self.plot_controls.chk_legend.isChecked()
        self.model.plot_config.show_std_dev = self.plot_controls.chk_std_dev.isChecked()
        self._update_plot()

    def _on_autoscale(self):
        """Handle autoscale request."""
        self.plot_canvas.autoscale()

    def _update_plot(self):
        """Update the plot with current data and settings."""
        self.plot_canvas.clear()

        visible_cases = self.model.get_visible_cases()
        if not visible_cases:
            self.plot_canvas.refresh()
            return

        config = self.model.plot_config
        sel_alphas = self.filter_toolbar.get_selected_alphas()
        sel_betas = self.filter_toolbar.get_selected_betas()
        # Selected tunnel-speed step (Mach filter). A single config now holds
        # every speed increment as distinct per-point Mach values; this
        # selects which step(s) to visualize (None = all).
        sel_mach = self.filter_toolbar.get_selected_mach()

        # Get axis variables based on plot type, then apply the X Axis
        # override when the user picked one.
        x_var, y_var = self._get_axis_vars(config.plot_type)
        x_override = self.plot_controls.get_x_var()
        if x_override:
            x_var = x_override

        # Plot each visible case
        for case in visible_cases:
            if not case.has_data:
                continue

            self._plot_case(case, x_var, y_var, sel_alphas, sel_betas,
                            sel_mach)

        # Set labels
        self.plot_canvas.set_labels(
            xlabel=self._x_axis_label(x_var, config),
            ylabel=config.y_label
        )

        # Grid and legend
        self.plot_canvas.toggle_grid(config.show_grid)
        if config.show_legend:
            self.plot_canvas.add_legend()
        else:
            self.plot_canvas.remove_legend()

        self.plot_canvas.refresh()

    def _get_axis_vars(self, plot_type: PlotType):
        """Get X and Y variable names for plot type.

        If the Custom Y dropdown is set, its value overrides the y_var
        for the current plot type while keeping the same x_var.
        """
        var_map = {
            PlotType.CL_VS_ALPHA: ("Alpha", "Cl"),
            PlotType.CD_VS_ALPHA: ("Alpha", "Cd"),
            PlotType.CL_VS_CD: ("Cd", "Cl"),
            PlotType.CM_VS_ALPHA: ("Alpha", "CPitch"),
            PlotType.CM_VS_CL: ("Cl", "CPitch"),
            PlotType.LD_VS_ALPHA: ("Alpha", "L/D"),
            PlotType.LATERAL_VS_BETA: ("Beta", "Lateral"),
            PlotType.CY_VS_ALPHA: ("Alpha", "Cs"),
            PlotType.CROLL_VS_ALPHA: ("Alpha", "CRoll"),
            PlotType.CYAW_VS_ALPHA: ("Alpha", "CYaw"),
            # Stability derivatives (central-difference, per deg)
            PlotType.CMA_VS_ALPHA: ("Alpha", "Cma"),
            PlotType.CLA_VS_ALPHA: ("Alpha", "CLa"),
            PlotType.SM_VS_ALPHA: ("Alpha", "StaticMargin"),
            PlotType.CYB_VS_ALPHA: ("Alpha", "CYb"),
            PlotType.CNB_VS_ALPHA: ("Alpha", "Cnb"),
            PlotType.CLB_VS_ALPHA: ("Alpha", "Clb"),
        }
        x_var, y_var = var_map.get(plot_type, ("Alpha", "Cl"))

        # Custom Y override - pulls from PlotTypeSelector dropdown
        try:
            custom_y = self.plot_controls.plot_selector.get_custom_y_var()
        except Exception:
            custom_y = ""
        if custom_y:
            y_var = custom_y
        return x_var, y_var

    def _unit_labels(self):
        """Unit labels for the model's output unit system, or None."""
        try:
            from utils.windtunnel.units import UNIT_LABELS, UnitSystem
            return UNIT_LABELS[UnitSystem[self.model.output_units]]
        except Exception:                                      # noqa: BLE001
            return None

    def _x_axis_label(self, x_var: str, config) -> str:
        """Axis label for the x variable actually being plotted.

        Falls back to the plot type's own label when the x variable came
        from the plot type, and to the bare variable name for a custom
        calculator output (whose units only the user knows).
        """
        if not self.plot_controls.get_x_var():
            return config.x_label
        label = _X_AXIS_LABELS.get(x_var)
        if label is None:
            return x_var
        if "{}" not in label:
            return label
        labels = self._unit_labels()
        if x_var == 'Q':
            return label.format(labels.pressure if labels else 'psi')
        return label.format(labels.velocity if labels else 'ft/s')

    def _convert_x(self, data: np.ndarray, x_var: str) -> np.ndarray:
        """Convert dimensional x data into the output unit system.

        Coefficients, angles, Mach and Reynolds number are dimensionless
        or already in degrees and pass through untouched.  q and U_inf are
        stored internally in psi and m/s, so they are converted the same
        way the table converts them, keeping the two views consistent.
        """
        if x_var not in ('Q', 'U_inf', 'Velocity'):
            return data
        try:
            from utils.windtunnel.units import UnitConverter, UnitSystem
            conv = UnitConverter(UnitSystem[self.model.output_units])
        except Exception:                                      # noqa: BLE001
            return data
        if x_var == 'Q':
            return conv.convert_pressure(data)
        return conv.convert_velocity(data)

    def _get_linewidth(self) -> float:
        """Get the current line width setting."""
        return self.plot_controls.get_linewidth()

    def _get_markersize(self) -> float:
        """Get the current marker size setting."""
        return self.plot_controls.get_markersize()

    def _on_context_menu(self, point_data: dict):
        """Handle right-click context menu from canvas."""
        case_id = point_data.get('case_id')
        if not case_id:
            return

        case = self.model.cases.get(case_id)
        if case is None:
            return

        alpha = point_data.get('alpha', 0.0)
        beta = point_data.get('beta', 0.0)

        # Determine y-variable from current plot type for channel auto-selection
        config = self.model.plot_config
        _, y_var = self._get_axis_vars(config.plot_type)

        menu = QMenu(self)
        action_time = menu.addAction(
            f"View Time History - {case.name} (\u03b1={alpha:.1f}\u00b0, \u03b2={beta:.1f}\u00b0)"
        )
        action_fft = menu.addAction(
            f"View FFT - {case.name} (\u03b1={alpha:.1f}\u00b0, \u03b2={beta:.1f}\u00b0)"
        )

        action = menu.exec(QCursor.pos())
        if action == action_time:
            self.view_time_history_requested.emit(case_id, alpha, beta, y_var)
        elif action == action_fft:
            self.view_fft_requested.emit(case_id, alpha, beta, y_var)

    @staticmethod
    def _mach_point_mask(case: TestCase, sel_mach: Optional[float]):
        """Flat boolean mask selecting points in the speed STEP whose Mach
        matches ``sel_mach``, or None when no Mach filtering applies (no
        selection, no per-point machs, or a length that would misalign with
        alpha/beta).

        Matching is on ``case.point_machs`` (each step reported at its mean
        measured Mach), not the raw per-point Mach, so selecting a step
        keeps every point of that step: the measured Mach wanders within a
        step and would drop most of them.
        """
        if sel_mach is None:
            return None
        machs = np.asarray(
            getattr(case, 'point_machs', np.array([]))).flatten()
        n_pts = int(np.asarray(case.alphas).size)
        if machs.size == 0 or machs.size != n_pts:
            return None
        return np.isclose(machs, sel_mach, atol=5e-4)

    def _plot_case(self, case: TestCase, x_var: str, y_var: str,
                   sel_alphas: Optional[List[float]],
                   sel_betas: Optional[List[float]],
                   sel_mach: Optional[float] = None):
        """Plot data for a single case with alpha/beta/Mach multi-select
        filtering. ``sel_mach`` (from the Mach combo) selects one tunnel-speed
        step to visualize when a config holds several (None = all steps)."""
        lw = self._get_linewidth()
        ms = self._get_markersize()
        beta_xaxis = (x_var == "Beta")
        # A speed variable on x sweeps the tunnel condition, not an angle,
        # so the trace grouping inverts: one trace per (alpha, beta) across
        # the speed steps.  Only 1-D data can hold several speed steps - a
        # 2-D (alpha x beta) grid exists only when the run held one speed -
        # so a 2-D case keeps the ordinary column path.
        speed_xaxis = (x_var in _X_SPEED_VARS and case.alphas.ndim != 2)

        # Per-point Mach mask (flat, aligned with flat alpha/beta) for the
        # 1-D multi-speed path; and a whole-case skip for a single-Mach 2-D
        # grid whose Mach doesn't match the selection.
        mach_mask_flat = self._mach_point_mask(case, sel_mach)
        if sel_mach is not None and case.alphas.ndim == 2:
            case_mach = (case.mach_number if case.mach_number is not None
                         else (float(np.mean(case.machs))
                               if len(case.machs) > 0 else None))
            if case_mach is not None and not np.isclose(
                    case_mach, sel_mach, atol=5e-4):
                return

        if case.alphas.ndim == 2:
            # --- 2D grid data (rows = alpha, cols = beta) ---
            n_rows, n_cols = case.alphas.shape

            if beta_xaxis:
                # --- 2D grid, beta on x-axis: one trace per alpha row ---
                beta_avg = np.mean(case.betas, axis=0)
                if sel_betas is not None:
                    beta_cols = [j for j in range(n_cols)
                                 if any(np.isclose(beta_avg[j], b, atol=0.15)
                                        for b in sel_betas)]
                else:
                    beta_cols = list(range(n_cols))

                alpha_avg = np.mean(case.alphas, axis=1)
                if sel_alphas is not None:
                    alpha_rows = [i for i in range(n_rows)
                                  if any(np.isclose(alpha_avg[i], a, atol=0.15)
                                         for a in sel_alphas)]
                else:
                    alpha_rows = list(range(n_rows))

                beta_cols.sort(key=lambda j: beta_avg[j])

                if not beta_cols or not alpha_rows:
                    return

                single_alpha = len(alpha_rows) == 1
                for idx, i in enumerate(alpha_rows):
                    x_data = self._convert_x(
                        self._get_row_data(case, x_var, i)[beta_cols], x_var)
                    y_data = self._get_row_data(case, y_var, i)[beta_cols]

                    if len(x_data) == 0 or len(y_data) == 0:
                        continue

                    alpha_val = np.mean(case.alphas[i, :])
                    if single_alpha:
                        label = case.name
                    else:
                        label = f"{case.name} \u03b1={alpha_val:.1f}\u00b0"

                    alpha_vals = case.alphas[i, beta_cols]
                    beta_vals = case.betas[i, beta_cols]
                    ls = _BETA_LINESTYLES[idx % len(_BETA_LINESTYLES)]
                    mk = _BETA_MARKERS[idx % len(_BETA_MARKERS)]
                    self.plot_canvas.plot(x_data, y_data, label=label,
                                          color=case.color, marker=mk,
                                          linestyle=ls, linewidth=lw,
                                          markersize=ms, case_id=case.id,
                                          alpha_arr=alpha_vals,
                                          beta_arr=beta_vals)
                    if self.model.plot_config.show_std_dev:
                        std_var = self._get_std_var(y_var)
                        if std_var:
                            y_std = self._get_row_data(
                                case, std_var, i)[beta_cols]
                            if len(y_std) == len(y_data):
                                self.plot_canvas.fill_between(
                                    x_data, y_data - y_std,
                                    y_data + y_std, color=case.color)
                return

            # Determine which columns (betas) to plot
            if sel_betas is not None:
                beta_avg = np.mean(case.betas, axis=0)
                beta_cols = [j for j in range(n_cols)
                             if any(np.isclose(beta_avg[j], b, atol=0.15) for b in sel_betas)]
            else:
                beta_cols = list(range(n_cols))

            # Determine which rows (alphas) to include
            alpha_avg = np.mean(case.alphas, axis=1)
            if sel_alphas is not None:
                alpha_rows = [i for i in range(n_rows)
                              if any(np.isclose(alpha_avg[i], a, atol=0.15) for a in sel_alphas)]
            else:
                alpha_rows = list(range(n_rows))

            # Sort alpha_rows by actual alpha value (ascending)
            alpha_rows.sort(key=lambda i: alpha_avg[i])

            if not beta_cols or not alpha_rows:
                return

            single_beta = len(beta_cols) == 1
            for idx, j in enumerate(beta_cols):
                x_data = self._convert_x(
                    self._get_col_data(case, x_var, j)[alpha_rows], x_var)
                y_data = self._get_col_data(case, y_var, j)[alpha_rows]

                if len(x_data) == 0 or len(y_data) == 0:
                    continue

                beta_val = np.mean(case.betas[:, j])
                if single_beta:
                    label = case.name
                else:
                    label = f"{case.name} \u03b2={beta_val:.1f}\u00b0"

                alpha_vals = case.alphas[alpha_rows, j]
                beta_vals = case.betas[alpha_rows, j]
                ls = _BETA_LINESTYLES[idx % len(_BETA_LINESTYLES)]
                mk = _BETA_MARKERS[idx % len(_BETA_MARKERS)]
                self.plot_canvas.plot(x_data, y_data, label=label,
                                      color=case.color, marker=mk,
                                      linestyle=ls, linewidth=lw,
                                      markersize=ms, case_id=case.id,
                                      alpha_arr=alpha_vals, beta_arr=beta_vals)
                if self.model.plot_config.show_std_dev:
                    std_var = self._get_std_var(y_var)
                    if std_var:
                        y_std = self._get_col_data(
                            case, std_var, j)[alpha_rows]
                        if len(y_std) == len(y_data):
                            self.plot_canvas.fill_between(
                                x_data, y_data - y_std,
                                y_data + y_std, color=case.color)
        else:
            # --- 1D flat data ---
            flat_alpha = case.alphas.flatten()
            flat_beta = case.betas.flatten()

            if speed_xaxis:
                # --- 1D, a speed variable on x: one trace per (alpha,
                # beta), walking across the speed steps ---
                self._plot_speed_sweep(
                    case, x_var, y_var, flat_alpha, flat_beta,
                    sel_alphas, sel_betas, mach_mask_flat, lw, ms)
                return

            if beta_xaxis:
                # --- 1D, beta on x-axis: one trace per unique alpha ---
                beta_mask = np.ones(len(flat_beta), dtype=bool)
                if sel_betas is not None:
                    beta_mask = np.zeros(len(flat_beta), dtype=bool)
                    for b in sel_betas:
                        beta_mask |= np.isclose(flat_beta, b, atol=0.15)

                unique_alphas = sorted(
                    set(round(float(a), 1) for a in flat_alpha))

                if sel_alphas is not None:
                    unique_alphas = [
                        a for a in unique_alphas
                        if any(np.isclose(a, sa, atol=0.15)
                               for sa in sel_alphas)]

                if not unique_alphas:
                    return

                single_alpha = len(unique_alphas) == 1

                for idx, alpha_val in enumerate(unique_alphas):
                    alpha_match = np.isclose(flat_alpha, alpha_val, atol=0.15)
                    mask = alpha_match & beta_mask
                    if mach_mask_flat is not None:
                        mask = mask & mach_mask_flat

                    if not np.any(mask):
                        continue

                    sort_order = np.argsort(flat_beta[mask])
                    x_data = self._convert_x(
                        self._get_var_data_1d(case, x_var, mask), x_var
                    )[sort_order]
                    y_data = self._get_var_data_1d(
                        case, y_var, mask)[sort_order]

                    if len(x_data) == 0 or len(y_data) == 0:
                        continue

                    if single_alpha:
                        label = case.name
                    else:
                        label = (f"{case.name}"
                                 f" \u03b1={alpha_val:.1f}\u00b0")

                    alpha_vals = flat_alpha[mask][sort_order]
                    beta_vals = flat_beta[mask][sort_order]
                    ls = _BETA_LINESTYLES[idx % len(_BETA_LINESTYLES)]
                    mk = _BETA_MARKERS[idx % len(_BETA_MARKERS)]
                    self.plot_canvas.plot(x_data, y_data, label=label,
                                          color=case.color, marker=mk,
                                          linestyle=ls, linewidth=lw,
                                          markersize=ms, case_id=case.id,
                                          alpha_arr=alpha_vals,
                                          beta_arr=beta_vals)
                    if self.model.plot_config.show_std_dev:
                        std_var = self._get_std_var(y_var)
                        if std_var:
                            y_std = self._get_var_data_1d(
                                case, std_var, mask)[sort_order]
                            if len(y_std) == len(y_data):
                                self.plot_canvas.fill_between(
                                    x_data, y_data - y_std,
                                    y_data + y_std, color=case.color)
                return

            # Build alpha mask (applies to all beta groups)
            alpha_mask = np.ones(len(flat_alpha), dtype=bool)
            if sel_alphas is not None:
                alpha_mask = np.zeros(len(flat_alpha), dtype=bool)
                for a in sel_alphas:
                    alpha_mask |= np.isclose(flat_alpha, a, atol=0.15)

            # Determine unique betas to iterate over
            unique_betas = sorted(set(round(float(b), 1) for b in flat_beta))

            if sel_betas is not None:
                unique_betas = [b for b in unique_betas
                                if any(np.isclose(b, sb, atol=0.15) for sb in sel_betas)]

            if not unique_betas:
                return

            single_beta = len(unique_betas) == 1

            # Group by tunnel-speed step so each speed is a SEPARATE
            # alpha-sweep trace (never a zigzag connecting points across
            # speeds). A specific Mach selection -> one group; "All" over a
            # multi-speed config -> one group per distinct step;
            # single-speed -> one ungrouped pass.
            #
            # The key is case.point_machs, which reports every point of a
            # step at that step's MEAN measured Mach.  Keying on the raw
            # per-point Mach instead would split a step into one trace per
            # point, because the tunnel does not hold an exact Mach across
            # an alpha sweep.
            machs_flat = np.asarray(
                getattr(case, 'point_machs', np.array([]))).flatten()
            have_mach = (machs_flat.size == len(flat_alpha)
                         and machs_flat.size > 0)
            if mach_mask_flat is not None:
                mach_groups = [(sel_mach, mach_mask_flat)]
            elif have_mach:
                distinct = sorted(set(round(float(m), 3) for m in machs_flat))
                mach_groups = ([(m, np.isclose(machs_flat, m, atol=5e-4))
                                for m in distinct] if len(distinct) > 1
                               else [(None, None)])
            else:
                mach_groups = [(None, None)]
            multi_mach = len(mach_groups) > 1

            trace_idx = 0
            for mach_val, mmask in mach_groups:
                for beta_val in unique_betas:
                    beta_mask = np.isclose(flat_beta, beta_val, atol=0.15)
                    mask = alpha_mask & beta_mask
                    if mmask is not None:
                        mask = mask & mmask

                    if not np.any(mask):
                        continue

                    # Sort by alpha within each (speed, beta) group
                    sort_order = np.argsort(flat_alpha[mask])

                    x_data = self._convert_x(
                        self._get_var_data_1d(case, x_var, mask),
                        x_var)[sort_order]
                    y_data = self._get_var_data_1d(case, y_var, mask)[sort_order]

                    if len(x_data) == 0 or len(y_data) == 0:
                        continue

                    parts = [case.name]
                    if multi_mach and mach_val is not None:
                        parts.append(f"M={mach_val:.3f}")
                    if not single_beta:
                        parts.append(f"\u03b2={beta_val:.1f}\u00b0")
                    label = " ".join(parts)

                    alpha_vals = flat_alpha[mask][sort_order]
                    beta_vals = flat_beta[mask][sort_order]
                    ls = _BETA_LINESTYLES[trace_idx % len(_BETA_LINESTYLES)]
                    mk = _BETA_MARKERS[trace_idx % len(_BETA_MARKERS)]
                    self.plot_canvas.plot(x_data, y_data, label=label,
                                          color=case.color, marker=mk,
                                          linestyle=ls, linewidth=lw,
                                          markersize=ms, case_id=case.id,
                                          alpha_arr=alpha_vals, beta_arr=beta_vals)
                    if self.model.plot_config.show_std_dev:
                        std_var = self._get_std_var(y_var)
                        if std_var:
                            y_std = self._get_var_data_1d(
                                case, std_var, mask)[sort_order]
                            if len(y_std) == len(y_data):
                                self.plot_canvas.fill_between(
                                    x_data, y_data - y_std,
                                    y_data + y_std, color=case.color)
                    trace_idx += 1

    def _plot_speed_sweep(self, case: TestCase, x_var: str, y_var: str,
                          flat_alpha: np.ndarray, flat_beta: np.ndarray,
                          sel_alphas: Optional[List[float]],
                          sel_betas: Optional[List[float]],
                          mach_mask_flat: Optional[np.ndarray],
                          lw: float, ms: float):
        """Draw one trace per (alpha, beta) across the tunnel-speed steps.

        This is the inverse of the ordinary grouping.  With alpha on x, a
        speed sweep draws one alpha sweep per speed step; with a speed
        variable on x it draws one speed sweep per angle, which is what
        makes a Mach sweep read as a curve instead of a column of points.

        Points are ordered by the x variable itself rather than by the
        speed setpoint, so the line never doubles back on a run where the
        tunnel did not settle monotonically.
        """
        pairs = []
        for a, b in zip(flat_alpha, flat_beta):
            key = (round(float(a), 1), round(float(b), 1))
            if key not in pairs:
                pairs.append(key)
        if sel_alphas is not None:
            pairs = [p for p in pairs
                     if any(np.isclose(p[0], sa, atol=0.15)
                            for sa in sel_alphas)]
        if sel_betas is not None:
            pairs = [p for p in pairs
                     if any(np.isclose(p[1], sb, atol=0.15)
                            for sb in sel_betas)]
        if not pairs:
            return
        pairs.sort()

        single_alpha = len({p[0] for p in pairs}) == 1
        single_beta = len({p[1] for p in pairs}) == 1

        for idx, (alpha_val, beta_val) in enumerate(pairs):
            mask = (np.isclose(flat_alpha, alpha_val, atol=0.15)
                    & np.isclose(flat_beta, beta_val, atol=0.15))
            if mach_mask_flat is not None:
                mask = mask & mach_mask_flat
            if not np.any(mask):
                continue

            x_data = self._convert_x(
                self._get_var_data_1d(case, x_var, mask), x_var)
            y_data = self._get_var_data_1d(case, y_var, mask)
            if len(x_data) == 0 or len(y_data) == 0:
                continue

            sort_order = np.argsort(x_data)
            x_data = x_data[sort_order]
            y_data = y_data[sort_order]

            parts = [case.name]
            if not single_alpha:
                parts.append(f"\u03b1={alpha_val:.1f}\u00b0")
            if not single_beta:
                parts.append(f"\u03b2={beta_val:.1f}\u00b0")
            label = " ".join(parts)

            ls = _BETA_LINESTYLES[idx % len(_BETA_LINESTYLES)]
            mk = _BETA_MARKERS[idx % len(_BETA_MARKERS)]
            self.plot_canvas.plot(
                x_data, y_data, label=label, color=case.color, marker=mk,
                linestyle=ls, linewidth=lw, markersize=ms, case_id=case.id,
                alpha_arr=flat_alpha[mask][sort_order],
                beta_arr=flat_beta[mask][sort_order])
            if self.model.plot_config.show_std_dev:
                std_var = self._get_std_var(y_var)
                if std_var:
                    y_std = self._get_var_data_1d(
                        case, std_var, mask)[sort_order]
                    if len(y_std) == len(y_data):
                        self.plot_canvas.fill_between(
                            x_data, y_data - y_std, y_data + y_std,
                            color=case.color)

    @staticmethod
    def _conform(case: TestCase, data: np.ndarray) -> np.ndarray:
        """Reshape a flat per-point array onto the case's alpha grid.

        The per-point tunnel conditions (Mach, Re, q, U_inf) are stored
        flat, one entry per reduced point, while alphas/betas may be a 2-D
        (alpha x beta) grid.  Indexing a flat array by row or column would
        silently take the wrong points, so conform it first.  Anything that
        does not match the grid size is returned untouched.
        """
        data = np.asarray(data)
        alphas = np.asarray(case.alphas)
        if (alphas.ndim == 2 and data.ndim == 1
                and data.size == alphas.size):
            return data.reshape(alphas.shape)
        return data

    def _resolve_derivative(self, case: TestCase, var: str) -> Optional[np.ndarray]:
        """Return the full derivative array for `var`, or None if not a derivative."""
        try:
            from utils.windtunnel.derivatives import is_derivative, get_derivative
        except Exception:
            return None
        if not is_derivative(var):
            return None
        return get_derivative(case, var)

    def _resolve_custom(self, case: TestCase, var: str) -> Optional[np.ndarray]:
        """
        Return the user-defined calculator output `var` for the case,
        or None if it's not a custom variable.

        Supports `<name>_std` lookup: if a direct hit fails and the
        name ends in `_std`, the parallel custom_vars_std dict is
        consulted.  This is how sigma shading flows through the
        same _get_*_data accessor path as primary values.
        """
        custom = getattr(case, 'custom_vars', None)
        if isinstance(custom, dict):
            arr = custom.get(var)
            if arr is not None:
                return np.asarray(arr)

        # Std-dev fallback for custom variables (e.g. 'Cp1_std')
        if var.endswith('_std'):
            base = var[:-len('_std')]
            stds = getattr(case, 'custom_vars_std', None)
            if isinstance(stds, dict):
                arr = stds.get(base)
                if arr is not None:
                    return np.asarray(arr)
        return None

    def _get_var_data(self, case: TestCase, sweep_data: Optional[dict],
                      var: str) -> np.ndarray:
        """Get data for a variable."""
        if var == "L/D":
            if sweep_data:
                return sweep_data['Cl'] / np.maximum(sweep_data['Cd'], 1e-10)
            else:
                return case.Cl / np.maximum(case.Cd, 1e-10)
        if var == "Lateral":
            var = "Cs"

        # Custom user-defined calculator outputs are checked first so
        # the user can shadow / override built-in coefficient names.
        custom = self._resolve_custom(case, var)
        if custom is not None:
            return custom

        # Derivatives are computed on demand from the case data
        deriv = self._resolve_derivative(case, var)
        if deriv is not None:
            return deriv

        if sweep_data and var.lower() in sweep_data:
            return sweep_data[var.lower()]
        if sweep_data and var in sweep_data:
            return sweep_data[var]

        return case.get_coefficient(var)

    def _get_col_data(self, case: TestCase, var: str, col: int) -> np.ndarray:
        """Get column data for 2D arrays."""
        if var == "L/D":
            return case.Cl[:, col] / np.maximum(case.Cd[:, col], 1e-10)
        if var == "Lateral":
            var = "Cs"

        custom = self._resolve_custom(case, var)
        if custom is not None:
            if custom.ndim == 2:
                return custom[:, col]
            return custom

        deriv = self._resolve_derivative(case, var)
        if deriv is not None:
            if deriv.ndim == 2:
                return deriv[:, col]
            return deriv

        data = self._conform(case, case.get_coefficient(var))
        if data.ndim == 2:
            return data[:, col]
        return data

    def _get_row_data(self, case: TestCase, var: str, row: int) -> np.ndarray:
        """Get row data for 2D arrays (beta sweep at fixed alpha)."""
        if var == "L/D":
            return case.Cl[row, :] / np.maximum(case.Cd[row, :], 1e-10)
        if var == "Lateral":
            var = "Cs"

        custom = self._resolve_custom(case, var)
        if custom is not None:
            if custom.ndim == 2:
                return custom[row, :]
            return custom

        deriv = self._resolve_derivative(case, var)
        if deriv is not None:
            if deriv.ndim == 2:
                return deriv[row, :]
            return deriv

        data = self._conform(case, case.get_coefficient(var))
        if data.ndim == 2:
            return data[row, :]
        return data

    def _get_var_data_1d(self, case: TestCase, var: str,
                         mask: np.ndarray) -> np.ndarray:
        """Get 1D variable data filtered by a boolean mask."""
        if var == "L/D":
            cl = case.Cl.flatten()[mask]
            cd = case.Cd.flatten()[mask]
            return cl / np.maximum(cd, 1e-10)
        if var == "Lateral":
            var = "Cs"

        custom = self._resolve_custom(case, var)
        if custom is not None:
            return custom.flatten()[mask]

        deriv = self._resolve_derivative(case, var)
        if deriv is not None:
            return deriv.flatten()[mask]

        data = case.get_coefficient(var)
        return data.flatten()[mask]

    def _get_std_var(self, var: str) -> Optional[str]:
        """Map a coefficient variable name to its std dev counterpart.

        Built-in coefficients map to their `<name>_std` field.  Custom
        calculator variables ALSO map to `<name>_std`, which is then
        resolved by _resolve_custom against case.custom_vars_std.
        Returns None for vars that don't have a std-dev companion.
        """
        std_map = {
            'Cl': 'Cl_std', 'CL': 'Cl_std',
            'Cd': 'Cd_std', 'CD': 'Cd_std',
            'Cs': 'Cs_std', 'CY': 'Cs_std',
            'CRoll': 'CRoll_std',
            'CPitch': 'CPitch_std', 'Cm': 'CPitch_std',
            'CYaw': 'CYaw_std', 'Cn': 'CYaw_std',
        }
        if var in std_map:
            return std_map[var]
        # Custom variable - use `<name>_std` which _resolve_custom
        # routes to case.custom_vars_std.  Only valid if the var is
        # actually in custom_vars on at least one currently-visible
        # case; we do a soft check via the model.
        for c in self.model.cases:
            cv = getattr(c, 'custom_vars', None)
            if isinstance(cv, dict) and var in cv:
                return var + '_std'
        return None

    def update_filters(self):
        """Update filter options from model."""
        alphas = self.model.get_available_alphas()
        self.filter_toolbar.set_alpha_values(alphas)

        betas = self.model.get_available_betas()
        self.filter_toolbar.set_beta_values(betas)

        machs = self.model.cases.all_mach_numbers
        self.filter_toolbar.set_mach_values(machs)
