"""
Plot Canvas Widget
==================

Matplotlib canvas embedded in PyQt6 with dark theme styling and interactive features.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QToolBar,
    QToolButton, QSizePolicy, QMenu, QLabel
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QAction

import matplotlib
matplotlib.use('QtAgg')

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter
from matplotlib import rcParams

from ..utils.themes import DarkTheme, get_plot_style

# Apply plot style globally (disables LaTeX to allow Unicode characters)
rcParams.update(get_plot_style())
from ..utils.icons import Icons


class DarkNavigationToolbar(NavigationToolbar2QT):
    """Navigation toolbar with dark theme styling."""

    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        self.setStyleSheet(f"""
            QToolBar {{
                background-color: {DarkTheme.BACKGROUND_LIGHT};
                border: none;
                spacing: 4px;
                padding: 4px;
            }}
            QToolButton {{
                background-color: transparent;
                border: none;
                padding: 6px;
                border-radius: 4px;
            }}
            QToolButton:hover {{
                background-color: {DarkTheme.HOVER};
            }}
            QToolButton:pressed {{
                background-color: {DarkTheme.SELECTION};
            }}
        """)


class PlotCanvas(QWidget):
    """
    Matplotlib canvas with interactive toolbar and dark theme styling.

    Features:
    - Interactive zoom/pan
    - Data point hover annotations
    - Cursor position display
    - Grid toggle
    - Save functionality

    Signals
    -------
    plot_updated : pyqtSignal
        Emitted when the plot is updated
    point_selected : pyqtSignal(str, float, float)
        Emitted when a data point is clicked (label, x, y)
    """

    plot_updated = pyqtSignal()
    point_selected = pyqtSignal(str, float, float)
    context_menu_requested = pyqtSignal(dict)  # nearest point data for right-click

    def __init__(self, parent=None, interactive=True):
        super().__init__(parent)

        self._interactive = interactive
        # Store plot data for hover/click detection
        self._plot_data: List[Dict[str, Any]] = []
        self._annotation = None
        self._highlighted_point = None

        self._setup_ui()
        self._apply_style()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.figure.patch.set_facecolor(DarkTheme.PLOT_BACKGROUND)

        # Create canvas
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)

        # Create axes
        self.ax: Axes = self.figure.add_subplot(111)

        # Create custom toolbar
        self.toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(self.toolbar_widget)
        toolbar_layout.setContentsMargins(8, 4, 8, 4)
        toolbar_layout.setSpacing(4)

        # Home button
        self.btn_home = QToolButton()
        self.btn_home.setIcon(Icons.home())
        self.btn_home.setToolTip("Reset view (H)")
        self.btn_home.clicked.connect(self.reset_view)
        toolbar_layout.addWidget(self.btn_home)

        # Zoom in
        self.btn_zoom_in = QToolButton()
        self.btn_zoom_in.setIcon(Icons.zoom_in())
        self.btn_zoom_in.setToolTip("Zoom to rectangle (Z)")
        self.btn_zoom_in.setCheckable(True)
        toolbar_layout.addWidget(self.btn_zoom_in)

        # Pan
        self.btn_pan = QToolButton()
        self.btn_pan.setIcon(Icons.compare())
        self.btn_pan.setToolTip("Pan (P)")
        self.btn_pan.setCheckable(True)
        toolbar_layout.addWidget(self.btn_pan)

        toolbar_layout.addSpacing(16)

        # Grid toggle
        self.btn_grid = QToolButton()
        self.btn_grid.setIcon(Icons.grid())
        self.btn_grid.setToolTip("Toggle grid (G)")
        self.btn_grid.setCheckable(True)
        self.btn_grid.setChecked(True)
        self.btn_grid.clicked.connect(self.toggle_grid)
        toolbar_layout.addWidget(self.btn_grid)

        # Autoscale button
        self.btn_autoscale = QToolButton()
        self.btn_autoscale.setText("Auto")
        self.btn_autoscale.setToolTip("Autoscale axes (A)")
        self.btn_autoscale.clicked.connect(self.autoscale)
        toolbar_layout.addWidget(self.btn_autoscale)

        toolbar_layout.addStretch()

        # Cursor position label
        self.lbl_cursor = QLabel("")
        self.lbl_cursor.setStyleSheet(f"""
            QLabel {{
                color: {DarkTheme.TEXT_SECONDARY};
                font-family: monospace;
                padding: 2px 8px;
            }}
        """)
        self.lbl_cursor.setMinimumWidth(180)
        toolbar_layout.addWidget(self.lbl_cursor)

        toolbar_layout.addSpacing(8)

        # Save button
        self.btn_save = QToolButton()
        self.btn_save.setIcon(Icons.save())
        self.btn_save.setToolTip("Save figure (Ctrl+S)")
        self.btn_save.clicked.connect(self.save_figure)
        toolbar_layout.addWidget(self.btn_save)

        # Add widgets to layout
        layout.addWidget(self.toolbar_widget)
        layout.addWidget(self.canvas)

        # Navigation toolbar (hidden, used for functionality)
        self.nav_toolbar = DarkNavigationToolbar(self.canvas, self)
        self.nav_toolbar.hide()

        # Connect mouse events for interactivity
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        if self._interactive:
            self.canvas.mpl_connect('button_press_event', self._on_mouse_click)

    def _apply_style(self):
        """Apply dark theme to the plot."""
        style = get_plot_style()

        self.ax.set_facecolor(style['axes.facecolor'])
        self.ax.tick_params(colors=style['xtick.color'])
        self.ax.spines['bottom'].set_color(style['axes.edgecolor'])
        self.ax.spines['top'].set_color(style['axes.edgecolor'])
        self.ax.spines['left'].set_color(style['axes.edgecolor'])
        self.ax.spines['right'].set_color(style['axes.edgecolor'])
        self.ax.xaxis.label.set_color(style['axes.labelcolor'])
        self.ax.yaxis.label.set_color(style['axes.labelcolor'])
        self.ax.title.set_color(style['axes.titlecolor'])

        if style['axes.grid']:
            self.ax.grid(True, color=style['grid.color'],
                         linestyle=style['grid.linestyle'],
                         linewidth=style['grid.linewidth'],
                         alpha=style['grid.alpha'])

        self._disable_scaling()

    def _connect_signals(self):
        """Connect toolbar signals."""
        self.btn_zoom_in.toggled.connect(self._on_zoom_toggled)
        self.btn_pan.toggled.connect(self._on_pan_toggled)

    def _on_zoom_toggled(self, checked: bool):
        """Handle zoom toggle."""
        if checked:
            self.btn_pan.setChecked(False)
            self.nav_toolbar.zoom()
        elif not self.btn_pan.isChecked():
            self.nav_toolbar.zoom()  # Toggle off

    def _on_pan_toggled(self, checked: bool):
        """Handle pan toggle."""
        if checked:
            self.btn_zoom_in.setChecked(False)
            self.nav_toolbar.pan()
        elif not self.btn_zoom_in.isChecked():
            self.nav_toolbar.pan()  # Toggle off

    def _on_mouse_move(self, event):
        """Handle mouse movement for cursor display and hover effects."""
        if event.inaxes == self.ax:
            self.lbl_cursor.setText(f"x={event.xdata:.3f}, y={event.ydata:.4f}")
            if self._interactive:
                self._update_hover(event.xdata, event.ydata)
        else:
            self.lbl_cursor.setText("")
            if self._interactive:
                self._clear_hover()

    def _on_mouse_click(self, event):
        """Handle mouse click for point selection and context menu."""
        if event.inaxes != self.ax:
            return
        if event.button == 1:  # Left click
            point = self._find_nearest_point(event.xdata, event.ydata)
            if point:
                self.point_selected.emit(point['label'], point['x'], point['y'])
        elif event.button == 3:  # Right click
            point = self._find_nearest_point(event.xdata, event.ydata)
            if point and point.get('case_id'):
                self.context_menu_requested.emit(point)

    def _find_nearest_point(self, x: float, y: float, threshold: float = 0.05) -> Optional[Dict]:
        """Find the nearest data point to the given coordinates."""
        if not self._plot_data:
            return None

        # Get axis limits for scaling threshold
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_scale = (xlim[1] - xlim[0]) * threshold
        y_scale = (ylim[1] - ylim[0]) * threshold

        nearest = None
        min_dist = float('inf')

        for data in self._plot_data:
            for i in range(len(data['x'])):
                dx = (data['x'][i] - x) / x_scale if x_scale != 0 else 0
                dy = (data['y'][i] - y) / y_scale if y_scale != 0 else 0
                dist = dx*dx + dy*dy

                if dist < min_dist and dist < 1.0:  # Within threshold
                    min_dist = dist
                    alpha_data = data.get('alpha')
                    beta_data = data.get('beta')
                    nearest = {
                        'label': data['label'],
                        'x': data['x'][i],
                        'y': data['y'][i],
                        'color': data.get('color', '#ffffff'),
                        'case_id': data.get('case_id'),
                        'alpha': float(alpha_data[i]) if alpha_data is not None else 0.0,
                        'beta': float(beta_data[i]) if beta_data is not None else 0.0,
                    }

        return nearest

    def _update_hover(self, x: float, y: float):
        """Update hover annotation."""
        point = self._find_nearest_point(x, y)

        if point:
            if self._annotation is None:
                self._annotation = self.ax.annotate(
                    '', xy=(0, 0), xytext=(10, 10),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.4', fc=DarkTheme.SURFACE,
                              ec=DarkTheme.BORDER, alpha=0.9),
                    color=DarkTheme.TEXT_PRIMARY,
                    fontsize=9
                )

            text = f"{point['label']}\nx={point['x']:.3f}\ny={point['y']:.4f}"
            self._annotation.set_text(text)
            self._annotation.xy = (point['x'], point['y'])
            self._annotation.set_visible(True)

            # Highlight the point
            if self._highlighted_point is not None:
                self._highlighted_point.remove()
            self._highlighted_point = self.ax.scatter(
                [point['x']], [point['y']],
                s=100, facecolors='none', edgecolors=point['color'],
                linewidths=2, zorder=10
            )

            self.canvas.draw_idle()
        else:
            self._clear_hover()

    def _clear_hover(self):
        """Clear hover annotation and highlight."""
        if self._annotation is not None:
            self._annotation.set_visible(False)
        if self._highlighted_point is not None:
            self._highlighted_point.remove()
            self._highlighted_point = None
        self.canvas.draw_idle()

    def reset_view(self):
        """Reset to default view."""
        self.nav_toolbar.home()

    def autoscale(self):
        """Autoscale axes to fit data."""
        self.ax.autoscale()
        self.ax.margins(0.05)
        self._disable_scaling()
        self.canvas.draw_idle()

    def toggle_grid(self, show: bool = None):
        """Toggle grid visibility."""
        if show is None:
            show = self.btn_grid.isChecked()
        self.ax.grid(show)
        self.canvas.draw_idle()

    def save_figure(self):
        """Save the figure to file."""
        self.nav_toolbar.save_figure()

    def clear(self):
        """Clear the plot."""
        self.ax.clear()
        self._plot_data.clear()
        self._annotation = None
        self._highlighted_point = None
        self._apply_style()
        self.canvas.draw_idle()

    def plot(self, x, y, label: str = None, color: str = None,
             marker: str = 'o', linestyle: str = '-', linewidth: float = 1.5,
             markersize: float = 6, **kwargs):
        """
        Add a plot to the canvas.

        Parameters
        ----------
        x : array-like
            X data
        y : array-like
            Y data
        label : str, optional
            Legend label
        color : str, optional
            Line color
        marker : str, optional
            Marker style
        linestyle : str, optional
            Line style
        linewidth : float, optional
            Line width
        markersize : float, optional
            Marker size
        **kwargs
            Additional matplotlib plot kwargs
        """
        # Extract custom kwargs that shouldn't go to matplotlib
        case_id = kwargs.pop('case_id', None)
        alpha_arr = kwargs.pop('alpha_arr', None)
        beta_arr = kwargs.pop('beta_arr', None)

        x = np.asarray(x)
        y = np.asarray(y)

        plot_kwargs = {
            'marker': marker,
            'linestyle': linestyle,
            'linewidth': linewidth,
            'markersize': markersize,
            'markeredgewidth': 1,
            'picker': 5,  # Enable picking for interactivity
        }
        if label:
            plot_kwargs['label'] = label
        if color:
            plot_kwargs['color'] = color

        plot_kwargs.update(kwargs)

        line, = self.ax.plot(x, y, **plot_kwargs)

        # Store data for hover/click detection
        if self._interactive:
            self._plot_data.append({
                'x': x,
                'y': y,
                'label': label or '',
                'color': line.get_color(),
                'case_id': case_id,
                'alpha': np.asarray(alpha_arr).flatten() if alpha_arr is not None else None,
                'beta': np.asarray(beta_arr).flatten() if beta_arr is not None else None,
            })

    def fill_between(self, x, y_lower, y_upper, color: str = None,
                     alpha: float = 0.2):
        """Draw a shaded region between y_lower and y_upper."""
        fill_color = color or '#1f77b4'
        self.ax.fill_between(np.asarray(x), np.asarray(y_lower),
                             np.asarray(y_upper), color=fill_color,
                             alpha=alpha, linewidth=0)

    def scatter(self, x, y, label: str = None, color: str = None,
                marker: str = 'o', size: int = 50, **kwargs):
        """
        Add scatter plot to the canvas.

        Parameters
        ----------
        x : array-like
            X data
        y : array-like
            Y data
        label : str, optional
            Legend label
        color : str, optional
            Point color
        marker : str, optional
            Marker style
        size : int, optional
            Marker size
        """
        x = np.asarray(x)
        y = np.asarray(y)

        scatter_kwargs = {
            'marker': marker,
            's': size,
            'picker': True,
        }
        if label:
            scatter_kwargs['label'] = label
        if color:
            scatter_kwargs['c'] = color

        scatter_kwargs.update(kwargs)

        self.ax.scatter(x, y, **scatter_kwargs)

        # Store data for hover/click detection
        self._plot_data.append({
            'x': x,
            'y': y,
            'label': label or '',
            'color': color or '#1f77b4'
        })

    def set_labels(self, xlabel: str = None, ylabel: str = None,
                   title: str = None):
        """Set axis labels and title."""
        style = get_plot_style()

        if xlabel:
            self.ax.set_xlabel(xlabel, color=style['axes.labelcolor'], fontsize=11)
        if ylabel:
            self.ax.set_ylabel(ylabel, color=style['axes.labelcolor'], fontsize=11)
        if title:
            self.ax.set_title(title, color=style['axes.titlecolor'], fontsize=12)

    def set_limits(self, xlim: tuple = None, ylim: tuple = None):
        """Set axis limits."""
        if xlim:
            self.ax.set_xlim(xlim)
        if ylim:
            self.ax.set_ylim(ylim)

    def add_legend(self, loc: str = 'best'):
        """Add legend to the plot."""
        # Check if there are any labeled artists
        handles, labels = self.ax.get_legend_handles_labels()
        if not labels:
            return  # No labeled artists, skip legend

        style = get_plot_style()
        legend = self.ax.legend(
            loc=loc,
            facecolor=style['legend.facecolor'],
            edgecolor=style['legend.edgecolor'],
            framealpha=style['legend.framealpha'],
            fontsize=9,
        )
        if legend:
            for text in legend.get_texts():
                text.set_color(style['text.color'])

    def remove_legend(self):
        """Remove the legend from the plot."""
        legend = self.ax.get_legend()
        if legend is not None:
            legend.remove()

    def add_horizontal_line(self, y: float, color: str = None,
                            linestyle: str = '--', label: str = None):
        """Add a horizontal reference line."""
        kwargs = {'color': color or DarkTheme.TEXT_SECONDARY,
                  'linestyle': linestyle, 'linewidth': 1, 'alpha': 0.7}
        if label:
            kwargs['label'] = label
        self.ax.axhline(y, **kwargs)

    def add_vertical_line(self, x: float, color: str = None,
                          linestyle: str = '--', label: str = None):
        """Add a vertical reference line."""
        kwargs = {'color': color or DarkTheme.TEXT_SECONDARY,
                  'linestyle': linestyle, 'linewidth': 1, 'alpha': 0.7}
        if label:
            kwargs['label'] = label
        self.ax.axvline(x, **kwargs)

    def _disable_scaling(self):
        """Disable scientific notation and offset on both axes."""
        for axis in [self.ax.xaxis, self.ax.yaxis]:
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            axis.set_major_formatter(formatter)

    def refresh(self):
        """Refresh the canvas."""
        self._disable_scaling()
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.plot_updated.emit()

    def plot_coefficient_sweep(self, cases: List[Any], x_var: str, y_var: str,
                               beta_filter: float = None):
        """
        Plot coefficient data from multiple cases.

        Parameters
        ----------
        cases : list
            List of TestCase objects
        x_var : str
            X-axis variable name
        y_var : str
            Y-axis variable name
        beta_filter : float, optional
            Filter to specific beta value
        """
        self.clear()

        for case in cases:
            if not case.visible or not case.has_data:
                continue

            if beta_filter is not None:
                data = case.get_sweep_at_beta(beta_filter)
                x_data = data.get(x_var.lower(), data.get('alpha', []))
                y_data = data.get(y_var, [])
                label = f"{case.name}"
            else:
                x_data = case.get_coefficient(x_var)
                y_data = case.get_coefficient(y_var)
                label = case.name

            if len(x_data) > 0 and len(y_data) > 0:
                # Handle 2D data
                if x_data.ndim == 2:
                    for i in range(x_data.shape[1]):
                        beta_val = np.mean(case.betas[:, i]) if case.betas.ndim == 2 else 0
                        self.plot(x_data[:, i], y_data[:, i],
                                  label=f"{case.name} Beta={beta_val:.0f} deg",
                                  color=case.color, marker=case.marker)
                else:
                    self.plot(x_data, y_data, label=label,
                              color=case.color, marker=case.marker)

        self.add_legend()
        self.refresh()
