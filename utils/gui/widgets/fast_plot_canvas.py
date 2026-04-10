"""
Fast Plot Canvas Widget
=======================

High-performance plotting canvas using pyqtgraph for interactive data visualization.
Provides significant speed improvements over matplotlib for interactive plots.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import re

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QToolButton, QLabel, QSizePolicy,
    QFileDialog, QMenu
)
from PyQt6.QtCore import pyqtSignal, Qt, QPointF
from PyQt6.QtGui import QColor, QFont

try:
    import pyqtgraph as pg
    from pyqtgraph import exporters
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False

from ..utils.themes import DarkTheme
from ..utils.icons import Icons


# LaTeX to Unicode conversion map
LATEX_TO_UNICODE = {
    r'\alpha': '\u03b1',      # α
    r'\beta': '\u03b2',       # β
    r'\gamma': '\u03b3',      # γ
    r'\delta': '\u03b4',      # δ
    r'\epsilon': '\u03b5',    # ε
    r'\theta': '\u03b8',      # θ
    r'\lambda': '\u03bb',     # λ
    r'\mu': '\u03bc',         # μ
    r'\pi': '\u03c0',         # π
    r'\rho': '\u03c1',        # ρ
    r'\sigma': '\u03c3',      # σ
    r'\phi': '\u03c6',        # φ
    r'\omega': '\u03c9',      # ω
    r'\Delta': '\u0394',      # Δ
    r'\Sigma': '\u03a3',      # Σ
    r'\infty': '\u221e',      # ∞
    r'\degree': '\u00b0',     # °
    r'\deg': '\u00b0',        # °
    r'\pm': '\u00b1',         # ±
    r'\leq': '\u2264',        # ≤
    r'\geq': '\u2265',        # ≥
    r'\neq': '\u2260',        # ≠
    r'\approx': '\u2248',     # ≈
    r'\cdot': '\u00b7',       # ·
    r'\times': '\u00d7',      # ×
}

# Unicode subscript characters (only ones that look good)
SUBSCRIPT_MAP = {
    '0': '\u2080', '1': '\u2081', '2': '\u2082', '3': '\u2083', '4': '\u2084',
    '5': '\u2085', '6': '\u2086', '7': '\u2087', '8': '\u2088', '9': '\u2089',
    'a': '\u2090', 'e': '\u2091', 'i': '\u1d62', 'o': '\u2092',
    'r': '\u1d63', 'u': '\u1d64', 'v': '\u1d65', 'x': '\u2093',
    'm': '\u2098', 'n': '\u2099', 'l': '\u2097',
}


def to_subscript(text: str) -> str:
    """Convert text to Unicode subscript where possible."""
    result = []
    for char in text:
        result.append(SUBSCRIPT_MAP.get(char, char))
    return ''.join(result)


def latex_to_unicode(text: str) -> str:
    """Convert LaTeX-style labels to Unicode with proper formatting."""
    if not text:
        return text

    # Remove $ delimiters
    text = text.replace('$', '')

    # Replace LaTeX commands with Unicode
    for latex, unicode_char in LATEX_TO_UNICODE.items():
        text = text.replace(latex, unicode_char)

    # Handle common aerodynamic coefficient patterns - use conventional notation
    # These are processed BEFORE general subscript handling
    text = text.replace('C_L', 'C\u029f')   # CL with small cap L
    text = text.replace('C_D', 'C\u1d05')   # CD with small cap D
    text = text.replace('C_Y', 'C\u028f')   # CY with small cap Y
    text = text.replace('C_m', 'Cₘ')        # Cm with subscript m
    text = text.replace('C_l', 'Cₗ')        # Cl with subscript l (roll)
    text = text.replace('C_n', 'Cₙ')        # Cn with subscript n

    # Handle remaining subscripts: _{xyz} -> subscript xyz
    def replace_subscript(match):
        content = match.group(1)
        return to_subscript(content)

    text = re.sub(r'_\{([^}]+)\}', replace_subscript, text)
    text = re.sub(r'_([a-zA-Z0-9])', lambda m: to_subscript(m.group(1)), text)

    return text


# Configure pyqtgraph for dark theme
if PYQTGRAPH_AVAILABLE:
    pg.setConfigOptions(
        antialias=True,
        background=DarkTheme.PLOT_BACKGROUND,
        foreground=DarkTheme.TEXT_PRIMARY
    )


class CrosshairCursor:
    """Crosshair cursor with data readout."""

    def __init__(self, plot_item):
        self.plot_item = plot_item
        self.vb = plot_item.vb

        # Create crosshair lines
        self.vline = pg.InfiniteLine(angle=90, movable=False,
                                      pen=pg.mkPen(DarkTheme.ACCENT, width=1, style=Qt.PenStyle.DashLine))
        self.hline = pg.InfiniteLine(angle=0, movable=False,
                                      pen=pg.mkPen(DarkTheme.ACCENT, width=1, style=Qt.PenStyle.DashLine))

        # Text item for data display
        self.text_item = pg.TextItem(color=DarkTheme.TEXT_PRIMARY, anchor=(0, 1))
        self.text_item.setFont(QFont("Consolas", 9))

        # Add to plot
        plot_item.addItem(self.vline, ignoreBounds=True)
        plot_item.addItem(self.hline, ignoreBounds=True)
        plot_item.addItem(self.text_item, ignoreBounds=True)

        self.hide()

    def show(self):
        self.vline.show()
        self.hline.show()
        self.text_item.show()

    def hide(self):
        self.vline.hide()
        self.hline.hide()
        self.text_item.hide()

    def update(self, x: float, y: float, nearest_point: dict = None):
        """Update crosshair position."""
        self.vline.setPos(x)
        self.hline.setPos(y)

        # Build text
        if nearest_point:
            text = f"{nearest_point['label']}\nx = {nearest_point['x']:.4f}\ny = {nearest_point['y']:.4f}"
        else:
            text = f"x = {x:.4f}\ny = {y:.4f}"

        self.text_item.setText(text)

        # Position text in upper left of view
        view_range = self.vb.viewRange()
        x_pos = view_range[0][0] + (view_range[0][1] - view_range[0][0]) * 0.02
        y_pos = view_range[1][1] - (view_range[1][1] - view_range[1][0]) * 0.02
        self.text_item.setPos(x_pos, y_pos)


class FastPlotCanvas(QWidget):
    """
    High-performance plot canvas using pyqtgraph.

    Provides the same API as PlotCanvas but with much better performance
    for interactive operations like zooming, panning, and filtering.

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

    # Marker symbol mapping from matplotlib to pyqtgraph
    MARKER_MAP = {
        'o': 'o',   # Circle
        's': 's',   # Square
        '^': 't',   # Triangle up
        'v': 't1',  # Triangle down
        'D': 'd',   # Diamond
        '<': 't2',  # Triangle left
        '>': 't3',  # Triangle right
        'p': 'p',   # Pentagon
        'h': 'h',   # Hexagon
        '*': 'star',
        '+': '+',
        'x': 'x',
    }

    def __init__(self, parent=None):
        super().__init__(parent)

        if not PYQTGRAPH_AVAILABLE:
            raise ImportError("pyqtgraph is required for FastPlotCanvas")

        self._plot_items: List[pg.PlotDataItem] = []
        self._plot_data: List[Dict[str, Any]] = []
        self._legend = None
        self._show_grid = True
        self._show_legend = True
        self._crosshair = None
        self._highlight_point = None

        self._setup_ui()
        self._apply_style()

    def _setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Custom toolbar
        self.toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(self.toolbar_widget)
        toolbar_layout.setContentsMargins(8, 4, 8, 4)
        toolbar_layout.setSpacing(4)

        # Reset view button
        self.btn_home = QToolButton()
        self.btn_home.setIcon(Icons.home())
        self.btn_home.setToolTip("Reset view (H)")
        self.btn_home.clicked.connect(self.reset_view)
        toolbar_layout.addWidget(self.btn_home)

        # Zoom button (toggle rectangle zoom mode)
        self.btn_zoom = QToolButton()
        self.btn_zoom.setIcon(Icons.zoom_in())
        self.btn_zoom.setToolTip("Zoom to rectangle (Z) - Right-click to reset")
        self.btn_zoom.setCheckable(True)
        self.btn_zoom.clicked.connect(self._toggle_zoom_mode)
        toolbar_layout.addWidget(self.btn_zoom)

        # Pan button
        self.btn_pan = QToolButton()
        self.btn_pan.setIcon(Icons.compare())
        self.btn_pan.setToolTip("Pan (drag with mouse)")
        self.btn_pan.setCheckable(True)
        self.btn_pan.clicked.connect(self._toggle_pan_mode)
        toolbar_layout.addWidget(self.btn_pan)

        toolbar_layout.addSpacing(8)

        # Grid toggle
        self.btn_grid = QToolButton()
        self.btn_grid.setIcon(Icons.grid())
        self.btn_grid.setToolTip("Toggle grid (G)")
        self.btn_grid.setCheckable(True)
        self.btn_grid.setChecked(True)
        self.btn_grid.clicked.connect(self.toggle_grid)
        toolbar_layout.addWidget(self.btn_grid)

        # Auto scale button
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
                font-family: Consolas, monospace;
                padding: 2px 8px;
            }}
        """)
        self.lbl_cursor.setMinimumWidth(200)
        toolbar_layout.addWidget(self.lbl_cursor)

        toolbar_layout.addSpacing(8)

        # Save button
        self.btn_save = QToolButton()
        self.btn_save.setIcon(Icons.save())
        self.btn_save.setToolTip("Save figure (Ctrl+S)")
        self.btn_save.clicked.connect(self.save_figure)
        toolbar_layout.addWidget(self.btn_save)

        layout.addWidget(self.toolbar_widget)

        # Create pyqtgraph plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Expanding)

        # Enable mouse interaction
        self.plot_widget.setMouseEnabled(x=True, y=True)

        layout.addWidget(self.plot_widget)

        # Get the plot item
        self.plot_item = self.plot_widget.getPlotItem()

        # Configure view box for better interaction
        self.plot_item.vb.setMouseMode(pg.ViewBox.RectMode)

        # Disable pyqtgraph's built-in right-click menu (we have our own toolbar)
        self.plot_item.vb.setMenuEnabled(False)

        # Create crosshair cursor
        self._crosshair = CrosshairCursor(self.plot_item)

        # Enable mouse tracking for cursor display
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_move)

        # Enable mouse click for point selection
        self.plot_widget.scene().sigMouseClicked.connect(self._on_mouse_click)

        # Right-click context menu via Qt (reliable across all pyqtgraph versions)
        self.plot_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.plot_widget.customContextMenuRequested.connect(self._on_right_click)

        # Connect view changed to update crosshair text position
        self.plot_item.vb.sigRangeChanged.connect(self._on_range_changed)

    def _disable_axis_scaling(self):
        """Force all axes to display raw data values with no scaling."""
        for axis_name in ['bottom', 'left', 'top', 'right']:
            ax = self.plot_item.getAxis(axis_name)
            ax.enableAutoSIPrefix(False)
            ax.autoSIPrefixScale = 1.0
            ax.labelUnitPrefix = ''

    def _apply_style(self):
        """Apply dark theme to the plot."""
        # Background
        self.plot_widget.setBackground(DarkTheme.PLOT_BACKGROUND)

        # Axis styling
        axis_pen = pg.mkPen(color=DarkTheme.TEXT_SECONDARY, width=1)

        for axis in ['bottom', 'left', 'top', 'right']:
            ax = self.plot_item.getAxis(axis)
            ax.setPen(axis_pen)
            ax.setTextPen(pg.mkPen(DarkTheme.TEXT_PRIMARY))
            ax.setStyle(tickFont=QFont("Segoe UI", 10))

        self._disable_axis_scaling()

        # Grid
        self.toggle_grid(self._show_grid)

        # Toolbar styling
        self.toolbar_widget.setStyleSheet(f"""
            QWidget {{
                background-color: {DarkTheme.BACKGROUND_LIGHT};
                border-bottom: 1px solid {DarkTheme.BORDER};
            }}
            QToolButton {{
                background-color: transparent;
                border: none;
                padding: 6px;
                border-radius: 4px;
                color: {DarkTheme.TEXT_PRIMARY};
            }}
            QToolButton:hover {{
                background-color: {DarkTheme.HOVER};
            }}
            QToolButton:checked {{
                background-color: {DarkTheme.SELECTION};
            }}
        """)

    def _toggle_zoom_mode(self, checked: bool):
        """Toggle rectangle zoom mode."""
        if checked:
            self.btn_pan.setChecked(False)
            self.plot_item.vb.setMouseMode(pg.ViewBox.RectMode)
        else:
            self.plot_item.vb.setMouseMode(pg.ViewBox.PanMode)

    def _toggle_pan_mode(self, checked: bool):
        """Toggle pan mode."""
        if checked:
            self.btn_zoom.setChecked(False)
            self.plot_item.vb.setMouseMode(pg.ViewBox.PanMode)
        else:
            self.plot_item.vb.setMouseMode(pg.ViewBox.RectMode)

    def _on_range_changed(self):
        """Handle view range change — re-enforce disabled axis scaling."""
        self._disable_axis_scaling()

    def _on_mouse_move(self, pos):
        """Handle mouse movement for cursor display and crosshair."""
        if self.plot_item.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_item.vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()

            # Find nearest point
            nearest = self._find_nearest_point(x, y)

            # Update cursor label
            if nearest:
                self.lbl_cursor.setText(
                    f"{nearest['label']}: ({nearest['x']:.3f}, {nearest['y']:.4f})"
                )
                self._crosshair.update(nearest['x'], nearest['y'], nearest)
                self._highlight_nearest(nearest)
            else:
                self.lbl_cursor.setText(f"x={x:.3f}, y={y:.4f}")
                self._crosshair.update(x, y)
                self._clear_highlight()

            self._crosshair.show()
        else:
            self.lbl_cursor.setText("")
            self._crosshair.hide()
            self._clear_highlight()

    def _highlight_nearest(self, point: dict):
        """Highlight the nearest data point."""
        if self._highlight_point:
            self.plot_item.removeItem(self._highlight_point)

        self._highlight_point = pg.ScatterPlotItem(
            [point['x']], [point['y']],
            size=15, pen=pg.mkPen(point['color'], width=2),
            brush=pg.mkBrush(None), symbol='o'
        )
        self.plot_item.addItem(self._highlight_point)

    def _clear_highlight(self):
        """Clear point highlight."""
        if self._highlight_point:
            self.plot_item.removeItem(self._highlight_point)
            self._highlight_point = None

    def _on_mouse_click(self, event):
        """Handle mouse click for point selection."""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.scenePos()
            if self.plot_item.sceneBoundingRect().contains(pos):
                mouse_point = self.plot_item.vb.mapSceneToView(pos)
                point = self._find_nearest_point(mouse_point.x(), mouse_point.y())
                if point:
                    self.point_selected.emit(point['label'], point['x'], point['y'])

    def _on_right_click(self, pos):
        """Handle right-click via Qt context menu for reliable operation."""
        # Map widget position to scene coordinates, then to data coordinates
        scene_pos = self.plot_widget.mapToScene(pos)
        if self.plot_item.sceneBoundingRect().contains(scene_pos):
            mouse_point = self.plot_item.vb.mapSceneToView(scene_pos)
            point = self._find_nearest_point(mouse_point.x(), mouse_point.y())
            if point and point.get('case_id'):
                self.context_menu_requested.emit(point)
            else:
                # Background menu with standard plot options
                menu = QMenu(self)
                action_view_all = menu.addAction("View All")
                menu.addSeparator()
                action_auto_x = menu.addAction("Auto Range X")
                action_auto_y = menu.addAction("Auto Range Y")
                menu.addSeparator()
                action_grid = menu.addAction("Toggle Grid")

                action = menu.exec(self.plot_widget.mapToGlobal(pos))
                if action == action_view_all:
                    self.autoscale()
                elif action == action_auto_x:
                    self.plot_item.vb.enableAutoRange(axis=pg.ViewBox.XAxis)
                elif action == action_auto_y:
                    self.plot_item.vb.enableAutoRange(axis=pg.ViewBox.YAxis)
                elif action == action_grid:
                    self._show_grid = not self._show_grid
                    self.btn_grid.setChecked(self._show_grid)
                    self.toggle_grid(self._show_grid)

    def _find_nearest_point(self, x: float, y: float, threshold: float = 0.08) -> Optional[Dict]:
        """Find the nearest data point to the given coordinates."""
        if not self._plot_data:
            return None

        # Get view range for scaling
        view_range = self.plot_item.viewRange()
        x_range = view_range[0][1] - view_range[0][0]
        y_range = view_range[1][1] - view_range[1][0]

        if x_range == 0 or y_range == 0:
            return None

        x_scale = x_range * threshold
        y_scale = y_range * threshold

        nearest = None
        min_dist = float('inf')

        for data in self._plot_data:
            x_arr = np.asarray(data['x'])
            y_arr = np.asarray(data['y'])

            # Vectorized distance calculation
            dx = (x_arr - x) / x_scale
            dy = (y_arr - y) / y_scale
            distances = dx*dx + dy*dy

            min_idx = np.argmin(distances)
            dist = distances[min_idx]

            if dist < min_dist and dist < 1.0:
                min_dist = dist
                alpha_data = data.get('alpha')
                beta_data = data.get('beta')
                nearest = {
                    'label': data['label'],
                    'x': float(x_arr[min_idx]),
                    'y': float(y_arr[min_idx]),
                    'color': data.get('color', '#ffffff'),
                    'case_id': data.get('case_id'),
                    'alpha': float(alpha_data[min_idx]) if alpha_data is not None else 0.0,
                    'beta': float(beta_data[min_idx]) if beta_data is not None else 0.0,
                }

        return nearest

    def reset_view(self):
        """Reset to default view (autoscale)."""
        self.plot_item.vb.enableAutoRange()
        self.plot_item.vb.autoRange(padding=0.05)

    def autoscale(self):
        """Autoscale axes to fit data."""
        # Disable auto range first to force recalculation
        self.plot_item.vb.disableAutoRange()
        # Re-enable and auto range
        self.plot_item.vb.enableAutoRange()
        self.plot_item.vb.autoRange(padding=0.05)

    def toggle_grid(self, show: bool = None):
        """Toggle grid visibility."""
        if show is None:
            show = self.btn_grid.isChecked()
        self._show_grid = show

        self.plot_item.showGrid(x=show, y=show, alpha=0.3 if show else 0)

    def save_figure(self):
        """Open interactive save dialog for customizing and exporting the plot."""
        from .save_image_dialog import SaveImageDialog

        # Hide crosshair during export
        if self._crosshair:
            self._crosshair.hide()

        dialog = SaveImageDialog(
            plot_items=list(self._plot_items),
            plot_data=list(self._plot_data),
            plot_item=self.plot_item,
            plot_widget=self.plot_widget,
            show_grid=self._show_grid,
            show_legend=self._legend is not None,
            parent=self
        )
        dialog.exec()

        # Restore crosshair
        if self._crosshair:
            self._crosshair.show()

    def clear(self):
        """Clear the plot."""
        # Remove legend BEFORE clearing items to avoid stale references
        if self._legend is not None:
            try:
                self._legend.scene().removeItem(self._legend)
            except Exception:
                pass
            self._legend = None
        # Clear pyqtgraph's internal legend reference to prevent auto-populating
        self.plot_item.legend = None

        self.plot_item.clear()
        self._plot_items.clear()
        self._plot_data.clear()
        self._clear_highlight()

        # Re-apply axis styling after clear (important for labels to show)
        self._apply_style()

        # Re-create crosshair after clear
        self._crosshair = CrosshairCursor(self.plot_item)

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
        """
        # Extract custom kwargs that shouldn't go to pyqtgraph
        case_id = kwargs.pop('case_id', None)
        alpha_arr = kwargs.pop('alpha_arr', None)
        beta_arr = kwargs.pop('beta_arr', None)

        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()

        # Filter out NaN/Inf values
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if len(x) == 0:
            return

        # Convert color
        if color:
            pen_color = QColor(color)
        else:
            # Use a default color from palette
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            pen_color = QColor(colors[len(self._plot_items) % len(colors)])

        # Convert marker
        symbol = self.MARKER_MAP.get(marker, 'o')

        # Create pen for line
        if linestyle == '-':
            pen = pg.mkPen(color=pen_color, width=linewidth)
        elif linestyle == '--':
            pen = pg.mkPen(color=pen_color, width=linewidth, style=Qt.PenStyle.DashLine)
        elif linestyle == ':':
            pen = pg.mkPen(color=pen_color, width=linewidth, style=Qt.PenStyle.DotLine)
        elif linestyle == '-.':
            pen = pg.mkPen(color=pen_color, width=linewidth, style=Qt.PenStyle.DashDotLine)
        elif linestyle == 'none' or linestyle == '' or linestyle is None:
            pen = None
        else:
            pen = pg.mkPen(color=pen_color, width=linewidth)

        # Create symbol brush
        symbol_brush = pg.mkBrush(color=pen_color)
        symbol_pen = pg.mkPen(color=pen_color, width=1)

        # Create plot item
        plot_item = self.plot_item.plot(
            x, y,
            pen=pen,
            symbol=symbol,
            symbolSize=markersize,
            symbolBrush=symbol_brush,
            symbolPen=symbol_pen,
            name=label
        )

        self._plot_items.append(plot_item)

        # Store data for hover/click detection
        self._plot_data.append({
            'x': x,
            'y': y,
            'label': label or '',
            'color': pen_color.name(),
            'case_id': case_id,
            'alpha': np.asarray(alpha_arr).flatten() if alpha_arr is not None else None,
            'beta': np.asarray(beta_arr).flatten() if beta_arr is not None else None,
        })

        # Ensure axis scaling stays disabled after adding data
        self._disable_axis_scaling()

    def fill_between(self, x, y_lower, y_upper, color: str = None,
                     alpha: float = 0.2):
        """Draw a shaded region between y_lower and y_upper."""
        x = np.asarray(x).flatten()
        y_lower = np.asarray(y_lower).flatten()
        y_upper = np.asarray(y_upper).flatten()

        fill_color = QColor(color) if color else QColor('#1f77b4')
        fill_color.setAlphaF(alpha)

        curve_lower = pg.PlotDataItem(x, y_lower)
        curve_upper = pg.PlotDataItem(x, y_upper)
        fill = pg.FillBetweenItem(curve_lower, curve_upper,
                                  brush=pg.mkBrush(fill_color))
        self.plot_item.addItem(fill)

    def scatter(self, x, y, label: str = None, color: str = None,
                marker: str = 'o', size: int = 50, **kwargs):
        """Add scatter plot to the canvas."""
        self.plot(x, y, label=label, color=color, marker=marker, linestyle='none')

    def set_labels(self, xlabel: str = None, ylabel: str = None,
                   title: str = None):
        """Set axis labels and title."""
        # Convert LaTeX to Unicode
        if xlabel:
            xlabel = latex_to_unicode(xlabel)
            bottom_axis = self.plot_item.getAxis('bottom')
            bottom_axis.setLabel(xlabel, color=DarkTheme.TEXT_PRIMARY)
            bottom_axis.label.setFont(QFont("Segoe UI", 12))

        if ylabel:
            ylabel = latex_to_unicode(ylabel)
            left_axis = self.plot_item.getAxis('left')
            left_axis.setLabel(ylabel, color=DarkTheme.TEXT_PRIMARY)
            left_axis.label.setFont(QFont("Segoe UI", 12))

        if title:
            title = latex_to_unicode(title)
            self.plot_item.setTitle(title, color=DarkTheme.TEXT_PRIMARY, size='14pt')

    def set_limits(self, xlim: tuple = None, ylim: tuple = None):
        """Set axis limits."""
        # Disable auto range when setting manual limits
        self.plot_item.vb.disableAutoRange()

        if xlim:
            self.plot_item.setXRange(xlim[0], xlim[1], padding=0)
        if ylim:
            self.plot_item.setYRange(ylim[0], ylim[1], padding=0)

    def add_legend(self, loc: str = 'best'):
        """Add legend to the plot."""
        if self._plot_data and self._legend is None:
            self._legend = self.plot_item.addLegend(
                offset=(10, 10),
                labelTextColor=DarkTheme.TEXT_PRIMARY,
                brush=pg.mkBrush(DarkTheme.SURFACE + 'dd'),
                pen=pg.mkPen(DarkTheme.BORDER)
            )
            # Retroactively add existing plot items (they were added while
            # plot_item.legend was None, so pyqtgraph skipped auto-adding)
            for item in self.plot_item.items:
                if isinstance(item, pg.PlotDataItem) and item.name():
                    self._legend.addItem(item, item.name())

    def remove_legend(self):
        """Remove the legend from the plot."""
        if self._legend is not None:
            try:
                self._legend.scene().removeItem(self._legend)
            except Exception:
                pass
            self._legend = None
        # Always clear pyqtgraph's internal reference to prevent auto-populating
        self.plot_item.legend = None

    def add_horizontal_line(self, y: float, color: str = None,
                            linestyle: str = '--', label: str = None):
        """Add a horizontal reference line."""
        pen_color = QColor(color) if color else QColor(DarkTheme.TEXT_SECONDARY)
        if linestyle == '--':
            pen = pg.mkPen(color=pen_color, width=1, style=Qt.PenStyle.DashLine)
        else:
            pen = pg.mkPen(color=pen_color, width=1)

        line = pg.InfiniteLine(pos=y, angle=0, pen=pen, label=label)
        self.plot_item.addItem(line)

    def add_vertical_line(self, x: float, color: str = None,
                          linestyle: str = '--', label: str = None):
        """Add a vertical reference line."""
        pen_color = QColor(color) if color else QColor(DarkTheme.TEXT_SECONDARY)
        if linestyle == '--':
            pen = pg.mkPen(color=pen_color, width=1, style=Qt.PenStyle.DashLine)
        else:
            pen = pg.mkPen(color=pen_color, width=1)

        line = pg.InfiniteLine(pos=x, angle=90, pen=pen, label=label)
        self.plot_item.addItem(line)

    def refresh(self):
        """Refresh the canvas."""
        # Auto-range after plotting if no manual limits set
        if not self._plot_items:
            return

        self._disable_axis_scaling()
        self.plot_item.vb.autoRange(padding=0.05)
        self.plot_updated.emit()


def is_available() -> bool:
    """Check if fast plotting is available (pyqtgraph installed)."""
    return PYQTGRAPH_AVAILABLE
