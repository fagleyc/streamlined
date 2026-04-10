"""
Save Image Dialog
=================

Interactive dialog for customizing plot appearance before saving.
Allows editing legend labels, line widths, marker sizes, colors,
marker types, axis limits, axis labels, and export themes.
Settings are persisted between uses via QSettings.
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QDoubleSpinBox, QComboBox,
    QPushButton, QFileDialog, QScrollArea, QWidget,
    QSpinBox, QSizePolicy, QCheckBox, QColorDialog, QToolButton
)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QColor, QFont

try:
    import pyqtgraph as pg
    from pyqtgraph import exporters
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False


# Export theme definitions
EXPORT_THEMES = {
    'White': {
        'background': '#ffffff',
        'axis_color': '#333333',
        'text_color': '#000000',
        'grid_color': '#cccccc',
        'legend_bg': '#ffffff',
        'legend_border': '#333333',
        'legend_text': '#000000',
    },
    'Dark': {
        'background': '#1e1e1e',
        'axis_color': '#a0a0a0',
        'text_color': '#e0e0e0',
        'grid_color': '#404040',
        'legend_bg': '#333333',
        'legend_border': '#555555',
        'legend_text': '#e0e0e0',
    },
    'Black': {
        'background': '#000000',
        'axis_color': '#808080',
        'text_color': '#ffffff',
        'grid_color': '#303030',
        'legend_bg': '#1a1a1a',
        'legend_border': '#444444',
        'legend_text': '#ffffff',
    },
    'Transparent': {
        'background': None,  # transparent
        'axis_color': '#333333',
        'text_color': '#000000',
        'grid_color': '#cccccc',
        'legend_bg': '#ffffffcc',
        'legend_border': '#333333',
        'legend_text': '#000000',
    },
}

# Marker types available for selection (display name -> pyqtgraph symbol)
MARKER_TYPES = {
    'Circle': 'o',
    'Square': 's',
    'Triangle Up': 't',
    'Triangle Down': 't1',
    'Diamond': 'd',
    'Triangle Left': 't2',
    'Triangle Right': 't3',
    'Pentagon': 'p',
    'Hexagon': 'h',
    'Star': 'star',
    'Plus': '+',
    'Cross': 'x',
}

# Reverse lookup: pyqtgraph symbol -> display name
_SYMBOL_TO_NAME = {v: k for k, v in MARKER_TYPES.items()}

# Settings key prefix
_SETTINGS_PREFIX = "save_image/"


class TraceEditor(QWidget):
    """Editor row for a single plot trace."""

    def __init__(self, index: int, label: str, color: str,
                 linewidth: float, markersize: float,
                 marker_symbol: str = 'o', parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        # Color button (clickable swatch)
        self.btn_color = QToolButton()
        self.btn_color.setFixedSize(22, 22)
        self._color = color
        self._update_color_swatch()
        self.btn_color.clicked.connect(self._pick_color)
        layout.addWidget(self.btn_color)

        # Legend label
        self.txt_label = QLineEdit(label)
        self.txt_label.setMinimumWidth(120)
        layout.addWidget(self.txt_label, stretch=2)

        # Line width
        self.spn_linewidth = QDoubleSpinBox()
        self.spn_linewidth.setRange(0.5, 10.0)
        self.spn_linewidth.setSingleStep(0.5)
        self.spn_linewidth.setValue(linewidth)
        self.spn_linewidth.setPrefix("W: ")
        self.spn_linewidth.setFixedWidth(80)
        layout.addWidget(self.spn_linewidth)

        # Marker size
        self.spn_markersize = QDoubleSpinBox()
        self.spn_markersize.setRange(0.0, 20.0)
        self.spn_markersize.setSingleStep(1.0)
        self.spn_markersize.setValue(markersize)
        self.spn_markersize.setPrefix("M: ")
        self.spn_markersize.setFixedWidth(80)
        layout.addWidget(self.spn_markersize)

        # Marker type
        self.cmb_marker = QComboBox()
        self.cmb_marker.addItems(list(MARKER_TYPES.keys()))
        marker_name = _SYMBOL_TO_NAME.get(marker_symbol, 'Circle')
        self.cmb_marker.setCurrentText(marker_name)
        self.cmb_marker.setFixedWidth(110)
        layout.addWidget(self.cmb_marker)

        self.index = index

    @property
    def color(self) -> str:
        return self._color

    @color.setter
    def color(self, value: str):
        self._color = value
        self._update_color_swatch()

    def _update_color_swatch(self):
        self.btn_color.setStyleSheet(
            f"background-color: {self._color}; border: 1px solid #888; border-radius: 2px;"
        )

    def _pick_color(self):
        c = QColorDialog.getColor(QColor(self._color), self, "Pick Trace Color")
        if c.isValid():
            self.color = c.name()
            # Emit change via linewidth signal (reuses existing connection)
            self.spn_linewidth.valueChanged.emit(self.spn_linewidth.value())

    def get_marker_symbol(self) -> str:
        return MARKER_TYPES.get(self.cmb_marker.currentText(), 'o')


class SaveImageDialog(QDialog):
    """Interactive dialog for customizing and saving plot images."""

    def __init__(self, plot_items: List['pg.PlotDataItem'],
                 plot_data: List[Dict[str, Any]],
                 plot_item: 'pg.PlotItem',
                 plot_widget: 'pg.PlotWidget',
                 current_labels: Dict[str, str] = None,
                 show_grid: bool = True,
                 show_legend: bool = True,
                 last_directory: str = '',
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Image")
        self.setMinimumSize(600, 650)

        self._plot_items = plot_items
        self._plot_data = plot_data
        self._plot_item = plot_item
        self._plot_widget = plot_widget
        self._current_labels = current_labels or {}
        self._show_grid = show_grid
        self._show_legend = show_legend
        self._last_directory = last_directory
        self._settings = QSettings("WindTunnelLab", "DataAnalyzer")

        self._build_ui()
        self._load_settings()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # --- Row 1: Theme, Format, Width ---
        top_row = QHBoxLayout()

        theme_group = QGroupBox("Export Theme")
        theme_layout = QHBoxLayout(theme_group)
        self.cmb_theme = QComboBox()
        self.cmb_theme.addItems(list(EXPORT_THEMES.keys()))
        self.cmb_theme.setCurrentText('White')
        self.cmb_theme.currentTextChanged.connect(self._update_preview)
        theme_layout.addWidget(self.cmb_theme)
        top_row.addWidget(theme_group)

        fmt_group = QGroupBox("Format")
        fmt_layout = QHBoxLayout(fmt_group)
        self.cmb_format = QComboBox()
        self.cmb_format.addItems(['PNG', 'SVG'])
        fmt_layout.addWidget(self.cmb_format)
        top_row.addWidget(fmt_group)

        res_group = QGroupBox("Width (px)")
        res_layout = QHBoxLayout(res_group)
        self.spn_width = QSpinBox()
        self.spn_width.setRange(640, 7680)
        self.spn_width.setSingleStep(160)
        self.spn_width.setValue(1920)
        res_layout.addWidget(self.spn_width)
        top_row.addWidget(res_group)

        layout.addLayout(top_row)

        # --- Row 2: Fonts, grid, legend ---
        font_row = QHBoxLayout()

        font_row.addWidget(QLabel("Label Font:"))
        self.spn_label_font = QSpinBox()
        self.spn_label_font.setRange(8, 36)
        self.spn_label_font.setValue(16)
        self.spn_label_font.setSuffix(" pt")
        self.spn_label_font.valueChanged.connect(self._update_preview)
        font_row.addWidget(self.spn_label_font)

        font_row.addWidget(QLabel("Tick Font:"))
        self.spn_tick_font = QSpinBox()
        self.spn_tick_font.setRange(6, 28)
        self.spn_tick_font.setValue(14)
        self.spn_tick_font.setSuffix(" pt")
        self.spn_tick_font.valueChanged.connect(self._update_preview)
        font_row.addWidget(self.spn_tick_font)

        font_row.addWidget(QLabel("Legend Font:"))
        self.spn_legend_font = QSpinBox()
        self.spn_legend_font.setRange(6, 28)
        self.spn_legend_font.setValue(12)
        self.spn_legend_font.setSuffix(" pt")
        self.spn_legend_font.valueChanged.connect(self._update_preview)
        font_row.addWidget(self.spn_legend_font)

        self.chk_grid = QCheckBox("Grid")
        self.chk_grid.setChecked(self._show_grid)
        self.chk_grid.stateChanged.connect(self._update_preview)
        font_row.addWidget(self.chk_grid)

        self.chk_legend = QCheckBox("Legend")
        self.chk_legend.setChecked(self._show_legend)
        self.chk_legend.stateChanged.connect(self._update_preview)
        font_row.addWidget(self.chk_legend)

        font_row.addStretch()
        layout.addLayout(font_row)

        # --- Row 3: Axis labels ---
        labels_group = QGroupBox("Axis Labels")
        labels_layout = QGridLayout(labels_group)

        # Get current axis labels from the plot
        x_label = ''
        y_label = ''
        src_bottom = self._plot_item.getAxis('bottom')
        src_left = self._plot_item.getAxis('left')
        if src_bottom.labelText:
            x_label = src_bottom.labelText
        if src_left.labelText:
            y_label = src_left.labelText

        labels_layout.addWidget(QLabel("X Label:"), 0, 0)
        self.txt_xlabel = QLineEdit(x_label)
        self.txt_xlabel.textChanged.connect(self._update_preview)
        labels_layout.addWidget(self.txt_xlabel, 0, 1)

        labels_layout.addWidget(QLabel("Y Label:"), 0, 2)
        self.txt_ylabel = QLineEdit(y_label)
        self.txt_ylabel.textChanged.connect(self._update_preview)
        labels_layout.addWidget(self.txt_ylabel, 0, 3)

        layout.addWidget(labels_group)

        # --- Row 4: Axis limits ---
        limits_group = QGroupBox("Axis Limits (leave blank for auto)")
        limits_layout = QGridLayout(limits_group)

        # Get current view range
        vb = self._plot_item.getViewBox()
        x_range, y_range = vb.viewRange()

        limits_layout.addWidget(QLabel("X Min:"), 0, 0)
        self.txt_xmin = QLineEdit(f"{x_range[0]:.4g}")
        self.txt_xmin.setFixedWidth(90)
        self.txt_xmin.editingFinished.connect(self._update_preview)
        limits_layout.addWidget(self.txt_xmin, 0, 1)

        limits_layout.addWidget(QLabel("X Max:"), 0, 2)
        self.txt_xmax = QLineEdit(f"{x_range[1]:.4g}")
        self.txt_xmax.setFixedWidth(90)
        self.txt_xmax.editingFinished.connect(self._update_preview)
        limits_layout.addWidget(self.txt_xmax, 0, 3)

        limits_layout.addWidget(QLabel("Y Min:"), 0, 4)
        self.txt_ymin = QLineEdit(f"{y_range[0]:.4g}")
        self.txt_ymin.setFixedWidth(90)
        self.txt_ymin.editingFinished.connect(self._update_preview)
        limits_layout.addWidget(self.txt_ymin, 0, 5)

        limits_layout.addWidget(QLabel("Y Max:"), 0, 6)
        self.txt_ymax = QLineEdit(f"{y_range[1]:.4g}")
        self.txt_ymax.setFixedWidth(90)
        self.txt_ymax.editingFinished.connect(self._update_preview)
        limits_layout.addWidget(self.txt_ymax, 0, 7)

        btn_auto = QPushButton("Auto")
        btn_auto.setFixedWidth(50)
        btn_auto.clicked.connect(self._auto_limits)
        limits_layout.addWidget(btn_auto, 0, 8)

        layout.addWidget(limits_group)

        # --- Trace editors ---
        traces_group = QGroupBox("Traces")
        traces_vlayout = QVBoxLayout(traces_group)

        # Header
        header = QHBoxLayout()
        header.addSpacing(26)  # color button
        lbl_h1 = QLabel("Legend Label")
        lbl_h1.setMinimumWidth(120)
        header.addWidget(lbl_h1, stretch=2)
        lbl_h2 = QLabel("Width")
        lbl_h2.setFixedWidth(80)
        header.addWidget(lbl_h2)
        lbl_h3 = QLabel("Marker Sz")
        lbl_h3.setFixedWidth(80)
        header.addWidget(lbl_h3)
        lbl_h4 = QLabel("Marker Type")
        lbl_h4.setFixedWidth(110)
        header.addWidget(lbl_h4)
        traces_vlayout.addLayout(header)

        # Scrollable trace list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(200)
        scroll_widget = QWidget()
        self.traces_inner_layout = QVBoxLayout(scroll_widget)
        self.traces_inner_layout.setContentsMargins(0, 0, 0, 0)
        self.traces_inner_layout.setSpacing(2)

        self.trace_editors: List[TraceEditor] = []
        for i, (item, data) in enumerate(zip(self._plot_items, self._plot_data)):
            label = data.get('label', f'Trace {i}')
            color = data.get('color', '#1f77b4')

            lw = 1.5
            ms = 6.0
            marker_sym = 'o'
            if item.opts.get('pen') is not None:
                try:
                    lw = item.opts['pen'].widthF()
                except (AttributeError, TypeError):
                    pass
            if item.opts.get('symbolSize') is not None:
                try:
                    ms = float(item.opts['symbolSize'])
                except (TypeError, ValueError):
                    pass
            if item.opts.get('symbol') is not None:
                marker_sym = item.opts['symbol']

            editor = TraceEditor(i, label, color, lw, ms, marker_sym)
            editor.spn_linewidth.valueChanged.connect(self._update_preview)
            editor.spn_markersize.valueChanged.connect(self._update_preview)
            editor.txt_label.textChanged.connect(self._update_preview)
            editor.cmb_marker.currentTextChanged.connect(self._update_preview)
            self.trace_editors.append(editor)
            self.traces_inner_layout.addWidget(editor)

        self.traces_inner_layout.addStretch()
        scroll.setWidget(scroll_widget)
        traces_vlayout.addWidget(scroll)

        # Bulk controls
        bulk_row = QHBoxLayout()
        bulk_row.addWidget(QLabel("Set All Widths:"))
        self.spn_bulk_lw = QDoubleSpinBox()
        self.spn_bulk_lw.setRange(0.5, 10.0)
        self.spn_bulk_lw.setSingleStep(0.5)
        self.spn_bulk_lw.setValue(1.5)
        bulk_row.addWidget(self.spn_bulk_lw)
        btn_apply_lw = QPushButton("Apply")
        btn_apply_lw.setFixedWidth(60)
        btn_apply_lw.clicked.connect(self._apply_bulk_linewidth)
        bulk_row.addWidget(btn_apply_lw)

        bulk_row.addSpacing(20)

        bulk_row.addWidget(QLabel("Set All Markers:"))
        self.spn_bulk_ms = QDoubleSpinBox()
        self.spn_bulk_ms.setRange(0.0, 20.0)
        self.spn_bulk_ms.setSingleStep(1.0)
        self.spn_bulk_ms.setValue(6.0)
        bulk_row.addWidget(self.spn_bulk_ms)
        btn_apply_ms = QPushButton("Apply")
        btn_apply_ms.setFixedWidth(60)
        btn_apply_ms.clicked.connect(self._apply_bulk_markersize)
        bulk_row.addWidget(btn_apply_ms)

        bulk_row.addStretch()
        traces_vlayout.addLayout(bulk_row)

        layout.addWidget(traces_group)

        # --- Preview ---
        if PYQTGRAPH_AVAILABLE:
            preview_group = QGroupBox("Preview")
            preview_layout = QVBoxLayout(preview_group)
            self.preview_widget = pg.PlotWidget()
            self.preview_widget.setMinimumHeight(220)
            self.preview_widget.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            self.preview_item = self.preview_widget.getPlotItem()
            self.preview_widget.setMouseEnabled(x=False, y=False)
            self.preview_item.vb.setMouseMode(pg.ViewBox.PanMode)
            preview_layout.addWidget(self.preview_widget)
            layout.addWidget(preview_group, stretch=1)

            self._update_preview()

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)

        btn_save = QPushButton("Save")
        btn_save.setDefault(True)
        btn_save.clicked.connect(self._do_save)
        btn_layout.addWidget(btn_save)

        layout.addLayout(btn_layout)

    # --- Settings persistence ---

    def _load_settings(self):
        """Load saved settings from QSettings."""
        s = self._settings
        p = _SETTINGS_PREFIX

        theme = s.value(p + "theme", "White")
        if theme in EXPORT_THEMES:
            self.cmb_theme.setCurrentText(theme)

        fmt = s.value(p + "format", "PNG")
        if fmt in ('PNG', 'SVG'):
            self.cmb_format.setCurrentText(fmt)

        width = s.value(p + "width", 1920, type=int)
        self.spn_width.setValue(width)

        label_font = s.value(p + "label_font", 16, type=int)
        self.spn_label_font.setValue(label_font)

        tick_font = s.value(p + "tick_font", 14, type=int)
        self.spn_tick_font.setValue(tick_font)

        legend_font = s.value(p + "legend_font", 12, type=int)
        self.spn_legend_font.setValue(legend_font)

    def _save_settings(self):
        """Persist current dialog settings to QSettings."""
        s = self._settings
        p = _SETTINGS_PREFIX

        s.setValue(p + "theme", self.cmb_theme.currentText())
        s.setValue(p + "format", self.cmb_format.currentText())
        s.setValue(p + "width", self.spn_width.value())
        s.setValue(p + "label_font", self.spn_label_font.value())
        s.setValue(p + "tick_font", self.spn_tick_font.value())
        s.setValue(p + "legend_font", self.spn_legend_font.value())

    # --- Bulk actions ---

    def _apply_bulk_linewidth(self):
        val = self.spn_bulk_lw.value()
        for editor in self.trace_editors:
            editor.spn_linewidth.setValue(val)

    def _apply_bulk_markersize(self):
        val = self.spn_bulk_ms.value()
        for editor in self.trace_editors:
            editor.spn_markersize.setValue(val)

    def _auto_limits(self):
        """Reset axis limits to auto-range from data."""
        all_x = []
        all_y = []
        for data in self._plot_data:
            all_x.extend(np.asarray(data['x']).flatten().tolist())
            all_y.extend(np.asarray(data['y']).flatten().tolist())

        if all_x:
            xmin, xmax = min(all_x), max(all_x)
            pad = (xmax - xmin) * 0.05 if xmax != xmin else 1.0
            self.txt_xmin.setText(f"{xmin - pad:.4g}")
            self.txt_xmax.setText(f"{xmax + pad:.4g}")

        if all_y:
            ymin, ymax = min(all_y), max(all_y)
            pad = (ymax - ymin) * 0.05 if ymax != ymin else 1.0
            self.txt_ymin.setText(f"{ymin - pad:.4g}")
            self.txt_ymax.setText(f"{ymax + pad:.4g}")

        self._update_preview()

    def _get_axis_limits(self):
        """Parse axis limit text fields. Returns (xmin, xmax, ymin, ymax) or None for auto."""
        def _parse(text):
            text = text.strip()
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                return None

        return _parse(self.txt_xmin.text()), _parse(self.txt_xmax.text()), \
               _parse(self.txt_ymin.text()), _parse(self.txt_ymax.text())

    # --- Preview ---

    def _update_preview(self, *_args):
        """Rebuild the preview plot with current dialog settings."""
        if not PYQTGRAPH_AVAILABLE:
            return

        self.preview_item.clear()
        if hasattr(self.preview_item, 'legend') and self.preview_item.legend is not None:
            try:
                self.preview_item.legend.scene().removeItem(self.preview_item.legend)
            except Exception:
                pass
            self.preview_item.legend = None

        theme_name = self.cmb_theme.currentText()
        theme = EXPORT_THEMES[theme_name]

        # Background
        if theme['background'] is None:
            self.preview_widget.setBackground(QColor(0, 0, 0, 0))
        else:
            self.preview_widget.setBackground(theme['background'])

        # Axis styling
        axis_pen = pg.mkPen(color=theme['axis_color'], width=1.5)
        text_pen = pg.mkPen(theme['text_color'])
        tick_font = QFont("Segoe UI", max(8, self.spn_tick_font.value() - 2))
        label_font = QFont("Segoe UI", max(8, self.spn_label_font.value() - 2),
                           QFont.Weight.Bold)

        for axis_name in ['bottom', 'left', 'top', 'right']:
            ax = self.preview_item.getAxis(axis_name)
            ax.setPen(axis_pen)
            ax.setTextPen(text_pen)
            ax.setStyle(tickFont=tick_font)
            if ax.label:
                ax.label.setFont(label_font)

        # Show top and right axes as border lines (no tick labels)
        for border_axis in ['top', 'right']:
            ax = self.preview_item.getAxis(border_axis)
            ax.setVisible(True)
            ax.setStyle(showValues=False)
            ax.setTicks([])

        # Axis labels from text fields
        xlabel = self.txt_xlabel.text()
        ylabel = self.txt_ylabel.text()
        if xlabel:
            self.preview_item.getAxis('bottom').setLabel(xlabel, color=theme['text_color'])
        if ylabel:
            self.preview_item.getAxis('left').setLabel(ylabel, color=theme['text_color'])

        # Grid
        show_grid = self.chk_grid.isChecked()
        self.preview_item.showGrid(x=show_grid, y=show_grid,
                                   alpha=0.3 if show_grid else 0)

        # Legend
        show_legend = self.chk_legend.isChecked()
        if show_legend:
            legend = self.preview_item.addLegend()
            legend.setLabelTextColor(theme['legend_text'])
            legend.setBrush(pg.mkBrush(theme['legend_bg']))
            legend.setPen(pg.mkPen(theme['legend_border']))

        # Plot traces
        for i, (item, data) in enumerate(zip(self._plot_items, self._plot_data)):
            if i >= len(self.trace_editors):
                break
            editor = self.trace_editors[i]

            x = np.asarray(data['x'])
            y = np.asarray(data['y'])
            color = QColor(editor.color)
            label = editor.txt_label.text()
            lw = editor.spn_linewidth.value()
            ms = editor.spn_markersize.value()
            symbol = editor.get_marker_symbol()

            # Reconstruct pen style from original
            pen_style = Qt.PenStyle.SolidLine
            orig_pen = item.opts.get('pen')
            if orig_pen is not None:
                try:
                    pen_style = orig_pen.style()
                except (AttributeError, TypeError):
                    pass

            pen = pg.mkPen(color=color, width=lw, style=pen_style)
            symbol_brush = pg.mkBrush(color=color)
            symbol_pen = pg.mkPen(color=color, width=1)

            self.preview_item.plot(
                x, y, pen=pen,
                symbol=symbol,
                symbolSize=ms,
                symbolBrush=symbol_brush,
                symbolPen=symbol_pen,
                name=label if show_legend else None
            )

        # Apply legend font size after traces are added
        if show_legend and self.preview_item.legend is not None:
            font_size = f'{max(6, self.spn_legend_font.value() - 2)}pt'
            for sample, lbl in self.preview_item.legend.items:
                lbl.setAttr('size', font_size)
                lbl.setText(lbl.text)

        # Apply axis limits
        xmin, xmax, ymin, ymax = self._get_axis_limits()
        if xmin is not None and xmax is not None:
            self.preview_item.setXRange(xmin, xmax, padding=0)
        if ymin is not None and ymax is not None:
            self.preview_item.setYRange(ymin, ymax, padding=0)

        if (xmin is None or xmax is None) and (ymin is None or ymax is None):
            self.preview_item.autoRange()

    # --- Save ---

    def _do_save(self):
        """Save the figure with the configured settings."""
        fmt = self.cmb_format.currentText()
        if fmt == 'PNG':
            filter_str = "PNG Image (*.png)"
            default_ext = ".png"
        else:
            filter_str = "SVG Vector (*.svg)"
            default_ext = ".svg"

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Figure", self._last_directory,
            f"{filter_str};;All Files (*.*)"
        )
        if not filepath:
            return

        if not filepath.lower().endswith(default_ext):
            filepath += default_ext

        self._save_settings()
        self._export_to_file(filepath)
        self.accept()

    def _export_to_file(self, filepath: str):
        """Apply theme and trace settings to the actual plot, export, then restore."""
        theme_name = self.cmb_theme.currentText()
        theme = EXPORT_THEMES[theme_name]

        # --- Save original state ---
        old_bg = self._plot_widget.backgroundBrush().color()
        old_axis_state = {}
        for axis_name in ['bottom', 'left', 'top', 'right']:
            ax = self._plot_item.getAxis(axis_name)
            old_axis_state[axis_name] = {
                'pen': ax.pen(),
                'text_pen': ax.textPen(),
                'label_text': ax.labelText,
            }

        old_trace_state = []
        for item in self._plot_items:
            old_trace_state.append({
                'pen': item.opts.get('pen'),
                'symbolSize': item.opts.get('symbolSize'),
                'symbol': item.opts.get('symbol'),
                'symbolBrush': item.opts.get('symbolBrush'),
                'symbolPen': item.opts.get('symbolPen'),
                'name': item.name(),
            })

        had_legend = self._plot_item.legend is not None
        old_view_range = self._plot_item.getViewBox().viewRange()

        # --- Apply export theme ---
        if theme['background'] is None:
            self._plot_widget.setBackground(QColor(0, 0, 0, 0))
        else:
            self._plot_widget.setBackground(theme['background'])

        axis_pen = pg.mkPen(color=theme['axis_color'], width=1.5)
        text_pen = pg.mkPen(theme['text_color'])
        export_font = QFont("Segoe UI", self.spn_tick_font.value())
        label_font = QFont("Segoe UI", self.spn_label_font.value(), QFont.Weight.Bold)

        for axis_name in ['bottom', 'left', 'top', 'right']:
            ax = self._plot_item.getAxis(axis_name)
            ax.setPen(axis_pen)
            ax.setTextPen(text_pen)
            ax.setStyle(tickFont=export_font)
            if ax.label:
                ax.label.setFont(label_font)

        # Show top and right axes as border lines (no tick labels)
        for border_axis in ['top', 'right']:
            ax = self._plot_item.getAxis(border_axis)
            ax.setVisible(True)
            ax.setStyle(showValues=False)
            ax.setTicks([])

        # Apply custom axis labels
        xlabel = self.txt_xlabel.text()
        ylabel = self.txt_ylabel.text()
        if xlabel:
            self._plot_item.getAxis('bottom').setLabel(xlabel, color=theme['text_color'])
        if ylabel:
            self._plot_item.getAxis('left').setLabel(ylabel, color=theme['text_color'])

        # Grid
        show_grid = self.chk_grid.isChecked()
        self._plot_item.showGrid(x=show_grid, y=show_grid,
                                 alpha=0.3 if show_grid else 0)

        # Remove existing legend
        if self._plot_item.legend is not None:
            try:
                self._plot_item.legend.scene().removeItem(self._plot_item.legend)
            except Exception:
                pass
            self._plot_item.legend = None

        # Apply trace settings
        for i, item in enumerate(self._plot_items):
            if i >= len(self.trace_editors):
                break
            editor = self.trace_editors[i]
            color = QColor(editor.color)
            lw = editor.spn_linewidth.value()
            ms = editor.spn_markersize.value()
            label = editor.txt_label.text()
            symbol = editor.get_marker_symbol()

            # Update pen width (preserve style)
            orig_pen = item.opts.get('pen')
            if orig_pen is not None:
                try:
                    pen_style = orig_pen.style()
                except (AttributeError, TypeError):
                    pen_style = Qt.PenStyle.SolidLine
                new_pen = pg.mkPen(color=color, width=lw, style=pen_style)
                item.setPen(new_pen)

            item.setSymbolSize(ms)
            item.setSymbol(symbol)
            item.setSymbolBrush(pg.mkBrush(color=color))
            item.setSymbolPen(pg.mkPen(color=color, width=1))
            item.opts['name'] = label

        # Add legend with export theme
        show_legend = self.chk_legend.isChecked()
        if show_legend:
            legend = self._plot_item.addLegend()
            legend.setLabelTextColor(theme['legend_text'])
            legend.setBrush(pg.mkBrush(theme['legend_bg']))
            legend.setPen(pg.mkPen(theme['legend_border']))
            for item in self._plot_items:
                name = item.opts.get('name', '')
                if name:
                    legend.addItem(item, name)
            # Apply legend font size
            font_size = f'{self.spn_legend_font.value()}pt'
            for sample, lbl in legend.items:
                lbl.setAttr('size', font_size)
                lbl.setText(lbl.text)

        # Apply axis limits
        xmin, xmax, ymin, ymax = self._get_axis_limits()
        if xmin is not None and xmax is not None:
            self._plot_item.setXRange(xmin, xmax, padding=0)
        if ymin is not None and ymax is not None:
            self._plot_item.setYRange(ymin, ymax, padding=0)

        # --- Export ---
        # Force layout update so axis labels are fully positioned
        self._plot_widget.repaint()
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        if filepath.lower().endswith('.svg'):
            exporter = exporters.SVGExporter(self._plot_item)
            exporter.export(filepath)
        else:
            # Render scene directly to QImage — captures all labels without
            # the QGraphicsView widget frame border
            from PyQt6.QtGui import QImage, QPainter
            from PyQt6.QtCore import QRectF

            scene = self._plot_widget.scene()
            source_rect = scene.itemsBoundingRect()

            target_width = self.spn_width.value()
            aspect = source_rect.height() / max(source_rect.width(), 1.0)
            target_height = int(target_width * aspect)

            image = QImage(target_width, target_height,
                           QImage.Format.Format_ARGB32)

            # Fill with theme background
            if theme['background'] is None:
                image.fill(QColor(0, 0, 0, 0))
            else:
                image.fill(QColor(theme['background']))

            painter = QPainter(image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            scene.render(painter,
                         QRectF(0, 0, target_width, target_height),
                         source_rect)
            painter.end()

            image.save(filepath)

        # --- Restore original state ---
        self._plot_widget.setBackground(old_bg)

        # Hide top/right border axes again
        for border_axis in ['top', 'right']:
            ax = self._plot_item.getAxis(border_axis)
            ax.setVisible(False)
            ax.setStyle(showValues=True)
            ax.setTicks(None)

        for axis_name, state in old_axis_state.items():
            ax = self._plot_item.getAxis(axis_name)
            ax.setPen(state['pen'])
            ax.setTextPen(state['text_pen'])

        # Restore trace settings
        for i, (item, state) in enumerate(zip(self._plot_items, old_trace_state)):
            if state['pen'] is not None:
                item.setPen(state['pen'])
            if state['symbolSize'] is not None:
                item.setSymbolSize(state['symbolSize'])
            if state['symbol'] is not None:
                item.setSymbol(state['symbol'])
            if state['symbolBrush'] is not None:
                item.setSymbolBrush(state['symbolBrush'])
            if state['symbolPen'] is not None:
                item.setSymbolPen(state['symbolPen'])
            item.opts['name'] = state['name']

        # Restore legend
        if self._plot_item.legend is not None:
            try:
                self._plot_item.legend.scene().removeItem(self._plot_item.legend)
            except Exception:
                pass
            self._plot_item.legend = None

        if had_legend:
            legend = self._plot_item.addLegend()
            from ..utils.themes import DarkTheme
            legend.setLabelTextColor(DarkTheme.TEXT_PRIMARY)
            legend.setBrush(pg.mkBrush(DarkTheme.SURFACE + 'dd'))
            legend.setPen(pg.mkPen(DarkTheme.BORDER))
            for item in self._plot_items:
                name = item.opts.get('name', '')
                if name:
                    legend.addItem(item, name)

        # Restore view range
        self._plot_item.setXRange(*old_view_range[0], padding=0)
        self._plot_item.setYRange(*old_view_range[1], padding=0)

        # Restore grid and axis style
        from ..utils.themes import DarkTheme
        self._plot_item.showGrid(x=self._show_grid, y=self._show_grid,
                                 alpha=0.3 if self._show_grid else 0)
        axis_pen = pg.mkPen(color=DarkTheme.TEXT_SECONDARY, width=1)
        for axis_name in ['bottom', 'left', 'top', 'right']:
            ax = self._plot_item.getAxis(axis_name)
            ax.setPen(axis_pen)
            ax.setTextPen(pg.mkPen(DarkTheme.TEXT_PRIMARY))
            ax.setStyle(tickFont=QFont("Segoe UI", 10))
