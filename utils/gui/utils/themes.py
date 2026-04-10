"""
Theme and Styling
=================

Dark theme styling for the application and matplotlib plots.
"""

from typing import Dict, Any
import matplotlib.pyplot as plt
from matplotlib import rcParams


class DarkTheme:
    """Dark theme color palette."""

    # Base colors
    BACKGROUND = "#1e1e1e"
    BACKGROUND_LIGHT = "#252526"
    BACKGROUND_LIGHTER = "#2d2d30"
    SURFACE = "#333333"

    # Text colors
    TEXT_PRIMARY = "#e0e0e0"
    TEXT_SECONDARY = "#a0a0a0"
    TEXT_DISABLED = "#606060"

    # Accent colors
    ACCENT = "#0078d4"
    ACCENT_LIGHT = "#3399ff"
    ACCENT_DARK = "#005a9e"

    # Status colors
    SUCCESS = "#4caf50"
    WARNING = "#ff9800"
    ERROR = "#f44336"
    INFO = "#2196f3"

    # Border colors
    BORDER = "#3f3f46"
    BORDER_LIGHT = "#4f4f56"

    # Selection
    SELECTION = "#264f78"
    HOVER = "#3a3a3c"

    # Plot colors
    PLOT_BACKGROUND = "#1e1e1e"
    PLOT_AXES = "#333333"
    PLOT_GRID = "#404040"
    PLOT_TEXT = "#e0e0e0"


def get_dark_stylesheet() -> str:
    """Generate the dark theme Qt stylesheet."""
    t = DarkTheme

    return f"""
    /* Main Window */
    QMainWindow {{
        background-color: {t.BACKGROUND};
    }}

    /* Central Widget */
    QWidget {{
        background-color: {t.BACKGROUND};
        color: {t.TEXT_PRIMARY};
        font-family: "Segoe UI", "SF Pro Display", sans-serif;
        font-size: 10pt;
    }}

    /* Frames and Group Boxes */
    QFrame {{
        background-color: {t.BACKGROUND_LIGHT};
        border: 1px solid {t.BORDER};
        border-radius: 4px;
    }}

    QGroupBox {{
        background-color: {t.BACKGROUND_LIGHT};
        border: 1px solid {t.BORDER};
        border-radius: 6px;
        margin-top: 12px;
        padding-top: 10px;
        font-weight: bold;
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0 5px;
        color: {t.TEXT_PRIMARY};
    }}

    /* Labels */
    QLabel {{
        background-color: transparent;
        border: none;
        color: {t.TEXT_PRIMARY};
    }}

    /* Buttons */
    QPushButton {{
        background-color: {t.SURFACE};
        border: 1px solid {t.BORDER};
        border-radius: 4px;
        padding: 6px 16px;
        color: {t.TEXT_PRIMARY};
        min-height: 24px;
    }}

    QPushButton:hover {{
        background-color: {t.HOVER};
        border-color: {t.ACCENT};
    }}

    QPushButton:pressed {{
        background-color: {t.ACCENT_DARK};
    }}

    QPushButton:disabled {{
        background-color: {t.BACKGROUND_LIGHTER};
        color: {t.TEXT_DISABLED};
        border-color: {t.BORDER};
    }}

    QPushButton:checked {{
        background-color: {t.ACCENT};
        border-color: {t.ACCENT_LIGHT};
    }}

    /* Primary Button Style */
    QPushButton[primary="true"] {{
        background-color: {t.ACCENT};
        border-color: {t.ACCENT};
        color: white;
    }}

    QPushButton[primary="true"]:hover {{
        background-color: {t.ACCENT_LIGHT};
    }}

    /* Tool Buttons */
    QToolButton {{
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 4px;
        padding: 4px;
    }}

    QToolButton:hover {{
        background-color: {t.HOVER};
        border-color: {t.BORDER};
    }}

    QToolButton:pressed {{
        background-color: {t.SELECTION};
    }}

    QToolButton:checked {{
        background-color: {t.ACCENT_DARK};
        border-color: {t.ACCENT};
    }}

    /* Line Edits */
    QLineEdit {{
        background-color: {t.SURFACE};
        border: 1px solid {t.BORDER};
        border-radius: 4px;
        padding: 6px 10px;
        color: {t.TEXT_PRIMARY};
        selection-background-color: {t.ACCENT};
    }}

    QLineEdit:focus {{
        border-color: {t.ACCENT};
    }}

    QLineEdit:disabled {{
        background-color: {t.BACKGROUND_LIGHTER};
        color: {t.TEXT_DISABLED};
    }}

    /* Spin Boxes */
    QSpinBox, QDoubleSpinBox {{
        background-color: {t.SURFACE};
        border: 1px solid {t.BORDER};
        border-radius: 4px;
        padding: 4px 8px;
        color: {t.TEXT_PRIMARY};
    }}

    QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {t.ACCENT};
    }}

    QSpinBox::up-button, QDoubleSpinBox::up-button,
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        background-color: {t.BACKGROUND_LIGHTER};
        border: none;
        width: 16px;
    }}

    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
        background-color: {t.HOVER};
    }}

    /* Combo Boxes */
    QComboBox {{
        background-color: {t.SURFACE};
        border: 1px solid {t.BORDER};
        border-radius: 4px;
        padding: 6px 10px;
        color: {t.TEXT_PRIMARY};
        min-width: 80px;
    }}

    QComboBox:hover {{
        border-color: {t.ACCENT};
    }}

    QComboBox:focus {{
        border-color: {t.ACCENT};
    }}

    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}

    QComboBox::down-arrow {{
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 6px solid {t.TEXT_SECONDARY};
        margin-right: 8px;
    }}

    QComboBox QAbstractItemView {{
        background-color: {t.SURFACE};
        border: 1px solid {t.BORDER};
        selection-background-color: {t.SELECTION};
        color: {t.TEXT_PRIMARY};
    }}

    /* Check Boxes */
    QCheckBox {{
        spacing: 8px;
        color: {t.TEXT_PRIMARY};
    }}

    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border: 1px solid {t.BORDER};
        border-radius: 3px;
        background-color: {t.SURFACE};
    }}

    QCheckBox::indicator:hover {{
        border-color: {t.ACCENT};
    }}

    QCheckBox::indicator:checked {{
        background-color: {t.ACCENT};
        border-color: {t.ACCENT};
    }}

    /* Radio Buttons */
    QRadioButton {{
        spacing: 8px;
        color: {t.TEXT_PRIMARY};
    }}

    QRadioButton::indicator {{
        width: 18px;
        height: 18px;
        border: 1px solid {t.BORDER};
        border-radius: 9px;
        background-color: {t.SURFACE};
    }}

    QRadioButton::indicator:hover {{
        border-color: {t.ACCENT};
    }}

    QRadioButton::indicator:checked {{
        background-color: {t.ACCENT};
        border-color: {t.ACCENT};
    }}

    /* Sliders */
    QSlider::groove:horizontal {{
        height: 6px;
        background-color: {t.SURFACE};
        border-radius: 3px;
    }}

    QSlider::handle:horizontal {{
        width: 16px;
        height: 16px;
        margin: -5px 0;
        background-color: {t.ACCENT};
        border-radius: 8px;
    }}

    QSlider::handle:horizontal:hover {{
        background-color: {t.ACCENT_LIGHT};
    }}

    /* Scroll Bars */
    QScrollBar:vertical {{
        background-color: {t.BACKGROUND};
        width: 12px;
        margin: 0;
    }}

    QScrollBar::handle:vertical {{
        background-color: {t.SURFACE};
        border-radius: 4px;
        min-height: 30px;
        margin: 2px;
    }}

    QScrollBar::handle:vertical:hover {{
        background-color: {t.HOVER};
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}

    QScrollBar:horizontal {{
        background-color: {t.BACKGROUND};
        height: 12px;
        margin: 0;
    }}

    QScrollBar::handle:horizontal {{
        background-color: {t.SURFACE};
        border-radius: 4px;
        min-width: 30px;
        margin: 2px;
    }}

    QScrollBar::handle:horizontal:hover {{
        background-color: {t.HOVER};
    }}

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}

    /* Tab Widget */
    QTabWidget::pane {{
        background-color: {t.BACKGROUND_LIGHT};
        border: 1px solid {t.BORDER};
        border-radius: 4px;
        top: -1px;
    }}

    QTabBar::tab {{
        background-color: {t.BACKGROUND_LIGHTER};
        border: 1px solid {t.BORDER};
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        padding: 8px 16px;
        margin-right: 2px;
        color: {t.TEXT_SECONDARY};
    }}

    QTabBar::tab:selected {{
        background-color: {t.BACKGROUND_LIGHT};
        color: {t.TEXT_PRIMARY};
        border-bottom: 2px solid {t.ACCENT};
    }}

    QTabBar::tab:hover:!selected {{
        background-color: {t.HOVER};
        color: {t.TEXT_PRIMARY};
    }}

    /* List Widget */
    QListWidget {{
        background-color: {t.BACKGROUND_LIGHT};
        border: 1px solid {t.BORDER};
        border-radius: 4px;
        padding: 4px;
    }}

    QListWidget::item {{
        padding: 8px;
        border-radius: 4px;
    }}

    QListWidget::item:selected {{
        background-color: {t.SELECTION};
    }}

    QListWidget::item:hover:!selected {{
        background-color: {t.HOVER};
    }}

    /* Tree Widget */
    QTreeWidget {{
        background-color: {t.BACKGROUND_LIGHT};
        border: 1px solid {t.BORDER};
        border-radius: 4px;
    }}

    QTreeWidget::item {{
        padding: 4px;
    }}

    QTreeWidget::item:selected {{
        background-color: {t.SELECTION};
    }}

    QTreeWidget::item:hover:!selected {{
        background-color: {t.HOVER};
    }}

    QHeaderView::section {{
        background-color: {t.BACKGROUND_LIGHTER};
        border: none;
        border-right: 1px solid {t.BORDER};
        border-bottom: 1px solid {t.BORDER};
        padding: 8px;
        color: {t.TEXT_PRIMARY};
    }}

    /* Table Widget */
    QTableWidget {{
        background-color: {t.BACKGROUND_LIGHT};
        border: 1px solid {t.BORDER};
        border-radius: 4px;
        gridline-color: {t.BORDER};
    }}

    QTableWidget::item {{
        padding: 6px;
    }}

    QTableWidget::item:selected {{
        background-color: {t.SELECTION};
    }}

    /* Splitter */
    QSplitter::handle {{
        background-color: {t.BORDER};
    }}

    QSplitter::handle:horizontal {{
        width: 2px;
    }}

    QSplitter::handle:vertical {{
        height: 2px;
    }}

    QSplitter::handle:hover {{
        background-color: {t.ACCENT};
    }}

    /* Menu Bar */
    QMenuBar {{
        background-color: {t.BACKGROUND_LIGHT};
        border-bottom: 1px solid {t.BORDER};
        padding: 2px;
    }}

    QMenuBar::item {{
        padding: 6px 12px;
        border-radius: 4px;
    }}

    QMenuBar::item:selected {{
        background-color: {t.HOVER};
    }}

    /* Menus */
    QMenu {{
        background-color: {t.SURFACE};
        border: 1px solid {t.BORDER};
        border-radius: 4px;
        padding: 4px;
    }}

    QMenu::item {{
        padding: 8px 24px 8px 16px;
        border-radius: 4px;
    }}

    QMenu::item:selected {{
        background-color: {t.SELECTION};
    }}

    QMenu::separator {{
        height: 1px;
        background-color: {t.BORDER};
        margin: 4px 8px;
    }}

    /* Tool Bar */
    QToolBar {{
        background-color: {t.BACKGROUND_LIGHT};
        border: none;
        border-bottom: 1px solid {t.BORDER};
        padding: 4px;
        spacing: 4px;
    }}

    QToolBar::separator {{
        background-color: {t.BORDER};
        width: 1px;
        margin: 4px 8px;
    }}

    /* Status Bar */
    QStatusBar {{
        background-color: {t.BACKGROUND_LIGHT};
        border-top: 1px solid {t.BORDER};
        color: {t.TEXT_SECONDARY};
    }}

    QStatusBar::item {{
        border: none;
    }}

    /* Progress Bar */
    QProgressBar {{
        background-color: {t.SURFACE};
        border: 1px solid {t.BORDER};
        border-radius: 4px;
        text-align: center;
        color: {t.TEXT_PRIMARY};
        height: 20px;
    }}

    QProgressBar::chunk {{
        background-color: {t.ACCENT};
        border-radius: 3px;
    }}

    /* Tooltips */
    QToolTip {{
        background-color: {t.SURFACE};
        border: 1px solid {t.BORDER};
        border-radius: 4px;
        padding: 6px;
        color: {t.TEXT_PRIMARY};
    }}

    /* Dock Widget */
    QDockWidget {{
        titlebar-close-icon: url(close.png);
        titlebar-normal-icon: url(float.png);
    }}

    QDockWidget::title {{
        background-color: {t.BACKGROUND_LIGHTER};
        padding: 8px;
        border-bottom: 1px solid {t.BORDER};
    }}
    """


def apply_theme(app) -> None:
    """Apply dark theme to the application."""
    app.setStyleSheet(get_dark_stylesheet())


def get_plot_style() -> Dict[str, Any]:
    """Get matplotlib style configuration for dark theme."""
    t = DarkTheme

    return {
        # Figure
        'figure.facecolor': t.PLOT_BACKGROUND,
        'figure.edgecolor': t.PLOT_BACKGROUND,

        # Axes
        'axes.facecolor': t.PLOT_AXES,
        'axes.edgecolor': t.BORDER,
        'axes.labelcolor': t.PLOT_TEXT,
        'axes.titlecolor': t.PLOT_TEXT,
        'axes.grid': True,
        'axes.axisbelow': True,

        # Grid
        'grid.color': t.PLOT_GRID,
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.5,

        # Text - disable LaTeX to allow Unicode characters (β, °)
        'text.usetex': False,
        'text.color': t.PLOT_TEXT,
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,

        # Ticks
        'xtick.color': t.PLOT_TEXT,
        'ytick.color': t.PLOT_TEXT,
        'xtick.direction': 'out',
        'ytick.direction': 'out',

        # Legend
        'legend.facecolor': t.SURFACE,
        'legend.edgecolor': t.BORDER,
        'legend.framealpha': 0.9,

        # Lines
        'lines.linewidth': 1.5,
        'lines.markersize': 6,

        # Savefig
        'savefig.facecolor': t.PLOT_BACKGROUND,
        'savefig.edgecolor': t.PLOT_BACKGROUND,
    }


def apply_plot_style() -> None:
    """Apply dark theme to matplotlib."""
    plt.rcParams.update(get_plot_style())


# Alias for main.py
def get_stylesheet() -> str:
    """Get the application stylesheet (alias for get_dark_stylesheet)."""
    return get_dark_stylesheet()


# Additional color constants for specific UI elements
PRIMARY = DarkTheme.ACCENT
SECONDARY = DarkTheme.TEXT_SECONDARY
