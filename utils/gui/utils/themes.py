"""
Theme and Styling
=================

Live-switchable application themes (USAFA Night / USAFA Day / Classic
Dark / Classic Light / High Contrast) for the Qt UI and matplotlib plots.

The engine lives in :mod:`themekit` (shared verbatim with Freestream).
This module keeps the historical API: ``DarkTheme`` still exposes
``BACKGROUND``, ``TEXT_SECONDARY``, ``PLOT_BACKGROUND``… — but the values
are the ACTIVE theme's, rewritten in place on every switch, so read them
when you build or paint, never copy them into module-level constants.
"""

from typing import Dict, Any

from . import themekit
from .themekit import (AppearanceDialog, BrandBadge, CommandPalette,  # noqa: F401
                       PALETTES, ThemeToggleButton, app_icon, current,
                       install_view_menu, logo_pixmap, make_splash_pixmap,
                       manager, themed_svg_icon)


class DarkTheme:
    """Active theme palette (name kept for compatibility — it is no longer
    always dark). Attributes are rewritten by the theme manager."""

    # Base colors
    BACKGROUND = ""
    BACKGROUND_LIGHT = ""
    BACKGROUND_LIGHTER = ""
    SURFACE = ""

    # Text colors
    TEXT_PRIMARY = ""
    TEXT_SECONDARY = ""
    TEXT_DISABLED = ""
    ON_ACCENT = ""

    # Accent colors
    ACCENT = ""
    ACCENT_LIGHT = ""
    ACCENT_DARK = ""
    INDICATOR = ""

    # Status colors
    SUCCESS = ""
    WARNING = ""
    ERROR = ""
    INFO = ""

    # Border colors
    BORDER = ""
    BORDER_LIGHT = ""

    # Selection
    SELECTION = ""
    HOVER = ""

    # Header (menu bar / status bar)
    HEADER_BG = ""
    HEADER_TEXT = ""

    # Plot colors
    PLOT_BACKGROUND = ""
    PLOT_AXES = ""
    PLOT_GRID = ""
    PLOT_TEXT = ""


#: Active theme alias with a neutral name for new code.
Theme = DarkTheme

themekit.manager().register_sink(DarkTheme, aliases={
    "BACKGROUND": "BG",
    "BACKGROUND_LIGHT": "BG_LIGHT",
    "BACKGROUND_LIGHTER": "BG_LIGHTER",
    "TEXT_PRIMARY": "TEXT",
    "TEXT_SECONDARY": "TEXT_DIM",
    "PLOT_BACKGROUND": "PLOT_FIG",
    "PLOT_AXES": "PLOT_BG",
    "PLOT_GRID": "AXIS",
    "PLOT_TEXT": "TEXT",
})


def _streamlined_extra_css(p, scale, density) -> str:
    """Streamlined-only rules layered on the shared sheet."""
    c = p.tokens
    return f"""
    /* Streamlined panels are QFrames: give them the card surface. Labels
       (also QFrames) are reset right after so they stay borderless. */
    QFrame {{
        background-color: {c['BG_LIGHT']};
        border: 1px solid {c['BORDER']};
        border-radius: 6px;
    }}
    QLabel {{ background-color: transparent; border: none; }}
    QStatusBar QLabel, QToolBar QLabel {{ border: none; }}
    QToolBar#mainToolbar {{ background-color: {c['HEADER_BG']};
        border-bottom: 2px solid {c['INDICATOR'] if p.branded else c['BORDER']};
        padding: 4px 8px; }}
    QToolBar#mainToolbar QToolButton {{ color: {c['HEADER_TEXT']}; }}
    QToolBar#mainToolbar QToolButton:hover {{
        background-color: {themekit.mix(c['HEADER_BG'], c['HEADER_TEXT'], 0.16)};
        border-color: transparent; }}
    QToolBar#mainToolbar QLabel#brandTitle {{ color: {c['HEADER_TEXT']}; }}
    QToolBar#mainToolbar QLabel#brandSub {{ color: {c['HEADER_DIM']}; }}
    QToolBar#mainToolbar::separator {{
        background-color: {themekit.mix(c['HEADER_BG'], c['HEADER_TEXT'], 0.25)}; }}
    """


themekit.manager().add_stylesheet_extra(_streamlined_extra_css)


def get_dark_stylesheet() -> str:
    """The ACTIVE theme's Qt stylesheet (name kept for compatibility)."""
    return themekit.manager().stylesheet()


def apply_theme(app=None) -> None:
    """Apply the active theme to the application."""
    themekit.manager().ensure_applied()


def get_plot_style() -> Dict[str, Any]:
    """Matplotlib rcParams for the active theme."""
    style = dict(themekit.mpl_rc(themekit.current()))
    style.update({
        # Text - disable LaTeX to allow Unicode characters (β, °)
        'text.usetex': False,
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.framealpha': 0.9,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
    })
    return style


def apply_plot_style() -> None:
    """Apply the active theme to matplotlib."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(get_plot_style())


# Alias for main.py
def get_stylesheet() -> str:
    """Get the application stylesheet (alias for get_dark_stylesheet)."""
    return get_dark_stylesheet()


def __getattr__(name):
    # PRIMARY / SECONDARY used to be import-time constants; resolve live
    if name == "PRIMARY":
        return DarkTheme.ACCENT
    if name == "SECONDARY":
        return DarkTheme.TEXT_SECONDARY
    raise AttributeError(name)
