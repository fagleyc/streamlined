"""Custom widgets for the Wind Tunnel GUI."""

from .plot_canvas import PlotCanvas
from .case_list import CaseListWidget, CaseListItem
from .filter_widgets import MultiSelectFilter, FilterToolbar, PlotTypeSelector

# Try to import fast plot canvas (requires pyqtgraph)
try:
    from .fast_plot_canvas import FastPlotCanvas, is_available as fast_plot_available
except ImportError:
    FastPlotCanvas = None
    fast_plot_available = lambda: False

__all__ = [
    'PlotCanvas',
    'FastPlotCanvas',
    'CaseListWidget',
    'CaseListItem',
    'MultiSelectFilter',
    'FilterToolbar',
    'PlotTypeSelector',
]
