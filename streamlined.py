#!/usr/bin/env python
"""
Wind Tunnel Data Reduction GUI
==============================

Launch script for the wind tunnel data reduction GUI application.

Usage:
    python streamlined.py

Requirements:
    - Python 3.8+
    - PyQt6
    - NumPy
    - Matplotlib
    - pandas (for export functionality)

The application provides:
    - Load and visualize wind tunnel test data (TDMS format)
    - Apply force balance and pressure calibrations
    - Compute aerodynamic coefficients (CL, CD, Cm, etc.)
    - Interactive plotting with filtering by beta/Mach/Reynolds
    - Export data to CSV/Excel
"""

import sys
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Check dependencies
def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    try:
        import PyQt6
    except ImportError:
        missing.append("PyQt6")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")

    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)


def main():
    """Launch the GUI application."""
    check_dependencies()

    # Disable LaTeX rendering BEFORE any matplotlib imports
    # This prevents Unicode character errors (β, °) in plot legends
    import matplotlib
    matplotlib.use('QtAgg')  # Set backend early
    matplotlib.rcParams['text.usetex'] = False
    matplotlib.rcParams['text.latex.preamble'] = ''

    # Also clear any cached tex settings
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = False

    from utils.gui.main import main as gui_main
    gui_main()


if __name__ == "__main__":
    main()
