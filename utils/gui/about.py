"""
About / version metadata for Streamlined.

Shared metadata template used across the wind-tunnel software ecosystem
(freestream, balance_cal, Streamlined): version, app name, author,
contact, and a compact version history (newest first).
"""

__version__ = "1.2.8"
APP_NAME = "Streamlined"
AUTHOR = "C. Fagley"
CONTACT = "casey.fagley@afacademy.af.edu"

# (version, iso_date, one_line) — newest first.
# Dates for 1.2.4+ from the repository history; earlier dates from the
# archived release zips in Versions/.
VERSION_HISTORY = [
    ("1.2.8", "2026-07-23",
     "External (ATE) balance reduction, freestream .h5/.mat run-file "
     "reading, TDMS shift tool, Help/Documentation system + About fix"),
    ("1.2.7", "2026-06-04",
     "Modular custom data calculator with template expansion; "
     "categorized MAT/HDF5 exports"),
    ("1.2.6", "2026-05-15",
     "Stability-derivative plots, selectable blockage corrections, "
     "COE export + standalone post-processor"),
    ("1.2.5", "2026-04-29",
     "Thermocouple cal auto-detect, raw export group, user-friendly "
     "error dialogs"),
    ("1.2.4", "2026-04-21",
     "Compressible isentropic tunnel conditions, Sutherland viscosity, "
     "moment-balance support"),
    ("1.2.3", "2026-03-16",
     "Multiple named geometries with per-case assignment; span-based "
     "Cl/Cn normalization"),
    ("1.2.2", "2026-03-16",
     "Interactive save-image dialog, std-dev shading, consolidated "
     "export dialog"),
    ("1.2.0", "2026-03-10",
     "Unsteady time-series HDF5/MAT export, per-case groups, "
     "multi-case time-history overlays"),
    ("1.1.0", "2026-03-05",
     "Plot-vs-beta toggle, MATLAB struct export, Excel export"),
    ("1.0.0", "2026-02",
     "Initial release: TDMS loading, balance/pressure calibration, "
     "BRF/WRF reduction, plotting, CSV/Excel/HDF5/MAT export"),
]

SUMMARY = (
    "Streamlined is a wind tunnel data reduction application that replaces "
    "the legacy MATLAB workflow: it reads raw force-balance runs (TDMS and "
    "freestream .h5/.mat run files), applies multi-order balance and "
    "pressure calibrations, transforms loads through body and wind "
    "reference frames with proper tare subtraction, computes compressible "
    "tunnel conditions and aerodynamic coefficients, and exports reduced "
    "data to CSV, Excel, HDF5, MAT, and COE formats."
)
