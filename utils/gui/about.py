"""
About / version metadata for Streamlined.

Shared metadata template used across the wind-tunnel software ecosystem
(freestream, balance_cal, Streamlined): version, app name, author,
contact, and a compact version history (newest first).
"""

__version__ = "1.3.7"
APP_NAME = "Streamlined"
AUTHOR = "C. Fagley"
CONTACT = "casey.fagley@afacademy.af.edu"

# (version, iso_date, one_line) — newest first.
# Dates for 1.2.4+ from the repository history; earlier dates from the
# archived release zips in Versions/.
VERSION_HISTORY = [
    ("1.3.7", "2026-09-21",
     "Mach filter no longer discards a single-speed dataset: the "
     "whole-case test keyed on the measured mean while the filter "
     "offers the commanded setpoint"),
    ("1.3.6", "2026-09-14",
     "Figure export: legend placement control (four corners), and the "
     "trace style, legend and grid are remembered between exports "
     "instead of resetting to defaults every time"),
    ("1.3.5", "2026-09-14",
     "Attitude offsets no longer rotate an EXTERNAL balance's loads: the "
     "balance is mount-fixed, so feeding the offset into its wind-axis "
     "resolution swung lift into drag and cut L/D by ~2.6x"),
    ("1.3.4", "2026-09-14",
     "Fix re-reducing a loaded case (adding an alpha offset after "
     "loading raised AttributeError), and fix raster figure export "
     "measuring hidden items and writing a huge mostly-empty image"),
    ("1.3.3", "2026-09-14",
     "Mach steps key and label on the COMMANDED Mach like alpha and "
     "beta do, so runs held a thousandth apart no longer split into "
     "extra filter entries; legacy M0p25 filename tokens are read as "
     "a Mach setpoint"),
    ("1.3.2", "2026-09-12",
     "Alpha, beta and Mach groups come from the COMMANDED value the "
     "run recorded, not the measured reading, so positioner jitter no "
     "longer splits one sweep angle into extra traces, filter entries "
     "and export columns"),
    ("1.3.1", "2026-09-09",
     "Alpha/beta attitude offsets in the model geometry (bent-sting "
     "rectification): added to every point's recorded attitude, "
     "air-on and air-off, before reduction"),
    ("1.3.0", "2026-09-01",
     "X Axis dropdown on the plot panel: plot against Mach, Re, q, U_inf, "
     "any coefficient or any calculator variable, replacing the 'Plot vs beta' "
     "checkbox; a speed variable on x draws one curve per angle"),
    ("1.2.9", "2026-08-28",
     "Speed sweeps group by their commanded setpoint, so each Mach "
     "step plots and exports as one curve; a run folder's processed "
     "output is no longer indexed as run data; unsteady MAT/HDF5 "
     "export fixed for external-balance runs"),
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
