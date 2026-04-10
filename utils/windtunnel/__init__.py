"""
Streamlined - Wind Tunnel Data Reduction Package
=================================================

A Python package for processing wind tunnel force balance measurements,
applying calibrations, and computing aerodynamic coefficients.

Author: C. Fagley

Basic Usage
-----------
>>> from windtunnel import DAQ
>>>
>>> # Create and configure DAQ object
>>> daq = DAQ()
>>> daq.set_swt_defaults()
>>>
>>> # Load calibrations
>>> daq.cal_balance('CalFiles/balance.vol', 'Linear')
>>> daq.cal_instruments('CalFiles/pressure.PCF')
>>>
>>> # Set geometry
>>> daq.set_geometry(MAC=2.86, S=18.75, MRC=[1.6, 0, 0], units='IPS')
>>>
>>> # Load and process data
>>> daq.load_data_directory('Run01')
>>> daq.reduce_datasets()
>>> daq.reduce_steady_state()
>>>
>>> # Generate plots
>>> daq.plot_ss_coeffs('alpha')
>>> plt.show()
"""

from .daq import DAQ
from .calibration import (
    read_vol_file, read_pcf_file, calc_coeffs,
    BalanceCalibration, PressureCalibration
)
from .data_io import (
    read_tdms_file, find_data_files, export_to_csv, export_to_excel
)
from .reduction import (
    reduce_raw, reduce_steady_state, to_dataframe,
    ReducedDataPoint, SteadyStateData,
    get_alpha_sweep, get_beta_sweep
)
from .coefficients import (
    calc_aero_coeffs, calc_tunnel_conditions,
    calc_lift_curve_slope, calc_zero_lift_alpha,
    calc_drag_polar_coeffs, calc_oswald_efficiency,
    AeroCoefficients, TunnelConditions
)
from .transforms import (
    calc_brf_forces, calc_wrf_forces,
    subtract_tare, Geometry,
    BRFForces, WRFForces
)
from .plotting import (
    plot_coefficients, plot_drag_polar, plot_pitching_moment,
    plot_lift_drag_ratio, plot_lateral_directional, plot_surface,
    setup_plot_style, is_latex_available, save_all_figures
)
from .utils import (
    str_subtract, bin_average, moving_average,
    apply_butterworth_filter, compute_derivative
)
from .uncertainty import (
    calc_coefficient_uncertainty, calc_precision_uncertainty,
    calc_reynolds_uncertainty, uncertainty_summary,
    CoefficientUncertainty, UncertaintyComponents
)
from .units import (
    UnitSystem, UnitLabels, UnitConverter, UNIT_LABELS, CONVERSION_FACTORS
)

__version__ = '1.0.0'
__author__ = 'C. Fagley'
__all__ = [
    # Main class
    'DAQ',

    # Calibration
    'read_vol_file',
    'read_pcf_file',
    'calc_coeffs',
    'BalanceCalibration',
    'PressureCalibration',

    # Data I/O
    'read_tdms_file',
    'find_data_files',
    'export_to_csv',
    'export_to_excel',

    # Reduction
    'reduce_raw',
    'reduce_steady_state',
    'to_dataframe',
    'ReducedDataPoint',
    'SteadyStateData',
    'get_alpha_sweep',
    'get_beta_sweep',

    # Coefficients
    'calc_aero_coeffs',
    'calc_tunnel_conditions',
    'calc_lift_curve_slope',
    'calc_zero_lift_alpha',
    'calc_drag_polar_coeffs',
    'calc_oswald_efficiency',
    'AeroCoefficients',
    'TunnelConditions',

    # Transforms
    'calc_brf_forces',
    'calc_wrf_forces',
    'subtract_tare',
    'Geometry',
    'BRFForces',
    'WRFForces',

    # Plotting
    'plot_coefficients',
    'plot_drag_polar',
    'plot_pitching_moment',
    'plot_lift_drag_ratio',
    'plot_lateral_directional',
    'plot_surface',
    'setup_plot_style',
    'is_latex_available',
    'save_all_figures',

    # Utilities
    'str_subtract',
    'bin_average',
    'moving_average',
    'apply_butterworth_filter',
    'compute_derivative',

    # Uncertainty
    'calc_coefficient_uncertainty',
    'calc_precision_uncertainty',
    'calc_reynolds_uncertainty',
    'uncertainty_summary',
    'CoefficientUncertainty',
    'UncertaintyComponents',

    # Units
    'UnitSystem',
    'UnitLabels',
    'UnitConverter',
    'UNIT_LABELS',
    'CONVERSION_FACTORS',
]
