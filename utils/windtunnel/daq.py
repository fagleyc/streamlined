"""
DAQ Class - Main Data Acquisition and Processing Class
======================================================

This class provides the main interface for wind tunnel data processing,
mirroring the functionality of the MATLAB DAQ class.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import warnings

from .calibration import (
    BalanceCalibration, read_vol_file, read_pcf_file, calc_coeffs
)
from .data_io import (
    read_tdms_file, read_run_file, find_data_files,
    classify_files_by_condition, copy_balance_markers,
    extract_alpha_beta_from_filename, extract_sort_key_from_filename,
    export_to_csv
)
from .transforms import Geometry, is_external_balance_data
from .reduction import (
    ReducedDataPoint, SteadyStateData,
    reduce_single_point, reduce_raw, reduce_steady_state,
    to_dataframe, get_alpha_sweep
)
from .plotting import (
    setup_plot_style, is_latex_available, plot_coefficients, plot_drag_polar,
    plot_pitching_moment, plot_lift_drag_ratio, save_all_figures
)
from .units import UnitSystem, UnitConverter, UnitLabels, UNIT_LABELS


@dataclass
class FacilitySettings:
    """Wind tunnel facility settings."""
    name: str = 'SWT'
    balance_type: str = 'Internal'
    balance_config: str = 'Force'
    pdiff: str = '220'
    p0_stil: str = '690'
    t0_stil: str = ''
    # Thermocouple calibration mode: 'auto' | 'degC' | 'degF'
    # 'auto' - per-sample detection (< 4 V = degC, else degF)
    # 'degC' - new thermocouple cal (0.1 V/degC)
    # 'degF' - old thermocouple cal (0.1 V/degF)
    temp_cal_mode: str = 'auto'


@dataclass
class FilterSettings:
    """Data filter settings."""
    filter_type: str = 'low'
    cutoff_freq: float = 10.0  # Hz (Wn = cutoff / (0.5 * fs))
    order: int = 4
    transient_time: float = 0.0


class DAQ:
    """
    Data Acquisition and Processing Class for Wind Tunnel Testing.

    This class provides methods to:
    - Load and apply force balance calibrations
    - Load and apply pressure transducer calibrations
    - Set model geometry parameters
    - Read and process TDMS data files
    - Reduce data to aerodynamic coefficients
    - Generate publication-quality plots

    Attributes
    ----------
    cal : BalanceCalibration
        Force balance calibration data
    pressure_cal : dict
        Pressure transducer calibration data
    geo : list
        List of Geometry objects
    fac : FacilitySettings
        Facility configuration
    filt : FilterSettings
        Filter configuration
    raw : list
        Raw data from TDMS files
    red : list
        Reduced data (forces, moments, coefficients)
    ss : SteadyStateData
        Steady-state averaged data

    Examples
    --------
    >>> # Basic usage
    >>> daq = DAQ()
    >>> daq.set_swt_defaults()
    >>> daq.cal_balance('CalFiles/balance.vol', 'Linear')
    >>> daq.cal_instruments('CalFiles/pressure.PCF')
    >>> daq.set_geometry(MAC=2.86, S=18.75, MRC=[1.6, 0, 0], units='IPS')
    >>> daq.load_data_directory('Run01')
    >>> daq.reduce_datasets()
    >>> daq.reduce_steady_state()
    >>> daq.plot_ss_coeffs('alpha')
    """

    def __init__(self):
        """Initialize DAQ object with empty data structures."""
        self.cal: Optional[BalanceCalibration] = None
        self.pressure_cal: Dict[str, Any] = {}
        self.geo: List[Geometry] = []
        self.fac = FacilitySettings()
        self.filt = FilterSettings()
        self.raw: List[Dict[str, Any]] = []
        self.red: List[ReducedDataPoint] = []
        self.ss: Optional[SteadyStateData] = None
        self._use_latex = False

        # Output unit system (default IPS for backward compatibility)
        self._output_units: str = 'IPS'
        self._converter: Optional[UnitConverter] = None

    def set_facility(self, facility: str, balance: str,
                     config: str = 'Force') -> 'DAQ':
        """
        Set facility properties.

        Parameters
        ----------
        facility : str
            Facility name: 'Subsonic', 'Lowspeed', or 'Trisonic'
        balance : str
            Balance type: 'Internal' or 'External'
        config : str
            Balance configuration: 'Force' or 'Moment'

        Returns
        -------
        DAQ
            Self for method chaining
        """
        facility_map = {
            'Subsonic': 'SWT',
            'Lowspeed': 'LSWT',
            'Trisonic': 'TST'
        }
        self.fac.name = facility_map.get(facility, facility)
        self.fac.balance_type = balance
        self.fac.balance_config = config
        return self

    def set_instruments(self, pdiff: str, p0_stil: str,
                        t0_stil: str = '') -> 'DAQ':
        """
        Set instrument channel names.

        Parameters
        ----------
        pdiff : str
            Differential pressure channel
        p0_stil : str
            Total pressure channel
        t0_stil : str
            Total temperature channel

        Returns
        -------
        DAQ
            Self for method chaining
        """
        self.fac.pdiff = pdiff
        self.fac.p0_stil = p0_stil
        self.fac.t0_stil = t0_stil
        return self

    def set_swt_defaults(self) -> 'DAQ':
        """
        Set default parameters for Subsonic Wind Tunnel (SWT).

        Returns
        -------
        DAQ
            Self for method chaining
        """
        self.fac = FacilitySettings(
            name='SWT',
            balance_type='Internal',
            balance_config='Force',
            pdiff='220',
            p0_stil='690',
            t0_stil=''
        )

        # Set up matplotlib style (auto-detects LaTeX availability)
        self._use_latex = is_latex_available()
        setup_plot_style(use_latex=self._use_latex)

        return self

    def set_lswt_defaults(self) -> 'DAQ':
        """Set default parameters for Low Speed Wind Tunnel."""
        self.fac = FacilitySettings(
            name='LSWT',
            balance_type='Internal',
            balance_config='Force',
            pdiff='220',
            p0_stil='',
            t0_stil=''
        )
        return self

    def set_filter(self, filter_type: str, cutoff: float,
                   order: int, transient: float = 0.0) -> 'DAQ':
        """
        Set filter parameters.

        Parameters
        ----------
        filter_type : str
            Filter type: 'low', 'high', 'bandpass'
        cutoff : float
            Cutoff frequency (normalized to Nyquist)
        order : int
            Filter order
        transient : float
            Transient time to remove from start of data

        Returns
        -------
        DAQ
            Self for method chaining
        """
        self.filt = FilterSettings(
            filter_type=filter_type,
            cutoff_freq=cutoff,
            order=order,
            transient_time=transient
        )
        return self

    def set_geometry(self, MAC: float, S: float, MRC: List[float],
                     units: str = 'IPS', span: float = 1.0,
                     **kwargs) -> 'DAQ':
        """
        Set model geometry parameters.

        Parameters
        ----------
        MAC : float
            Mean Aerodynamic Chord (reference length for CPitch and Re)
        S : float
            Reference area
        MRC : list
            Moment Reference Center shift [dx, dy, dz]
        units : str
            Input units: 'IPS', 'FPS', 'MKS', 'CGS'
        span : float
            Reference span (for CRoll and CYaw normalization)
        **kwargs
            Additional geometry parameters

        Returns
        -------
        DAQ
            Self for method chaining

        Notes
        -----
        All internal calculations use IPS (inch-pound-second) units.
        """
        # Unit conversion factors to IPS (multiply input by these to get inches/sq in)
        conversions = {
            'IPS': (1.0, 1.0),          # already in inches, sq inches
            'FPS': (12.0, 144.0),       # feet to inches (1 ft = 12 in, 1 sq ft = 144 sq in)
            'MKS': (39.3701, 1550.0031),  # meters to inches (1 m = 39.37 in, 1 mÂ² = 1550 sq in)
            'CGS': (0.3937, 0.155),     # cm to inches (1 cm = 0.3937 in, 1 cmÂ² = 0.155 sq in)
        }

        cL, cS = conversions.get(units, (1.0, 1.0))

        geo = Geometry(
            C=MAC * cL,
            S=S * cS,
            b=span * cL,
            mshift=np.array(MRC) * cL
        )

        # Add any extra parameters
        for key, value in kwargs.items():
            setattr(geo, key, value)

        self.geo.append(geo)
        return self

    def set_output_units(self, units: str) -> 'DAQ':
        """
        Set the output unit system for display and export.

        All internal calculations remain in IPS. This setting only affects
        how values are displayed and exported.

        Parameters
        ----------
        units : str
            Output unit system: 'IPS', 'FPS', 'MKS', or 'CGS'

        Returns
        -------
        DAQ
            Self for method chaining

        Notes
        -----
        - IPS: inch-pound-second (lbf, lb-in, psi, ft/s)
        - FPS: foot-pound-second (lbf, lb-ft, psf, ft/s)
        - MKS: SI units (N, N-m, Pa, m/s)
        - CGS: centimeter-gram-second (dyn, dyn-cm, Pa, cm/s)
        """
        units_upper = units.upper()
        if units_upper not in ['IPS', 'FPS', 'MKS', 'CGS']:
            warnings.warn(f"Unknown unit system '{units}', using IPS")
            units_upper = 'IPS'

        self._output_units = units_upper
        self._converter = UnitConverter(UnitSystem[units_upper])
        return self

    @property
    def output_units(self) -> str:
        """Get the current output unit system."""
        return self._output_units

    @property
    def unit_labels(self) -> UnitLabels:
        """Get unit labels for the current output system."""
        if self._converter is None:
            return UNIT_LABELS[UnitSystem.IPS]
        return self._converter.get_labels()

    @property
    def converter(self) -> UnitConverter:
        """Get the unit converter for the current output system."""
        if self._converter is None:
            self._converter = UnitConverter(UnitSystem.IPS)
        return self._converter

    def cal_balance(self, filepath: str, cal_type: str = 'Linear') -> 'DAQ':
        """
        Load and process force balance calibration.

        Parameters
        ----------
        filepath : str
            Path to .vol calibration file
        cal_type : str
            Calibration type: 'Linear', 'Quadratic', or 'Cubic'

        Returns
        -------
        DAQ
            Self for method chaining
        """
        self.cal = read_vol_file(filepath)

        if cal_type not in ['Linear', 'Quadratic', 'Cubic']:
            warnings.warn(f"Unknown calibration type '{cal_type}', using Linear")
            cal_type = 'Linear'

        self.cal = calc_coeffs(self.cal, cal_type)

        return self

    def cal_instruments(self, filepath: str) -> 'DAQ':
        """
        Load pressure transducer calibrations.

        Parameters
        ----------
        filepath : str
            Path to .PCF calibration file

        Returns
        -------
        DAQ
            Self for method chaining
        """
        self.pressure_cal = read_pcf_file(filepath)
        return self

    def load_data_directory(self, directory: str,
                            pattern: str = '*.tdms') -> 'DAQ':
        """
        Load all data files from a directory.

        Files are dispatched on extension (TDMS, HDF5 .h5/.hdf5, or
        MATLAB .mat) via :func:`read_run_file`; the default pattern
        preserves the historical TDMS behavior.

        Parameters
        ----------
        directory : str
            Directory containing data files
        pattern : str
            Glob pattern for file matching (e.g. '*.tdms', '*.h5')

        Returns
        -------
        DAQ
            Self for method chaining
        """
        files = find_data_files(directory, pattern)
        classified = classify_files_by_condition(files)

        # Organize files alpha -> beta -> speed (speed is a first-class
        # sweep dimension alongside alpha/beta)
        air_on_sorted = sorted(classified['AirOn'],
                               key=lambda f: extract_sort_key_from_filename(str(f)))
        air_off_sorted = sorted(classified['AirOff'],
                                key=lambda f: extract_sort_key_from_filename(str(f)))

        # Pair air-on with air-off files
        for i, on_file in enumerate(air_on_sorted):
            raw_entry = {'AirOn': {}, 'AirOff': {}}

            raw_on, _ = read_run_file(str(on_file))
            raw_entry['AirOn'] = raw_on.data
            raw_entry['AirOn']['Time'] = raw_on.time
            copy_balance_markers(raw_on, raw_entry['AirOn'])

            # Find matching air-off file
            alpha_on, beta_on = extract_alpha_beta_from_filename(str(on_file))

            if i < len(air_off_sorted):
                off_file = air_off_sorted[i]
                alpha_off, beta_off = extract_alpha_beta_from_filename(str(off_file))

                # Check if alpha/beta match
                if np.isclose(alpha_on, alpha_off, atol=0.5) and np.isclose(beta_on, beta_off, atol=0.5):
                    raw_off, _ = read_run_file(str(off_file))
                else:
                    # Use first AirOff file as tare
                    raw_off, _ = read_run_file(str(air_off_sorted[0]))
                raw_entry['AirOff'] = raw_off.data
                raw_entry['AirOff']['Time'] = raw_off.time
                copy_balance_markers(raw_off, raw_entry['AirOff'])

            self.raw.append(raw_entry)

        return self

    def config_selection_load(self, directory: str) -> Tuple['DAQ', List[str]]:
        """
        Load data with automatic configuration detection.

        Parameters
        ----------
        directory : str
            Directory containing TDMS files

        Returns
        -------
        tuple
            (self, list of configuration names)
        """
        self.load_data_directory(directory)
        configs = [f"Config_{i}" for i in range(len(self.raw))]
        return self, configs

    def reduce_datasets(self) -> 'DAQ':
        """
        Reduce all raw data to forces and coefficients.

        Returns
        -------
        DAQ
            Self for method chaining
        """
        if not self.raw:
            raise ValueError("No raw data loaded. Call load_data_directory first.")

        # External (ATE) balance data carries resolved loads and needs
        # no .vol calibration; only internal (bridge-volt) data does.
        needs_cal = any(
            not is_external_balance_data(entry.get('AirOn') or {})
            for entry in self.raw
        )
        if needs_cal and not self.cal:
            raise ValueError("No calibration loaded. Call cal_balance first.")

        if not self.geo:
            raise ValueError("No geometry set. Call set_geometry first.")

        self.red = reduce_raw(
            self.raw,
            self.cal,
            self.geo[0],  # Use first geometry
            self.pressure_cal,
            facility=self.fac.name,
            balance_config=self.fac.balance_config,
            pdiff_channel=self.fac.pdiff,
            p0_channel=self.fac.p0_stil,
            temp_cal_mode=getattr(self.fac, 'temp_cal_mode', 'auto'),
        )
        return self

    def reduce_2d_datasets(self) -> 'DAQ':
        """
        Reduce 2D array of raw data (multiple configurations).

        Returns
        -------
        DAQ
            Self for method chaining
        """
        return self.reduce_datasets()

    def reduce_steady_state(self) -> 'DAQ':
        """
        Reduce time-series data to steady-state values.

        Returns
        -------
        DAQ
            Self for method chaining
        """
        if not self.red:
            raise ValueError("No reduced data. Call reduce_datasets first.")

        self.ss = reduce_steady_state(self.red)
        return self

    def plot_ss_coeffs(self, x_var: str = 'alpha',
                       save_dir: Optional[str] = None) -> Dict:
        """
        Plot steady-state coefficients.

        Parameters
        ----------
        x_var : str
            Independent variable: 'alpha' or 'beta'
        save_dir : str, optional
            Directory to save figures

        Returns
        -------
        dict
            Dictionary of Figure objects
        """
        if self.ss is None:
            raise ValueError("No steady-state data. Call reduce_steady_state first.")

        return plot_coefficients(self.ss, x_var, use_latex=self._use_latex,
                                 save_dir=save_dir)

    def plot_drag_polar(self) -> 'Figure':
        """Plot drag polar (Cd vs Cl)."""
        if self.ss is None:
            raise ValueError("No steady-state data. Call reduce_steady_state first.")
        return plot_drag_polar(self.ss, use_latex=self._use_latex)

    def plot_pitching_moment(self, x_var: str = 'Cl') -> 'Figure':
        """Plot pitching moment coefficient."""
        if self.ss is None:
            raise ValueError("No steady-state data. Call reduce_steady_state first.")
        return plot_pitching_moment(self.ss, x_var, use_latex=self._use_latex)

    def plot_lift_drag_ratio(self) -> 'Figure':
        """Plot lift-to-drag ratio."""
        if self.ss is None:
            raise ValueError("No steady-state data. Call reduce_steady_state first.")
        return plot_lift_drag_ratio(self.ss, use_latex=self._use_latex)

    def export_steady_state(self, filepath: str) -> None:
        """
        Export steady-state data to CSV.

        Values are converted to the current output unit system.
        Column headers include unit labels.

        Parameters
        ----------
        filepath : str
            Output file path
        """
        if self.ss is None:
            raise ValueError("No steady-state data. Call reduce_steady_state first.")

        df = to_dataframe(self.ss)

        # Get unit labels
        labels = self.unit_labels
        conv = self.converter

        # Rename columns to include units for dimensional quantities
        column_renames = {
            'Q': f'Q [{labels.pressure}]',
            'U_inf': f'U_inf [{labels.velocity}]',
            'rho': f'rho [{labels.density}]',
            'T': f'T [{labels.temperature}]',
        }

        # Convert dimensional values if not IPS
        if self._output_units != 'IPS':
            # Convert tunnel conditions columns if present
            if 'Q' in df.columns:
                df['Q'] = conv.convert_pressure(df['Q'])
            if 'U_inf' in df.columns:
                df['U_inf'] = conv.convert_velocity(df['U_inf'])
            if 'rho' in df.columns:
                df['rho'] = conv.convert_density(df['rho'])
            if 'T' in df.columns:
                df['T'] = conv.convert_temperature(df['T'])

        # Rename columns to include units
        df = df.rename(columns=column_renames)

        df.to_csv(filepath, index=False)

    def save_all_figures(self, directory: str,
                         formats: List[str] = ['png', 'pdf']) -> None:
        """
        Save all current figures to a directory.

        Parameters
        ----------
        directory : str
            Output directory
        formats : list
            File formats to save
        """
        import matplotlib.pyplot as plt

        figures = {}
        for num in plt.get_fignums():
            fig = plt.figure(num)
            name = fig.get_label() or f'figure_{num}'
            figures[name] = fig

        save_all_figures(figures, directory, formats)

    def clear_data(self) -> 'DAQ':
        """Clear all loaded data."""
        self.raw = []
        self.red = []
        self.ss = None
        return self

    def __repr__(self) -> str:
        """String representation of DAQ object."""
        status = []
        status.append(f"DAQ Object")
        status.append(f"  Facility: {self.fac.name}")
        status.append(f"  Balance: {self.fac.balance_type} ({self.fac.balance_config})")
        status.append(f"  Calibration: {'Loaded' if self.cal else 'Not loaded'}")
        status.append(f"  Geometry: {len(self.geo)} configuration(s)")
        status.append(f"  Output Units: {self._output_units}")
        status.append(f"  Raw data: {len(self.raw)} files")
        status.append(f"  Reduced data: {len(self.red)} points")
        status.append(f"  Steady-state: {'Computed' if self.ss else 'Not computed'}")
        return '\n'.join(status)
