"""
Calibration File Readers and Processing
=======================================

Functions to read force balance calibration files (.vol) and
pressure transducer calibration files (.PCF), and compute
calibration coefficients.
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class BalanceDescription:
    """Force balance description metadata."""
    balance_type: str = ""
    serial_number: str = ""
    outer_diameter: str = ""


@dataclass
class MaxLoads:
    """Maximum balance loads and their units."""
    values: Dict[str, float] = field(default_factory=dict)
    units: Dict[str, str] = field(default_factory=dict)


@dataclass
class Distances:
    """Distance measurements from balance calibration."""
    values: Dict[str, float] = field(default_factory=dict)


@dataclass
class BalanceCalibration:
    """Complete balance calibration data structure."""
    info: Dict[str, str] = field(default_factory=dict)
    description: BalanceDescription = field(default_factory=BalanceDescription)
    max_loads: MaxLoads = field(default_factory=MaxLoads)
    distances: Distances = field(default_factory=Distances)
    force_channels: List[str] = field(default_factory=list)
    row_indices: List[int] = field(default_factory=list)
    force: np.ndarray = field(default_factory=lambda: np.array([]))
    volts: np.ndarray = field(default_factory=lambda: np.array([]))
    coeffs: np.ndarray = field(default_factory=lambda: np.array([]))
    force_est: np.ndarray = field(default_factory=lambda: np.array([]))
    cal_type: str = "Linear"
    r_squared: np.ndarray = field(default_factory=lambda: np.array([]))
    bias: np.ndarray = field(default_factory=lambda: np.array([]))
    file: str = ""


@dataclass
class PressureCalibration:
    """Pressure transducer calibration data."""
    name: str = ""
    cal_date: str = ""
    slope: float = 0.0
    slope_noise: float = 0.0
    exc_volt: str = ""
    units: str = ""


def _parse_line(line: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse a calibration file line with '-->' separator.

    Parameters
    ----------
    line : str
        Line to parse

    Returns
    -------
    tuple
        (key, value) or (None, None) if parsing fails
    """
    line = line.strip()
    parts = line.split('-->')
    if len(parts) > 1:
        key = re.sub(r'\s+', '', parts[0])
        value = parts[1].strip()
        return key, value
    return None, None


def read_vol_file(filepath: str) -> BalanceCalibration:
    """
    Read a force balance voltage calibration file (.vol format).

    Parameters
    ----------
    filepath : str
        Path to the .vol calibration file

    Returns
    -------
    BalanceCalibration
        Calibration data structure containing all parsed data

    Notes
    -----
    The .vol file format contains:
    - Balance description metadata
    - Maximum balance loads
    - Distance measurements
    - Force/voltage calibration curves for each channel (positive and negative)
    """
    cal = BalanceCalibration()
    cal.file = filepath

    channel_data = []  # List of (multiplier, channel, nloads, data)

    with open(filepath, 'r') as f:
        current_field = 'Info'
        in_header = True

        for line in f:
            line = line.strip()
            if not line:
                continue

            # Check for section header
            if line.startswith('['):
                field_name = re.sub(r'[\[\]\s]', '', line)
                current_field = field_name
                in_header = False

                # Handle force/voltage sections
                if 'pos' in field_name.lower() or 'neg' in field_name.lower():
                    channel = field_name[:-3]  # Remove 'pos' or 'neg'
                    if 'pos' in field_name.lower():
                        multiplier = 1
                    else:
                        multiplier = -1

                    # Read number of loads
                    next_line = next(f).strip()
                    _, nloads_str = _parse_line(next_line)
                    nloads = int(nloads_str)

                    # Skip header line
                    next(f)

                    # Read data lines
                    data = []
                    for _ in range(nloads):
                        data_line = next(f).strip()
                        values = [float(v) for v in data_line.split(',')]
                        data.append(values)

                    channel_data.append({
                        'multiplier': multiplier,
                        'channel': channel,
                        'nloads': nloads,
                        'data': np.array(data)
                    })
                continue

            # Parse content based on section
            if in_header:
                key, value = _parse_line(line)
                if key:
                    cal.info[key] = value

            elif current_field == 'BalanceDescription':
                key, value = _parse_line(line)
                if key:
                    key_lower = key.lower()
                    if 'type' in key_lower:
                        cal.description.balance_type = value
                    elif 'serial' in key_lower:
                        cal.description.serial_number = value
                    elif 'diameter' in key_lower:
                        cal.description.outer_diameter = value

            elif current_field == 'MaximalBalanceLoads':
                key, value = _parse_line(line)
                if key:
                    # Clean up key
                    clean_key = re.sub(r'[()]', '', key)
                    parts = value.split()
                    cal.max_loads.values[clean_key] = float(parts[0])
                    if len(parts) > 1:
                        cal.max_loads.units[clean_key] = parts[1]

            elif current_field == 'Distances':
                key, value = _parse_line(line)
                if key:
                    cal.distances.values[key] = float(value)

    # Process channel data into Force and Volts matrices
    if channel_data:
        total_loads = sum(cd['nloads'] for cd in channel_data)
        n_channels = 6

        Force = np.zeros((total_loads, n_channels))
        Volts = np.zeros((total_loads, n_channels))

        row_idx = 0
        current_channel = channel_data[0]['channel']
        col = 0
        cal.force_channels.append(current_channel)
        cal.row_indices.append(row_idx)

        for cd in channel_data:
            if cd['channel'] != current_channel:
                col += 1
                current_channel = cd['channel']
                cal.force_channels.append(current_channel)
                cal.row_indices.append(row_idx)

            data = cd['data']
            nloads = cd['nloads']
            multiplier = cd['multiplier']

            # Force values (first column, with sign)
            Force[row_idx:row_idx + nloads, col] = np.abs(data[:, 0]) * multiplier

            # Find zero-load indices for offset correction
            zero_indices = np.where(data[:, 0] == 0)[0]
            if len(zero_indices) > 0:
                zero_offset = np.mean(data[zero_indices, 1:7], axis=0)
            else:
                zero_offset = np.zeros(6)

            # Voltage values (columns 1-6, normalized by excitation voltage column 7)
            Volts[row_idx:row_idx + nloads, :] = (data[:, 1:7] - zero_offset) / data[:, 7:8]

            row_idx += nloads

        cal.force = Force
        cal.volts = Volts

    return cal


def read_pcf_file(filepath: str) -> Dict[str, PressureCalibration]:
    """
    Read a pressure calibration file (.PCF format).

    Parameters
    ----------
    filepath : str
        Path to the .PCF calibration file

    Returns
    -------
    dict
        Dictionary mapping transducer names to PressureCalibration objects

    Notes
    -----
    The .PCF file format contains pressure transducer calibration data
    including slope, date, and units for multiple transducers.
    """
    calibrations = {}

    with open(filepath, 'r') as f:
        current_section = None

        for line in f:
            line = line.strip()
            if not line:
                continue

            # Check for section header
            if line.startswith('['):
                section_name = re.sub(r'[\[\]\s]', '', line)
                # Convert to valid Python identifier
                field_name = 'P' + section_name.replace('-', '_')
                current_section = section_name

                # Read the next few lines for this transducer
                try:
                    comment_line = next(f).strip()

                    # Skip if this is a "load" entry (not a pressure cal)
                    if 'load' in comment_line.lower():
                        current_section = None
                        continue

                    cal = PressureCalibration()
                    cal.name = section_name

                    # Parse calibration date
                    date_line = next(f).strip()
                    _, cal.cal_date = _parse_line(date_line)

                    # Parse slope (positive)
                    slope_line = next(f).strip()
                    _, slope_str = _parse_line(slope_line)
                    cal.slope = float(slope_str)

                    # Parse slope (negative) - stored as slope_noise
                    slope_n_line = next(f).strip()
                    _, slope_n_str = _parse_line(slope_n_line)
                    cal.slope_noise = float(slope_n_str)

                    # Parse excitation voltage
                    exc_line = next(f).strip()
                    _, cal.exc_volt = _parse_line(exc_line)

                    # Parse units
                    units_line = next(f).strip()
                    _, cal.units = _parse_line(units_line)

                    calibrations[field_name] = cal

                except (StopIteration, ValueError):
                    continue

    return calibrations


def form_higher_order_terms(v: np.ndarray, order: int) -> np.ndarray:
    """
    Form polynomial terms for calibration fitting.

    Parameters
    ----------
    v : np.ndarray
        Input voltage matrix (n_samples x n_channels)
    order : int
        Polynomial order (1=linear, 2=quadratic, 3=cubic)

    Returns
    -------
    np.ndarray
        Matrix with polynomial terms
    """
    if order == 1:
        return v
    elif order == 2:
        return np.hstack([v, v * v])
    elif order == 3:
        return np.hstack([v, v * v, v * v * v])
    else:
        raise ValueError(f"Order {order} not supported. Use 1, 2, or 3.")


def calc_coeffs(cal: BalanceCalibration, cal_type: str = 'Linear') -> BalanceCalibration:
    """
    Calculate calibration coefficients using least-squares fitting.

    Parameters
    ----------
    cal : BalanceCalibration
        Calibration data with Force and Volts matrices populated
    cal_type : str
        Type of fit: 'Linear', 'Quadratic', or 'Cubic'

    Returns
    -------
    BalanceCalibration
        Updated calibration with Coeffs, Force_est, R_squared, and Bias

    Notes
    -----
    Uses least-squares regression to fit the relationship:
        Force = Volts_poly @ Coeffs
    where Volts_poly contains the polynomial terms of the voltage data.
    """
    if cal.force.size == 0 or cal.volts.size == 0:
        raise ValueError("Force and Volts data must be populated before calculating coefficients")

    # Form polynomial terms
    order_map = {'Linear': 1, 'Quadratic': 2, 'Cubic': 3}
    order = order_map.get(cal_type, 1)

    volts_poly = form_higher_order_terms(cal.volts, order)

    # Solve least-squares: Volts_poly @ Coeffs = Force
    coeffs, residuals, rank, s = np.linalg.lstsq(volts_poly, cal.force, rcond=None)

    # Estimate forces
    force_est = volts_poly @ coeffs

    # Calculate R-squared for each channel
    ss_res = np.sum((cal.force - force_est) ** 2, axis=0)
    ss_tot = np.sum((cal.force - np.mean(cal.force, axis=0)) ** 2, axis=0)
    r_squared = 1 - ss_res / ss_tot

    # Calculate bias (RMSE)
    bias = np.sqrt(np.sum((cal.force - force_est) ** 2, axis=0) / len(cal.force))

    # Update calibration object
    cal.coeffs = coeffs
    cal.force_est = force_est
    cal.cal_type = cal_type
    cal.r_squared = r_squared
    cal.bias = bias

    return cal
