"""
Utility Functions
=================

Helper functions for wind tunnel data processing.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import re


def find_closest_in_time(on_files: List[Path], off_files: List[Path],
                         configs: List[str]) -> List[str]:
    """
    Find the closest air-off files in time for each air-on file.

    Parameters
    ----------
    on_files : list
        List of air-on file paths
    off_files : list
        List of air-off file paths
    configs : list
        Configuration names

    Returns
    -------
    list
        Matched configuration names for air-off files
    """
    # Get modification times
    on_times = {f: f.stat().st_mtime for f in on_files}

    matched_configs = []
    for on_file in on_files:
        on_time = on_times[on_file]

        # Find closest air-off file
        min_diff = float('inf')
        closest_config = configs[0] if configs else None

        for off_file in off_files:
            off_time = off_file.stat().st_mtime
            diff = abs(on_time - off_time)
            if diff < min_diff:
                min_diff = diff
                # Extract config from filename
                for cfg in configs:
                    if cfg.lower() in str(off_file).lower():
                        closest_config = cfg
                        break

        matched_configs.append(closest_config)

    return matched_configs


def extract_unique_configurations(files: List[Path],
                                  ignore_parts: List[str]) -> List[str]:
    """
    Extract unique configuration names from file list.

    Parameters
    ----------
    files : list
        List of file paths
    ignore_parts : list
        Parts of filename to ignore when determining configuration

    Returns
    -------
    list
        Unique configuration names
    """
    configs = set()

    for f in files:
        name = f.stem

        # Remove ignored parts
        for part in ignore_parts:
            name = re.sub(part, '', name, flags=re.IGNORECASE)

        # Remove alpha/beta values
        name = re.sub(r'[-_]?\d+\.?\d*', '', name)

        # Clean up
        name = re.sub(r'[-_]+', '_', name).strip('_')

        if name:
            configs.add(name)

    return sorted(list(configs))


def append_struct_fields(base: Dict[str, Any],
                         new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Append fields from one dictionary to another.

    Parameters
    ----------
    base : dict
        Base dictionary
    new : dict
        Dictionary with new fields to add

    Returns
    -------
    dict
        Merged dictionary
    """
    result = base.copy()
    result.update(new)
    return result


def concatenate_structs(structs: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    Concatenate multiple data dictionaries along axis 0.

    Parameters
    ----------
    structs : list
        List of dictionaries with numpy arrays

    Returns
    -------
    dict
        Concatenated dictionary
    """
    if not structs:
        return {}

    result = {}
    keys = structs[0].keys()

    for key in keys:
        arrays = [s[key] for s in structs if key in s]
        if arrays:
            result[key] = np.concatenate(arrays, axis=0)

    return result


def str_subtract(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively subtract corresponding values in two dictionaries.

    Parameters
    ----------
    d1 : dict
        First dictionary (minuend)
    d2 : dict
        Second dictionary (subtrahend)

    Returns
    -------
    dict
        Result of subtraction

    Notes
    -----
    - For nested dictionaries, recursively subtract
    - For arrays, subtract element-wise (handling length mismatches)
    - 'Time' fields are preserved from d1 without subtraction
    """
    result = {}

    for key in d1:
        if isinstance(d1[key], dict):
            if key in d2 and isinstance(d2[key], dict):
                result[key] = str_subtract(d1[key], d2[key])
            else:
                result[key] = d1[key]
        elif 'time' in key.lower():
            result[key] = d1[key]
        elif key in d2:
            val1 = np.asarray(d1[key])
            val2 = np.asarray(d2[key])
            min_len = min(len(val1), len(val2))
            result[key] = val1[:min_len] - val2[:min_len]
        else:
            result[key] = d1[key]

    return result


def apply_butterworth_filter(data: np.ndarray, cutoff: float,
                             fs: float, order: int = 4,
                             filter_type: str = 'low') -> np.ndarray:
    """
    Apply Butterworth filter to data.

    Parameters
    ----------
    data : np.ndarray
        Input data
    cutoff : float
        Cutoff frequency in Hz
    fs : float
        Sampling frequency in Hz
    order : int
        Filter order
    filter_type : str
        Filter type: 'low', 'high', 'band'

    Returns
    -------
    np.ndarray
        Filtered data
    """
    from scipy.signal import butter, filtfilt

    nyq = 0.5 * fs
    normalized_cutoff = cutoff / nyq

    b, a = butter(order, normalized_cutoff, btype=filter_type)
    return filtfilt(b, a, data)


def compute_derivative(data: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute time derivative using central differences.

    Parameters
    ----------
    data : np.ndarray
        Input data
    dt : float
        Time step

    Returns
    -------
    np.ndarray
        Derivative
    """
    deriv = np.zeros_like(data)
    deriv[1:-1] = (data[2:] - data[:-2]) / (2 * dt)
    deriv[0] = (data[1] - data[0]) / dt
    deriv[-1] = (data[-1] - data[-2]) / dt
    return deriv


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """
    Apply moving average filter.

    Parameters
    ----------
    data : np.ndarray
        Input data
    window : int
        Window size

    Returns
    -------
    np.ndarray
        Smoothed data
    """
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='same')


def interpolate_to_common_time(data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
                               target_dt: Optional[float] = None) -> Dict[str, np.ndarray]:
    """
    Interpolate multiple time series to a common time base.

    Parameters
    ----------
    data_dict : dict
        Dictionary mapping names to (time, data) tuples
    target_dt : float, optional
        Target time step. If None, uses minimum dt from data.

    Returns
    -------
    dict
        Dictionary with interpolated data including 'Time' key
    """
    from scipy.interpolate import interp1d

    # Find time range and minimum dt
    t_min = max(t[0] for t, d in data_dict.values())
    t_max = min(t[-1] for t, d in data_dict.values())

    if target_dt is None:
        target_dt = min(np.mean(np.diff(t)) for t, d in data_dict.values())

    common_time = np.arange(t_min, t_max, target_dt)

    result = {'Time': common_time}

    for name, (time, data) in data_dict.items():
        interp_func = interp1d(time, data, kind='cubic',
                               bounds_error=False, fill_value='extrapolate')
        result[name] = interp_func(common_time)

    return result


def bin_average(x: np.ndarray, y: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute bin-averaged values.

    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y : np.ndarray
        Dependent variable
    n_bins : int
        Number of bins

    Returns
    -------
    tuple
        (bin_centers, bin_averages)
    """
    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    bin_indices = np.digitize(x, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_averages = np.array([y[bin_indices == i].mean()
                             if np.sum(bin_indices == i) > 0 else np.nan
                             for i in range(n_bins)])

    return bin_centers, bin_averages


def compute_uncertainty_propagation(values: Dict[str, float],
                                    uncertainties: Dict[str, float],
                                    formula: str) -> Tuple[float, float]:
    """
    Compute uncertainty propagation for a formula.

    Parameters
    ----------
    values : dict
        Variable values
    uncertainties : dict
        Variable uncertainties
    formula : str
        Mathematical formula as string

    Returns
    -------
    tuple
        (computed_value, uncertainty)

    Notes
    -----
    Uses numerical differentiation for partial derivatives.
    """
    import sympy as sp

    # Create symbols
    symbols = {name: sp.Symbol(name) for name in values}

    # Parse formula
    expr = sp.sympify(formula, locals=symbols)

    # Compute value
    computed_value = float(expr.subs(values))

    # Compute uncertainty via error propagation
    variance = 0
    for name, sym in symbols.items():
        if name in uncertainties:
            partial = sp.diff(expr, sym)
            partial_val = float(partial.subs(values))
            variance += (partial_val * uncertainties[name]) ** 2

    uncertainty = np.sqrt(variance)

    return computed_value, uncertainty


def format_coefficient_table(ss: 'SteadyStateData',
                             alpha: Optional[float] = None,
                             beta: Optional[float] = None) -> str:
    """
    Format steady-state data as a text table.

    Parameters
    ----------
    ss : SteadyStateData
        Steady-state data
    alpha : float, optional
        Filter to specific alpha
    beta : float, optional
        Filter to specific beta

    Returns
    -------
    str
        Formatted table string
    """
    from .reduction import SteadyStateData

    lines = []
    lines.append("=" * 80)
    lines.append("Aerodynamic Coefficients")
    lines.append("=" * 80)
    lines.append(f"{'Alpha':>8} {'Beta':>8} {'Cl':>10} {'Cd':>10} {'Cm':>10} {'L/D':>10}")
    lines.append("-" * 80)

    alphas = ss.alphas.flatten()
    betas = ss.betas.flatten()
    Cl = ss.Cl.flatten()
    Cd = ss.Cd.flatten()
    Cm = ss.CPitch.flatten()

    for i in range(len(alphas)):
        if alpha is not None and not np.isclose(alphas[i], alpha, atol=0.5):
            continue
        if beta is not None and not np.isclose(betas[i], beta, atol=0.5):
            continue

        ld = Cl[i] / Cd[i] if Cd[i] != 0 else np.inf
        lines.append(f"{alphas[i]:8.2f} {betas[i]:8.2f} {Cl[i]:10.4f} {Cd[i]:10.4f} "
                     f"{Cm[i]:10.4f} {ld:10.2f}")

    lines.append("=" * 80)

    return '\n'.join(lines)
