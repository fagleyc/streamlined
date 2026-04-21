"""
Data Reduction Module
=====================

Functions for reducing raw wind tunnel data to aerodynamic coefficients.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from .calibration import BalanceCalibration
from .transforms import (
    Geometry, BRFForces, WRFForces,
    calc_brf_forces, calc_wrf_forces,
    subtract_wrf_forces
)
from .coefficients import (
    AeroCoefficients, TunnelConditions,
    calc_tunnel_conditions, calc_aero_coeffs
)


@dataclass
class ReducedDataPoint:
    """Reduced data for a single test point."""
    alpha: np.ndarray = field(default_factory=lambda: np.array([]))
    beta: np.ndarray = field(default_factory=lambda: np.array([]))
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    air_on: Dict[str, Any] = field(default_factory=dict)
    air_off: Dict[str, Any] = field(default_factory=dict)
    aero: Dict[str, Any] = field(default_factory=dict)
    tunnel: TunnelConditions = field(default_factory=TunnelConditions)
    coeffs: AeroCoefficients = field(default_factory=AeroCoefficients)
    brf_on: BRFForces = field(default_factory=BRFForces)
    brf_off: BRFForces = field(default_factory=BRFForces)
    wrf_on: WRFForces = field(default_factory=WRFForces)
    wrf_off: WRFForces = field(default_factory=WRFForces)
    wrf_aero: WRFForces = field(default_factory=WRFForces)


@dataclass
class SteadyStateData:
    """Steady-state (averaged) aerodynamic data."""
    alphas: np.ndarray = field(default_factory=lambda: np.array([]))
    betas: np.ndarray = field(default_factory=lambda: np.array([]))
    Cl: np.ndarray = field(default_factory=lambda: np.array([]))
    Cd: np.ndarray = field(default_factory=lambda: np.array([]))
    Cs: np.ndarray = field(default_factory=lambda: np.array([]))
    CRoll: np.ndarray = field(default_factory=lambda: np.array([]))
    CPitch: np.ndarray = field(default_factory=lambda: np.array([]))
    CYaw: np.ndarray = field(default_factory=lambda: np.array([]))
    # Standard deviations (from time-series within each point)
    Cl_std: np.ndarray = field(default_factory=lambda: np.array([]))
    Cd_std: np.ndarray = field(default_factory=lambda: np.array([]))
    Cs_std: np.ndarray = field(default_factory=lambda: np.array([]))
    CRoll_std: np.ndarray = field(default_factory=lambda: np.array([]))
    CPitch_std: np.ndarray = field(default_factory=lambda: np.array([]))
    CYaw_std: np.ndarray = field(default_factory=lambda: np.array([]))
    pressure_coeffs: Dict[str, np.ndarray] = field(default_factory=dict)
    indices: np.ndarray = field(default_factory=lambda: np.array([]))


def reduce_single_point(raw_on: Dict[str, np.ndarray],
                        raw_off: Dict[str, np.ndarray],
                        cal: BalanceCalibration,
                        geo: Geometry,
                        pressure_cal: Dict[str, Any],
                        facility: str = 'SWT',
                        balance_config: str = 'Force',
                        pdiff_channel: str = '220',
                        p0_channel: str = '690',
                        temp_cal_mode: str = 'auto') -> ReducedDataPoint:
    """
    Reduce a single test point from raw data to coefficients.

    Parameters
    ----------
    raw_on : dict
        Air-on raw data
    raw_off : dict
        Air-off (tare) raw data
    cal : BalanceCalibration
        Balance calibration
    geo : Geometry
        Model geometry
    pressure_cal : dict
        Pressure transducer calibrations
    facility : str
        Facility name
    balance_config : str
        Balance configuration
    pdiff_channel : str
        Differential pressure channel
    p0_channel : str
        Total pressure channel

    Returns
    -------
    ReducedDataPoint
        Reduced data including coefficients
    """
    result = ReducedDataPoint()

    # Store position data from AirON
    result.alpha = raw_on.get('Alpha', np.array([0.0]))
    result.beta = raw_on.get('Beta', np.array([0.0]))
    result.time = raw_on.get('Time', np.array([0.0]))

    # Get position data from AirOFF (may differ from AirON if using a single tare)
    alpha_off = raw_off.get('Alpha', np.array([0.0]))
    beta_off = raw_off.get('Beta', np.array([0.0]))

    # Calculate BRF forces for air-on and air-off
    result.brf_on = calc_brf_forces(raw_on, cal, geo, balance_config)
    result.brf_off = calc_brf_forces(raw_off, cal, geo, balance_config)

    # Calculate WRF forces - CRITICAL: each uses its OWN alpha/beta!
    # This is essential for proper tare subtraction when tare is at different angle
    result.wrf_on = calc_wrf_forces(result.brf_on, result.alpha, result.beta)
    result.wrf_off = calc_wrf_forces(result.brf_off, alpha_off, beta_off)

    # Subtract tare (air-off from air-on) in WRF
    result.wrf_aero = subtract_wrf_forces(result.wrf_on, result.wrf_off)

    # Calculate tunnel conditions (from air-on data)
    result.tunnel = calc_tunnel_conditions(
        raw_on, pressure_cal, facility,
        pdiff_channel, p0_channel, geo.C,
        temp_cal_mode=temp_cal_mode,
    )

    # Extract pressure port data if available
    pressure_data = {}
    for key in raw_on:
        if key.startswith('P') and key[1:].isdigit():
            # This is a pressure port
            on_val = raw_on[key]
            off_val = raw_off.get(key, np.zeros_like(on_val))
            # Subtract only the mean (DC component) of air-off tare
            pressure_data[key] = on_val - np.mean(off_val)

    # Calculate coefficients
    result.coeffs = calc_aero_coeffs(
        result.wrf_aero, result.tunnel.Q,
        geo.C, geo.S, pressure_data
    )

    return result


def reduce_raw(raw_data_list: List[Dict[str, Dict[str, np.ndarray]]],
               cal: BalanceCalibration,
               geo: Geometry,
               pressure_cal: Dict[str, Any],
               facility: str = 'SWT',
               balance_config: str = 'Force',
               pdiff_channel: str = '220',
               p0_channel: str = '690',
               temp_cal_mode: str = 'auto') -> List[ReducedDataPoint]:
    """
    Reduce a list of raw data points.

    Parameters
    ----------
    raw_data_list : list
        List of dictionaries with 'AirOn' and 'AirOff' keys
    cal : BalanceCalibration
        Balance calibration
    geo : Geometry
        Model geometry
    pressure_cal : dict
        Pressure calibrations
    facility : str
        Facility name
    balance_config : str
        Balance configuration
    pdiff_channel : str
        Differential pressure channel
    p0_channel : str
        Total pressure channel

    Returns
    -------
    list
        List of ReducedDataPoint objects
    """
    results = []

    for raw in raw_data_list:
        air_on = raw.get('AirOn', raw.get('AirON', {}))
        air_off = raw.get('AirOff', raw.get('AirOFF', {}))

        if not air_on:
            continue

        # If no air_off, use air_on as tare (will give zero aero)
        if not air_off:
            air_off = air_on

        reduced = reduce_single_point(
            air_on, air_off, cal, geo, pressure_cal,
            facility, balance_config, pdiff_channel, p0_channel,
            temp_cal_mode=temp_cal_mode,
        )
        results.append(reduced)

    return results


def reduce_steady_state(reduced_data: List[ReducedDataPoint]) -> SteadyStateData:
    """
    Reduce time-series data to steady-state (mean) values.

    Parameters
    ----------
    reduced_data : list
        List of ReducedDataPoint objects

    Returns
    -------
    SteadyStateData
        Steady-state averaged data organized by alpha/beta

    Notes
    -----
    Data is sorted and organized by rounded alpha and beta values
    (rounded to nearest 0.5 degrees).
    """
    n_points = len(reduced_data)

    if n_points == 0:
        return SteadyStateData()

    # Extract mean values for each point
    alphas = np.array([np.mean(rd.alpha) for rd in reduced_data])
    betas = np.array([np.mean(rd.beta) for rd in reduced_data])

    Cl = np.array([np.mean(rd.coeffs.Cl) for rd in reduced_data])
    Cd = np.array([np.mean(rd.coeffs.Cd) for rd in reduced_data])
    Cs = np.array([np.mean(rd.coeffs.Cs) for rd in reduced_data])
    CRoll = np.array([np.mean(rd.coeffs.CRoll) for rd in reduced_data])
    CPitch = np.array([np.mean(rd.coeffs.CPitch) for rd in reduced_data])
    CYaw = np.array([np.mean(rd.coeffs.CYaw) for rd in reduced_data])

    # Extract standard deviations for each point
    Cl_std = np.array([np.std(rd.coeffs.Cl) for rd in reduced_data])
    Cd_std = np.array([np.std(rd.coeffs.Cd) for rd in reduced_data])
    Cs_std = np.array([np.std(rd.coeffs.Cs) for rd in reduced_data])
    CRoll_std = np.array([np.std(rd.coeffs.CRoll) for rd in reduced_data])
    CPitch_std = np.array([np.std(rd.coeffs.CPitch) for rd in reduced_data])
    CYaw_std = np.array([np.std(rd.coeffs.CYaw) for rd in reduced_data])

    # Round alpha and beta for sorting
    alpha_int = np.round(alphas * 2) / 2
    beta_int = np.round(betas * 2) / 2

    # Sort by alpha then beta
    ab = np.column_stack([alpha_int, beta_int])
    sort_idx = np.lexsort((ab[:, 1], ab[:, 0]))

    # Get unique combinations
    n_alpha = len(np.unique(alpha_int))
    n_beta = len(np.unique(beta_int))

    # Create output
    ss = SteadyStateData()
    ss.indices = sort_idx

    # Reshape if we have a grid structure
    if n_alpha * n_beta == n_points:
        ss.alphas = alphas[sort_idx].reshape(n_alpha, n_beta)
        ss.betas = betas[sort_idx].reshape(n_alpha, n_beta)
        ss.Cl = Cl[sort_idx].reshape(n_alpha, n_beta)
        ss.Cd = Cd[sort_idx].reshape(n_alpha, n_beta)
        ss.Cs = Cs[sort_idx].reshape(n_alpha, n_beta)
        ss.CRoll = CRoll[sort_idx].reshape(n_alpha, n_beta)
        ss.CPitch = CPitch[sort_idx].reshape(n_alpha, n_beta)
        ss.CYaw = CYaw[sort_idx].reshape(n_alpha, n_beta)
        ss.Cl_std = Cl_std[sort_idx].reshape(n_alpha, n_beta)
        ss.Cd_std = Cd_std[sort_idx].reshape(n_alpha, n_beta)
        ss.Cs_std = Cs_std[sort_idx].reshape(n_alpha, n_beta)
        ss.CRoll_std = CRoll_std[sort_idx].reshape(n_alpha, n_beta)
        ss.CPitch_std = CPitch_std[sort_idx].reshape(n_alpha, n_beta)
        ss.CYaw_std = CYaw_std[sort_idx].reshape(n_alpha, n_beta)
    else:
        # Just return sorted arrays
        ss.alphas = alphas[sort_idx]
        ss.betas = betas[sort_idx]
        ss.Cl = Cl[sort_idx]
        ss.Cd = Cd[sort_idx]
        ss.Cs = Cs[sort_idx]
        ss.CRoll = CRoll[sort_idx]
        ss.CPitch = CPitch[sort_idx]
        ss.CYaw = CYaw[sort_idx]
        ss.Cl_std = Cl_std[sort_idx]
        ss.Cd_std = Cd_std[sort_idx]
        ss.Cs_std = Cs_std[sort_idx]
        ss.CRoll_std = CRoll_std[sort_idx]
        ss.CPitch_std = CPitch_std[sort_idx]
        ss.CYaw_std = CYaw_std[sort_idx]

    # Handle pressure coefficients
    for rd in reduced_data:
        for key, value in rd.coeffs.pressure_coeffs.items():
            if key not in ss.pressure_coeffs:
                ss.pressure_coeffs[key] = []
            ss.pressure_coeffs[key].append(np.mean(value))

    # Convert to arrays and reshape
    for key in ss.pressure_coeffs:
        arr = np.array(ss.pressure_coeffs[key])
        if n_alpha * n_beta == n_points:
            ss.pressure_coeffs[key] = arr[sort_idx].reshape(n_alpha, n_beta)
        else:
            ss.pressure_coeffs[key] = arr[sort_idx]

    return ss


def to_dataframe(ss: SteadyStateData) -> 'pd.DataFrame':
    """
    Convert SteadyStateData to a pandas DataFrame.

    Parameters
    ----------
    ss : SteadyStateData
        Steady-state data

    Returns
    -------
    pd.DataFrame
        DataFrame with coefficient data
    """
    import pandas as pd

    data = {
        'Alpha': ss.alphas.flatten(),
        'Beta': ss.betas.flatten(),
        'Cl': ss.Cl.flatten(),
        'Cd': ss.Cd.flatten(),
        'Cs': ss.Cs.flatten(),
        'CRoll': ss.CRoll.flatten(),
        'CPitch': ss.CPitch.flatten(),
        'CYaw': ss.CYaw.flatten(),
    }

    for key, value in ss.pressure_coeffs.items():
        data[key] = value.flatten()

    return pd.DataFrame(data)


def get_alpha_sweep(ss: SteadyStateData, beta: float = 0.0,
                    tolerance: float = 0.5) -> Dict[str, np.ndarray]:
    """
    Extract an alpha sweep at a specific beta value.

    Parameters
    ----------
    ss : SteadyStateData
        Steady-state data
    beta : float
        Target beta value
    tolerance : float
        Tolerance for beta matching

    Returns
    -------
    dict
        Dictionary with alpha and coefficient arrays
    """
    if ss.alphas.ndim == 2:
        # Grid data - find the beta column
        beta_avg = np.mean(ss.betas, axis=0)
        idx = np.argmin(np.abs(beta_avg - beta))

        return {
            'alpha': ss.alphas[:, idx],
            'Cl': ss.Cl[:, idx],
            'Cd': ss.Cd[:, idx],
            'Cs': ss.Cs[:, idx],
            'CRoll': ss.CRoll[:, idx],
            'CPitch': ss.CPitch[:, idx],
            'CYaw': ss.CYaw[:, idx],
        }
    else:
        # 1D data - filter by beta
        mask = np.abs(ss.betas - beta) < tolerance
        return {
            'alpha': ss.alphas[mask],
            'Cl': ss.Cl[mask],
            'Cd': ss.Cd[mask],
            'Cs': ss.Cs[mask],
            'CRoll': ss.CRoll[mask],
            'CPitch': ss.CPitch[mask],
            'CYaw': ss.CYaw[mask],
        }


def get_beta_sweep(ss: SteadyStateData, alpha: float = 0.0,
                   tolerance: float = 0.5) -> Dict[str, np.ndarray]:
    """
    Extract a beta sweep at a specific alpha value.

    Parameters
    ----------
    ss : SteadyStateData
        Steady-state data
    alpha : float
        Target alpha value
    tolerance : float
        Tolerance for alpha matching

    Returns
    -------
    dict
        Dictionary with beta and coefficient arrays
    """
    if ss.alphas.ndim == 2:
        # Grid data - find the alpha row
        alpha_avg = np.mean(ss.alphas, axis=1)
        idx = np.argmin(np.abs(alpha_avg - alpha))

        return {
            'beta': ss.betas[idx, :],
            'Cl': ss.Cl[idx, :],
            'Cd': ss.Cd[idx, :],
            'Cs': ss.Cs[idx, :],
            'CRoll': ss.CRoll[idx, :],
            'CPitch': ss.CPitch[idx, :],
            'CYaw': ss.CYaw[idx, :],
        }
    else:
        # 1D data - filter by alpha
        mask = np.abs(ss.alphas - alpha) < tolerance
        return {
            'beta': ss.betas[mask],
            'Cl': ss.Cl[mask],
            'Cd': ss.Cd[mask],
            'Cs': ss.Cs[mask],
            'CRoll': ss.CRoll[mask],
            'CPitch': ss.CPitch[mask],
            'CYaw': ss.CYaw[mask],
        }
