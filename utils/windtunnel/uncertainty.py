"""
Uncertainty Analysis Module
===========================

Functions for computing measurement uncertainties in wind tunnel testing.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

from .reduction import ReducedDataPoint, SteadyStateData
from .coefficients import AeroCoefficients


@dataclass
class UncertaintyComponents:
    """Uncertainty components for a measurement."""
    precision: float = 0.0      # Random/precision error (2-sigma)
    bias: float = 0.0           # Systematic/bias error
    total: float = 0.0          # Total uncertainty (RSS)


@dataclass
class CoefficientUncertainty:
    """Uncertainty values for aerodynamic coefficients."""
    Cl: UncertaintyComponents = field(default_factory=UncertaintyComponents)
    Cd: UncertaintyComponents = field(default_factory=UncertaintyComponents)
    Cs: UncertaintyComponents = field(default_factory=UncertaintyComponents)
    CRoll: UncertaintyComponents = field(default_factory=UncertaintyComponents)
    CPitch: UncertaintyComponents = field(default_factory=UncertaintyComponents)
    CYaw: UncertaintyComponents = field(default_factory=UncertaintyComponents)


def calc_precision_uncertainty(data: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate precision (random) uncertainty from repeated measurements.

    Parameters
    ----------
    data : np.ndarray
        Repeated measurement data
    confidence : float
        Confidence level (default 0.95 for 95%)

    Returns
    -------
    float
        Precision uncertainty (2-sigma for 95% confidence)

    Notes
    -----
    For 95% confidence with large sample size, uses 2 * std / sqrt(n)
    """
    n = len(data)
    if n < 2:
        return 0.0

    std = np.std(data, ddof=1)

    # Student's t multiplier for 95% confidence
    if n > 30:
        t_mult = 2.0
    else:
        from scipy.stats import t
        t_mult = t.ppf((1 + confidence) / 2, n - 1)

    return t_mult * std / np.sqrt(n)


def calc_balance_uncertainty(cal_bias: np.ndarray,
                             forces: np.ndarray,
                             balance_type: str = 'Internal') -> Dict[str, float]:
    """
    Calculate force balance measurement uncertainty.

    Parameters
    ----------
    cal_bias : np.ndarray
        Calibration bias (RMSE) for each channel
    forces : np.ndarray
        Measured forces
    balance_type : str
        'Internal' or 'External'

    Returns
    -------
    dict
        Uncertainty for each force/moment component
    """
    if balance_type == 'Internal':
        channel_names = ['N1', 'N2', 'Y1', 'Y2', 'Axial', 'Roll']
    else:
        channel_names = ['Drag', 'Side', 'Lift', 'Roll', 'Pitch', 'Yaw']

    uncertainties = {}
    for i, name in enumerate(channel_names):
        if i < len(cal_bias):
            uncertainties[name] = cal_bias[i]

    return uncertainties


def calc_tunnel_uncertainty(Q: float, S: float, C: float,
                            dQ: float = 0.01, dS: float = 0.001,
                            dC: float = 0.001) -> Dict[str, float]:
    """
    Calculate uncertainty contributions from tunnel conditions.

    Parameters
    ----------
    Q : float
        Dynamic pressure
    S : float
        Reference area
    C : float
        Reference chord
    dQ : float
        Fractional uncertainty in Q
    dS : float
        Fractional uncertainty in S
    dC : float
        Fractional uncertainty in C

    Returns
    -------
    dict
        Uncertainty contributions
    """
    return {
        'Q': dQ * Q,
        'S': dS * S,
        'C': dC * C
    }


def propagate_coefficient_uncertainty(force_unc: float,
                                      Q: float, S: float, C: float,
                                      dQ_frac: float = 0.01,
                                      dS_frac: float = 0.001) -> float:
    """
    Propagate uncertainty through coefficient calculation.

    For Cx = F / (Q * S):
        dCx/Cx = sqrt((dF/F)^2 + (dQ/Q)^2 + (dS/S)^2)

    Parameters
    ----------
    force_unc : float
        Force uncertainty
    Q : float
        Dynamic pressure
    S : float
        Reference area
    C : float
        Reference chord (for moments)
    dQ_frac : float
        Fractional uncertainty in Q
    dS_frac : float
        Fractional uncertainty in S

    Returns
    -------
    float
        Coefficient uncertainty
    """
    # Force coefficient uncertainty
    denom = Q * S

    # Relative uncertainty
    rel_unc_sq = (force_unc / denom) ** 2 + dQ_frac ** 2 + dS_frac ** 2

    return np.sqrt(rel_unc_sq)


def calc_coefficient_uncertainty(reduced_data: ReducedDataPoint,
                                 cal_bias: np.ndarray,
                                 dQ_frac: float = 0.01,
                                 dS_frac: float = 0.001) -> CoefficientUncertainty:
    """
    Calculate complete uncertainty for aerodynamic coefficients.

    Parameters
    ----------
    reduced_data : ReducedDataPoint
        Reduced data point
    cal_bias : np.ndarray
        Calibration bias for each balance channel
    dQ_frac : float
        Fractional uncertainty in dynamic pressure
    dS_frac : float
        Fractional uncertainty in reference area

    Returns
    -------
    CoefficientUncertainty
        Uncertainty for all coefficients
    """
    unc = CoefficientUncertainty()

    Q = np.mean(reduced_data.tunnel.Q)
    S = 1.0  # Need to pass geometry
    C = 1.0

    # Precision uncertainty from time series
    coeffs = reduced_data.coeffs

    for coeff_name in ['Cl', 'Cd', 'Cs', 'CRoll', 'CPitch', 'CYaw']:
        coeff_data = getattr(coeffs, coeff_name)
        comp = UncertaintyComponents()

        # Precision (random) uncertainty
        comp.precision = calc_precision_uncertainty(coeff_data)

        # Bias uncertainty from calibration
        # This is simplified - full calculation requires error propagation
        avg_bias = np.mean(cal_bias) if len(cal_bias) > 0 else 0.0
        comp.bias = avg_bias / (Q * S) if Q * S > 0 else 0.0

        # Total (RSS)
        comp.total = np.sqrt(comp.precision ** 2 + comp.bias ** 2)

        setattr(unc, coeff_name, comp)

    return unc


def calc_reynolds_uncertainty(rho: float, U: float, L: float, mu: float,
                              drho_frac: float = 0.01,
                              dU_frac: float = 0.01,
                              dL_frac: float = 0.001) -> Tuple[float, float]:
    """
    Calculate Reynolds number and its uncertainty.

    Re = rho * U * L / mu

    Parameters
    ----------
    rho : float
        Air density
    U : float
        Freestream velocity
    L : float
        Reference length
    mu : float
        Dynamic viscosity
    drho_frac : float
        Fractional uncertainty in density
    dU_frac : float
        Fractional uncertainty in velocity
    dL_frac : float
        Fractional uncertainty in length

    Returns
    -------
    tuple
        (Reynolds number, uncertainty)
    """
    Re = rho * U * L / mu

    # Relative uncertainty
    dRe_frac = np.sqrt(drho_frac ** 2 + dU_frac ** 2 + dL_frac ** 2)

    return Re, Re * dRe_frac


def uncertainty_summary(unc: CoefficientUncertainty) -> str:
    """
    Generate a text summary of coefficient uncertainties.

    Parameters
    ----------
    unc : CoefficientUncertainty
        Coefficient uncertainties

    Returns
    -------
    str
        Formatted summary
    """
    lines = []
    lines.append("=" * 70)
    lines.append("COEFFICIENT UNCERTAINTY SUMMARY")
    lines.append("=" * 70)
    lines.append(f"{'Coeff':>10} {'Precision':>12} {'Bias':>12} {'Total':>12}")
    lines.append("-" * 70)

    for name in ['Cl', 'Cd', 'Cs', 'CRoll', 'CPitch', 'CYaw']:
        comp = getattr(unc, name)
        lines.append(f"{name:>10} {comp.precision:12.6f} {comp.bias:12.6f} {comp.total:12.6f}")

    lines.append("=" * 70)

    return '\n'.join(lines)


def write_uncertainty_table(ss: SteadyStateData,
                            unc: CoefficientUncertainty,
                            filepath: str,
                            alpha: Optional[float] = None,
                            beta: Optional[float] = None) -> None:
    """
    Write uncertainty table to file.

    Parameters
    ----------
    ss : SteadyStateData
        Steady-state data
    unc : CoefficientUncertainty
        Uncertainties
    filepath : str
        Output file path
    alpha : float, optional
        Filter to specific alpha
    beta : float, optional
        Filter to specific beta
    """
    with open(filepath, 'w') as f:
        f.write("Wind Tunnel Data Uncertainty Analysis\n")
        f.write("=" * 70 + "\n\n")

        if alpha is not None:
            f.write(f"Alpha = {alpha:.1f} deg\n")
        if beta is not None:
            f.write(f"Beta = {beta:.1f} deg\n")
        f.write("\n")

        f.write(f"{'Coefficient':>12} {'Value':>12} {'Uncertainty':>12} {'%':>8}\n")
        f.write("-" * 50 + "\n")

        # Get mean values
        coeffs = ['Cl', 'Cd', 'Cs', 'CRoll', 'CPitch', 'CYaw']
        for name in coeffs:
            data = getattr(ss, name)
            unc_val = getattr(unc, name).total
            mean_val = np.mean(data)
            pct = 100 * unc_val / abs(mean_val) if mean_val != 0 else 0

            f.write(f"{name:>12} {mean_val:12.5f} {unc_val:12.5f} {pct:8.2f}\n")

        f.write("\n" + "=" * 70 + "\n")
