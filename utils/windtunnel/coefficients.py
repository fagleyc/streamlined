"""
Aerodynamic Coefficient Calculations
====================================

Functions for computing aerodynamic coefficients from forces and moments.
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from .transforms import WRFForces


@dataclass
class AeroCoefficients:
    """Aerodynamic coefficients."""
    Cl: np.ndarray = field(default_factory=lambda: np.array([]))   # Lift coefficient
    Cd: np.ndarray = field(default_factory=lambda: np.array([]))   # Drag coefficient
    Cs: np.ndarray = field(default_factory=lambda: np.array([]))   # Side force coefficient
    CRoll: np.ndarray = field(default_factory=lambda: np.array([]))  # Roll moment coefficient
    CPitch: np.ndarray = field(default_factory=lambda: np.array([]))  # Pitch moment coefficient
    CYaw: np.ndarray = field(default_factory=lambda: np.array([]))  # Yaw moment coefficient
    pressure_coeffs: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class TunnelConditions:
    """Wind tunnel flow conditions."""
    Q: np.ndarray = field(default_factory=lambda: np.array([]))       # Dynamic pressure (psi)
    Q_mks: np.ndarray = field(default_factory=lambda: np.array([]))   # Dynamic pressure (Pa)
    rho: np.ndarray = field(default_factory=lambda: np.array([]))     # Static air density (kg/m^3)
    P_tot: np.ndarray = field(default_factory=lambda: np.array([]))   # Total (stagnation) pressure (Pa)
    P_static: np.ndarray = field(default_factory=lambda: np.array([]))  # Static pressure (Pa)
    T: np.ndarray = field(default_factory=lambda: np.array([]))       # Static temperature (C)
    T0: np.ndarray = field(default_factory=lambda: np.array([]))      # Total (stagnation) temperature (C)
    U_inf: np.ndarray = field(default_factory=lambda: np.array([]))   # Freestream velocity (m/s)
    Re: np.ndarray = field(default_factory=lambda: np.array([]))      # Reynolds number
    Mach: np.ndarray = field(default_factory=lambda: np.array([]))    # Mach number
    a: np.ndarray = field(default_factory=lambda: np.array([]))       # Speed of sound (m/s)


# Physical constants
TORR_TO_PA = 133.32
PSI_TO_PA = 6894.75729
C_TO_K = 273.15
R_AIR = 287.058  # J/(kg*K) - specific gas constant for air
KGM3_TO_LB_IN3 = 3.61273e-5
GAMMA = 1.4  # ratio of specific heats

# Sutherland's law constants for dynamic viscosity of air
MU_REF = 1.716e-5   # Pa*s, reference dynamic viscosity
T_REF = 273.15       # K, reference temperature
S_SUTH = 110.4       # K, Sutherland constant for air


def _convert_thermocouple_to_celsius(raw_temp: np.ndarray,
                                     temp_cal_mode: str = 'auto') -> np.ndarray:
    """
    Convert raw thermocouple voltage to Celsius.

    Two calibrations have been used with this facility's thermocouple:
      - Old: 0.1 V/degF  (raw * 10 gives degF, then convert to degC)
      - New: 0.1 V/degC  (raw * 10 gives degC directly)

    Parameters
    ----------
    raw_temp : np.ndarray
        Raw thermocouple voltage time-series.
    temp_cal_mode : str
        One of:
          'auto'    - auto-detect per sample (< 4 V assume degC, else degF).
                      A voltage near 4 V splits room-temperature degF (~7 V
                      at 70 degF) from room-temperature degC (~2 V at 20 degC).
          'degC'    - force new calibration: raw * 10 is already degC.
          'degF'    - force old calibration: raw * 10 is degF, convert to degC.

    Returns
    -------
    np.ndarray
        Temperature in Celsius.
    """
    mode = (temp_cal_mode or 'auto').lower()
    scaled = raw_temp * 10.0  # 0.1 V/deg -> deg (either scale)

    if mode == 'degc':
        return scaled
    if mode == 'degf':
        return (scaled - 32.0) * 5.0 / 9.0

    # 'auto' - per-sample decision based on raw voltage magnitude
    # At typical room temperature: degC slope gives ~2 V, degF slope ~7 V.
    # Threshold of 4 V cleanly separates the two regimes.
    raw_arr = np.asarray(raw_temp, dtype=float)
    is_degf = raw_arr >= 4.0
    out_degc = scaled.copy() if isinstance(scaled, np.ndarray) else np.array(scaled)
    if np.any(is_degf):
        out_degc = np.where(is_degf, (scaled - 32.0) * 5.0 / 9.0, scaled)
    return out_degc


def calc_tunnel_conditions(raw_data: Dict[str, np.ndarray],
                           pressure_cal: Dict[str, Any],
                           facility: str = 'SWT',
                           pdiff_channel: str = '220',
                           p0_channel: str = '690',
                           ref_length_in: float = 1.0,
                           temp_cal_mode: str = 'auto') -> TunnelConditions:
    """
    Calculate wind tunnel flow conditions.

    Parameters
    ----------
    raw_data : dict
        Raw data containing Pdiff, Ptot, Temp channels
    pressure_cal : dict
        Pressure transducer calibration data
    facility : str
        Facility name: 'SWT', 'LSWT', or 'TST'
    pdiff_channel : str
        Name of differential pressure calibration channel
    p0_channel : str
        Name of total pressure calibration channel
    ref_length_in : float
        Reference length in inches for Reynolds number
    temp_cal_mode : str
        Thermocouple calibration mode: 'auto', 'degC', or 'degF'.
        'auto' detects per-sample from voltage magnitude.

    Returns
    -------
    TunnelConditions
        Calculated tunnel conditions
    """
    cond = TunnelConditions()

    # Ensure channel names have 'P' prefix
    if not pdiff_channel.startswith('P'):
        pdiff_channel = 'P' + pdiff_channel
    if not p0_channel.startswith('P'):
        p0_channel = 'P' + p0_channel

    if facility == 'SWT':
        # Get calibration slopes
        pdiff_slope = pressure_cal[pdiff_channel].slope
        p0_slope = pressure_cal[p0_channel].slope

        gm1 = GAMMA - 1.0  # gamma - 1 = 0.4
        g_ratio = gm1 / GAMMA  # (gamma-1)/gamma = 2/7

        # ---------------------------------------------------------------
        # Measured quantities (converted to SI)
        # ---------------------------------------------------------------

        # Differential pressure: dP = P0 - P_static (psi, then Pa)
        dP_psi = raw_data['Pdiff'] * pdiff_slope
        dP_Pa = dP_psi * PSI_TO_PA

        # Total (stagnation) pressure P0 (Pa)
        P0_Pa = raw_data['Ptot'] * p0_slope * PSI_TO_PA

        # Total (stagnation) temperature T0
        # Thermocouple is in the settling chamber, so it reads T0.
        # Slope is 0.1 V/deg (either degF or degC depending on cal vintage).
        # See _convert_thermocouple_to_celsius for the branch logic.
        T0_C = _convert_thermocouple_to_celsius(
            raw_data['Temp'], temp_cal_mode)
        T0_K = T0_C + C_TO_K

        # ---------------------------------------------------------------
        # Derived static pressure
        # ---------------------------------------------------------------
        P_static = P0_Pa - dP_Pa

        # Guard against zero or negative static pressure
        P_static = np.maximum(P_static, 1.0)

        # ---------------------------------------------------------------
        # Isentropic relations
        # ---------------------------------------------------------------

        # Pressure ratio and isentropic core term:
        #   isentropic_term = (P0 / P_static)^((gamma-1)/gamma) - 1
        isentropic_term = np.power(P0_Pa / P_static, g_ratio) - 1.0
        isentropic_term = np.maximum(isentropic_term, 0.0)

        # Compressible dynamic pressure:
        #   q = (gamma / (gamma-1)) * P_static * isentropic_term
        Q_Pa = (GAMMA / gm1) * P_static * isentropic_term

        # Mach number from isentropic pressure ratio:
        #   M = sqrt( (2 / (gamma-1)) * isentropic_term )
        Mach = np.sqrt((2.0 / gm1) * isentropic_term)

        # Static temperature from total temperature:
        #   T_static = T0 / (1 + (gamma-1)/2 * M^2)
        T_static_K = T0_K / (1.0 + 0.5 * gm1 * Mach**2)

        # Static density from ideal gas law (using static quantities):
        #   rho = P_static / (R_air * T_static)
        rho = P_static / (R_AIR * T_static_K)

        # Speed of sound (using static temperature):
        #   a = sqrt(gamma * R_air * T_static)
        a = np.sqrt(GAMMA * R_AIR * T_static_K)

        # Freestream velocity:
        #   U = M * a
        U_inf = Mach * a

        # Dynamic viscosity via Sutherland's law:
        #   mu = mu_ref * (T/T_ref)^(3/2) * (T_ref + S) / (T + S)
        mu = MU_REF * np.power(T_static_K / T_REF, 1.5) * \
            (T_REF + S_SUTH) / (T_static_K + S_SUTH)

        # Reynolds number:
        #   Re = rho * U * L / mu
        ref_length_m = ref_length_in * 0.0254
        Re = rho * U_inf * ref_length_m / mu

        # ---------------------------------------------------------------
        # Store results
        # ---------------------------------------------------------------
        cond.Q = Q_Pa / PSI_TO_PA        # Dynamic pressure in psi (for coefficients)
        cond.Q_mks = Q_Pa                # Dynamic pressure in Pa
        cond.P_tot = P0_Pa               # Total (stagnation) pressure (Pa)
        cond.P_static = P_static         # Static pressure (Pa)
        cond.T0 = T0_C                   # Total (stagnation) temperature (C)
        cond.T = T_static_K - C_TO_K     # Static temperature (C)
        cond.rho = rho                   # Static density (kg/m^3)
        cond.Mach = Mach                 # Mach number
        cond.a = a                       # Speed of sound (m/s)
        cond.U_inf = U_inf               # Freestream velocity (m/s)
        cond.Re = Re                     # Reynolds number

    elif facility == 'LSWT':
        # Low Speed Wind Tunnel
        pdiff_slope = pressure_cal[pdiff_channel].slope
        cond.Q = raw_data['Pdiff'] * pdiff_slope
        cond.Q_mks = cond.Q * PSI_TO_PA

    elif facility == 'TST':
        # Trisonic Tunnel - compressible flow calculation
        # This requires additional channels and more complex calculations
        pass

    return cond


def calc_aero_coeffs(wrf: WRFForces,
                     Q: np.ndarray,
                     C: float,
                     S: float,
                     pressure_data: Optional[Dict[str, np.ndarray]] = None,
                     b: float = None) -> AeroCoefficients:
    """
    Calculate aerodynamic coefficients from WRF forces.

    Parameters
    ----------
    wrf : WRFForces
        Wind reference frame forces and moments
    Q : np.ndarray
        Dynamic pressure (consistent units with forces)
    C : float
        Reference chord length
    S : float
        Reference area
    b : float, optional
        Reference span for roll/yaw moments. Defaults to C if not provided.

    Returns
    -------
    AeroCoefficients
        Calculated aerodynamic coefficients

    Notes
    -----
    Coefficient definitions:
        Cl = Lift / (q * S)
        Cd = Drag / (q * S)
        Cs = Side / (q * S)
        CRoll = Roll / (q * S * b)    (span-based)
        CPitch = Pitch / (q * S * C)  (chord-based)
        CYaw = Yaw / (q * S * b)     (span-based)
    """
    if b is None:
        b = C

    coeffs = AeroCoefficients()

    # Denominator for force coefficients
    denom = Q * S

    # Avoid division by zero
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)

    # Force coefficients
    coeffs.Cl = wrf.Lift / denom
    coeffs.Cd = wrf.Drag / denom
    coeffs.Cs = wrf.Side / denom

    # Pitching moment uses chord (C)
    denom_pitch = denom * C
    coeffs.CPitch = wrf.Pitch / denom_pitch

    # Roll and yaw moments use span (b)
    denom_span = denom * b
    coeffs.CRoll = wrf.Roll / denom_span
    coeffs.CYaw = wrf.Yaw / denom_span

    # Pressure coefficients if available
    if pressure_data:
        for key, data in pressure_data.items():
            coeffs.pressure_coeffs[f'C{key}'] = data / Q

    return coeffs


def calc_lift_curve_slope(alpha: np.ndarray, Cl: np.ndarray,
                          alpha_range: tuple = (-5, 10)) -> float:
    """
    Calculate lift curve slope (Cl_alpha) in the linear region.

    Parameters
    ----------
    alpha : np.ndarray
        Angle of attack in degrees
    Cl : np.ndarray
        Lift coefficient
    alpha_range : tuple
        Range of alpha to use for linear fit

    Returns
    -------
    float
        Lift curve slope (per degree)
    """
    mask = (alpha >= alpha_range[0]) & (alpha <= alpha_range[1])
    if np.sum(mask) < 2:
        return np.nan

    coeffs = np.polyfit(alpha[mask], Cl[mask], 1)
    return coeffs[0]


def calc_zero_lift_alpha(alpha: np.ndarray, Cl: np.ndarray,
                         alpha_range: tuple = (-5, 10)) -> float:
    """
    Calculate zero-lift angle of attack.

    Parameters
    ----------
    alpha : np.ndarray
        Angle of attack in degrees
    Cl : np.ndarray
        Lift coefficient
    alpha_range : tuple
        Range of alpha to use for linear fit

    Returns
    -------
    float
        Zero-lift angle of attack in degrees
    """
    mask = (alpha >= alpha_range[0]) & (alpha <= alpha_range[1])
    if np.sum(mask) < 2:
        return np.nan

    coeffs = np.polyfit(alpha[mask], Cl[mask], 1)
    return -coeffs[1] / coeffs[0]


def calc_drag_polar_coeffs(Cl: np.ndarray, Cd: np.ndarray) -> Dict[str, float]:
    """
    Fit drag polar: Cd = Cd0 + K * Cl^2

    Parameters
    ----------
    Cl : np.ndarray
        Lift coefficient
    Cd : np.ndarray
        Drag coefficient

    Returns
    -------
    dict
        Dictionary with 'Cd0' and 'K' values
    """
    # Fit Cd = Cd0 + K * Cl^2
    X = np.column_stack([np.ones_like(Cl), Cl ** 2])
    coeffs, _, _, _ = np.linalg.lstsq(X, Cd, rcond=None)

    return {'Cd0': coeffs[0], 'K': coeffs[1]}


def calc_oswald_efficiency(K: float, AR: float) -> float:
    """
    Calculate Oswald efficiency factor.

    Parameters
    ----------
    K : float
        Drag polar coefficient (Cd = Cd0 + K*Cl^2)
    AR : float
        Wing aspect ratio

    Returns
    -------
    float
        Oswald efficiency factor e
    """
    return 1 / (np.pi * AR * K)


def calc_max_lift_drag_ratio(Cd0: float, K: float) -> Dict[str, float]:
    """
    Calculate maximum lift-to-drag ratio and the Cl at which it occurs.

    Parameters
    ----------
    Cd0 : float
        Zero-lift drag coefficient
    K : float
        Induced drag factor

    Returns
    -------
    dict
        Dictionary with 'L_D_max' and 'Cl_max_LD'
    """
    Cl_max_LD = np.sqrt(Cd0 / K)
    L_D_max = 1 / (2 * np.sqrt(Cd0 * K))

    return {'L_D_max': L_D_max, 'Cl_max_LD': Cl_max_LD}


def calc_static_margin(Cm_alpha: float, Cl_alpha: float) -> float:
    """
    Calculate static margin as fraction of MAC.

    Parameters
    ----------
    Cm_alpha : float
        Pitching moment slope (per degree)
    Cl_alpha : float
        Lift curve slope (per degree)

    Returns
    -------
    float
        Static margin (positive = stable)
    """
    return -Cm_alpha / Cl_alpha
