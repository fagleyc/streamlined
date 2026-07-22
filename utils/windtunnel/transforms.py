"""
Coordinate Transformations
==========================

Functions for transforming forces and moments between reference frames:
- Body Reference Frame (BRF)
- Wind Reference Frame (WRF)
- Balance axis to body axis transformations
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from .calibration import BalanceCalibration, form_higher_order_terms


@dataclass
class BRFForces:
    """Body Reference Frame forces and moments."""
    Fx: np.ndarray = field(default_factory=lambda: np.array([]))
    Fy: np.ndarray = field(default_factory=lambda: np.array([]))
    Fz: np.ndarray = field(default_factory=lambda: np.array([]))
    Mx: np.ndarray = field(default_factory=lambda: np.array([]))
    My: np.ndarray = field(default_factory=lambda: np.array([]))
    Mz: np.ndarray = field(default_factory=lambda: np.array([]))
    elements: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class WRFForces:
    """Wind Reference Frame forces and moments."""
    Lift: np.ndarray = field(default_factory=lambda: np.array([]))
    Drag: np.ndarray = field(default_factory=lambda: np.array([]))
    Side: np.ndarray = field(default_factory=lambda: np.array([]))
    Roll: np.ndarray = field(default_factory=lambda: np.array([]))
    Pitch: np.ndarray = field(default_factory=lambda: np.array([]))
    Yaw: np.ndarray = field(default_factory=lambda: np.array([]))


# Resolved external-balance (ATE) channels: wind-axis loads in
# engineering units, NOT bridge volts. Order matches WRFForces fields.
EXTERNAL_LOAD_CHANNELS = ('Lift', 'Drag', 'Side', 'Roll', 'Pitch', 'Yaw')

# Bridge channels that mark raw internal-balance (volts) data.
_BRIDGE_CHANNELS = ('N1', 'N2', 'Y1', 'Y2',
                    'AftPitch', 'AftYaw', 'FwdPitch', 'FwdYaw')


def is_external_balance_data(raw_data: Dict[str, Any]) -> bool:
    """
    True when a raw channel dict carries resolved external-balance
    (ATE) loads rather than internal-balance bridge volts.

    Prefers an explicit ``balance_type`` marker (carried through from
    the file's ``RawData.properties`` by callers that propagate it) and
    falls back to structural detection: resolved load channels present
    with no bridge channels.

    Parameters
    ----------
    raw_data : dict
        Raw channel dictionary as produced by the data_io readers.

    Returns
    -------
    bool
        True for external-balance (already-resolved) data.

    Notes
    -----
    Also accepts a ``RawData`` object directly (its ``balance_type``
    property / ``data`` dict are used).
    """
    if not hasattr(raw_data, 'get'):                  # RawData object
        marker = getattr(raw_data, 'balance_type', None)
        raw_data = getattr(raw_data, 'data', {}) or {}
    else:
        marker = raw_data.get('balance_type')
    if isinstance(marker, bytes):
        marker = marker.decode('utf-8', errors='replace')
    if isinstance(marker, str):
        return marker.strip().lower() == 'external'

    has_loads = all(ch in raw_data for ch in ('Lift', 'Drag', 'Pitch'))
    has_bridges = any(ch in raw_data for ch in _BRIDGE_CHANNELS)
    return has_loads and not has_bridges


def wrf_from_resolved_loads(raw_data: Dict[str, np.ndarray],
                            n_samples: Optional[int] = None) -> WRFForces:
    """
    Build WRFForces directly from resolved external-balance channels.

    ATE external-balance files already carry wind-axis loads (Lift,
    Drag, Side and Roll, Pitch, Yaw moments) in engineering units, so
    neither the bridge-to-force calibration (:func:`calc_brf_forces`)
    nor the body-to-wind rotation (:func:`calc_wrf_forces`) applies —
    the channels pass straight through under their in-file names.

    Parameters
    ----------
    raw_data : dict
        Raw channel dictionary containing the resolved load channels.
    n_samples : int, optional
        Sample count used to zero-fill missing channels; inferred from
        the first available load channel when omitted.

    Returns
    -------
    WRFForces
        Wind reference frame loads, passed through unmodified.
    """
    if n_samples is None:
        for ch in EXTERNAL_LOAD_CHANNELS:
            if ch in raw_data:
                n_samples = len(np.atleast_1d(raw_data[ch]))
                break
        else:
            n_samples = 0

    wrf = WRFForces()
    for ch in EXTERNAL_LOAD_CHANNELS:
        value = raw_data.get(ch)
        if value is None:
            arr = np.zeros(n_samples)
        else:
            arr = np.atleast_1d(np.asarray(value, dtype=float))
        setattr(wrf, ch, arr)
    return wrf


@dataclass
class Geometry:
    """Model geometry reference values."""
    C: float = 1.0  # Reference chord length (inches)
    S: float = 1.0  # Reference area (sq inches)
    b: float = 1.0  # Reference span (inches) — used for CRoll, CYaw
    mshift: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))  # MRC shift (x, y, z)
    flip: bool = False


def get_distance_values(cal: BalanceCalibration) -> Dict[str, float]:
    """
    Extract distance values from calibration data.

    Parameters
    ----------
    cal : BalanceCalibration
        Calibration data

    Returns
    -------
    dict
        Dictionary with dx1, dx2, dy1, dy2 values
    """
    distances = cal.distances.values
    result = {'dx1': 0.0, 'dx2': 0.0, 'dy1': 0.0, 'dy2': 0.0}

    search_terms = {
        'dx1': ['x1', 'N1'],
        'dx2': ['x2', 'N2'],
        'dy1': ['y1', 'Y1'],
        'dy2': ['y2', 'Y2']
    }

    for key, terms in search_terms.items():
        for dist_name, dist_value in distances.items():
            for term in terms:
                if term in dist_name:
                    result[key] = dist_value
                    break

    return result


def calc_brf_forces(raw_data: Dict[str, np.ndarray],
                    cal: BalanceCalibration,
                    geo: Geometry,
                    balance_config: str = 'Force') -> BRFForces:
    """
    Calculate Body Reference Frame (BRF) forces from raw balance data.

    Parameters
    ----------
    raw_data : dict
        Raw data dictionary containing balance channel voltages
    cal : BalanceCalibration
        Balance calibration data
    geo : Geometry
        Model geometry
    balance_config : str
        Balance configuration: 'Force' or 'Moment'

    Returns
    -------
    BRFForces
        Body reference frame forces and moments

    Notes
    -----
    For Force configuration:
        - Forces are direct sums of element pairs
        - Moments are calculated using distances and MRC shifts

    For Moment configuration:
        - Forces are calculated from element differences
        - Moments are averages with MRC corrections

    Raises
    ------
    ValueError
        If ``raw_data`` carries resolved external-balance loads
        (balance_type == 'external'): those channels are already forces
        and moments, so pushing them through the volts-to-forces
        calibration would silently corrupt them.
    """
    if is_external_balance_data(raw_data):
        raise ValueError(
            "raw_data carries resolved external-balance loads "
            "(Lift/Drag/... in engineering units), not bridge volts; "
            "bridge-to-force calibration does not apply. Use "
            "wrf_from_resolved_loads() to pass the loads through."
        )

    # Internal (sting) balance data is bridge volts: a fitted .vol
    # balance calibration is mandatory. Fail with a clear message
    # instead of an AttributeError deep in the math.
    if cal is None or np.size(getattr(cal, 'coeffs', np.array([]))) == 0:
        raise ValueError(
            "internal-balance data (bridge volts) requires a fitted "
            "balance calibration: load a .vol file (e.g. via "
            "DAQ.cal_balance / read_vol_file + calc_coeffs) before "
            "reducing."
        )

    brf = BRFForces()

    # Get distance values
    dist = get_distance_values(cal)
    dx1, dx2 = dist['dx1'], dist['dx2']
    dy1, dy2 = dist['dy1'], dist['dy2']

    # Get excitation voltage
    if 'Excitation' in raw_data:
        excitation = raw_data['Excitation']
    else:
        excitation = np.ones(len(list(raw_data.values())[0]))

    # Build raw voltage matrix based on balance configuration
    if balance_config == 'Force':
        # Try different naming conventions
        if 'N1' in raw_data:
            raw_volts = np.column_stack([
                raw_data['N1'] / excitation,
                raw_data['N2'] / excitation,
                raw_data['Y1'] / excitation,
                raw_data['Y2'] / excitation,
                raw_data['Axial'] / excitation,
                raw_data['Roll'] / excitation
            ])
        else:
            # Alternative naming convention
            raw_volts = np.column_stack([
                raw_data['AftPitch'] / excitation,
                raw_data['AftYaw'] / excitation,
                raw_data['FwdPitch'] / excitation,
                raw_data['FwdYaw'] / excitation,
                raw_data['Axial'] / excitation,
                raw_data['Roll'] / excitation
            ])
    else:  # Moment configuration
        # Try moment-balance naming first, fall back to force-balance names
        if 'AftPitch' in raw_data:
            ch1, ch2, ch3, ch4 = 'AftPitch', 'AftYaw', 'FwdPitch', 'FwdYaw'
        else:
            # Map force-balance channel names to moment-balance positions
            ch1, ch2, ch3, ch4 = 'N1', 'N2', 'Y1', 'Y2'
        raw_volts = np.column_stack([
            raw_data[ch1] / excitation,
            raw_data[ch2] / excitation,
            raw_data[ch3] / excitation,
            raw_data[ch4] / excitation,
            raw_data['Axial'] / excitation,
            raw_data['Roll'] / excitation,
        ])

        # Adjust distances for moment balance
        if dx1 > 2:
            dx1 /= 2
            dx2 /= 2
            dy1 /= 2
            dy2 /= 2

    # Apply calibration
    order_map = {'Linear': 1, 'Quadratic': 2, 'Cubic': 3}
    order = order_map.get(cal.cal_type, 1)
    X = form_higher_order_terms(raw_volts, order)
    elements = X @ cal.coeffs

    brf.elements = elements

    # Calculate forces and moments based on configuration
    mshift = geo.mshift

    if balance_config == 'Force':
        brf.Fz = elements[:, 0] + elements[:, 1]  # N1 + N2
        brf.Fy = elements[:, 2] + elements[:, 3]  # Y1 + Y2
        brf.Fx = elements[:, 4]                    # Axial

        # Moments with MRC shift corrections
        brf.Mx = elements[:, 5] - brf.Fy * mshift[2]
        brf.My = (elements[:, 0] * (dx1 - mshift[0]) -
                  elements[:, 1] * (dx2 + mshift[0]) -
                  brf.Fx * mshift[2])
        brf.Mz = (elements[:, 2] * (dy1 + mshift[1]) -
                  elements[:, 3] * (dy2 - mshift[1]) -
                  brf.Fy * mshift[0])

    else:  # Moment configuration
        brf.Fz = (elements[:, 0] - elements[:, 2]) / (dx1 + dx2)
        brf.Fy = (elements[:, 1] - elements[:, 3]) / (dy1 + dy2)
        brf.Fx = elements[:, 4]

        brf.Mx = elements[:, 5] - brf.Fy * mshift[2]
        brf.My = ((elements[:, 0] + elements[:, 2]) / 2 -
                  brf.Fx * mshift[2] - brf.Fz * mshift[0])
        brf.Mz = ((elements[:, 1] + elements[:, 3]) / 2 +
                  brf.Fx * mshift[1] + brf.Fy * mshift[0])

    return brf


def calc_wrf_forces(brf: BRFForces, alpha: np.ndarray, beta: np.ndarray) -> WRFForces:
    """
    Transform Body Reference Frame forces to Wind Reference Frame.

    Parameters
    ----------
    brf : BRFForces
        Body reference frame forces
    alpha : np.ndarray
        Angle of attack in degrees
    beta : np.ndarray
        Sideslip angle in degrees

    Returns
    -------
    WRFForces
        Wind reference frame forces and moments

    Notes
    -----
    Transformation equations:
        Lift = -Fx*sin(a) + Fz*cos(a)
        Drag = Fx*cos(b)*cos(a) - Fy*sin(b) + Fz*cos(b)*sin(a)
        Side = Fx*sin(b)*cos(a) + Fy*cos(b) + Fz*sin(b)*sin(a)

    Moments are transferred directly (body and wind axes aligned for moments).
    """
    wrf = WRFForces()

    # Convert to radians
    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(beta)

    cos_a = np.cos(alpha_rad)
    sin_a = np.sin(alpha_rad)
    cos_b = np.cos(beta_rad)
    sin_b = np.sin(beta_rad)

    # Transform forces
    wrf.Lift = cos_a * brf.Fz - sin_a * brf.Fx

    wrf.Drag = (brf.Fx * cos_b * cos_a -
                brf.Fy * sin_b +
                brf.Fz * sin_a * cos_b)

    wrf.Side = (brf.Fx * sin_b * cos_a +
                brf.Fy * cos_b +
                brf.Fz * sin_a * sin_b)

    # Moments transfer directly
    wrf.Roll = brf.Mx
    wrf.Pitch = brf.My
    wrf.Yaw = brf.Mz

    return wrf


def subtract_tare(air_on: Dict[str, np.ndarray],
                  air_off: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Subtract tare (air-off) data from air-on data.

    Parameters
    ----------
    air_on : dict
        Air-on data dictionary
    air_off : dict
        Air-off (tare) data dictionary

    Returns
    -------
    dict
        Tare-corrected data

    Notes
    -----
    Subtracts air_off from air_on for all matching keys except 'Time'.
    Handles different array lengths by truncating to shorter length.
    """
    result = {}

    for key in air_on:
        if key.lower() == 'time':
            result[key] = air_on[key]
        elif key in air_off:
            on_data = air_on[key]
            off_data = air_off[key]

            # Handle length mismatch
            min_len = min(len(on_data), len(off_data))
            result[key] = on_data[:min_len] - off_data[:min_len]
        else:
            result[key] = air_on[key]

    return result


def subtract_brf_forces(on: BRFForces, off: BRFForces) -> BRFForces:
    """
    Subtract tare BRF forces.

    Parameters
    ----------
    on : BRFForces
        Air-on BRF forces
    off : BRFForces
        Air-off (tare) BRF forces

    Returns
    -------
    BRFForces
        Tare-corrected forces
    """
    result = BRFForces()

    for attr in ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']:
        on_val = getattr(on, attr)
        off_val = getattr(off, attr)
        min_len = min(len(on_val), len(off_val))
        setattr(result, attr, on_val[:min_len] - off_val[:min_len])

    return result


def subtract_wrf_forces(on: WRFForces, off: WRFForces) -> WRFForces:
    """
    Subtract tare WRF forces.

    Parameters
    ----------
    on : WRFForces
        Air-on WRF forces
    off : WRFForces
        Air-off (tare) WRF forces

    Returns
    -------
    WRFForces
        Tare-corrected forces
    """
    result = WRFForces()

    for attr in ['Lift', 'Drag', 'Side', 'Roll', 'Pitch', 'Yaw']:
        on_val = getattr(on, attr)
        off_val = getattr(off, attr)
        # Subtract only the mean (DC component) of air-off tare,
        # preserving the time-varying content of air-on
        setattr(result, attr, on_val - np.mean(off_val))

    return result
