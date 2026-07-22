"""
External (ATE) Balance Reduction
================================

Port of the external-balance handling from the MATLAB pipeline that was
used on this tunnel (see ``deprecated/`` for the originals):

* ``deprecated/scripts/calc_coeffs.m`` ('External' case) — the external
  balance needs no volts->forces calibration: its channels are already
  resolved loads. The MATLAB chain works in ``lb`` / ``in-lb``
  (``d.Units = {'lb' 'lb' 'lb' 'in-lb' 'in-lb' 'in-lb'}``) with channel
  order ``d.Forcechan = {'Drag' 'Side' 'Lift' 'Roll' 'Pitch' 'Yaw'}``
  and per-channel calibration bias values taken from the cal file.
* ``deprecated/scripts/DAQ_reduce_raw.m`` — air-on and air-off points
  are each carried to wind-axis loads, then tared:
  ``o(i).Aero = str_subtract(o(i).AirON, o(i).AirOFF)`` (Streamlined's
  established :func:`~.transforms.subtract_wrf_forces` subtracts the
  tare *mean* instead of sample-wise, preserving air-on dynamics; the
  external path plugs into that same tare step).
* ``deprecated/scripts/DAQ_calc_coeffs.m`` — coefficients divide the
  tared wind-axis loads by ``Q*S`` (forces) and ``Q*S*C`` (all three
  moments use the reference chord C, not span).
* ``deprecated/scripts/DPM_calc_BRF_forces.m`` — the MRC shift
  (``geo.mshift``) is applied at element level for the INTERNAL balance
  only; the MATLAB pipeline never transfers the external ATE loads to a
  different moment reference center (equivalent to mshift == 0).
* ``deprecated/scripts/calc_uncertainty_Extbalance.m`` +
  ``calc_uncertainty.m`` — external-balance bias / precision / total
  uncertainty, ported in :func:`calc_uncertainty_ext_balance`.

Unit note: the Freestream ATE_Balance group streams loads in N / N*m
(``meta.channels.ATE_Balance.<chan>.unit == 'N'`` in the run files),
while the historical reduction chain above works in lb / in-lb, with Q
in psi and S / C in in^2 / in. :func:`external_loads_to_ips` performs
that conversion when the run file marks SI loads.

span_config: Freestream mode-2 files carry a ``span_config`` marker
('full' / 'half'). The MATLAB pipeline contains no half-span handling
(no span logic anywhere in deprecated/scripts), so the marker is carried
through in ``RawData.properties`` but intentionally applies no scaling
here.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Sequence

from .transforms import WRFForces

# ---------------------------------------------------------------------------
# Constants ported from deprecated/scripts/calc_coeffs.m ('External' case)
# ---------------------------------------------------------------------------

# Channel order of the external balance calibration (d.Forcechan)
EXTERNAL_CHANNEL_ORDER = ('Drag', 'Side', 'Lift', 'Roll', 'Pitch', 'Yaw')

# Per-channel calibration bias, "From Cal file" (calc_coeffs.m d.Bias)
EXTERNAL_CAL_BIAS = np.array(
    [0.0164, 0.0368, 0.0238, 0.0201, 0.0158, 0.0081])

# Engineering units of each channel in the historical chain (d.Units)
EXTERNAL_CAL_UNITS = ('lb', 'lb', 'lb', 'in-lb', 'in-lb', 'in-lb')

# ---------------------------------------------------------------------------
# Unit conversions (ATE streams N / N*m; chain works in lb / in-lb,
# see calc_coeffs.m Units field)
# ---------------------------------------------------------------------------

LBF_TO_N = 4.4482216152605
N_TO_LBF = 1.0 / LBF_TO_N                 # force: N -> lbf
NM_TO_INLB = 1.0 / (LBF_TO_N * 0.0254)    # moment: N*m -> in*lb

_FORCE_CHANNELS = ('Lift', 'Drag', 'Side')
_MOMENT_CHANNELS = ('Roll', 'Pitch', 'Yaw')

# raw-dict marker values that flag newton-based (SI) resolved loads
_SI_MARKERS = ('n', 'si', 'mks', 'newton', 'newtons', 'n*m', 'nm')


def external_loads_in_si(raw_data: Dict[str, Any]) -> bool:
    """
    True when a raw channel dict carries a ``load_units`` marker flagging
    newton-based resolved loads (as written by Freestream: the
    ATE_Balance channel unit attribute is 'N' / 'N*m').

    The marker is propagated from ``RawData.properties['load_units']``
    by the directory loaders. Dicts without a marker are assumed to
    already be in the chain's native lb / in-lb (legacy behavior).
    """
    marker = raw_data.get('load_units') if hasattr(raw_data, 'get') else None
    if isinstance(marker, bytes):
        marker = marker.decode('utf-8', errors='replace')
    return isinstance(marker, str) and marker.strip().lower() in _SI_MARKERS


def external_loads_to_ips(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert resolved external-balance loads from SI to the chain's
    native units: N -> lbf (Lift/Drag/Side), N*m -> in*lb
    (Roll/Pitch/Yaw).

    The historical reduction works in lb / in-lb with Q in psi and
    S, C in inches (deprecated/scripts/calc_coeffs.m 'External':
    ``d.Units = {'lb' 'lb' 'lb' 'in-lb' 'in-lb' 'in-lb'}``), so the
    Freestream ATE stream (N / N*m) must be converted before q
    normalization. Conversion only happens when the dict carries an SI
    ``load_units`` marker (see :func:`external_loads_in_si`); unmarked
    dicts pass through untouched, preserving the historical
    pass-through behavior for data already in lb / in-lb.

    Returns a shallow copy with the six load channels converted (other
    channels are shared by reference); the input dict is not modified.
    """
    if not external_loads_in_si(raw_data):
        return raw_data

    converted = dict(raw_data)
    for ch in _FORCE_CHANNELS:
        if ch in converted:
            converted[ch] = np.asarray(converted[ch], dtype=float) * N_TO_LBF
    for ch in _MOMENT_CHANNELS:
        if ch in converted:
            converted[ch] = np.asarray(converted[ch], dtype=float) * NM_TO_INLB
    converted['load_units'] = 'lb'
    return converted


def transfer_external_loads_to_mrc(wrf: WRFForces,
                                   alpha: np.ndarray,
                                   beta: np.ndarray,
                                   mshift: Sequence[float]) -> WRFForces:
    """
    Transfer resolved wind-axis moments to a shifted moment reference
    center.

    The MATLAB pipeline applies the MRC shift for the INTERNAL balance
    only, at element level inside deprecated/scripts/DPM_calc_BRF_forces.m
    ('Force' case); the external ATE loads were never re-referenced
    (equivalent to mshift == [0, 0, 0], which makes this a no-op). This
    helper extends that internal-path convention to resolved loads: the
    wind-axis forces are rotated back to body axes (inverse of the
    rotation in deprecated/scripts/DPM_calc_WRF_forces.m), the
    DPM_calc_BRF_forces.m net-force moment-arm terms are applied,

        Mx = Mx - Fy*mz                      (Roll)
        My = My - Fz*mx - Fx*mz              (Pitch)
        Mz = Mz + Fy*my - Fy*mx              (Yaw)

    and the moments transfer directly back to the wind frame (moments
    pass straight through BRF<->WRF in DPM_calc_WRF_forces.m).

    Parameters
    ----------
    wrf : WRFForces
        Resolved wind-axis loads (forces lb, moments in-lb — or any
        consistent unit system where mshift shares the length unit).
    alpha, beta : np.ndarray
        Model attitude in degrees (per-sample or scalar).
    mshift : sequence of 3 floats
        MRC shift (dx, dy, dz) in the same length units as the moments'
        arm (inches for the historical chain).

    Returns
    -------
    WRFForces
        Loads with the moments re-referenced; forces unchanged.
    """
    mx, my, mz = (float(m) for m in np.asarray(mshift, dtype=float))
    if mx == 0.0 and my == 0.0 and mz == 0.0:
        return wrf  # faithful MATLAB behavior: external loads untouched

    a = np.deg2rad(np.atleast_1d(np.asarray(alpha, dtype=float)))
    b = np.deg2rad(np.atleast_1d(np.asarray(beta, dtype=float)))
    n = max(len(np.atleast_1d(wrf.Lift)), len(a), len(b))
    a = np.broadcast_to(a, (n,)) if a.size == 1 else a
    b = np.broadcast_to(b, (n,)) if b.size == 1 else b

    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)

    # Forward rotation from DPM_calc_WRF_forces.m:
    #   Lift = -sa*Fx           + ca*Fz
    #   Drag =  cb*ca*Fx - sb*Fy + sa*cb*Fz
    #   Side =  sb*ca*Fx + cb*Fy + sa*sb*Fz
    # Solve the 3x3 system per sample for the body-axis forces.
    R = np.empty((n, 3, 3))
    R[:, 0, 0] = -sa
    R[:, 0, 1] = 0.0
    R[:, 0, 2] = ca
    R[:, 1, 0] = cb * ca
    R[:, 1, 1] = -sb
    R[:, 1, 2] = sa * cb
    R[:, 2, 0] = sb * ca
    R[:, 2, 1] = cb
    R[:, 2, 2] = sa * sb

    lds = np.stack([np.broadcast_to(np.atleast_1d(wrf.Lift), (n,)),
                    np.broadcast_to(np.atleast_1d(wrf.Drag), (n,)),
                    np.broadcast_to(np.atleast_1d(wrf.Side), (n,))], axis=1)
    F = np.linalg.solve(R, lds[..., None])[..., 0]
    Fx, Fy, Fz = F[:, 0], F[:, 1], F[:, 2]

    out = WRFForces()
    out.Lift = wrf.Lift
    out.Drag = wrf.Drag
    out.Side = wrf.Side
    # Moment-arm terms per DPM_calc_BRF_forces.m ('Force' case), applied
    # to the directly-transferring moments (Roll=Mx, Pitch=My, Yaw=Mz):
    out.Roll = wrf.Roll - Fy * mz
    out.Pitch = wrf.Pitch - Fz * mx - Fx * mz
    out.Yaw = wrf.Yaw + Fy * my - Fy * mx
    return out


# ---------------------------------------------------------------------------
# Uncertainty — port of deprecated/scripts/calc_uncertainty_Extbalance.m
# (bias) and calc_uncertainty.m (precision / total)
# ---------------------------------------------------------------------------

# Our coefficient names -> the moment-case names used in the MATLAB
# switch (calc_uncertainty_Extbalance.m cases 'Cmx'/'Cmy'/'Cmz'). In the
# original, DAQ_calc_coeffs.m produced CRoll/CPitch/CYaw so those switch
# cases only fired for pipelines using the Cm* naming; the formulas are
# identical, so the port applies them to the CRoll/CPitch/CYaw names.
_MOMENT_CASE_MAP = {'CRoll': 'Cmx', 'CPitch': 'Cmy', 'CYaw': 'Cmz'}


def calc_precision_uncertainty_cases(values_by_case: List[np.ndarray],
                                     confidence: float = 0.975
                                     ) -> np.ndarray:
    """
    Precision (random) error across repeat cases, ported from
    deprecated/scripts/calc_uncertainty.m:

        prec = std(temp, [], 2) * tinv(.975, ncase) / sqrt(ncase)

    (Note: the MATLAB uses ``ncase`` degrees of freedom in ``tinv``,
    not ncase-1 — reproduced exactly.)

    Parameters
    ----------
    values_by_case : list of np.ndarray
        One time-series (or scalar array) per repeat case; all are
        truncated to the shortest length.
    confidence : float
        One-sided t quantile (MATLAB uses .975).

    Returns
    -------
    np.ndarray
        Per-sample precision error (2-sided 95% half-width).
    """
    from scipy.stats import t as t_dist

    ncase = len(values_by_case)
    if ncase < 2:
        n = len(np.atleast_1d(values_by_case[0])) if values_by_case else 0
        return np.zeros(n)

    nmin = min(len(np.atleast_1d(v)) for v in values_by_case)
    temp = np.column_stack([np.atleast_1d(v)[:nmin] for v in values_by_case])
    # MATLAB std(...,[],2) is the sample std (N-1 normalization)
    return (np.std(temp, axis=1, ddof=1)
            * t_dist.ppf(confidence, ncase) / np.sqrt(ncase))


@np.errstate(divide='ignore', invalid='ignore')
def calc_uncertainty_ext_balance(coeffs: Dict[str, np.ndarray],
                                 loads: np.ndarray,
                                 alpha_deg: np.ndarray,
                                 Q: np.ndarray,
                                 S: float,
                                 C: float,
                                 cal_bias: np.ndarray = EXTERNAL_CAL_BIAS,
                                 config: str = 'Vertical',
                                 prec: Optional[Dict[str, np.ndarray]] = None
                                 ) -> Dict[str, Dict[str, Any]]:
    """
    External-balance bias / total uncertainty, ported from
    deprecated/scripts/calc_uncertainty_Extbalance.m ('Vertical'
    mounting config; 'Horizontal' was not implemented in the original).

    Parameters
    ----------
    coeffs : dict
        Coefficient name -> per-sample array (the fields of
        d.red(1).Coeffs; names Cl/Cd/Cs and CRoll/CPitch/CYaw or
        Cmx/Cmy/Cmz).
    loads : np.ndarray, shape (n, 6)
        Balance loads in the calibration channel order
        ``EXTERNAL_CHANNEL_ORDER`` = (Drag, Side, Lift, Roll, Pitch,
        Yaw), in lb / in-lb (fb = d.red(1).BRF.Elems in the original).
    alpha_deg : np.ndarray
        Angle of attack in degrees (converted to rad, as in the .m).
    Q : np.ndarray
        Dynamic pressure (psi).
    S, C : float
        Reference area (in^2) and chord (in).
    cal_bias : np.ndarray, shape (6,)
        Per-channel balance bias (bfb = d.cal.Bias; defaults to the
        calc_coeffs.m 'From Cal file' values).
    config : str
        Mounting configuration; only 'Vertical' is implemented,
        matching the original.
    prec : dict, optional
        Coefficient name -> precision-error array (d.unc.prec), used
        for the total; missing entries are treated as zero.

    Returns
    -------
    dict
        {'InfCoeffs': {...}, 'bias': {...}, 'total': {...}} mirroring
        the MATLAB d.unc structure. Bias entries hold the per-source
        biases plus 'total'; 'total' holds sqrt(bias^2 + prec^2).

    Notes
    -----
    Constants are reproduced exactly from the .m file, including its
    quirks: ba = deg2rad(0.05); bQ = .0005*mean(Q); bS = .005^2 (the
    source squares the area bias before use); bC = .005; the 'Cmy' case
    uses fb(:,6) in the influence coefficients but bfb(5) in the bias
    RSS.
    """
    if config != 'Vertical':
        # calc_uncertainty_Extbalance.m: 'Horizontal config not implemented'
        raise NotImplementedError(
            "Only the 'Vertical' mounting config is implemented "
            "(as in calc_uncertainty_Extbalance.m)")

    fb = np.atleast_2d(np.asarray(loads, dtype=float))
    bfb = np.asarray(cal_bias, dtype=float)
    alpha = np.deg2rad(np.asarray(alpha_deg, dtype=float))
    Q = np.asarray(Q, dtype=float)
    prec = prec or {}

    # Bias constants — verbatim from calc_uncertainty_Extbalance.m
    ba = np.deg2rad(0.05)        # attitude bias [rad]
    bQ = 0.0005 * np.mean(Q)     # dynamic-pressure bias
    bS = 0.005 ** 2              # area bias (pre-squared in the source)
    bC = 0.005                   # chord bias

    inf_coeffs: Dict[str, Dict[str, np.ndarray]] = {}
    bias: Dict[str, Dict[str, Any]] = {}
    total: Dict[str, np.ndarray] = {}

    for name in coeffs:
        case = _MOMENT_CASE_MAP.get(name, name)
        ic: Dict[str, np.ndarray] = {}
        bt = None

        if case == 'Cl':
            # Cl = [Fy*cos(a) - Fx*sin(a)]/QS  (Vertical mount)
            ic['pCpFy'] = np.cos(alpha) / Q / S
            ic['pCpFx'] = -np.sin(alpha) / Q / S
            ic['pCpa'] = -(fb[:, 1] * np.sin(alpha)
                           - fb[:, 0] * np.cos(alpha)) / Q / S
            ic['pCpQ'] = -0.5 * (fb[:, 1] * np.cos(alpha)
                                 - fb[:, 0] * np.sin(alpha)) / (Q ** 2) / S
            ic['pCpS'] = -0.5 * (fb[:, 1] * np.cos(alpha)
                                 - fb[:, 0] * np.sin(alpha)) / (S ** 2) / Q
            bt = np.sqrt((ic['pCpFx'] * bfb[0]) ** 2
                         + (ic['pCpFy'] * bfb[1]) ** 2
                         + (ic['pCpQ'] * bQ) ** 2
                         + (ic['pCpa'] * ba) ** 2
                         + (ic['pCpS'] * bS) ** 2)
            bias[name] = {'Fy': bfb[0], 'Fx': bfb[1], 'Q': bQ,
                          'a': ba, 'S': bS, 'total': bt}

        elif case == 'Cd':
            # Cd = [Fx*cos(a) + Fy*sin(a)]/QS
            ic['pCpFy'] = np.sin(alpha) / Q / S
            ic['pCpFx'] = np.cos(alpha) / Q / S
            ic['pCpQ'] = -0.5 * (fb[:, 1] * np.sin(alpha)
                                 + fb[:, 0] * np.cos(alpha)) / (Q ** 2) / S
            ic['pCpS'] = -0.5 * (fb[:, 1] * np.sin(alpha)
                                 + fb[:, 0] * np.cos(alpha)) / (S ** 2) / Q
            bt = np.sqrt((ic['pCpFx'] * bfb[0]) ** 2
                         + (ic['pCpFy'] * bfb[1]) ** 2
                         + (ic['pCpQ'] * bQ) ** 2
                         + (ic['pCpS'] * bS) ** 2)
            bias[name] = {'Fy': bfb[0], 'Fx': bfb[1], 'Q': bQ,
                          'S': bS, 'total': bt}

        elif case == 'Cs':
            # Cs = Fz/QS
            ic['pCpFz'] = 1.0 / Q / S
            ic['pCpQ'] = -0.5 * fb[:, 2] / (Q ** 2) / S
            ic['pCpS'] = -0.5 * fb[:, 2] / (S ** 2) / Q
            bt = np.sqrt((ic['pCpFz'] * bfb[2]) ** 2
                         + (ic['pCpQ'] * bQ) ** 2
                         + (ic['pCpS'] * bS) ** 2)
            bias[name] = {'Fz': bfb[2], 'Q': bQ, 'S': bS, 'total': bt}

        elif case == 'Cmx':
            # Cmx = Roll/QSC
            ic['pCpMx'] = 1.0 / Q / S / C
            ic['pCpQ'] = -0.5 * fb[:, 3] / (Q ** 2) / S / C
            ic['pCpS'] = -0.5 * fb[:, 3] / (S ** 2) / Q / C
            ic['pCpC'] = -0.5 * fb[:, 3] / (C ** 2) / Q / S
            bt = np.sqrt((ic['pCpMx'] * bfb[3]) ** 2
                         + (ic['pCpQ'] * bQ) ** 2
                         + (ic['pCpC'] * bC) ** 2
                         + (ic['pCpS'] * bS) ** 2)
            bias[name] = {'Mx': bfb[3], 'Q': bQ, 'C': bC, 'S': bS,
                          'total': bt}

        elif case == 'Cmy':
            # Cmy = Pitch/QSC — the source uses fb(:,6) in the
            # influence coefficients but bfb(5) in the RSS (faithful)
            ic['pCpMy'] = 1.0 / Q / S / C
            ic['pCpQ'] = -0.5 * fb[:, 5] / (Q ** 2) / S / C
            ic['pCpS'] = -0.5 * fb[:, 5] / (S ** 2) / Q / C
            ic['pCpC'] = -0.5 * fb[:, 5] / (C ** 2) / Q / S
            bt = np.sqrt((ic['pCpMy'] * bfb[4]) ** 2
                         + (ic['pCpQ'] * bQ) ** 2
                         + (ic['pCpC'] * bC) ** 2
                         + (ic['pCpS'] * bS) ** 2)
            bias[name] = {'My': bfb[4], 'Q': bQ, 'S': bS, 'C': bC,
                          'total': bt}

        elif case == 'Cmz':
            # Cmz = Yaw/QSC
            ic['pCpMz'] = 1.0 / Q / S / C
            ic['pCpQ'] = -0.5 * fb[:, 5] / (Q ** 2) / S / C
            ic['pCpS'] = -0.5 * fb[:, 5] / (S ** 2) / Q / C
            ic['pCpC'] = -0.5 * fb[:, 5] / (C ** 2) / Q / S
            bt = np.sqrt((ic['pCpMz'] * bfb[5]) ** 2
                         + (ic['pCpQ'] * bQ) ** 2
                         + (ic['pCpC'] * bC) ** 2
                         + (ic['pCpS'] * bS) ** 2)
            bias[name] = {'Mz': bfb[5], 'Q': bQ, 'S': bS, 'C': bC,
                          'total': bt}

        else:
            continue  # non-coefficient / pressure fields: no bias case

        inf_coeffs[name] = ic
        p = np.asarray(prec.get(name, 0.0), dtype=float)
        total[name] = np.sqrt(bt ** 2 + p ** 2)

    return {'InfCoeffs': inf_coeffs, 'bias': bias, 'total': total}


def build_load_matrix(wrf: WRFForces) -> np.ndarray:
    """
    Stack WRF loads into an (n, 6) matrix in the external calibration
    channel order (Drag, Side, Lift, Roll, Pitch, Yaw — calc_coeffs.m
    d.Forcechan), the layout expected by
    :func:`calc_uncertainty_ext_balance`.
    """
    cols = [np.atleast_1d(getattr(wrf, ch)) for ch in EXTERNAL_CHANNEL_ORDER]
    n = max(len(c) for c in cols)
    cols = [np.broadcast_to(c, (n,)) if c.size == 1 else c[:n] for c in cols]
    return np.column_stack(cols)
