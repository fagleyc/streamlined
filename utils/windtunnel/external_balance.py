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
('full' / 'half') written from the ATE driver's model-span mapping. The
marker selects the RESOLUTION ALGORITHM — see
:func:`resolve_external_wrf`. The two mountings do not merely differ by
a scale factor; they permute which balance channel carries which model
load, and the half-span mounting additionally needs an alpha rotation:

full span ("Horizontal" mounting)
    The model pitches on the incidence strut ABOVE the balance, so the
    balance never tilts with alpha. Its Lift/Drag/Side channels are
    already the wind-axis loads and pass straight through — the
    behaviour this module has always had.

half span ("Vertical" mounting, ``d.geo.config == 'Vertical'`` in the
MATLAB)
    A semispan model stands on the turntable with its span vertical,
    and alpha is driven by the YAW drive. Per the ATE manual
    (AID-010-10015-1 §3.2) "during Yaw rotation the entire Balance
    moves with the model", so the balance axes are BODY-fixed: its
    horizontal channels must be rotated by alpha, and its vertical
    channel measures the model's SIDE force, not lift. The MATLAB
    encodes exactly this in the coefficient definitions inside
    ``deprecated/scripts/calc_uncertainty_Extbalance.m`` ('Vertical'):

        Cl  = [Fy*cos(a) - Fx*sin(a)]/QS   Fy = fb(:,2) = Side channel
        Cd  = [Fx*cos(a) + Fy*sin(a)]/QS   Fx = fb(:,1) = Drag channel
        Cs  = Fz/QS                        Fz = fb(:,3) = Lift channel
        Cmy = .../QSC  using fb(:,6)       the YAW channel carries the
                                           model's PITCHING moment

    (that last one is why the 'Cmy' case reads ``fb(:,6)`` under a
    comment saying ``Cmy = Mz/QSC`` — it is the axis permutation, not a
    typo. The ``bfb(5)`` in the same block IS a typo; see
    :func:`calc_uncertainty_ext_balance`.)

Load units: the OGI's engineering-unit setting is operator-selectable
(AID-012-10015-1, Settings -> Units: "Kg and Kgm, N and Nm, Lb and
Lbft", or raw counts) and is NOT reported on the wire, so the recorded
per-channel unit attribute is the only witness. ``load_units`` therefore
distinguishes all four systems, including lb-with-lb*FT, whose moments
need a x12 conversion the historical lb/in-lb pass-through would miss.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Sequence

from .transforms import WRFForces, wrf_from_resolved_loads

# ---------------------------------------------------------------------------
# Model-span / mounting configuration
# ---------------------------------------------------------------------------

#: full-span model on the incidence strut — the MATLAB 'Horizontal' mount
SPAN_FULL = 'full'
#: semispan model on the turntable — the MATLAB 'Vertical' mount
SPAN_HALF = 'half'
SPAN_CONFIGS = (SPAN_FULL, SPAN_HALF)

#: span_config -> the mounting name used by d.geo.config in the MATLAB
SPAN_TO_MOUNT = {SPAN_FULL: 'Horizontal', SPAN_HALF: 'Vertical'}

_HALF_ALIASES = ('half', 'halfspan', 'half-span', 'half_span',
                 'semispan', 'semi-span', 'semi_span', 'vertical')


def normalize_span_config(value: Any) -> str:
    """
    Normalize a span/mount marker to ``'full'`` or ``'half'``.

    Accepts the Freestream marker ('full' / 'half'), the MATLAB mount
    name ('Horizontal' / 'Vertical'), bytes, and None. Anything
    unrecognised — including a missing marker — falls back to
    ``'full'``, which is the historical pass-through behaviour.
    """
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='replace')
    if isinstance(value, np.ndarray):
        value = value.item() if value.size == 1 else ''
    if not isinstance(value, str):
        return SPAN_FULL
    return SPAN_HALF if value.strip().lower() in _HALF_ALIASES else SPAN_FULL

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
KGF_TO_N = 9.80665                        # standard gravity

_FORCE_CHANNELS = ('Lift', 'Drag', 'Side')
_MOMENT_CHANNELS = ('Roll', 'Pitch', 'Yaw')

# ``load_units`` marker -> (force scale to lbf, moment scale to in*lb).
# The four entries are the OGI's four engineering-unit settings plus the
# chain's own native system:
#   'N'    N     / N*m      (Freestream default; the ATE app's label)
#   'kg'   kgf   / kgf*m
#   'lbft' lbf   / lbf*ft   <- the OGI's "Lb and Lbft" setting
#   'lb'   lbf   / in*lbf   <- already native; the historical assumption
_UNIT_SCALES = {
    'n':    (N_TO_LBF, NM_TO_INLB),
    'kg':   (KGF_TO_N * N_TO_LBF, KGF_TO_N * NM_TO_INLB),
    'lbft': (1.0, 12.0),
    'lb':   (1.0, 1.0),
}

# marker values that flag newton-based (SI) resolved loads
_SI_MARKERS = ('n', 'si', 'mks', 'newton', 'newtons', 'n*m', 'nm')


def _unit_marker(raw_data: Dict[str, Any]) -> str:
    """Normalized ``load_units`` marker, or '' when absent/unknown."""
    marker = raw_data.get('load_units') if hasattr(raw_data, 'get') else None
    if isinstance(marker, bytes):
        marker = marker.decode('utf-8', errors='replace')
    if isinstance(marker, np.ndarray):
        marker = marker.item() if marker.size == 1 else ''
    if not isinstance(marker, str):
        return ''
    m = marker.strip().lower()
    if m in _SI_MARKERS:
        return 'n'
    return m if m in _UNIT_SCALES else ''


def external_loads_in_si(raw_data: Dict[str, Any]) -> bool:
    """
    True when a raw channel dict carries a ``load_units`` marker flagging
    newton-based resolved loads (as written by Freestream: the
    ATE_Balance channel unit attribute is 'N' / 'N*m').

    The marker is propagated from ``RawData.properties['load_units']``
    by the directory loaders. Dicts without a marker are assumed to
    already be in the chain's native lb / in-lb (legacy behavior).
    """
    return _unit_marker(raw_data) == 'n'


def external_loads_to_ips(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert resolved external-balance loads to the chain's native units:
    lbf for Lift/Drag/Side and in*lb for Roll/Pitch/Yaw.

    The historical reduction works in lb / in-lb with Q in psi and
    S, C in inches (deprecated/scripts/calc_coeffs.m 'External':
    ``d.Units = {'lb' 'lb' 'lb' 'in-lb' 'in-lb' 'in-lb'}``), so a stream
    in any other system must be converted before q normalization.

    The OGI's unit setting is operator-selectable and is not carried on
    the wire, so the conversion is driven entirely by the recorded
    ``load_units`` marker (see :data:`_UNIT_SCALES`). Note that the
    OGI's pound setting pairs lbf with lbf*FT, not in*lbf — a x12
    difference on every moment coefficient that the old
    "not-SI-so-pass-it-through" rule silently swallowed. An absent or
    unrecognised marker still passes through untouched (the chain's own
    native units), preserving the legacy behaviour.

    Returns a shallow copy with the six load channels converted (other
    channels are shared by reference); the input dict is not modified.
    """
    marker = _unit_marker(raw_data)
    if not marker or marker == 'lb':
        return raw_data

    f_scale, m_scale = _UNIT_SCALES[marker]
    converted = dict(raw_data)
    for ch in _FORCE_CHANNELS:
        if ch in converted:
            converted[ch] = np.asarray(converted[ch], dtype=float) * f_scale
    for ch in _MOMENT_CHANNELS:
        if ch in converted:
            converted[ch] = np.asarray(converted[ch], dtype=float) * m_scale
    converted['load_units'] = 'lb'
    return converted


# ---------------------------------------------------------------------------
# Span-aware resolution of the balance channels into wind-axis loads
# ---------------------------------------------------------------------------

def resolve_external_wrf(raw_data: Dict[str, Any],
                         alpha_deg: Any = None,
                         span_config: Any = SPAN_FULL,
                         n_samples: Optional[int] = None) -> WRFForces:
    """
    Turn the six resolved ATE channels into wind-axis model loads.

    Which channel carries which model load depends on how the model is
    mounted, because the balance yaws with the model but never pitches
    with it (AID-010-10015-1 §3.2).

    ``span_config='full'`` (the MATLAB 'Horizontal' mount)
        Alpha comes from the incidence strut above the balance, so the
        balance stays level and its channels ARE the wind-axis loads.
        Straight pass-through — identical to
        :func:`~.transforms.wrf_from_resolved_loads`.

    ``span_config='half'`` (the MATLAB 'Vertical' mount)
        The semispan model stands on the turntable and alpha is the YAW
        drive, so the balance rotates with the model. Its vertical
        channel points along the model SPAN and its two horizontal
        channels are body-fixed::

            Lift  =  Side*cos(a) - Drag*sin(a)
            Drag  =  Drag*cos(a) + Side*sin(a)
            Side  =  Lift
            Roll  =  Roll
            Pitch =  Yaw          (model pitch is about the vertical axis)
            Yaw   =  Pitch

        The three force lines are the coefficient definitions written in
        ``calc_uncertainty_Extbalance.m`` ('Vertical'); the moment
        permutation is the same axis relabelling, and is what makes that
        file's 'Cmy' case read the Yaw column.

    Parameters
    ----------
    raw_data : dict
        Raw channel dict holding the six resolved load channels, already
        in consistent units (run it through :func:`external_loads_to_ips`
        first).
    alpha_deg : array-like, optional
        Angle of attack in degrees, per-sample or scalar. Required for
        the half-span rotation; ignored for full span. Missing alpha is
        treated as zero.
    span_config : str
        ``'full'`` / ``'half'`` (or the MATLAB mount names); anything
        unrecognised falls back to ``'full'``.
    n_samples : int, optional
        Sample count used to zero-fill missing channels.

    Returns
    -------
    WRFForces
        Wind-axis loads about the balance's virtual centre.
    """
    span = normalize_span_config(span_config)
    if span == SPAN_FULL:
        return wrf_from_resolved_loads(raw_data, n_samples=n_samples)

    # ½ span: start from the pass-through stack so channel presence,
    # zero-fill and length handling stay in one place, then relabel.
    ch = wrf_from_resolved_loads(raw_data, n_samples=n_samples)

    if alpha_deg is None:
        alpha_deg = raw_data.get('Alpha', 0.0) if hasattr(raw_data, 'get') \
            else 0.0
    a = np.deg2rad(np.atleast_1d(np.asarray(alpha_deg, dtype=float)))
    n = len(np.atleast_1d(ch.Drag))
    if a.size == 1:
        a = np.broadcast_to(a, (n,))
    elif a.size != n:                     # tare taken at a different length
        a = np.resize(a, n)
    ca, sa = np.cos(a), np.sin(a)

    out = WRFForces()
    out.Lift = ch.Side * ca - ch.Drag * sa
    out.Drag = ch.Drag * ca + ch.Side * sa
    out.Side = ch.Lift
    out.Roll = ch.Roll
    out.Pitch = ch.Yaw
    out.Yaw = ch.Pitch
    return out


def transfer_external_loads_to_mrc(wrf: WRFForces,
                                   alpha: np.ndarray,
                                   beta: np.ndarray,
                                   mshift: Sequence[float],
                                   span_config: Any = SPAN_FULL) -> WRFForces:
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
    span_config : str
        ``'full'`` / ``'half'``. On the ½-span mount the body forces
        come straight back out of the alpha rotation instead of a 3x3
        solve, and the arm terms are applied in the same model body
        axes; ``mshift`` means the same thing on both mounts (a shift
        of the model's moment reference centre).

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

    if normalize_span_config(span_config) == SPAN_HALF:
        # ½ span: no 3x3 inverse is needed, because the resolution in
        # resolve_external_wrf is itself a plane rotation by alpha plus
        # a fixed relabelling. Undo it to get the model body forces:
        #   Fx (axial)  = Drag*cos(a) - Lift*sin(a)   (the Drag channel)
        #   Fy (side)   = Side                        (the Lift channel)
        #   Fz (normal) = Lift*cos(a) + Drag*sin(a)   (the Side channel)
        Fx = wrf.Drag * ca - wrf.Lift * sa
        Fy = np.broadcast_to(np.atleast_1d(wrf.Side), (n,))
        Fz = wrf.Lift * ca + wrf.Drag * sa
        out = WRFForces()
        out.Lift, out.Drag, out.Side = wrf.Lift, wrf.Drag, wrf.Side
        out.Roll = wrf.Roll - Fy * mz
        out.Pitch = wrf.Pitch - Fz * mx - Fx * mz
        out.Yaw = wrf.Yaw + Fy * my - Fy * mx
        return out

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
                                 span_config: Any = None,
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
    source squares the area bias before use); bC = .005.

    Two deliberate DEVIATIONS from the source, both in the moment
    cases and both documented inline: the 'Cmy' block's bias term is
    taken from the channel its influence coefficients actually read,
    and 'Cmz' reads the Pitch channel instead of repeating 'Cmy'"s Yaw
    column. See :func:`resolve_external_wrf` for why those two
    coefficients read the "wrong-looking" channels in the first place.
    """
    # span_config, when given, is authoritative; `config` stays as the
    # historical MATLAB mount name so existing callers keep working.
    mount = (SPAN_TO_MOUNT[normalize_span_config(span_config)]
             if span_config is not None else config)
    if mount == 'Horizontal':
        return _uncertainty_full_span(coeffs, loads, Q, S, C,
                                      cal_bias, prec or {})
    if mount != 'Vertical':
        raise ValueError(
            f"unknown mounting config {mount!r} — expected 'Vertical' "
            f"(½ span) or 'Horizontal' (full span)")

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
            # Cmy = model pitching moment / QSC. On the Vertical mount
            # that moment is about the VERTICAL axis, so it is the
            # balance's Yaw channel — fb[:, 5] — which is why the source
            # reads fb(:,6) here under a comment saying "Cmy = Mz/QSC".
            # DEVIATION: the source then pairs it with bfb(5) (the Pitch
            # channel's bias); that mismatch is a typo, and the bias of
            # the channel actually being read is used instead.
            ic['pCpMy'] = 1.0 / Q / S / C
            ic['pCpQ'] = -0.5 * fb[:, 5] / (Q ** 2) / S / C
            ic['pCpS'] = -0.5 * fb[:, 5] / (S ** 2) / Q / C
            ic['pCpC'] = -0.5 * fb[:, 5] / (C ** 2) / Q / S
            bt = np.sqrt((ic['pCpMy'] * bfb[5]) ** 2
                         + (ic['pCpQ'] * bQ) ** 2
                         + (ic['pCpC'] * bC) ** 2
                         + (ic['pCpS'] * bS) ** 2)
            bias[name] = {'My': bfb[5], 'Q': bQ, 'S': bS, 'C': bC,
                          'total': bt}

        elif case == 'Cmz':
            # Cmz = model yawing moment / QSC. Vertical-mount partner of
            # the case above: the model's yaw axis is horizontal, so it
            # is the balance's PITCH channel, fb[:, 4].
            # DEVIATION: the source reads fb(:,6) here too — a
            # copy-paste of the Cmy block that would make Cmy and Cmz
            # share one channel and leave the Pitch channel unused.
            ic['pCpMz'] = 1.0 / Q / S / C
            ic['pCpQ'] = -0.5 * fb[:, 4] / (Q ** 2) / S / C
            ic['pCpS'] = -0.5 * fb[:, 4] / (S ** 2) / Q / C
            ic['pCpC'] = -0.5 * fb[:, 4] / (C ** 2) / Q / S
            bt = np.sqrt((ic['pCpMz'] * bfb[4]) ** 2
                         + (ic['pCpQ'] * bQ) ** 2
                         + (ic['pCpC'] * bC) ** 2
                         + (ic['pCpS'] * bS) ** 2)
            bias[name] = {'Mz': bfb[4], 'Q': bQ, 'S': bS, 'C': bC,
                          'total': bt}

        else:
            continue  # non-coefficient / pressure fields: no bias case

        inf_coeffs[name] = ic
        p = np.asarray(prec.get(name, 0.0), dtype=float)
        total[name] = np.sqrt(bt ** 2 + p ** 2)

    return {'InfCoeffs': inf_coeffs, 'bias': bias, 'total': total}


@np.errstate(divide='ignore', invalid='ignore')
def _uncertainty_full_span(coeffs: Dict[str, np.ndarray],
                           loads: np.ndarray,
                           Q: np.ndarray,
                           S: float,
                           C: float,
                           cal_bias: np.ndarray,
                           prec: Dict[str, np.ndarray]
                           ) -> Dict[str, Dict[str, Any]]:
    """
    Bias / total uncertainty for the full-span ('Horizontal') mount,
    which ``calc_uncertainty_Extbalance.m`` left as a ``disp('Horizontal
    config not implemented')`` stub.

    On this mount the balance never tilts with the model, so each
    coefficient reads one channel with no attitude term::

        Cl = Lift/QS   Cd = Drag/QS   Cs = Side/QS
        CRoll = Roll/QSC   CPitch = Pitch/QSC   CYaw = Yaw/QSC

    which drops the ``pCpa`` sensitivity that only the Vertical mount's
    alpha rotation creates. Constants (bQ, bS, bC) match the Vertical
    branch exactly.
    """
    fb = np.atleast_2d(np.asarray(loads, dtype=float))
    bfb = np.asarray(cal_bias, dtype=float)
    Q = np.asarray(Q, dtype=float)

    bQ = 0.0005 * np.mean(Q)
    bS = 0.005 ** 2
    bC = 0.005

    # coefficient -> (column in EXTERNAL_CHANNEL_ORDER, partial name,
    #                 is it a moment?)
    plan = {'Cl': (2, 'pCpFz', False), 'Cd': (0, 'pCpFx', False),
            'Cs': (1, 'pCpFy', False), 'CRoll': (3, 'pCpMx', True),
            'CPitch': (4, 'pCpMy', True), 'CYaw': (5, 'pCpMz', True)}

    inf_coeffs: Dict[str, Dict[str, np.ndarray]] = {}
    bias: Dict[str, Dict[str, Any]] = {}
    total: Dict[str, np.ndarray] = {}

    for name in coeffs:
        entry = plan.get(name) or plan.get(
            {v: k for k, v in _MOMENT_CASE_MAP.items()}.get(name, ''))
        if entry is None:
            continue                     # pressure / non-coefficient field
        col, partial, is_moment = entry
        denom_c = C if is_moment else 1.0

        ic: Dict[str, np.ndarray] = {
            partial: 1.0 / Q / S / denom_c,
            'pCpQ': -0.5 * fb[:, col] / (Q ** 2) / S / denom_c,
            'pCpS': -0.5 * fb[:, col] / (S ** 2) / Q / denom_c,
        }
        terms = [(ic[partial] * bfb[col]) ** 2,
                 (ic['pCpQ'] * bQ) ** 2,
                 (ic['pCpS'] * bS) ** 2]
        entry_bias: Dict[str, Any] = {'Q': bQ, 'S': bS,
                                      partial[3:]: bfb[col]}
        if is_moment:
            ic['pCpC'] = -0.5 * fb[:, col] / (C ** 2) / Q / S
            terms.append((ic['pCpC'] * bC) ** 2)
            entry_bias['C'] = bC

        bt = np.sqrt(sum(terms))
        entry_bias['total'] = bt
        inf_coeffs[name] = ic
        bias[name] = entry_bias
        p = np.asarray(prec.get(name, 0.0), dtype=float)
        total[name] = np.sqrt(bt ** 2 + p ** 2)

    return {'InfCoeffs': inf_coeffs, 'bias': bias, 'total': total}


def build_load_matrix(wrf: WRFForces) -> np.ndarray:
    """
    Stack WRF loads into an (n, 6) matrix in the external calibration
    channel order (Drag, Side, Lift, Roll, Pitch, Yaw — calc_coeffs.m
    d.Forcechan), the layout expected by
    :func:`calc_uncertainty_ext_balance`.

    Correct for a full-span run, where the wind-axis loads and the
    balance channels are the same six numbers. On a ½-span run they are
    not (see :func:`resolve_external_wrf`) and the uncertainty needs the
    RAW channels — use :func:`build_load_matrix_from_channels` there,
    which is what ``fb = d.red(1).BRF.Elems`` meant in the MATLAB.
    """
    cols = [np.atleast_1d(getattr(wrf, ch)) for ch in EXTERNAL_CHANNEL_ORDER]
    n = max(len(c) for c in cols)
    cols = [np.broadcast_to(c, (n,)) if c.size == 1 else c[:n] for c in cols]
    return np.column_stack(cols)


def build_load_matrix_from_channels(raw_data: Dict[str, Any],
                                    n_samples: Optional[int] = None
                                    ) -> np.ndarray:
    """
    Stack the RAW balance channels into the (n, 6) matrix in
    ``EXTERNAL_CHANNEL_ORDER``, mount-independent.

    This is the direct analogue of the MATLAB ``d.red(1).BRF.Elems``:
    what the six load cells read, before any mount-dependent relabelling
    or alpha rotation.
    """
    arrays = []
    for ch in EXTERNAL_CHANNEL_ORDER:
        value = raw_data.get(ch)
        arrays.append(None if value is None
                      else np.atleast_1d(np.asarray(value, dtype=float)))
    if n_samples is None:
        lengths = [len(a) for a in arrays if a is not None]
        n_samples = max(lengths) if lengths else 0
    cols = [np.zeros(n_samples) if a is None
            else (np.broadcast_to(a, (n_samples,)) if a.size == 1
                  else np.resize(a, n_samples))
           for a in arrays]
    return np.column_stack(cols) if cols else np.zeros((0, 6))
