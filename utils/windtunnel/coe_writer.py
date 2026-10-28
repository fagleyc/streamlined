"""
COE File Writer
===============

Write reduced wind-tunnel data to the legacy Reduce2 .COE format so the
output can be opened by the existing Excel post-processing tool (which
produces stability-derivative plots, blockage corrections, and static
margin).

Format reference: see brandt/5gat_Atail_B0.COE for an example.

Each .COE file contains one alpha sweep at a fixed beta and a fixed
tunnel speed. A Streamlined case that steps through several betas or
several Mach numbers is therefore exported as one .COE file per
(Mach step, beta), grouped on the COMMANDED values (case.point_machs,
case.point_betas), the same keys the GUI filters and plots on.

Author: C. Fagley
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# Unit conversions used by the COE format (legacy IPS / Imperial)
PSI_TO_PA = 6894.75729
PA_TO_PSI = 1.0 / PSI_TO_PA
MS_TO_FTS = 3.28084           # m/s -> ft/s
DEGC_TO_DEGF_FACTOR = 9.0 / 5.0


def _c_to_f(t_c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return t_c * DEGC_TO_DEGF_FACTOR + 32.0


def _safe_mean(arr) -> float:
    """Mean of an array-like; returns 0.0 if empty / None."""
    if arr is None:
        return 0.0
    try:
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            return 0.0
        return float(np.mean(a))
    except Exception:
        return 0.0


def _extract_point_row(pt, geo_chord_in: float) -> Dict[str, float]:
    """
    Build the 35 COE columns for one ReducedDataPoint.

    Quantities come from the per-point tunnel conditions, body-frame
    forces, and aero coefficients on the ReducedDataPoint object.
    """
    alpha = _safe_mean(getattr(pt, 'alpha', None))
    beta = _safe_mean(getattr(pt, 'beta', None))

    tc = getattr(pt, 'tunnel', None)
    Ma = _safe_mean(getattr(tc, 'Mach', None)) if tc else 0.0
    Re = _safe_mean(getattr(tc, 'Re', None)) if tc else 0.0
    T0_C = _safe_mean(getattr(tc, 'T0', None)) if tc else 0.0
    T_C = _safe_mean(getattr(tc, 'T', None)) if tc else 0.0
    a_ms = _safe_mean(getattr(tc, 'a', None)) if tc else 0.0
    U_ms = _safe_mean(getattr(tc, 'U_inf', None)) if tc else 0.0
    P0_pa = _safe_mean(getattr(tc, 'P_tot', None)) if tc else 0.0
    Pstatic_pa = _safe_mean(getattr(tc, 'P_static', None)) if tc else 0.0
    q_psi = _safe_mean(getattr(tc, 'Q', None)) if tc else 0.0

    # pDiff = P0 - P_static measured by the differential transducer.
    # For incompressible flow this equals q; for compressible flow it
    # is slightly higher. Compute directly from the static / total
    # pressures so the COE file carries the raw measurement.
    if P0_pa > 0 and Pstatic_pa > 0:
        pdiff_psi = (P0_pa - Pstatic_pa) * PA_TO_PSI
    else:
        pdiff_psi = q_psi

    # Body-frame raw elements (pre-tare; mean of air-on minus mean of air-off)
    brf_on = getattr(pt, 'brf_on', None)
    brf_off = getattr(pt, 'brf_off', None)

    def _elem(idx):
        if brf_on is None:
            return 0.0
        on_el = getattr(brf_on, 'elements', None)
        if on_el is None or on_el.ndim != 2 or on_el.shape[1] <= idx:
            return 0.0
        on_mean = float(np.mean(on_el[:, idx]))
        if brf_off is not None:
            off_el = getattr(brf_off, 'elements', None)
            if (off_el is not None and off_el.ndim == 2
                    and off_el.shape[1] > idx):
                on_mean -= float(np.mean(off_el[:, idx]))
        return on_mean

    N1 = _elem(0)
    N2 = _elem(1)
    Y1 = _elem(2)
    Y2 = _elem(3)
    Ax = _elem(4)
    Roll_elem = _elem(5)  # raw roll element; not directly used below

    def _brf(attr):
        """Aerodynamic BRF component: air-on minus mean of air-off."""
        if brf_on is None:
            return 0.0
        on_arr = getattr(brf_on, attr, None)
        if on_arr is None or len(on_arr) == 0:
            return 0.0
        val = float(np.mean(on_arr))
        if brf_off is not None:
            off_arr = getattr(brf_off, attr, None)
            if off_arr is not None and len(off_arr) > 0:
                val -= float(np.mean(off_arr))
        return val

    Fz = _brf('Fz')
    Fy = _brf('Fy')
    Fx = _brf('Fx')
    Mx = _brf('Mx')
    My = _brf('My')
    Mz = _brf('Mz')

    # Aerodynamic coefficients - wind frame (from pt.coeffs, post-tare)
    coeffs = getattr(pt, 'coeffs', None)
    Cl = _safe_mean(getattr(coeffs, 'Cl', None)) if coeffs else 0.0
    Cd = _safe_mean(getattr(coeffs, 'Cd', None)) if coeffs else 0.0
    CRoll = _safe_mean(getattr(coeffs, 'CRoll', None)) if coeffs else 0.0
    CPitch = _safe_mean(getattr(coeffs, 'CPitch', None)) if coeffs else 0.0
    CYaw = _safe_mean(getattr(coeffs, 'CYaw', None)) if coeffs else 0.0
    LD = Cl / Cd if abs(Cd) > 1e-10 else 0.0

    # Body-frame force/moment coefficients (non-dim by qS, qSc)
    qS = q_psi  # numerator: forces in lbf, denominator: q [psi] * S [in^2]
    # We will compute these with proper qS later when we have S - placeholder
    # via passed geo_chord_in. The caller normalizes.
    return {
        'Alpha': alpha,
        'Beta': beta,
        'Ma': Ma,
        'Re': Re,
        'T0Stil_F': _c_to_f(T0_C),
        'Tinf_F': _c_to_f(T_C),
        'a_fts': a_ms * MS_TO_FTS,
        'Vinf_fts': U_ms * MS_TO_FTS,
        'p0Stil_psia': P0_pa * PA_TO_PSI,
        'pInf_psia': Pstatic_pa * PA_TO_PSI,
        'qInf_psi': q_psi,
        'pDiff_psi': pdiff_psi,
        'N1': N1,
        'N2': N2,
        'N': Fz,
        'Y1': Y1,
        'Y2': Y2,
        'Y': Fy,
        'Ax': Fx,
        'PiMom': My,
        'YaMom': Mz,
        'RoMom': Mx,
        # Body-frame force coefficients: F / (q * S)
        '_Fx_for_cn': Fz,
        '_Fy_for_cy': Fy,
        '_Ax_for_cax': Fx,
        '_My_for_cpi': My,
        '_Mz_for_cya': Mz,
        '_Mx_for_cro': Mx,
        # Wind-frame (post-tare) coefficients - already MRC-shifted by Streamlined
        'CLift': Cl,
        'CDrag': Cd,
        'LD': LD,
        'CPitchSh': CPitch,
        'CYawSh': CYaw,
        'CRollSh': CRoll,
    }


def _compute_body_coeffs(row: Dict[str, float], ref_area_in2: float,
                         ref_chord_in: float, ref_span_in: float) -> Dict[str, float]:
    """Compute body-frame coefficients given row values and reference geometry."""
    q = row['qInf_psi']
    qS = q * ref_area_in2
    qSc = qS * ref_chord_in
    qSb = qS * ref_span_in
    if abs(qS) < 1e-12:
        qS = 1e-12
    if abs(qSc) < 1e-12:
        qSc = 1e-12
    if abs(qSb) < 1e-12:
        qSb = 1e-12
    return {
        'CN': row['_Fx_for_cn'] / qS,
        'CY': row['_Fy_for_cy'] / qS,
        'CAx': row['_Ax_for_cax'] / qS,
        'CAxBD': 0.0,
        # Body-frame moment coefficients (about balance center, no MRC shift).
        # Since Streamlined's reduction applies MRC shift uniformly, we set
        # these equal to the shifted versions; if MRC shift is zero they are
        # identical anyway. This matches the legacy Reduce2 behavior in cases
        # where MomentXShift = MomentZShift = 0.
        'CPiMom': row['CPitchSh'],
        'CYaMom': row['CYawSh'],
        'CRoMom': row['CRollSh'],
    }


def _format_row(row: Dict[str, float], body: Dict[str, float]) -> str:
    """Format one TEST RUN data row as CSV.

    Each column uses a precision tuned to match the legacy Reduce2
    output, so any downstream parser that expects fixed decimal places
    is satisfied.  The column order matches the COE [TEST RUN] header.
    """
    # Column-specific precisions (decimal places).  None = scientific.
    cols = [
        (row['Alpha'],        2),  # Alpha
        (row['Beta'],         2),  # Beta
        (row['Ma'],           4),  # Mach
        (row['Re'],        'sci'), # Reynolds
        (row['T0Stil_F'],     4),  # T0 stil [degF]
        (row['Tinf_F'],       4),  # Tinf [degF]
        (row['a_fts'],        4),  # speed of sound [ft/s]
        (row['Vinf_fts'],     4),  # Vinf [ft/s]
        (row['p0Stil_psia'],  4),  # P0 stil [psia]
        (row['pInf_psia'],    4),  # pInf [psia]
        (row['qInf_psi'],     4),  # qInf [psi]
        (row['pDiff_psi'],    4),  # pDiff [psi]
        (row['N1'],           4),  # N1 [lb]
        (row['N2'],           4),  # N2 [lb]
        (row['N'],            4),  # N [lb]
        (row['Y1'],           4),  # Y1 [lb]
        (row['Y2'],           4),  # Y2 [lb]
        (row['Y'],            4),  # Y [lb]
        (row['Ax'],           4),  # Ax [lb]
        (row['PiMom'],        4),  # PiMom [in-lb]
        (row['YaMom'],        4),  # YaMom [in-lb]
        (row['RoMom'],        4),  # RoMom [in-lb]
        (body['CN'],          4),  # CN
        (body['CY'],          4),  # CY
        (body['CAx'],         4),  # CAx
        (body['CAxBD'],    'sci'), # CAxBD (legacy uses scientific)
        (body['CPiMom'],      4),  # CPiMom
        (body['CYaMom'],      4),  # CYaMom
        (body['CRoMom'],      4),  # CRoMom
        (row['CLift'],        4),  # CLift
        (row['CDrag'],        4),  # CDrag
        (row['LD'],           4),  # L/D
        (row['CPitchSh'],     4),  # CPitchSh
        (row['CYawSh'],       4),  # CYawSh
        (row['CRollSh'],      4),  # CRollSh
    ]
    return ', '.join(_fmt_num(v, p) for v, p in cols)


def _fmt_num(v: float, precision=4) -> str:
    """Format a number for the COE TEST RUN table.

    Parameters
    ----------
    v : float
        Value to format.
    precision : int or 'sci'
        Number of decimal places, or 'sci' for scientific notation
        (NNN.NNNe+NN).  Use 'sci' for columns like Reynolds number
        and CAxBD where the legacy file always uses scientific form.
    """
    if not np.isfinite(v):
        return '0' + ('.' + '0' * precision if isinstance(precision, int) else '.000e+00')
    if precision == 'sci':
        return f'{v:.3e}'
    return f'{v:.{precision}f}'


# -----------------------------------------------------------------------------
# Header block builders
# -----------------------------------------------------------------------------

def _build_header(case, beta_value: float,
                  case_geometry: Dict[str, Any],
                  balance_cal_file: str = '',
                  output_path: str = '',
                  mach_value: Optional[float] = None,
                  p_atm_psi: float = 0.0,
                  excluded: Optional[List[str]] = None) -> str:
    """Build the COE header sections (everything before [TEST RUN]).

    ``case_geometry`` must already be in inches (see _geometry_ips).
    ``p_atm_psi`` is the mean stilling-chamber pressure of the points in
    this file, the proxy for atmosphere the legacy header carries.
    """
    now = datetime.datetime.now()
    date_str = now.strftime('%m-%d-%Y')
    time_str = now.strftime('%H:%M:%S')

    mac_in = case_geometry.get('mac', 1.0)
    span_in = case_geometry.get('span', 1.0)
    ref_area_in2 = case_geometry.get('ref_area', 1.0)
    mrc = case_geometry.get('mrc', [0.0, 0.0, 0.0])
    mrc_x = float(mrc[0]) if len(mrc) > 0 else 0.0
    mrc_z = float(mrc[2]) if len(mrc) > 2 else 0.0

    # Balance arm distances - default values if not available
    dx1 = dx2 = 1.5
    dy1 = dy2 = 1.25
    daq = getattr(case, 'daq', None)
    if daq is not None:
        cal = getattr(daq, 'cal', None)
        if cal is not None:
            try:
                from .transforms import get_distance_values
                dists = get_distance_values(cal)
                dx1 = dists.get('dx1', dx1)
                dx2 = dists.get('dx2', dx2)
                dy1 = dists.get('dy1', dy1)
                dy2 = dists.get('dy2', dy2)
            except Exception:
                pass

    serial = ''
    if daq is not None and getattr(daq, 'cal', None) is not None:
        serial = getattr(daq.cal.description, 'serial_number', '') or ''

    lines = [
        'Reduce2 2.0',
        f'Coe Data Filename --> {output_path}',
        f'Date -->  {date_str}',
        f'Time -->  {time_str}',
        'Red Data Filename --> (none - generated by Streamlined)',
        '*****',
        '[Comment]',
        (f'Exported by Streamlined for case {case.name!r} at beta = '
         f'{beta_value:g} deg'
         + (f', Mach = {mach_value:.3f}' if mach_value is not None
            else '')),
        *[f'Point left out ({note})' for note in (excluded or [])],
        '',
        '[Conditions]',
        f'Atmosphere Pressure -->  {p_atm_psi:.2f} [psia]',
        'Stilling Chamber Temperature --> In Column',
        'Machnumber -->  In Column',
        '',
        '[Force/Moment Balance]',
        f'Calibration Voltage Filename: --> {balance_cal_file}',
        f'Moment Calibration File (Calibration Slopes) --> {balance_cal_file}',
        f'Date --> {date_str}',
        f'Serial Number --> {serial}',
        f'Distance from balance center to N1 --> {dx1:.3f} [in]',
        f'Distance from balance center to N2 --> {dx2:.3f} [in]',
        f'Distance from balance center to Y1 --> {dy1:.3f} [in]',
        f'Distance from balance center to Y2 --> {dy2:.3f} [in]',
        '',
        '[Reference Length]',
        f'ReferenceLength --> {mac_in:.2f} [in]',
        f'ReferenceArea   --> {ref_area_in2:.2f} [in^2]',
        f'MeanAreaChord   --> {mac_in:.2f} [in]',
        f'SpanWidth       --> {span_in:.2f} [in]',
        f'MomentXShift    --> {mrc_x:.2f} [in]',
        f'MomentZShift    --> {mrc_z:.2f} [in]',
        '',
        '[Alpha Effective]',
        'Alpha OffSet --> 0 [ deg]',
        '',
        '',
        '[TEST RUN]',
        ('Alpha, Beta, Machnumber, Reynolds Number, , TInf, a, Vinf, '
         'Absolute pressure gage, pInf, qInf, Differential pressue gage, '
         'Task, Task, N, Task, Task, Y, Task, Pitching Moment, '
         'Yawing Moment, Task, CN, CY, CAx, CAxBD, CPiMom, CYaMom, CRoMom, '
         'CLift, CDrag, Lift/Drag, CPiMomSh, CYaMomSh, CRoMomSh'),
        ('Alpha, Beta, Ma, Re, T0Stil, Tinf, a, Vinf, p0Stil, pInf, qInf, '
         'pDiff, N1, N2, N, Y1, Y2, Y, Ax, PiMom, YaMom, RoMom, CN, CY, CAx, '
         'CAxBD, CPiMom, CYaMom, CRoMom, CLift, CDrag, L/D, CPiMomSh, '
         'CYaMomSh, CRoMomSh'),
        ('[ \xb0 ], [ \xb0 ], [-], [-], [\xb0F], [\xb0F], [ft\\sec], '
         '[ft\\sec], [psia], [psia], [psia], [psia], [lb], [lb], [lb], '
         '[lb], [lb], [lb], [lb], [in-lb], [in-lb], [in-lb], [-], [-], '
         '[-], [-], [-], [-], [-], [-], [-], [-], [-], [-], [-]'),
    ]
    return '\n'.join(lines)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def _geometry_ips(case_geometry: Dict[str, Any]) -> Dict[str, Any]:
    """Reference geometry in inches, whatever units it was entered in.

    The geometry dialog takes MKS or IPS (geo['units']); the COE format
    and the q[psi]*S[in^2] normalisation below need inches.
    """
    geo = dict(case_geometry or {})
    units = geo.get('units') or 'IPS'
    try:
        from .units import geometry_to_ips
    except Exception:                                      # noqa: BLE001
        return geo

    def _len(v):
        return float(geometry_to_ips(float(v), 'length', units))

    for key in ('mac', 'span'):
        if key in geo and geo[key] is not None:
            geo[key] = _len(geo[key])
    if geo.get('ref_area') is not None:
        geo['ref_area'] = float(geometry_to_ips(
            float(geo['ref_area']), 'area', units))
    if geo.get('mrc') is not None:
        geo['mrc'] = [_len(v) for v in geo['mrc']]
    geo['units'] = 'IPS'
    return geo


def _per_point(case, attr: str, n: int) -> Optional[np.ndarray]:
    """case.<attr> flattened to one value per reduced point, or None.

    case arrays and case.daq.red are in the same point order (the
    reduction builds one from the other); anything not exactly one
    value per point is unusable as a key.
    """
    try:
        a = np.asarray(getattr(case, attr), dtype=float).ravel()
    except Exception:                                      # noqa: BLE001
        return None
    return a if a.size == n else None


def _invalid_points(case, n: int) -> Dict[int, str]:
    """{point index: reason} for points whose coefficients are not valid.

    The same checks DataController._tunnel_off_points reports after a
    load: measured Mach below half a Mach command (tunnel off or spinning
    down), or the positioner more than 1 deg from the commanded angle.
    Streamlined keeps such points and shows the note; a .COE row has no
    flag, and the Excel tool and coe_postprocess.py differentiate along
    the sweep, so they are left out of the rows and named in [Comment].
    """
    out: Dict[int, str] = {}

    def _note(i: int, why: str) -> None:
        out[i] = f'{out[i]}; {why}' if i in out else why

    for axis, nom_attr, meas_attr in (('alpha', 'alpha_nominal', 'alphas'),
                                      ('beta', 'beta_nominal', 'betas')):
        nom = _per_point(case, nom_attr, n)
        meas = _per_point(case, meas_attr, n)
        if nom is None or meas is None:
            continue
        for i in np.flatnonzero(np.abs(meas - nom) > 1.0):
            _note(int(i), f'positioner off its point: measured {axis} '
                          f'{meas[i]:.2f} deg, commanded {nom[i]:g}')
    if str(getattr(case, 'speed_unit', '')).strip().lower() == 'mach':
        cmd = _per_point(case, 'speeds', n)
        meas = _per_point(case, 'machs', n)
        if cmd is not None and meas is not None:
            with np.errstate(invalid='ignore'):
                low = np.isfinite(cmd) & (cmd > 0) & (meas < 0.5 * cmd)
            for i in np.flatnonzero(low):
                _note(int(i), f'tunnel off or spinning down: measured Mach '
                              f'{meas[i]:.3f} at a {cmd[i]:g} command')
    return out


def _num_tag(v: float) -> str:
    """Filename-safe number: 0 -> '0', -5 -> '-5', 2.5 -> '2p5', 0.25 -> '0p25'."""
    v = round(float(v), 3)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f'{v:g}'.replace('.', 'p')


def _group_points(case, red, beta_tolerance: float):
    """Group reduced points into COE files: {(mach, beta, leg): [idx]}.

    Keys are the COMMANDED values the GUI uses (case.point_machs,
    case.point_betas). Without them the measured beta is bucketed by
    beta_tolerance and every point shares one Mach group, which is the
    old behaviour. A hysteresis run (case.sweep_dirs 'up'/'dn') splits
    into its two legs, because one file is one monotonic alpha sweep.
    """
    n = len(red)
    machs = _per_point(case, 'point_machs', n)
    betas = _per_point(case, 'point_betas', n)
    try:
        dirs = np.asarray(getattr(case, 'sweep_dirs', [])).ravel()
    except Exception:                                      # noqa: BLE001
        dirs = np.array([])
    if dirs.size != n:
        dirs = np.array([''] * n)

    groups: Dict[Tuple, List[int]] = {}
    for i, pt in enumerate(red):
        if betas is not None and np.isfinite(betas[i]):
            beta = round(float(betas[i]), 2)
        else:
            measured = _safe_mean(getattr(pt, 'beta', None))
            beta = round(round(measured / beta_tolerance)
                         * beta_tolerance, 2)
        mach = None
        if machs is not None and np.isfinite(machs[i]):
            mach = round(float(machs[i]), 3)
        leg = str(dirs[i] or '').strip().lower()
        groups.setdefault((mach, beta, leg), []).append(i)

    # A leg tag only means something where both legs exist at that
    # (Mach, beta); a lone leg is just the sweep.
    merged: Dict[Tuple, List[int]] = {}
    for (mach, beta, leg), idx in groups.items():
        legs = {k[2] for k in groups if k[0] == mach and k[1] == beta}
        if not ({'up', 'dn'} <= legs):
            leg = ''
        merged.setdefault((mach, beta, leg), []).extend(idx)
    return merged


def write_coe_files(case, output_dir: str,
                    case_geometry: Dict[str, Any],
                    balance_cal_file: str = '',
                    beta_tolerance: float = 0.5) -> List[str]:
    """
    Write one .COE file per (Mach step, beta) for the given case.

    Parameters
    ----------
    case : TestCase
        Streamlined case with raw DAQ data attached (case.daq.red).
    output_dir : str
        Directory to write the .COE files into.
    case_geometry : dict
        Geometry dict (mac, ref_area, span, mrc, units). Converted to
        inches here when it was entered in MKS.
    balance_cal_file : str
        Path to the .vol calibration file (for the COE header).
    beta_tolerance : float
        Degrees within which MEASURED betas are grouped into the same
        file; only used when the case has no commanded beta.

    Returns
    -------
    list of str
        The full paths of the .COE files that were written.

    Files are named <case>_b<beta>.COE; when the case holds more than
    one Mach step the step is added, <case>_M0p30_b0.COE, and a
    hysteresis run adds _up / _dn.
    """
    daq = getattr(case, 'daq', None)
    red = getattr(daq, 'red', None) if daq is not None else None
    if not red:
        raise ValueError(
            f"Case '{case.name}' has no raw DAQ data attached; cannot "
            "export to COE (the legacy format requires per-point "
            "tunnel conditions and balance elements).")

    # Sanitize name for file system
    safe_name = case.name
    for ch in '/\\:*?"<>|':
        safe_name = safe_name.replace(ch, '_')
    safe_name = safe_name.strip()
    if not safe_name:
        safe_name = 'case'

    geo = _geometry_ips(case_geometry)
    mac_in = geo.get('mac', 1.0)
    ref_area_in2 = geo.get('ref_area', 1.0)
    span_in = geo.get('span', 1.0)

    groups = _group_points(case, red, beta_tolerance)
    multi_mach = len({k[0] for k in groups}) > 1
    alphas_cmd = _per_point(case, 'point_alphas', len(red))
    invalid = _invalid_points(case, len(red))

    out_paths = []
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _order(key):
        mach, beta, leg = key
        return (mach if mach is not None else -1.0, beta, leg != 'up', leg)

    for key in sorted(groups, key=_order):
        mach, beta, leg = key
        excluded = []
        for i in groups[key]:
            if i in invalid:
                a = alphas_cmd[i] if alphas_cmd is not None else np.nan
                excluded.append(f'alpha {a:g}: {invalid[i]}')
        idx = [i for i in groups[key] if i not in invalid]
        if not idx:
            continue

        # Per-point rows, sorted on the commanded alpha (measured as a
        # tie-break) so the sweep is monotonic; a down leg runs downward.
        rows = []
        for i in idx:
            row = _extract_point_row(red[i], mac_in)
            body = _compute_body_coeffs(
                row, ref_area_in2, mac_in, span_in)
            a_key = (alphas_cmd[i] if alphas_cmd is not None
                     and np.isfinite(alphas_cmd[i]) else row['Alpha'])
            rows.append(((a_key, row['Alpha']), row, body))
        rows.sort(key=lambda r: r[0], reverse=(leg == 'dn'))

        tags = []
        if multi_mach and mach is not None:
            # two decimals like the acquisition's M0p25 token, three
            # only when the step needs them
            m = f'{mach:.2f}' if abs(mach - round(mach, 2)) < 1e-9 \
                else f'{mach:.3f}'
            tags.append(f"_M{m.replace('.', 'p')}")
        tags.append(f'_b{_num_tag(beta)}')
        if leg:
            tags.append(f'_{leg}')
        fname = f"{safe_name}{''.join(tags)}.COE"
        full_path = out_dir / fname

        p0 = [r[1]['p0Stil_psia'] for r in rows if r[1]['p0Stil_psia'] > 0]
        header = _build_header(
            case, beta, geo,
            balance_cal_file=balance_cal_file,
            output_path=str(full_path),
            mach_value=mach,
            p_atm_psi=float(np.mean(p0)) if p0 else 0.0,
            excluded=excluded)

        body_lines = [_format_row(r, b) for _, r, b in rows]
        content = header + '\n' + '\n'.join(body_lines) + '\n'
        full_path.write_text(content, encoding='utf-8')
        out_paths.append(str(full_path))

    return out_paths
