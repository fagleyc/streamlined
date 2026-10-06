"""
Plot Variables
==============

The ONE registry of quantities the plot can put on an axis: each entry
says what group it belongs to (coefficients, wind-axis loads, body-axis
loads, balance elements, tunnel conditions, attitude, stability
derivatives), what physical KIND it is (so dimensional data is converted
into the output unit system), and how to label the axis.

The axis selector builds its lists from here, the plot panel converts and
labels from here, and TestCase.get_coefficient resolves the keys - so a
new quantity is added in one place.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PlotVar:
    key: str            # TestCase.get_coefficient / PlotPanel name
    label: str          # menu text
    group: str          # menu section
    kind: str           # 'coeff' | 'force' | 'moment' | 'pressure' |
                        # 'velocity' | 'density' | 'temperature' | 'angle'
                        # | 'none' (dimensionless, not a coefficient)
    axis: str           # axis label; '{}' is filled with the unit
    tip: str = ""


COEFFICIENTS = "Coefficients"
WIND = "Wind-axis loads"
BODY = "Body-axis loads"
ELEMENTS = "Balance elements"
TUNNEL = "Tunnel conditions"
ATTITUDE = "Attitude"
DERIVATIVES = "Stability derivatives"
CUSTOM = "Calculator outputs"

GROUP_ORDER = (COEFFICIENTS, WIND, BODY, ELEMENTS, TUNNEL, ATTITUDE,
               DERIVATIVES, CUSTOM)

_VARS: Tuple[PlotVar, ...] = (
    # ── coefficients (wind axes) ──
    PlotVar("Cl", "CL  lift coefficient", COEFFICIENTS, "coeff", r"$C_L$"),
    PlotVar("Cd", "CD  drag coefficient", COEFFICIENTS, "coeff", r"$C_D$"),
    PlotVar("Cs", "CY  side-force coefficient", COEFFICIENTS, "coeff",
            r"$C_Y$"),
    PlotVar("CRoll", "Cl  rolling-moment coefficient", COEFFICIENTS, "coeff",
            r"$C_l$"),
    PlotVar("CPitch", "Cm  pitching-moment coefficient", COEFFICIENTS,
            "coeff", r"$C_m$"),
    PlotVar("CYaw", "Cn  yawing-moment coefficient", COEFFICIENTS, "coeff",
            r"$C_n$"),
    PlotVar("L/D", "L/D  lift-to-drag ratio", COEFFICIENTS, "none", "L/D"),
    # ── wind-axis loads (air-on minus tare, at the MRC) ──
    PlotVar("Lift", "Lift", WIND, "force", "Lift [{}]"),
    PlotVar("Drag", "Drag", WIND, "force", "Drag [{}]"),
    PlotVar("Side", "Side force", WIND, "force", "Side force [{}]"),
    PlotVar("RollMoment", "Rolling moment", WIND, "moment",
            "Rolling moment [{}]"),
    PlotVar("PitchMoment", "Pitching moment", WIND, "moment",
            "Pitching moment [{}]"),
    PlotVar("YawMoment", "Yawing moment", WIND, "moment",
            "Yawing moment [{}]"),
    # ── body-axis loads (the wind loads rotated back through alpha/beta) ──
    PlotVar("Fx", "Fx  axial force", BODY, "force", "$F_x$ [{}]",
            "Body-axis force along x"),
    PlotVar("Fy", "Fy  side force", BODY, "force", "$F_y$ [{}]"),
    PlotVar("Fz", "Fz  normal force", BODY, "force", "$F_z$ [{}]"),
    PlotVar("Mx", "Mx  rolling moment", BODY, "moment", "$M_x$ [{}]"),
    PlotVar("My", "My  pitching moment", BODY, "moment", "$M_y$ [{}]"),
    PlotVar("Mz", "Mz  yawing moment", BODY, "moment", "$M_z$ [{}]"),
    # ── balance elements (labels replaced by the recording balance's
    #    channel names, see element_vars) ──
    PlotVar("elem_N1", "Element 1", ELEMENTS, "force", "Element 1 [{}]"),
    PlotVar("elem_N2", "Element 2", ELEMENTS, "force", "Element 2 [{}]"),
    PlotVar("elem_Y1", "Element 3", ELEMENTS, "force", "Element 3 [{}]"),
    PlotVar("elem_Y2", "Element 4", ELEMENTS, "force", "Element 4 [{}]"),
    PlotVar("elem_Ax", "Element 5", ELEMENTS, "force", "Element 5 [{}]"),
    PlotVar("elem_Roll", "Element 6", ELEMENTS, "moment",
            "Element 6 [{}]"),
    # ── tunnel conditions ──
    PlotVar("Mach", "Mach", TUNNEL, "none", "Mach"),
    PlotVar("Q", "q  dynamic pressure", TUNNEL, "pressure", "q [{}]"),
    PlotVar("U_inf", "U∞  velocity", TUNNEL, "velocity",
            r"$U_\infty$ [{}]"),
    PlotVar("Re", "Re  Reynolds number", TUNNEL, "none", "Re"),
    PlotVar("rho", "ρ  density", TUNNEL, "density", r"$\rho$ [{}]"),
    PlotVar("T", "T  temperature", TUNNEL, "temperature", "T [{}]"),
    PlotVar("P0", "P0  total pressure", TUNNEL, "pressure", "$P_0$ [{}]"),
    # ── attitude ──
    PlotVar("Alpha", "α  angle of attack", ATTITUDE, "angle",
            r"$\alpha$ [deg]"),
    PlotVar("Beta", "β  sideslip", ATTITUDE, "angle",
            r"$\beta$ [deg]"),
    # ── stability derivatives (central difference, per deg) ──
    PlotVar("CLa", "CLα  lift-curve slope", DERIVATIVES, "none",
            r"$C_{L_\alpha}$ [1/deg]"),
    PlotVar("Cma", "Cmα  pitch stability", DERIVATIVES, "none",
            r"$C_{m_\alpha}$ [1/deg]"),
    PlotVar("StaticMargin", "Static margin  -Cmα/CLα",
            DERIVATIVES, "none", "Static margin"),
    PlotVar("CYb", "CYβ", DERIVATIVES, "none",
            r"$C_{Y_\beta}$ [1/deg]", "Needs at least two betas"),
    PlotVar("Cnb", "Cnβ  directional stability", DERIVATIVES, "none",
            r"$C_{n_\beta}$ [1/deg]", "Needs at least two betas"),
    PlotVar("Clb", "Clβ  lateral stability", DERIVATIVES, "none",
            r"$C_{l_\beta}$ [1/deg]", "Needs at least two betas"),
)

BY_KEY: Dict[str, PlotVar] = {v.key: v for v in _VARS}

#: element slot keys in balance-channel order
ELEMENT_KEYS = ("elem_N1", "elem_N2", "elem_Y1", "elem_Y2", "elem_Ax",
                "elem_Roll")

#: what an old Plot Type selection meant, as (x, y) - saved configurations
#: and the model's PlotType keep working
LEGACY_PLOT_TYPES = {
    "CL_VS_ALPHA": ("Alpha", "Cl"), "CD_VS_ALPHA": ("Alpha", "Cd"),
    "CL_VS_CD": ("Cd", "Cl"), "CM_VS_ALPHA": ("Alpha", "CPitch"),
    "CM_VS_CL": ("Cl", "CPitch"), "LD_VS_ALPHA": ("Alpha", "L/D"),
    "LATERAL_VS_BETA": ("Beta", "Cs"), "CY_VS_ALPHA": ("Alpha", "Cs"),
    "CROLL_VS_ALPHA": ("Alpha", "CRoll"), "CYAW_VS_ALPHA": ("Alpha", "CYaw"),
    "CMA_VS_ALPHA": ("Alpha", "Cma"), "CLA_VS_ALPHA": ("Alpha", "CLa"),
    "SM_VS_ALPHA": ("Alpha", "StaticMargin"),
    "CYB_VS_ALPHA": ("Alpha", "CYb"), "CNB_VS_ALPHA": ("Alpha", "Cnb"),
    "CLB_VS_ALPHA": ("Alpha", "Clb"),
}


def all_vars() -> Tuple[PlotVar, ...]:
    return _VARS


def grouped(extra: Iterable[str] = (),
            element_names: Optional[Iterable[str]] = None
            ) -> List[Tuple[str, List[PlotVar]]]:
    """[(group, [PlotVar, ...]), ...] in menu order.

    ``extra`` adds calculator outputs; ``element_names`` relabels the six
    balance elements with the recording balance's channel names."""
    names = list(element_names or [])
    out: Dict[str, List[PlotVar]] = {g: [] for g in GROUP_ORDER}
    for v in _VARS:
        if v.group == ELEMENTS and len(names) == len(ELEMENT_KEYS):
            name = names[ELEMENT_KEYS.index(v.key)]
            v = PlotVar(v.key, name, v.group, v.kind,
                        name + " [{}]", v.tip)
        out[v.group].append(v)
    for name in extra:
        if name not in BY_KEY:
            out[CUSTOM].append(PlotVar(name, name, CUSTOM, "none", name))
    return [(g, out[g]) for g in GROUP_ORDER if out[g]]


def unit_label(kind: str, output_units: str) -> str:
    """The unit of a ``kind`` in the output unit system."""
    try:
        from utils.windtunnel.units import UNIT_LABELS, UnitSystem
        labels = UNIT_LABELS[UnitSystem[output_units]]
        return {"force": labels.force, "moment": labels.moment,
                "pressure": labels.pressure, "velocity": labels.velocity,
                "density": labels.density,
                "temperature": labels.temperature}.get(kind, "")
    except Exception:                                  # noqa: BLE001
        return {"force": "lbf", "moment": "in-lb", "pressure": "psi",
                "velocity": "ft/s", "density": "slug/ft^3",
                "temperature": "degF"}.get(kind, "")


def axis_label(key: str, output_units: str = "IPS",
               element_names: Optional[Iterable[str]] = None) -> str:
    """Axis label for ``key``; unknown keys (calculator outputs) label
    with their own name, since only the user knows their units."""
    var = BY_KEY.get(key)
    if var is None:
        return key
    label = var.axis
    names = list(element_names or [])
    if var.group == ELEMENTS and len(names) == len(ELEMENT_KEYS):
        label = names[ELEMENT_KEYS.index(key)] + " [{}]"
    if "{}" in label:
        label = label.format(unit_label(var.kind, output_units))
    return label


def convert(data: np.ndarray, key: str, output_units: str = "IPS"
            ) -> np.ndarray:
    """Dimensional data from its stored unit (lbf, lb-in, psi, m/s,
    kg/m^3, degC) into the output unit system; everything else passes."""
    var = BY_KEY.get(key)
    if var is None or var.kind in ("coeff", "angle", "none"):
        return data
    try:
        from utils.windtunnel.units import UnitConverter, UnitSystem
        conv = UnitConverter(UnitSystem[output_units])
    except Exception:                                  # noqa: BLE001
        return data
    fn = {"force": conv.convert_force, "moment": conv.convert_moment,
          "pressure": conv.convert_pressure,
          "velocity": conv.convert_velocity,
          "density": conv.convert_density,
          "temperature": conv.convert_temperature}[var.kind]
    return fn(np.asarray(data, dtype=float))


def body_from_wind(lift, drag, side, alpha_deg, beta_deg):
    """Body-axis (Fx, Fy, Fz) from wind-axis (Lift, Drag, Side).

    Exact inverse of transforms.calc_wrf_forces, solved point by point
    from that transform's own 3x3 matrix, so the sign conventions can
    never drift apart:
        Lift = -sa Fx           + ca Fz
        Drag =  cb ca Fx - sb Fy + cb sa Fz
        Side =  sb ca Fx + cb Fy + sb sa Fz
    """
    lift, drag, side = (np.asarray(v, dtype=float).ravel()
                        for v in (lift, drag, side))
    a = np.deg2rad(np.asarray(alpha_deg, dtype=float).ravel())
    b = np.deg2rad(np.asarray(beta_deg, dtype=float).ravel())
    n = lift.size
    if not (drag.size == side.size == a.size == b.size == n) or n == 0:
        empty = np.array([])
        return empty, empty, empty
    sa, ca, sb, cb = np.sin(a), np.cos(a), np.sin(b), np.cos(b)
    mats = np.empty((n, 3, 3))
    mats[:, 0] = np.stack([-sa, np.zeros(n), ca], axis=1)
    mats[:, 1] = np.stack([cb * ca, -sb, cb * sa], axis=1)
    mats[:, 2] = np.stack([sb * ca, cb, sb * sa], axis=1)
    rhs = np.stack([lift, drag, side], axis=1)[..., None]
    fx, fy, fz = np.linalg.solve(mats, rhs)[..., 0].T
    return fx, fy, fz
