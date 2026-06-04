"""
Custom Data Calculator
======================

A modular expression-based calculator for deriving additional
quantities from reduced wind-tunnel data.  Modeled loosely on the
ParaView Calculator filter: you define a NAMED template like

    name template:        "Cp{i}"
    expression template:  "(P{i} - p_inf) / q_inf"
    index variable:       "i"
    index range:          "1..32"

and the engine expands it into 32 derived variables (Cp1, Cp2, ..., Cp32)
computed from the available data on each test point's time-series.

Rules can also be index-free (a single template with no `{}`):

    name = "L_over_D"
    expression = "Cl / Cd"

Variables available to expressions
----------------------------------
The expression namespace exposes everything Streamlined knows about
a reduced data point:

  Raw DAQ channels by name:        P1, P32, N1, N2, Y1, Y2, Axial,
                                   Roll, Excitation, Pdiff, Ptot,
                                   Temp, Alpha, Beta, Time, ...
  Tunnel conditions:               Q (Q_psi), q_pa (Q_Pa), p0 (P_tot),
                                   p_static (P_static), p_inf (P_static),
                                   q_inf (Q_psi), T0, T_static, U_inf,
                                   Mach, Re, a, rho
  BRF forces (air-on minus mean air-off):
                                   Fx, Fy, Fz, Mx, My, Mz
  WRF aerodynamic forces:          Lift, Drag, Side, Roll_W (roll moment
                                   in wind frame), Pitch_W, Yaw_W
  Coefficients:                    Cl, Cd, Cs, CRoll, CPitch, CYaw
  Numpy math functions:            sin, cos, tan, exp, log, sqrt, abs,
                                   minimum, maximum, where, mean, std,
                                   pi (numpy.pi)

Each variable is a 1-D numpy array of length n_samples for the test
point being evaluated.  Scalar literals and tunnel-mean values work
naturally with array broadcasting.

Author: C. Fagley
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# Allowed names available to expression eval.  Numpy math + a handful
# of convenience aggregators.  No file / module access of any kind.
_ALLOWED_BUILTINS = {
    'abs': abs, 'min': min, 'max': max,
    'sum': sum, 'len': len, 'round': round,
    'float': float, 'int': int,
}

_NUMPY_FUNCS = {
    'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
    'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
    'atan2': np.arctan2,
    'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
    'exp': np.exp, 'log': np.log, 'log10': np.log10, 'log2': np.log2,
    'sqrt': np.sqrt, 'cbrt': np.cbrt,
    'pow': np.power, 'power': np.power,
    'sign': np.sign,
    'fabs': np.fabs,
    'floor': np.floor, 'ceil': np.ceil,
    'minimum': np.minimum, 'maximum': np.maximum,
    'clip': np.clip, 'where': np.where,
    'mean': np.mean, 'std': np.std, 'median': np.median,
    'sum_arr': np.sum,
    'pi': np.pi, 'e': np.e,
}


# ----------------------------------------------------------------------
# Rule dataclass
# ----------------------------------------------------------------------

@dataclass
class CalcRule:
    """
    A single user-defined calculation rule.

    name_template
        Output variable name template.  May contain a `{index_var}`
        placeholder to expand into multiple outputs.  Example: "Cp{i}".

    expression
        Python expression evaluated against a per-point namespace.
        The same `{index_var}` placeholder is substituted before eval.
        Example: "(P{i} - p_inf) / q_inf".

    index_var
        Placeholder name (default 'i').  Set empty string for
        single-output rules with no expansion.

    index_range
        How to enumerate the index.  Supports:
          - "1..32"        inclusive integer range
          - "1, 2, 5, 10"  explicit list
          - "auto:P{i}"    auto-detect from available variable names
                           (only indices that have a matching input
                           variable get a rule)
          - ""             single output (no expansion)

    enabled
        If False the rule is skipped during evaluation but kept in
        the model so it can be re-enabled later.

    description
        Free-form note shown in the UI.
    """
    name_template: str = ''
    expression: str = ''
    index_var: str = 'i'
    index_range: str = ''
    enabled: bool = True
    description: str = ''


# ----------------------------------------------------------------------
# Template expansion
# ----------------------------------------------------------------------

_AUTO_RE = re.compile(r'^auto\s*:\s*(.+)$')


def expand_rule(rule: CalcRule,
                available_vars: List[str]) -> List[Tuple[str, str]]:
    """
    Expand a rule into concrete (output_name, expression) pairs.

    Parameters
    ----------
    rule : CalcRule
        The user-defined rule.
    available_vars : list[str]
        Names of variables actually present in the dataset.  Used by
        the 'auto:...' index range to limit expansion to inputs that
        exist.

    Returns
    -------
    list of (output_name, expression) tuples ready for evaluation.
    Empty if the rule is disabled, malformed, or the expansion yields
    nothing applicable.
    """
    if not rule or not rule.enabled:
        return []
    if not rule.name_template or not rule.expression:
        return []

    placeholder = '{' + (rule.index_var or 'i') + '}'

    # No placeholder -> single output, no expansion needed
    if placeholder not in rule.name_template and placeholder not in rule.expression:
        return [(rule.name_template, rule.expression)]

    indices = _enumerate_indices(rule, available_vars)
    out = []
    for i in indices:
        name = rule.name_template.replace(placeholder, str(i))
        expr = rule.expression.replace(placeholder, str(i))
        out.append((name, expr))
    return out


def _enumerate_indices(rule: CalcRule,
                       available_vars: List[str]) -> List[int]:
    """Parse rule.index_range into a list of indices."""
    rng = (rule.index_range or '').strip()
    if not rng:
        return []

    m = _AUTO_RE.match(rng)
    if m:
        pattern = m.group(1).strip()
        placeholder = '{' + (rule.index_var or 'i') + '}'
        # Build a regex that matches "P1", "P32", etc.
        rx = '^' + re.escape(pattern).replace(
            re.escape(placeholder), r'(\d+)') + '$'
        rx_compiled = re.compile(rx)
        indices = sorted({
            int(m.group(1)) for v in available_vars
            if (m := rx_compiled.match(v))
        })
        return indices

    if '..' in rng:
        # e.g. "1..32"
        try:
            lo_s, hi_s = rng.split('..', 1)
            lo, hi = int(lo_s.strip()), int(hi_s.strip())
            if hi >= lo:
                return list(range(lo, hi + 1))
        except ValueError:
            return []
        return []

    # Comma-separated explicit list, e.g. "1, 5, 10"
    out = []
    for part in rng.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


# ----------------------------------------------------------------------
# Expression evaluation
# ----------------------------------------------------------------------


def _safe_globals() -> dict:
    """Build the restricted globals dict for eval()."""
    g = {'__builtins__': _ALLOWED_BUILTINS}
    g.update(_NUMPY_FUNCS)
    return g


def evaluate(expression: str, namespace: Dict[str, Any]) -> Any:
    """
    Evaluate an expression against a variable namespace.

    Raises
    ------
    ValueError
        If the expression references an undefined name or contains
        a syntax error.  The original Python exception is wrapped to
        give a clearer message to GUI users.
    """
    if not expression.strip():
        raise ValueError("Empty expression")
    try:
        return eval(expression, _safe_globals(), namespace)
    except NameError as e:
        raise ValueError(
            f"Undefined variable in expression '{expression}': {e}")
    except SyntaxError as e:
        raise ValueError(
            f"Syntax error in expression '{expression}': {e}")
    except Exception as e:
        raise ValueError(
            f"Failed to evaluate '{expression}': "
            f"{type(e).__name__}: {e}")


# ----------------------------------------------------------------------
# Per-point variable namespace
# ----------------------------------------------------------------------

def build_namespace(pt) -> Dict[str, np.ndarray]:
    """
    Build the variable namespace for one ReducedDataPoint.

    Exposes raw DAQ channels, tunnel conditions, BRF / WRF forces,
    and aero coefficients as 1-D numpy arrays of length n_samples.
    """
    ns: Dict[str, Any] = {}

    # Raw DAQ channels (air_on dictionary)
    if hasattr(pt, 'air_on') and isinstance(pt.air_on, dict):
        for key, val in pt.air_on.items():
            try:
                ns[key] = np.asarray(val, dtype=float)
            except Exception:
                pass

    # Convenience: alpha, beta, time at top level
    for attr in ('alpha', 'beta', 'time'):
        v = getattr(pt, attr, None)
        if v is not None:
            try:
                ns[attr] = np.asarray(v, dtype=float)
            except Exception:
                pass

    # Tunnel conditions
    tc = getattr(pt, 'tunnel', None)
    if tc is not None:
        # Both verbose and short aliases
        mappings = {
            'Q': 'Q', 'q': 'Q', 'q_psi': 'Q',
            'q_pa': 'Q_mks', 'Q_mks': 'Q_mks',
            'p0': 'P_tot', 'P_tot': 'P_tot', 'P0': 'P_tot',
            'p_static': 'P_static', 'P_static': 'P_static',
            'p_inf': 'P_static',
            'q_inf': 'Q',
            'T0': 'T0',
            'T': 'T', 'T_static': 'T',
            'U_inf': 'U_inf', 'U': 'U_inf',
            'Mach': 'Mach', 'M': 'Mach',
            'Re': 'Re',
            'a': 'a', 'sound_speed': 'a',
            'rho': 'rho', 'density': 'rho',
        }
        for alias, src in mappings.items():
            v = getattr(tc, src, None)
            if v is not None and not (isinstance(v, np.ndarray)
                                       and v.size == 0):
                try:
                    ns[alias] = np.asarray(v, dtype=float)
                except Exception:
                    pass

    # BRF aerodynamic forces (air-on minus mean air-off)
    brf_on = getattr(pt, 'brf_on', None)
    brf_off = getattr(pt, 'brf_off', None)
    for attr in ('Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'):
        if brf_on is not None:
            on_arr = getattr(brf_on, attr, None)
            if on_arr is not None and np.asarray(on_arr).size > 0:
                v = np.asarray(on_arr, dtype=float)
                if brf_off is not None:
                    off = getattr(brf_off, attr, None)
                    if off is not None and np.asarray(off).size > 0:
                        v = v - float(np.mean(off))
                ns[attr] = v

    # WRF forces (aerodynamic, already tare-subtracted)
    wrf = getattr(pt, 'wrf_aero', None)
    if wrf is not None:
        for src, alias in [('Lift', 'Lift'), ('Drag', 'Drag'),
                            ('Side', 'Side'),
                            ('Roll', 'Roll_W'), ('Pitch', 'Pitch_W'),
                            ('Yaw', 'Yaw_W')]:
            v = getattr(wrf, src, None)
            if v is not None and np.asarray(v).size > 0:
                ns[alias] = np.asarray(v, dtype=float)

    # Coefficient time-series
    cf = getattr(pt, 'coeffs', None)
    if cf is not None:
        for attr in ('Cl', 'Cd', 'Cs', 'CRoll', 'CPitch', 'CYaw'):
            v = getattr(cf, attr, None)
            if v is not None and np.asarray(v).size > 0:
                ns[attr] = np.asarray(v, dtype=float)

    return ns


def available_variables(case) -> List[str]:
    """
    Return the names available for expression use in the first
    test point of the case.  Useful for auto-detect of pressure
    port enumeration.
    """
    if case is None:
        return []
    daq = getattr(case, 'daq', None)
    red = getattr(daq, 'red', None) if daq is not None else None
    if not red:
        return []
    try:
        ns = build_namespace(red[0])
    except Exception:
        return []
    return sorted(ns.keys())


# ----------------------------------------------------------------------
# Apply rules to a case
# ----------------------------------------------------------------------

def evaluate_rule_on_point(rule_expression: str,
                            pt) -> Optional[np.ndarray]:
    """Evaluate the (expanded) expression against one point's namespace."""
    try:
        ns = build_namespace(pt)
        result = evaluate(rule_expression, ns)
        if np.isscalar(result):
            return np.array([float(result)])
        return np.asarray(result, dtype=float)
    except Exception:
        return None


def apply_rules_to_case(case, rules: List[CalcRule]) -> Dict[str, np.ndarray]:
    """
    Evaluate each enabled rule on each test point and return a dict
    of {output_name: per-point-mean array}.  The array shape matches
    case.alphas (1-D flat or 2-D grid).

    Parameters
    ----------
    case : TestCase
        Must have a populated case.daq.red list.
    rules : list of CalcRule
        Active rules to evaluate.

    Returns
    -------
    dict[str, np.ndarray]
        Output variable name -> per-point mean array shaped to match
        case.alphas.  Failed evaluations produce NaN arrays.
    """
    out: Dict[str, np.ndarray] = {}
    if not case or not case.has_data:
        return out

    daq = getattr(case, 'daq', None)
    red = getattr(daq, 'red', None) if daq is not None else None
    if not red:
        return out

    # Discover available variables from the first point
    av_vars = available_variables(case)
    if not av_vars:
        return out

    target_shape = case.alphas.shape
    sort_idx = None
    ss = getattr(daq, 'ss', None)
    if ss is not None and hasattr(ss, 'indices'):
        try:
            sort_idx = np.asarray(ss.indices)
        except Exception:
            sort_idx = None

    for rule in rules:
        for output_name, expr in expand_rule(rule, av_vars):
            scalars = []
            for pt in red:
                arr = evaluate_rule_on_point(expr, pt)
                if arr is None or arr.size == 0:
                    scalars.append(np.nan)
                else:
                    scalars.append(float(np.mean(arr)))
            flat = np.asarray(scalars, dtype=float)

            # Apply sort_idx and reshape to match case grid
            if (sort_idx is not None and sort_idx.size > 0
                    and len(flat) >= sort_idx.size):
                try:
                    sorted_flat = flat[sort_idx]
                except Exception:
                    sorted_flat = flat
            else:
                sorted_flat = flat

            try:
                if target_shape != sorted_flat.shape:
                    out[output_name] = sorted_flat.reshape(target_shape)
                else:
                    out[output_name] = sorted_flat
            except ValueError:
                out[output_name] = sorted_flat
    return out


def expanded_output_names(rules: List[CalcRule],
                           case=None) -> List[str]:
    """
    Return the flat list of every expanded output name across all
    enabled rules.  Useful for populating plot / time-history menus.
    """
    av = available_variables(case) if case is not None else []
    names: List[str] = []
    for rule in rules:
        for name, _ in expand_rule(rule, av):
            if name not in names:
                names.append(name)
    return names


def evaluate_timeseries(rule_expression: str, pt) -> Optional[np.ndarray]:
    """
    Evaluate one (already-expanded) expression on a single point and
    return the full time-series result.  Used by the time history view.
    """
    return evaluate_rule_on_point(rule_expression, pt)


# ----------------------------------------------------------------------
# Serialization helpers
# ----------------------------------------------------------------------

def rule_to_dict(rule: CalcRule) -> dict:
    return asdict(rule)


def rule_from_dict(d: dict) -> CalcRule:
    return CalcRule(
        name_template=d.get('name_template', ''),
        expression=d.get('expression', ''),
        index_var=d.get('index_var', 'i'),
        index_range=d.get('index_range', ''),
        enabled=bool(d.get('enabled', True)),
        description=d.get('description', ''),
    )


def rules_to_dicts(rules: List[CalcRule]) -> List[dict]:
    return [rule_to_dict(r) for r in rules]


def rules_from_dicts(dicts: List[dict]) -> List[CalcRule]:
    if not isinstance(dicts, list):
        return []
    return [rule_from_dict(d) for d in dicts if isinstance(d, dict)]
