"""
Stability Derivatives
=====================

Compute longitudinal and lateral stability derivatives from reduced
wind-tunnel coefficient data using central finite differences.

The supported derivatives are:

    Cma         = d(CPitch) / d(alpha)        [per deg]
    CLa         = d(CL)     / d(alpha)        [per deg]
    Static Margin = -Cma / CLa                [non-dim]
    CYb         = d(CY)     / d(beta)         [per deg]
    Cnb         = d(CYaw)   / d(beta)         [per deg]
    Clb         = d(CRoll)  / d(beta)         [per deg]

The independent variables (alpha, beta) need not be uniformly spaced;
central differences are computed using the actual independent-variable
spacing on either side of each interior point.  Forward / backward
differences are used at the array endpoints.

For 2-D gridded cases (n_alpha x n_beta):
  - Alpha derivatives operate along axis 0 (alpha varies down the rows)
  - Beta  derivatives operate along axis 1 (beta varies along columns)

For 1-D flat cases:
  - Alpha derivatives use np.argsort(alpha) sorting then central diff
  - Beta derivatives require multiple beta values at matching alphas;
    a NaN array is returned if the input has only one unique beta.

All routines return arrays of the same shape as the input coefficient
array so the caller can slice them with the same indexing logic used
for the raw coefficients.

Author: C. Fagley
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def central_diff(y: np.ndarray, x: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute the central finite difference d(y)/d(x) along the given axis.

    Endpoints use one-sided differences. Non-uniform spacing in x is
    handled correctly by using the actual neighbor-to-neighbor spacing
    rather than assuming a constant dx.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable. Shape (..., N, ...).
    x : np.ndarray
        Independent variable. Same shape as y or 1-D with length N
        along the diff axis.
    axis : int
        Axis along which to differentiate.

    Returns
    -------
    np.ndarray
        Derivative dy/dx with the same shape as y. NaN-filled where
        the diff cannot be evaluated (e.g. fewer than 2 points).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    # Broadcast x to match y shape (if x is 1-D, expand along axis)
    if x.shape != y.shape:
        target = [1] * y.ndim
        target[axis] = y.shape[axis]
        x = np.broadcast_to(x.reshape(target), y.shape).copy()

    n = y.shape[axis]
    if n < 2:
        return np.full_like(y, np.nan)

    # numpy gradient handles non-uniform spacing along an axis
    # by accepting a 1-D coordinate array per axis. For our use
    # (x varying along the same axis as y, possibly different
    # along orthogonal axes), iterate over slices manually.
    out = np.empty_like(y)

    # Move diff axis to the front for easier slicing
    y_moved = np.moveaxis(y, axis, 0)  # shape (n, *rest)
    x_moved = np.moveaxis(x, axis, 0)
    out_moved = np.empty_like(y_moved)

    # Interior central differences
    for i in range(1, n - 1):
        dx = x_moved[i + 1] - x_moved[i - 1]
        # Avoid divide-by-zero (collinear repeated points)
        dx_safe = np.where(np.abs(dx) < 1e-12, np.nan, dx)
        out_moved[i] = (y_moved[i + 1] - y_moved[i - 1]) / dx_safe

    # Endpoints: one-sided
    dx0 = x_moved[1] - x_moved[0]
    dx0_safe = np.where(np.abs(dx0) < 1e-12, np.nan, dx0)
    out_moved[0] = (y_moved[1] - y_moved[0]) / dx0_safe

    dxN = x_moved[-1] - x_moved[-2]
    dxN_safe = np.where(np.abs(dxN) < 1e-12, np.nan, dxN)
    out_moved[-1] = (y_moved[-1] - y_moved[-2]) / dxN_safe

    # Restore original axis order
    return np.moveaxis(out_moved, 0, axis)


def _alpha_derivative(case, attr: str) -> np.ndarray:
    """Compute d(attr)/d(alpha) over a case's coefficient grid."""
    y = getattr(case, attr, None)
    if y is None or np.asarray(y).size == 0:
        return np.array([])
    y = np.asarray(y, dtype=float)
    alphas = np.asarray(case.alphas, dtype=float)

    if y.ndim == 2 and alphas.ndim == 2:
        # 2D grid: diff along axis 0 (alpha varies down rows)
        return central_diff(y, alphas, axis=0)

    # 1D fallback: sort by alpha, diff, restore original order
    if y.ndim == 1:
        a_flat = alphas.flatten()
        y_flat = y.flatten()
        order = np.argsort(a_flat)
        a_sorted = a_flat[order]
        y_sorted = y_flat[order]
        d_sorted = central_diff(y_sorted, a_sorted, axis=0)
        # Restore to original index order
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))
        return d_sorted[inv_order]

    return np.full_like(y, np.nan)


def _beta_derivative(case, attr: str) -> np.ndarray:
    """Compute d(attr)/d(beta) over a case's coefficient grid."""
    y = getattr(case, attr, None)
    if y is None or np.asarray(y).size == 0:
        return np.array([])
    y = np.asarray(y, dtype=float)
    betas = np.asarray(case.betas, dtype=float)

    if y.ndim == 2 and betas.ndim == 2:
        # 2D grid: diff along axis 1 (beta varies along columns)
        # Require at least 2 unique betas
        if y.shape[1] < 2:
            return np.full_like(y, np.nan)
        return central_diff(y, betas, axis=1)

    # 1D: group rows by alpha, compute diff across betas within each
    # alpha group. If a group has only one beta, the result is NaN.
    if y.ndim == 1:
        a_flat = np.asarray(case.alphas, dtype=float).flatten()
        b_flat = betas.flatten()
        y_flat = y.flatten()
        out = np.full_like(y_flat, np.nan)
        # Bucket alphas to integer-rounded keys for grouping
        keys = np.round(a_flat * 2) / 2  # nearest 0.5 deg
        for key in np.unique(keys):
            idx = np.where(keys == key)[0]
            if len(idx) < 2:
                continue
            # Sort by beta within this alpha group
            order = idx[np.argsort(b_flat[idx])]
            b_sub = b_flat[order]
            y_sub = y_flat[order]
            d_sub = central_diff(y_sub, b_sub, axis=0)
            out[order] = d_sub
        return out

    return np.full_like(y, np.nan)


# -----------------------------------------------------------------------------
# Public derivative accessors
# -----------------------------------------------------------------------------

DERIVATIVE_NAMES = {
    # name -> (kind, source_attr)
    'Cma': ('alpha', 'CPitch'),
    'CLa': ('alpha', 'Cl'),
    'CDa': ('alpha', 'Cd'),
    'CYb': ('beta', 'Cs'),
    'Cnb': ('beta', 'CYaw'),
    'Clb': ('beta', 'CRoll'),
    'StaticMargin': ('special', None),  # = -Cma / CLa
}


def is_derivative(var: str) -> bool:
    """Return True if `var` names one of the supported derivatives."""
    return var in DERIVATIVE_NAMES


def get_derivative(case, name: str) -> np.ndarray:
    """
    Return the requested stability derivative array for the case.

    Parameters
    ----------
    case : TestCase
        Reduced Streamlined case.
    name : str
        One of: 'Cma', 'CLa', 'CDa', 'CYb', 'Cnb', 'Clb', 'StaticMargin'.

    Returns
    -------
    np.ndarray
        Same shape as case.Cl. NaN where the derivative cannot be
        computed (single-point alpha or beta).
    """
    if name not in DERIVATIVE_NAMES:
        return np.array([])

    if name == 'StaticMargin':
        cma = _alpha_derivative(case, 'CPitch')
        cla = _alpha_derivative(case, 'Cl')
        if cma.size == 0 or cla.size == 0:
            return np.array([])
        # SM = -Cma / CLa ; guard against zero CLa
        cla_safe = np.where(np.abs(cla) < 1e-10, np.nan, cla)
        return -cma / cla_safe

    kind, attr = DERIVATIVE_NAMES[name]
    if kind == 'alpha':
        return _alpha_derivative(case, attr)
    if kind == 'beta':
        return _beta_derivative(case, attr)
    return np.array([])
