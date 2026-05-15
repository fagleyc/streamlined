"""
Wind Tunnel Blockage and Wall-Effect Corrections
=================================================

A family of blockage and wall-effect correction methods that can be
applied to reduced wind-tunnel data. Each method returns a corrected
copy of CL, CD, and alpha; the original arrays are never modified.

The full list of supported approaches:

    'none'           - No correction (default; values pass through)

    'pope_kirsten'   - Pope-Harper corrections specialized for the
                       Kirsten Wind Tunnel constants:
                           lambda = 1.0, k = 0.333,
                           delta  = 0.141, sigma = 0.011
                       This is the legacy "Reduce2" / brandt approach.

    'pope_generic'   - Pope-Harper corrections with user-supplied
                       facility constants lambda, k, delta, sigma.

    'maskell'        - Maskell's correction for stalled / bluff-body
                       data where wake blockage dominates.
                       Applies an additive correction to q based on
                       CD and the model frontal area.

    'glauert_closed' - Classical Glauert closed-test-section lift
                       interference: alpha correction proportional
                       to CL, plus a small q increment.

References:
    Pope, A. and Harper, J. J., "Low-Speed Wind Tunnel Testing",
        Wiley, 1966.  (Solid blockage, wake blockage, streamline
        curvature corrections.)
    Maskell, E. C., "A Theory of the Blockage Effects on Bluff
        Bodies and Stalled Wings in a Closed Wind Tunnel",
        ARC R&M 3400, 1965.

All corrections are written so that the "off" / "none" case returns
the inputs unchanged (backward-compatible default behavior).

Author: C. Fagley
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np


# Available method identifiers
METHOD_NONE = 'none'
METHOD_POPE_KIRSTEN = 'pope_kirsten'
METHOD_POPE_GENERIC = 'pope_generic'
METHOD_MASKELL = 'maskell'
METHOD_GLAUERT_CLOSED = 'glauert_closed'

AVAILABLE_METHODS = [
    METHOD_NONE,
    METHOD_POPE_KIRSTEN,
    METHOD_POPE_GENERIC,
    METHOD_MASKELL,
    METHOD_GLAUERT_CLOSED,
]

# Method display labels for the GUI
METHOD_LABELS = {
    METHOD_NONE: 'None (no correction)',
    METHOD_POPE_KIRSTEN: 'Pope-Harper (Kirsten Wind Tunnel)',
    METHOD_POPE_GENERIC: 'Pope-Harper (generic facility)',
    METHOD_MASKELL: "Maskell (stalled / bluff-body)",
    METHOD_GLAUERT_CLOSED: 'Glauert (closed test section)',
}

# Pope-Harper coefficients for the Kirsten Wind Tunnel.  Source: the
# legacy Reduce2 spreadsheet ("Tunnel correction terms: Lambda = 1.0,
# k = .333, delta = .141, sigma = .011").
KIRSTEN_CONSTANTS = {
    'lambda': 1.0,
    'k': 0.333,
    'delta': 0.141,
    'sigma': 0.011,
}


@dataclass
class BlockageConfig:
    """User-facing configuration for a single blockage correction run."""
    method: str = METHOD_NONE

    # Test section reference dimensions (consistent units; only used in ratios)
    test_section_area_in2: float = 0.0      # full cross-section area C
    test_section_width_in: float = 0.0      # B
    test_section_height_in: float = 0.0     # H

    # Pope-Harper coefficients (filled with KIRSTEN_CONSTANTS for
    # 'pope_kirsten' or user-supplied for 'pope_generic')
    lambda_: float = 1.0
    k: float = 0.333
    delta: float = 0.141
    sigma: float = 0.011

    # Solid blockage interpolation: frontal area at two pitch angles.
    # Linear interpolation between these two anchor points gives the
    # frontal area as a function of alpha.
    frontal_area_alpha_low_in2: float = 0.0
    frontal_area_alpha_low_deg: float = 0.0
    frontal_area_alpha_high_in2: float = 0.0
    frontal_area_alpha_high_deg: float = 0.0

    # Wing reference area for streamline curvature term and wake blockage
    reference_area_in2: float = 1.0

    def is_active(self) -> bool:
        """True iff a non-trivial method is selected."""
        return self.method != METHOD_NONE


@dataclass
class BlockageResult:
    """The output of applying a correction to one alpha-sweep."""
    alpha_corrected_deg: np.ndarray = field(
        default_factory=lambda: np.array([]))
    Cl_corrected: np.ndarray = field(default_factory=lambda: np.array([]))
    Cd_corrected: np.ndarray = field(default_factory=lambda: np.array([]))
    epsilon: np.ndarray = field(default_factory=lambda: np.array([]))
    method: str = METHOD_NONE
    notes: str = ''


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _interpolate_frontal_area(alpha_deg: np.ndarray,
                              cfg: BlockageConfig) -> np.ndarray:
    """Linear interpolation of frontal area as a function of alpha."""
    a_lo = cfg.frontal_area_alpha_low_deg
    a_hi = cfg.frontal_area_alpha_high_deg
    A_lo = cfg.frontal_area_alpha_low_in2
    A_hi = cfg.frontal_area_alpha_high_in2

    if a_hi <= a_lo:
        # Degenerate range; just use the average
        return np.full_like(alpha_deg, 0.5 * (A_lo + A_hi))

    slope = (A_hi - A_lo) / (a_hi - a_lo)
    return A_lo + slope * (alpha_deg - a_lo)


def _solid_blockage_pope(area_frontal: np.ndarray,
                         cfg: BlockageConfig) -> np.ndarray:
    """
    Pope solid blockage: eps_sb = K * S_frontal / C^(3/2),
    using the facility's K (= sigma * (h/w)^(3/2) compressed form, but
    we follow the legacy spreadsheet which expresses it directly via
    K1 = sigma * lambda) and the test section area C.

    Implementation follows the legacy Reduce2 form:
        eps_sb = K1 * S_frontal / C^(3/2)
    with K1 absorbed into a single 'sigma' coefficient at the model
    scale being tested.  Falls back to zero if test_section_area_in2
    is unset.
    """
    C = cfg.test_section_area_in2
    if C <= 0:
        return np.zeros_like(area_frontal)
    # Use sigma * lambda as the consolidated solid-blockage constant.
    K1 = cfg.sigma * cfg.lambda_
    return K1 * area_frontal / np.power(C, 1.5)


def _wake_blockage_pope(Cd_u: np.ndarray, cfg: BlockageConfig) -> np.ndarray:
    """Pope wake blockage: eps_wb = (S/(4*C)) * CD_u."""
    C = cfg.test_section_area_in2
    S = cfg.reference_area_in2
    if C <= 0:
        return np.zeros_like(Cd_u)
    return (S / (4.0 * C)) * Cd_u


# -----------------------------------------------------------------------------
# Method implementations
# -----------------------------------------------------------------------------

def _apply_none(alpha_deg, Cl, Cd, cfg) -> BlockageResult:
    return BlockageResult(
        alpha_corrected_deg=np.asarray(alpha_deg, dtype=float).copy(),
        Cl_corrected=np.asarray(Cl, dtype=float).copy(),
        Cd_corrected=np.asarray(Cd, dtype=float).copy(),
        epsilon=np.zeros_like(np.asarray(alpha_deg, dtype=float)),
        method=METHOD_NONE,
        notes='No correction applied.',
    )


def _apply_pope(alpha_deg, Cl, Cd, cfg, *, label) -> BlockageResult:
    """Pope-Harper combined correction (solid + wake + streamline)."""
    alpha_deg = np.asarray(alpha_deg, dtype=float)
    Cl_u = np.asarray(Cl, dtype=float)
    Cd_u = np.asarray(Cd, dtype=float)

    A_frontal = _interpolate_frontal_area(alpha_deg, cfg)
    eps_sb = _solid_blockage_pope(A_frontal, cfg)
    eps_wb = _wake_blockage_pope(Cd_u, cfg)
    eps_total = eps_sb + eps_wb

    # Apply Pope corrections:
    #   q_corrected = q_u * (1 + eps_total)^2
    #   CL  = CL_u  / (1 + eps_total)^2
    #   CD  = CD_u  / (1 + eps_total)^2 - eps_sb (solid blockage drag tare)
    #   d_alpha = delta * (S/C) * CL_u * 57.296  [deg]   (streamline curv.)
    scale = (1.0 + eps_total) ** 2
    Cl_c = Cl_u / scale
    Cd_c = Cd_u / scale - eps_sb
    if cfg.test_section_area_in2 > 0:
        d_alpha_deg = (cfg.delta * cfg.reference_area_in2
                       / cfg.test_section_area_in2) * Cl_u * 57.295779513
    else:
        d_alpha_deg = np.zeros_like(alpha_deg)
    alpha_c = alpha_deg + d_alpha_deg

    return BlockageResult(
        alpha_corrected_deg=alpha_c,
        Cl_corrected=Cl_c,
        Cd_corrected=Cd_c,
        epsilon=eps_total,
        method=label,
        notes=(f'Pope-Harper (lambda={cfg.lambda_}, k={cfg.k}, '
               f'delta={cfg.delta}, sigma={cfg.sigma}). '
               f'Streamline curvature shifts alpha by '
               f'delta*(S/C)*CL*57.3 deg.'),
    )


def _apply_maskell(alpha_deg, Cl, Cd, cfg) -> BlockageResult:
    """
    Maskell stalled / bluff-body correction:
        eps_M = (theta * S / C) * CD_u
    where theta ~ 5/2 for stalled wings (Maskell's value).
    The freestream q is then scaled by (1 + eps_M) and CL/CD divided
    by that factor.  No streamline curvature term.
    """
    alpha_deg = np.asarray(alpha_deg, dtype=float)
    Cl_u = np.asarray(Cl, dtype=float)
    Cd_u = np.asarray(Cd, dtype=float)

    C = cfg.test_section_area_in2
    S = cfg.reference_area_in2
    theta = 2.5
    if C > 0:
        eps_M = theta * (S / C) * Cd_u
    else:
        eps_M = np.zeros_like(Cd_u)

    scale = 1.0 + eps_M
    Cl_c = Cl_u / scale
    Cd_c = Cd_u / scale

    return BlockageResult(
        alpha_corrected_deg=alpha_deg.copy(),
        Cl_corrected=Cl_c,
        Cd_corrected=Cd_c,
        epsilon=eps_M,
        method=METHOD_MASKELL,
        notes='Maskell bluff-body / stalled-wing correction (theta=5/2).',
    )


def _apply_glauert_closed(alpha_deg, Cl, Cd, cfg) -> BlockageResult:
    """
    Glauert classical lift interference for closed test sections:
        d_alpha = delta_0 * (S/C) * CL_u * 57.296  [deg]
        d_Cd    = delta_0 * (S/C) * CL_u^2
    A small q increment is also applied if requested via sigma.
    """
    alpha_deg = np.asarray(alpha_deg, dtype=float)
    Cl_u = np.asarray(Cl, dtype=float)
    Cd_u = np.asarray(Cd, dtype=float)

    C = cfg.test_section_area_in2
    S = cfg.reference_area_in2
    delta_0 = cfg.delta if cfg.delta else 0.125  # classical value

    if C > 0:
        ratio = S / C
        d_alpha_deg = delta_0 * ratio * Cl_u * 57.295779513
        d_Cd = delta_0 * ratio * Cl_u ** 2
    else:
        d_alpha_deg = np.zeros_like(alpha_deg)
        d_Cd = np.zeros_like(Cd_u)

    return BlockageResult(
        alpha_corrected_deg=alpha_deg + d_alpha_deg,
        Cl_corrected=Cl_u.copy(),
        Cd_corrected=Cd_u + d_Cd,
        epsilon=np.zeros_like(alpha_deg),
        method=METHOD_GLAUERT_CLOSED,
        notes=(f'Glauert closed-section lift interference '
               f'(delta_0={delta_0}).'),
    )


# -----------------------------------------------------------------------------
# Public dispatcher
# -----------------------------------------------------------------------------

def apply_blockage_correction(alpha_deg, Cl, Cd,
                              cfg: BlockageConfig) -> BlockageResult:
    """
    Apply the selected blockage correction to a single alpha sweep.

    Inputs Cl, Cd, alpha_deg must be 1-D arrays of the same length.
    For multi-dimensional sweeps (alpha x beta), apply once per
    constant-beta row.
    """
    if cfg is None or cfg.method == METHOD_NONE:
        return _apply_none(alpha_deg, Cl, Cd, cfg)

    if cfg.method == METHOD_POPE_KIRSTEN:
        # Force the Kirsten constants regardless of what's in cfg
        cfg_k = BlockageConfig(**{**cfg.__dict__, **{
            'lambda_': KIRSTEN_CONSTANTS['lambda'],
            'k': KIRSTEN_CONSTANTS['k'],
            'delta': KIRSTEN_CONSTANTS['delta'],
            'sigma': KIRSTEN_CONSTANTS['sigma'],
        }})
        return _apply_pope(alpha_deg, Cl, Cd, cfg_k,
                           label=METHOD_POPE_KIRSTEN)

    if cfg.method == METHOD_POPE_GENERIC:
        return _apply_pope(alpha_deg, Cl, Cd, cfg,
                           label=METHOD_POPE_GENERIC)

    if cfg.method == METHOD_MASKELL:
        return _apply_maskell(alpha_deg, Cl, Cd, cfg)

    if cfg.method == METHOD_GLAUERT_CLOSED:
        return _apply_glauert_closed(alpha_deg, Cl, Cd, cfg)

    # Unknown method - return inputs untouched
    return _apply_none(alpha_deg, Cl, Cd, cfg)


def correction_summary(cfg: BlockageConfig) -> str:
    """Short human-readable description of the active correction."""
    if cfg is None or not cfg.is_active():
        return 'No blockage correction'
    label = METHOD_LABELS.get(cfg.method, cfg.method)
    return label
