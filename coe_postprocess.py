"""
COE Post-Processor
==================

Standalone script that loads one or more legacy Reduce2 .COE files and
generates the same set of plots that the 5GAT Atail.xlsm spreadsheet
produces, without any Excel / VBA / ActiveX dependency.

Outputs:
    - CL vs Alpha, CD vs Alpha, CD vs CL
    - Cm vs Alpha, Cm vs CL
    - L/D vs Alpha
    - Cma vs Alpha           (central-difference longitudinal stability)
    - CLa vs Alpha           (central-difference lift curve slope)
    - Static Margin vs Alpha (-Cma / CLa)
    - CY vs Alpha, Cn vs Alpha, Cl (roll) vs Alpha
    - CYb / Cnb / Clb vs Alpha  (only when >= 2 input files at
      different betas are supplied)
    - Blockage-corrected alpha, CL, CD overlays (optional)

Usage
-----
    # Single COE file: longitudinal plots only
    python coe_postprocess.py case_B0.COE

    # Two files at different beta: also produces beta-derivatives
    python coe_postprocess.py case_B0.COE case_B10.COE

    # Apply a blockage correction:
    python coe_postprocess.py case.COE --blockage pope_kirsten \\
        --test-section-area 1152 --reference-area 81.5 \\
        --frontal-low 10.0 --frontal-low-alpha -4 \\
        --frontal-high 15.0 --frontal-high-alpha 16

    # Save PNG plots to a directory instead of showing them
    python coe_postprocess.py case.COE -o ./plots

Run `python coe_postprocess.py --help` for the full option list.

This script is functionally a drop-in replacement for the 5GAT
Atail.xlsm spreadsheet.  It has no ActiveX or macro dependencies.

Author: C. Fagley
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Ensure the local utils package is importable when run from the repo
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.windtunnel.coe_reader import read_coe_file, COEData
from utils.windtunnel.derivatives import central_diff
from utils.windtunnel.blockage import (
    BlockageConfig, apply_blockage_correction,
    AVAILABLE_METHODS, METHOD_LABELS,
)


def _alpha_sort(coe: COEData) -> Tuple[np.ndarray, np.ndarray]:
    """Sort the COE data by ascending alpha, return order index."""
    alpha = coe['Alpha']
    order = np.argsort(alpha)
    return alpha[order], order


def _plot_basic(ax, coe: COEData, x_key: str, y_key: str,
                label: str, color: str):
    """Plot one curve sorted by x_key (typically Alpha)."""
    if x_key == 'Alpha':
        _, order = _alpha_sort(coe)
    elif x_key == 'CLift':
        order = np.argsort(coe['CLift'])
    elif x_key == 'CDrag':
        order = np.argsort(coe['CDrag'])
    else:
        order = np.arange(len(coe[x_key]))
    ax.plot(coe[x_key][order], coe[y_key][order],
            marker='o', linewidth=1.5, label=label, color=color)


def _alpha_derivative(coe: COEData, y_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Central-difference y w.r.t. alpha; returns (alpha_sorted, dy/dalpha)."""
    alpha, order = _alpha_sort(coe)
    y_sorted = coe[y_key][order]
    return alpha, central_diff(y_sorted, alpha, axis=0)


def _beta_derivative_pair(coe_lo: COEData, coe_hi: COEData,
                           y_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finite-difference y w.r.t. beta between two cases at different beta.
    Returns (alpha_array, d(y)/d(beta)) on the alpha grid of the LOW
    case (linearly interpolating the HIGH case to the same alphas if
    the grids differ).
    """
    alpha_lo, order_lo = _alpha_sort(coe_lo)
    y_lo = coe_lo[y_key][order_lo]

    alpha_hi, order_hi = _alpha_sort(coe_hi)
    y_hi = coe_hi[y_key][order_hi]

    beta_lo = float(np.mean(coe_lo['Beta']))
    beta_hi = float(np.mean(coe_hi['Beta']))
    d_beta = beta_hi - beta_lo
    if abs(d_beta) < 1e-9:
        return alpha_lo, np.full_like(alpha_lo, np.nan)

    # Resample the high-beta case onto the low-beta alpha grid
    y_hi_resampled = np.interp(alpha_lo, alpha_hi, y_hi)
    return alpha_lo, (y_hi_resampled - y_lo) / d_beta


def _apply_blockage_to_coe(coe: COEData,
                            cfg: BlockageConfig) -> dict:
    """Return a dict with corrected alpha, CL, CD for the COE data."""
    alpha = coe['Alpha']
    Cl = coe['CLift']
    Cd = coe['CDrag']
    result = apply_blockage_correction(alpha, Cl, Cd, cfg)
    return {
        'alpha': result.alpha_corrected_deg,
        'CL': result.Cl_corrected,
        'CD': result.Cd_corrected,
        'epsilon': result.epsilon,
        'method': result.method,
    }


# ----------------------------------------------------------------------
# Plot dispatcher
# ----------------------------------------------------------------------

_DEFAULT_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd',
                   '#ff7f0e', '#8c564b']


def _figure_save_or_show(fig, name: str, out_dir: Optional[Path]) -> None:
    if out_dir is None:
        return
    fig.savefig(out_dir / f'{name}.png', dpi=150, bbox_inches='tight')


def make_all_plots(coes: List[COEData],
                   blockage_cfg: Optional[BlockageConfig] = None,
                   out_dir: Optional[Path] = None) -> List:
    """Render the full plot suite for one or more COE files."""
    figs = []

    labels = []
    for coe in coes:
        b = float(np.mean(coe['Beta']))
        name = Path(coe.filepath).stem
        labels.append(f'{name} (beta={b:.1f} deg)')

    colors = _DEFAULT_COLORS[:len(coes)]

    # ---- Standard polars ----
    polar_configs = [
        ('CL_vs_Alpha', 'Alpha', 'CLift', r'$\alpha$ [deg]', r'$C_L$'),
        ('CD_vs_Alpha', 'Alpha', 'CDrag', r'$\alpha$ [deg]', r'$C_D$'),
        ('CD_vs_CL',    'CLift', 'CDrag', r'$C_L$',          r'$C_D$'),
        ('Cm_vs_Alpha', 'Alpha', 'CPiMomSh', r'$\alpha$ [deg]', r'$C_m$'),
        ('Cm_vs_CL',    'CLift', 'CPiMomSh', r'$C_L$',         r'$C_m$'),
        ('CY_vs_Alpha', 'Alpha', 'CY', r'$\alpha$ [deg]', r'$C_Y$'),
        ('Cn_vs_Alpha', 'Alpha', 'CYaMomSh', r'$\alpha$ [deg]', r'$C_n$'),
        ('Cl_roll_vs_Alpha', 'Alpha', 'CRoMomSh', r'$\alpha$ [deg]',
         r'$C_l$ (roll)'),
        ('LD_vs_Alpha', 'Alpha', 'LD', r'$\alpha$ [deg]', r'L/D'),
    ]

    for name, xk, yk, xlab, ylab in polar_configs:
        fig, ax = plt.subplots(figsize=(7, 5))
        for coe, lab, col in zip(coes, labels, colors):
            _plot_basic(ax, coe, xk, yk, lab, col)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_title(name.replace('_', ' '))
        _figure_save_or_show(fig, name, out_dir)
        figs.append(fig)

    # ---- Alpha derivatives (Cma, CLa, Static Margin) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for coe, lab, col in zip(coes, labels, colors):
        alpha, Cma = _alpha_derivative(coe, 'CPiMomSh')
        _, CLa = _alpha_derivative(coe, 'CLift')
        with np.errstate(divide='ignore', invalid='ignore'):
            sm = np.where(np.abs(CLa) > 1e-10, -Cma / CLa, np.nan)
        axes[0].plot(alpha, Cma, marker='o', label=lab, color=col)
        axes[1].plot(alpha, CLa, marker='o', label=lab, color=col)
        axes[2].plot(alpha, sm, marker='o', label=lab, color=col)
    axes[0].set_title('Cma vs Alpha')
    axes[0].set_xlabel(r'$\alpha$ [deg]')
    axes[0].set_ylabel(r'$C_{m\alpha}$ [1/deg]')
    axes[1].set_title('CLa vs Alpha')
    axes[1].set_xlabel(r'$\alpha$ [deg]')
    axes[1].set_ylabel(r'$C_{L\alpha}$ [1/deg]')
    axes[2].set_title('Static Margin vs Alpha')
    axes[2].set_xlabel(r'$\alpha$ [deg]')
    axes[2].set_ylabel('Static Margin')
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    _figure_save_or_show(fig, 'Stability_Long', out_dir)
    figs.append(fig)

    # ---- Beta derivatives: require 2 COEs at different beta ----
    if len(coes) >= 2:
        # Sort by beta
        coes_by_beta = sorted(
            coes, key=lambda c: float(np.mean(c['Beta'])))
        beta_lo = float(np.mean(coes_by_beta[0]['Beta']))
        beta_hi = float(np.mean(coes_by_beta[-1]['Beta']))
        if abs(beta_hi - beta_lo) > 1e-6:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            alpha_arr, CYb = _beta_derivative_pair(
                coes_by_beta[0], coes_by_beta[-1], 'CY')
            _, Cnb = _beta_derivative_pair(
                coes_by_beta[0], coes_by_beta[-1], 'CYaMomSh')
            _, Clb = _beta_derivative_pair(
                coes_by_beta[0], coes_by_beta[-1], 'CRoMomSh')
            axes[0].plot(alpha_arr, CYb, marker='o',
                         label=f'beta {beta_lo:.0f} -> {beta_hi:.0f}',
                         color=colors[0])
            axes[1].plot(alpha_arr, Cnb, marker='o',
                         label=f'beta {beta_lo:.0f} -> {beta_hi:.0f}',
                         color=colors[0])
            axes[2].plot(alpha_arr, Clb, marker='o',
                         label=f'beta {beta_lo:.0f} -> {beta_hi:.0f}',
                         color=colors[0])
            axes[0].set_title('CYb vs Alpha')
            axes[0].set_xlabel(r'$\alpha$ [deg]')
            axes[0].set_ylabel(r'$C_{Y\beta}$ [1/deg]')
            axes[1].set_title('Cnb vs Alpha')
            axes[1].set_xlabel(r'$\alpha$ [deg]')
            axes[1].set_ylabel(r'$C_{n\beta}$ [1/deg]')
            axes[2].set_title('Clb vs Alpha')
            axes[2].set_xlabel(r'$\alpha$ [deg]')
            axes[2].set_ylabel(r'$C_{l\beta}$ [1/deg]')
            for ax in axes:
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9)
            fig.tight_layout()
            _figure_save_or_show(fig, 'Stability_Lat', out_dir)
            figs.append(fig)

    # ---- Blockage corrections ----
    if blockage_cfg is not None and blockage_cfg.is_active():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for coe, lab, col in zip(coes, labels, colors):
            corr = _apply_blockage_to_coe(coe, blockage_cfg)
            alpha_raw = coe['Alpha']
            order = np.argsort(alpha_raw)
            axes[0].plot(alpha_raw[order], coe['CLift'][order],
                         marker='o', linestyle='--', color=col,
                         label=f'{lab} (raw)')
            axes[0].plot(corr['alpha'][order], corr['CL'][order],
                         marker='s', color=col,
                         label=f'{lab} (corrected)')
            axes[1].plot(coe['CDrag'][order], coe['CLift'][order],
                         marker='o', linestyle='--', color=col,
                         label=f'{lab} (raw)')
            axes[1].plot(corr['CD'][order], corr['CL'][order],
                         marker='s', color=col,
                         label=f'{lab} (corrected)')
        axes[0].set_xlabel(r'$\alpha$ [deg]')
        axes[0].set_ylabel(r'$C_L$')
        axes[0].set_title(
            f'CL vs Alpha ({METHOD_LABELS.get(blockage_cfg.method, blockage_cfg.method)})')
        axes[1].set_xlabel(r'$C_D$')
        axes[1].set_ylabel(r'$C_L$')
        axes[1].set_title(
            f'Drag Polar ({METHOD_LABELS.get(blockage_cfg.method, blockage_cfg.method)})')
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        _figure_save_or_show(fig, 'Blockage_Corrected', out_dir)
        figs.append(fig)

    return figs


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Post-process Reduce2 .COE files and generate '
                    'stability / blockage plots without Excel.')
    p.add_argument('coe_files', nargs='+',
                   help='One or more .COE file paths. With two files '
                        'at different beta, beta-derivative plots are '
                        'also produced.')
    p.add_argument('-o', '--output-dir', default=None,
                   help='If set, save plots as PNG files into this '
                        'directory instead of showing them.')
    p.add_argument('--blockage', default='none',
                   choices=AVAILABLE_METHODS,
                   help='Blockage correction approach (default: none).')
    p.add_argument('--test-section-area', type=float, default=0.0,
                   help='Test section cross-section area C [in^2].')
    p.add_argument('--reference-area', type=float, default=None,
                   help='Wing reference area S [in^2]. Defaults to '
                        'the value in the COE header.')
    p.add_argument('--lambda-', type=float, default=1.0, dest='lambda_',
                   help='Pope-Harper lambda (generic only).')
    p.add_argument('--k', type=float, default=0.333,
                   help='Pope-Harper k (generic only).')
    p.add_argument('--delta', type=float, default=0.141,
                   help='Pope-Harper delta (generic / Glauert).')
    p.add_argument('--sigma', type=float, default=0.011,
                   help='Pope-Harper sigma (generic only).')
    p.add_argument('--frontal-low', type=float, default=0.0,
                   help='Frontal area at low alpha [in^2].')
    p.add_argument('--frontal-low-alpha', type=float, default=0.0,
                   help='Low-alpha anchor [deg].')
    p.add_argument('--frontal-high', type=float, default=0.0,
                   help='Frontal area at high alpha [in^2].')
    p.add_argument('--frontal-high-alpha', type=float, default=20.0,
                   help='High-alpha anchor [deg].')
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    coes = []
    for fp in args.coe_files:
        coes.append(read_coe_file(fp))
        print(f'Read {fp}: {coes[-1].n_rows} rows, '
              f'beta={float(np.mean(coes[-1]["Beta"])):.2f} deg')

    # Reference area defaults: take from first COE header if not given
    ref_area = (args.reference_area if args.reference_area is not None
                else coes[0].header.ref_area_in2)

    cfg = BlockageConfig(
        method=args.blockage,
        test_section_area_in2=args.test_section_area,
        reference_area_in2=ref_area,
        lambda_=args.lambda_,
        k=args.k,
        delta=args.delta,
        sigma=args.sigma,
        frontal_area_alpha_low_in2=args.frontal_low,
        frontal_area_alpha_low_deg=args.frontal_low_alpha,
        frontal_area_alpha_high_in2=args.frontal_high,
        frontal_area_alpha_high_deg=args.frontal_high_alpha,
    )

    out_dir = None
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    figs = make_all_plots(coes, blockage_cfg=cfg, out_dir=out_dir)

    if out_dir:
        print(f'\nWrote {len(figs)} plots to {out_dir}')
    else:
        plt.show()

    return 0


if __name__ == '__main__':
    sys.exit(main())
