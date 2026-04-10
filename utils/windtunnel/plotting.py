"""
Plotting Module
===============

Functions for generating standard aerodynamic plots with publication-quality formatting.
"""

import shutil

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from .reduction import SteadyStateData

# Cache the result so we only check once per process
_latex_available: Optional[bool] = None


def is_latex_available() -> bool:
    """Check whether a LaTeX installation is available on this system."""
    global _latex_available
    if _latex_available is None:
        _latex_available = shutil.which('latex') is not None
    return _latex_available


def setup_plot_style(use_latex: bool = True) -> None:
    """
    Set up matplotlib style to match MATLAB defaults.

    Parameters
    ----------
    use_latex : bool
        Whether to use LaTeX for text rendering.
        If True but LaTeX is not installed, falls back to non-LaTeX rendering.
    """
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.5,
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.figsize': (8, 6),
        'figure.dpi': 100,
    })

    if use_latex and is_latex_available():
        plt.rcParams.update({
            'text.usetex': True,
            'font.family': 'serif',
            'text.latex.preamble': r'\usepackage{amsmath}',
        })
    else:
        plt.rcParams.update({
            'text.usetex': False,
        })


def _get_coeff_label(coeff_name: str, use_latex: bool = True) -> str:
    """Get proper label for a coefficient."""
    if use_latex:
        labels = {
            'Cl': r'$C_L$',
            'Cd': r'$C_D$',
            'Cs': r'$C_S$',
            'CRoll': r'$C_\ell$',
            'CPitch': r'$C_m$',
            'CYaw': r'$C_n$',
        }
    else:
        labels = {
            'Cl': 'CL',
            'Cd': 'CD',
            'Cs': 'CS',
            'CRoll': 'Cl (roll)',
            'CPitch': 'Cm',
            'CYaw': 'Cn',
        }
    return labels.get(coeff_name, coeff_name)


def plot_coefficients(ss: SteadyStateData,
                      x_var: str = 'alpha',
                      coefficients: Optional[List[str]] = None,
                      beta_values: Optional[List[float]] = None,
                      alpha_values: Optional[List[float]] = None,
                      use_latex: bool = True,
                      save_dir: Optional[str] = None) -> Dict[str, Figure]:
    """
    Plot aerodynamic coefficients vs angle of attack or sideslip.

    Parameters
    ----------
    ss : SteadyStateData
        Steady-state data to plot
    x_var : str
        Independent variable: 'alpha' or 'beta'
    coefficients : list, optional
        List of coefficient names to plot. Default: all
    beta_values : list, optional
        Beta values to plot (for alpha sweeps)
    alpha_values : list, optional
        Alpha values to plot (for beta sweeps)
    use_latex : bool
        Whether to use LaTeX formatting
    save_dir : str, optional
        Directory to save figures

    Returns
    -------
    dict
        Dictionary mapping coefficient names to Figure objects
    """
    if coefficients is None:
        coefficients = ['Cl', 'Cd', 'Cs', 'CRoll', 'CPitch', 'CYaw']

    figures = {}

    for coeff in coefficients:
        if not hasattr(ss, coeff):
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        coeff_data = getattr(ss, coeff)

        if ss.alphas.ndim == 2:
            # Grid data
            if x_var == 'alpha':
                x_data = ss.alphas
                fixed_data = ss.betas
                fixed_label = r'$\beta$' if use_latex else 'Beta'
            else:
                x_data = ss.betas
                fixed_data = ss.alphas
                fixed_label = r'$\alpha$' if use_latex else 'Alpha'

            # Plot each slice
            n_slices = x_data.shape[1] if x_var == 'alpha' else x_data.shape[0]

            for i in range(n_slices):
                if x_var == 'alpha':
                    x = x_data[:, i]
                    y = coeff_data[:, i]
                    fixed_val = np.mean(fixed_data[:, i])
                else:
                    x = x_data[i, :]
                    y = coeff_data[i, :]
                    fixed_val = np.mean(fixed_data[i, :])

                label = f'{fixed_label}={fixed_val:.1f}°'
                ax.plot(x, y, 'o-', label=label)

        else:
            # 1D data
            if x_var == 'alpha':
                ax.plot(ss.alphas, coeff_data, 'o-')
            else:
                ax.plot(ss.betas, coeff_data, 'o-')

        # Labels
        if use_latex:
            if x_var == 'alpha':
                ax.set_xlabel(r'$\alpha$ [deg]')
            else:
                ax.set_xlabel(r'$\beta$ [deg]')
        else:
            if x_var == 'alpha':
                ax.set_xlabel('Alpha [deg]')
            else:
                ax.set_xlabel('Beta [deg]')

        ax.set_ylabel(_get_coeff_label(coeff, use_latex))
        ax.grid(True)
        ax.legend(loc='best')

        fig.tight_layout()
        figures[coeff] = fig

        if save_dir:
            save_path = Path(save_dir) / f'{coeff}_vs_{x_var}.png'
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return figures


def plot_drag_polar(ss: SteadyStateData,
                    beta_values: Optional[List[float]] = None,
                    use_latex: bool = True,
                    show_parabola: bool = True) -> Figure:
    """
    Plot drag polar (Cd vs Cl).

    Parameters
    ----------
    ss : SteadyStateData
        Steady-state data
    beta_values : list, optional
        Beta values to include
    use_latex : bool
        Whether to use LaTeX formatting
    show_parabola : bool
        Whether to show parabolic fit

    Returns
    -------
    Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if ss.alphas.ndim == 2:
        n_beta = ss.alphas.shape[1]
        for i in range(n_beta):
            Cl = ss.Cl[:, i]
            Cd = ss.Cd[:, i]
            beta_val = np.mean(ss.betas[:, i])

            if use_latex:
                label = rf'$\beta={beta_val:.1f}^\circ$'
            else:
                label = f'Beta={beta_val:.1f} deg'

            ax.plot(Cl, Cd, 'o-', label=label)

            if show_parabola and i == 0:
                # Fit and plot parabola for first beta
                from .coefficients import calc_drag_polar_coeffs
                polar = calc_drag_polar_coeffs(Cl, Cd)
                Cl_fit = np.linspace(Cl.min(), Cl.max(), 100)
                Cd_fit = polar['Cd0'] + polar['K'] * Cl_fit ** 2
                ax.plot(Cl_fit, Cd_fit, '--', color='gray', alpha=0.7,
                        label=f"Cd0={polar['Cd0']:.4f}, K={polar['K']:.4f}")
    else:
        ax.plot(ss.Cl, ss.Cd, 'o-')

        if show_parabola:
            from .coefficients import calc_drag_polar_coeffs
            polar = calc_drag_polar_coeffs(ss.Cl, ss.Cd)
            Cl_fit = np.linspace(ss.Cl.min(), ss.Cl.max(), 100)
            Cd_fit = polar['Cd0'] + polar['K'] * Cl_fit ** 2
            ax.plot(Cl_fit, Cd_fit, '--', color='gray', alpha=0.7,
                    label=f"Cd0={polar['Cd0']:.4f}, K={polar['K']:.4f}")

    if use_latex:
        ax.set_xlabel(r'$C_L$')
        ax.set_ylabel(r'$C_D$')
    else:
        ax.set_xlabel('CL')
        ax.set_ylabel('CD')

    ax.grid(True)
    ax.legend(loc='best')
    fig.tight_layout()

    return fig


def plot_pitching_moment(ss: SteadyStateData,
                         x_var: str = 'Cl',
                         use_latex: bool = True) -> Figure:
    """
    Plot pitching moment coefficient.

    Parameters
    ----------
    ss : SteadyStateData
        Steady-state data
    x_var : str
        X-axis variable: 'Cl' or 'alpha'
    use_latex : bool
        Whether to use LaTeX formatting

    Returns
    -------
    Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if ss.alphas.ndim == 2:
        n_beta = ss.alphas.shape[1]
        for i in range(n_beta):
            if x_var == 'Cl':
                x = ss.Cl[:, i]
            else:
                x = ss.alphas[:, i]

            y = ss.CPitch[:, i]
            beta_val = np.mean(ss.betas[:, i])

            if use_latex:
                label = rf'$\beta={beta_val:.1f}^\circ$'
            else:
                label = f'Beta={beta_val:.1f} deg'

            ax.plot(x, y, 'o-', label=label)
    else:
        if x_var == 'Cl':
            ax.plot(ss.Cl, ss.CPitch, 'o-')
        else:
            ax.plot(ss.alphas, ss.CPitch, 'o-')

    if use_latex:
        if x_var == 'Cl':
            ax.set_xlabel(r'$C_L$')
        else:
            ax.set_xlabel(r'$\alpha$ [deg]')
        ax.set_ylabel(r'$C_m$')
    else:
        if x_var == 'Cl':
            ax.set_xlabel('CL')
        else:
            ax.set_xlabel('Alpha [deg]')
        ax.set_ylabel('Cm')

    ax.grid(True)
    ax.legend(loc='best')
    fig.tight_layout()

    return fig


def plot_lift_drag_ratio(ss: SteadyStateData,
                         use_latex: bool = True) -> Figure:
    """
    Plot lift-to-drag ratio vs angle of attack.

    Parameters
    ----------
    ss : SteadyStateData
        Steady-state data
    use_latex : bool
        Whether to use LaTeX formatting

    Returns
    -------
    Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if ss.alphas.ndim == 2:
        n_beta = ss.alphas.shape[1]
        for i in range(n_beta):
            alpha = ss.alphas[:, i]
            L_D = ss.Cl[:, i] / ss.Cd[:, i]
            beta_val = np.mean(ss.betas[:, i])

            if use_latex:
                label = rf'$\beta={beta_val:.1f}^\circ$'
            else:
                label = f'Beta={beta_val:.1f} deg'

            ax.plot(alpha, L_D, 'o-', label=label)
    else:
        L_D = ss.Cl / ss.Cd
        ax.plot(ss.alphas, L_D, 'o-')

    if use_latex:
        ax.set_xlabel(r'$\alpha$ [deg]')
        ax.set_ylabel(r'$L/D$')
    else:
        ax.set_xlabel('Alpha [deg]')
        ax.set_ylabel('L/D')

    ax.grid(True)
    ax.legend(loc='best')
    fig.tight_layout()

    return fig


def plot_lateral_directional(ss: SteadyStateData,
                             use_latex: bool = True) -> Tuple[Figure, Figure]:
    """
    Plot lateral-directional coefficients vs beta.

    Parameters
    ----------
    ss : SteadyStateData
        Steady-state data
    use_latex : bool
        Whether to use LaTeX formatting

    Returns
    -------
    tuple
        (side_force_fig, moment_fig)
    """
    # Side force figure
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    if ss.betas.ndim == 2:
        n_alpha = ss.betas.shape[0]
        for i in range(n_alpha):
            beta = ss.betas[i, :]
            Cs = ss.Cs[i, :]
            alpha_val = np.mean(ss.alphas[i, :])

            if use_latex:
                label = rf'$\alpha={alpha_val:.1f}^\circ$'
            else:
                label = f'Alpha={alpha_val:.1f} deg'

            ax1.plot(beta, Cs, 'o-', label=label)
    else:
        ax1.plot(ss.betas, ss.Cs, 'o-')

    if use_latex:
        ax1.set_xlabel(r'$\beta$ [deg]')
        ax1.set_ylabel(r'$C_S$')
    else:
        ax1.set_xlabel('Beta [deg]')
        ax1.set_ylabel('CS')

    ax1.grid(True)
    ax1.legend(loc='best')
    fig1.tight_layout()

    # Moment figure
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(14, 6))

    if ss.betas.ndim == 2:
        n_alpha = ss.betas.shape[0]
        for i in range(n_alpha):
            beta = ss.betas[i, :]
            CRoll = ss.CRoll[i, :]
            CYaw = ss.CYaw[i, :]
            alpha_val = np.mean(ss.alphas[i, :])

            if use_latex:
                label = rf'$\alpha={alpha_val:.1f}^\circ$'
            else:
                label = f'Alpha={alpha_val:.1f} deg'

            ax2.plot(beta, CRoll, 'o-', label=label)
            ax3.plot(beta, CYaw, 'o-', label=label)
    else:
        ax2.plot(ss.betas, ss.CRoll, 'o-')
        ax3.plot(ss.betas, ss.CYaw, 'o-')

    if use_latex:
        ax2.set_xlabel(r'$\beta$ [deg]')
        ax2.set_ylabel(r'$C_\ell$')
        ax3.set_xlabel(r'$\beta$ [deg]')
        ax3.set_ylabel(r'$C_n$')
    else:
        ax2.set_xlabel('Beta [deg]')
        ax2.set_ylabel('Cl (roll)')
        ax3.set_xlabel('Beta [deg]')
        ax3.set_ylabel('Cn')

    ax2.grid(True)
    ax3.grid(True)
    ax2.legend(loc='best')
    ax3.legend(loc='best')
    fig2.tight_layout()

    return fig1, fig2


def plot_surface(ss: SteadyStateData,
                 coeff: str = 'Cl',
                 use_latex: bool = True,
                 colormap: str = 'RdBu_r') -> Figure:
    """
    Create a 3D surface plot of a coefficient.

    Parameters
    ----------
    ss : SteadyStateData
        Steady-state data (must be 2D grid)
    coeff : str
        Coefficient name to plot
    use_latex : bool
        Whether to use LaTeX formatting
    colormap : str
        Matplotlib colormap name

    Returns
    -------
    Figure
        Matplotlib figure
    """
    if ss.alphas.ndim != 2:
        raise ValueError("Surface plot requires 2D grid data")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    coeff_data = getattr(ss, coeff)

    surf = ax.plot_surface(ss.alphas, ss.betas, coeff_data,
                           cmap=colormap, alpha=0.8,
                           linewidth=0, antialiased=True)

    if use_latex:
        ax.set_xlabel(r'$\alpha$ [deg]')
        ax.set_ylabel(r'$\beta$ [deg]')
        ax.set_zlabel(_get_coeff_label(coeff, use_latex))
    else:
        ax.set_xlabel('Alpha [deg]')
        ax.set_ylabel('Beta [deg]')
        ax.set_zlabel(_get_coeff_label(coeff, use_latex))

    fig.colorbar(surf, shrink=0.5, aspect=10, label=_get_coeff_label(coeff, use_latex))
    fig.tight_layout()

    return fig


def save_all_figures(figures: Dict[str, Figure],
                     directory: str,
                     formats: List[str] = ['png', 'pdf'],
                     dpi: int = 150) -> None:
    """
    Save all figures to a directory.

    Parameters
    ----------
    figures : dict
        Dictionary mapping names to Figure objects
    directory : str
        Output directory
    formats : list
        List of file formats to save
    dpi : int
        Resolution for raster formats
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    for name, fig in figures.items():
        for fmt in formats:
            filepath = directory / f'{name}.{fmt}'
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
