"""
Test Case Model
===============

Represents a single wind tunnel test case with its data and metadata.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from datetime import datetime
from pathlib import Path
import uuid

if TYPE_CHECKING:
    from utils.windtunnel.units import UnitConverter


@dataclass
class TunnelConditions:
    """
    Tunnel flow conditions for a test case.

    Attributes
    ----------
    Q : np.ndarray
        Dynamic pressure (psi)
    Q_mks : np.ndarray
        Dynamic pressure (Pa)
    U_inf : np.ndarray
        Freestream velocity (m/s)
    rho : np.ndarray
        Air density (kg/m^3)
    T : np.ndarray
        Temperature (C)
    P_tot : np.ndarray
        Total pressure (Pa)
    Re : np.ndarray
        Reynolds number
    Mach : np.ndarray
        Mach number
    """
    Q: np.ndarray = field(default_factory=lambda: np.array([]))
    Q_mks: np.ndarray = field(default_factory=lambda: np.array([]))
    U_inf: np.ndarray = field(default_factory=lambda: np.array([]))
    rho: np.ndarray = field(default_factory=lambda: np.array([]))
    T: np.ndarray = field(default_factory=lambda: np.array([]))
    P_tot: np.ndarray = field(default_factory=lambda: np.array([]))
    Re: np.ndarray = field(default_factory=lambda: np.array([]))
    Mach: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def mean_Q(self) -> float:
        """Mean dynamic pressure (psi)."""
        return float(np.mean(self.Q)) if len(self.Q) > 0 else 0.0

    @property
    def mean_velocity(self) -> float:
        """Mean freestream velocity (m/s)."""
        return float(np.mean(self.U_inf)) if len(self.U_inf) > 0 else 0.0

    @property
    def mean_Re(self) -> float:
        """Mean Reynolds number."""
        return float(np.mean(self.Re)) if len(self.Re) > 0 else 0.0

    @property
    def mean_Mach(self) -> float:
        """Mean Mach number."""
        return float(np.mean(self.Mach)) if len(self.Mach) > 0 else 0.0

    @property
    def mean_density(self) -> float:
        """Mean air density (kg/m^3)."""
        return float(np.mean(self.rho)) if len(self.rho) > 0 else 0.0

    @property
    def mean_temperature(self) -> float:
        """Mean temperature (C)."""
        return float(np.mean(self.T)) if len(self.T) > 0 else 0.0


@dataclass
class TestCase:
    """
    Represents a single wind tunnel test case.

    Attributes
    ----------
    id : str
        Unique identifier for the case
    name : str
        Display name for the case
    filepath : Path
        Path to the source data file(s)
    date : datetime
        Test date
    run_number : int
        Run number identifier
    visible : bool
        Whether to display this case in plots
    color : str
        Plot color for this case
    marker : str
        Plot marker style
    metadata : dict
        Additional test metadata
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    filepath: Optional[Path] = None
    date: Optional[datetime] = None
    run_number: int = 0
    visible: bool = True
    color: str = "#1f77b4"
    marker: str = "o"
    linestyle: str = "-"
    geometry_name: str = "Default"
    calibration_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Test conditions (mean values for display)
    mach_number: Optional[float] = None
    reynolds_number: Optional[float] = None
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    velocity: Optional[float] = None
    density: Optional[float] = None

    # Angle ranges
    alpha_min: float = 0.0
    alpha_max: float = 0.0
    beta_values: List[float] = field(default_factory=list)

    # Processed data (stored as numpy arrays)
    alphas: np.ndarray = field(default_factory=lambda: np.array([]))
    betas: np.ndarray = field(default_factory=lambda: np.array([]))
    Cl: np.ndarray = field(default_factory=lambda: np.array([]))
    Cd: np.ndarray = field(default_factory=lambda: np.array([]))
    Cs: np.ndarray = field(default_factory=lambda: np.array([]))
    CRoll: np.ndarray = field(default_factory=lambda: np.array([]))
    CPitch: np.ndarray = field(default_factory=lambda: np.array([]))
    CYaw: np.ndarray = field(default_factory=lambda: np.array([]))

    # Standard deviations (from time-series within each point)
    Cl_std: np.ndarray = field(default_factory=lambda: np.array([]))
    Cd_std: np.ndarray = field(default_factory=lambda: np.array([]))
    Cs_std: np.ndarray = field(default_factory=lambda: np.array([]))
    CRoll_std: np.ndarray = field(default_factory=lambda: np.array([]))
    CPitch_std: np.ndarray = field(default_factory=lambda: np.array([]))
    CYaw_std: np.ndarray = field(default_factory=lambda: np.array([]))

    # Per-point tunnel conditions arrays
    machs: np.ndarray = field(default_factory=lambda: np.array([]))
    reynolds: np.ndarray = field(default_factory=lambda: np.array([]))
    velocities: np.ndarray = field(default_factory=lambda: np.array([]))
    densities: np.ndarray = field(default_factory=lambda: np.array([]))
    temperatures: np.ndarray = field(default_factory=lambda: np.array([]))
    dynamic_pressures: np.ndarray = field(default_factory=lambda: np.array([]))
    total_pressures: np.ndarray = field(default_factory=lambda: np.array([]))

    # Blockage-corrected arrays (populated when a tunnel correction is
    # active; otherwise empty and the uncorrected arrays are primary).
    alphas_corr: np.ndarray = field(default_factory=lambda: np.array([]))
    Cl_corr: np.ndarray = field(default_factory=lambda: np.array([]))
    Cd_corr: np.ndarray = field(default_factory=lambda: np.array([]))
    blockage_epsilon: np.ndarray = field(
        default_factory=lambda: np.array([]))
    blockage_method: str = 'none'

    # User-defined calculator results: variable name -> per-point mean
    # array reshaped to match self.alphas.shape.  Populated by
    # DataController._apply_calc_rules_to_case after reduction.
    custom_vars: Dict[str, np.ndarray] = field(default_factory=dict)
    # Parallel dict of per-point std-dev arrays (same shape as the
    # mean arrays).  Used for sigma shading on plots.
    custom_vars_std: Dict[str, np.ndarray] = field(default_factory=dict)

    # Tunnel conditions object
    tunnel_conditions: TunnelConditions = field(default_factory=TunnelConditions)

    # WRF forces (stored in internal IPS units: lbf)
    lift_forces: np.ndarray = field(default_factory=lambda: np.array([]))
    drag_forces: np.ndarray = field(default_factory=lambda: np.array([]))
    side_forces: np.ndarray = field(default_factory=lambda: np.array([]))

    # WRF moments (stored in internal IPS units: lb-in)
    roll_moments: np.ndarray = field(default_factory=lambda: np.array([]))
    pitch_moments: np.ndarray = field(default_factory=lambda: np.array([]))
    yaw_moments: np.ndarray = field(default_factory=lambda: np.array([]))

    # Balance element forces (stored in internal IPS units: lbf)
    elem_N1: np.ndarray = field(default_factory=lambda: np.array([]))
    elem_N2: np.ndarray = field(default_factory=lambda: np.array([]))
    elem_Y1: np.ndarray = field(default_factory=lambda: np.array([]))
    elem_Y2: np.ndarray = field(default_factory=lambda: np.array([]))
    elem_Ax: np.ndarray = field(default_factory=lambda: np.array([]))
    elem_Roll: np.ndarray = field(default_factory=lambda: np.array([]))

    # Raw DAQ object reference
    daq: Any = None

    def __post_init__(self):
        """Initialize computed fields."""
        if not self.name and self.filepath:
            self.name = Path(self.filepath).stem

    @property
    def has_data(self) -> bool:
        """Check if case has processed data."""
        return len(self.alphas) > 0

    @property
    def n_points(self) -> int:
        """Number of data points."""
        return len(self.alphas.flatten()) if self.alphas.size > 0 else 0

    @property
    def has_tunnel_conditions(self) -> bool:
        """Check if tunnel conditions are available."""
        return (len(self.tunnel_conditions.Q) > 0 or
                self.mach_number is not None or
                len(self.machs) > 0)

    @property
    def description(self) -> str:
        """Generate a short description."""
        parts = []
        if self.run_number:
            parts.append(f"Run {self.run_number}")

        # Prefer mean values from tunnel_conditions
        mach = self.mach_number
        if mach is None and len(self.machs) > 0:
            mach = float(np.mean(self.machs))
        if mach is None and self.tunnel_conditions.mean_Mach > 0:
            mach = self.tunnel_conditions.mean_Mach
        if mach:
            parts.append(f"M={mach:.2f}")

        re = self.reynolds_number
        if re is None and len(self.reynolds) > 0:
            re = float(np.mean(self.reynolds))
        if re is None and self.tunnel_conditions.mean_Re > 0:
            re = self.tunnel_conditions.mean_Re
        if re:
            parts.append(f"Re={re/1e6:.2f}M")

        return ", ".join(parts) if parts else self.name

    def update_from_tunnel_conditions(self):
        """Update mean values from tunnel conditions arrays."""
        if len(self.tunnel_conditions.Mach) > 0:
            self.mach_number = self.tunnel_conditions.mean_Mach
        if len(self.tunnel_conditions.Re) > 0:
            self.reynolds_number = self.tunnel_conditions.mean_Re
        if len(self.tunnel_conditions.U_inf) > 0:
            self.velocity = self.tunnel_conditions.mean_velocity
        if len(self.tunnel_conditions.rho) > 0:
            self.density = self.tunnel_conditions.mean_density
        if len(self.tunnel_conditions.T) > 0:
            self.temperature = self.tunnel_conditions.mean_temperature
        if len(self.tunnel_conditions.Q) > 0:
            self.pressure = self.tunnel_conditions.mean_Q

        # Also update per-point arrays
        if len(self.tunnel_conditions.Mach) > 0:
            self.machs = self.tunnel_conditions.Mach.flatten()
        if len(self.tunnel_conditions.Re) > 0:
            self.reynolds = self.tunnel_conditions.Re.flatten()
        if len(self.tunnel_conditions.U_inf) > 0:
            self.velocities = self.tunnel_conditions.U_inf.flatten()
        if len(self.tunnel_conditions.rho) > 0:
            self.densities = self.tunnel_conditions.rho.flatten()
        if len(self.tunnel_conditions.T) > 0:
            self.temperatures = self.tunnel_conditions.T.flatten()
        if len(self.tunnel_conditions.Q) > 0:
            self.dynamic_pressures = self.tunnel_conditions.Q.flatten()

    def get_coefficient(self, name: str) -> np.ndarray:
        """Get coefficient data by name."""
        coeff_map = {
            'Cl': self.Cl, 'CL': self.Cl,
            'Cd': self.Cd, 'CD': self.Cd,
            'Cs': self.Cs, 'CY': self.Cs,
            'CRoll': self.CRoll, 'Cl_roll': self.CRoll,
            'CPitch': self.CPitch, 'Cm': self.CPitch,
            'CYaw': self.CYaw, 'Cn': self.CYaw,
            'Alpha': self.alphas, 'alpha': self.alphas,
            'Beta': self.betas, 'beta': self.betas,
            'Mach': self.machs, 'mach': self.machs,
            'Re': self.reynolds, 're': self.reynolds,
            'Q': self.dynamic_pressures, 'q': self.dynamic_pressures,
            'Velocity': self.velocities, 'velocity': self.velocities,
            'U_inf': self.velocities, 'u_inf': self.velocities,
            # Standard deviations
            'Cl_std': self.Cl_std, 'CL_std': self.Cl_std,
            'Cd_std': self.Cd_std, 'CD_std': self.Cd_std,
            'Cs_std': self.Cs_std, 'CY_std': self.Cs_std,
            'CRoll_std': self.CRoll_std,
            'CPitch_std': self.CPitch_std, 'Cm_std': self.CPitch_std,
            'CYaw_std': self.CYaw_std, 'Cn_std': self.CYaw_std,
        }
        return coeff_map.get(name, np.array([]))

    def get_sweep_at_beta(self, beta: float, tolerance: float = 0.5) -> Dict[str, np.ndarray]:
        """Extract data at a specific beta angle."""
        if self.betas.ndim == 2:
            # Grid data
            beta_avg = np.mean(self.betas, axis=0)
            idx = np.argmin(np.abs(beta_avg - beta))
            return {
                'alpha': self.alphas[:, idx],
                'Cl': self.Cl[:, idx],
                'Cd': self.Cd[:, idx],
                'Cs': self.Cs[:, idx],
                'CRoll': self.CRoll[:, idx],
                'CPitch': self.CPitch[:, idx],
                'CYaw': self.CYaw[:, idx],
                'beta': beta_avg[idx]
            }
        else:
            # 1D data
            mask = np.abs(self.betas - beta) < tolerance
            return {
                'alpha': self.alphas[mask],
                'Cl': self.Cl[mask],
                'Cd': self.Cd[mask],
                'Cs': self.Cs[mask],
                'CRoll': self.CRoll[mask],
                'CPitch': self.CPitch[mask],
                'CYaw': self.CYaw[mask],
                'beta': beta
            }

    @property
    def has_forces(self) -> bool:
        """Check if WRF forces are available."""
        return len(self.lift_forces) > 0

    @property
    def has_moments(self) -> bool:
        """Check if WRF moments are available."""
        return len(self.roll_moments) > 0

    def get_forces_display(self, converter: 'UnitConverter') -> Dict[str, np.ndarray]:
        """
        Return forces converted to display units.

        Parameters
        ----------
        converter : UnitConverter
            Unit converter for the desired output system

        Returns
        -------
        dict
            Dictionary with 'Lift', 'Drag', 'Side' arrays in output units
        """
        return {
            'Lift': converter.convert_force(self.lift_forces) if len(self.lift_forces) > 0 else np.array([]),
            'Drag': converter.convert_force(self.drag_forces) if len(self.drag_forces) > 0 else np.array([]),
            'Side': converter.convert_force(self.side_forces) if len(self.side_forces) > 0 else np.array([]),
        }

    def get_moments_display(self, converter: 'UnitConverter') -> Dict[str, np.ndarray]:
        """
        Return moments converted to display units.

        Parameters
        ----------
        converter : UnitConverter
            Unit converter for the desired output system

        Returns
        -------
        dict
            Dictionary with 'Roll', 'Pitch', 'Yaw' arrays in output units
        """
        return {
            'Roll': converter.convert_moment(self.roll_moments) if len(self.roll_moments) > 0 else np.array([]),
            'Pitch': converter.convert_moment(self.pitch_moments) if len(self.pitch_moments) > 0 else np.array([]),
            'Yaw': converter.convert_moment(self.yaw_moments) if len(self.yaw_moments) > 0 else np.array([]),
        }

    def get_tunnel_conditions_display(self, converter: 'UnitConverter') -> Dict[str, Any]:
        """
        Return tunnel conditions converted to display units.

        Parameters
        ----------
        converter : UnitConverter
            Unit converter for the desired output system

        Returns
        -------
        dict
            Dictionary with converted tunnel condition values
        """
        result = {}

        # Dynamic pressure (from psi)
        if self.pressure is not None:
            result['Q'] = converter.convert_pressure(self.pressure)
        elif len(self.dynamic_pressures) > 0:
            result['Q'] = converter.convert_pressure(np.mean(self.dynamic_pressures))
        else:
            result['Q'] = None

        # Velocity (from m/s)
        if self.velocity is not None:
            result['U_inf'] = converter.convert_velocity(self.velocity)
        elif len(self.velocities) > 0:
            result['U_inf'] = converter.convert_velocity(np.mean(self.velocities))
        else:
            result['U_inf'] = None

        # Density (from kg/m^3)
        if self.density is not None:
            result['rho'] = converter.convert_density(self.density)
        elif len(self.densities) > 0:
            result['rho'] = converter.convert_density(np.mean(self.densities))
        else:
            result['rho'] = None

        # Temperature (from Celsius)
        if self.temperature is not None:
            result['T'] = converter.convert_temperature(self.temperature)
        elif len(self.temperatures) > 0:
            result['T'] = converter.convert_temperature(np.mean(self.temperatures))
        else:
            result['T'] = None

        # Dimensionless quantities (no conversion needed)
        result['Mach'] = self.mach_number if self.mach_number is not None else (
            float(np.mean(self.machs)) if len(self.machs) > 0 else None
        )
        result['Re'] = self.reynolds_number if self.reynolds_number is not None else (
            float(np.mean(self.reynolds)) if len(self.reynolds) > 0 else None
        )

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'filepath': str(self.filepath) if self.filepath else None,
            'run_number': self.run_number,
            'visible': self.visible,
            'color': self.color,
            'marker': self.marker,
            'metadata': self.metadata,
            'mach_number': self.mach_number,
            'reynolds_number': self.reynolds_number,
            'velocity': self.velocity,
            'density': self.density,
            'temperature': self.temperature,
            'pressure': self.pressure,
            'alpha_min': self.alpha_min,
            'alpha_max': self.alpha_max,
            'beta_values': self.beta_values,
        }


class CaseCollection:
    """
    Collection of test cases with management operations.
    """

    def __init__(self):
        self._cases: Dict[str, TestCase] = {}
        self._order: List[str] = []  # Maintain insertion order

    def add(self, case: TestCase) -> None:
        """Add a test case to the collection."""
        self._cases[case.id] = case
        if case.id not in self._order:
            self._order.append(case.id)

    def remove(self, case_id: str) -> Optional[TestCase]:
        """Remove and return a test case."""
        if case_id in self._cases:
            case = self._cases.pop(case_id)
            self._order.remove(case_id)
            return case
        return None

    def get(self, case_id: str) -> Optional[TestCase]:
        """Get a test case by ID."""
        return self._cases.get(case_id)

    def __getitem__(self, case_id: str) -> TestCase:
        return self._cases[case_id]

    def __contains__(self, case_id: str) -> bool:
        return case_id in self._cases

    def __len__(self) -> int:
        return len(self._cases)

    def __iter__(self):
        """Iterate in order."""
        for case_id in self._order:
            yield self._cases[case_id]

    @property
    def visible_cases(self) -> List[TestCase]:
        """Get all visible cases."""
        return [c for c in self if c.visible]

    @property
    def all_beta_values(self) -> List[float]:
        """Get all unique beta values across cases (rounded to 1 decimal)."""
        betas = set()
        for case in self:
            if case.has_data:
                if case.betas.ndim == 2:
                    betas.update(round(float(v), 1) for v in np.mean(case.betas, axis=0))
                else:
                    betas.update(round(float(v), 1) for v in case.betas.flatten())
        return sorted(betas)

    @property
    def all_alpha_values(self) -> List[float]:
        """Get all unique alpha values across cases (rounded to 1 decimal)."""
        alphas = set()
        for case in self:
            if case.has_data:
                if case.alphas.ndim == 2:
                    # 2D: each row is one alpha; take row mean
                    alphas.update(round(float(v), 1) for v in np.mean(case.alphas, axis=1))
                else:
                    alphas.update(round(float(v), 1) for v in case.alphas.flatten())
        return sorted(alphas)

    @property
    def all_mach_numbers(self) -> List[float]:
        """Get all unique PER-POINT Mach numbers across cases.

        A single config now holds every speed step (a velocity/Mach sweep is
        one case with multiple distinct per-point Mach values), so the filter
        must enumerate the DISTINCT per-point Machs — not one-per-case — for
        the user to select which speed step to visualize.

        Rounded to 3 decimals: a low-speed (LSWT) sweep spans only Mach
        0-0.1, so 2 decimals would merge adjacent speed steps (e.g. Hz 20 ->
        M0.031 and Hz 25 -> M0.038 both round to 0.03). 3 decimals keeps the
        distinct steps visible for both the low-speed (0-0.1) and subsonic
        (0-0.6) tunnels.
        """
        machs = set()
        for case in self:
            if not case.has_data:
                continue
            if len(case.machs) > 0:
                for m in np.asarray(case.machs).flatten():
                    machs.add(round(float(m), 3))
            elif case.mach_number is not None:
                machs.add(round(case.mach_number, 3))
        return sorted(machs)

    @property
    def all_reynolds_numbers(self) -> List[float]:
        """Get all unique Reynolds numbers."""
        reynolds = set()
        for case in self:
            if case.reynolds_number is not None:
                reynolds.add(case.reynolds_number)
            elif len(case.reynolds) > 0:
                reynolds.add(float(np.mean(case.reynolds)))
        return sorted(reynolds)

    @property
    def all_velocities(self) -> List[float]:
        """Get all unique velocities."""
        velocities = set()
        for case in self:
            if case.velocity is not None:
                velocities.add(round(case.velocity, 1))
            elif len(case.velocities) > 0:
                velocities.add(round(float(np.mean(case.velocities)), 1))
        return sorted(velocities)

    def clear(self) -> None:
        """Remove all cases."""
        self._cases.clear()
        self._order.clear()
