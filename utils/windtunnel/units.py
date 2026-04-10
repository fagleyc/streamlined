"""
Unit System Module
==================

Provides unit conversion capabilities for wind tunnel data output.
Supports IPS, FPS, MKS, and CGS unit systems.

All internal calculations remain in IPS (inch-pound-second).
Conversion happens only at output/display time.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Union
import numpy as np


class UnitSystem(Enum):
    """Supported unit systems for output."""
    IPS = "IPS"  # inch-pound-second (internal)
    FPS = "FPS"  # foot-pound-second
    MKS = "MKS"  # meter-kilogram-second (SI)
    CGS = "CGS"  # centimeter-gram-second


@dataclass(frozen=True)
class UnitLabels:
    """Unit labels for display purposes."""
    length: str
    area: str
    force: str
    moment: str
    pressure: str
    velocity: str
    density: str
    temperature: str


# Unit labels for each system
UNIT_LABELS: Dict[UnitSystem, UnitLabels] = {
    UnitSystem.IPS: UnitLabels(
        length="in",
        area="sq in",
        force="lbf",
        moment="lb-in",
        pressure="psi",
        velocity="ft/s",
        density="slug/ft^3",
        temperature="degF"
    ),
    UnitSystem.FPS: UnitLabels(
        length="ft",
        area="sq ft",
        force="lbf",
        moment="lb-ft",
        pressure="psf",
        velocity="ft/s",
        density="slug/ft^3",
        temperature="degF"
    ),
    UnitSystem.MKS: UnitLabels(
        length="m",
        area="m^2",
        force="N",
        moment="N-m",
        pressure="Pa",
        velocity="m/s",
        density="kg/m^3",
        temperature="degC"
    ),
    UnitSystem.CGS: UnitLabels(
        length="cm",
        area="cm^2",
        force="dyn",
        moment="dyn-cm",
        pressure="Pa",
        velocity="cm/s",
        density="g/cm^3",
        temperature="degC"
    ),
}


# Conversion factors from internal IPS units to each output system
# Internal units:
#   - Length: inches
#   - Area: square inches
#   - Force: lbf
#   - Moment: lb-in
#   - Pressure: psi
#   - Velocity: stored as m/s internally (from tunnel conditions)
#   - Density: stored as kg/m^3 internally (from tunnel conditions)
#   - Temperature: stored as Celsius internally

CONVERSION_FACTORS: Dict[UnitSystem, Dict[str, float]] = {
    UnitSystem.IPS: {
        'length': 1.0,              # in -> in
        'area': 1.0,                # sq in -> sq in
        'force': 1.0,               # lbf -> lbf
        'moment': 1.0,              # lb-in -> lb-in
        'pressure': 1.0,            # psi -> psi
        'velocity_from_mps': 3.28084,    # m/s -> ft/s
        'density_from_kgm3': 0.00194032, # kg/m^3 -> slug/ft^3
        'temp_offset': 32.0,        # C -> F: T*9/5 + 32
        'temp_scale': 1.8,          # C -> F multiplier
    },
    UnitSystem.FPS: {
        'length': 1.0 / 12.0,       # in -> ft
        'area': 1.0 / 144.0,        # sq in -> sq ft
        'force': 1.0,               # lbf -> lbf
        'moment': 1.0 / 12.0,       # lb-in -> lb-ft
        'pressure': 144.0,          # psi -> psf (lb/ft^2)
        'velocity_from_mps': 3.28084,    # m/s -> ft/s
        'density_from_kgm3': 0.00194032, # kg/m^3 -> slug/ft^3
        'temp_offset': 32.0,        # C -> F
        'temp_scale': 1.8,
    },
    UnitSystem.MKS: {
        'length': 0.0254,           # in -> m
        'area': 0.00064516,         # sq in -> m^2
        'force': 4.44822,           # lbf -> N
        'moment': 0.112985,         # lb-in -> N-m
        'pressure': 6894.76,        # psi -> Pa
        'velocity_from_mps': 1.0,   # m/s -> m/s (no conversion)
        'density_from_kgm3': 1.0,   # kg/m^3 -> kg/m^3 (no conversion)
        'temp_offset': 0.0,         # C -> C (no conversion)
        'temp_scale': 1.0,
    },
    UnitSystem.CGS: {
        'length': 2.54,             # in -> cm
        'area': 6.4516,             # sq in -> cm^2
        'force': 444822.0,          # lbf -> dyn (1 lbf = 444822 dyn)
        'moment': 11298484.0,       # lb-in -> dyn-cm
        'pressure': 6894.76,        # psi -> Pa (CGS uses Pa for pressure)
        'velocity_from_mps': 100.0, # m/s -> cm/s
        'density_from_kgm3': 0.001, # kg/m^3 -> g/cm^3
        'temp_offset': 0.0,         # C -> C (no conversion)
        'temp_scale': 1.0,
    },
}


class UnitConverter:
    """
    Converts values from internal units to the specified output unit system.

    Internal units (IPS-based):
    - Length: inches
    - Area: square inches
    - Force: lbf
    - Moment: lb-in
    - Pressure: psi

    Tunnel conditions are stored in metric:
    - Velocity: m/s
    - Density: kg/m^3
    - Temperature: Celsius

    Parameters
    ----------
    output_system : UnitSystem
        The target unit system for output

    Examples
    --------
    >>> converter = UnitConverter(UnitSystem.MKS)
    >>> converter.convert_force(10.0)  # 10 lbf -> N
    44.4822
    >>> converter.convert_length(12.0)  # 12 inches -> m
    0.3048
    """

    def __init__(self, output_system: UnitSystem = UnitSystem.IPS):
        self.output_system = output_system
        self._factors = CONVERSION_FACTORS[output_system]
        self._labels = UNIT_LABELS[output_system]

    def convert_length(self, value_inches: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert length from inches to output units.

        Parameters
        ----------
        value_inches : float or np.ndarray
            Length value(s) in inches

        Returns
        -------
        float or np.ndarray
            Length value(s) in output units
        """
        return value_inches * self._factors['length']

    def convert_area(self, value_sq_inches: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert area from square inches to output units.

        Parameters
        ----------
        value_sq_inches : float or np.ndarray
            Area value(s) in square inches

        Returns
        -------
        float or np.ndarray
            Area value(s) in output units
        """
        return value_sq_inches * self._factors['area']

    def convert_force(self, value_lbf: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert force from lbf to output units.

        Parameters
        ----------
        value_lbf : float or np.ndarray
            Force value(s) in lbf

        Returns
        -------
        float or np.ndarray
            Force value(s) in output units (lbf, N, or dyn)
        """
        return value_lbf * self._factors['force']

    def convert_moment(self, value_lb_in: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert moment from lb-in to output units.

        Parameters
        ----------
        value_lb_in : float or np.ndarray
            Moment value(s) in lb-in

        Returns
        -------
        float or np.ndarray
            Moment value(s) in output units (lb-in, lb-ft, N-m, or dyn-cm)
        """
        return value_lb_in * self._factors['moment']

    def convert_pressure(self, value_psi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert pressure from psi to output units.

        Parameters
        ----------
        value_psi : float or np.ndarray
            Pressure value(s) in psi

        Returns
        -------
        float or np.ndarray
            Pressure value(s) in output units (psi, psf, or Pa)
        """
        return value_psi * self._factors['pressure']

    def convert_velocity(self, value_mps: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert velocity from m/s (internal storage) to output units.

        Parameters
        ----------
        value_mps : float or np.ndarray
            Velocity value(s) in m/s

        Returns
        -------
        float or np.ndarray
            Velocity value(s) in output units (ft/s, m/s, or cm/s)
        """
        return value_mps * self._factors['velocity_from_mps']

    def convert_density(self, value_kgm3: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert density from kg/m^3 (internal storage) to output units.

        Parameters
        ----------
        value_kgm3 : float or np.ndarray
            Density value(s) in kg/m^3

        Returns
        -------
        float or np.ndarray
            Density value(s) in output units (slug/ft^3, kg/m^3, or g/cm^3)
        """
        return value_kgm3 * self._factors['density_from_kgm3']

    def convert_temperature(self, value_celsius: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert temperature from Celsius (internal storage) to output units.

        Parameters
        ----------
        value_celsius : float or np.ndarray
            Temperature value(s) in Celsius

        Returns
        -------
        float or np.ndarray
            Temperature value(s) in output units (degF or degC)
        """
        return value_celsius * self._factors['temp_scale'] + self._factors['temp_offset']

    def get_labels(self) -> UnitLabels:
        """
        Get unit labels for the current output system.

        Returns
        -------
        UnitLabels
            Dataclass containing unit label strings
        """
        return self._labels

    @property
    def length_label(self) -> str:
        """Get length unit label."""
        return self._labels.length

    @property
    def area_label(self) -> str:
        """Get area unit label."""
        return self._labels.area

    @property
    def force_label(self) -> str:
        """Get force unit label."""
        return self._labels.force

    @property
    def moment_label(self) -> str:
        """Get moment unit label."""
        return self._labels.moment

    @property
    def pressure_label(self) -> str:
        """Get pressure unit label."""
        return self._labels.pressure

    @property
    def velocity_label(self) -> str:
        """Get velocity unit label."""
        return self._labels.velocity

    @property
    def density_label(self) -> str:
        """Get density unit label."""
        return self._labels.density

    @property
    def temperature_label(self) -> str:
        """Get temperature unit label."""
        return self._labels.temperature

    def __repr__(self) -> str:
        return f"UnitConverter(output_system={self.output_system.value})"
