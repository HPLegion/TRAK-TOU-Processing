from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.constants import atomic_mass as AMU
from scipy.constants import elementary_charge as Q_E
from scipy.constants import speed_of_light as C_0


def _velocity(time: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Compute velocity as time derivative of position"""
    if len(time) > 2:
        return np.gradient(position, time, edge_order=2)
    else:
        return np.zeros_like(time)


@dataclass(frozen=True)
class Particle:
    """Container for particle trajectory data
    Exposes derived information for convenient use

    Velocity is derived from the particle position
    This makes certain assumptions about the domain edges
    See: numpy gradient function docs

    CHANGES TO THE INTERNAL DATA/ARRAYS ARE NOT PERMITTED!
    """

    pid: int  #: Particle ID
    mass: float  #: Mass (amu)
    charge: int  #: Charge number
    current: float  #: Current (unknown units)

    t: np.ndarray  #: Timestep (s)
    x: np.ndarray  #: X position (m)
    y: np.ndarray  #: Y position (m)
    z: np.ndarray  #: Z position (m)

    def __len__(self) -> int:
        """Number of sample points"""
        return len(self.t)

    @property
    def has_data(self) -> bool:
        """True iff trajectory consists of at least two points"""
        return len(self) > 2

    @property
    def mass_si(self) -> float:
        """Mass (kg)"""
        return self.mass * AMU

    @property
    def charge_si(self) -> float:
        """Charge (C)"""
        return self.charge * Q_E

    @cached_property
    def v_x(self) -> np.ndarray:
        """X velocity (m/s)"""
        return _velocity(self.t, self.x)

    @cached_property
    def v_y(self) -> np.ndarray:
        """Y velocity (m/s)"""
        return _velocity(self.t, self.y)

    @cached_property
    def v_z(self) -> np.ndarray:
        """Z velocity (m/s)"""
        return _velocity(self.t, self.z)

    @cached_property
    def v_abs(self):
        """Absolute velocity / speed (m/s)"""
        v_vec = np.column_stack((self.v_x, self.v_y, self.v_z))
        v_abs = np.linalg.norm(v_vec, axis=1)
        return v_abs

    @cached_property
    def r(self) -> np.ndarray:
        """Cylindrical coordinate r (m)"""
        return np.sqrt(self.x**2 + self.y**2)

    @cached_property
    def phi_rad(self) -> np.ndarray:
        """Cylindrical coordinate phi (rad)"""
        return np.arctan2(self.y, self.x)

    @property
    def phi(self) -> np.ndarray:
        """Cylindrical coordinate phi (deg)"""
        return np.rad2deg(self.phi_rad)

    @cached_property
    def v_r(self) -> np.ndarray:
        """Radial velocity (m/s)"""
        return _velocity(self.t, self.r)

    @cached_property
    def omega_phi_rad(self) -> np.ndarray:
        """Angular speed (rad/s)"""
        return _velocity(self.t, np.unwrap(self.phi_rad))

    @property
    def omega_phi(self) -> np.ndarray:
        """Angular speed (deg/s)"""
        return np.rad2deg(self.omega_phi_rad)

    @property
    def v_phi(self) -> np.ndarray:
        """Tangential velocity (m/s)"""
        return self.r * self.omega_phi_rad

    @property
    def beta_x(self) -> np.ndarray:
        """X relativistic beta (velocity in units of c)"""
        return self.v_x / C_0

    @property
    def beta_y(self) -> np.ndarray:
        """Y relativistic beta (velocity in units of c)"""
        return self.v_y / C_0

    @property
    def beta_z(self) -> np.ndarray:
        """Z relativistic beta (velocity in units of c)"""
        return self.v_z / C_0

    @property
    def beta_abs(self) -> np.ndarray:
        """Absolute relativistic beta (velocity in units of c)"""
        return self.v_abs / C_0

    @cached_property
    def gamma(self) -> np.ndarray:
        """Relativistic gamma"""
        return 1 / np.sqrt(1 - self.beta_abs**2)

    @property
    def kin_energy(self) -> np.ndarray:
        """Kinetic energy (eV)"""
        return self.kin_energy_si / Q_E

    @property
    def kin_energy_si(self) -> np.ndarray:
        """Kinetic energy (J)"""
        return (self.gamma - 1) * self.mass_si * C_0**2

    @property
    def kin_energy_long(self) -> np.ndarray:
        """Longitudinal (parallel ot z) kinetic energy (eV)"""
        return self.kin_energy * np.cos(self.ang_with_z_rad) ** 2

    @property
    def kin_energy_trans(self) -> np.ndarray:
        """Transverse (perpendicular ot z) kinetic energy (eV)"""
        return self.kin_energy * np.sin(self.ang_with_z_rad) ** 2

    @property
    def p_x(self) -> np.ndarray:
        """X normalised momentum p_x (in units of beta*gamma)"""
        return self.beta_x * self.gamma

    @property
    def p_y(self) -> np.ndarray:
        """Y normalised momentum p_y (in units of beta*gamma)"""
        return self.beta_y * self.gamma

    @property
    def p_z(self) -> np.ndarray:
        """Z normalised momentum p_z (in units of beta*gamma)"""
        return self.beta_z * self.gamma

    @property
    def p_abs(self) -> np.ndarray:
        """Absolute normalised momentum p (in units of beta*gamma)"""
        return self.beta_abs * self.gamma

    @property
    def p_x_si(self) -> np.ndarray:
        """X momentum (kg/m/s)"""
        return self.p_x * self.mass_si * C_0

    @property
    def p_y_si(self) -> np.ndarray:
        """Y momentum (kg/m/s)"""
        return self.p_y * self.mass_si * C_0

    @property
    def p_z_si(self) -> np.ndarray:
        """Z momentum (kg/m/s)"""
        return self.p_z * self.mass_si * C_0

    @property
    def p_abs_si(self) -> np.ndarray:
        """Absolute momentum (kg/m/s)"""
        return self.p_abs * self.mass_si * C_0

    @property
    def ang_with_z_rad(self) -> np.ndarray:
        """Angle between velocity vector and Z axis (rad)"""
        return np.arccos(self.v_z / self.v_abs)

    @property
    def ang_with_z(self) -> np.ndarray:
        """Angle between velocity vector and Z axis (deg)"""
        return np.rad2deg(self.ang_with_z_rad)

    def _range_mask(self, zmin: float | None = None, zmax: float | None = None) -> np.ndarray:
        """Mask for a given z range"""
        if not zmin:  # TODO: CHECK FOR NONE
            zmin = np.amin(self.z)
        if not zmax:  # TODO: CHECK FOR NONE
            zmax = np.amax(self.z)
        return (zmin <= self.z) & (self.z <= zmax)

    def max_ang_with_z(
        self, zmin: float | None = None, zmax: float | None = None
    ) -> tuple[float, float]:
        """Max. angle with the Z axis along this trajectory.
        Z range can be limited by setting arguments.
        If the max value occurs more than once the first appearance is returned.
        Returns a tuple (z, angle(z)) (m, deg)
        """
        mask = self._range_mask(zmin=zmin, zmax=zmax)
        arg = np.argmax(self.ang_with_z[mask])
        return (self.z[mask][arg], self.ang_with_z[mask][arg])

    def max_kin_energy_trans(
        self, zmin: float | None = None, zmax: float | None = None
    ) -> tuple[float, float]:
        """Max. transverse e_kin along this trajectory.
        Z range can be limited by setting arguments.
        If the max value occurs more than once the first appearance is returned.
        Returns a tuple (z, e_trans(z)) (m, eV)
        """
        mask = self._range_mask(zmin=zmin, zmax=zmax)
        arg = np.argmax(self.kin_energy_trans[mask])
        return (self.z[mask][arg], self.kin_energy_trans[mask][arg])

    def mean_ang_with_z(
        self, zmin: float | None = None, zmax: float | None = None
    ) -> tuple[float, float]:
        """Mean angle with the Z axis along this trajectory.
        Z range can be limited by setting arguments.
        Returns a tuple (mean_z_in_range, mean_angle_in_range) (m, deg)
        """
        mask = self._range_mask(zmin=zmin, zmax=zmax)
        return (self.z[mask].mean(), self.ang_with_z[mask].mean())

    def mean_kin_energy_trans(
        self, zmin: float | None = None, zmax: float | None = None
    ) -> tuple[float, float]:
        """Mean transverse e_kin along this trajectory.
        Z range can be limited by setting arguments.
        Returns a tuple (mean_z_in_range, mean_trans_ekin_in_range) (m, eV)
        """
        mask = self._range_mask(zmin=zmin, zmax=zmax)
        return (self.z[mask].mean(), self.kin_energy_trans[mask].mean())
