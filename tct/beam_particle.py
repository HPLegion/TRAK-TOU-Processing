"""
Contains the class TouParticle
"""

import numpy as np
from scipy.constants import (speed_of_light as C_0,
                             atomic_mass as AMU,
                             elementary_charge as Q_E,
                             #  pi as PI
                            )
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

class Particle:
    """
    A data container for particle information imported from a TOU file.
    Also extends the trajectory data with useful particle information in an easy to access way
    The velocity is derived from the particle position
    This makes certain assumptions about the domain edges
    See: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.gradient.html

    CHANGES TO THE INTERNAL DATA (PROPERTIES) ARE NOT FORSEEN AND WILL NOT BE HANDLED CORRECTLY
    THEY MAY, HOWEVER, BE POSSIBLE!
    """
    def __init__(self, trajectory, constants, safe_mode=False):
        """
        Constructor
        trajectory -- pandas dataframe as produced by the import_tou functions (i.e t,x,y,z format)
        constants -- dict with particle constants as produced by the import_tou functions
        safe mode -- In safe mode only copies of data structures are read and returned, this
                     protects the internal data structure from unwanted manipulation but may
                     decrease performance
        """
        self._safe_mode = safe_mode
        if self._safe_mode:
            self._tou_data = trajectory.copy()
        else:
            self._tou_data = trajectory

        self._id = constants["id"]
        self._mass = constants["mass"]
        self._charge = constants["charge"]
        self._current = constants["current"]
        # compute velocities
        if self.has_data:
            # compute the beta value of all timesteps (velocity)
            self._tou_data["v_x"] = np.gradient(self.x, self.t, edge_order=2)
            self._tou_data["v_y"] = np.gradient(self.y, self.t, edge_order=2)
            self._tou_data["v_z"] = np.gradient(self.z, self.t, edge_order=2)

        else:
            self._tou_data["v_x"] = []
            self._tou_data["v_y"] = []
            self._tou_data["v_z"] = []

        # Initialise fields that will be computed on demand and then memorised
        # This is for potentially more "demanding" computations where memoisation may be faster
        self._gamma = None
        self._v_abs = None

    @property
    def tou_data(self):
        """Dataframe holding the TOU Data"""
        if self._safe_mode:
            return self._tou_data.copy()
        else:
            return self._tou_data

    @property
    def has_data(self):
        """returns True if trajectory frame has at least two lines, False otherwise"""
        return self._tou_data.shape[0] > 2

    @property
    def pid(self):
        """Tou File Particle ID"""
        return self._id

    @property
    def current(self):
        """Current in unknown units"""
        return self._current

    @property
    def mass(self):
        """Mass in atomic mass units"""
        return self._mass

    @property
    def mass_si(self):
        """Mass in kg"""
        return self._mass * AMU

    @property
    def charge(self):
        """Charge number"""
        return self._charge

    @property
    def charge_si(self):
        """Charge in C"""
        return self._charge * Q_E

    @property
    def t(self):
        """numpy array of the timesteps column"""
        if self._safe_mode:
            return self.tou_data['t'].values.copy()
        else:
            return self.tou_data['t'].values

    @property
    def x(self):
        """numpy array of the x column (pos in m)"""
        if self._safe_mode:
            return self.tou_data['x'].values.copy()
        else:
            return self.tou_data['x'].values

    @property
    def y(self):
        """numpy array of the y column (pos in m)"""
        if self._safe_mode:
            return self.tou_data['y'].values.copy()
        else:
            return self.tou_data['y'].values

    @property
    def z(self):
        """numpy array of the z column (pos in m)"""
        if self._safe_mode:
            return self.tou_data['z'].values.copy()
        else:
            return self.tou_data['z'].values

    @property
    def r(self):
        """
        numpy array of the radius coordinate (cylinder coordinates) r = sqrt(x**2 + y**2) (in m)
        """
        return np.sqrt(self.x**2 + self.y**2)

    @property
    def phi(self):
        """
        numpy array of phi corrdinate (cylinder coordinates) phi = atan2(y, x) (in deg)
        """
        return np.rad2deg(np.arctan2(self.y, self.x))

    @property
    def v_x(self):
        """numpy array of the v_x column (velocity in m/s)"""
        if self._safe_mode:
            return self.tou_data['v_x'].values.copy()
        else:
            return self.tou_data['v_x'].values

    @property
    def v_y(self):
        """numpy array of the v_y column (velocity in m/s)"""
        if self._safe_mode:
            return self.tou_data['v_y'].values.copy()
        else:
            return self.tou_data['v_y'].values

    @property
    def v_z(self):
        """numpy array of the v_z column (velocity in m/s)"""
        if self._safe_mode:
            return self.tou_data['v_z'].values.copy()
        else:
            return self.tou_data['v_z'].values

    @property
    def v_abs(self):
        """numpy array of the absolute velocity (speed, in m/s)"""
        if self._v_abs is None:
            v_vec = np.column_stack((self.v_x, self.v_y, self.v_z))
            self._v_abs = np.linalg.norm(v_vec, axis=1)

        if self._safe_mode:
            return self._v_abs.copy()
        else:
            return self._v_abs

    @property
    def beta_x(self):
        """numpy array of beta_x (velocity in units of c_0)"""
        return self.v_x / C_0

    @property
    def beta_y(self):
        """numpy array of beta_y (velocity in units of c_0)"""
        return self.v_y / C_0

    @property
    def beta_z(self):
        """numpy array of beta_z (velocity in units of c_0)"""
        return self.v_z / C_0

    @property
    def beta_abs(self):
        """numpy array of the absolute velocity beta (speed, in units of c_0)"""
        return self.v_abs / C_0

    @property
    def gamma(self):
        """numpy array of gamma (velocity in units of c_0)"""
        #Compute _gamma on first demand
        if self._gamma is None:
            self._gamma = 1/np.sqrt(1-self.beta_abs**2)

        if self._safe_mode:
            return self._gamma.copy()
        else:
            return self._gamma

    @property
    def kin_energy_si(self):
        """Kinetic Energy in J"""
        return (self.gamma-1)*self.mass_si*C_0**2
        # return self.v_abs**2 * self.mass_si / 2

    @property
    def kin_energy(self):
        """Kinetic Energy in eV"""
        return self.kin_energy_si / Q_E

    @property
    def kin_energy_long(self):
        """Longitudinal (parallel ot z) Kinetic Energy in eV"""
        return self.kin_energy * np.cos(self.ang_with_z_rad)**2

    @property
    def kin_energy_trans(self):
        """Transverse (perpendicular ot z) Kinetic Energy in eV"""
        return self.kin_energy * np.sin(self.ang_with_z_rad)**2

    @property
    def p_x(self):
        """numpy array of normalised momentum p_x (in units of beta*gamma)"""
        return self.beta_x * self.gamma

    @property
    def p_y(self):
        """numpy array of normalised momentum p_y (in units of beta*gamma)"""
        return self.beta_y * self.gamma

    @property
    def p_z(self):
        """numpy array of normalised momentum p_z (in units of beta*gamma)"""
        return self.beta_z * self.gamma

    @property
    def p_abs(self):
        """numpy array of absolute normalised momentum p (in units of beta*gamma)"""
        return self.beta_abs * self.gamma

    @property
    def p_x_si(self):
        """numpy array of momentum p_x (in kg/m/s)"""
        return self.p_x * self.mass_si * C_0

    @property
    def p_y_si(self):
        """numpy array of momentum p_y (in kg/m/s)"""
        return self.p_y * self.mass_si * C_0

    @property
    def p_z_si(self):
        """numpy array of momentum p_z (in kg/m/s)"""
        return self.p_z * self.mass_si * C_0

    @property
    def p_abs_si(self):
        """numpy array of absolute momentum p (in kg/m/s)"""
        return self.p_abs * self.mass_si * C_0

    @property
    def ang_with_z_rad(self):
        """The angle between the particles velocity vector and the z axis (radians)"""
        return np.arccos(self.v_z/self.v_abs)

    @property
    def ang_with_z(self):
        """The angle between the particles velocity vector and the z axis (degree)"""
        return np.rad2deg(self.ang_with_z_rad)

    def _range_mask(self, zmin=None, zmax=None):
        """
        returns a numpy mask for a given range, replacing with limits of available data if None
        """
        if not zmin:
            zmin = np.amin(self.z)
        if not zmax:
            zmax = np.amax(self.z)
        return (zmin <= self.z) & (self.z <= zmax)

    def max_ang_with_z(self, zmin=None, zmax=None):
        """
        finds the maximum angle with the z axis along the trajectory, the z range can be limited
        if the max value occurs more than once the first appearance is returned
        returns a tuple (z, angle(z)) (angle in degrees)
        """
        mask = self._range_mask(zmin=zmin, zmax=zmax)
        arg = np.argmax(self.ang_with_z[mask])
        return (self.z[mask][arg], self.ang_with_z[mask][arg])

    def max_kin_energy_trans(self, zmin=None, zmax=None):
        """
        finds the maximum transverse e_kin along the trajectory, the z range can be limited
        if the max value occurs more than once the first appearance is returned
        returns a tuple (z, angle(z)) (e_kin in eV)
        """
        mask = self._range_mask(zmin=zmin, zmax=zmax)
        arg = np.argmax(self.kin_energy_trans[mask])
        return (self.z[mask][arg], self.kin_energy_trans[mask][arg])

    def mean_ang_with_z(self, zmin=None, zmax=None):
        """
        finds the mean angle with the z axis along the trajectory, the z range can be limited
        if the max value occurs more than once the first appearance is returned
        returns a tuple (mean_z_in_range, mean_angle_in_range) (angle in degrees)
        """
        mask = self._range_mask(zmin=zmin, zmax=zmax)
        return (self.z[mask].mean(), self.ang_with_z[mask].mean())

    def mean_kin_energy_trans(self, zmin=None, zmax=None):
        """
        finds the mean transverse e_kin along the trajectory, the z range can be limited
        if the max value occurs more than once the first appearance is returned
        returns a tuple (mean_z_in_range, mean_trans_ekin_in_range) (e_kin in eV)
        """
        mask = self._range_mask(zmin=zmin, zmax=zmax)
        return (self.z[mask].mean(), self.kin_energy_trans[mask].mean())


class Beam:
    """
    A class holding a number of trajectories forming a beam
    Provides convenience functions and properties
    """
    def __init__(self, particles):
        """
        Init method

        particles is a list of TouParticle objects
        """
        self.particles = particles
        self.zmin = np.min([np.min(p.z) for p in self.particles])
        self.zmax = np.max([np.max(p.z) for p in self.particles])

    @property
    def current(self):
        """Total beam current"""
        return sum(p.current for p in self.particles)

    @property
    def ntr(self):
        """Number of trajectories"""
        return len(self.particles)

    def plot_trajectories(self, x="z", y="r", ax=None, nskip=1, **kwargs):
        """
        Plots two properties of each trajectory at all available timesteps for all trajectories
        """
        if not ax:
            _, ax = plt.subplots()

        for p in self.particles[::nskip]:
            ax.plot(getattr(p, x), getattr(p, y), **kwargs)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        return ax.figure


    def _interpolate_along_z(self, quantity, zmin=None, zmax=None, tr_id=0):
        """
        Takes the z coordiantes of trajectory "tr_id" and resamples a given quantity for these
        z values for all trajectories, zrange can be limited
        """
        # Take z positions of selected particle as sample points
        zref = np.sort(self.particles[tr_id].z)
        if not (zmin or zmax):
            zmin = np.min(zref)
        zref = np.clip(zref, a_min=zmin, a_max=zmax)
        # Interpolate the radii at each zref
        qs = []
        for p in self.particles:
            qs.append(np.interp(zref, p.z, getattr(p, quantity)))
        return (zref, qs)

    ### The following method is flawed, the computation of the mean radius as done here, does
    ### not make sense
    # def mean_radius(self, zmin=None, zmax=None):
    #     """Compute the mean radius of the beam at a series of z positions"""
    #     zref, rs = self._interpolate_along_z("r", zmin=zmin, zmax=zmax)
    #     nz = len(zref)
    #     rs = np.stack(rs)
    #     rs = np.split(rs.T, nz)

    #     currents = np.array([p.current for p in self._particles])
    #     rmean = []
    #     weights = currents/self.current
    #     for r in rs:
    #         dr = np.mean(np.abs(np.diff(r))) #Approximation of particle distance/annulus thicknes
    #         rmean.append(np.sqrt(np.sum(weights * 2 * r * dr)))
    #     rmean = np.array(rmean)
    #     return (zref, rmean)

    # TODO: Implement more robustly, right now it just takes trajectory #0
    def outer_radius(self, zmin=None, zmax=None):
        """
        Compute the max radius of the beam at a series of z positions
        """
        zref, rs = self._interpolate_along_z("r", zmin=zmin, zmax=zmax)
        rs = np.stack(rs)
        rmax = rs.T.max(axis=1)
        return (zref, rmax)

    def outer_radius_characteristics(self, zmin=None, zmax=None):
        """
        Compute some characteristic properties of the edge of the beam (outermost trajectory)

        Returns
        (mean z pos, mean radius, std radius, period)
        """
        z, r = self.outer_radius(zmin=zmin, zmax=zmax)

        peaks, dips = _peaks_and_dips_args(z, r)

        if len(peaks) < 2 and len(dips) < 2:
            # if len(peaks) == 1 and len(dips) == 1:
            #     minz, maxz = min(peaks[0], dips[0]), max(peaks[0], dips[0])
            #     period = 2 * (maxz-minz)
            #     rngmsk = np.nonzero((minz < z) & (z <= maxz))
            #     rmean = r[rngmsk].mean()
            #     rstd = r[rngmsk].std()
            #     return ((minz+maxz)/2, rmean, rstd, period)
            # else:
            return (z.mean(), r.mean(), r.std(), z.max()-z.min())

        if len(peaks) > len(dips):
            zsp = z[peaks]
        else:
            zsp = z[dips]

        period = np.diff(zsp).mean()
        rngmsk = np.nonzero((zsp[0] <= z) & (z <= zsp[-1]))

        rmean = r[rngmsk].mean()
        rstd = r[rngmsk].std()

        return (zsp.mean(), rmean, rstd, period)

    def plot_outer_radius(self, zmin=None, zmax=None, ax=None):
        """
        Plot the outermost radius of the beam (outermost particle) and mark maxima and minima

        Returns
        figure object
        """
        z, r = self.outer_radius(zmin=zmin, zmax=zmax)

        peaks, dips = _peaks_and_dips_args(z, r)

        if not ax:
            _, ax = plt.subplots()

        ax.plot(z, r, "k")
        ax.plot(z[peaks], r[peaks], 'ro')
        ax.plot(z[dips], r[dips], 'bo')
        ax.set_xlabel("z")
        ax.set_ylabel("r max")
        return ax.figure

def _peaks_and_dips_args(x, y):
    """Returns the indices of local maxima and minima in y"""
    yp = np.gradient(y, x)
    peaks = np.nonzero((yp[:-1] > 0) & (yp[1:] < 0))[0]
    dips = np.nonzero((yp[:-1] < 0) & (yp[1:] > 0))[0]
    return peaks, dips
