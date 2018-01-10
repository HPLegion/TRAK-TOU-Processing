"""
Contains the class SimpleTouParticle
"""

import numpy as np
from scipy.constants import (speed_of_light as C_0,
                             atomic_mass as AMU,
                             elementary_charge as Q_E)

class SimpleTouParticle:
    """
    A simple data container for particle information imported from a TOU file.
    Also extends the trajectory data with useful particle information in an easy to access way
    The velocity is derived from the particle position
    This makes certain assumptions about the domain edges
    See: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.gradient.html
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
        self._gamma = None

    @property
    def tou_data(self):
        """Dataframe holding the TOU Data"""
        if self._safe_mode:
            return self._tou_data.copy()
        else:
            return self._tou_data

    # Alias for tou_data
    trajectory = tou_data

    @property
    def id(self):
        """Tou File Particle ID"""
        return self._id

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
    def gamma(self):
        """numpy array of gamma (velocity in units of c_0)"""
        #Compute _gamma on first demand
        if self._gamma is None:
            beta_vec = np.column_stack((self.beta_x, self.beta_y, self.beta_z))
            self._gamma = 1/np.sqrt(1-np.linalg.norm(beta_vec, axis=1)**2)

        if self._safe_mode:
            return self._gamma.copy()
        else:
            return self._gamma

    @property
    def kin_energy_si(self):
        """Kinetic Energy in J"""
        return (self.gamma-1)*self.mass_si*C_0**2

    @property
    def kin_energy(self):
        """Kinetic Energy in eV"""
        return self.kin_energy_si / Q_E

    @property
    def has_data(self):
        """returns True if trajectory frame is not empty, False otherwise"""
        return not self.tou_data.empty



            # # from the betas also compute the gamma factor
            # temp_mom = np.column_stack((self._trajectory["px"], self._trajectory["py"],
            #                             self._trajectory["pz"]))
            # gamma = 1/np.sqrt(1-np.linalg.norm(temp_mom, axis=1))

            # # multiply beta with gamma to get the full normalised momentum
            # self._trajectory["px"] *= gamma
            # self._trajectory["py"] *= gamma
            # self._trajectory["pz"] *= gamma