"""
Contains the class SimpleTouParticle
"""

import numpy as np
from scipy.constants import speed_of_light as C_0
class SimpleTouParticle:
    """
    A simple data container for particle information imported from a TOU file.
    Also extends the trajecotry frame with velocity columns computed using numpy.gradient
    This makes certain assumptions about the domain edges
    See: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.gradient.html
    """
    def __init__(self, trajectory, constants):
        """
        Constructor
        trajectory -- pandas dataframe as produced by the import_tou functions (i.e t,x,y,z format)
        constants -- dict with particle constants as produced by the import_tou functions
        """
        self._trajectory = trajectory
        self._id = constants["id"]
        self._mass = constants["mass"]
        self._charge = constants["charge"]
        # compute velocities
        if not self._trajectory.empty:
            self._trajectory["vx"] = np.gradient(self._trajectory["x"], self._trajectory["t"], edge_order=2)
            self._trajectory["vy"] = np.gradient(self._trajectory["y"], self._trajectory["t"], edge_order=2)
            self._trajectory["vz"] = np.gradient(self._trajectory["z"], self._trajectory["t"], edge_order=2)
        else:
            self._trajectory["vx"] = []
            self._trajectory["vy"] = []
            self._trajectory["vz"] = []

    @property
    def trajectory(self):
        """Dataframe holding the TOU Data"""
        return self._trajectory

    @property
    def id(self):
        """Particle ID"""
        return self._id

    @property
    def mass(self):
        """Mass"""
        return self._mass

    @property
    def charge(self):
        """Charge"""
        return self._charge

    @property
    def has_data(self):
        """returns True if trajectory frame is not empty, False otherwise"""
        return not self._trajectory.empty
