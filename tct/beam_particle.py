"""
Contains the class TouParticle
"""
from __future__ import annotations

from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np

from .particle import Particle


class Beam:
    """
    A class holding a number of trajectories forming a beam
    Provides convenience functions and properties
    """
    def __init__(self, particles: list[Particle]):
        """
        Init method

        particles is a list of TouParticle objects
        """
        self.particles = particles

    @property
    def current(self):
        """Total beam current"""
        return sum(p.current for p in self.particles)

    @property
    def ntr(self):
        """Number of trajectories"""
        return len(self.particles)

    def plot_trajectories(self, x="z", y="r", ax=None, p_slice=None, **kwargs):
        """
        Plots two properties of each trajectory at all available timesteps for all trajectories
        """
        if not ax:
            _, ax = plt.subplots()

        if p_slice is None:
            p_slice = np.s_[:]

        if isinstance(p_slice, int):
            p = self.particles[p_slice]
            ax.plot(getattr(p, x), getattr(p, y), **kwargs)
        else:
            for p in self.particles[p_slice]:
                ax.plot(getattr(p, x), getattr(p, y), **kwargs)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        return ax.figure


    @lru_cache(maxsize=10)
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

    @lru_cache(maxsize=10)
    def outer_radius(self, zmin=None, zmax=None):
        """
        Compute the max radius of the beam at a series of z positions
        """
        zref, rs = self._interpolate_along_z("r", zmin=zmin, zmax=zmax)
        rs = np.stack(rs)
        rmax = rs.T.max(axis=1)
        return (zref, rmax)

    @lru_cache(maxsize=10)
    def outer_radius_extrema(self, zmin=None, zmax=None):
        """Returns a list of all local maxima and minima of the outer radius in a given range"""
        z, r = self.outer_radius(zmin=zmin, zmax=zmax)
        peaks_arg, dips_arg = _peaks_and_dips_args(z, r)
        peaks = dict(z=z[peaks_arg], r=r[peaks_arg])
        dips = dict(z=z[dips_arg], r=r[dips_arg])
        return peaks, dips

    def outer_radius_closest_max(self, z0):
        """
        returns the closest maximum of the outer radius
        """
        peaks, _ = self.outer_radius_extrema()
        arg = np.argmin(np.abs(peaks["z"]-z0))
        return peaks["z"][arg], peaks["r"][arg]

    def outer_radius_closest_min(self, z0):
        """
        returns the closest maximum of the outer radius
        """
        _, dips = self.outer_radius_extrema()
        arg = np.argmin(np.abs(dips["z"]-z0))
        return dips["z"][arg], dips["r"][arg]

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
        peaks, dips = self.outer_radius_extrema()

        if not ax:
            _, ax = plt.subplots()

        ax.plot(z, r, "k")
        ax.plot(peaks["z"], peaks["r"], 'ro')
        ax.plot(dips["z"], dips["r"], 'bo')
        ax.set_xlabel("z")
        ax.set_ylabel("r max")
        return ax.figure

def _peaks_and_dips_args(x, y):
    """Returns the indices of local maxima and minima in y"""
    yp = np.gradient(y, x)
    peaks = np.nonzero((yp[:-1] > 0) & (yp[1:] < 0))[0]
    dips = np.nonzero((yp[:-1] < 0) & (yp[1:] > 0))[0]
    return peaks, dips
