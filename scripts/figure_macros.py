"""Contains some convenient plotting functions"""

import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_trajectories(beam, egeo=None, bgeo=None, ax=None, xlim=None, ylim=None, title=None):
    """Plots trajectories of particles in the beam together with geometries if provided"""
    if not ax:
        _, ax = plt.subplots(figsize=(12, 9))

    if bgeo:
        plot_geometry(bgeo, ax=ax, edgecolor="k", facecolor="cornflowerblue")
    if egeo:
        plot_geometry(egeo, ax=ax, edgecolor="k", facecolor="tab:gray")

    beam.plot_trajectories(ax=ax, lw=".75")

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title)
    ax.set_xlabel("$z$ (m)")
    ax.set_ylabel("$r$ (m)")
    plt.tight_layout()
    return ax.figure

def plot_geometry(geo, ax=None, **kwargs):
    """Plots a list of regions presenting a problem geometry as mpl patches"""
    if not ax:
        _, ax = plt.subplots(figsize=(12, 9))
    for rp in geo:
        patch = mpl.patches.PathPatch(rp, **kwargs)
        ax.add_patch(patch)
    plt.tight_layout()
    return ax.figure

def plot_ang_with_z(beam, ax=None, xlim=None, ylim=None, title=None):
    """
    Plot the pitch angles (w.r.t. e_z) of particles vs their z position
    """
    if not ax:
        _, ax = plt.subplots(figsize=(12, 9))

    beam.plot_trajectories(y="ang_with_z", ax=ax, lw=".75")#, color="k")

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title)
    ax.set_xlabel("$z$ (m)")
    ax.set_ylabel(r"$\phi$ (deg)")
    plt.tight_layout()

    return ax.figure
