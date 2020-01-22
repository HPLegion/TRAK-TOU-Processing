"""Script for quickly generating some plots of TRAK simulation results"""

### Imports and Definitions
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.integrate
from scipy.stats import linregress
from scipy.constants import (
    elementary_charge as Q_E,
    electron_mass as M_E,
    epsilon_0 as EPS_0,
    pi as PI,
    Boltzmann as K_B,
)

from tct.simulations import Trak

### Additional definitions
ETA = Q_E/M_E


### Plot Settings
ZLIM = [-.057, .1]
RLIM = [0, .01]
ZLIM_ZOOM = [-.057, 0.0]
RLIM_ZOOM = [0.0, .006]
BLIM = [0.0, 0.2]



### Helper functions
def brillouin_radius(I, v_z, B):
    return np.sqrt(2*I/(ETA * PI * EPS_0 * v_z * B**2))


def herrmann_radius(I, v_z, B, T_c, r_c, B_c):
    r_B = brillouin_radius(I, v_z, B)
    radix = 1 + 4 * ((8 * K_B * T_c * r_c**2)/(Q_E * ETA * r_B**4 * B**2) +\
                     (r_c**4 * B_c**2)/(r_B**4 * B**2))
    return r_B * np.sqrt(0.5 + 0.5 * np.sqrt(radix))



### Main pipline
def pipeline(filepath):
    ### Parse filepath and create output directory if necessary
    filedir, filename = os.path.split(filepath)
    filenamestub, _ = os.path.splitext(filename)
    outputdir = os.path.join(filedir, "./Brillouin_lowres")

    if not os.path.exists(outputdir):
        os.mkdir(outputdir)


    ### Search for magnetic field line scan and load it
    bfile = [f for f in os.listdir(filedir) if "BREX_RING_AXIS_SCAN" in f]
    if len(bfile) != 1:
        raise FileNotFoundError("Magnetic field file not found or ambiguous.")
    else:
        magdf = pd.read_csv(os.path.join(filedir, bfile[0]), comment="#", sep=r"\s+")
        magdf.Z = magdf.Z/1000
        field_interp = scipy.interpolate.interp1d(magdf.Z, magdf.Bz)


    ### Load simulation and define some convenience variables
    sim = Trak(filepath)
    beam = sim.beam
    current = beam.current
    p = beam.particles[-1]
    r_c = p.r[0]
    T_c = sim.emission.T_c
    z_c = p.z[0]


    ### Compute needed quantities
    B_shift = sim.permag.shift
    B_c = field_interp(z_c - B_shift)
    B_p = field_interp(p.z - B_shift)

    psi_c = B_c * PI * r_c**2
    psi = B_p * PI * p.r**2

    rho = current / (PI * p.r**2 * p.v_z)
    omega_p = np.sqrt(ETA * rho/EPS_0)
    omega_H = ETA * B_p / 2
    r_B = brillouin_radius(current, p.v_z, B_p)
    r_H = herrmann_radius(current, p.v_z, B_p, T_c, r_c, B_c)

    dw_r = -Q_E * B_p * p.v_phi
    w_r = scipy.integrate.cumtrapz(dw_r, p.t, initial=0)

    plot_title = (
        f"{filenamestub} --- "
        f"$I = {int(current*1e3)}$ mA, "
        f"$B_c = {int(B_c*1e4)}$ G, "
        f"$T_c = {int(T_c)}$ K"
        )

    ### Generate Plot: Report sheet
    fig, axs = plt.subplots(3, figsize=(8, 11), sharex=True, gridspec_kw={'hspace': 0})

    ax = axs[0]
    sim.plot_trajectories(ax=ax, p_slice=np.s_[-1], label="$r$", title=None)
    ax.plot(p.z, r_B, "--", label="$r_B$")
    ax.plot(p.z, r_H, "--", label="$r_H$")
    ax.legend()
    ax.set_ylim(RLIM)
    ax.set_ylabel("$r$ (m)")
    ax.grid()
    ax2 = ax.twinx()
    ax2.plot(magdf.Z + B_shift, magdf.Bz, "tab:red", label="B_z(r=0)")
    ax2.set_ylim(BLIM)
    ax2.set_ylabel("$B_z$ (T)", color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")

    ax = axs[1]
    ax.plot(p.z, 1e-6*p.v_r, label="$v_r$")
    ax.plot(p.z, 1e-6*p.v_phi, label=r"$v_\phi$")
    ax.plot(p.z, 1e-6*p.v_z, label="$v_z$")
    ax.set_ylabel("$v$ (mm/ns)")
    ax.grid()
    ax.legend()

    ax = axs[2]
    # ax.plot(p.z, p.omega_phi_rad / (2*omega_H), label=r"$\omega_\phi/(2\omega_H)$")
    ax.plot(p.z, p.r/r_B, label=r"$r/r_B$")
    ax.plot(p.z, p.r/r_H, label=r"$r/r_H$")
    ax.plot(p.z, omega_p**2/(2*omega_H**2), label=r"$\omega_p^2/(2\omega_H^2)$")
    ax.plot(p.z, 1-(psi_c/psi)**2, label=r"$1-(\psi_c/\psi)^2$")
    ax.set_ylim(-5, 5)
    ax.grid()
    ax.legend()

    # ax = axs[3]
    # ax.plot(p.z, dw_r)
    # ax.set_ylabel(r"$-q_e B_z v_\phi$ (N)")
    # ax2 = ax.twinx()
    # ax2.plot(p.z, w_r, "tab:red")
    # ax2.set_ylabel(r"$m_e\int [-q_e B_z v_\phi] \,dt$ (kg m/s)", color="tab:red")
    # ax2.tick_params(axis='y', labelcolor="tab:red")
    # ax.grid()


    axs[0].set_xlim(ZLIM)
    axs[-1].set_xlabel("$z$ (m)")
    axs[0].set_title(plot_title)

    plt.tight_layout()
    # plt.show()
    fig.savefig(os.path.join(outputdir, f"{filenamestub}_report.png"), dpi=300)
    axs[0].set_xlim(ZLIM_ZOOM)
    fig.savefig(os.path.join(outputdir, f"{filenamestub}_zoom.png"), dpi=300)
    plt.close(fig)


    ### r vs B figure
    valid = p.z > 0
    invalid = np.logical_not(valid)
    res = linregress(np.log(B_p[valid]), np.log(p.r[valid]))
    xf, yf = B_p, np.exp(res.intercept)*B_p**res.slope

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(B_p[valid], p.r[valid], "tab:blue", label="fitted")
    ax.plot(B_p[invalid], p.r[invalid], "tab:red", label="not fitted")
    ax.plot(xf, yf, "k", label=f"{np.exp(res.intercept):.5f}B^{res.slope:.3f}")
    ax.set(
        ylabel="r (m)",
        xlabel="B (T)",
        title=plot_title
    )
    plt.tight_layout()
    ax.legend()
    # plt.show()
    fig.savefig(os.path.join(outputdir, f"{filenamestub}_compression.png"), dpi=300)
    plt.close()