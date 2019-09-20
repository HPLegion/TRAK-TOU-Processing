import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate
import scipy.integrate
from scipy.constants import (
    elementary_charge as Q_E,
    electron_mass as M_E,
    pi as PI
)
import tct

#### SETTINGS ####
TOUFILE = os.path.abspath(r"M:\HANNES CST FILES\NONADIABATIC GUN\trak\RMNA5_30ES_M31+8mm_0_8A_0_5kGsOrig_.TOU")
MAGFILE_AX = os.path.abspath(r"M:\HANNES CST FILES\NONADIABATIC GUN\trak\FMRNA4_3_1_0KGS+008.TXT")
MAGFILE_MAT = os.path.abspath(r"M:\HANNES CST FILES\NONADIABATIC GUN\trak\FMRNA4_3_1_0KGS+008_RINGS.MTX")
PID = -5
TID = 5

#### DATA IMPORT ####
beam = tct.import_tou_as_beam(TOUFILE)
part = beam.particles

magdf_ax = pd.read_csv(MAGFILE_AX, comment="#", sep=r"\s+")
magdf_ax.Z = magdf_ax.Z/1000
b_z_ax = scipy.interpolate.interp1d(magdf_ax.Z, magdf_ax.Bz)

magdf_mat = pd.read_csv(MAGFILE_MAT, comment="#", sep=r"\s+")
magdf_mat.Z = magdf_mat.Z/1000
magdf_mat.R = magdf_mat.R/1000
b_z_mat = scipy.interpolate.interp2d(magdf_mat.Z, magdf_mat.R, magdf_mat.Bz, kind="cubic")

magdf_mat_pivot = magdf_mat[["Z", "R", "Bz"]]
magdf_mat_pivot = magdf_mat_pivot.pivot("Z", "R")

#### INTERPOLANT DEFINITION ####
def enclosed_flux(z, r):
    def f(_r):
        return 2* PI * _r * b_z_mat(z, _r)
    return scipy.integrate.quad(f, 0, r)[0]
enclosed_flux = np.vectorize(enclosed_flux)

p = part[PID]

z_0 = p.z[TID]; print("z_0", z_0)
r_c = p.r[TID]; print("r_c", r_c)
b_c = b_z_ax(z_0); print("b_c", b_c)
v_phi_c_rad = p.v_phi_rad[TID]; print("v_phi_c_rad", v_phi_c_rad)
flux_c = enclosed_flux(z_0, r_c); print("flux_c", flux_c)
def busch_radius(b, v_phi_rad):
    A = v_phi_c_rad - Q_E/2/M_E * b_c
    B = v_phi_rad - Q_E/2/M_E * b
    return r_c * np.sqrt(A/B)

def busch_radius_flux(flux, v_phi_rad):
    A = Q_E/(2*PI*M_E) * (flux - flux_c) / v_phi_rad
    B = r_c**2 * v_phi_c_rad / v_phi_rad
    return np.sqrt(A + B)


#### MAGFIELD PLOTS ####
z_samp = np.linspace(0, .200, 200)
r_samp = np.linspace(0, 0.003, 100)
zmsh, rmsh = np.meshgrid(z_samp, r_samp)

fig, axs = plt.subplots(2, 3)

axs[0, 0].plot(z_samp, b_z_ax(z_samp), label="line scan")
axs[0, 0].plot(z_samp, b_z_mat(z_samp, 0), "--", label="area scan")
axs[0, 0].set_xlabel("z (m)")
axs[0, 0].set_ylabel("B_z (T)")
axs[0, 0].set_title("B_z on axis")
axs[0, 0].legend()

_x, _y = np.meshgrid(magdf_mat_pivot.columns.levels[1].values, magdf_mat_pivot.index.values)
_p = axs[0, 1].contourf(_y, _x, magdf_mat_pivot.values, levels=100)
fig.colorbar(_p, ax=axs[0, 1])
axs[0, 1].set_xlabel("z (m)")
axs[0, 1].set_ylabel("r (m)")
axs[0, 1].set_title("Raw area scan")

_p = axs[1, 1].contourf(zmsh, rmsh, b_z_mat(z_samp, r_samp), levels=100)
fig.colorbar(_p, ax=axs[1, 1])
axs[1, 1].set_xlabel("z (m)")
axs[1, 1].set_ylabel("r (m)")
axs[1, 1].set_title("Interpolated area scan")

axs[0, 2].plot(z_samp, enclosed_flux(z_samp, 0.001), label="integrated")
axs[0, 2].plot(z_samp, 0.001**2 * PI * b_z_ax(z_samp), "--", label="approximated")
axs[0, 2].set_xlabel("z (m)")
axs[0, 2].set_ylabel("enclosed flux")
axs[0, 2].legend()
axs[0, 2].set_title("Flux enclosed in 1 mm radius")

axs[1, 2].plot(p.z, enclosed_flux(p.z, p.r), label="integrated")
axs[1, 2].plot(p.z, p.r**2 * PI * b_z_ax(p.z), "--", label="approximated")
axs[1, 2].set_xlabel("z (m)")
axs[1, 2].set_ylabel("enclosed flux")
axs[1, 2].legend()
axs[1, 2].set_title(f"Enclosed flux (particle: {PID})")


#### TRAJECTORY PLOTS ####

fig, axs = plt.subplots(2, 3)
axs[0, 0].plot(p.z, p.r)
axs[0, 0].set_xlabel("z (m)")
axs[0, 0].set_ylabel("r (m)")
axs[0, 0].set_title(f"Radius (particle: {PID})")

axs[1, 0].plot(p.z, p.v_r)
axs[1, 0].set_xlabel("z (m)")
axs[1, 0].set_ylabel("v_r (m/s)")
axs[1, 0].set_title(f"Radial velocity (particle: {PID})")

axs[0, 1].plot(p.z, p.phi)
axs[0, 1].set_xlabel("z (m)")
axs[0, 1].set_ylabel("phi (deg)")
axs[0, 1].set_title(f"Azimuth (particle: {PID})")

axs[1, 1].plot(p.z, p.v_phi_rad)
axs[1, 1].set_xlabel("z (m)")
axs[1, 1].set_ylabel("v_phi_rad (1/s)")
axs[1, 1].set_title(f"Azimuthal velocity (particle: {PID})")

axs[0, 2].plot(p.z, p.r, label="true")
axs[0, 2].plot(p.z, busch_radius(b_z_ax(p.z), p.v_phi_rad), "--", label="busch")
axs[0, 2].set_xlabel("z (m)")
axs[0, 2].set_ylabel("busch r (m)")
axs[0, 2].legend()
axs[0, 2].set_title(f"Busch radius axial field (timestep: {TID})")

axs[1, 2].plot(p.z, p.r, label="true")
axs[1, 2].plot(p.z, busch_radius_flux(enclosed_flux(p.z, p.r), p.v_phi_rad), "--", label="busch")
axs[1, 2].set_xlabel("z (m)")
axs[1, 2].set_ylabel("busch r (m)")
axs[1, 2].set_ylim(axs[0, 2].get_ylim())
axs[1, 2].legend()
axs[1, 2].set_title(f"Busch radius integrated field (timestep: {TID})")

#### BEAM PLOTS ####
fig, axs = plt.subplots(1, 2)

for par in part[::20]:
    axs[0].plot(par.z, par.r)
axs[0].set_xlabel("z (m)")
axs[0].set_ylabel("r (m)")

for par in part[::20]:
    axs[1].plot(par.x, par.y)
axs[1].set_aspect("equal")
axs[1].set_xlabel("x (m)")
axs[1].set_ylabel("y (m)")




plt.show()
