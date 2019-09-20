import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate
import scipy.integrate
from scipy.constants import (
    elementary_charge as Q_E,
    electron_mass as M_E,
    pi as PI,
    epsilon_0 as EPS_0,
)
import tct

#### DERIVED CONSTANTS ####
ETA = Q_E / M_E # Electron Charge to mass ratio


#### SETTINGS ####
PID = -1
TID = 5
I_BEAM = 0.711 #Amps
R_C = 0.001 # m Cathode radius
DIR = r'M:\HANNES CST FILES\NONADIABATIC GUN\trak\\'
TOUFILE = os.path.abspath(DIR + r"RMNA5_30ES_M31+8mm_0_8A_0_5kGsOrig_.TOU")
MAGFILE_AX = os.path.abspath(DIR + r"FMRNA4_3_1_0KGS+008.TXT")
MAGFILE_MAT = os.path.abspath(DIR + r"FMRNA4_3_1_0KGS+008_RINGS.MTX")
ESTATFILE_AX = os.path.abspath(DIR + r"ERMNA1_30_0_7A.TXT")
ESTATFILE_MAT = os.path.abspath(DIR + r"ERMNA1_30_0_7A_RINGS.MTX")


#### DATA IMPORT AND INTERPOLANTS ####
#region
beam = tct.import_tou_as_beam(TOUFILE)
particles = beam.particles
print(beam.current)

magdf_ax = pd.read_csv(MAGFILE_AX, comment="#", sep=r"\s+")
magdf_ax.Z = magdf_ax.Z/1000
b_z_ax = scipy.interpolate.interp1d(magdf_ax.Z, magdf_ax.Bz)
b_r_ax = scipy.interpolate.interp1d(magdf_ax.Z, magdf_ax.Br)


magdf_mat = pd.read_csv(MAGFILE_MAT, comment="#", sep=r"\s+")
magdf_mat.Z = magdf_mat.Z/1000
magdf_mat.R = magdf_mat.R/1000
# b_z_mat = scipy.interpolate.interp2d(magdf_mat.Z, magdf_mat.R, magdf_mat.Bz, kind="cubic")
# b_r_mat = scipy.interpolate.interp2d(magdf_mat.Z, magdf_mat.R, magdf_mat.Br, kind="cubic")


b_z_mat_pivot = magdf_mat[["Z", "R", "Bz"]]
b_z_mat_pivot = b_z_mat_pivot.pivot("Z", "R")
b_r_mat_pivot = magdf_mat[["Z", "R", "Br"]]
b_r_mat_pivot = b_r_mat_pivot.pivot("Z", "R")
b_z_mat = scipy.interpolate.RectBivariateSpline(b_z_mat_pivot.index.values, b_z_mat_pivot.columns.levels[1].values, b_z_mat_pivot.values)
b_r_mat = scipy.interpolate.RectBivariateSpline(b_r_mat_pivot.index.values, b_r_mat_pivot.columns.levels[1].values, b_r_mat_pivot.values)

estatdf_ax = pd.read_csv(ESTATFILE_AX, comment="#", sep=r"\s+")
estatdf_ax.Z = estatdf_ax.Z/1000
e_z_ax = scipy.interpolate.interp1d(estatdf_ax.Z, estatdf_ax.Ez)
e_r_ax = scipy.interpolate.interp1d(estatdf_ax.Z, estatdf_ax.Er)

estatdf_mat = pd.read_csv(ESTATFILE_MAT, comment="#", sep=r"\s+")
estatdf_mat.Z = estatdf_mat.Z/1000
estatdf_mat.R = estatdf_mat.R/1000
# e_z_mat = scipy.interpolate.interp2d(estatdf_mat.Z, estatdf_mat.R, estatdf_mat.Ez, kind="cubic")
# e_r_mat = scipy.interpolate.interp2d(estatdf_mat.Z, estatdf_mat.R, estatdf_mat.Er, kind="cubic")


e_z_mat_pivot = estatdf_mat[["Z", "R", "Ez"]]
e_z_mat_pivot = e_z_mat_pivot.pivot("Z", "R")
e_r_mat_pivot = estatdf_mat[["Z", "R", "Er"]]
e_r_mat_pivot = e_r_mat_pivot.pivot("Z", "R")
e_z_mat = scipy.interpolate.RectBivariateSpline(e_z_mat_pivot.index.values, e_z_mat_pivot.columns.levels[1].values, e_z_mat_pivot.values)
e_r_mat = scipy.interpolate.RectBivariateSpline(e_r_mat_pivot.index.values, e_r_mat_pivot.columns.levels[1].values, e_r_mat_pivot.values)
#endregion

# ### MAGFIELD PLOTS ####
# #region
# z_samp = np.linspace(0, .200, 200)
# r_samp = np.linspace(0, 0.003, 100)
# zmsh, rmsh = np.meshgrid(z_samp, r_samp)

# fig, axs = plt.subplots(2, 2)

# _x, _y = np.meshgrid(e_z_mat_pivot.columns.levels[1].values, e_z_mat_pivot.index.values)
# _p1 = axs[0, 0].contourf(_y, _x, e_z_mat_pivot.values, levels=100)
# fig.colorbar(_p1, ax=axs[0, 0])
# axs[0, 0].set_xlabel("z (m)")
# axs[0, 0].set_ylabel("r (m)")
# axs[0, 0].set_title("Raw area scan")

# _p2 = axs[1, 0].contourf(zmsh, rmsh, e_z_mat(z_samp, r_samp).T, levels=_p1.levels)
# fig.colorbar(_p2, ax=axs[1, 0])
# axs[1, 0].set_xlabel("z (m)")
# axs[1, 0].set_ylabel("r (m)")
# axs[1, 0].set_title("Interpolated area scan")

# _x, _y = np.meshgrid(b_z_mat_pivot.columns.levels[1].values, b_z_mat_pivot.index.values)
# _p1 = axs[0, 1].contourf(_y, _x, b_z_mat_pivot.values, levels=100)
# fig.colorbar(_p1, ax=axs[0, 1])
# axs[0, 1].set_xlabel("z (m)")
# axs[0, 1].set_ylabel("r (m)")
# axs[0, 1].set_title("Raw area scan")

# _p2 = axs[1, 1].contourf(zmsh, rmsh, b_z_mat(z_samp, r_samp).T, levels=_p1.levels)
# fig.colorbar(_p2, ax=axs[1, 1])
# axs[1, 1].set_xlabel("z (m)")
# axs[1, 1].set_ylabel("r (m)")
# axs[1, 1].set_title("Interpolated area scan")

# z_samp = np.linspace(0, .200, 200)
# r_samp = np.linspace(0, 0.003, 100)
# zmsh, rmsh = np.meshgrid(z_samp, r_samp)
# #endregion

# ### MAGFIELD PLOTS ####
# #region
# z_samp = np.linspace(0, .200, 200)
# r_samp = np.linspace(0, 0.003, 100)
# zmsh, rmsh = np.meshgrid(z_samp, r_samp)

# fig, axs = plt.subplots(2, 2)

# _x, _y = np.meshgrid(e_r_mat_pivot.columns.levels[1].values, e_r_mat_pivot.index.values)
# _p1 = axs[0, 0].contourf(_y, _x, e_r_mat_pivot.values, levels=100)
# fig.colorbar(_p1, ax=axs[0, 0])
# axs[0, 0].set_xlabel("z (m)")
# axs[0, 0].set_ylabel("r (m)")
# axs[0, 0].set_title("Raw area scan")

# _p2 = axs[1, 0].contourf(zmsh, rmsh, e_r_mat(z_samp, r_samp).T, levels=_p1.levels)
# fig.colorbar(_p2, ax=axs[1, 0])
# axs[1, 0].set_xlabel("z (m)")
# axs[1, 0].set_ylabel("r (m)")
# axs[1, 0].set_title("Interpolated area scan")

# _x, _y = np.meshgrid(b_r_mat_pivot.columns.levels[1].values, b_r_mat_pivot.index.values)
# _p1 = axs[0, 1].contourf(_y, _x, b_r_mat_pivot.values, levels=100)
# fig.colorbar(_p1, ax=axs[0, 1])
# axs[0, 1].set_xlabel("z (m)")
# axs[0, 1].set_ylabel("r (m)")
# axs[0, 1].set_title("Raw area scan")

# _p2 = axs[1, 1].contourf(zmsh, rmsh, b_r_mat(z_samp, r_samp).T, levels=_p1.levels)
# fig.colorbar(_p2, ax=axs[1, 1])
# axs[1, 1].set_xlabel("z (m)")
# axs[1, 1].set_ylabel("r (m)")
# axs[1, 1].set_title("Interpolated area scan")

# z_samp = np.linspace(0, .200, 200)
# r_samp = np.linspace(0, 0.003, 100)
# zmsh, rmsh = np.meshgrid(z_samp, r_samp)
# #endregion

# #### Line Interpolator Plots ####
# #region
# fig, axs = plt.subplots(2, 2)
# z_samp = np.linspace(0, .200, 200)

# axs[0, 0].plot(estatdf_ax.Z, estatdf_ax.Ez)
# axs[0, 0].plot(z_samp, e_z_ax(z_samp))
# axs[0, 0].set_xlabel("z (m)")
# axs[0, 0].set_ylabel("E_z (V/m)")

# axs[1, 0].plot(estatdf_ax.Z, estatdf_ax.Er)
# axs[1, 0].plot(z_samp, e_r_ax(z_samp))
# axs[1, 0].set_xlabel("z (m)")
# axs[1, 0].set_ylabel("E_r (V/m)")

# axs[0, 1].plot(magdf_ax.Z, magdf_ax.Bz)
# axs[0, 1].plot(z_samp, b_z_ax(z_samp))
# axs[0, 1].set_xlabel("z (m)")
# axs[0, 1].set_ylabel("B_z (T)")

# axs[1, 1].plot(magdf_ax.Z, magdf_ax.Br)
# axs[1, 1].plot(z_samp, b_r_ax(z_samp))
# axs[1, 1].set_xlabel("z (m)")
# axs[1, 1].set_ylabel("B_r (T)")
# #endregion

p = particles[PID]
T_0 = p.t[TID]
R_0 = p.r[TID]
THETA_0 = np.unwrap(p.phi_rad)[TID]
Z_0 = p.z[TID]
V_R_0 = p.v_r[TID]
V_THETA_0 = p.v_phi_rad[TID]
V_Z_0 = p.v_z[TID]
B_Z_0 = b_z_mat(p.z[TID], 0)

# dt_rad = lambda x: np.interp(x,
#     np.array([0, 0.66, 0.661, 2.88, 2.881, 4.29, 30, 30.001, 40, 40.001, 200])/1000,
#     np.array([1.1, 2.25, 10, 10, 2, 4.5, 4.5, 10, 10, 5, 5])/1000
#     )
dt_rad = lambda x: np.interp(x,
    np.array([0, 0.66, 2.88, 4.29, 30, 40, 200])/1000,
    np.array([1.1, 2.25, 2, 4.5, 4.5, 5, 5])/1000
    )

def rhs(_t, y):
    r, theta, z, v_r, v_z = y
    e_r_sc = - I_BEAM * r / (2 * PI * EPS_0 * v_z * r**2)
    e_z = e_z_mat(z, r)
    e_r = e_r_mat(z, r) + e_r_sc
    b_z = b_z_mat(z, r)
    b_r = b_r_mat(z, r)
    v_theta = ETA/2 * (b_z - (R_0/r)**2 * B_Z_0) + (R_0/r)**2 * V_THETA_0
    v_r_dot = - ETA * e_r - ETA * r * v_theta * b_z + r * v_theta**2
    v_z_dot = - ETA * e_z + ETA * r * v_theta * b_r
    return v_r, v_theta, v_z, v_r_dot, v_z_dot

def get_rhs_with_e_sc_z(e_sc_z_interp):
    def _rhs(_t, y):
        r, theta, z, v_r, v_z = y
        e_z = e_z_mat(z, r) + e_sc_z_interp(z)
        e_r = e_r_mat(z, r) - I_BEAM * r / (2 * PI * EPS_0 * v_z * r**2)
        b_z = b_z_mat(z, r)
        b_r = b_r_mat(z, r)
        v_theta = ETA/2 * (b_z - (R_0/r)**2 * B_Z_0) + (R_0/r)**2 * V_THETA_0
        v_r_dot = - ETA * e_r - ETA * r * v_theta * b_z + r * v_theta**2
        v_z_dot = - ETA * e_z + ETA * r * v_theta * b_r
        return v_r, v_theta, v_z, v_r_dot, v_z_dot
    return _rhs
y0 = [R_0, THETA_0, Z_0, V_R_0, V_Z_0]
print(y0)

e_sc_z_interp = scipy.interpolate.interp1d([-1,1], [0,0])

# plt.ion()
# fig, ax = plt.subplots()
# line = ax.plot([],[])
# ax.set_xlabel("z (m)")
# ax.set_ylabel("r (m)")
# ax.set_xlim(0, .2)
# ax.set_ylim(0, .01)
# plt.show()
rs = []
C = .8
for k in range(20):
    print(k)
    rhs = get_rhs_with_e_sc_z(e_sc_z_interp)
    # sol = scipy.integrate.solve_ivp(rhs, [T_0, 0], y0, max_step=1.0e-11)
    sol = scipy.integrate.solve_ivp(rhs, [T_0, 5e-9], y0, max_step=1.0e-11)
    # if k > 0:
    #     r_old, theta_old, z_old, v_r_old, v_z_old = r, theta, z, v_r, v_z
    #     v_theta_old = v_theta
    r, theta, z, v_r, v_z = sol.y[0, :], sol.y[1, :], sol.y[2, :], sol.y[3, :], sol.y[4, :]
    f = lambda y: rhs(0, y)
    temp = np.apply_along_axis(f, 0, sol.y)
    v_theta = temp[1,:]
    # if k > 0:
    #     r = C * r_old + (1-C) * r
    #     theta = C * theta_old + (1-C) * theta
    #     z = C * z_old + (1-C) * z
    #     v_r = C * v_r_old + (1-C) * v_r
    #     v_theta = C * v_theta_old + (1-C) * v_theta
    #     v_z = C * v_z_old + (1-C) * v_z
    # if k >= 0:
    #     e_sc_z_interp_old = e_sc_z_interp


    e_sc_z =  -I_BEAM / (2 * PI * EPS_0 * v_z**2) * np.log(r/dt_rad(z)) * np.gradient(v_z, z)
    # e_sc_z[0] = - e_z_mat(z[0], r[0])
    # if k >= 0:
    #     e_sc_z = C * e_sc_z_interp_old(z) + (1-C) * e_sc_z
    # e_sc_z_interp = scipy.interpolate.interp1d(z, e_sc_z, fill_value="extrapolate")
    def e_sc_z_interp(z0):
        _z = z.copy()
        _v_z = v_z.copy()
        _r_e = r.copy()
        _v_z = scipy.interpolate.interp1d(_z, _v_z, fill_value="extrapolate")
        _r_e = scipy.interpolate.interp1d(_z, _r_e, fill_value="extrapolate")
        def f(x):
            # print(x)
            return I_BEAM/(PI * EPS_0 * _v_z(x) * _r_e(x)**2)
        def g(x):
            return I_BEAM/(2 * PI * EPS_0 * _v_z(x) * _r_e(x)**2) * (1 - (x-z0)/np.sqrt((x-z0)**2+_r_e(x)**2))
        samp = np.linspace(0, 0.2, 1000)
        return np.trapz(g(samp), samp)#- e_z_mat(0, .001)[0]
    e_sc_z = np.array([e_sc_z_interp(__x) for __x in z])
        # samp = np.linspace(0, z0, 1000)
        # return np.trapz(f(samp), samp)- e_z_mat(0, .001)[0]
        # return scipy.integrate.quad(f, 0, z0)[0] #- e_z_mat(0, .001)
    # plt.figure()
    # _x = np.linspace(0, 0.2, 100)
    # _fx = np.array([e_sc_z_interp(__x) for __x in _x])
    # plt.plot(_x, _fx)
    # plt.show()
    rs.append((z,r))
    # line[0].set_xdata = z
    # line[0].set_ydata = r
    # fig.canvas.draw()
    # plt.pause(.1)
#### TRAJECTORY PLOTS ####

# fig, ax = plt.subplots()
# for _z, _r in rs:
#     ax.plot(_z,_r)

    fig, axs = plt.subplots(2, 3)
    axs[0, 0].plot(p.z, p.r)
    axs[0, 0].plot(z, r)
    axs[0, 0].set_xlabel("z (m)")
    axs[0, 0].set_ylabel("r (m)")
    axs[0, 0].set_title(f"Radius (particle: {PID})")

    axs[1, 0].plot(p.z, p.v_r)
    axs[1, 0].plot(z, v_r)
    axs[1, 0].set_xlabel("z (m)")
    axs[1, 0].set_ylabel("v_r (m/s)")
    axs[1, 0].set_title(f"Radial velocity (particle: {PID})")

    axs[0, 1].plot(p.z, np.unwrap(p.phi_rad))
    axs[0, 1].plot(z, theta)
    axs[0, 1].set_xlabel("z (m)")
    axs[0, 1].set_ylabel("phi (rad)")
    axs[0, 1].set_title(f"Azimuth (particle: {PID})")

    axs[1, 1].plot(p.z, p.v_phi_rad)
    axs[1, 1].plot(z, v_theta)
    axs[1, 1].set_xlabel("z (m)")
    axs[1, 1].set_ylabel("v_phi_rad (1/s)")
    axs[1, 1].set_title(f"Azimuthal velocity (particle: {PID})")

    axs[0, 2].plot(p.t, p.z)
    axs[0, 2].plot(sol.t, z)
    axs[0, 2].set_xlabel("t (s)")
    axs[0, 2].set_ylabel("z (m)")
    axs[0, 2].set_title(f"Axial position (particle: {PID})")

    axs[1, 2].plot(p.z, p.v_z)
    axs[1, 2].plot(z, v_z)
    axs[1, 2].set_xlabel("z (m)")
    axs[1, 2].set_ylabel("v_z (m/s)")
    axs[1, 2].set_title(f"Axial velocity (particle: {PID})")

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(z, dt_rad(z))
    axs[0].set_xlabel("z (m)")
    axs[0].set_ylabel("dt_rad (m)")

    axs[1].plot(z, e_sc_z)
    axs[1].plot(z, e_z_mat.ev(z, r))
    axs[1].plot(z, e_z_mat.ev(z, r) + e_sc_z)
    axs[1].set_xlabel("z (m)")
    axs[1].set_ylabel("E_z^SC (V/m)")

    plt.show()