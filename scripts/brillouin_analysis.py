"""Script for quickly generating some plots of TRAK simulation results"""
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy.interpolate
import scipy.integrate

from scipy.constants import (
    elementary_charge as Q_E,
    electron_mass as M_E,
    epsilon_0 as EPS_0,
    pi as PI
)
ETA = Q_E/M_E

from tct.simulations import Trak


Z_0 = -0.057 # emission surface

ZLIM = [-.057, .1]
RLIM = [0, .01]
ZLIM_ZOOM = [-.057, 0.0]
RLIM_ZOOM = [0.0, .006]
BLIM = [0.0, 0.2]

dirs = os.listdir(r"M:\REX_NA_GUN")
dirs = ["M:\\REX_NA_GUN\\" + d for d in dirs if "REX_NA" in d]

for CWD in tqdm(dirs[:]):

# CWD = r"M:\REX_NA_GUN\REX_NA_DRAWING_ringwidth_+1mm_1300K"
    os.chdir(CWD)
    if not os.path.exists("./Brillouin"):
        os.mkdir("./Brillouin")

    try:
        magdf = pd.read_csv("./BREX_RING_AXIS_SCAN.TXT", comment="#", sep=r"\s+")
        magdf.Z = magdf.Z/1000
        field_interp = scipy.interpolate.interp1d(magdf.Z, magdf.Bz)
    except:
        continue

    files = os.listdir(CWD)
    files = [f for f in files if f.endswith(".tin")]
    files = [f for f in files if "repot" not in f]


    for f in tqdm(files[:]):
        # print(f)
        try:
            fnamestub = f[:-4]
            trak = Trak(f)
            beam = trak.beam
            p_edge = trak.beam.particles[-1]
            current = trak.beam.current

            # Compute needed quantities
            b_shift = trak.permag.shift
            b_c = field_interp(Z_0 - b_shift)
            b_p_edge = field_interp(p_edge.z - b_shift)

            psi_c = b_c * PI * p_edge.r[0]**2
            psi = b_p_edge * PI * p_edge.r**2

            rho = current / (PI * p_edge.r**2 * p_edge.v_z)
            omega_p = np.sqrt(ETA * rho/EPS_0)
            omega_H = ETA * b_p_edge / 2
            r_B = np.sqrt(2*current/(ETA * PI * EPS_0 * p_edge.v_z * b_p_edge**2))

            dw_r = -Q_E * b_p_edge * p_edge.v_phi * p_edge.v_r
            w_r = scipy.integrate.cumtrapz(dw_r, p_edge.t, initial=0)


            ################ Report sheet
            fig, axs = plt.subplots(4, figsize=(8, 14), sharex=True, gridspec_kw={'hspace': 0})
            axs[0].set_xlim(ZLIM)
            axs[-1].set_xlabel("$z$ (m)")
            axs[0].set_title(f"{fnamestub} --- $B_c = {int(psi_c*1e4)}$ Gs, $I = {int(current*1e3)}$ mA")

            ax = axs[0]
            trak.estat.plot_elements(ax=ax, edgecolor="k", facecolor="tab:gray")
            trak.permag.plot_elements(ax=ax, edgecolor="k", facecolor="tab:blue")
            ax.plot(p_edge.z, p_edge.r, label="$r$")
            ax.plot(p_edge.z, r_B, "--", label="$r_B$")
            ax.legend()
            ax.set_ylim(RLIM)
            ax.set_ylabel("$r$ (m)")
            ax.grid()
            ax2 = ax.twinx()
            ax2.plot(magdf.Z + b_shift, magdf.Bz, "tab:red", label="B_z(r=0)")
            ax2.set_ylim(BLIM)
            ax2.set_ylabel("$B_z$ (T)", color="tab:red")
            ax2.tick_params(axis='y', labelcolor="tab:red")

            ax = axs[1]
            ax.plot(p_edge.z, 1e-6*p_edge.v_r, label="$v_r$")
            ax.plot(p_edge.z, 1e-6*p_edge.v_phi, label=r"$v_\phi$")
            ax.plot(p_edge.z, 1e-6*p_edge.v_z, label="$v_z$")
            ax.set_ylabel("$v$ (mm/ns)")
            ax.grid()
            ax.legend()

            ax = axs[2]
            ax.plot(p_edge.z, p_edge.omega_phi_rad / (2*omega_H), label=r"$\omega_\phi/(2\omega_H)$")
            ax.plot(p_edge.z, p_edge.r/r_B, label=r"$r/r_B$")
            ax.plot(p_edge.z, omega_p**2/(2*omega_H**2), label=r"$\omega_p^2/(2\omega_H^2)$")
            ax.plot(p_edge.z, 1-(psi_c/psi)**2, label=r"$1-(\psi_c/\psi)^2$")
            ax.set_ylim(-5, 5)
            ax.grid()
            ax.legend()

            ax = axs[3]
            ax.plot(p_edge.z, dw_r/Q_E/1e12, label=r"$-q_e B_z v_\phi v_r$")
            ax.set_ylabel(r"$-q_e B_z v_\phi v_r$ (keV/ns)")
            ax2 = ax.twinx()
            ax2.plot(p_edge.z, w_r/Q_E/1e3, "tab:red", label=r"$\int -q_e B_z v_\phi v_r dt$")
            ax2.set_ylabel(r"$\int -q_e B_z v_\phi v_r \,dt$ (keV)", color="tab:red")
            ax2.tick_params(axis='y', labelcolor="tab:red")
            # ax2.set_ylim(bottom=0)
            ax.grid()

            plt.tight_layout()
            # plt.show()
            fig.savefig(f"./Brillouin/{fnamestub}_report.png")
            axs[0].set_xlim(ZLIM_ZOOM)
            fig.savefig(f"./Brillouin/{fnamestub}_report_zoom.png")
            plt.close(fig)

        except:
            continue


        # # ################ Trajectory plot
        # fig = trak.plot_trajectories()
        # fig.gca().set(title=f"{fnamestub} - {beam.current} A", xlim=XLIM, ylim=YLIM)
        # plt.tight_layout()
        # fig.savefig(f"./Brillouin/{fnamestub}.png")
        # plt.close(fig)

        # # ################ Transverse plot
        # fig, ax = plt.subplots(figsize=(12, 9))
        # for p in beam.particles[::5]:
        #     ax.plot(p.x, p.y)
        # ax.set(xlabel="x (m)", ylabel="y (m)", title=f"{fnamestub} - {beam.current} A")
        # fig.savefig(f"./Brillouin/{fnamestub}_transverse.png")
        # plt.close(fig)

        # ################ Trajectory plot with fields
        # fig = trak.plot_trajectories(efield={"fill":True})
        # ax = fig.gca()
        # ax.set(title=f"{fnamestub} - {beam.current} A", xlim=XLIM, ylim=YLIM)
        # plt.tight_layout()

        # ax2 = ax.twinx()
        # ax2.plot(magdf.Z + b_shift, magdf.Bz, "tab:blue", label="B_z(r=0)")
        # ax2.set_ylim((0, 1.0))
        # ax2.set_ylabel("$B_z$ (T)", color="tab:blue")
        # ax2.tick_params(axis='y', labelcolor="tab:blue")
        # plt.tight_layout()

        # fig.savefig(f"./Brillouin/{fnamestub}_fields.png")

        # ax.set(xlim=XLIM_ZOOM, ylim=YLIM_ZOOM)
        # fig.savefig(f"./Brillouin/{fnamestub}_fields_zoom.png")
        # plt.close(fig)
