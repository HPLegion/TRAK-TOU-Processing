import os
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
import numpy as np
import scipy.interpolate

from tct.simulations import Trak

CWD = r"C:\TRAKTEMP\REX_NA_DRAWING_ringwidth_+1mm_1300K"
XLIM = [-.057, .1]
YLIM = [0, .015]
YLIM_ACCEPT = [0, 30]

os.chdir(CWD)

magdf = pd.read_csv("./BREX_RING_AXIS_SCAN.TXT", comment="#", sep=r"\s+")
magdf.Z = magdf.Z/1000
field_interp = scipy.interpolate.interp1d(magdf.Z, magdf.Bz)

files = os.listdir(CWD)
files = [f for f in files if f.endswith(".tin")]
files = [f for f in files if "repot" not in f]


for f in tqdm.tqdm(files):
    # print(f)
    fnamestub = f[:-4]

    # bshift = float(fnamestub.split("_")[-1][:-2])/1000
    # bgeo = import_min_as_regions("BRex_ring.MIN", xshift=bshift, scale=0.001)
    # bgeo = [r.to_mpl_path() for r in bgeo[1:]] # Skip domain boundaries during conversion
    trak = Trak(f)
    beam = trak.beam

    # ################ Trajecotry Plot
    # fig = trak.plot_trajectories()
    # fig.gca().set(title=f"{fnamestub} - {beam.current} A", xlim=XLIM, ylim=YLIM)
    # plt.tight_layout()
    # fig.savefig(fnamestub + "_2.png")
    # # plt.show()
    # plt.close(fig)

    # ################ Trajecotry Plot
    # fig,ax = plt.subplots(figsize=(12, 9))
    # for p in beam.particles[::5]:
    #     ax.plot(p.x, p.y)
    # ax.set_xlabel("x (m)")
    # ax.set_ylabel("y (m)")
    # ax.set_title(f"{fnamestub} - {beam.current} A")
    # fig.savefig(fnamestub + "_transverse_2.png")
    # # plt.show()
    # plt.close(fig)

    ################ Trajecotry with fields Plot
    # fig, ax = plt.subplots(figsize=(12, 9))
    # estatfile = trak.estat.input_file_name[:-3] + "mtx"
    # estatdf_mat = pd.read_csv(estatfile, comment="#", sep=r"\s+", skiprows=[0,1,2,3,5])
    # estatdf_mat.Z = estatdf_mat.Z/1000
    # estatdf_mat.R = estatdf_mat.R/1000

    # phi_mat_pivot = estatdf_mat[["Z", "R", "Phi"]]
    # phi_mat_pivot = phi_mat_pivot.pivot("Z", "R")
    # phi_mat = scipy.interpolate.RectBivariateSpline(phi_mat_pivot.index.values, phi_mat_pivot.columns.levels[1].values, phi_mat_pivot.values)

    # z_samp = np.linspace(*XLIM, 200)
    # r_samp = np.linspace(*YLIM, 100)
    # zmsh, rmsh = np.meshgrid(z_samp, r_samp)

    # _cont = ax.contourf(zmsh, rmsh, 1+phi_mat(z_samp, r_samp).T, levels=21, zorder=1, cmap="plasma",
    #                     vmin=-10000, vmax=0, extend="both")
    # cbar = fig.colorbar(_cont, ax=ax)
    # cbar.ax.set_ylabel("Potential (V)")
    # # plt.show()

    # trak.plot_trajectories(ax=ax, c="k")
    fig = trak.plot_trajectories(efield={"fill":True})
    ax = fig.gca()
    ax.set(title=f"{fnamestub} - {beam.current} A", xlim=XLIM, ylim=YLIM)
    plt.tight_layout()


    ax2 = ax.twinx()
    # ax2.plot(magdf.Z + bshift, magdf.Bz, "tab:blue", label="B_z on axis")
    ax2.plot(magdf.Z, magdf.Bz, "tab:blue", label="B_z on axis")
    ax2.set_ylim((0, 1.0))
    ax2.set_ylabel("$B_z$ (T)", color="tab:blue")
    ax2.tick_params(axis='y', labelcolor="tab:blue")
    plt.tight_layout()

    fig.savefig(fnamestub + "_fields_2.png")
    ax.set_xlim(-.057, 0)
    ax.set_ylim(0, 0.006)
    fig.savefig(fnamestub + "_fields_zoom_2.png")
    plt.close(fig)

#     # ################ Acceptance Plot
#     fig = plot_ang_with_z(beam, ylim=YLIM_ACCEPT, xlim=XLIM, title=fnamestub + " - Acceptance")
#     ax = fig.gca()

#     for bmax in [1.5, 2.0, 2.5]:
#         accept = np.rad2deg(np.arcsin(np.sqrt(magdf.Bz/bmax)))
#         ax.plot(magdf.Z, accept, "--", label=str(bmax)+" T")
#     ax.legend()

#     ax2 = ax.twinx()
#     ax2.plot(magdf.Z, magdf.Bz, "b-")
#     ax2.set_ylim((0, 1.0))
#     ax2.set_ylabel("$B_z$ (T)")
#     plt.tight_layout()

#     fig.savefig(f[:-4] + "_acceptance.png")
#     # plt.show()
#     plt.close(fig)

# beam = import_tou_as_beam("Rex1mm500Gs_36_28_6+0_7mm_long_anode_-5000V_0_8A.TOU")
# beam.plot_trajectories(y="kin_energy")
# beam.plot_trajectories(y="kin_energy_long")
# beam.plot_trajectories(y="kin_energy_trans")
# plt.show()

# def angle_evo():
#     fig, ax = plt.subplots(figsize=(12, 9))

#     ax2 = ax.twinx()
#     ax2.plot(magdf.Z, magdf.Bz, "b-")
#     ax2.set_ylim((0, 2.0))
#     ax2.set_ylabel("$B_z$ (T)")

#     pos = 0.025
#     z = np.linspace(pos, .200, 100)
#     b0 = field_interp(pos)
#     growth = np.sqrt(field_interp(z)/b0)
#     for ang0 in np.linspace(0, np.pi/6, 5):
#         ang = np.rad2deg(np.arcsin((np.sin(ang0) * growth)))
#         ax.plot(z, ang, "k--")



#     for bmax in [1.0, 1.5, 2.0, 2.5, 3.0]:
#         accept = np.rad2deg(np.arcsin(np.sqrt(magdf.Bz/bmax)))
#         ax.plot(magdf.Z, accept, "-", label=str(bmax)+" T")
#     ax.legend()


#     ax.set_xlim(-.1, 0.2)
#     ax.set_ylim(0, 180)
#     ax.set_title("Acceptance")
#     ax.set_xlabel("$z$ (m)")
#     ax.set_ylabel("$\phi$ (deg)")
#     plt.tight_layout()
#     return fig
# fig = angle_evo()
# plt.show()


    # ################### Relection Plot
    # fig, ax = plt.subplots(figsize=(12, 9))

    # ax.plot(magdf.Z, magdf.Bz, "k-", label="Field")
    # ax.legend()

    # for p in b._particles:
    #     cf = field_interp(p.z) / np.sin(p.ang_with_z_rad)**2
    #     ax.plot(p.z, cf, lw=".75")

    # ax.set_xlim(XLIM)
    # ax.set_ylim((0, 5.0))
    # ax.set_title(f + " - Reflection Field")
    # ax.set_xlabel("$z$ (m)")
    # ax.set_ylabel("Critical $B_z$ (T)")
    # plt.tight_layout()
    # plt.show()
    # # plt.savefig(f[:-4] + "_reflection.png")
    # plt.close(fig)
    # # print(f)
    # ################### Relection Plot