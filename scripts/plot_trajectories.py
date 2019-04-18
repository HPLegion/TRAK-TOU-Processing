import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
import numpy as np
import scipy.interpolate

from tct import import_min_as_regions, import_tou_as_beam
from figure_macros import plot_ang_with_z, plot_trajectories

CWD = r"C:\TRAKTEMP\REXGUN_THERMAL\long_anode_xs_5mm"
XLIM = [-.06, .05]
YLIM = [0, .006]
YLIM_ACCEPT = [0, 30]

os.chdir(CWD)

egeo = import_min_as_regions("ERex1mm_500Gs_long_anode.min", scale=0.001)
# bgeo = import_min_as_regions("BRex_36_28_6.MIN", xshift=0.0007, scale=0.001)
egeo = [r.to_mpl_path() for r in egeo[1:]] # Skip domain boundaries during conversion
# bgeo = [r.to_mpl_path() for r in bgeo[1:]] # Skip domain boundaries during conversion

# magdf = pd.read_csv("./FBREX-36-28_6+0_7MM_500GS_AXIS_SCAN.TXT", comment="#", sep=r"\s+")
# magdf.Z = magdf.Z/1000
# field_interp = scipy.interpolate.interp1d(magdf.Z, magdf.Bz)

files = os.listdir(CWD)
files = [f for f in files if f.endswith("TOU")]


for f in tqdm.tqdm(files):
    fnamestub = f[:-4]
    beam = import_tou_as_beam(f)

    ################ Trajecotry Plot
    fig = plot_trajectories(beam, egeo=egeo, title=f, xlim=XLIM, ylim=YLIM)
    # fig.savefig(fnamestub + ".png")
    plt.show()
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

beam = import_tou_as_beam("Rex1mm500Gs_36_28_6+0_7mm_long_anode_-5000V_0_8A.TOU")
beam.plot_trajectories(y="kin_energy")
beam.plot_trajectories(y="kin_energy_long")
beam.plot_trajectories(y="kin_energy_trans")
plt.show()

def angle_evo():
    fig, ax = plt.subplots(figsize=(12, 9))

    ax2 = ax.twinx()
    ax2.plot(magdf.Z, magdf.Bz, "b-")
    ax2.set_ylim((0, 2.0))
    ax2.set_ylabel("$B_z$ (T)")

    pos = 0.025
    z = np.linspace(pos, .200, 100)
    b0 = field_interp(pos)
    growth = np.sqrt(field_interp(z)/b0)
    for ang0 in np.linspace(0, np.pi/6, 5):
        ang = np.rad2deg(np.arcsin((np.sin(ang0) * growth)))
        ax.plot(z, ang, "k--")



    for bmax in [1.0, 1.5, 2.0, 2.5, 3.0]:
        accept = np.rad2deg(np.arcsin(np.sqrt(magdf.Bz/bmax)))
        ax.plot(magdf.Z, accept, "-", label=str(bmax)+" T")
    ax.legend()


    ax.set_xlim(-.1, 0.2)
    ax.set_ylim(0, 180)
    ax.set_title("Acceptance")
    ax.set_xlabel("$z$ (m)")
    ax.set_ylabel("$\phi$ (deg)")
    plt.tight_layout()
    return fig
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