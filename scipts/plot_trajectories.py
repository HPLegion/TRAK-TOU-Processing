import os
import matplotlib.pyplot as plt
from trak_geometry import parse_trak_geometry
from import_tou import beam_from_tou
import matplotlib as mpl
import tqdm
import pandas as pd
import numpy as np
import scipy.interpolate

CWD = "C:\TRAKTEMP\REXGUN_THERMAL\long_anode_l_highres"
XLIM = [-.06, .2]
YLIM = [0, .005]
YLIM_ACCEPT = [0, 180]

os.chdir(CWD)

egeo = parse_trak_geometry("ERex1mm_500Gs_long_anode_l.min", scale=0.001)
bgeo = parse_trak_geometry("BRex_36_28_6.MIN", xshift=0.0007, scale=0.001)
[print(reg.to_svg_path()) for reg in egeo]

fig, ax = plt.subplots()
ax.set_xlabel("z")
ax.set_ylabel("r")
ax.set_title("test")
ax.set_xlim()
ax.set_ylim()
for reg in egeo[1:]:
    path = reg.to_mpl_path()
    patch = mpl.patches.PathPatch(path, edgecolor="k", facecolor="tab:gray")
    ax.add_patch(patch)
plt.show()

# magdf = pd.read_csv("./FBREX-36-28_6+0_7MM_500GS_AXIS_SCAN.TXT", comment="#", sep=r"\s+")
# magdf.Z = magdf.Z/1000
# field_interp = scipy.interpolate.interp1d(magdf.Z, magdf.Bz)

# files = os.listdir(CWD)
# files = [f for f in files if f.endswith("TOU")]


# for f in ["Rex1mm500Gs_36_28_6+0_7mm_long_anode_0V_0_8A.TOU"]:#tqdm.tqdm(files):
    
#     b = beam_from_tou(f)

#     ################ Trajectory Plot
#     fig, ax = plt.subplots(figsize=(12, 9))

#     # for reg in bgeo[1:]:
#     #     path = reg.to_mpl_path()
#     #     patch = mpl.patches.PathPatch(path, edgecolor="b", fill=False)
#     #     ax.add_patch(patch)

#     for reg in egeo[1:]:
#         path = reg.to_mpl_path()
#         patch = mpl.patches.PathPatch(path, edgecolor="k", facecolor="tab:gray")
#         ax.add_patch(patch)

#     b.plot_trajectories(ax=ax, lw=".75")

#     ax.set_xlim(XLIM)
#     ax.set_ylim(YLIM)
#     ax.set_title(f)
#     ax.set_xlabel("$z$ (m)")
#     ax.set_ylabel("$r$ (m)")
#     plt.tight_layout()
#     # plt.show()
#     plt.savefig(f[:-4] + ".png")
#     plt.close(fig)
#     # print(f)
#     ################ Trajectory Plot

    # ################ Acceptance Plot
    # fig, ax = plt.subplots(figsize=(12, 9))
    
    # ax2 = ax.twinx()
    # ax2.plot(magdf.Z, magdf.Bz, "b-")
    # ax2.set_ylim((0, 1.0))
    # ax2.set_ylabel("$B_z$ (T)")

    # b.plot_trajectories(y="ang_with_z", ax=ax, lw=".75", color="k")

    # for bmax in [1.0, 1.5, 2.0, 2.5, 3.0]:
    #     accept = np.rad2deg(np.arcsin(np.sqrt(magdf.Bz/bmax)))
    #     ax.plot(magdf.Z, accept, "--", label=str(bmax)+" T")
    # ax.legend()


    # ax.set_xlim(XLIM)
    # ax.set_ylim(YLIM_ACCEPT)
    # ax.set_title(f + " - Acceptance")
    # ax.set_xlabel("$z$ (m)")
    # ax.set_ylabel("$\phi$ (deg)")
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig(f[:-4] + "_acceptance.png")
    # plt.close(fig)
    # # print(f)
    # ################ Acceptance Plot

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

# fig, ax = plt.subplots(figsize=(12, 9))
    
# ax2 = ax.twinx()
# ax2.plot(magdf.Z, magdf.Bz, "b-")
# ax2.set_ylim((0, 2.0))
# ax2.set_ylabel("$B_z$ (T)")

# pos = 0.025
# z = np.linspace(pos, .200, 100)
# b0 = field_interp(pos)
# growth = np.sqrt(field_interp(z)/b0)
# for ang0 in np.linspace(0, np.pi/6, 5):
#     ang = np.rad2deg(ang0 * growth)
#     ax.plot(z, ang, "k--")



# for bmax in [1.0, 1.5, 2.0, 2.5, 3.0]:
#     accept = np.rad2deg(np.arcsin(np.sqrt(magdf.Bz/bmax)))
#     ax.plot(magdf.Z, accept, "-", label=str(bmax)+" T")
# ax.legend()


# ax.set_xlim(-.1, 0.2)
# ax.set_ylim(YLIM_ACCEPT)
# ax.set_title("Acceptance")
# ax.set_xlabel("$z$ (m)")
# ax.set_ylabel("$\phi$ (deg)")
# plt.tight_layout()
# plt.show()
# # plt.savefig(f[:-4] + "_acceptance.png")
# plt.close(fig)