import os
import matplotlib.pyplot as plt
from trak_geometry import parse_trak_geometry
from import_tou import beam_from_tou
import matplotlib as mpl

os.chdir("C:\TRAKTEMP\REXGUN_THERMAL\long_anode_s")

egeo = parse_trak_geometry("./ERex1mm_500Gs_long_anode.min", scale=0.001)
bgeo = parse_trak_geometry("BRex_36_28_6.MIN", xshift=0.0007, scale=0.001)

beams = [
    "Rex1mm500Gs_36_28_6+0_7mm_long_anode_0V_0_1A.TOU",
    "Rex1mm500Gs_36_28_6+0_7mm_long_anode_0V_0_2A.TOU",
    "Rex1mm500Gs_36_28_6+0_7mm_long_anode_0V_0_3A.TOU",
    "Rex1mm500Gs_36_28_6+0_7mm_long_anode_0V_0_4A.TOU",
    "Rex1mm500Gs_36_28_6+0_7mm_long_anode_0V_0_5A.TOU",
    "Rex1mm500Gs_36_28_6+0_7mm_long_anode_0V_0_6A.TOU",
    "Rex1mm500Gs_36_28_6+0_7mm_long_anode_0V_0_7A.TOU",
    "Rex1mm500Gs_36_28_6+0_7mm_long_anode_0V_0_8A.TOU"
]

fig, ax = plt.subplots()
for reg in bgeo[1:]:
    path = reg.to_mpl_path()
    patch = mpl.patches.PathPatch(path, edgecolor="b", fill=False)
    ax.add_patch(patch)
for reg in egeo[1:]:
    path = reg.to_mpl_path()
    patch = mpl.patches.PathPatch(path, edgecolor="k")
    ax.add_patch(patch)
for bf in beams:
    b = beam_from_tou(bf)
    b.plot_outer_radius(ax=ax)
plt.legend()
# ax.grid()
ax.axis("equal")
ax.set_xlabel("z")
ax.set_ylabel("r")
plt.show()
