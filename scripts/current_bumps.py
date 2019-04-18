import os
import matplotlib.pyplot as plt
from tct import import_min_as_regions, import_tou_as_beam
import matplotlib as mpl

CWD = r"C:\TRAKTEMP\REXGUN_THERMAL\long_anode_xs_5mm"
os.chdir(CWD)

egeo = import_min_as_regions("ERex1mm_500Gs_long_anode.min", scale=0.001)
bgeo = import_min_as_regions("BRex_36_28_6.MIN", xshift=0.0007, scale=0.001)
egeo = [r.to_mpl_path() for r in egeo[1:]] # Skip domain boundaries during conversion
bgeo = [r.to_mpl_path() for r in bgeo[1:]] # Skip domain boundaries during conversion

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

# beams = os.listdir(CWD)
# beams = [f for f in beams if f.endswith("TOU")]

fig, ax = plt.subplots()
for reg in bgeo[1:]:
    patch = mpl.patches.PathPatch(reg, edgecolor="b", fill=False)
    ax.add_patch(patch)
for reg in egeo[1:]:
    patch = mpl.patches.PathPatch(reg, edgecolor="k")
    ax.add_patch(patch)
for bf in beams:
    b = import_tou_as_beam(bf)
    b.plot_outer_radius(ax=ax)
plt.legend()
# ax.grid()
# ax.axis("equal")
ax.set_xlabel("z")
ax.set_ylabel("r")
plt.show()
