import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# df = pd.read_csv(r"C:\TRAKTEMP\REXGUN_THERMAL\700Gs\Rex1mm700Gs_32_27_23_7_+2_0mm_long_anode_2019-04-30-11-27-06\Rex1mm700Gs_32_27_23_7_+2_0mm_long_anode_2019-04-30-11-27-06_resultdump.csv")
df = pd.read_csv(r"C:\TRAKTEMP\REXGUN_THERMAL\500Gs\long_anode_xs_5mm\Rex1mm500Gs_36_28_6+0_7mm_long_anode_2019-04-17-17-05-28\Rex1mm500Gs_36_28_6+0_7mm_long_anode_2019-04-17-17-05-28_resultdump.csv")

df.EP_MIN_curden = df.EP_MIN_curden/10000
df.EP_AVG_curden = df.EP_AVG_curden/10000
df.EP_MAX_curden = df.EP_MAX_curden/10000

df.EP_MIN_curden_prj = df.EP_MIN_curden_prj/10000
df.EP_AVG_curden_prj = df.EP_AVG_curden_prj/10000
df.EP_MAX_curden_prj = df.EP_MAX_curden_prj/10000

df.EP_AVG_elong_prj_min = df.EP_AVG_elong_prj_min.apply(str).apply(np.complex)

p0 = df.groupby("p0")
p1 = df.groupby("p1")

# plt.figure()
# p1.plot("EP_ektheo", "EP_AVG_curden")
# plt.show()

fig = plt.figure()
for name, g in p1:
    plt.plot(g.EP_ektheo, g.EP_AVG_curden_prj, label=name)
plt.legend()
plt.xlabel("E (eV)")
plt.ylabel("j (A/cm^2)")
fig.savefig("./cden_500Gs.pdf")
plt.show()

fig = plt.figure()
for name, g in p1:
    plt.plot(g.EP_ektheo, np.real_if_close(g.EP_AVG_elong_prj_min.values), label=name)
plt.legend()
plt.xlabel("E (eV)")
plt.ylabel("E + SC (eV)")
plt.ylim(0, 10000)
fig.savefig("./scret_500Gs.pdf")
plt.show()