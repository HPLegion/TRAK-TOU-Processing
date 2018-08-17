from import_tou import particles_from_tou
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.constants import speed_of_light, elementary_charge, atomic_mass


# def get_start_transverse_offset(particle):
#     return np.sqrt(particle.trajectory["x"].iloc[0]**2 + particle.trajectory["y"].iloc[0]**2)

# particles = particles_from_tou("sample_big.TOU", zmin=.00, zmax=0.2)
particles = particles_from_tou("NA53FC1F_0_1A_0_4kGs.TOU")#, zmin=.12, zmax=0.175)
# particles = particles_from_tou("./debug-files/Bgun2Fs_0_1A_0_4kGs.TOU")#, zmin=.25, zmax=0.38)
plt.figure("angles")
plt.figure("traject")
plt.figure("tx")
plt.figure("ty")
plt.figure("tz")
plt.figure("xy")
plt.figure("vtrans")
plt.figure("energy")
plt.figure("energy_long")
plt.figure("energy_trans")
plt.figure("traject2")
plt.figure("traject2").add_subplot(111, projection='3d')
plt.xlabel("x")
plt.ylabel("y")
for par in particles:
    plt.figure("angles")
    plt.plot(par.z, par.ang_with_z)
    plt.figure("traject")
    plt.plot(par.z, np.sqrt(par.x**2 + par.y**2))
    plt.figure("tx")
    plt.plot(par.t, par.x)
    plt.figure("ty")
    plt.plot(par.t, par.y)
    plt.figure("tz")
    plt.plot(par.t, par.z)
    plt.figure("xy")
    plt.plot(par.x, par.y)
    plt.figure("traject2")
    plt.plot(par.x, par.y, par.z)
    plt.figure("vtrans")
    plt.plot(par.z, np.sqrt(par.v_x**2 + par.v_y**2))
    plt.figure("energy")
    plt.plot(par.z, par.kin_energy)
    plt.figure("energy_long")
    plt.plot(par.z, par.kin_energy_long)
    plt.figure("energy_trans")
    plt.plot(par.z, par.kin_energy_trans)
    # print(par.max_ang_with_z(zmin=0.005))

plt.show()
