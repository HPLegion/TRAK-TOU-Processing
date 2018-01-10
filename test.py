from import_tou import particles_from_tou
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.constants import speed_of_light, elementary_charge, atomic_mass

# particles = []
# #for particle in read_tou_blockwise("sample.TOU", zmin=3.776107E-01, zmax=3.797542E-01):
# for particle in read_tou_blockwise("sample_big.TOU"):
# #for particle in read_tou_blockwise("sample.TOU", zmin=3.776107E-01, zmax=-1):
#     particles.append( SimpleTouParticle(particle[0], particle[1]))



def get_total_norm_momentum(particle):
    px = particle.trajectory["px"]
    py = particle.trajectory["py"]
    pz = particle.trajectory["pz"]
    mom = np.sqrt(px**2 + py**2 + pz**2)
    return mom

def get_si_mass(particle):
    return particle.mass * atomic_mass

def get_total_momentum(particle):
    return get_total_norm_momentum(particle) * speed_of_light * get_si_mass(particle)

def get_angles(particle):
    mom = get_total_norm_momentum(particle)
    return np.arccos(particle.trajectory["pz"]/mom)



def energy(particle):
    si_mom = get_total_momentum(particle)
    si_mass = get_si_mass(particle)
    return (np.sqrt((si_mom*speed_of_light)**2 + (si_mass*(speed_of_light**2))**2)-si_mass*(speed_of_light**2))/elementary_charge

def get_max_angle(particle):
    return np.max(get_angles(particle))

def get_start_transverse_offset(particle):
    return np.sqrt(particle.trajectory["x"].iloc[0]**2 + particle.trajectory["y"].iloc[0]**2)

particles = particles_from_tou("sample_big.TOU", zmin=.00, zmax=0.2)
plt.figure("angles")
plt.figure("traject")
plt.figure("vtrans")
plt.figure("energy")
# plt.figure("traject").add_subplot(111, projection='3d')
plt.xlabel("x")
plt.ylabel("y")
for particle in particles:
    # print(particle.id)
    # print(180/3.154159*get_angles(particle))
    plt.figure("angles")
    plt.plot(particle.trajectory["z"], np.gradient(get_angles(particle),particle.trajectory.z))
    plt.figure("traject")
    plt.plot(particle.trajectory["z"], np.sqrt(particle.trajectory["x"]**2 + particle.trajectory["y"]**2))
    plt.figure("vtrans")
    plt.plot(particle.trajectory["pz"], np.sqrt(particle.trajectory["px"]**2 + particle.trajectory["py"]**2))
    plt.figure("energy")
    plt.plot(particle.trajectory["z"], energy(particle))
    print(particle.mass)

    # if particle.id%10==0:
        # plt.plot(particle.trajectory["x"], particle.trajectory["y"], particle.trajectory["z"])

plt.figure()
# plt.plot([get_start_transverse_offset(p) for p in particles], [get_max_angle(p) for p in particles])
plt.plot([p.trajectory.z[0] for p in particles], [get_start_transverse_offset(p) for p in particles])
plt.show()
# for p in particles:
#     plt.plot(p.id, np.min(p.trajectory.vz),"rx")
#     print(p.id)
#     print(np.min(p.trajectory.vz))
# plt.show()