from import_tou import particles_from_tou
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# particles = []
# #for particle in read_tou_blockwise("sample.TOU", zmin=3.776107E-01, zmax=3.797542E-01):
# for particle in read_tou_blockwise("sample_big.TOU"):
# #for particle in read_tou_blockwise("sample.TOU", zmin=3.776107E-01, zmax=-1):
#     particles.append( SimpleTouParticle(particle[0], particle[1]))




def get_angles(particle):
    vx = particle.trajectory["vx"]
    vy = particle.trajectory["vy"]
    vz = particle.trajectory["vz"]
    lng = np.sqrt(vx**2 + vy**2 + vz**2)
    return np.arccos(vz/lng)

def get_max_angle(particle):
    return np.max(get_angles(particle))

def get_start_transverse_offset(particle):
    return np.sqrt(particle.trajectory["x"].iloc[0]**2 + particle.trajectory["y"].iloc[0]**2)

particles = particles_from_tou("sample_big.TOU", zmin=.005, zmax=0.2)
plt.figure("angles")
plt.figure("traject")
plt.figure("vtrans")
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
    plt.plot(particle.trajectory["z"], np.sqrt(particle.trajectory["vx"]**2 + particle.trajectory["vy"]**2))
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