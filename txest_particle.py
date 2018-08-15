import numpy as np
from matplotlib import pyplot as plt
R = 5e6 / (2*np.pi)
w = 2*np.pi*5
vzz = 5e7
x0 = 2
y0 = -1
sx = 3
sy = 4


t = np.arange(0,1,1e-5)

x = sx * R * np.sin(w*t)
y = sy * R * np.cos(w*t)
z = vzz*t

vx = sx * R * w * np.cos(w*t)
vy = - sy * R * w * np.sin(w*t)
vz = vzz *t/t

vabs = np.sqrt(vx**2 + vy**2 + vz**2)
vt = np.sqrt(vx**2 + vy**2)

phi = np.arccos(vz/vabs)

plt.figure()
plt.plot(t, x)
plt.xlabel("t")
plt.title("x")

plt.figure()
plt.plot(t, y)
plt.xlabel("t")
plt.title("y")

plt.figure()
plt.plot(t, z)
plt.xlabel("t")
plt.title("z")

plt.figure()
plt.plot(t, vx)
plt.xlabel("t")
plt.title("vx")

plt.figure()
plt.plot(t, vy)
plt.xlabel("t")
plt.title("vy")

plt.figure()
plt.plot(t, vz)
plt.xlabel("t")
plt.title("vz")

plt.figure()
plt.plot(t, phi)
plt.xlabel("t")
plt.title("phi")

plt.show()
