import matplotlib.pyplot as plt
import numpy

data = numpy.genfromtxt("../../data/HSE-data/2H-MoS2_new_POTCAR.agr",
                        skip_header=515,
                        skip_footer=22)

z = data[:, 0]; rho = data[:, 1] / 1e-6
zero = numpy.zeros_like(z)
center = 13.80
left = 12.04
right = 15.17
z = z-center
plt.figure(figsize=(3.5, 3.5))
plt.style.use("science")
plt.fill_between(z, zero, rho, where=rho >= zero, facecolor="r")
plt.fill_between(z, zero, rho, where=rho <= zero, facecolor="g")
plt.plot([-10, 10], [0, 0], "--")
plt.axvline(x=left-center)
plt.axvline(x=right-center)
plt.ylim(-8, 8)
plt.xlim(-10, 10)
plt.savefig("../../tmp_img/density_MoS2.svg")
