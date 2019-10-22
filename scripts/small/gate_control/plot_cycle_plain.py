import numpy
import matplotlib.pyplot as plt
plt.style.use("science")

filename = "IV_cycles.csv"
data = numpy.genfromtxt(filename, delimiter=",",
                        skip_header=13)
cycles = data[:, -2]
cond = numpy.where(cycles >= 1)[0]
Vg = data[cond, 0]
Id = numpy.abs(data[cond, 1])
t = data[cond, 4]
t = t - t[0]

fig = plt.figure(figsize=(7.5, 2.1))
ax = fig.add_subplot(111)
# ax2 = ax.twinx()
ax.set_yscale("log")
y = numpy.linspace(*ax.get_xlim(), 10)
xx, yy = numpy.meshgrid(y, t)
_, zz = numpy.meshgrid(y, Vg)
# pc = ax.pcolor(yy, xx, zz, alpha=0.8, vmax=100, vmin=-50)
ax.plot(t, Id, "-o", markersize=3, rasterized=True)
ax.set_ylim(8e-8, 1e-4)
# fig.colorbar(pc, ticks=[-50, 0, 50, 100])
fig.savefig("cycles_plain.svg")

fig = plt.figure(figsize=(7.5, 2.1))
ax = fig.add_subplot(111)
# ax2 = ax.twinx()
# ax.set_yscale("log")
ax.plot([], [])
# pc = ax.pcolor(yy, xx, zz, alpha=0.8, vmax=100, vmin=-50)
ax.plot(t, Vg, "-o", markersize=3, rasterized=True)
ax.set_yticks([-50, 0, 50, 100])
# ax.set_ylim(8e-8, 1e-4)
# fig.colorbar(pc, ticks=[-50, 0, 50, 100])
fig.savefig("cycles_plain_Vg.svg")
