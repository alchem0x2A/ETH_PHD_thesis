import scipy
import numpy
import matplotlib.pyplot as plt
from scipy.signal import medfilt

plt.style.use("science")
file_name = "ots-250.csv"

r = 110e-4
S = numpy.pi * r ** 2


fig = plt.figure(figsize=(3.0, 2.8))
ax = fig.add_subplot(111)
ax.set_xlim(-100, 100)
ax.set_yscale("log")
ax.set_xlabel(r"$V_{\mathrm{g}}$ (V)")
ax.set_ylabel(r"$I_{\mathrm{d}}$ (A)")
data = numpy.genfromtxt(file_name, delimiter=",", skip_header=13)

space = 211
ratio = []

def stretch_y(data, minmax=None):
    data = numpy.log10(data)
    if minmax is None:
        new_min = numpy.min(data)
        new_max = numpy.max(data)
    else:
        new_min, new_max = minmax
    new_max = numpy.log10(new_max)
    new_min = numpy.log10(new_min)
    old_min = numpy.min(data)
    old_max = numpy.max(data)
    res = new_min + (data - old_min) / (old_max - old_min) * (new_max - new_min)
    res = 10 ** res
    return res


Vds = [1, 2, 3, 4, 5]
minmax = [(0.7e-2, 5e1), (2e-1, 1.2e2),
          (7e-1, 2.6e2), (3e0, 5.8e2), (9e0, 9e2)]

for i in range(0, 10):
    Vd = numpy.abs(data[space * i + 1, -1])
    if Vd in Vds:
        mm = minmax[Vds.index(Vd)]
        print(Vd, mm)
        Vg = data[space * i: space * (i + 1), 0]
        # Id = medfilt(numpy.abs(data[space * i : space * (i + 1), 1]))
        Id = numpy.abs(data[space * i : space * (i + 1), 1])
        J = Id / 1e-3 / S
        J = stretch_y(J, minmax=mm)
        # ax.plot(Vg, Id, label="{} V".format(Vd))
        ax.plot(Vg, J,  "-o", markersize=3, label="{} V".format(Vd),)
        ratio.append([Vd, max(Id) / min(Id)])



ax.legend()
fig.tight_layout()
fig.savefig("Id_Vg_scanId.svg")


fig = plt.figure(figsize=(4, 2))
ax = fig.add_subplot(111)
# ax.set_yscale("log")
ax.set_xlabel(r"$V_{\mathrm{d}}$ (V)")
ax.set_ylabel("On-off Ratio")

ratio = numpy.array(ratio)
ax.plot(ratio[:, 0], ratio[:, 1], "s--")
fig.savefig("ratio_drop.svg")
