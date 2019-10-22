import matplotlib.pyplot as plt
plt.style.use("science")
import numpy
import scipy

doping = ["ots-1", "plasma-1"]
gr = ["ots-gr", "plasma-gr"]
ratio = [1.8, 1.0]

fig = plt.figure(figsize=(4, 4))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax1.set_xlabel(r"$V_{\mathrm{g}}$ (V)")
ax1.set_ylabel(r"$I_{\mathrm{d}}$ Transistor (A)")
ax1.set_yscale("log")
ax2.set_ylabel(r"$I_{\mathrm{d}}$ Graphene ($\mathrm{\mu}$A)")


colors = ["#4286f4", "#f49b41"]

for ind, m in enumerate(doping):
    file_name = "{}.csv".format(m)
    data = numpy.genfromtxt(file_name,
                            delimiter=",",
                            skip_header=13)
    data = data[: 201, :]
    Vg = data[:, 0]
    Id = numpy.abs(data[:, 1])
    print(m, max(Id) / min(Id))
    ax1.plot(Vg, Id, label=m, color=colors[ind])

for ind, m in enumerate(gr):
    file_name = "{}.csv".format(m)
    data = numpy.genfromtxt(file_name,
                            delimiter=",",
                            skip_header=13)
    data = data[: 201, :]
    Vg = data[:, 0]
    Id = numpy.abs(data[:, 1])
    print(m)
    ax2.plot(Vg, Id * ratio[ind] / 10 ** -6, "--", label=m, color=colors[ind])



ax1.legend()
fig.tight_layout()
fig.savefig("comparison_doping.pdf")
