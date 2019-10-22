import numpy
import matplotlib
import matplotlib.pyplot as plt

files = {("gr", "glovebox"): "gr_glovebox.csv",
         ("f16", "glovebox"): "F16CuPc_glovebox.csv",
         }

ratios = {("gr", "glovebox"): 1.0 / 1.9,
          ("f16", "glovebox"): 1.0 / 1.6,
          }

fig = plt.figure(figsize=(4, 4))
plt.style.use("science")
ax = fig.add_subplot(111)
ax.set_xlabel("$V_{\\mathrm{g}}$ (V)")
ax.set_ylabel("$\\rho$ (k$\Omega$)")
ax.set_xlim(-100, 100)

for name in files:
    print_name = " ".join(name)
    f_name = files[name]
    data = numpy.genfromtxt(f_name,
                            skip_header=13,
                            delimiter=",")
    Vg = data[:, 0]
    Id = data[:, 1]
    Vd = data[0, -1]
    Rds = Vd / Id * ratios[name]
    ax.plot(Vg + 25, Rds / 1000, label=print_name)

ax.legend(loc=0)
fig.savefig("gr_dirac_point_glovebox.pdf")
