import matplotlib.pyplot as plt
plt.style.use("science")
import numpy
import scipy

data = numpy.genfromtxt("Id_Vg.csv",
                        delimiter=",",
                        skip_header=14)
V_g = data[:, 0]
I_d = data[:, 1]
cycles = data[:, -2]
condition_cycle = numpy.where(cycles == 1)

V_g_valid = V_g[condition_cycle]
I_d_valid = I_d[condition_cycle]

figure = plt.figure(figsize=(3, 3))
ax = figure.add_subplot(111)
ax.set_xlim(-100, 100)
ax.set_yscale("log")
ax.set_xlabel("$V_{\\mathrm{g}}$ (V)")
ax.set_ylabel("$I_{\\mathrm{d}}$ (A)")
print(max(I_d_valid) / min(I_d_valid))
ax.plot(V_g_valid, I_d_valid, "o")
figure.savefig("best_IV.svg")




