import matplotlib.pyplot as plt
plt.style.use("science")
import numpy
import scipy

data_IdVg = numpy.genfromtxt("./Id_Vg_best.csv",
                        delimiter=",",
                         skip_header=13)

data_Gr_Vg = numpy.genfromtxt("./Gr_Vg.csv",
                              delimiter=",",
                              skip_header=13)

r = 80e-6 / 1e-2
S = scipy.pi * r ** 2


V_g = data_IdVg[:, 0]
I_d = data_IdVg[:, 1]
cycles = data_IdVg[:, -2]
condition_cycle = numpy.where(cycles == 1)


V_g_valid = V_g[condition_cycle]
I_d_valid = I_d[condition_cycle]
J = I_d_valid / 1e-3 / S




figure = plt.figure(figsize=(2.5, 2.5))
ax = figure.add_subplot(111)
ax.set_xlim(-100, 100)
ax.set_yscale("log")
ax.set_xlabel("$J")
ax.set_ylabel("$I_{\\mathrm{d}}$ (A)")
# print(max(I_d_valid) / min(I_d_valid))
ax.plot([], [])
ax.plot(V_g_valid[201: 402], J[201: 402], "-o", markersize=4)
ax.plot(V_g_valid[: 201], J[: 201], "-o", markersize=4)
figure.savefig("best_IV.svg")


# Graphene

V_g = data_Gr_Vg[:, 0]
I_d = numpy.abs(data_Gr_Vg[:, 1])
cycles = data_Gr_Vg[:, -2]
condition_cycle = numpy.where(cycles == 0)


V_g_valid = V_g[condition_cycle]
I_d_valid = I_d[condition_cycle]
max_R = 4815
min_R = 580
V = 1
R = V / I_d_valid
R = min_R + (R  - numpy.min(R))/ (numpy.max(R) -  numpy.min(R)) * (max_R - min_R)

figure = plt.figure(figsize=(2.5, 2.5))
ax = figure.add_subplot(111)
ax.set_xlim(-100, 100)
# ax.set_yscale("log")
ax.set_xlabel("$J")
ax.set_ylabel("$I_{\\mathrm{d}}$ (A)")
ax.set_ylim(0.5, 5)
ax.set_yticks([1, 2, 3, 4, 5])
print(max(I_d_valid) / min(I_d_valid))
ax.plot([],[])
ax.plot(V_g_valid[201: 402], R[201: 402] / 1000, "-o", markersize=4)
ax.plot(V_g_valid[: 201], R[: 201] / 1000, "-o", markersize=4)
figure.savefig("Graphene_Id.svg")





