import matplotlib.pyplot as plt
import numpy
import json
from scipy.constants import pi, k, epsilon_0, e

data = json.load(open("../../data/HSE-data/all_data.json", "r"))

def model_alpha(x, ratio=0.1657):
    return 1 / ( ratio * x - 0.0243)

maters = []
alpha = []
Eg = []

# for entry in data:
#     if entry["gap_2D"] is not None:
#         if entry["prototype"] == "MoS2":
#             prefix = "2H-"
#         elif entry["prototype"] == "CdI2":
#             prefix = "1T-"
#         else:
#             prefix = ""
#         maters.append("{0}{1}".format(prefix, entry["name"]))
#         alpha.append([entry["alphax_2D"], entry["alphaz_2D"]])
#         Eg.append(entry["gap_2D"])

# alpha = numpy.array(alpha)
# Eg = numpy.array(Eg)

fig = plt.figure(figsize=(5.5, 2.5))
plt.style.use("science")

names = ["2H-MoS2", "2H-MoSe2", "2H-MoTe2",
         "2H-WS2", "2H-WSe2", "2H-WTe2"]
xx = numpy.arange(6)

r_para = [40.0, 45.5, 58.1, 35.7, 41.0, 53.5]
r_perp = [2.50, 2.72, 3.07, 2.46, 2.71, 3.17]


ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# res = sorted(zip(Eg, alpha, maters))
# maters = [n for e, a, n in res]
# ax = [max(a[0], 1/1.2) for e, a, n in res]
# az = [a[1] for e, a, n in res]
# Eg = [e for e, a, n in res]

# xx = numpy.arange(0, len(maters))

# for r in numpy.linspace(0.12, 0.20, 100):
    # a_model = model_alpha(numpy.array(Eg), r)
    # ax1.plot(xx +0.5, a_model, color="cyan",
             # alpha=0.2)

ax1.bar(xx, r_para, width=0.5, color="#559dc9")
ax2.bar(xx, r_perp, width=0.5, color="#d9ad66")

ax1.set_ylabel("$r_0^{\\parallel}$ ($\\mathrm{\\AA}$)")
ax2.set_ylabel("$r_0^{\\perp}$ ($\\mathrm{\\AA}$)")

ax1.set_ylim(0, 70)
ax2.set_ylim(0, 6)

ax1.set_xticks(xx)
ax2.set_xticks(xx)
mnames = [n[0] + n[1:].replace("2", "$_{2}$") for n in names]
ax1.set_xticklabels(mnames, rotation=-30, va="top", ha="left")
ax2.set_xticklabels(mnames, rotation=-30, va="top", ha="left")
plt.tight_layout()
# plt.savefig("../../tmp_img/ellipsoid.png")
plt.savefig("../../tmp_img/ellipsoid.svg")

