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
for entry in data:
    if entry["gap_2D"] is not None:
        if entry["prototype"] == "MoS2":
            prefix = "2H-"
        elif entry["prototype"] == "CdI2":
            prefix = "1T-"
        else:
            prefix = ""
        maters.append("{0}{1}".format(prefix, entry["name"]))
        alpha.append([entry["alphax_2D"], entry["alphaz_2D"]])
        Eg.append(entry["gap_2D"])

# alpha = numpy.array(alpha)
# Eg = numpy.array(Eg)

fig = plt.figure(figsize=(8, 4.0))
plt.style.use("science")

ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

res = sorted(zip(Eg, alpha, maters))
maters = [n for e, a, n in res]
ax = [max(a[0], 1/1.2) for e, a, n in res]
az = [a[1] for e, a, n in res]
Eg = [e for e, a, n in res]

xx = numpy.arange(0, len(maters))

# for r in numpy.linspace(0.12, 0.20, 100):
    # a_model = model_alpha(numpy.array(Eg), r)
    # ax1.plot(xx +0.5, a_model, color="cyan",
             # alpha=0.2)

ax2.plot(Eg, "o-", markersize=6)
ax1.bar(xx, ax, color="#FFCC00", alpha=0.7)
ax1.bar(xx, az, color="#FF5555", alpha=0.5)
ax1.set_xticks(range(0, len(maters)))
mnames = [n[0] + n[1:].replace("2", "$_{2}$") for n in maters]
ax1.set_xticklabels(mnames, rotation="vertical")
ax1.set_ylim(0, 11)
plt.tight_layout()
plt.savefig("../../tmp_img/all_data.svg")

