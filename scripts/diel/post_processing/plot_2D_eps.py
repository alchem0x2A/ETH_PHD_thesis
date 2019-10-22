import matplotlib.pyplot as plt
import numpy
from scipy.constants import pi, epsilon_0

plt.figure(figsize=(3.5, 3.5))
plt.style.use("science")
alpha = 5
delta = numpy.linspace(4.5, 5.5, 1000)
eps = 1 / (1 - alpha / delta)
# print(eps)
plt.plot((delta - alpha)/alpha * 100, eps, "o", markersize=2)
plt.axhline(y=1, ls="--")
plt.ylim(-100, 100)
plt.xlabel("Difference between $\\delta_{\\mathrm{2D}}$ and $\\alpha^{\\perp}/\\varepsilon_{0}$ (%)")
plt.ylabel("Calculated $\\varepsilon_{\\mathrm{2D}}^{\\perp}$")
plt.savefig("../../tmp_img/scale_error.svg")
