import numpy
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0, pi

def eps_para(alpha, L):
    return 1 + 4 * pi * alpha / L

def eps_perp(alpha, L):
    return 1 / (1 - 4 * pi * alpha / L)

def plot_eps_para(ax, alpha_0, dev, trans=0.7):
    L = numpy.linspace(10, 60, 100)
    [ax.plot(L, eps_para(a, L), c="#839abf",
             alpha=(1 - abs(a - alpha_0) / dev) * trans) \
     for a in numpy.linspace(alpha_0 - dev * 0.999,
                             alpha_0 + dev * 0.999)]
    return

def plot_eps_perp(ax, alpha_0, dev, trans=0.7):
    L = numpy.linspace(10, 60, 100)
    [ax.plot(L, eps_perp(a, L), c="#cca384",
             alpha=(1 - abs(a - alpha_0) / dev) * trans) \
     for a in numpy.linspace(alpha_0 - dev * 0.999,
                             alpha_0 + dev * 0.999)]
    return

plt.style.use("science")
fig = plt.figure(figsize=(5, 1.6))
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
L = numpy.linspace(10, 60, 100)
#color plots
plot_eps_para(ax, alpha_0=8.5, dev=4)
plot_eps_perp(ax2, alpha_0=0.47, dev=0.155)

# [ax.plot(L, eps_para(a, L), c="#839abf", alpha=(1 - abs(a - 9.0) / 4)) for a in numpy.linspace(5, 13, 100)]
# ax.set_title("$\\alpha^{\\parallel} = 9 \\pm 4$")
# ax.set_xlabel("L")
# ax.set_ylabel("Model $\\varepsilon^{\\parallel}$")

# [ax2.plot(L, eps_perp(a, L), c="#cca384", alpha=(1 - abs(a - 0.5) / 0.1)) for a in numpy.linspace(0.4, 0.6, 100)]
# ax2.set_title("$\\alpha^{\\perp} = 0.5 \\pm 0.1$")
# ax2.set_xlabel("L")
# ax2.set_ylabel("Model $\\varepsilon^{\\perp}$")
fig.tight_layout()
fig.savefig("../../tmp_img/model_eps_parallel.svg")
