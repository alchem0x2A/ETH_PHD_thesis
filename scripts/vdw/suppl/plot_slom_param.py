import numpy
# import os, os.path
import matplotlib.pyplot as plt
plt.style.use("science")
# from numpy import meshgrid
from ..utils.kkr import kkr, matsubara_freq
from ..utils.lifshitz import alpha_to_eps, g_amb_alpha, g_amb
from ..utils.eps_tools import data_bulk
from . import data_path, img_path
from helper import gridplots, grid_labels, savepgf
# from os.path import join, exists, dirname, abspath

# curdir = abspath(dirname(__file__))
# img_path = join(curdir, "../../img", "suppl")

# if not exists(img_path):
    # os.makedirs(img_path)

freq_matsu = matsubara_freq(numpy.arange(0, 1000),
                             mode="energy")

# Cannot change!
def get_eps(index):
    entry = data_bulk[index]
    eps = numpy.sqrt(entry["eps_x_iv"] * entry["eps_z_iv"])
    gap = entry["gap"]
    freq = entry["freq_imag"]
    if gap > 0:
        omega_half = freq[(eps - 1) <= (eps[0] - 1) / 2][0]
        omega_p = numpy.sqrt((eps[0] - 1) * omega_half ** 2)
    else:
        omega_half = 0
        omega_p = 0
    print(data_bulk[index]["name"], omega_half, omega_p)
    return gap, omega_half, omega_p, eps[0]

def plot_main():
    fig, ax = gridplots(1, 2, ratio=2, r=0.7)
    ax1 = ax[0]
    ax2 = ax[1]
    res = []
    for i in range(len(data_bulk)):
        r_ = get_eps(i)
        if (r_[0] != 0) and (r_[2] > 5) and (r_[3] < 200):          # Not a metal
            res.append(list(r_))
    
    res = numpy.array(res)
    ax1.plot(res[:, 0], res[:, 1], "s", markersize=4, alpha=0.8,
             color="#4286f4")
    ax2.plot(res[:, 0], res[:, 2], "o", markersize=4, alpha=0.8,
             color="#ffbc49")
    p = numpy.polyfit(res[:, 0], res[:, 1], deg=1)
    print(p)
    xx = numpy.linspace(0, 12)
    ax1.plot(xx, numpy.poly1d(p)(xx), "--", color="grey")
    ax1.text(x=10, y=14, s="$\\hbar \\omega_{\\mathrm{g}}$" + \
             "={0:.1f}".format(*p) + \
             "$E_{\\mathrm{g}}^{\\mathrm{Bulk}}$" + "+{1:.1f} eV$".format(*p),
             ha="right")

    ax2.axhline(y=numpy.mean(res[:, 2]), ls="--", color="grey")
    ea = numpy.mean(res[:, 2])
    ax2.text(x=4, y=12, s="Average $\\hbar \\omega_{\\mathrm{p}}$=" +\
             "{0:.1f} eV".format(ea),
             ha='left',
             va="top")
    # ax2.plot(xx, 3.632 * xx ** 0.564, "--")
    ax2.set_ylim(6, 20)
    ax1.set_xlabel("$E_{\\mathrm{g}}^{\\mathrm{Bulk}}$ (eV)")
    ax2.set_xlabel("$E_{\\mathrm{g}}^{\\mathrm{Bulk}}$ (eV)")
    ax1.set_ylabel("$\\hbar \\omega_{\\mathrm{g}}$ (eV)")
    ax2.set_ylabel("$\\hbar \\omega_{\\mathrm{p}}$ (eV)")

    
    # ax1.text(0, 0.97, s="a", size=12,
             # weight="bold", va="top",
             # transform=fig.transFigure
    # )
    # ax2.text(0.33, 0.97, s="b", size=12,
             # weight="bold", va="top",
             # transform=fig.transFigure
    # )
    grid_labels(fig, ax, reserved_space=(0, 0))
    savepgf(fig, img_path / "eps_3D_slom.pgf")
    
    


if __name__ == "__main__":
    plot_main()
