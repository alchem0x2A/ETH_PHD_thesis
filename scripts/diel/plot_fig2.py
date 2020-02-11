import numpy as np
from . import data_path, img_path
from helper import gridplots, grid_labels, savepgf
from helper import get_color, add_cbar, add_img_ax
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from scipy.constants import epsilon_0, pi
from scipy.optimize import curve_fit
import os


def calc_alpha_freq(L=20,
                    direction="in_plane",
                    method="PBE"):
    assert direction in ("in_plane", "out_of_plane")
    assert method in ("PBE", "GW")
    base_dir = data_path / "distance" / "GW" /\
               "{0}/{1:d}/{2}/".format(method, L, direction)
    img_file = base_dir / "{0}_imaginary_epsilon.dat".format(direction)
    real_file = base_dir / "{0}_real_epsilon.dat".format(direction)
    img_data = np.genfromtxt(img_file)
    real_data = np.genfromtxt(real_file)
    freq = img_data[:, 0]
    eps = real_data[:, 1] + 1j * img_data[:, 1]
    if direction == "in_plane":
        alpha = (eps - 1) * L / (np.pi * 4)
    else:
        alpha = (eps - 1) * L / (np.pi * 4)
    return freq, alpha, eps


def plot_ax(fig, ax, method="PBE",
            direction="in_plane",
            legend=True):
    ax1, ax2 = ax
    for ax_ in [ax1, ax2]:
        ax_.set_xlabel("$\\hbar \\omega$ (eV)")
    if method == "GW":
        if direction == "in_plane":
            lim = [0, 15]
        else:
            # lim = [13, 17]
            # lim = [13, 17]
            lim = [0, 20]
    else:
        if direction == "in_plane":
            # lim = [4, 8]
            lim = [0, 12]
        else:
            # lim = [8, 12]
            lim = [0, 12]
    # for ax in [ax3, ax4]:
        # ax.set_xlim(6, )
    # Set ylabel
    if direction == "in_plane":
        tag = "\\parallel"
    else:
        tag = "\\perp"

    ax1.set_ylabel(
        "Re [$\\varepsilon^{{{0}}}_{{\\mathrm{{SL}}}}(\\omega)$]".format(tag))
    ax2.set_ylabel(
        "Re [$\\alpha^{{{0}}}_{{\\mathrm{{2D}}}}(\\omega)/(4\\pi \\varepsilon_0)$] (Å)".format(tag))
    # ax3.set_ylabel(
        # "Im $\\varepsilon^{{{0}}}_{{\\mathrm{{SL}}}}(\\omega)$".format(tag))
    # ax4.set_ylabel(
        # "Im $\\alpha^{{{0}}}_{{\\mathrm{{2D}}}}(\\omega)/(4\\pi \\varepsilon_0)$ (Å)".format(tag))

    for L in (20, 30, 40, 50, 60):
        freq, alpha, eps = calc_alpha_freq(
            L, direction=direction, method=method)
        cond = np.where((freq > lim[0]) & (freq < lim[1]))
        freq = freq[cond]
        alpha = alpha[cond]
        eps = eps[cond]
        # freq_out, alpha_out, eps_out = calc_alpha_freq(L, direction="out_of_plane", method=method)
        ax1.plot(freq, eps.real, label="{} Å".format(
            L), linewidth=1.25)
        ax2.plot(freq, alpha.real, linewidth=1.4)
        # ax3.plot(freq, eps.imag, linewidth=1.25)
        # ax4.plot(freq, alpha.imag, linewidth=1.25)
    if legend:
        l = ax1.legend(labelspacing=0.2)
        l.set_title("{} $L$".format(dict(GW="G$_{0}$W$_{0}$", PBE="PBE")[method]))
    return


def plot_main():
    fig, ax = gridplots(2, 2, r=1, ratio=1.25)
    plot_ax(fig, ax[0: 2], direction="in_plane", legend=True)
    plot_ax(fig, ax[2: ], direction="out_of_plane", legend=False)
    grid_labels(fig, ax)
    savepgf(fig, img_path / "freq-compare.pgf", preview=True)


if __name__ == '__main__':
    plot_main()
