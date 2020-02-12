import numpy as np
from . import data_path, img_path
from helper import gridplots, grid_labels, savepgf
from helper import get_color, add_cbar, add_img_ax
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os


from scipy.constants import epsilon_0, pi

from scipy.stats import linregress
from scipy.constants import pi
# from model_eps import plot_eps_para, plot_eps_perp
import re
import ase
import ase.io


root = data_path / "multi-mx2"

colors = {"mos2": "#1f77b4",
          "mose2": "#ff7f0e",
          "mote2": "#2ca02c",
          "ws2": "#d62728",
          "wse2": "#9467bd",
          "wte2": "#8c564b",
          }


def get_eps_matrix(mater, n):
    work_dir = root / "{0}/{1}/eps".format(mater, n)
    outcar = work_dir / "OUTCAR.COMPLETE"
    assert outcar.exists()
    pattern = "".join((r"MACROSCOPIC STATIC DIELECTRIC TENSOR ",
                       r"\(including local field effects in DFT\)",
                       r"[\s\n]+[-]{20,}\n(.+?)[-]{20,}"))
    content = open(outcar.as_posix(), "r").read()
    matches = re.findall(pattern, content, re.DOTALL)
    assert(len(matches) > 0)    # At least one calculated!
    eps_string = matches[-1]
    eps_matrix = np.fromstring(eps_string.strip(),
                               dtype="float",
                               sep=" ")
    assert eps_matrix.shape[0] == 9
    eps_matrix = eps_matrix.reshape(3, 3)    # Reshape to Tensor
    return eps_matrix


def get_lattice_L(mater, n):
    work_dir = root / "{0}/{1}/eps".format(mater, n)
    poscar = work_dir / "POSCAR"
    assert poscar.exists()
    atoms = ase.io.read(poscar, format="vasp")
    L = atoms.get_cell()[-1, -1]
    return L


def plot_ax(fig, ax):
    ax1, ax2 = ax

    n_range = np.arange(1, 7)
    ext_n = np.linspace(0, 7)

    for m in ["Mo", "W"]:
        for x in ["S", "Se", "Te"]:
            mater = "{0}{1}2".format(m, x)
            alpha_para = []
            alpha_perp = []
            for n in n_range:
                eps_matrix = get_eps_matrix(mater, n)
                L = get_lattice_L(mater, n)
                eps_para = (eps_matrix[0, 0] + eps_matrix[1, 1]) / 2
                eps_perp = eps_matrix[2, 2]
                alpha_para_ = L * (eps_para - 1) / (4 * pi)
                alpha_perp_ = L * (eps_perp - 1) / eps_perp / (4 * pi)
                alpha_para.append(alpha_para_)
                alpha_perp.append(alpha_perp_)
            alpha_para = np.array(alpha_para)
            alpha_perp = np.array(alpha_perp)
            k_para, b_para, *_1 = linregress(x=n_range, y=alpha_para)
            k_perp, b_perp, *_2 = linregress(x=n_range, y=alpha_perp)
            print(b_para, _1, b_perp, _2)
            l1, = ax1.plot(n_range, alpha_para, "o", markersize=5,
                           label="2H-" + mater.replace("2", "$_{2}$"))
            l2, = ax2.plot(n_range, alpha_perp, "o", markersize=5,
                           label="2H-" + mater.replace("2", "$_{2}$"))
            ax1.plot(ext_n, k_para * ext_n + b_para, "--", color=l1.get_c())
            ax2.plot(ext_n, k_perp * ext_n + b_perp, "--", color=l2.get_c())

    ax1.set_xlabel("Number of Layers ($N$)")
    ax1.set_ylabel(
        "$\\alpha_{\\mathrm{NL}}^{\\parallel}/(4 \\pi \\varepsilon_0)$ (Å)")
    ax2.set_xlabel("Number of Layers ($N$)")
    ax2.set_ylabel(
        "$\\alpha_{\\mathrm{NL}}^{\\perp} / (4 \\pi \\varepsilon_0)$ (Å)")
    ax1.text(x=0.95, y=0.02, s="$\\alpha_{\\mathrm{NL}}^{\\parallel} = N \\alpha_{\\mathrm{2D}}^{\\parallel}$ \n $R^{2} > 0.9999$",
             ha="right", va="bottom",
             transform=ax1.transAxes)
    ax2.text(x=0.95, y=0.02, s="$\\alpha_{\\mathrm{NL}}^{\\perp} = N \\alpha_{\\mathrm{2D}}^{\\perp}$ \n $R^{2} > 0.9999$",
             ha="right", va="bottom",
             transform=ax2.transAxes)
    ax2.legend()

    ax_img = inset_axes(ax1, width="30%", height="30%",
                        loc="upper left")
    add_img_ax(ax_img, img_path / "sub_img" / "NL.png")
    ax_img.text(x=0.5, y=-0.02, s="$N$ layers",
                ha="center",
                va="top",
                transform=ax_img.transAxes)


def plot_main():
    fig, ax = gridplots(1, 2, r=0.85, ratio=2)
    plot_ax(fig, ax)
    grid_labels(fig, ax)

    savepgf(fig, img_path / "fig-NL.pgf", preview=True)


if __name__ == '__main__':
    plot_main()
