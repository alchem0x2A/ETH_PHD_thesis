import numpy as np
from . import data_path, img_path
from helper import gridplots, grid_labels, savepgf
from helper import get_color, add_cbar, add_img_ax
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os


from scipy.constants import epsilon_0, pi
from scipy.optimize import curve_fit, fsolve
from scipy.constants import pi, k, epsilon_0, e
from pathlib import Path
import json

import ase.db
import warnings
from ase.data import covalent_radii
from ase.io import read, write
from scipy.stats import linregress
from scipy.optimize import curve_fit
from .gpaw_data import get_data
from .epfl_data import get_data_epfl, get_data_epfl2
import csv


def convert_name(s):
    s = s.replace("2", "$_{2}$")
    s = s.replace("3", "$_{3}$")
    s = s.replace("4", "$_{4}$")
    return s


def plot_a(fig, ax):
    f_data = data_path / "HSE-data" / "all_data.json"
    data = json.load(open(f_data, "r"))

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

    ax1 = ax
    ax2 = ax1.twinx()

    res = sorted(zip(Eg, alpha, maters))
    maters = [n for e, a, n in res]
    ax = [max(a[0], 1/1.2) for e, a, n in res]
    az = [a[1] for e, a, n in res]
    Eg = [e for e, a, n in res]

    xx = np.arange(0, len(maters))

    # for r in numpy.linspace(0.12, 0.20, 100):
    # a_model = model_alpha(numpy.array(Eg), r)
    # ax1.plot(xx +0.5, a_model, color="cyan",
    # alpha=0.2)

    lg,  = ax2.plot(Eg, "o-", markersize=4.5)
    ax1.bar(xx, ax, color="#FFCC00", alpha=0.7,
            label="$\\alpha_{\\mathrm{2D}}^{\\parallel}$")
    ax1.bar(xx, az, color="#FF5555", alpha=0.7,
            label="$\\alpha_{\\mathrm{2D}}^{\\perp}$")
    ax1.set_xticks(range(0, len(maters)))
    mnames = [n[0] + convert_name(n[1:]) for n in maters]
    ax1.set_xticklabels(mnames, rotation="vertical",
                        size="x-small")
    ax1.tick_params("x", pad=2)
    ax1.set_ylim(0, 11)
    ax1.text(x=1, y=0.8,
             s="⇨", size="x-large",
             ha="right",
             color=lg.get_c(),
             transform=ax1.transAxes)
    ax1.legend(loc="upper center")
    ax1.set_ylabel("Polarizability / ($4 \\pi \\varepsilon_0$) (Å)")
    ax2.set_ylabel("$E_{\\mathrm{g}}$ (eV)")


def plot_bc(fig, ax):
    """
    Extract the alpha from the HSE xlsx files
    """
    ax_b, ax_c = ax

    db_file = data_path / "gpaw_data" / "c2db_small.db"
    if not os.path.exists(db_file):
        raise FileExistsError(("Please download the c2db data into ../../data/gpaw_data/ folder,"
                               "from https://cmr.fysik.dtu.dk/_downloads/c2db.db"))

    db = ase.db.connect(db_file)

    materials = []
    eps_x = []
    eps_z = []
    alpha_x = []
    alpha_z = []
    Eg_HSE = []
    thick = []

    def get_thick(atom_row):
        pos = atom_row.positions[:, -1]
        diff = covalent_radii[atom_row.numbers]
        zmax = np.max(pos + diff) - np.min(pos - diff)
        return zmax

    # REad VASP result
    f_hse = data_path / "HSE-data" / "2D_HSE.csv"
    reader = csv.reader(open(f_hse.as_posix(),
                             encoding="utf8"))
    next(reader)                    # skip line1
    for row in reader:
        if row[4] != "":
            name, proto = row[: 2]
            print(name, proto)
            L, E, ex, ey, ez, *_ = map(float, row[2:])
            if ez < ex:
                eps_z.append(ez)
                materials.append("-".join((name, proto)))
                e_xy = np.sqrt(ex * ey)
                ax = (e_xy - 1) / (4 * pi) * L
                az = (1 - 1/ez) * L / (4 * pi)
                ax = max(1 / 1.2, ax)
                eps_x.append(e_xy)
                eps_z.append(ez)
                alpha_x.append(ax)
                alpha_z.append(az)
                Eg_HSE.append(E)
                if proto == "ABX3":
                    thick.append(8.0)
                else:
                    mol = list(db.select(formula=name, prototype=proto))[0]
                    thick.append(get_thick(mol))

    print(len(alpha_x))
    alpha_x = np.array(alpha_x)
    alpha_z = np.array(alpha_z)
    Eg_HSE = np.array(Eg_HSE)
    thick = np.array(thick)

    colors = {"MoS2": "#AA0000", "CdI2": "#2C5AA0",
              "GaS": "#FFCC00", "BN": "#A05A2C",
              "P": "#447821", "CH": "#FF6600",
              "ABX3": "#6600FF"}

    cs = [colors[mat.split("-")[-1]] for mat in materials]

    def fit_func(x, a, b):
        return b / x

    exclude_eps = ["AgNO2", "Cd(OH)2", "Ca(OH)2", "SnF4",
                   "Li2(OH)2", "Rb2Cl2", "LiBH4", "NaCN",
                   "Mg(OH)2", "Na2(OH)2", "PbF4", "AgO4Cl", "Ag2I2"]

    materials_qe, *qe_data = get_data_epfl2(exclude=exclude_eps)
    gp_data = get_data()
    print(qe_data[0].shape)

    # x-direction
    # gpaw
    ss = 64
    ax_b.scatter(gp_data[2], 1 / gp_data[0], marker="s", alpha=0.3,
                 s=ss,
                 c="#1ba17a",
                 # label=("$\\alpha_{\\mathrm{2D}}^{\\parallel}$ Ref.XXX")
    )

    ax_b.scatter(qe_data[2], 1 / qe_data[0], marker="^", alpha=0.3,
                  s=ss,
                 c="#1ba17a",
                 # label=("$\\alpha_{\\mathrm{2D}}^{\\parallel}$ Ref.YYY")
    )

    ax_b.scatter(Eg_HSE, 1 / (alpha_x), marker="o",
                 edgecolors=None, alpha=0.5,
                 s=ss,
                 c=cs)
    res = linregress(x=Eg_HSE, y=1 / (alpha_x))
    xx = np.linspace(0, 8.5)
    yy = res.slope * xx + res.intercept
    ax_b.plot(xx, yy, "--")
    ax_b.text(x=0.01, y=0.99,
              s="$y={0:.2f}x{1:+.2f},\\ R^2={2:.2f}$".format(res.slope,
                                                             res.intercept,
                                                             res.rvalue),
              ha="left", va="top", size="small",
              transform=ax_b.transAxes)
    ax_b.set_xlabel("$E_{\\mathrm{g}}$ (eV)")
    ax_b.set_ylabel(
        "$(4 \\pi \\varepsilon_0)/\\alpha_{\\mathrm{2D}}^{\\parallel}$ (Å$^{-1}$)")
    ax_b.set_xlim(0, 8.5)
    ax_b.set_ylim(np.min(yy), 1.5)
    # lg = ax_b.legend(loc=0)
    # for lh in lg.legendHandles:
        # lh.set_alpha(0.80)

    # z-direction
    # gpaw
    ax_c.scatter(gp_data[3], gp_data[1] * 4 * pi, marker="s", alpha=0.3,
                 c="#cca384",
                 s=ss,
                 # label=("$\\alpha_{\\mathrm{2D}}^{\\perp}$ Ref.XXX")
    )

    ax_c.scatter(qe_data[-1], qe_data[1] * 4 * pi, marker="^", alpha=0.3,
                 c="#cca384",
                 s=ss,
                 # label=("$\\alpha_{\\mathrm{2D}}^{\\perp}$ Ref.YYY")
    )

    # hse
    ax_c.scatter(thick, alpha_z * 4 * pi, marker="o",
                 s=ss,
                 edgecolors=None,
                 alpha=0.5, c=cs)
    res = linregress(x=thick, y=alpha_z * 4 * pi)
    xx = np.linspace(1.5, 10.5)
    yy = res.slope * xx + res.intercept
    ax_c.plot(xx, yy, "--")
    ax_c.set_xlim(1.5, 10.5)
    ax_c.text(x=0.01, y=0.99,
              s="$y={0:.2f}x{1:+.2f},\\ R^2={2:.2f}$".format(res.slope,
                                                             res.intercept,
                                                             res.rvalue),
              ha="left", va="top", size="small",
              transform=ax_c.transAxes)
    ax_c.set_xlabel("$\\delta^{\\mathrm{cov}}_{\\mathrm{2D}}$ (Å)")
    ax_c.set_ylabel(
        "$\\alpha_{\\mathrm{2D}}^{\\perp} / (4 \\pi \\varepsilon_0)$ (Å)")
    ax_c.set_ylim(*(ax_c.get_xlim()))
    # lg = ax_c.legend(loc=0)
    # for lh in lg.legendHandles:
        # lh.set_alpha(0.80)

    ax_img = inset_axes(ax_c, width="40%", height="20%",
                        loc="lower right")
    add_img_ax(ax_img, img_path / "sub_img" / "delta_cov.png")
    ax_img.text(x=0, y=0.5,
                s="$\\delta_{\\mathrm{2D}}^{\\mathrm{cov}}$",
                ha="right",
                size="small",
                transform=ax_img.transAxes)

def add_legend_right(fig, ax):
    """Add dummy legend at right side of ax"""
    colors = {"MoS2": "#AA0000", "CdI2": "#2C5AA0",
              "GaS": "#FFCC00", "BN": "#A05A2C",
              "P": "#447821", "CH": "#FF6600",
              "ABX3": "#6600FF"}
    symbs = {"MoS2": "2H-MX$_2$", "CdI2": "1T-MX$_2$",
              "GaS": "GaS", "BN": "BN",
              "P": "P$_4$", "CH": "CH",
              "ABX3": "ABX$_3$"}
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ss = 64
    for mater in symbs.keys():
        ax.scatter(-1, -1, marker="o", s=ss,
                   edgecolors=None,
                   alpha=0.8, c=colors[mater],
                   label=symbs[mater])
    ax.scatter(-1, -1, marker="s", alpha=0.8,
               c="#1ba17a",
               s=ss,
               label=("$\\alpha_{\\mathrm{2D}}^{\\parallel}$ Ref.XXX"))

    ax.scatter(-1, -1, marker="^", alpha=0.5,
               c="#1ba17a",
               s=ss,
               label=("$\\alpha_{\\mathrm{2D}}^{\\parallel}$ Ref.YYY"))
    
    ax.scatter(-1, -1, marker="s", alpha=0.5,
               c="#cca384",
               s=ss,
               label=("$\\alpha_{\\mathrm{2D}}^{\\perp}$ Ref.XXX"))

    ax.scatter(-1, -1, marker="^", alpha=0.8,
               c="#cca384",
               s=ss,
               label=("$\\alpha_{\\mathrm{2D}}^{\\perp}$ Ref.YYY"))

    # Legend outside
    ax.legend(loc="center left",
              bbox_to_anchor=(1.02, 0, 0.2, 1.0),
              bbox_transform=ax.transAxes)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    

def replace_cite(f):
    s_all = ""
    with open(f, "r") as fo:
        s_all = fo.read()
    s_all = s_all.replace("XXX", "~\\cite{Haastrup_2018_database}")
    s_all = s_all.replace("YYY", "~\\cite{Mounet_2018_database}")
    with open(f, "w") as fw:
        fw.write(s_all)


def plot_main():
    h = 0.80
    fig, ax = gridplots(2, 2, r=1, ratio=(2 - h) * 1.1,
                        span=[(0, 0, 1, 2),
                              (1, 0, 1, 1),
                              (1, 1, 1, 1)],
                        gridspec_kw=dict(height_ratios=(h,
                                                        2 - h)))
    plot_a(fig, ax[0])
    plot_bc(fig, ax[1:])
    add_legend_right(fig, ax[2])
    grid_labels(fig, ax, offsets=[(0, 0),
                                  (0, -0.05),
                                  (-0.05, -0.05)])

    savepgf(fig, img_path / "fig-universal.pgf", preview=True)
    replace_cite(img_path / "fig-universal.pgf")


if __name__ == '__main__':
    plot_main()
