import numpy as np
from . import data_path, img_path
from helper import gridplots, grid_labels, savepgf
from helper import get_color, add_cbar, add_img_ax
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import ase.db
from ase.io.trajectory import Trajectory
from gpaw import GPAW
import warnings
import numpy
import matplotlib.pyplot as plt
from ase.data import covalent_radii
from scipy.stats import linregress
from scipy.optimize import curve_fit
import os, os.path
from scipy.constants import pi, epsilon_0
import scipy
import csv

"""
Extract the alpha from the HSE xlsx files
"""

db_file = data_path / "2D-bulk" / "bulk.db"


db = ase.db.connect(db_file)

materials = []
eps_x = []
eps_z = []
alpha_x = []
alpha_z = []
eps_x_3D = []
eps_z_3D = []
Eg_HSE = []
thick = []

def get_thick(atom_row):
    # Get thickness from 2D material
    pos = atom_row.positions[:, -1]
    diff = covalent_radii[atom_row.numbers]
    zmax = np.max(pos + diff) - np.min(pos - diff)
    return zmax

def get_bulk(name, proto, id=None, method="gpaw"):
    # Get bulk properties
    if id is None:
        try:
            res = list(db.select(formula=name, prototype=proto))
        except KeyError:
            return None
        if len(res) == 0:
            return None
        r = res[0]
    else:
        r = db.get(id)
    try:
        if method.lower() == "gpaw":
            L = r.bulk_L
            eps_para = (r.bulk_eps_x + r.bulk_eps_y) / 2
            eps_perp = r.bulk_eps_z
            e = r.gap_hse
        # VASP version below:
        elif method.lower() == "vasp":
            L = r.bulk_L_vasp
            eps_para = (r.bulk_eps_x_vasp + r.bulk_eps_y_vasp) / 2
            eps_perp = r.bulk_eps_z_vasp
            if r.bulk_gap < 0:
                e = r.gap_hse
            else:
                e = r.bulk_gap
        else:
            return None
        if eps_para < 0 or eps_perp < 0:
            return None
    except Exception:
        return None
    return L, eps_para, eps_perp, e


reader = csv.reader(open(data_path / "HSE-data" / "2D_HSE.csv",
                         encoding="utf8"))
next(reader)                    # skip line1
for row in reader:
    if row[4] != "":
        name, proto = row[: 2]
        print(name, proto)
        L, E, ex, ey, ez, *_ = map(float, row[2:])
        if ez < ex:
            e_xy = np.sqrt(ex * ey)
            ax = (e_xy - 1) / (4 * pi) * L
            az = (1 - 1/ez) * L / (4 * pi)
            bulk_res = get_bulk(name, proto, method="vasp")
            if bulk_res is not None:
                materials.append("-".join((name, proto)))
                eps_x.append(e_xy); eps_z.append(ez)
                alpha_x.append(ax); alpha_z.append(az)
                L_3D, ex_3D, ez_3D, e = bulk_res
                print(L_3D, ex_3D, ez_3D)
                ex_simu = 1 + 4 * pi * ax / L_3D
                ez_simu = 1 / (1 - 4 * pi * az / L_3D)
                print(name, proto, ex_3D, ex_simu)
                eps_x_3D.append((ex_3D, ex_simu))
                eps_z_3D.append((ez_3D, ez_simu))
                Eg_HSE.append(E)


alpha_x = np.array(alpha_x)
alpha_z = np.array(alpha_z)
eps_x_3D = np.array(eps_x_3D)
eps_z_3D = np.array(eps_z_3D)
Eg_HSE = np.array(Eg_HSE)
# thick = np.array(thick)

eps_x_gpaw = []
eps_z_gpaw = []
alpha_z_gpaw = []
Eg_gpaw = []
L_gpaw = []
for db_id in range(1, db.count() + 1):  # db index starts with 1
    mol = db.get(db_id)
    try:
        ax = (mol.alphax +  mol.alphay) / 2
        az = mol.alphaz
        L, ex, ez, e = get_bulk(None, None, db_id, method="gpaw")
        ex_simu = 1 + 4 * pi * ax / L
        ez_simu = 1 / (1 - 4 * pi * az / L)
        # ez_simu = 4 * pi * az / L
        eps_x_gpaw.append((ex, ex_simu))
        eps_z_gpaw.append((ez, ez_simu))
        alpha_z_gpaw.append(az)
        L_gpaw.append(L)
        Eg_gpaw.append(e)
    except Exception:
        continue
eps_x_gpaw = np.array(eps_x_gpaw)
eps_z_gpaw = np.array(eps_z_gpaw)
alpha_z_gpaw = np.array(alpha_z_gpaw)
Eg_gpaw = np.array(Eg_gpaw)
L_gpaw = np.array(L_gpaw)
print(Eg_HSE.shape)
print(eps_x_3D.shape, eps_z_3D.shape)

print(Eg_gpaw.shape)
print(eps_x_gpaw.shape, eps_z_gpaw.shape)

def plot_ax(fig, ax):
    ax1, ax2 = ax
    def fit_func(x, a,b):
        return a+b / x

    upper = 25
    cond = np.where(eps_x_gpaw[:, 0] < upper)
    cond2 = np.where(eps_x_3D[:, 0] < upper)

    # x-direction
    res = linregress(eps_x_gpaw[:, 1][cond], eps_x_gpaw[:, 0][cond])
    print(res)
    res2 = linregress(eps_x_3D[:, 1][cond2], eps_x_3D[:, 0][cond2])
    xx = np.linspace(0, 30)
    yy = res.slope * xx + res.intercept
    ax1.plot(xx, yy, "-.")
    ax1.text(x=0.02, y=0.98,
             s="$y={0:.2f}x{1:+.2f}$ \n $R^{{2}}={2:.2f}".format(res.slope,
                                                                 res.intercept,
                                                                 res.rvalue),
             ha="left", va="top",
             transform=ax1.transAxes)
    ax1.plot(eps_x_gpaw[:, 1][cond], eps_x_gpaw[:, 0][cond], "s",
             markersize=8,
             alpha=0.2, color="grey")
    ss = 64
    ax1.scatter(eps_x_3D[:, 1][cond2], eps_x_3D[:, 0][cond2],
                c=Eg_HSE[cond2],
                s=ss,
                alpha=0.5,
                cmap="jet")
    ax1.plot(np.linspace(1, upper), np.linspace(1, upper), "-",
             linewidth=3, alpha=0.5)
    ax1.set_xlim(1, upper)
    ax1.set_ylim(1, upper)
    # cb = ax1.colorbar()
    # cb.ax.set_title("$E_{\mathrm{g}}$ (eV)")
    ax1.set_xlabel("Model-predicted $\\varepsilon_{\\mathrm{Bulk}}^{\\parallel}$")
    ax1.set_ylabel("DFT-calculated $\\varepsilon_{\\mathrm{Bulk}}^{\\parallel}$")

    # z-direction
    upper = 10
    cond = np.where((eps_z_gpaw[:, 0] < upper) & (eps_z_gpaw[:, 1] > 0) & (eps_z_gpaw[:, 1] < upper) )
    cond2 = np.where((eps_z_3D[:, 0] < upper) & (0 < eps_z_3D[:, 1]) & (eps_z_3D[:, 1] < upper))
    res = linregress(eps_z_gpaw[:, 1][cond], eps_z_gpaw[:, 0][cond])
    print(res)
    res2 = linregress(eps_z_3D[:, 1][cond2], eps_z_3D[:, 0][cond2])
    print(res2)
    xx = np.linspace(1, upper)
    yy = res.slope * xx + res.intercept
    ax2.text(x=0.98, y=0.3,
             s="$y={0:.2f}x{1:+.2f}$ \n $R^{{2}}={2:.2f}".format(res.slope,
                                                                 res.intercept,
                                                                 res.rvalue),
             ha="right", va="center",
             transform=ax2.transAxes)

    ez = eps_z_gpaw[cond]; az = alpha_z_gpaw[cond]; ll = L_gpaw[cond]

    def model(X, A, B, C):
        a = X[:, 0]; L = X[:, 1]
        return A - B / (1 + 4 * pi * a / L - C)


    ax2.plot(xx, yy, "-.")
    ax2.plot(np.linspace(1, upper), np.linspace(1, upper),
             "-", linewidth=3, alpha=0.5, label="$y=x$")
    ax2.plot(eps_z_gpaw[:, 1][cond], eps_z_gpaw[:, 0][cond],
             "s",
             markersize=8,
             alpha=0.2, color="brown", label="PBE")
    ax2.scatter(eps_z_3D[:, 1][cond2],
                eps_z_3D[:, 0][cond2], alpha=0.5,
                s=ss,
                c=Eg_HSE[cond2],
                cmap="jet",
                label="HSE"
    )

    ax2.set_ylim(1, upper)
    # plt.set_xlim(1, 2)
    ax2.set_xlim(1, upper)
    
    # cb = ax2.colorbar()
    # cb.ax.set_title("$E_{\mathrm{g}}$ (eV)")
    ax2.set_xlabel("Model-predicted $\\varepsilon_{\\mathrm{Bulk}}^{\\perp}$")
    ax2.set_ylabel("DFT-calculated $\\varepsilon_{\\mathrm{Bulk}}^{\\perp}$")
    ax2.legend(loc="lower right")

    ax_cb = inset_axes(ax2, width="100%", height="100%",
                       loc="lower left",
                       bbox_to_anchor=(1.05, 0.0, 0.06, 0.6),
                       bbox_transform=ax2.transAxes)
    cb = add_cbar(fig, None, min=Eg_HSE[cond2].min(),
             max=Eg_HSE[cond2].max(), cax=ax_cb)
    cb.ax.set_title("$E_{\\mathrm{g}}$ (eV)", ha="center", pad=7)

    ax_img = inset_axes(ax1, width="100%", height="100%",
                        loc="lower left",
                        bbox_to_anchor=(0.5, 0.02, 0.3, 0.3),
                        bbox_transform=ax1.transAxes)
    add_img_ax(ax_img, img_path / "sub_img" / "L_bulk.png", index=1)
    ax_img.text(x=1.01, y=90 / 256,
                s="$L_{\\mathrm{Bulk}}$",
                ha="left", va="center", size="small",
                transform=ax_img.transAxes)



def plot_main():
    fig, ax = gridplots(1, 2, r=0.95, ratio=2)
    plot_ax(fig, ax)
    grid_labels(fig, ax)

    savepgf(fig, img_path / "fig-2D-3D.pgf", preview=True)


if __name__ == '__main__':
    plot_main()
