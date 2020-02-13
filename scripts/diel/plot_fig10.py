import numpy as np
from . import data_path, img_path
from helper import gridplots, grid_labels, savepgf
from helper import get_color, add_cbar, add_img_ax
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import ase.db
import numpy
import matplotlib.pyplot as plt
from ase.data import covalent_radii
from .gpaw_data import get_data
import csv

"""
Extract the alpha from the HSE xlsx files
"""

def get_2D3D():
    aniso_data = data_path / "other_dimension" / "2D3D.npz"
    if not aniso_data.exists():  # then need to create
        db_file = data_path / "gpaw_data" / "c2db_small.db"
        bulk_file = data_path / "2D-bulk/bulk.db"
        if not db_file.exists():
            raise FileExistsError(("Please download the c2db data into data/gpaw_data/ folder,"
                                   "from https://cmr.fysik.dtu.dk/_downloads/c2db.db"))

        def get_bulk(name, proto, id=None, method="gpaw"):
            # Get bulk properties
            if id is None:
                res = list(db.select(formula=name, prototype=proto))
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
                    if r.bulk_gap_vasp < 0:
                        r = r.gap_hse
                    else:
                        r = r.bulk_gap_vasep
                else:
                    return None
                if eps_para < 0 or eps_perp < 0:
                    return None
            except Exception:
                return None
            return L, eps_para, eps_perp, e

        db = ase.db.connect(db_file)
        bulk_db = ase.db.connect(bulk_file)

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

        reader = csv.reader(open(data_path / "HSE-data" / "2D_HSE.csv",
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
                    ax = (e_xy - 1) / (4 * np.pi) * L
                    az = (1 - 1/ez) * L / (4 * np.pi)
                    ax = max(1 / 1.2, ax)
                    eps_x.append(e_xy); eps_z.append(ez)
                    alpha_x.append(ax); alpha_z.append(az)
                    Eg_HSE.append(E)
                    mol = list(db.select(formula=name, prototype=proto))[0]
                    thick.append(get_thick(mol))

        print(len(alpha_x))
        alpha_x = np.array(alpha_x)
        alpha_z = np.array(alpha_z)
        Eg_HSE = np.array(Eg_HSE)
        thick = np.array(thick)

        cond = np.where(Eg_HSE > 0.6)
        Eg_HSE = Eg_HSE[cond]
        alpha_x = alpha_x[cond]
        alpha_z = alpha_z[cond]
        thick = thick[cond]

        gp_data = get_data()

        import relation_2D3D as bulk

        # cnt_eg = np.sqrt(cnt_r * B / (cnt_x - A))
        Eg_2D = np.append(gp_data[2], Eg_HSE)
        Eg_3D = np.append(bulk.Eg_gpaw, bulk.Eg_HSE)
        eta_2D = np.append(gp_data[1] / gp_data[0],  alpha_z / alpha_x)
        print(len(eta_2D))
        eta_3D = np.append(np.min([bulk.eps_z_gpaw[:, 0] / bulk.eps_x_gpaw[:, 0],
                                   bulk.eps_x_gpaw[:, 0] / bulk.eps_z_gpaw[:, 0]],
                                  axis=0),
                           np.min([bulk.eps_z_3D[:, 0] / bulk.eps_x_3D[:, 0],
                                   bulk.eps_x_3D[:, 0] / bulk.eps_z_3D[:, 0]],
                                  axis=0), )
        np.savez(aniso_data,
                    **{"Eg_2D": Eg_2D, "Eg_3D": Eg_3D,
                     "eta_2D": eta_2D, "eta_3D": eta_3D})
    else:
        d = np.load(aniso_data)
        Eg_2D = d["Eg_2D"]; Eg_3D = d["Eg_3D"]
        eta_2D = d["eta_2D"]; eta_3D = d["eta_3D"]
    return Eg_2D, Eg_3D, eta_2D, eta_3D



def plot_ax(fig, ax):
    Eg_2D, Eg_3D, eta_2D, eta_3D = get_2D3D()
    ax.scatter(Eg_2D, eta_2D, marker="^", alpha=0.1, s=40,
               linewidth=0)
    ax.scatter(Eg_3D, eta_3D, marker="s", alpha=0.1, s=40,
               linewidth=0)

    def anisotropy(data):
        a_max = np.max(data, axis=1)
        a_min = np.min(data, axis=1)
        return a_min / a_max

    def anis_from_file(file_name):
        data = np.genfromtxt(file_name, delimiter=",",
                             comments="#")  # csv ending
        Eg = data[:, 1]
        anis = anisotropy(data[:, 2:5])
        return Eg, anis


    marks = {"CNT": "o",
             "polyacene": "p",
             "MPc": "<",
             "covalent": ">",
             "fullerene": "*",
             "polyacetylene": "D"}
    for f in ["CNT", "covalent", "polyacene", "MPc", "fullerene", "polyacetylene"]:
        f_name = data_path / "other_dimension" / "{}.csv".format(f)
        Eg, anis = anis_from_file(f_name)
        ax.scatter(Eg, anis, label=f,
                   marker=marks[f], s=49,
                   alpha=0.6,
                   linewidth=0)

    xx = yy = np.linspace(0, 8, 100)
    ax.plot(xx, np.ones_like(xx), "--")
    yy = 0.048 * xx + 0.087

    # ax.set_title("$y={0:.4f}x+{1:.4f},\ R^2={2:.4f}$".format(res.slope, res.intercept, res.rvalue))
    ax.set_xlabel("$E_{\mathrm{g}}$ (eV)")
    ax.set_ylabel("Dielectric Anisotropy $g$")
    ax.plot(xx, yy, color="k", alpha=0.5)
    ax.fill_between(xx, y1=yy, y2=np.ones_like(xx),
                    linewidth=0,
                    color="orange", alpha=0.1, zorder=0)
    ax.fill_between(xx, y1=np.zeros_like(xx), y2=yy,
                    linewidth=0,
                    color="green", alpha=0.1, zorder=0)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1.05)
    # ax.set_ylabel("$\\alpha_{zz}/(4\\pi \\varepsilon_0)$ ($\\AA$)")

def plot_legend(fig, ax):
    ax.set_axis_off()
    ax_img = inset_axes(ax, width="100%", height="100%",
                        bbox_to_anchor=(-0.1, -0.05, 1.1, 1.1),
                        bbox_transform=ax.transAxes)
    add_img_ax(ax_img, img_path / "sub_img" / "dimensions.png")
    w, h = 300, 955
    texts = ["Bulk Covalent Materials",
             "Bulk vdW Materials",
             "2D Materials",
             "Planar OSc",
             "CNT",
             "Linear Polyacene",
             "Polyacetylene",
             "Fullerenes"]
    for i, t in enumerate(texts[::-1]):
        x = 1.02
        ymin = 42
        ymax = 881
        y = np.linspace(ymin, ymax, len(texts))[i] / h
        ax_img.text(x=x, y=y, s=t, ha="left", va="center",
                    transform=ax_img.transAxes)
    for i, y in enumerate([42, 406, 645, 880]):
        x = 0.76
        y = y / h
        ax_img.text(x=x, y=y,
                    s="{0:d}D".format(i),
                    ha="right", va="center",
                    transform=ax_img.transAxes)


def plot_main():
    w = 1.4
    fig, ax = gridplots(1, 2, r=0.65,
                        ratio=2 / w,
                        gridspec_kw=dict(width_ratios=(w, 2 - w)))
    # grid_labels(fig, ax)
    plot_ax(fig, ax[0])
    plot_legend(fig, ax[1])
    savepgf(fig, img_path / "fig-aniso.pgf", preview=True)


if __name__ == '__main__':
    plot_main()
