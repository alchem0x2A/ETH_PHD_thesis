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

db_file = "../../data/2D-bulk/bulk.db"


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
    zmax = numpy.max(pos + diff) - numpy.min(pos - diff)
    return zmax

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

reader = csv.reader(open("../../data/HSE-data/2D_HSE.csv", encoding="utf8"))
next(reader)                    # skip line1
for row in reader:
    if row[4] != "":
        name, proto = row[: 2]
        print(name, proto)
        L, E, ex, ey, ez, *_ = map(float, row[2:])
        if ez < ex:
            e_xy = numpy.sqrt(ex * ey)
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
                # ez_simu = 4 * pi * az / L_3D
                print(name, proto, ex_3D, ex_simu)
                eps_x_3D.append((ex_3D, ex_simu))
                eps_z_3D.append((ez_3D, ez_simu))
                Eg_HSE.append(E)
            # mol = list(db.select(formula=name, prototype=proto))[0]
            # thick.append(get_thick(mol))

alpha_x = numpy.array(alpha_x)
alpha_z = numpy.array(alpha_z)
eps_x_3D = numpy.array(eps_x_3D)
eps_z_3D = numpy.array(eps_z_3D)
Eg_HSE = numpy.array(Eg_HSE)
# thick = numpy.array(thick)

eps_x_gpaw = []
eps_z_gpaw = []
alpha_z_gpaw = []
Eg_gpaw = []
L_gpaw = []
for db_id in range(1, db.count() + 1):  # db index starts with 1
    mol = db.get(db_id)
    # if any(hasattr(mol, key) is False for key in ["alphax", "alphay", "alphaz",
                                         # "bulk_L", "bulk_eps_x", "bulk_eps_y",
                                         # "bulk_eps_z"]):
        # continue
    # if mol.bulk_calculated is False:
        # continue
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
eps_x_gpaw = numpy.array(eps_x_gpaw)
eps_z_gpaw = numpy.array(eps_z_gpaw)
alpha_z_gpaw = numpy.array(alpha_z_gpaw)
Eg_gpaw = numpy.array(Eg_gpaw)
L_gpaw = numpy.array(L_gpaw)
print(Eg_HSE.shape)
print(eps_x_3D.shape, eps_z_3D.shape)

print(Eg_gpaw.shape)
print(eps_x_gpaw.shape, eps_z_gpaw.shape)

# cond = numpy.where(Eg_HSE > 0.6)
# Eg_HSE = Eg_HSE[cond]
# alpha_x = alpha_x[cond]
# alpha_z = alpha_z[cond]
# thick = thick[cond]
if __name__ == "__main__":

    img_path = "../../tmp_img/"
    plt.style.use("science")


    def fit_func(x, a,b):
        return a+b / x

    upper = 25
    cond = numpy.where(eps_x_gpaw[:, 0] < upper)
    cond2 = numpy.where(eps_x_3D[:, 0] < upper)

    # x-direction
    # MAE=numpy.mean(numpy.abs(eps_x_3D[:, 1][eps_x_3D[:, 0] < 30] - eps_x_3D[:, 0][eps_x_3D[:, 0]<30]))
    # print(MAE)
    # MAE=numpy.mean(numpy.abs(eps_x_3D[:, 1][eps_x_3D[:, 0] < 30] - eps_x_3D[:, 0][eps_x_3D[:, 0]<30]))
    # print(MAE)
    plt.figure(figsize=(2.8, 2.5))
    res = linregress(eps_x_gpaw[:, 1][cond], eps_x_gpaw[:, 0][cond])
    print(res)
    res2 = linregress(eps_x_3D[:, 1][cond2], eps_x_3D[:, 0][cond2])
    print(res2)
    xx = numpy.linspace(0, 30)
    yy = res.slope * xx + res.intercept
    plt.plot(xx, yy, "-.")
    plt.plot(eps_x_gpaw[:, 1][cond], eps_x_gpaw[:, 0][cond], "s",
             alpha=0.2, color="grey")

    plt.scatter(eps_x_3D[:, 1][cond2], eps_x_3D[:, 0][cond2],
                c=Eg_HSE[cond2],
                alpha=0.5,
                cmap="jet")
    print()
    plt.plot(numpy.linspace(1, upper), numpy.linspace(1, upper), "--")
    plt.xlim(1, upper)
    plt.ylim(1, upper)
    cb = plt.colorbar()
    cb.ax.set_title("$E_{\mathrm{g}}$ (eV)")
    plt.ylabel("eps bulk xx from DFT")
    plt.xlabel("eps bulk xx from alpha")
    plt.tight_layout()
    plt.savefig(os.path.join(img_path, "eps_3D_xx.svg"))

    # z-direction
    upper = 10
    cond = numpy.where((eps_z_gpaw[:, 0] < upper) & (eps_z_gpaw[:, 1] > 0) & (eps_z_gpaw[:, 1] < upper) )
    cond2 = numpy.where((eps_z_3D[:, 0] < upper) & (0 < eps_z_3D[:, 1]) & (eps_z_3D[:, 1] < upper))
    plt.figure(figsize=(2.8, 2.5))
    res = linregress(eps_z_gpaw[:, 1][cond], eps_z_gpaw[:, 0][cond])
    print(res)
    res2 = linregress(eps_z_3D[:, 1][cond2], eps_z_3D[:, 0][cond2])
    print(res2)
    xx = numpy.linspace(1, upper)
    yy = res.slope * xx + res.intercept

    ez = eps_z_gpaw[cond]; az = alpha_z_gpaw[cond]; ll = L_gpaw[cond]

    def model(X, A, B, C):
        a = X[:, 0]; L = X[:, 1]
        return A - B / (1 + 4 * pi * a / L - C)

    from scipy.optimize import curve_fit
    # xx = numpy.hstack()
    # res = curve_fit(model, az)
    plt.plot(xx, yy, "-.")
    plt.plot(eps_z_gpaw[:, 1][cond], eps_z_gpaw[:, 0][cond],
             "s",
             alpha=0.2, color="brown")
    plt.scatter(eps_z_3D[:, 1][cond2],
                eps_z_3D[:, 0][cond2], alpha=0.5,
                c=Eg_HSE[cond2],
                cmap="jet"
    )

    plt.plot(numpy.linspace(1, upper), numpy.linspace(1, upper), "--")
    plt.ylim(1, upper)
    # plt.xlim(1, 2)
    plt.xlim(1, upper)
    cb = plt.colorbar()
    cb.ax.set_title("$E_{\mathrm{g}}$ (eV)")
    plt.ylabel("eps bulk zz from DFT")
    plt.xlabel("eps bulk zz from alpha")
    plt.tight_layout()
    plt.savefig(os.path.join(img_path, "eps_3D_zz.svg"))
