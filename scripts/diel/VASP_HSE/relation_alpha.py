import ase.db
import warnings
import numpy
import matplotlib.pyplot as plt
from ase.data import covalent_radii
from ase.io import read, write
from scipy.stats import linregress
from scipy.optimize import curve_fit
import os, os.path
from scipy.constants import pi, epsilon_0
from gpaw_data import get_data
from epfl_data import get_data_epfl, get_data_epfl2
import scipy
import csv

"""
Extract the alpha from the HSE xlsx files
"""

db_file = "../../data/gpaw_data/c2db.db"
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
    zmax = numpy.max(pos + diff) - numpy.min(pos - diff)
    return zmax


# REad VASP result
reader = csv.reader(open("../../data/HSE-data/2D_HSE.csv", encoding="utf8"))
next(reader)                    # skip line1
for row in reader:
    if row[4] != "":
        name, proto = row[: 2]
        print(name, proto)
        L, E, ex, ey, ez, *_ = map(float, row[2:])
        if ez < ex:
            eps_z.append(ez)
            materials.append("-".join((name, proto)))
            e_xy = numpy.sqrt(ex * ey)
            ax = (e_xy - 1) / (4 * pi) * L
            az = (1 - 1/ez) * L / (4 * pi)
            ax = max(1 / 1.2, ax)
            eps_x.append(e_xy); eps_z.append(ez)
            alpha_x.append(ax); alpha_z.append(az)
            Eg_HSE.append(E)
            if proto == "ABX3":
                thick.append(8.0)
            else:
                mol = list(db.select(formula=name, prototype=proto))[0]
                thick.append(get_thick(mol))



print(len(alpha_x))
alpha_x = numpy.array(alpha_x)
alpha_z = numpy.array(alpha_z)
Eg_HSE = numpy.array(Eg_HSE)
thick = numpy.array(thick)

# cond = numpy.where(Eg_HSE > 0.6)
# materials = materials[cond]
# Eg_HSE = Eg_HSE[cond]
# alpha_x = alpha_x[cond]
# alpha_z = alpha_z[cond]
# thick = thick[cond]

colors = {"MoS2": "#AA0000", "CdI2": "#2C5AA0",
          "GaS": "#FFCC00", "BN": "#A05A2C",
          "P": "#447821", "CH": "#FF6600",
          "ABX3": "#6600FF"}

print(materials)
cs = [colors[mat.split("-")[-1]] for mat in materials]
print(cs)

img_path = "../../tmp_img/"
plt.style.use("science")

# plt.figure(figsize=(7, 3.5))
# plt.subplot(121)
# plt.scatter(Eg_HSE, alpha_x, marker="o", alpha=0.5, c=cs)
def fit_func(x, a,b):
    return b / x


exclude_eps = ["AgNO2", "Cd(OH)2", "Ca(OH)2", "SnF4",
               "Li2(OH)2", "Rb2Cl2", "LiBH4", "NaCN",
               "Mg(OH)2", "Na2(OH)2", "PbF4", "AgO4Cl", "Ag2I2"]
# exclude_eps = []
# materials_qe, *qe_data = get_data_epfl(exclude=exclude_eps)
materials_qe, *qe_data = get_data_epfl2(exclude=exclude_eps)
gp_data = get_data()
print(qe_data[0].shape)

# x-direction
fig = plt.figure(figsize=(2.5, 2.5))
ax = fig.add_subplot(111)
# gpaw
ax.scatter(gp_data[2], 1 / gp_data[0], marker="s", alpha=0.1,
           c="#7289af", label="$\\alpha_{\\mathrm{2D}}^{\\parallel}$ Ref. xx")

ax.scatter(qe_data[2], 1 / qe_data[0], marker="^", alpha=0.1,
           c="#7289af", label="$\\alpha_{\\mathrm{2D}}^{\\parallel}$ Ref. yy")

# [ax.text(x=qe_data[3][i], y=1 / qe_data[0][i], s=materials_qe[i],
         # size="x-small",
         # ha="left", va="top", alpha=0.5) for i in range(len(materials_qe)) if 1 / qe_data[0][i] > 0.55]
# hse
ax.scatter(Eg_HSE, 1 / (alpha_x), marker="o",
            edgecolors=None, alpha=0.5,
            c=cs)
res = linregress(x=Eg_HSE, y=1 / (alpha_x))
# print(res)
xx = numpy.linspace(0, 8.5)
yy = res.slope * xx + res.intercept
# for mat, x, y in zip(materials, Eg_HSE ** -p, alpha_x):
    # plt.text(x=x, y=y * 4 * pi, s=mat)
ax.plot(xx, yy, "--")
# ax.set_title("$y={0:.4f}x+{1:.4f},\ R^2={2:.4f}$".format(res.slope, res.intercept, res.rvalue))
ax.set_xlabel("$E_{\\rm{g}}$ (eV)")
ax.set_ylabel("$(4 \\pi \\varepsilon_0)/\\alpha_{\\mathrm{2D}}^{\\parallel}$ ($\\mathrm{\\AA}^{-1}$)")
ax.set_xlim(0, 8.5)
ax.set_ylim(numpy.min(yy), 1.5)
# ax.set_xticks([1,3,5,7])
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(img_path, "alpha_xx_1_Eg_HSE.svg"))

# z-direction
fig = plt.figure(figsize=(2.5, 2.5))
ax = fig.add_subplot(111)
# gpaw
ax.scatter(gp_data[3], gp_data[1] * 4 * pi, marker="s", alpha=0.1,
           c="#cca384", label="$\\alpha_{\\mathrm{2D}}^{\\perp}$ Ref. xx")

ax.scatter(qe_data[-1], qe_data[1] * 4 * pi, marker="^", alpha=0.1,
           c="#cca384", label="$\\alpha_{\\mathrm{2D}}^{\\perp}$ Ref. xx")
# [ax.text(x=qe_data[-1][i], y=qe_data[1][i], s=materials_qe[i],
         # ha="left", va="top", size="x-small", alpha=0.5) \
         # for i in range(len(materials_qe)) if qe_data[1][i] < 0.8 * qe_data[-1][i] * 0.078]
# hse
ax.scatter(thick, alpha_z * 4 * pi, marker="o",
            edgecolors=None,
            alpha=0.5, c=cs)
res = linregress(x=thick, y=alpha_z * 4 * pi)
# res = linregress(x=numpy.hstack([thick, gp_data[3], qe_data[-1]]),
                 # y=numpy.hstack([alpha_z, gp_data[1], qe_data[1]]))
# res = linregress(x=gp_data[3], y=gp_data[1])
print(res)
xx = numpy.linspace(1.5, 10.5)
yy = res.slope * xx + res.intercept
ax.plot(xx, yy, "--")
ax.plot(xx, xx, "--")
# plt.colorbar()
ax.set_xlim(1.5, 10.5)
# ax.set_ylim(0.08, numpy.max(yy))
# ax.set_title("$y={0:.2f}x+{1:.2f},\ R^2={2:.2f}$".format(res.slope, res.intercept, res.rvalue))
ax.set_xlabel("$\\delta_{\\mathrm{cov}}$ ($\\mathrm{\\AA}$)")
ax.set_ylabel("$\\alpha_{\\mathrm{2D}}^{\\perp} / (4 \\pi \\varepsilon_0)$ ($\\mathrm{\\AA}$)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(img_path, "alpha_zz_thick_HSE.svg"))

