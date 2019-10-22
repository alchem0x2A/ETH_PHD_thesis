import ase.db
import warnings
import numpy
import matplotlib.pyplot as plt
from ase.data import covalent_radii
from scipy.stats import linregress
import os, os.path
from scipy.constants import pi, epsilon_0

db_file = "../../data/gpaw_data/c2db.db"
if not os.path.exists(db_file):
    raise FileExistsError(("Please download the c2db data into ../../data/gpaw_data/ folder,"
                   "from https://cmr.fysik.dtu.dk/_downloads/c2db.db"))


subset = [l.strip() for l in open("list.txt").readlines() if l != "\n"]

db = ase.db.connect(db_file)

materials = []
alpha_x = []
alpha_z = []
Eg_HSE = []
Eg_GW = []
Eg_PBE = []
thick = []

def get_thick(atom_row):
    pos = atom_row.positions[:, -1]
    diff = covalent_radii[atom_row.numbers]
    zmax = numpy.max(pos + diff) - numpy.min(pos - diff)
    return zmax

candidates = db.select(selection="gap_gw>0.5")

for mol in candidates:
    if "Cr" in mol.formula:     # CrS2 stuffs are not correct?
        continue
    if mol.formula not in subset:
        continue
    print("{0}-{1}".format(mol.formula, mol.prototype))
    togo = True
    for attrib in ("gap", "gap_hse",
                   "gap_gw", "alphax", "alphaz"):
        if not hasattr(mol, attrib):
            warnings.warn("{0} doesn't have attribute {1}!".format(mol.formula,
                                                                   attrib))
            togo = False
    if togo is not True:
        warnings.warn("{0} not calculated!".format(mol.formula))
        continue
    materials.append("{0}-{1}".format(mol.formula, mol.prototype))
    alpha_x.append(mol.alphax)
    alpha_z.append(mol.alphaz)
    Eg_HSE.append(mol.gap_hse)
    Eg_GW.append(mol.gap_gw)
    Eg_PBE.append(mol.gap)
    thick.append(get_thick(mol))

print(len(alpha_x))
alpha_x = numpy.array(alpha_x)
alpha_z = numpy.array(alpha_z)
Eg_HSE = numpy.array(Eg_HSE)
Eg_GW = numpy.array(Eg_GW)
Eg_PBE = numpy.array(Eg_PBE)
thick = numpy.array(thick)

img_path = "../../tmp_img/"
plt.style.use("science")

plt.figure(figsize=(7, 3.5))
plt.subplot(121)
plt.plot(Eg_HSE, alpha_x * 4 * pi, "o", alpha=0.5)
plt.xlabel("$E_{\\rm{g}}$ (eV)")
plt.ylabel("$\\alpha_{xx} / \\epsilon_0$ ($\\AA$)")

plt.subplot(122)
plt.plot(Eg_HSE, alpha_z * 4 * pi, "o", alpha=0.5)
plt.xlabel("$E_{\\rm{g}}$ (eV)")
plt.ylabel("$\\alpha_{zz} / \\epsilon_0$ ($\\AA$)")

plt.tight_layout()
plt.savefig(os.path.join(img_path, "alpha_Eg_subset.svg"))

# x-direction
plt.figure(figsize=(3.5, 3.5))
plt.plot(Eg_HSE, 1 / alpha_x, "o", alpha=0.5)
res = linregress(x=Eg_HSE, y=1 / alpha_x)
xx = numpy.linspace(min(Eg_HSE), max(Eg_HSE))
yy = res.slope * xx + res.intercept
plt.plot(xx, yy, "--")
plt.text(x=0.8, y=4, s="$y={0:.2f}x+{1:.2f},\ R^2={2:.2f}$".format(res.slope, res.intercept, res.rvalue))
plt.xlabel("$1/E_{\\rm{g}}$ (1/eV)")
plt.ylabel("$\\alpha_{xx} / \\epsilon_0$ ($\\AA$)")
plt.savefig(os.path.join(img_path, "alpha_xx_1_Eg_subset.svg"))

# z-direction
plt.figure(figsize=(3.5, 3.5))
plt.plot(thick, alpha_z * 4 * pi, "o", alpha=0.5)
res = linregress(x=thick, y=alpha_z * 4 * pi)
xx = numpy.linspace(min(thick), max(thick))
yy = res.slope * xx + res.intercept
plt.plot(xx, yy, "--")
plt.text(x=2.5, y=0.40, s="$y={0:.2f}x+{1:.2f},\ R^2={2:.2f}$".format(res.slope, res.intercept, res.rvalue))
plt.xlabel("Thickness ($\\AA$)")
plt.ylabel("$\\alpha_{zz} / \\epsilon_0$ ($\\AA$)")
plt.savefig(os.path.join(img_path, "alpha_zz_thick_subset.svg"))
