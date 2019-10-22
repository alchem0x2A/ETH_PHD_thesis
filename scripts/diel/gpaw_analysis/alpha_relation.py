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


db = ase.db.connect(db_file)

materials = []
alpha_x = []
alpha_z = []
Eg_HSE = []
Eg_GW = []
Eg_PBE = []
thick = []
n_2D = []
polar = []

valence = numpy.load("../post_processing/valence.npy")
pol = numpy.load("../post_processing/valence.npy")

def get_thick(atom_row):
    pos = atom_row.positions[:, -1]
    diff = covalent_radii[atom_row.numbers]
    zmax = numpy.max(pos + diff) - numpy.min(pos - diff)
    vals = valence[atom_row.numbers]  # valence electrons
    atom_pol = pol[atom_row.numbers]
    A = atom_row.cell_area
    return zmax, sum(vals) / A, sum(atom_pol) / A

candidates = db.select(selection="gap_gw>0.5")


for mol in candidates:
    if "Cr" in mol.formula:     # CrS2 stuffs are not correct?
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
    delta, n, apol = get_thick(mol)
    thick.append(delta)
    n_2D.append(n)
    polar.append(apol)

print(len(alpha_x))
alpha_x = numpy.array(alpha_x)
alpha_z = numpy.array(alpha_z)
Eg_HSE = numpy.array(Eg_HSE)
Eg_GW = numpy.array(Eg_GW)
Eg_PBE = numpy.array(Eg_PBE)
thick = numpy.array(thick)
n_2D = numpy.array(n_2D)
polar = numpy.array(polar)

img_path = "../../tmp_img/"
plt.style.use("science")

# plt.figure(figsize=(7, 3.5))                           #
# plt.subplot(121)                                       #
# plt.plot(Eg_HSE, alpha_x * 4 * pi, "o", alpha=0.5)     #
# plt.xlabel("$E_{\\rm{g}}$ (eV)")                       #
# plt.ylabel("$\\alpha_{xx}/\\varepsilon_0$ ($\\AA$)")   #
#                                                        #
# plt.subplot(122)                                       #
# plt.plot(Eg_HSE, alpha_z * 4 * pi, "o", alpha=0.5)     #
# plt.xlabel("$E_{\\rm{g}}$ (eV)")                       #
# plt.ylabel("$\\alpha_{zz} / \\varepsilon_0$ ($\\AA$)") #
#                                                        #
# plt.tight_layout()                                     #
# plt.savefig(os.path.join(img                           #
                         # _path, "alpha_Eg_original.svg"))

# x-direction
plt.figure(figsize=(3.5, 3.5))
plt.plot(Eg_HSE, 1 / (alpha_x), "o", alpha=0.5)
k, b, r, *_ = linregress(x=Eg_HSE, y=1/alpha_x)
print(k, b, r)
xx = numpy.linspace(0.3, 8)
yy = k * xx + b
plt.plot(xx, yy, "--")
plt.xlabel("$E_{\\rm{g}}$ (eV)")
plt.ylabel("$(4 \\pi \\varepsilon_0)/\\alpha^{\parallel}_{\\rm{2D}}$ ($\\AA^{-1}$)")
plt.savefig(os.path.join(img_path, "alpha_xx_1_Eg.svg"))

# z-direction
plt.figure(figsize=(3.5, 3.5))
plt.plot(thick, alpha_z, "o", alpha=0.5)
# plt.plot(polar, alpha_z, "o", alpha=0.5)
k, b, r, *_ = linregress(x=thick, y=alpha_z)
print(k, b, r)
xx = numpy.linspace(2, 10)
yy = k * xx + b
# yyy = 1 / (4 * pi) * xx - 0.05
plt.plot(xx, yy, "--")
# plt.plot(xx, yyy, "--")
plt.xlabel("Thickness ($\\AA$)")
plt.ylabel("$\\alpha^{\\perp}_{\\rm{2D}} / (4 \pi \\varepsilon_0)$ ($\\AA$)")
plt.savefig(os.path.join(img_path, "alpha_zz_thick.svg"))

#x-direction
plt.figure(figsize=(3.5, 3.5))
# plt.plot(thick, alpha_z, "o", alpha=0.5)
plt.plot(polar, alpha_z, "o", alpha=0.5)
# k, b, r, *_ = linregress(x=thick, y=alpha_z)
# print(k, b, r)
# xx = numpy.linspace(2, 10)
# yy = k * xx + b
# yyy = 1 / (4 * pi) * xx - 0.05
# plt.plot(xx, yy, "--")
# plt.plot(xx, yyy, "--")
plt.text(x=2, y=10, s="$\\alpha^{\\perp}_{\\rm{2D}} = \\frac{\\hbar^2 e^2 \\rho_e}{m_e E_{\mathrm{g}}^2}$")
plt.xlabel("Total Atomic Polarizability per Area (Bohr$^3$)")
plt.ylabel("$\\alpha^{\\perp}_{\\rm{2D}} / (4 \pi \\varepsilon_0)$ ($\\AA$)")
plt.savefig(os.path.join(img_path, "alpha_zz_polar.svg"))

# z-direction with atomic polarizability
plt.figure(figsize=(3.5, 3.5))
# plt.plot(thick, alpha_z, "o", alpha=0.5)
plt.plot(polar, alpha_x, "o", alpha=0.5)
k, b, r, *_ = linregress(x=thick, y=alpha_z)
print(k, b, r)
# xx = numpy.linspace(2, 10)
# yy = k * xx + b
# yyy = 1 / (4 * pi) * xx - 0.05
# plt.plot(xx, yy, "--")
# plt.plot(xx, yyy, "--")
plt.xlabel("Total Atomic Polarizability per Area (Bohr$^3$)")
plt.ylabel("$\\alpha^{\\parallel}_{\\rm{2D}} / (4 \pi \\varepsilon_0)$ ($\\AA$)")
plt.savefig(os.path.join(img_path, "alpha_xx_polar.svg"))



