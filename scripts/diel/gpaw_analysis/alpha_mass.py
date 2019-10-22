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
emass = []
hmass = []

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
    for attrib in ("emass1", "hmass1",
                   "alphax", "alphaz",
    ):
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
    if hasattr(mol, "emass2"):
        em = (mol.emass1 + mol.emass2) / 2
    else:
        em = mol.emass1
    emass.append(em)
    if hasattr(mol, "hmass2"):
        hm = (mol.hmass1 + mol.hmass2) / 2
    else:
        hm = mol.hmass1
    hmass.append(hm)

print(len(alpha_x))
alpha_x = numpy.array(alpha_x)
alpha_z = numpy.array(alpha_z)
emass = numpy.array(emass)
hmass = numpy.array(hmass)

img_path = "../../tmp_img/"
plt.style.use("science")


# x-direction
plt.figure(figsize=(3.5, 3.5))
plt.xlim(0.05, 1.5)

l2, *_ = plt.plot(emass, alpha_x, "o", alpha=0.4)
plt.xlabel("$m_{e}^{*}/m_0$ ")
plt.ylabel("$\\alpha^{\parallel}_{\\rm{2D}}/(4 \\pi \\varepsilon_0)$ ($\\AA$)")
plt.savefig(os.path.join(img_path, "gpaw_alphaxx_emass.svg"))

plt.figure(figsize=(3.5, 3.5))
plt.xlim(0.05, 1.5)
l2, *_ = plt.plot(hmass, alpha_x, "o", alpha=0.4)
plt.xlabel("$m_{h}^{*}/m_0$ ")
plt.ylabel("$\\alpha^{\parallel}_{\\rm{2D}}/(4 \\pi \\varepsilon_0)$ ($\\AA$)")
plt.savefig(os.path.join(img_path, "gpaw_alphaxx_hmass.svg"))

plt.figure(figsize=(3.5, 3.5))
plt.xlim(0.05, 1.5)
l2, *_ = plt.plot(emass, alpha_z, "o", alpha=0.4)
plt.xlabel("$m_{e}^{*}/m_0$ ")
plt.ylabel("$\\alpha^{\perp}_{\\rm{2D}}/(4 \\pi \\varepsilon_0)$ ($\\AA$)")
plt.savefig(os.path.join(img_path, "gpaw_alphazz_emass.svg"))

plt.figure(figsize=(3.5, 3.5))
plt.xlim(0.05, 1.5)
l2, *_ = plt.plot(hmass, alpha_z, "o", alpha=0.4)
plt.xlabel("$m_{h}^{*}/m_0$ ")
plt.ylabel("$\\alpha^{\perp}_{\\rm{2D}}/(4 \\pi \\varepsilon_0)$ ($\\AA$)")
plt.savefig(os.path.join(img_path, "gpaw_alphazz_hmass.svg"))





