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
Eg_PBE_direct=[]
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
                   "gap_gw","dir_gap","dir_gap_hse", "dir_gap_gw",
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
    Eg_HSE.append((mol.gap_hse, mol.dir_gap_hse))
    Eg_GW.append((mol.gap_gw, mol.dir_gap_gw))
    Eg_PBE.append((mol.gap, mol.dir_gap))
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


# x-direction
plt.figure(figsize=(3.5, 3.5))
for i, data in enumerate((Eg_HSE, Eg_PBE, Eg_GW)):
    plt.cla()
    candidates = ("HSE", "PBE", "GW")
    l1, *_ = plt.plot(data[:, 0], 1 / (alpha_x), "o", alpha=0.4,
                      label="{} Minimal Gap".format(candidates[i]),
                      color="red")
    l2, *_ = plt.plot(data[:, 1], 1 / (alpha_x), "o", alpha=0.4,
                      label="{} Direct Gap".format(candidates[i]),
                      color="green")
    xx = numpy.linspace(0.3, 10)
    k, b, r, *_ = linregress(x=data[:, 0], y=1/alpha_x)
    print(k, b, r)
    yy = k * xx + b
    plt.plot(xx, yy, "--", color=l1.get_c())
    k, b, r, *_ = linregress(x=data[:, 1], y=1/alpha_x)
    print(k, b, r)
    yy = k * xx + b
    plt.plot(xx, yy, "--", color=l2.get_c())
    plt.xlabel("$E_{\\rm{g}}$ (eV)")
    plt.title(candidates[i])
    plt.ylabel("$(4 \\pi \\varepsilon_0)/\\alpha^{\\parallel}_{\\rm{2D}}$ ($\\AA^{-1}$)")
    plt.legend()
    plt.savefig(os.path.join(img_path, "gpaw_alphaxx_all_Eg_{}.svg").format(candidates[i]))





