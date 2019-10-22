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

candidates = db.select(selection="bse_binding>0,dir_gap_gw>0,alphax>0")

materials = []
alpha_x = []
E_opt = []

for mol in candidates:
    if "Cr" in mol.formula:     # CrS2 stuffs are not correct?
        continue
    print("{0}-{1}".format(mol.formula, mol.prototype))
    materials.append("{0}-{1}".format(mol.formula, mol.prototype))
    alpha_x.append((mol.alphax + mol.alphay) / 2)
    E_opt.append(mol.dir_gap_gw - mol.bse_binding)

alpha_x = numpy.array(alpha_x)
E_opt = numpy.array(E_opt)

img_path = "../../tmp_img/"
plt.style.use("science")



# x-direction
plt.figure(figsize=(3.5, 3.5))
plt.plot(E_opt, 1 / (alpha_x), "o", alpha=0.5)
k, b, r, *_ = linregress(x=E_opt, y=1 / alpha_x)
print(k, b, r)
xx = numpy.linspace(0.0, 8)
yy = k * xx + b
plt.xlim(0, 8)
plt.plot(xx, yy, "--")
plt.xlabel("$E_{\\rm{g}}^{\\rm{opt}}$ (eV)")
plt.ylabel("$(4 \\pi \\varepsilon_0)/\\alpha^{\parallel}_{\\rm{2D}}$ ($\\AA^{-1}$)")
plt.savefig(os.path.join(img_path, "alpha_x_E_opt.svg"))

