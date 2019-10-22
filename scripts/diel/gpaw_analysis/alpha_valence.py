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
omega_Eg = []
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
    for attrib in ("gap_hse", "emass1",
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
    em = mol.emass1
    zmax, ne, _ = get_thick(mol)
    if em > 0.1:
        omega_Eg.append(ne /  mol.gap_gw ** 2)
        alpha_x.append(mol.alphax)
        alpha_z.append(mol.alphaz)
    # omega_Eg.append(ne)

print(len(alpha_x))
alpha_x = numpy.array(alpha_x)
alpha_z = numpy.array(alpha_z)
omega_Eg = numpy.array(omega_Eg)

img_path = "../../tmp_img/"
plt.style.use("science")


# x-direction
plt.figure(figsize=(3.5, 3.5))

l2, *_ = plt.plot(omega_Eg, alpha_x, "o", alpha=0.4)
k, b, r, *_ = linregress(omega_Eg, alpha_x)
print(k, b, r)
xx = numpy.linspace(min(omega_Eg), max(omega_Eg))
plt.plot(xx, 10.97 * xx, "--")
plt.xlim(0, 5)
plt.ylim(0, 20)
plt.text(x=2, y=10, s="$\\alpha^{\\parallel}_{\\rm{2D}}=\\frac{\\hbar^2 e^2 \\sigma_{\\mathrm{2D}}^{\\mathrm{V}}}{m_e E_{\mathrm{g}}^2}$")
plt.xlabel("$\\sigma^{\mathrm{V}}_{\mathrm{2D}}/E_{\mathrm{g}}^{2}$ ($\\AA^{-2}$∙eV$^{-2}$) ")
plt.ylabel("$\\alpha^{\parallel}_{\\rm{2D}}/(4 \\pi \\varepsilon_0)$ ($\\AA$)")
plt.savefig(os.path.join(img_path, "gpaw_alphaxx_omega_Eg.svg"))
