import ase.db
import warnings
import numpy
import matplotlib.pyplot as plt
from ase.data import covalent_radii
from ase.calculators.vasp import Vasp
from ase.atoms import Atoms
from scipy.stats import linregress
import os, os.path
import shutil
from scipy.constants import pi, epsilon_0

db_file = "../../data/gpaw_data/c2db.db"
if not os.path.exists(db_file):
    raise FileExistsError(("Please download the c2db data into ../../data/gpaw_data/ folder,"
                   "from https://cmr.fysik.dtu.dk/_downloads/c2db.db"))


db = ase.db.connect(db_file)

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


root_dir = "../../VASP/"
curr_dir = os.path.dirname(os.path.abspath(__file__))
for mol in candidates:
    if "Cr" in mol.formula:     # CrS2 stuffs are not correct?
        continue
    thick, *_ = get_thick(mol)
    start_guess = thick + 0.25            #Start guess of thickness
    tag = "{0}-{1}".format(mol.formula, mol.prototype)
    print(tag)
    togo = True
    for attrib in ("alphax", "alphaz"):
        if not hasattr(mol, attrib):
            warnings.warn("{0} doesn't have attribute {1}!".format(mol.formula,
                                                                   attrib))
            togo = False
    if togo is not True:
        warnings.warn("{0} not calculated!".format(mol.formula))
        continue
    base_dir = os.path.join(root_dir, tag)
    if os.path.exists(base_dir) is not True:
        os.makedirs(base_dir)
    os.chdir(base_dir)
    calc = Vasp(restart=None,
                xc="vdw-df2",
                encut=800,
                kpts={"gamma":True,
                      "density":5.0},
                setups="gw",
                ismear=0,
                sigma=0.001,
                ediff=1e-7,
                algo="Accurate")
    atoms = Atoms(symbols=mol.formula,
                  positions=mol.positions,
                  cell=mol.cell,
                  pbc=[True, ] * 3)
    atoms.cell[-1, -1] = start_guess
    atoms.center(axis=2)                  #center along z

    calc.set_atoms(atoms)
    calc.initialize(atoms)
    calc.clean()
    calc.write_input(atoms)
    # Done with the input
os.chdir(curr_dir)    
