import ase.db
import warnings
import numpy
import matplotlib.pyplot as plt
from ase.data import covalent_radii
from scipy.stats import linregress
from scipy.constants import pi, epsilon_0
from pathlib import Path
from . import data_path
import subprocess

db_file = data_path / "gpaw_data" / "c2db_small.db"
if not db_file.exists():
    raise FileExistsError(("Please download the c2db data into ../../data/gpaw_data/ folder,"
                   "from https://cmr.fysik.dtu.dk/_downloads/c2db.db"))


db = ase.db.connect(db_file.as_posix())
valence = numpy.load(data_path / "post_processing" / "atom_pol.npy")
pol = numpy.load(data_path / "post_processing" / "valence.npy")


def get_thick(atom_row):
    pos = atom_row.positions[:, -1]
    diff = covalent_radii[atom_row.numbers]
    zmax = numpy.max(pos + diff) - numpy.min(pos - diff)
    vals = valence[atom_row.numbers]  # valence electrons
    atom_pol = pol[atom_row.numbers]
    A = atom_row.cell_area
    return zmax, sum(vals) / A, sum(atom_pol) / A


def get_data():
    candidates = db.select(selection="gap_gw>0.5")
    candidates = db.select(selection="gap_gw>0.05")
    materials = []
    alpha_x = []
    alpha_z = []
    Eg_HSE = []
    Eg_GW = []
    Eg_PBE = []
    thick = []
    n_2D = []
    polar = []

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
    return alpha_x, alpha_z, Eg_HSE, thick
