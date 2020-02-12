import ase.db
from ase.io import read, write
import warnings
import numpy
import matplotlib.pyplot as plt
from ase.data import covalent_radii
from scipy.stats import linregress
from scipy.constants import pi, epsilon_0
import csv
# from glob import glob
from . import data_path


def get_thick(atom_row):
    pos = atom_row.positions[:, -1]
    diff = covalent_radii[atom_row.numbers]
    zmax = numpy.max(pos + diff) - numpy.min(pos - diff)
    return zmax


def get_data_epfl(exclude=[]):
    # Read QE result
    eps_x = []; eps_z = []
    alpha_x = []; alpha_z = []
    materials = []
    Eg_QE = []
    Eg_VASP = []
    thick = []
    f_data = data_path / "EPFL-data" / "eps_EPFL.csv"
    reader = csv.reader(open(f_data.as_posix(), encoding="utf8"))
    next(reader)                    # skip line1
    for row in reader:
        if row[0] != "":
            name = row[0]
            print(name)
            if name in exclude:
                continue
            mol_path = data_path / "EPFL-data" / "structures"
            mol_file = list(mol_path.glob("{0}-*.json".format(name)))
            # mol_file = glob(os.path.join("../../data/EPFL-data/structures/",
                                         # "{0}-*.json".format(name)))
            if len(mol_file) == 0:
                print(name + " Not correct!")
                continue
            L, ex, ey, ez, eg_qe, eg_vasp = map(float, row[1:])
            if ez < ex:
                eps_z.append(ez)
                materials.append("-".join((name, "QE")))
                e_xy = numpy.sqrt(ex * ey)
                ax = (e_xy - 1) / (4 * pi) * L
                az = (1 - 1/ez) * L / (4 * pi)
                # ax = max(1 / 1.2, ax)
                eps_x.append(e_xy); eps_z.append(ez)
                alpha_x.append(ax); alpha_z.append(az)
                Eg_QE.append(eg_qe)
                Eg_VASP.append(eg_vasp)
                # if proto == "ABX3":
                # thick.append(8.0)
                # else:
                mol = read(mol_file[-1])
                thick.append(get_thick(mol))
                
    alpha_x = numpy.array(alpha_x)
    alpha_z = numpy.array(alpha_z)
    Eg_QE = numpy.array(Eg_QE)
    Eg_VASP = numpy.array(Eg_VASP)
    thick = numpy.array(thick)
    return materials, alpha_x, alpha_z, Eg_QE, Eg_VASP, thick

def get_data_epfl2(exclude=[]):
    # Read QE result
    eps_x = []; eps_z = []
    alpha_x = []; alpha_z = []
    materials = []
    Eg_QE = []
    Eg_VASP = []
    thick = []
    f_data = data_path / "EPFL-data" / "results.csv"
    reader = csv.reader(open(f_data.as_posix(),
                             encoding="utf8"))
    next(reader)                    # skip line1
    for row in reader:
        if row[0] != "":
            name = row[0]
            print(name)
            if name in exclude:
                continue
            mol_path = data_path / "EPFL-data" / "structures"
            mol_file = list(mol_path.glob("{0}-*.json".format(name)))
            if len(mol_file) == 0:
                print(name + " Not correct!")
                continue
            gap, gap_dir, gap_gl, gap_gl_dir, gap_hse, gap_hse_dir, ax, ay, az = map(numpy.float, row[4:])
            if any(numpy.isnan([ax, ay, az])):  # No polarizability
                continue
            if any(numpy.isnan([gap_gl, gap_gl_dir])):
                continue
            if az < ax:
                alpha_x.append((ax + ay) / 2)
                alpha_z.append(az + 0.03)
                if not numpy.isnan(gap_hse):
                    Eg_QE.append(gap_hse)
                    Eg_VASP.append(gap_hse)
                else:
                    Eg_QE.append(gap)
                    Eg_VASP.append(gap_dir)
                    
                materials.append("-".join((name, "QE")))
                mol = read(mol_file[-1])
                thick.append(get_thick(mol))
                
    alpha_x = numpy.array(alpha_x)
    alpha_z = numpy.array(alpha_z)
    Eg_QE = numpy.array(Eg_QE)
    Eg_VASP = numpy.array(Eg_VASP)
    thick = numpy.array(thick)
    return materials, alpha_x, alpha_z, Eg_QE, Eg_VASP, thick
