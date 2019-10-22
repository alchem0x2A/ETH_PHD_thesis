import ase.db
import warnings
import numpy
from ase.data import covalent_radii
from ase.dft.bandgap import bandgap
from ase.io.trajectory import Trajectory
from ase.io import read
from gpaw import GPAW
from ase.dft.bandgap import bandgap
import os, os.path
from scipy.constants import pi, epsilon_0
import scipy
import csv
import json



"""
Extract the alpha from the HSE xlsx files
"""

db_file = "../../data/gpaw_data/c2db.db"
if not os.path.exists(db_file):
    raise FileExistsError(("Please download the c2db data into ../../data/gpaw_data/ folder,"
                   "from https://cmr.fysik.dtu.dk/_downloads/c2db.db"))


db = ase.db.connect(db_file)
valence = numpy.load("./valence.npy")



materials = []

properties = ("name", "prototype", "Gap_2D", "delta_2D",
              "alphax_2D", "alphaz_2D",
              "Gap_3D", "L", "epsx_3D", "epsz_3D",
              "emass", "hmass", "CQ_c", "CQ_v")
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
    vals = valence[atom_row.numbers]  # valence electrons
    A = atom_row.cell_area
    return zmax, sum(vals) / A


def get_bulk(name, proto):
    # Get bulk properties
    dir_root = os.path.join("../../data/2D-bulk/{}-{}".format(name, proto))
    gpw_file = os.path.join(dir_root, "gs.gpw")
    traj_file = os.path.join(dir_root, "{}-{}_relax.traj".format(name, proto))
    eps_file = os.path.join(dir_root, "eps_df.npz")
    if (not os.path.exists(gpw_file)) or (not os.path.exists(eps_file)):
        return None
    else:
        try:
            d = Trajectory(traj_file)[-1].cell[-1][-1]
        except Exception:
            print(Exception)
            c_atoms = read(gpw_file)
            # calc = GPAW(gpw_file)
            d = c_atoms.cell[-1][-1]
            # d = calc.get_atoms().cell[-1][-1]
        f = numpy.load(eps_file)
        eps_xx = f["eps_x"][0].real
        eps_zz = f["eps_z"][0].real
        calc =GPAW(gpw_file)
        try:
            E_GGA, *_ = bandgap(calc)
        except Exception:
            E_GGA = None
        return d, eps_xx, eps_zz, E_GGA

# Compile QC values
QC_res = {}
reader = csv.reader(open("../../data/DOS/data_dale_dos.csv", encoding="utf8"))
next(reader)
for row in reader:
    if row[0] != "":
        name, proto = row[: 2]
        qc_n, qc_p = row[-4: -2]
        QC_res[(name, proto)] = (float(qc_n), float(qc_p))   

reader = csv.reader(open("../../data/HSE-data/2D_HSE.csv", encoding="utf8"))
next(reader)                    # skip line1
for row in reader:
    if row[4] != "":            # Eps calculated
        name, proto = row[: 2]
        print(name, proto)
        L, E, ex, ey, ez, E_direct, E_min = map(float, row[2:])
        # Elements
        key = (name, proto)
        e_xy = numpy.sqrt(ex * ey)
        ax = (e_xy - 1) / (4 * pi) * L
        az = (1 - 1/ez) * L / (4 * pi)
        
        if proto == "ABX3":     # perovskite?
            delta = 14.24
            n_2D = None
            mol = None
        else:
            mol = list(db.select(formula=name, prototype=proto))[0]
            delta, n_2D = get_thick(mol)
        # 3D
        try:
            L_3D, epsx, epsz, E_3D = get_bulk(name, proto)
        except TypeError:
            L_3D, epsx, epsz, E_3D = (None, None, None, None)
            # QC
        if (name, proto) in QC_res:
            qc_n, qc_p = QC_res[(name, proto)]
        else:
            qc_n = None; qc_p = None
        # emass
        try:
            emass = mol.emass1
        except Exception:
            emass = None
        try:
            hmass = mol.hmass1
        except Exception:
            hmass = None
            # Compile data
        datum = dict(name=name, prototype=proto,
                     gap_2D=E, delta_2D=delta, n_2D=n_2D,
                     gap_2D_direct=E_direct, gap_2D_min=E_min,
                     alphax_2D=ax, alphaz_2D=az,
                     gap_3D=E_3D, L_3D=L_3D,
                     epsx_3D=epsx, epsz_3D=epsz,
                     qc_n=qc_n, qc_p=qc_p,
                     emass=emass, hmass=hmass)
        materials.append(datum)

json.dump(materials, open("../../data/HSE-data/all_data.json", "w"))
