import numpy as np
from ase.atoms import Atoms
from ase.io import read, write
from ase.build import mx2, nanotube
from . import data_path
from ase.visualize import view

def build_nt():
    for l in (10, 20, 30, 40,):
        nt = nanotube(n=4, m=4, length=l)
        f = data_path / "nt_{0}.xyz".format(l)
        write(f.as_posix(), nt)

def build_mx2():
    mx = mx2(size=(40, 40, 1))
    f = data_path / "mx2.xyz"
    write(f.as_posix(), mx)

def build_mx2_rect(x=20, y=20, label=""):
    mx = mx2(size=(100, 100, 1))
    m = [a for a in mx if ((0 < a.x < x) and (0 < a.y < y))]
    m = Atoms(m)
    f = data_path / "mx2_rect{0}.xyz".format(label)
    write(f.as_posix(), m)

def build_mx2_tri(a=20, label=""):
    mx = mx2(size=(a, a, 1))
    m = [a for a in mx if (a.x >= 0) and (np.arctan(a.y / (a.x + 1e-10))) <= np.pi / 3]
    m = Atoms(m)
    f = data_path / "mx2_tri{0}.xyz".format(label)
    write(f.as_posix(), m)

def build_perovskite(a=20, n=2):
    mol = read((data_path / "cspbbr3-single.xyz").as_posix())
    m = mol * (a, a, n)
    natoms = [a_ for a_ in m if (a_.x <= (a - 1 + 0.05) * (mol.cell[0, 0])) \
           and (a_.y <= (a - 1 + 0.05) * (mol.cell[1, 1])) \
           and (a_.z <= (n - 1 + 0.05) * (mol.cell[2, 2]))]
    m = Atoms(natoms)
    f = data_path / "perov_{0}_{1}.xyz".format(a, n)
    write(f.as_posix(), m)

if __name__ == '__main__':
    build_nt()
    build_mx2()
    for x in range(20, 100, 10):
        build_mx2_rect(x=x, y=x, label=x)
    for x in range(5, 30, 5):
        build_mx2_tri(a=x, label=x)

    for a in range(5, 30, 5):
        for n in range(1, 5):
            build_perovskite(a=a, n=n)
