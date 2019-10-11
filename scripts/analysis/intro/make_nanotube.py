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

if __name__ == '__main__':
    build_nt()
    build_mx2()
