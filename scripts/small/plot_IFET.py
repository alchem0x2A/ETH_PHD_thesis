import numpy as np
import matplotlib as mpl
from dill import loads
from . import data_path, img_path
import os.path
from os.path import join, exists, abspath, dirname
from matplotlib.colors import Normalize
from helper import gridplots
import scipy.signal
from scipy.signal import medfilt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import cos, arccos, pi, degrees, radians, exp

mpl.use("Agg")
mpl.rcParams["text.usetex"] = False
mpl.rcParams["svg.fonttype"] = "none"


#Plot IdVg
def plot_idvg():
    fig, ax = gridplots(1, 1, r=0.4, ratio=1)
    data_IdVg = np.genfromtxt(data_path / "IFET" / "IdVg" / "Id_Vg_best.csv",
                              delimiter=",",
                              skip_header=13)

    r = 80e-6 / 1e-2
    S = scipy.pi * r ** 2


    V_g = data_IdVg[:, 0]
    I_d = data_IdVg[:, 1]
    cycles = data_IdVg[:, -2]
    condition_cycle = np.where(cycles == 1)


    V_g_valid = V_g[condition_cycle]
    I_d_valid = I_d[condition_cycle]
    J = I_d_valid / 1e-3 / S

    ax.set_xlim(-100, 100)
    ax.set_yscale("log")
    ax.set_xlabel("$J")
    ax.set_ylabel("$I_{\\mathrm{d}}$ (A)")
    # print(max(I_d_valid) / min(I_d_valid))
    ax.plot([], [])
    ax.plot(V_g_valid[201: 402], J[201: 402], "-o", markersize=4)
    ax.plot(V_g_valid[: 201], J[: 201], "-o", markersize=4)
    fig.savefig(img_path / "best_IV.svg")

# Plot IdVd
def plot_idvd():
    from scipy.signal import medfilt

    file_name = data_path / "IFET" / "Id_scan" /  "ots-250.csv"
    fig, ax = gridplots(1, 1, r=0.4, ratio=1)
    r = 280e-4
    S = np.pi * r ** 2

    ax.set_xlim(-100, 100)
    ax.set_yscale("log")
    ax.set_xlabel(r"$V_{\mathrm{G}}$ (V)")
    ax.set_ylabel(r"$J_{\mathrm{DS}}$ (mA$\cdot{}$cm$^{-2}$)")
    data = np.genfromtxt(file_name, delimiter=",", skip_header=13)

    space = 211
    ratio = []

    Vds = [1, 2, 3, 4, 5]

    for i in range(0, 10):
        Vd = np.abs(data[space * i + 1, -1])
        if Vd in Vds:
            mm = minmax[Vds.index(Vd)]
            print(Vd, mm)
            Vg = data[space * i: space * (i + 1), 0]
            Id = medfilt(np.abs(data[space * i : space * (i + 1), 1]))
            J = Id / 1e-3 / S
            ax.plot(Vg, J,  "-o", markersize=3, label="{} V".format(Vd),)
            ratio.append([Vd, max(Id) / min(Id)])

    fig.savefig(img_path / "Id_scan.svg")


def plot_main():
    plot_idvg()
    plot_idvd()


if __name__ == '__main__':
    print(("The script outputs IdVg and IdVd plots into {} as svg!"
           " Do manual editing after").format(img_path.as_posix()))
    
    plot_main()
