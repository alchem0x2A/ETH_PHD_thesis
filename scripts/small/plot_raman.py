import numpy as np
import matplotlib as mpl
from . import data_path, img_path
import os.path
from os.path import join, exists, abspath, dirname
from matplotlib.colors import Normalize
from helper import gridplots
import scipy.signal
from scipy.signal import medfilt
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.use("Agg")
mpl.rcParams["text.usetex"] = False
mpl.rcParams["svg.fonttype"] = "none"

#Plot 1D and 2D raman spectra



def plot_1D():
    fig, ax = gridplots(1, 1, r=0.6, ratio=1.618)
    file_temp = "sio2{0}-f16.txt"
    maters = ["", "-gr"]
    strings = ["Gr", "SiO$_{2}$"]
    
    for mat, name in zip(maters, strings):
        f_name = data_path / "Raman" / file_temp.format(mat)
        data = np.genfromtxt(f_name)
        cond_ = (data[:, 0] > 1200) & (data[:, 0] < 2000)
        data_plot = data[cond_]
        data_plot[:, 1] = (data_plot[:, 1] - min(data_plot[:, 1])) \
                          / (max(data_plot[:, 1]) - min(data_plot[:, 1]))\
                          + maters.index(mat) * 0.75
        ax.plot(data_plot[:, 0], data_plot[:, 1],
                label="F$_{16}$CuPc/" + name)
    ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax.set_ylabel(r"Intensity (a.u.)")
    ax.legend(loc=0)
    fig.savefig(img_path / "rama_1d.svg")


def read_2d(filename):
    # data = np.genfromtxt(filename, delimiter=",")
    data = np.load(filename)
    wavenumber = data[:, 0]
    spectra = data[:, 1:]
    return wavenumber, spectra


def apply_filter(wavenumber, spectra, filter_name, w_min, w_max):
    if filter_name == "max":
        filter_func = np.max
    elif filter_name == "min":
        filter_func = np.min
    elif filter_name == "mean":
        filter_func = np.mean
    else:
        raise ValueError("Unknown filter!")

    condition = np.where((wavenumber > w_min) & (wavenumber < w_max))
    spectra_filtered = filter_func(spectra[condition, :], axis=1)[0]
    # The returned spectra is a vector
    return spectra_filtered


def plot_Raman_2D(data, vmin_, vmax_, filename, interp=1024):
    """
    Only to plot the Raman data, no manipulation!
    data is 1XN^2 matrix
    """
    size = int(data.shape[0]**0.5)
    data = data.reshape(size, size)
    data = scipy.signal.medfilt2d(data)
    fig, ax = gridplots(1, 1, r=0.3, ratio=1.2)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin_, vmax=vmax_, interpolation="bicubic")
    ax.set_xticks([])
    ax.set_yticks([])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.set_ylabel("I(1380)/I(1315)")
    fig.savefig(filename)

def plot_f16(wavenumber, spectra, filename, vmin=None, vmax=None,):
    max_1 = apply_filter(wavenumber, spectra, "max", 1290, 1330)
    max_2 = apply_filter(wavenumber, spectra, "max", 1350, 1390)
    avg = apply_filter(wavenumber, spectra, "mean", 1240, 1280)
    res = (max_2 - avg)/(max_1 - avg)
    if vmin is None:
        vmin = min(res)
    if vmax is None:
        vmax = max(res)
    plot_Raman_2D(res, vmin_=vmin, vmax_=vmax, filename=filename)

def plot_2D():
    wavenumber, spectra = read_2d(data_path / "Raman/sample-1-2.npy")
    plot_f16(wavenumber, spectra,
             img_path / "raman_2d.svg", vmin=0.5, vmax=1.7)

if __name__ == '__main__':
    print(("The script outputs 1D and 2D Raman spectra into {} as svg!"
           " Do manual editing after").format(img_path.as_posix()))
    plot_1D()
    plot_2D()
