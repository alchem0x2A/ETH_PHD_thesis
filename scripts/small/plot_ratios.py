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

#Plot IFET ratios

data_all = np.load(data_path / "IFET" / "all_data_onoff.npz", allow_pickle=True)
data_onoff = data_all["data"]
fil = loads(data_all["f"].tobytes())

def get_data(i):
    data = data_onoff[i]
    data = data[: 201, :]
    Vg_raw = data[:, 0]
    Id_raw = medfilt(np.abs(data[:, 1]))
    cond = np.where((Vg_raw > -80) & (Id_raw > 5e-11))
    Vg_now = Vg_raw[cond]
    Id_now = Id_raw[cond]
    # fit = interp1d(Vg_now, Id_now,
                   # kind="zero",
                   # fill_value="extrapolate")
    VV = np.linspace(-100, 100, 1000)
    # II = np.exp(fit(VV))
    II = Id_now
    rat = fil(np.max(II) / np.min(II))
    
    return np.max(II), np.min(II), rat

def plot_main():
    fig, ax = gridplots(1, 1, r=0.5, ratio=1)
    ratios = []
    for i in range(len(data_onoff)):
        max_, min_, rat = get_data(i)
        ratios.append(rat)
    ratios = np.array(ratios)

    bins = 10 ** (np.arange(1, 6.5, 0.5))
    ax.set_xlabel("log(Ratio)")
    ax.set_ylabel("Count")
    ax.set_xscale("log")
    ax.hist(ratios, bins=bins)
    ax.set_ylim(0, 25)
    fig.savefig(img_path / "ratios_hist.svg")

if __name__ == '__main__':
    print(("The script outputs contact angle models into {} as svg!"
           " Do manual editing after").format(img_path.as_posix()))
    
    plot_main()
