import numpy as np
from .gixd_pp import get_powder_data, get_gixd_data, angle_dist
from .gixd_pp import data_gixd
from . import img_path, data_path
import matplotlib as mpl
from matplotlib.colors import Normalize
from helper import gridplots
from scipy.interpolate import interp1d

mpl.use("Agg")
mpl.rcParams["text.usetex"] = False
mpl.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
# import os.path
# from os.path import join, exists, abspath, dirname

# plt.style.use("science")

# plt.figure(figsize=(6, 3))
# qq, ii = get_powder_data(width=0.02)
# plt.plot(qq, ii / max(ii))
# plt.savefig(join(img_path, "xrd_q.png"))

maters = dict()
date_old = "0625"
date_new = "0726"
gr = "graphene"
nogr = "no-graphene"
date_old1 = "1008-old"
date_new1 = "1008-new"

maters = {}
# maters["Plasma_gr_old"] = dict(name="Plasma_nobg", date=date_old, condition=gr)
# maters["OTS_gr_old"] = dict(name="OTS_nobg", date=date_old, condition=gr)
# maters["Au_gr_old"] = dict(name="Au_nobg", date=date_old, condition=gr)
# maters["Plasma_gr_new"] = dict(name="Plasma_nobg", date=date_new, condition=gr)
# maters["OTS_gr_new"] = dict(name="OTS_nobg", date=date_new, condition=gr)
maters["Au_gr_new"] = dict(name="Au_nobg", date=date_new, condition=gr)
maters["Plasma_nogr_new"] = dict(name="Plasma_nobg", date=date_new, condition=nogr)
# maters["OTS_nogr_new"] = dict(name="OTS_nobg", date=date_new, condition=nogr)
maters["Au_nogr_new"] = dict(name="Au_nobg", date=date_new, condition=nogr)

maters["Plasma_gr_old1"] = dict(name="Plasma", date=date_old1, condition=gr)
maters["Au_gr_old1"] = dict(name="Au", date=date_old1, condition=gr)
maters["Plasma_gr_new1"] = dict(name="Plasma", date=date_new1, condition=gr)
maters["Au_gr_new1"] = dict(name="Au", date=date_new1, condition=gr)
# maters["Plasma_gr_old1"] = dict(name="Plasma_nobg", date=date_old1, condition=gr)
# maters["Au_gr_old1"] = dict(name="Au_nobg", date=date_old1, condition=gr)
# maters["Plasma_gr_new1"] = dict(name="Plasma_nobg", date=date_new1, condition=gr)
# maters["Au_gr_new1"] = dict(name="Au_nobg", date=date_new1, condition=gr)


short_names = ("Plasma", "OTS", "Au")

stanford_Data = np.genfromtxt(data_gixd /  "gixd_stanford.csv",
                              delimiter=",")
stanford_q = stanford_Data[:, 0] * 10
stanford_i = stanford_Data[:, 1]
stanford_i = (stanford_i - np.min(stanford_i)) /\
             (np.max(stanford_i) - np.min(stanford_i))



bpe_q = np.linspace(3.5, 20, 1024)
bpe_i = np.zeros_like(bpe_q)
count = 0
for i in range(1, 6):
    # bpe_Data = np.genfromtxt(data_gixd /  "bpe-powder{:d}.csv".format(i),
                             # delimiter=",")
    # bpe_Data = np.genfromtxt(data_gixd /  "bpe-recryst{:d}.csv".format(i),
                             # delimiter=",")
    bpe_Data = np.genfromtxt(data_gixd /  "bpe-gixd{:d}.csv".format(i),
                             delimiter=",")
    lim = 10
    q_ = bpe_Data[lim:, 0]
    i_ = 256 - bpe_Data[lim:, 1]
    if q_.max() < bpe_q.max():
        continue
    else:
        ii_ = interp1d(q_, i_)(bpe_q)
        bpe_i += ii_
        count += 1

bpe_i = bpe_i / count               # Average
bpe_i = (bpe_i - np.min(bpe_i)) / (np.max(bpe_i) - np.min(bpe_i))
# print(bpe_q, bpe_i)


    
exp_xrd_data = np.genfromtxt(data_gixd / "powder.csv",
                             delimiter=",")
exp_theta = exp_xrd_data[:, 0]
exp_i = exp_xrd_data[:, 1]
exp_i = (exp_i - exp_i.min()) / (exp_i.max() - exp_i.min())
l_wave = 0.154059               # Wavelength in nm
exp_q = np.sin(np.radians(exp_theta) / 2) * 4 * np.pi / l_wave

def plot_1D_profile(ax, q, intensity,
                    scale=1.2,
                    offset=0, name=""):
    ax.plot(q, intensity * scale + offset, label=name)
    return

def powder_data(q_range=(3.5, 22.5), width=0.02):
    qq, ii = get_powder_data(width=width)
    qmin, qmax = q_range
    cond = np.where((qq >= qmin) & (qq <= qmax))
    qq = qq[cond]
    ii = ii[cond]
    ii = ii / np.max(ii)
    return qq, ii


def chi_avg_data(mater_name,
                 q_range_2D=(20, -1, -0.5, 25),
                 q_range_1D=(3.5, 22.5)):
    # name = mater["name"]
    # date = mater["date"]
    # condition = mater["condition"]
    # name_file = "{0}_nobg".format(name)
    # Get actual data
    X, Y, data = get_gixd_data(**maters[mater_name],
                               q_range=q_range_2D)
    q_1D, spectrum, spectrum_norm = angle_dist(X, Y, data, q_range=q_range_1D)
    return q_1D, spectrum_norm


def plot_compare():
    # fig = plt.figure(figsize=(6, 8))
    fig, ax = gridplots(ratio=2)
    ax1 = ax
    # New data
    plot_1D_profile(ax1, stanford_q, stanford_i,
                    offset=-3.5, scale=5, name="Stanford")
    for i, mater_name in enumerate(["{0}_gr_new1".format(c) for c in short_names]):
        if mater_name in maters.keys():
            q, intensity = chi_avg_data(mater_name)
            plot_1D_profile(ax1, q, intensity, offset=-i, scale=2,
                            name="{0}/Gr/BPE".format(mater_name.split("_")[0]))
            # plot_1D_profile(ax1, q, intensity ** 0.5, offset=-i, scale=2,
                            # name="{0}/Gr/BPE".format(mater_name.split("_")[0]))
    for i, mater_name in enumerate(["{0}_nogr_new".format(c) for c in short_names]):
        if mater_name in maters.keys():
            q, intensity = chi_avg_data(mater_name)
            plot_1D_profile(ax1, q, intensity, offset=-i, scale=2,
                            name="{0}/BPE".format(mater_name.split("_")[0]))
    
    qq, ii = powder_data()
    # plot_1D_profile(ax1, qq, ii, offset=-3.5, scale=5, name="Power Sample")
    plot_1D_profile(ax1, bpe_q, bpe_i, offset=-3.5, scale=3, name="Powder Sample")
    # plot_1D_profile(ax1, bpe_q, bpe_i ** 0.5, offset=-3.5, scale=3, name="Powder Sample")
    # ax1.set_ylim(*ax1.get_ylim())
    # plot_1D_profile(ax1, exp_q, exp_i * 5, offset=-3.5, scale=5, name="Powder Exp")

    # ax1.set_title("New data (07.26)")
    ax1.set_xlabel("$|q|$ (nm)")
    ax1.set_yticks([])
    ax1.set_ylabel("Intensity (a.u.)")
    ax1.legend(loc=0)
    ax1.set_xlim(12, 20.5)
    ax1.set_xlim(3, 12)
    # ax1.set_ylim(-3.5, -0.5)


    fig.savefig(img_path /  "compare_gixd_1D_nobg.svg")

def plot_nogr():
    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(111)
    # New data
    for i, mater_name in enumerate(["{0}_nogr_new".format(c) for c in short_names]):
        q, intensity = chi_avg_data(mater_name)
        plot_1D_profile(ax1, q, intensity, offset=-i, scale=3, name=mater_name)

    qq, ii = powder_data()
    plot_1D_profile(ax1, qq, ii, offset=-2.5, scale=10, name="Powder XRD")

    ax1.set_xlabel("$|q|$ (nm)")
    ax1.set_yticks([])
    ax1.legend(loc=0)

    fig.tight_layout()
    fig.savefig(img_path /  "compare_gixd_nogr.png")


if __name__ == "__main__":
    plot_compare()
    # plot_nogr()
