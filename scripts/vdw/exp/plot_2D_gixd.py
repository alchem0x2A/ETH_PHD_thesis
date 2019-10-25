import numpy as np
from .gixd_pp import get_powder_data, get_gixd_data, angle_dist
from .gixd_pp import read_param, data_gixd
import matplotlib as mpl
from . import data_path, img_path
import os.path
from os.path import join, exists, abspath, dirname
from matplotlib.colors import Normalize
from helper import gridplots

mpl.use("Agg")
mpl.rcParams["text.usetex"] = False
mpl.rcParams["svg.fonttype"] = "none"

# plt.style.use("science")


# plt.figure(figsize=(6, 3))
# qq, ii = get_powder_data(width=0.02)
# plt.plot(qq, ii / max(ii))
# plt.savefig(join(img_path, "xrd_q.png"))

maters = dict()
date_old = "0625"
date_new = "0726"
date_new2 = "1008"
date_old1 = "1008-old"
date_new1 = "1008-new"
gr = "graphene"
nogr = "no-graphene"

maters = {}
# maters["Plasma_gr_old"] = dict(name="Plasma", date=date_old, condition=gr)
# maters["OTS_gr_old"] = dict(name="OTS", date=date_old, condition=gr)
# maters["Au_gr_old"] = dict(name="Au", date=date_old, condition=gr)
# maters["Plasma_gr_new"] = dict(name="Plasma", date=date_new, condition=gr)
# maters["OTS_gr_new"] = dict(name="OTS", date=date_new, condition=gr)
# maters["Au_gr_new"] = dict(name="Au", date=date_new, condition=gr)
# maters["Plasma_nogr_new"] = dict(name="Plasma", date=date_new, condition=nogr)
# maters["OTS_nogr_new"] = dict(name="OTS", date=date_new, condition=nogr)
# maters["Au_nogr_new"] = dict(name="Au", date=date_new, condition=nogr)

# maters["Au_gr_new2"] = dict(name="Au", date=date_new2, condition=gr)
# maters["Plasma_gr_new2"] = dict(name="Plasma", date=date_new2, condition=gr)

maters["Plasma_gr_old1"] = dict(name="Plasma", date=date_old1, condition=gr)
# maters["OTS_gr_old"] = dict(name="OTS", date=date_old1, condition=gr)
maters["Au_gr_old1"] = dict(name="Au", date=date_old1, condition=gr)
maters["Plasma_gr_new1"] = dict(name="Plasma", date=date_new1, condition=gr)
# maters["OTS_gr_old"] = dict(name="OTS", date=date_old1, condition=gr)
maters["Au_gr_new1"] = dict(name="Au", date=date_new1, condition=gr)

short_names = ("Plasma", "OTS", "Au")


def show_gixd(ax, mater_name,
              v_minmax=None,
              q_range=(8, -0.1, 12, 20)):
    X, Y, data = get_gixd_data(**maters[mater_name], q_range=q_range)
    if v_minmax is None:
        param = read_param(join(data_gixd, "scaling.json"),
                           **maters[mater_name])
        v_min, v_max = param["vminmax"]
    else:
        v_min, v_max = v_minmax
    if "new2" in mater_name:
        data = 255 - data
        # norm=None
    ax.imshow(data,
              vmin=v_min, vmax=v_max,
              extent=q_range,
              rasterized=True,
              # norm=Normalize(vmin=v_min, vmax=v_max),
              cmap="inferno")
    ax.set_xlabel("$q_{xy}$ (nm$^{-1}$)")
    ax.set_ylabel("$q_{z}$ (nm$^{-1}$)")
    return


def show_large():
    for mater_name in maters.keys():
        fig, ax = gridplots(r=0.5, ratio=1)
        ax.set_aspect('equal')
        # fig = plt.figure(figsize=(3, 5))
        show_gixd(ax, mater_name, q_range=(15, -0.1, -0.1, 20))
        fig.tight_layout()
        fig.savefig(img_path / "2D_{0}_large.svg".format(mater_name))
        fig.savefig(img_path / "2D_{0}_large.png".format(mater_name))

def show_small():
    for mater_name in maters.keys():
        fig, ax = gridplots(r=0.5, ratio=1)
        ax.set_aspect('equal')
        show_gixd(ax, mater_name, q_range=(8, -0.1, 14, 20))
        fig.tight_layout()
        fig.savefig(img_path / "2D_{0}_small.svg".format(mater_name))
        fig.savefig(img_path / "2D_{0}_small.png".format(mater_name))

def show_nobg():
    for mater_name in maters.keys():
        fig, ax = gridplots(r=0.5, ratio=1)
        ax.set_aspect('equal')
        # fig = plt.figure(figsize=(3, 5))
        show_gixd(ax, mater_name, q_range=(6, -2, 2, 10))
        fig.tight_layout()
        fig.savefig(img_path / "2D_{0}_nobg.svg".format(mater_name))
        fig.savefig(img_path / "2D_{0}_nobg.png".format(mater_name))


if __name__ == "__main__":
    show_large()
    show_small()
    # show_nobg()
