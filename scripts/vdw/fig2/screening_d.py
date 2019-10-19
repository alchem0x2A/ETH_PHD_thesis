import numpy
# import os, os.path
# from os.path import join, exists, dirname, abspath
# import matplotlib.pyplot as plt
# plt.style.use("science")

# Add the utils
from ..utils.kkr import kkr, matsubara_freq
from ..utils.lifshitz import alpha_to_eps, g_amb_alpha, g_amb, g_amb_part, g_amb_alpha_part, transparency_single
from ..utils.img_tools import get_color, add_cbar
from ..utils.eps_tools import get_alpha, get_eps, get_index, data_2D
from . import data_path, img_path

from scipy.optimize import curve_fit


# curdir = abspath(dirname(__file__))
# img_path = join(curdir, "../../img/", "fig2")

# if not exists(img_path):
    # os.makedirs(img_path)


a = ("SiO2-exp")
b = ("Si3N4-exp")

ind_a = get_index(a, "bulk")
ind_b = get_index(b, "bulk")




# EPS
eps_a, freq, *_ = get_eps(ind_a)
eps_b, *_ = get_eps(ind_b)


def get_screening_gap(force=False):
    ds = numpy.linspace(2, 15, 32) * 1e-9
    res_file = data_path / "2D" / "vdW_screen_bg.npz"
    if force is not True:
        if res_file.exists():
            data = numpy.load(res_file)
            return data["names"], data["Eg"], data["d"], data["G"]


    gs = []
    Egs = []
    names = []
    # Vacuum case
    for ind_m in range(-1, len(data_2D)):
        if ind_m == -1:
            name = "Vacuum"; eg = 1e4
            alpha_m = numpy.zeros((3, 1024))
            freq_alpha = numpy.linspace(0, 150, 1024)
        else:
            alpha_m, freq_alpha, eg, *_ = get_alpha(ind_m)
            formula = data_2D[ind_m]["formula"]
            prototype = data_2D[ind_m]["prototype"]
            name = "{}-{}".format(formula, prototype)
        names.append(name)
        Egs.append(eg)
        gs_ = []
        for d_  in ds:
            gs_.append(g_amb_alpha(eps_a, alpha_m, eps_b,
                                   freq, freq_alpha, d_))
            print(d_)
        gs.append(gs_)
    gs = numpy.array(gs)
    Egs = numpy.array(Egs)
    numpy.savez(res_file, names=names, Eg=Egs, ds=ds, G=gs)
    return names, Egs, ds, gs

def main(force=False):
    get_screening_gap(force)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run main function")
    parser.add_argument("-f", "--force", dest="force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
