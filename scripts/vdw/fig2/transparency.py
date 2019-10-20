import numpy
# import os, os.path
# from os.path import join, exists, dirname, abspath

# Add the utils
from ..utils.kkr import kkr, matsubara_freq
from ..utils.lifshitz import alpha_to_eps, g_amb_alpha, g_amb, g_amb_part, g_amb_alpha_part, transparency_single
from ..utils.img_tools import get_color, add_cbar
from ..utils.eps_tools import get_alpha, get_eps, get_index, data_2D
from . import data_path, img_path


# curdir = abspath(dirname(__file__))
# img_path = join(curdir, "../../img/", "fig2")

# if not exists(img_path):
    # os.makedirs(img_path)


# for i in range(len(data_2D)):
    # print(i, data_2D[i]["name"])



def get_energy_freq_dep(mater_2D, mater_bulk_a,
                        mater_bulk_b=None, d=1e-9, renormalized=True):
    # print(d)
    eps_a, freq_matsu, *_ = get_eps(mater_bulk_a)
    if mater_bulk_b is  None:
        eps_b = eps_a
    else:
        eps_b, *_ = get_eps(mater_bulk_b)
    if mater_2D >=0:            # A true material
        alpha, freq_alpha, *_ = get_alpha(mater_2D)
        g_part = g_amb_alpha_part(eps_a, alpha, eps_b,
                                  freq_matsu, freq_alpha, d, renormalized=renormalized)
    else:                       # vacuum
        eps_v = numpy.ones_like(eps_a)
        g_part = g_amb_part(eps_a, eps_v, eps_b, freq_matsu, d, renormalized=renormalized)
    return g_part, freq_matsu

# a = ("InSb-zincblende")
# b = ("InSb-zincblende")
# a = ("InSb-exp")
# b = ("InSb-exp")
a = "Si3N4-exp"
b = "SiO2-exp"

ind_a = get_index(a, "bulk")
ind_b = get_index(b, "bulk")

# EPS
eps_a, freq, *_ = get_eps(ind_a)
eps_b, *_ = get_eps(ind_b)
# Egs = []
d = 5e-9

def get_trans(d_=d, force=False):
    res_file = data_path / "2D" /  "transparency_{:.1f}.npz".format(d_ / 1e-9)
    if force is not True:
        if res_file.exists():
            data = numpy.load(res_file, allow_pickle=True)
            return data["names"], data["Eg"], data["G"], data["transparency"], data["freq_matsu"]



    Egs = []
    transparency = []
    names = []
    G = []
    # Vacuum
    g_0, *_ = get_energy_freq_dep(-1, ind_a, ind_b, d_)
    Egs.append(1e4)
    G.append(g_0)
    names.append("Vacuum")
    transparency.append(numpy.ones_like(g_0))
    for ind_m in range(0, len(data_2D)):
        alpha_m, freq_alpha, Eg, *_ = get_alpha(ind_m)
        formula = data_2D[ind_m]["formula"]
        prototype = data_2D[ind_m]["prototype"]
        names.append("{}-{}".format(formula, prototype))
        Egs.append(Eg)
        g_part, freq_matsu = get_energy_freq_dep(ind_m, ind_a, ind_b, d_)
        G.append(g_part)
        eta = g_part / g_0
        transparency.append(eta)
    Egs = numpy.array(Egs)
    G = numpy.array(G)
    transparency = numpy.array(transparency)
    numpy.savez(res_file, names=names, Eg=Egs, G=G, transparency=transparency, freq_matsu=freq_matsu)
    return names, Egs, G, transparency, freq_matsu

def main(**kargs):
    get_trans(**kargs)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run main function")
    parser.add_argument("-f", "--force", dest="force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
