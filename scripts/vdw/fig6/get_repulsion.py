import numpy
# import os, os.path
# import matplotlib
# import matplotlib.pyplot as plt
import json
# from numpy import meshgrid
from ..utils.kkr import kkr, matsubara_freq
from ..utils.lifshitz import alpha_to_eps, g_amb_alpha, g_amb, transparency_single
from ..utils.eps_tools import get_alpha, get_eps, get_index, data_2D, data_bulk
from . import data_path
# from os.path import join, exists, dirname, abspath
# plt.style.use("science")

# curdir = os.path.abspath(os.path.dirname(__file__))

# img_path = join(curdir, "../../img", "fig6")
# file_class = join(curdir, "../../data/bulk", "class.csv")
# if not exists(img_path):
    # os.makedirs(img_path)

freq_matsu = matsubara_freq(n=numpy.arange(1000), mode="energy")
# index_a = get_index("Au-exp", "bulk")
# print(index_a)
# print(data_bulk[index_a])


def get_energy(index_m, index_a, d_min=8e-10, d_max=3e-9, force=False):
    res_file = data_path / "pho_amb_{:d}_{:d}.npz".format(index_m, index_a)
    if res_file.exists() and (force is not True):
        data = numpy.load(res_file)
        return data["dd"], data["phi_amb"]
    else:
        alpha_m, freq_alpha, *_ = get_alpha(index_m, correction=False)
        eps_a, freq, *_ = get_eps(index_a)
        eps_b = numpy.ones_like(eps_a)
        dd = numpy.linspace(d_min, d_max, 50)
        e_abs = []
        for d in dd:
            e_abs.append(g_amb_alpha(eps_a, alpha_m, eps_b,
                                     freq, freq_alpha, d))
        e_abs = numpy.array(e_abs)
        numpy.savez(res_file, dd=dd, phi_amb=e_abs)
        return dd, e_abs

def get_energy_two_body(index_m, index_a, delta=3.3e-10,
                        d_min=8e-10, d_max=10e-9, force=False):
    res_file = data_path / "pho_mb_{:d}_{:d}.npz".format(index_m, index_a)
    if res_file.exists() and (force is not True):
        data = numpy.load(res_file)
        return data["dd"], data["phi_mb"]
    else:
        eps_a, freq, *_ = get_eps(index_a)
        alpha_m, freq_alpha, *_ = get_alpha(index_m, correction=False)
        eps_m_x = alpha_to_eps(alpha_m[0], delta, direction="x")
        eps_m_z = alpha_to_eps(alpha_m[2], delta, direction="z")
        eps_m_xiv = kkr(freq_alpha, eps_m_x, freq)
        eps_m_ziv = kkr(freq_alpha, eps_m_z, freq)
        # KKR Transform
        eps_m = numpy.array([eps_m_xiv, eps_m_xiv, eps_m_ziv])
        eps_v = numpy.ones_like(eps_a)
        dd = numpy.linspace(d_min, d_max, 50)
        e_abs = []
        for d in dd:
            ea = g_amb(eps_a, eps_v, eps_m,
                       freq,  d)
            eb = g_amb(eps_a, eps_v, eps_m,
                       freq, d - delta)
            delta_e = eb - ea
            e_abs.append(delta_e)
        e_abs = numpy.array(e_abs)
        numpy.savez(res_file, dd=dd, phi_mb=e_abs)
        return dd, e_abs
    

# def plot():
#     fig = plt.figure(figsize=(6, 3))
#     ax1 = fig.add_subplot(121)
#     ax2 = fig.add_subplot(122)
#     names = {3:"Gr",
#             27:"h-BN"}
#     for ind_m in [3, 27]:
#         dd, e_abs = get_energy(ind_m)
#         dd, e_n = get_energy_two_body(ind_m)
#         ax1.plot(dd / 1e-10, e_abs * 1000, label="{}".format(names[ind_m]))
#         ax2.plot(dd / 1e-10, (e_abs + e_n) * 1000, label="{} Total".format(names[ind_m]))
#         ax2.plot(dd / 1e-10, e_n * 1000, "--", label="{} Two-body only".format(names[ind_m]))
#     ax1.axhline(y=0, ls="--")
#     ax1.set_ylabel("$\\Delta \\Phi^{amb}$ (mJm$^{-2}$)")
#     ax1.set_xlabel("$d$ ($\\rm{\\AA}$)")
#     ax1.set_title("3-body Interactions")

#     ax2.axhline(y=0, ls="--")
#     ax2.set_ylabel("$\\Delta \\Phi^{tot}$ (mJm$^{-2}$)")
#     # ax2.set_ylabel("3 body vs 2 body")
#     ax2.set_xlabel("$d$ ($\\rm{\\AA}$)")
#     ax2.set_title("Total interactions")
#     ax1.legend()
#     ax2.legend()
#     fig.tight_layout()
#     fig.savefig(join(img_path, "rep_gold.png"))


if __name__ == "__main__":
    raise NotImplementedError("Don't run this module as main!")
