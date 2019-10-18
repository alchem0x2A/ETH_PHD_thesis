import numpy

# Add the utils
from ..utils.kkr import kkr, matsubara_freq
from ..utils.lifshitz import alpha_to_eps, g_amb_alpha, g_amb
from ..utils.img_tools import get_color, add_cbar
from ..utils.eps_tools import get_alpha, get_index, data_2D
from . import data_path, img_path

from helper import gridplots, grid_labels, savepgf

import matplotlib.pyplot as plt



freq_matsu = matsubara_freq(numpy.arange(0, 1000),
                             mode="energy")

from ase.data import covalent_radii
from ase.db import connect
db_fname = (data_path / "2D"/ "c2db_minimal.db").as_posix()
db = connect(db_fname)

def omega_half(alpha, freq, d):
    # d is in nm, convert to Angstrom
    d = d / 1e-10
    pi = numpy.pi
    alpha_x = (alpha[0] + alpha[1]) / 2
    alpha_z = alpha[2]
    # eps_x = 1 + 4 * pi * alpha_x / d
    # eps_z = 1 / (1 - 4 * pi * alpha_z / d)
    eps_x = alpha_to_eps(alpha_x, d, direction="x", imag=False)
    eps_z = alpha_to_eps(alpha_z, d, direction="z", imag=False)

    # Convert to kkr!
    eps_x_iv = kkr(freq, eps_x.imag, freq_matsu)
    eps_z_iv = kkr(freq, eps_z.imag, freq_matsu)
    alpha_x_iv = kkr(freq, alpha_x.imag, freq_matsu) - 1  # 
    alpha_z_iv = kkr(freq, alpha_z.imag, freq_matsu) - 1
    # print(alpha_x_iv[0], alpha_x[0].real)
    eps_x0 = eps_x_iv[0]
    eps_z0 = eps_z_iv[0]
    omega_half_x = freq_matsu[(eps_x_iv - 1) <= ((eps_x0 - 1) / 2)][0]
    omega_half_z = freq_matsu[(eps_z_iv - 1) <= ((eps_z0 - 1) / 2)][0]

    omega_alpha_z = freq_matsu[alpha_z_iv <= (alpha_z_iv[0] / 2)][0]
    return omega_half_x, omega_half_z, omega_alpha_z

def get_transition_freq(d_=2):
    d = d_ * 1e-9
    res_file = data_path / "2D" / "transition_freq_{:.1f}.npz".format(d_)
    if res_file.exists():
        data = numpy.load(res_file, allow_pickle=True)
        return data["names"], data["Eg"], data["trans_para"], data["trans_perp"]
    # Else create from scratch
    names = []
    Eg = []
    trans_para = []
    trans_perp = []
    list_matter = range(len(data_2D))
    for i in list_matter:
        print(i)
        alpha, freq, eg = get_alpha(i)
        formula = data_2D[i]["formula"]
        prototype = data_2D[i]["prototype"]
        names.append("{}-{}".format(formula, prototype))
        if alpha[2][0].real > 1:
            continue            # probably bad data
        omega_half_x, omega_half_z, omega_alpha_z = omega_half(alpha, freq, d)
        atoms = list(db.select(formula=formula, prototype=prototype))
        #
        Eg.append(eg)
        trans_para.append(omega_half_x)
        trans_perp.append(omega_half_z)
    Eg = numpy.array(Eg)
    trans_para = numpy.array(trans_para)
    trans_perp = numpy.array(trans_perp)
    numpy.savez(res_file.as_posix(), names=names, Eg=Eg,
                trans_para=trans_para, trans_perp=trans_perp)
        

# def plot_decay(d_=2, list_matter=None, list_pick=[]):
#     d = d_ * 1e-9                # in nm
#     # fig = plt.figure(figsize=(7.5, 2.5))
#     # ax1 = fig.add_subplot(131)
#     # ax2 = fig.add_subplot(133)
#     # ax3 = fig.add_subplot(132)

#     # fig, ax = gridplots(1, 3, r=0.8)
#     # ax1 = ax[0]
#     # ax2 = ax[1]
#     # ax3 = ax[2]
    
#     if list_matter is None:
#         list_matter = range(len(data_2D))

#     res = []
#     for i in list_matter:
#     # for i in range(5):
#         print(i)
#         alpha, freq, Eg = get_alpha(i)
#         formula = data_2D[i]["formula"]
#         prototype = data_2D[i]["prototype"]
#         print(formula, prototype)
#         # print(data_2D[i]["name"], alpha[2][0].real)
#         if alpha[2][0].real > 1:
#             continue            # probably bad data
#         # print(alpha)
#         # print(Eg)
#         omega_half_x, omega_half_z, omega_alpha_z = omega_half(alpha, freq, d)
#         print(Eg, omega_half_x, omega_half_z)
#         atoms = list(db.select(formula=formula, prototype=prototype))
#         print(atoms)
#         # if len(atoms) == 0:
#             # continue
#         # else:
#             # d = get_thick(atoms[0])
#         # if max(1 / g) > 1.1:
#             # continue
#         # print(freq.shape, g.shape)
#         # if i in list_pick:
#             # alpha = 1
#         # else:
#             # alpha = 0.08
#         alpha_plot = 0.5
#         # ax1.plot(Eg, omega_half_x, "^", alpha=alpha_plot, color="#6196ed")
#         # ax2.plot(alpha[2][0].real, omega_half_z, "s", alpha=alpha_plot, color="#fcaa67")
#         # ax2.plot(d, omega_half_z, "s", alpha=alpha_plot, color="#fcaa67")
#         # ax3.plot(Eg, omega_half_z, "o", alpha=alpha_plot, color="#ef645b")
#         # res.append([alpha[2][0].real, omega_alpha_z])
#         res.append([Eg, omega_half_x, omega_half_z])
#     res = numpy.array(res)
#     print(res)
#     numpy.savetxt("alpha0.csv", X=numpy.array(res), delimiter=",")
        
#         # ax3.plot(freq_matsu, 1 / g, "o", color=get_color(Eg), markersize=3,
#                     # alpha=alpha)
#     # False cmap
#     # ax1.axhline(y=1, color="k", ls="--")
#     ax1.plot(res[:, 0], res[:, 1], "^")
#     ax1.set_xlabel("$E_{\\mathrm{g}}$ (eV)")
#     ax1.set_title("In-plane")
#     # ax1.set_xscale("log")
#     # ax1.set_yscale("log")
#     ax1.set_ylim(0.9, 14)
#     ax1.set_ylabel("$\\hbar \\xi_{\\mathrm{tr}}^{\\parallel}$ (eV)")

#     # ax2.axhline(y=1, color="k", ls="--")
#     ax2.plot(res[:, 0], res[:, 2], "^")
#     ax2.set_xlabel("$\\alpha_{0}^{\\perp}$ ($\\mathrm{\\AA{}}$)")
#     # ax2.set_xscale("log")
#     # ax2.set_yscale("log")
#     # ax2.set_ylim(0.9, 20)
#     # ax2.set_xlim(0, 1)
#     ax2.set_ylabel("$\\hbar \\xi_{1/2}$ (eV)")
#     ax2.set_title("Out-of-plane")
#     # cbar  = add_cbar(fig, plt.gca(), n_max=8)
#     # plt.title("d = {} nm".format(d_))
#     # cbar.ax.set_title("$Eg$ (eV)")

#     ax3.set_xlabel("$E_{\\mathrm{g}}$ (eV)")
#     # ax2.set_xscale("log")
#     # ax2.set_yscale("log")
#     # ax2.set_ylim(0.9, 20)
#     ax3.set_ylabel("$\\hbar \\xi_{\\mathrm{tr}}^{\\perp}$ (eV)")
#     ax3.set_title("Out-of-plane")

#     '''
#     ax3.axhline(y=1, color="k", ls="--")
#     ax3.set_xlabel("$\\hbar \\omega$ (eV)")
#     ax3.set_xscale("log")
#     ax3.set_ylabel("$1 / g_m^2$")
#     cbar  = add_cbar(fig, ax3, n_max=8)
#     # .title("d = {} nm".format(d_))
#     cbar.ax.set_title("$Eg$ (eV)")
#     '''
#     # fig.tight_layout()
#     # ax1.text(0.0, 0.95, s="a", size=15, weight="bold",
#              # transform=fig.transFigure, va="top")
#     # ax2.text(0.33, 0.95, s="b", size=15, weight="bold",
#              # transform=fig.transFigure, va="top")
#     # ax3.text(0.66, 0.95, s="c", size=15, weight="bold",
#              # transform=fig.transFigure, va="top")
             
#     # fig.savefig(img_path / "omega_half_d_{:1f}.png".format(d_))
#     savepgf(fig, img_path / "omega_half.pgf")

def main():
    d = 2
    get_transition_freq(d)

if __name__ == "__main__":
    # list_matter = [i for i, m in enumerate(data_2D) \
                   # if m["name"] in ("MoS2-MoS2", "BN-BN", "C2-C")]
        # plot_dgm2(d, list_pick=list_matter)
    main()
