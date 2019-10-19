import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import cumtrapz
# import os, os.path
# from os.path import join, exists, abspath, dirname
from . import data_path, img_path
from helper import gridplots, grid_labels, savepgf
from ..utils.eps_tools import get_alpha, get_index
from ..utils.lifshitz import alpha_to_eps, g_amb_alpha
from ..utils.kkr import kkr, matsubara_freq
from ..utils.eps_tools import get_eps, get_alpha, data_2D
# from .screening_d import get_screening_gap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import interp1d
from scipy.stats import linregress



candidates = ["C2-C",
              # "MoS2-MoS2",
              "BN-BN",]

disp_names = {
    # "MoS2": "2H-MoS$_{2}$",
               "BN": "hBN",
               "C2": "Gr"}

# disp_names = {"MoS2": "2H-MoS$_{2}$",
               # "BN": "hBN",
               # "C2": "Gr"}

def cal_power(d, g, kernel=5):
    xx = np.linspace(d[0], d[-1], 1024)
    yy = interp1d(d, g, kind="cubic")(xx)
    new_xx = xx[:-kernel]
    power = []
    for i in range(len(xx) - kernel):
        x_ = np.log(xx[i: i + kernel])
        y_ = np.log(np.abs(yy[i: i + kernel]))
        k, b, *_ = linregress(x_, y_)
        power.append(k)
    return new_xx, np.array(power)

a = ("SiO2-exp")
# a = ("Au-exp")
b = ("Si3N4-exp")
# a = ("Si3N4-exp")

ind_a = get_index(a, "bulk")
ind_b = get_index(b, "bulk")




# EPS
eps_a, freq, *_ = get_eps(ind_a)
eps_b, *_ = get_eps(ind_b)


def get_screening_gap(min=2, max=15, force=False, ratio=1.3):
    ds = np.linspace(min, max, 128) * 1e-9
    res_file = data_path / "2D" / "vdW_screen_fig4_{0:.1f}_{1:.1f}.npz".format(min, max)
    if force is not True:
        if res_file.exists():
            data = np.load(res_file)
            return data["names"], data["Eg"], data["ds"], data["G"]


    gs = []
    Egs = []
    names = []
    # Vacuum case
    for ind_m in range(-1, len(data_2D)):
        if ind_m == -1:
            name = "Vacuum"; eg = 1e4
            alpha_m = np.zeros((3, 1024))
            freq_alpha = np.linspace(0, 150, 1024)
        else:
            alpha_m, freq_alpha, eg, *_ = get_alpha(ind_m)
            formula = data_2D[ind_m]["formula"]
            prototype = data_2D[ind_m]["prototype"]
            if not formula in disp_names.keys():
                continue
            name = "{}-{}".format(formula, prototype)
        names.append(name)
        Egs.append(eg)
        gs_ = []
        for d_  in ds:
            gs_.append(g_amb_alpha(eps_a, alpha_m, eps_b,
                                   freq, freq_alpha, d_))
            print(d_)
        gs.append(gs_)
    gs = np.array(gs)
    Egs = np.array(Egs)
    np.savez(res_file, names=names, Eg=Egs, ds=ds, G=gs)
    return names, Egs, ds, gs

        
    

def plot_main():
    fig, ax = gridplots(1, 1, r=0.7, ratio=1.33)
    names, Egs, ds, gs = get_screening_gap(min=1, max=20,)

    ax_inset = inset_axes(ax, width="70%", height="70%",
                          loc="lower right",
                          bbox_to_anchor=(0.25, 0.15, 0.75, 0.75),
                          bbox_transform=ax.transAxes)

    for can in candidates:
        i = list(names).index(can)
        name = disp_names[can.split("-")[0]]
        ax_inset.plot(ds, -gs[i])
        j = 25
        ax_inset.text(ds[j] * 1.05, -gs[0][j], s="←$\Phi^{0} \propto d^{-2}$",
                      color="grey", ha="left", va="bottom")

        # Plot the power
        new_dd, power = cal_power(ds, gs[i])
        j = 75
        l, = ax.plot(new_dd / 1e-9, -power)
        ax.text(new_dd[j] / 1e-9 * 1.2, -power[j], s=name, color=l.get_c())
    ax_inset.loglog(ds, -gs[0], ls="--", color="grey")
    ax_inset.tick_params(axis="both", labelsize="small")
    ax_inset.set_xlabel("$d$ (m)", size="small", labelpad=-0.5)
    ax_inset.set_ylabel(r"$\Phi$ (J·m$^{-2}$)", size="small", labelpad=-0.5)
    ax.set_xlabel("$x$ (nm)")
    ax.set_ylabel("$p$")
    ax.axhline(y=2, ls="--", color="grey")
    ax.set_ylim(0.5, 2.2)
    ax.text(x=5, y=2.05, s="Vacuum limit", va="bottom", ha="left")


    # 

    # Left
    # ax_ = ax
    # ax_.plot(freq_real, eps_para, label="In-plane")
    # ax_.plot(freq_real, eps_perp, label="Out-of-plane")
    # ax_.set_xlabel(r"$\hbar \omega$ (eV)")
    # ax_.set_ylabel(r"Im[$\varepsilon_{\mathrm{m}}(\omega)$]")
    # ax_.set_xlim(0, 30)
    # ax_.text(x=0.5, y=0.5, s=r"m=2H-MoS$_{2}$ $d$=2 nm", size="small",
             # ha="center",
             # transform=ax_.transAxes)
    # l = ax_.legend(loc=0)
    # l.set_title("m=2H-MoS$_{2}$ $d$=2 nm")

    # Right
    # ax_ = ax[1]
    # ax_.yaxis.tick_right()
    # ax_.yaxis.set_label_position("right")
    # ax_.plot(freq_matsu, eps_para_iv, label="In-plane")
    # ax_.plot(freq_matsu, eps_perp_iv, label="Out-of-plane")
    # # l = ax_.legend(loc=0)
    # ax_.set_xlabel(r"$\hbar \xi$ (eV)")
    # ax_.set_ylabel(r"$\varepsilon_{\mathrm{m}}(\xi)$")
    # ax_.set_xlim(0.16, 30)
    # ax_.set_ylim(0.5, 5)
    # ax_.axhline(y=1, ls="--", color="grey", alpha=0.6)
    # ax_.set_xscale("log")

    # # ax_.text(x=0.63, y=0.82, s=r"",
    # #          ha="left",
    # #          va="top",
    # #          transform=ax_.transAxes)
    # #
    # dummy = fig.add_subplot(111)
    # dummy.set_axis_off()

    # bbox_props = dict(boxstyle="rarrow, pad=0.5", fc="#cfcfcf", alpha=0.8)
    # t = dummy.text(0.5, 0.5, "KKR", ha="center", va="center",
    #             transform=fig.transFigure,
    #             bbox=bbox_props)

    # # bb = t.get_bbox_patch()
    # # bb.set_boxstyle("rarrow", pad=0.6)
    
    # # dummy.annotate("", xy=(0.55, 0.5), xytext=(0.45, 0.5),
    #                # xycoords="figure fraction",
    #                # ha="center", va="bottom",
    #                # arrowprops=dict(arrowstyle="->")
                   
    # # )
    savepgf(fig, img_path / "power_law.pgf")
    # fig.savefig(join(img_path, "integ_xx.svg"))

if __name__ == "__main__":
    plot_main()
