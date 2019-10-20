import numpy
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
import os, os.path
from os.path import join, exists, abspath, dirname
from . import data_path, img_path
from helper import gridplots, grid_labels, savepgf
from ..utils.eps_tools import get_alpha, get_index
from ..utils.lifshitz import alpha_to_eps
from ..utils.kkr import kkr, matsubara_freq


def get_eps_both(mater, d=2e-9):
    freq_matsu = matsubara_freq(numpy.arange(0, 1000),
                                 mode="energy")
    alpha_real, freq_real, *_ = get_alpha(get_index(mater, kind="2D"))
    # By default, returns the imaginary part
    eps_para = alpha_to_eps(alpha_real[0], d, direction="x")
    eps_perp = alpha_to_eps(alpha_real[-1], d, direction="z")
    eps_para_iv = kkr(freq_real, eps_para, freq_matsu)
    eps_perp_iv = kkr(freq_real, eps_perp, freq_matsu)
    return freq_real, eps_para, eps_perp, freq_matsu, eps_para_iv, eps_perp_iv
    

def plot_main():
    mater=("MoS2", "MoS2")
    fig, ax = gridplots(1, 2, r=0.9, ratio=2.5)
    freq_real, eps_para, eps_perp, \
    freq_matsu, eps_para_iv, eps_perp_iv = get_eps_both(mater)

    # Left
    ax_ = ax[0]
    ax_.plot(freq_real, eps_para, label="In-plane")
    ax_.plot(freq_real, eps_perp, label="Out-of-plane")
    ax_.set_xlabel(r"$\hbar \omega$ (eV)")
    ax_.set_ylabel(r"Im[$\varepsilon_{\mathrm{m}}(\omega)$]")
    ax_.set_xlim(0, 30)
    # ax_.text(x=0.5, y=0.5, s=r"m=2H-MoS$_{2}$ $d$=2 nm", size="small",
             # ha="center",
             # transform=ax_.transAxes)
    l = ax_.legend(loc=0)
    l.set_title("m=2H-MoS$_{2}$ $d$=2 nm")

    # Right
    ax_ = ax[1]
    ax_.yaxis.tick_right()
    ax_.yaxis.set_label_position("right")
    ax_.plot(freq_matsu, eps_para_iv, label="In-plane")
    ax_.plot(freq_matsu, eps_perp_iv, label="Out-of-plane")
    # l = ax_.legend(loc=0)
    ax_.set_xlabel(r"$\hbar \xi$ (eV)")
    ax_.set_ylabel(r"$\varepsilon_{\mathrm{m}}(i \xi)$")
    ax_.set_xlim(0.16, 30)
    ax_.set_ylim(0.5, 5)
    ax_.axhline(y=1, ls="--", color="grey", alpha=0.6)
    ax_.set_xscale("log")

    # ax_.text(x=0.63, y=0.82, s=r"",
    #          ha="left",
    #          va="top",
    #          transform=ax_.transAxes)
    #
    dummy = fig.add_subplot(111)
    dummy.set_axis_off()

    bbox_props = dict(boxstyle="rarrow, pad=0.5", fc="#cfcfcf", alpha=0.8)
    t = dummy.text(0.5, 0.5, "KKR", ha="center", va="center",
                transform=fig.transFigure,
                bbox=bbox_props)

    # bb = t.get_bbox_patch()
    # bb.set_boxstyle("rarrow", pad=0.6)
    
    # dummy.annotate("", xy=(0.55, 0.5), xytext=(0.45, 0.5),
                   # xycoords="figure fraction",
                   # ha="center", va="bottom",
                   # arrowprops=dict(arrowstyle="->")
                   
    # )
    savepgf(fig, img_path / "eps_kkr.pgf")
    # fig.savefig(join(img_path, "integ_xx.svg"))

if __name__ == "__main__":
    plot_main()
