import numpy
# import os, os.path
import matplotlib as mpl
# from numpy import meshgrid
from ..utils.kkr import kkr, matsubara_freq
from ..utils.lifshitz import alpha_to_eps, g_amb_alpha, g_amb
from ..utils.lifshitz import g_amb_alpha_part, g_amb_part
# from os.path import join, exists, abspath, dirname
from ..utils.eps_tools import get_eps, get_alpha, data_2D, data_bulk, file_bulk, file_2D, get_index
import multiprocessing
from multiprocessing import Pool
from scipy.interpolate import interp1d
from . import img_path, data_path
from helper import gridplots, grid_labels, savepgf, add_cbar, get_color
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# curdir = os.path.abspath(os.path.dirname(__file__))
# img_path = os.path.join(curdir, "../../img/suppl")

candidates =  [("vac", "vac"),
               ("BN", "BN"),
               ("MoS2", "MoS2"),
               ("C2", "C")]

disp_names = {"vac": "Vac",
              "MoS2": "MoS$_{2}$",
               "BN": "hBN",
               "C2": "Gr"}

bulks = ["Cu-exp",
         "BN-zincblende",
         "GaAs-zincblende"]

bulk_title = {"Cu-exp": r"Cu $E_{\mathrm{g}}^{\mathrm{Bulk}}$=0 eV",
         "BN-zincblende": r"GaAs $E_{\mathrm{g}}^{\mathrm{Bulk}}$=1.4 eV",
              "GaAs-zincblende": r"cubic BN $E_{\mathrm{g}}^{\mathrm{Bulk}}$=6.4 eV",}




def get_energy_freq_dep(mater_2D, mater_bulk_a,
                        mater_bulk_b=None, d=1e-9,
                        renormalized=True, force=False):
    ind_a = get_index(mater_bulk_a, kind="bulk")
    # print(ind_a)
    eps_a, freq_matsu, *_ = get_eps(ind_a)
    if mater_bulk_b is None:
        eps_b = eps_a
        ind_b = ind_a
    else:
        ind_b = get_index(mater_bulk_b, kind="bulk")
        eps_b, *_ = get_eps(ind_b)
    if "vac" not in mater_2D:
        ind_2D = get_index(mater_2D, kind="2D")
    else:
        ind_2D = -1

    res_file = data_path / "bulk" /  "g_contrib_{0}_{1}_{2}_{3:.1f}.npz".format(ind_2D,
                                                                      ind_a, ind_b, d / 1e-9)

    if res_file.exists() and (force is not True):
        data = numpy.load(res_file)
        return data["freq_matsu"], data["g"]

    else:
        print(ind_a, ind_b, ind_2D, d)
        if ind_2D >= 0:
            alpha, freq_alpha, *_ = get_alpha(ind_2D)
            g_part = g_amb_alpha_part(eps_a, alpha, eps_b,
                                      freq_matsu, freq_alpha, d, renormalized=renormalized)
        else:                       # vacuum
            eps_v = numpy.ones_like(eps_a)
            g_part = g_amb_part(eps_a, eps_v, eps_b, freq_matsu, d, renormalized=renormalized)
        numpy.savez(res_file, freq_matsu=freq_matsu, g=g_part)
        return  freq_matsu, g_part



# 2D materials:

def plot_main():
    old_size =  mpl.rcParams["font.size"]
    mpl.rcParams["font.size"] = 8
    cbar_r = 0.1
    span = [(i, j, 1, 1) for i in range(4) for j in range(3)] + [(0, 3, 4, 1)]
    fig, ax = gridplots(4, 4, r=1, ratio=1.2,
                        span=span,
                        gridspec_kw=dict(width_ratios=[(4 - cbar_r) / 3,] * 3 \
                                         + [cbar_r / 4,]))
    ax[-1].set_axis_off()
    for i, m_2D in enumerate(candidates):
        ax[i * 3].text(x=-0.4, y=0.5, s=disp_names[m_2D[0]], ha="right",
                       transform=ax[i * 3].transAxes, size="large")
        for j, m_bulk in enumerate(bulks):
            ds = numpy.linspace(1, 10, 10) * 1e-9
            for d_ in ds:
                c = get_color(d_ / 1e-9, 1, 10, name="rainbow")
                freq_, g_ = get_energy_freq_dep(m_2D, m_bulk, d=d_)
                g_ = g_ * 1e6
                ax[i * 3 + j].plot(freq_, -g_, color=c)
                ax[i * 3 + j].set_xscale("log")
            ax[i * 3 + j].text(x=0.95, y=0.95, s="i" * (j + 1),
                               weight="bold", va="top", ha="right",
                               transform=ax[i * 3 + j].transAxes)
            if i != 3:
                ax[i * 3 + j].set_xticklabels([])
                ax[i * 3 + j].tick_params(bottom=False)
    for j, m_bulk in enumerate(bulks):
        ax[j].set_title(bulk_title[m_bulk], size="large", pad=5)
        
    for i in range(4):
        ax[i * 3].set_ylabel(r"$|G(i \xi)|$ (μJ·m$^{-2}$)")
    for i in range(3):
        ax[9 + i].set_xlabel(r"$\hbar \xi$ (eV)")

    cax = inset_axes(ax[-1], width="400%", height="30%",
                     loc="lower left")
    cb = add_cbar(fig, min=1, max=10, cax=cax)
    cb.ax.set_title("$d$ (nm)", pad=4)

    grid_labels(fig, [ax[i] for i in range(0, 12, 3)],
                reserved_space=(0, -0.05))
    
    savepgf(fig, img_path / "g_distance_bulk.pgf")
    mpl.rcParams["font.size"] = old_size
    
    

if __name__ == "__main__":
    plot_main()
