import numpy as np
from . import data_path, img_path
from .constants import Const
from . import equations as eqs
from scipy.integrate import cumtrapz
from scipy.io import loadmat
from scipy.optimize import fsolve
from scipy.interpolate import BivariateSpline
from helper import gridplots, grid_labels, savepgf
from helper import get_color, add_cbar, add_img_ax
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_ab(fig, ax):
    """Plot sigma vs Layer number """
    ax_a, ax_b = ax
    ND = 1e18 * 1e6
    psi_b = eqs.func_Psi_B(ND)
    Dim_M = 5
    Dim_N = 100
    sigma_M = np.linspace(-2, 2, Dim_M) * 1e13 * 1e4 * Const.q
    delta = 0.01 * 1e13 * 1e4 * Const.q
    psi_0 = Const.phi_i - psi_b - Const.phi_g0

    for i in range(Dim_M):
        eta = np.empty(Dim_N)
        ss = np.empty(Dim_N)
        guess = psi_0 / 2       # Don't know why scipy's sensitivity changes
        for j in range(Dim_N):
            sigma_GS, phi_GS, sigma_S, psi_S = eqs.fsolve_NL(j + 1,
                                                             sigma_M[i],
                                                             guess, psi_b)
            guess = psi_S
            sigma_GS_1, phi_GS_1, sigma_S_1, psi_S_1  = eqs.fsolve_NL(j + 1,
                                                                      sigma_M[i] + delta,
                                                                      guess, psi_b)

            ss[j] = sigma_S
            eta[j] = -(sigma_S_1-sigma_S) / 2 / delta
# %         guess = sigma_GS(1);
        l1, = ax_a.plot(np.arange(1, Dim_N + 1), ss / (Const.q * 1e12 * 1e4),
                     's-')
        l2, = ax_b.plot(np.arange(1, Dim_N + 1), eta,
                        's-',
                        label="{0:.1f}".format(sigma_M[i] / 1e13 / 1e4 / Const.q))

    ax_a.set_xlabel("Graphene Layer Number")
    ax_a.set_ylabel("$\\sigma_{\\mathrm{S}}$ (10$^{13}$ $e \\cdot{}$cm$^{-2}$)")
    ax_a.set_xscale("log")
    ax_a.set_ylim(-0.5, 4.0)
    ax_a.text(x=0.5, y=0.98,
              s=("$\\sigma_{\\mathrm{M}} = -2 \\times{}10^{13} "
                 "\\sim{} 2 \\times{} 10^{13}$"
                 " $e \\cdot{}$cm$^{-2}$"),
              ha="center", va="top",
              transform=ax_a.transAxes)
    ax_a.text(x=0.98, y=0.02,
              s=("$n_{0} = 10^{18}$"
                 " cm$^{-3}$"),
              ha="right", va="bottom",
              transform=ax_a.transAxes)

    ax_a.annotate(s="",
                  xy=(0.3, 0.4),
                  xytext=(0.5, 0.75),
                  xycoords="axes fraction",
                  arrowprops=dict(arrowstyle="->",
                                  connectionstyle="arc3,rad=0.3"))

    ax_b.set_xlabel("Graphene Layer Number")
    ax_b.set_ylabel("$\\eta^{\\mathrm{EF}}(N)$")
    ax_b.set_xscale("log")
    lg = ax_b.legend(loc=0)
    lg.set_title("$\\sigma_{\\mathrm{M}}$ ($10^{13}$ $e \\cdot$cm$^{-2}$)")


def plot_c(fig, ax):
    NDs = [18, 10, 2]
    Dim_M = len(NDs)
    Dim_N = 100
    sigma_M = -1 * 1e13 * 1e4 * Const.q

    for i in range(Dim_M):
        ND = 10 ** (NDs[i]) * 1e6
        psi_b = eqs.func_Psi_B(ND)
        psi_0 = Const.phi_i - psi_b - Const.phi_g0
        guess = psi_0 / 2
        for j in range(Dim_N):
            sigma_GS, phi_GS, sigma_S, psi_S = eqs.fsolve_NL(j + 1,
                                                             sigma_M,
                                                             guess, psi_b)
            guess = psi_S       # Gruadually improbe psi_S
        ax.plot(range(1, Dim_N + 1), phi_GS,
                label="$10^{{{0:d}}}$".format(NDs[i]))
    ax.set_xlim(1, Dim_N)
    ax.axhline(y=4.6, ls="--", color="k", alpha=0.5)
    ax.set_ylim(4.4, 4.85)
    ax.set_xlabel('$x$ (Layers of graphene)')
    ax.set_ylabel('$\\phi_{\\mathrm{G}}(x)$ (V)')
    lg = ax.legend(loc="lower left")
    lg.set_title("$n_0$ (cm$^{-3}$)")


def plot_main():
    h = 1.1
    fig, ax = gridplots(2, 2, r=0.90, ratio=h, span=[(0, 0, 1, 1),
                                                       (0, 1, 1, 1),
                                                       (1, 0, 1, 2)],
                        gridspec_kw=dict(height_ratios=(h,
                                                        2 - h)))
    plot_ab(fig, ax[:2])
    plot_c(fig, ax[2])
                        
    grid_labels(fig, ax)
    savepgf(fig, img_path / "ML-eta.pgf", preview=True)
    
    
if __name__ == '__main__':
    plot_main()
