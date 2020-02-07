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

# Plot the figures for fig 6
# import matplotlib
# matplotlib.rcdefaults()


def plot_a(fig, ax, dim=256):
    """Plot the surface charge change with bias
    Note for n-type Semiconductor, forward bias is negative"""

    ND = 1e16 * 1e6
    Psi_b0 = eqs.func_Psi_B(ND)

    V_bias = np.linspace(-2, 2, dim)
    NQ_gate = np.linspace(-2, 2, 9) * 1e13
    Q_gate = NQ_gate * Const.q * 1e4

    for j in range(Q_gate.shape[0]):
        psi_s_b = np.empty(dim)
        psi_s = np.empty(dim)
        for i in range(dim):
            psi_s_b[i] = fsolve(lambda Psi:
                                eqs.solve_psi_s(Psi,
                                                Psi_b0,
                                                Q_gate[j],
                                                -V_bias[i]),
                                -0.5 * V_bias[i] - 0.5)[0]
            psi_s[i] = fsolve(lambda Psi:
                              eqs.solve_psi_s(Psi,
                                              Psi_b0,
                                              Q_gate[j], 0), 0)[0] - V_bias[i]
        ax.plot(V_bias, psi_s_b)

    ax.set_xlabel("$V_{\\mathrm{b}}$ (V)")
    ax.set_ylabel("$\\psi_{\\mathrm{0}}$ (V)")
    ax.set_ylim(-1.3, 0.55)
    ax.text(x=0.5, y=0.98,
            s=("$\\sigma_{\\mathrm{M}} = -2 \\times{} 10^{13} \\sim{}"
               "2 \\times{} 10^{13}$ $e \\cdot{}$cm$^{-2}$"),
            ha="center", va="top",
            transform=ax.transAxes)
    ax.text(x=0.02, y=0.02,
            s="Forward bias",
            ha="left", va="bottom",
            transform=ax.transAxes)
    ax.text(x=0.98, y=0.02,
            s="Reverse bias",
            ha="right", va="bottom",
            transform=ax.transAxes)
    ax.fill_betweenx(y=ax.get_ylim(),
                     x1=-2, x2=0, color="red", alpha=0.2, linewidth=0,
                     zorder=0)
    ax.fill_betweenx(y=ax.get_ylim(),
                     x1=0, x2=2, color="blue", alpha=0.2, linewidth=0,
                     zorder=0)
    ax.annotate(s="",
                xy=(0.75, 0.4),
                xytext=(0.4, 0.2),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle="->"))


def plot_b(fig, ax, dim=256):
    """Plot the Fermi level of graphene vs. Q_gate under bias;
    %Note for n-type Semiconductor, forward bias is negative"""

    ND = 1e16 * 1e6
    Psi_b0 = eqs.func_Psi_B(ND)

    V_bias = np.linspace(-0.75, 0.75, 7)
    NQ_gate = np.linspace(-2, 2, dim) * 1e13
    Q_gate = NQ_gate * Const.q * 1e4

    for j in range(V_bias.shape[0]):
        psi_s_b = np.empty(dim)
        Delta_EF_gr = np.empty(dim)
        for i in range(dim):
            psi_s_b[i] = fsolve(lambda Psi:
                                eqs.solve_psi_s(Psi, Psi_b0,
                                                Q_gate[i], -V_bias[j]), 0)[0]
            Delta_EF_gr[i] = - eqs.func_delta_phi_g(eqs.func_q_g(Q_gate[i],
                                                    eqs.func_E_psi(psi_s_b[i],
                                                                   Psi_b0)))
        ax.plot(NQ_gate / 1e13, Delta_EF_gr)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel("$\\sigma_{\\mathrm{M}}$ ($10^{13}$ $e\\cdot{}$cm$^{-2}$)")
    ax.set_ylabel("$\\Delta E_{\\mathrm{F, G}}$ (eV)")

    ax.text(x=0.5, y=0.98,
            s="$V_{\\mathrm{b}} = -0.75 \\sim{} 0.75 V$",
            ha="center", va="top", transform=ax.transAxes)
    ax.annotate(s="",
                xy=(0.4, 0.75),
                xytext=(0.6, 0.25),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle="->"))
    


def plot_main():
    fig, ax = gridplots(1, 2, r=0.90, ratio=2)
    plot_a(fig, ax[0])
    plot_b(fig, ax[1])

    grid_labels(fig, ax, offsets=[(0, 0), (0.02, 0)])
    savepgf(fig, img_path / "bias-effect-res.pgf", preview=True)


if __name__ == '__main__':
    plot_main()
