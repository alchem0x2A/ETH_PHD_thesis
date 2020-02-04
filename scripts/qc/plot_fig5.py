import numpy as np
from . import data_path, img_path
from .constants import Const
from . import equations as eqs
from scipy.optimize import fsolve
from scipy.interpolate import BivariateSpline
from helper import gridplots, grid_labels, savepgf
from helper import get_color, add_cbar, add_img_ax
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Plot the figures for fig 3
filename_temp = "{0}-range_{1}_{2}.dat"
ranges = ["large", "small"]
maters = ["Si", "G"]
states = ['p-doped', 'Intrinsic', 'n-doped']
"large-range_G_intrinsic.dat"

def get_data(rng, mater, state):
    assert rng in ranges
    assert mater in maters
    assert state in states
    fn = data_path / "abinit" / filename_temp.format(rng, mater, state)
    data = np.genfromtxt(fn)
    return data[:, 0], data[:, 1]


def plot_a(fig, ax):
    """Plot sigma_G vs sigma_M"""
    # Large scale
    x_large = np.linspace(-20, 20, 256)
    for i, state in enumerate(states):
        x, y = get_data("large", "G", state)
        p = np.polyfit(x, y, deg=1)
        y_large = np.poly1d(p)(x_large)
        l, = ax.plot(x, y, "s")
        l2, = ax.plot(x_large, y_large, color=l.get_c())
    ax.set_ylim(-12, 6)
    ax.set_xlabel("$\\sigma_{\\mathrm{M}}$ (10$^{13}$ $e \cdot{}$cm$^{-2}$)")
    ax.set_ylabel("$\\sigma_{\\mathrm{G}}$ (10$^{13}$ $e \cdot{}$cm$^{-2}$)")

    # Small
    ax_small = inset_axes(ax, width="100%", height="100%", loc="lower left",
                          bbox_to_anchor=(0.05, 0.05, 0.4, 0.4),
                          bbox_transform=ax.transAxes)
    for i, state in enumerate(states):
        x, y = get_data("small", "G", state)
        l, = ax_small.plot(x, y, "-s", markersize=2, linewidth=0.5)
    ax_small.tick_params(direction='in', length=2, width=0.5,
                         pad=0.5, labelsize="x-small")
    ax_small.set_ylim(-2, 2)
    ax_small.set_yticks([-2, 0, 2])
    # ax_small.set_xlabel("$\\sigma_{\\mathrm{M}}$ (10$^{13}$ $e \cdot{}$cm$^{-2}$)",
                        # size="x-small")
    # ax_small.set_ylabel("$\\sigma_{\\mathrm{G}}$ (10$^{13}$ $e \cdot{}$cm$^{-2}$)",
                        # size="x-small")
    

def plot_b(fig, ax):
    """Plot sigma_G vs sigma_M"""
    # Large scale
    x_large = np.linspace(-20, 20, 256)
    for i, state in enumerate(states):
        x, y = get_data("large", "Si", state)
        p = np.polyfit(x, y, deg=1)
        y_large = np.poly1d(p)(x_large)
        l, = ax.plot(x, y, "s")
        l2, = ax.plot(x_large, y_large, color=l.get_c(),
                      label="{0} $\\eta^{{\\mathrm{{EF}}}} = {1:.3f}$".format(state,
                                                                              -p[0]))
    ax.set_xlabel("$\\sigma_{\\mathrm{M}}$ (10$^{13}$ $e \cdot{}$cm$^{-2}$)")
    ax.set_ylabel("$\\sigma_{\\mathrm{S}}$ (10$^{13}$ $e \cdot{}$cm$^{-2}$)")

        # Small
    ax_small = inset_axes(ax, width="100%", height="100%", loc="lower left",
                          bbox_to_anchor=(0.05, 0.05, 0.4, 0.4),
                          bbox_transform=ax.transAxes)
    for i, state in enumerate(states):
        x, y = get_data("small", "Si", state)
        l, = ax_small.plot(x, y, "-s", markersize=2, linewidth=0.5)
    ax_small.tick_params(direction='in', length=2,
                         width=0.5, pad=0.5, labelsize="x-small")
    ax_small.set_ylim(-3, 3)
    # ax_small.set_xlabel("$\\sigma_{\\mathrm{M}}$ (10$^{13}$ $e \cdot{}$cm$^{-2}$)",
                        # size="x-small")
    # ax_small.set_ylabel("$\\sigma_{\\mathrm{S}}$ (10$^{13}$ $e \cdot{}$cm$^{-2}$)",
                        # size="x-small")
    ax.set_ylim(-23, 15)
    ax.legend(loc="upper right",
              handlelength=1,
              fontsize="x-small",
              markerscale=0.5)

def plot_c(fig, ax):
    # ax.set_xticks([]); ax.set_yticks([])
    ax.set_axis_off()
    ax_top = inset_axes(ax, width="100%", height="100%", loc="lower left",
                        bbox_to_anchor=(-0.15, 0.6, 1.15, 0.4),
                        bbox_transform=ax.transAxes,
                        borderpad=0)
    # ax_top.set_xticks([]); ax_top.set_yticks([])

    ax_bottom = inset_axes(ax, width="100%", height="100%", loc="lower left",
                           # bbox_to_anchor=(-0.1, 0, 1.1, 1),
                           bbox_to_anchor=(-0.15, 0.0, 1.15, 0.4),
                           bbox_transform=ax.transAxes,
                           borderpad=0)
                           # bbox_to_anchor=(0.5, 0, 1.0, 0.3),
    # ax_bottom.set_xticks([]); ax_bottom.set_yticks([])
    add_img_ax(ax_top, img_path / "sub_img" / "pos.png")
    add_img_ax(ax_bottom, img_path / "sub_img" / "neg.png")
    ax_top.text(x=0.5, y=0.5,
                ha="center",
                s="$\\sigma_{\\mathrm{M}} = -0.98 \\times 10^{13}$ $e \cdot$cm$^{-2}$",
                color="#3939f4",
                transform=ax.transAxes)
    
    ax.text(x=0.5, y=-0.1,
            s="$\\sigma_{\\mathrm{M}} = 0.98 \\times 10^{13}$ $e \cdot$cm$^{-2}$",
            ha="center",
            color="#e62828",
            transform=ax.transAxes)


   


def plot_main():
    w = 1.15
    fig, ax = gridplots(1, 3, r=1, ratio=3 / w,
                        gridspec_kw=dict(width_ratios=(w, w, 3 - 2 * w)))
    plot_a(fig, ax[0])
    plot_b(fig, ax[1])
    plot_c(fig, ax[2])
    grid_labels(fig, ax)
    savepgf(fig, img_path / "first_principles.pgf")
    

if __name__ == '__main__':
    plot_main()
