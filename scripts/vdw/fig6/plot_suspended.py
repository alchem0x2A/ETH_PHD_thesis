from . import data_path, img_path
from helper import grid_labels, gridplots, add_img_ax, savepgf
from .get_repulsion import get_energy, get_energy_two_body
from ..utils.eps_tools import get_index
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np


maters_m = [("C2", "C"),
            ("BN", "BN")]

disp_names = { "BN": "hBN",
               "C2": "Gr"}

def plot_main():
    # plot figure!
    fig, ax = gridplots(1, 2, ratio=2, r=0.98)

    # Set limit
    ax[0].set_ylim(-0.2, 0.1)
    ax[0].axhspan(ymin=0, ymax=0.1, color="red", alpha=0.05)
    ax[0].axhspan(ymin=-0.2, ymax=0, color="green", alpha=0.05)
    ax[1].set_ylim(-1.6, 0.8)
    ax[1].axhspan(ymin=0, ymax=0.8, color="red", alpha=0.05)
    ax[1].axhspan(ymin=-1.6, ymax=0, color="green", alpha=0.05)
    
    # First part the water
    for i, m in enumerate(maters_m):
        ind_a = get_index("H2O-exp", "bulk")
        ind_m = get_index(m, "2D")
        name = disp_names[m[0]]
        dd, Phi_amb = get_energy(ind_m, ind_a)
        dd, Phi_mb = get_energy_two_body(ind_m, ind_a)
        Phi_tot = Phi_amb + Phi_mb
        # if i == 0:
            # l1, = ax[0].plot(dd / 1e-10, Phi_amb * 1000, "^",
                            # label=r"$\Phi^{\mathrm{AmB}}$",
                             # alpha=0.3)
            # l2, = ax[0].plot(dd / 1e-10, Phi_mb * 1000, "s",
                       # label=r"$\Phi^{\mathrm{2D-B}}$",
                       # color=l1.get_c(), alpha=0.3)
            # l3, = ax[0].plot(dd / 1e-10, Phi_tot * 1000, "o",
                       # label=r"$\Phi^{\mathrm{tot}}$", color=l1.get_c())
        # else:
        l1, = ax[0].plot(dd / 1e-10, Phi_amb * 1000, "^",
                         alpha=0.3)
        l2, = ax[0].plot(dd / 1e-10, Phi_mb * 1000, "s",
                   color=l1.get_c(), alpha=0.3)
        l3, = ax[0].plot(dd / 1e-10, Phi_tot * 1000, "o",
                         color=l1.get_c())
            
        
        ind_a = get_index("Au-exp", "bulk")
    # for i, m in enumerate(maters_m):
        # ind_m = get_index(m, "2D")
        # name = disp_names[m[0]]
        dd, Phi_amb = get_energy(ind_m, ind_a)
        dd, Phi_mb = get_energy_two_body(ind_m, ind_a)
        Phi_tot = Phi_amb + Phi_mb
        l, = ax[1].plot(dd / 1e-10,
                        Phi_amb * 1000, "^", label="{}".format(name),
                        alpha=0.3)
        ax[1].plot(dd / 1e-10, Phi_mb * 1000, "s",
                   label="{} Two-body only".format(name),
                   color=l.get_c(), alpha=0.3)
        ax[1].plot(dd / 1e-10, Phi_tot * 1000, "o",
                   label="{} Total".format(name), color=l.get_c())
        if i == 1:
            x = 15; y = 0.55
        else:
            x = 25;  y = 0.15
        ax[1].text(x=x, y=y, s=name,
                   ha="left", color=l.get_c())




    # Treat legend
    lines = ax[1].get_lines()
    leg = ax[1].legend([(lines[0], lines[3]),
                        (lines[1], lines[4]),
                        (lines[2], lines[5])],
                       [r"$\Phi^{\mathrm{AmB}}$",
                        r"$\Phi^{\mathrm{2D-B}}$",
                        r"$\Phi^{\mathrm{tot}}$"],
                       handler_map={tuple: HandlerTuple(ndivide=None)},
                       loc="upper right")

    # Add text

    # Style change
    for ax_ in ax:
        ax_.axhline(y=0, ls="--", color="grey")
    for ax_ in ax:
        ax_.set_xlabel(r"$d$ (\AA{})")
        ax_.set_ylabel(r"$\Phi^{\mathrm{vdW}}$ (mJ·m$^{-2}$)")

    # Add water cube
    inset_left = inset_axes(ax[0],
                            width="100%", height="100%",
                            loc="lower center",
                            bbox_to_anchor=(0.4, 0, 0.55, 0.5),
                            bbox_transform=ax[0].transAxes)
    add_img_ax(inset_left, img_path / "3D" / "water_cube.png")

    inset_right = inset_axes(ax[1],
                            width="100%", height="100%",
                            loc="lower center",
                            bbox_to_anchor=(0.3, 0, 0.65, 0.5),
                            bbox_transform=ax[1].transAxes)
    add_img_ax(inset_right, img_path / "3D" / "gold_cluster.png")
    # zoom in!
    # ax[1].set_ylim(-1, 0.1)
    # ax[3].set_ylim(-1, 0.6)
    grid_labels(fig, ax, reserved_space=(0, 0))
    savepgf(fig, img_path / "suspend_energy.pgf")

if __name__ == "__main__":
    plot_main()
