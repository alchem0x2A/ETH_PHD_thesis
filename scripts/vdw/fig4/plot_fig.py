import numpy as np
from . import data_path, img_path
from .screen_bulk import get_transparency
from .slom import get_slom
from helper import gridplots, grid_labels, savepgf
from helper import get_color, add_cbar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit

d = 2                           # Default is 2 nm

candidates = [("C2", "C"),
              ("MoS2", "MoS2"),
               ("BN", "BN"),]

disp_names = {"MoS2": "2H-MoS$_{2}$",
               "BN": "hBN",
               "C2": "Graphene"}



def plot_main():
    w_cax = 0.05
    fig, ax = gridplots(2, 4,
                        span=[(0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 1, 1),
                              (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 1, 1),
                              (0, 3, 2, 1)],
                        gridspec_kw=dict(width_ratios=[4 * (1 - w_cax) / 3,] * 3 + [w_cax * 4, ]),
                        ratio=1.5)


    for i, can in enumerate(candidates):
        formula, prototype = can
        name = disp_names[formula]
        eta_bulk = get_transparency(can)
        Eg_a = eta_bulk[:, 0]
        Eg_b = eta_bulk[:, 2]
        eta = eta_bulk[:, -1]
        XX, YY, eta_slom = get_slom(can)
        ax[i].scatter(Eg_a, Eg_b, c=eta, cmap="rainbow",
                       s=36, marker="s", alpha=0.3, vmax=0.85, vmin=-0.1,
                      rasterized=True)
        ax[i].set_title(name)
        ax[i].set_xticklabels([])
        ax[i].tick_params(bottom=False, labelbottom=False)
        ax[i + 3].pcolor(XX, YY, eta_slom, cmap="rainbow", rasterized=True, vmin=-0.1, vmax=0.7)
        ax[i + 3].set_xlabel(r"$E_{\mathrm{g}}^{\mathrm{A}}$ (eV)")
        if i > 0:
            ax[i].set_yticklabels([])
            ax[i].tick_params(left=False, labelleft=False)
            ax[i + 3].set_yticklabels([])
            ax[i + 3].tick_params(left=False, labelleft=False)
        else:
            ax[i].set_ylabel(r"$E_{\mathrm{g}}^{\mathrm{B}}$ (eV)")
            ax[i + 3].set_ylabel(r"$E_{\mathrm{g}}^{\mathrm{B}}$ (eV)")
    
        

        
    print("finish loading")

    # Figure right
    ax[-1].set_axis_off()
    cax_inside = inset_axes(ax[-1], height="50%", width="45%",
                            loc="center left")
    cb1 = add_cbar(fig, None, -0.1, 0.85,  cax=cax_inside,  shrink=0.6)
    cb1.ax.set_title(r"$\eta^{\mathrm{vdW}}$ ", pad=5, ha="center")

    
    grid_labels(fig, [ax[0], ax[3]], reserved_space=(0, -0.05),)
    labels = ["i", "ii", "iii"]
    for i in range(6):
        ii = i % 3
        ax_ = ax[i]
        ax_.set_xlim(0, 12)
        ax_.set_ylim(0, 12)
        ax_.set_aspect("equal")
        ax_.text(x=0.05, y=0.98,
                 s=labels[ii], weight="bold",
                 ha="left", va="top",
                 transform=ax_.transAxes)
    savepgf(fig, img_path / "bulk_compare_slom.pgf")
    

if __name__ == '__main__':
    plot_main()
