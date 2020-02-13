import numpy as np
from . import data_path, img_path
from helper import gridplots, grid_labels, savepgf
from helper import get_color, add_cbar, add_img_ax
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


import json
from scipy.constants import pi, k, epsilon_0, e

data = json.load(open(data_path / "HSE-data/all_data.json", "r"))


def model_alpha(x, ratio=0.1657):
    return 1 / (ratio * x - 0.0243)

def plot_img(fig, ax):
    ax.set_axis_off()
    ax_img = inset_axes(ax, height="100%", width="100%",
                        bbox_to_anchor=(-0.1, 0, 1.2, 1),
                        bbox_transform=ax.transAxes)
    add_img_ax(ax_img, img_path / "sub_img" / "ellip.png")
    ax_img.text(x=0.5, y=0.7,
                s="$r_{0}^{\\perp} = \\alpha_{\\mathrm{2D}}^{\\perp}/2\\varepsilon_0$",
                ha="center", va="center",
                transform=ax.transAxes)
    ax_img.text(x=0.90, y=0.5,
                s="$r_0^{\\parallel} = \\alpha_{\\mathrm{2D}}^{\\parallel}/2\\varepsilon_0$",
                ha="center", va="center",
                transform=ax.transAxes)
    


def plot_ax(fig, ax):
    ax1, ax2 = ax
    maters = []
    alpha = []
    Eg = []

    names = ["2H-MoS2", "2H-MoSe2", "2H-MoTe2",
             "2H-WS2", "2H-WSe2", "2H-WTe2"]
    xx = np.arange(6)

    r_para = [40.0, 45.5, 58.1, 35.7, 41.0, 53.5]
    r_perp = [2.50, 2.72, 3.07, 2.46, 2.71, 3.17]


    ax1.bar(xx, r_para, width=0.5, color="#559dc9")
    ax2.bar(xx, r_perp, width=0.5, color="#d9ad66")

    ax1.set_ylabel("$r_0^{\\parallel}$ (Å)")
    ax2.set_ylabel("$r_0^{\\perp}$ (Å)")

    ax1.set_ylim(0, 70)
    ax2.set_ylim(0, 6)

    ax1.set_xticks(xx)
    ax2.set_xticks(xx)

    ax1.text(x=0.98, y=0.98,
             s="In-plane",
             ha="right", va="top",
             transform=ax1.transAxes)

    ax2.text(x=0.98, y=0.98,
             s="Out-of-plane",
             ha="right", va="top",
             transform=ax2.transAxes)
    mnames = [n[0] + n[1:].replace("2", "$_{2}$") for n in names]
    ax1.set_xticklabels(mnames, rotation=-30, va="top", ha="left")
    ax2.set_xticklabels(mnames, rotation=-30, va="top", ha="left")




def plot_main():
    h = 1.1
    fig, ax = gridplots(2, 2, r=0.9, ratio=(2 - h) * 1.1,
                        span=[(0, 0, 1, 2),
                              (1, 0, 1, 1),
                              (1, 1, 1, 1)],
                        gridspec_kw=dict(height_ratios=(h,
                                                        2 - h)))
    grid_labels(fig, ax)
    plot_img(fig, ax[0])
    plot_ax(fig, ax[1:])
    savepgf(fig, img_path / "fig-ellipsoid.pgf", preview=True)


if __name__ == '__main__':
    plot_main()
