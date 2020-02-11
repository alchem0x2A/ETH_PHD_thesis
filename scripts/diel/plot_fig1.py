import numpy as np
from . import data_path, img_path
from helper import gridplots, grid_labels, savepgf
from helper import get_color, add_cbar, add_img_ax
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from scipy.constants import epsilon_0, pi
from scipy.optimize import curve_fit
import os


def fit_para(L, d, eps_2D):
    return (eps_2D - 1) * d / L  + 1

def fit_vert(L, d, eps_2D):
    return 1 / (d / L * (1 / eps_2D - 1) + 1)

def eps_para(alpha, L):
    return 1 + 4 * pi * alpha / L


def eps_perp(alpha, L):
    return 1 / (1 - 4 * pi * alpha / L)

def plot_a(fig, ax):
    """Insert binary image here"""
    ax.set_axis_off()
    ax_large = inset_axes(ax,
                          width="100%",
                          height="100%",
                          loc="upper center",
                          bbox_to_anchor=(-0.15, -0.05, 1.3, 1.1),
                          bbox_transform=ax.transAxes,
                          )
    add_img_ax(ax_large, img_path / "sub_img" / "scheme_crop.png")


def plot_bc(fig, ax):
    def convert_name(s):
        dict = {"mos2": "2H-MoS$_2$",
              "mose2": "2H-MoSe$_2$",
              "mote2": "2H-MoTe$_2$",
              "ws2": "2H-WS$_2$",
              "wse2": "2H-WSe$_2$",
              "wte2": "2H-WTe$_2$",}
        return dict[s]
    # ax_b, ax_c = ax
    # ax_b_t = inset_axes(ax_b, height="60%", width="100%",
    #                     loc="upper center")
    ax_b_t, ax_c_t, ax_b_b, ax_c_b = ax
    ax_b_b.set_axis_off()
    ax_c_b.set_axis_off()
    ax_b_b = inset_axes(ax_b_b, height="115%", width="100%",
                        loc="lower center")
    ax_c_b = inset_axes(ax_c_b, height="115%", width="100%",
                        loc="lower center")
    # ax_c_t = inset_axes(ax_c, height="60%", width="100%",
    #                     loc="upper center")
    # ax_c_b = inset_axes(ax_c, height="40%", width="100%",
    #                     loc="lower center")

    root = data_path / "distance"
    g = os.walk(root)
    names = next(g)[1]

    colors = {"mos2": "#1f77b4",
              "mose2": "#ff7f0e",
              "mote2": "#2ca02c",
              "ws2": "#d62728",
              "wse2": "#9467bd",
              "wte2": "#8c564b",}

    data_all = {k: dict(para=None, perp=None) 
                for k, v in colors.items()}

    for i, item in enumerate(g):
        if "old" in item:
            continue
        for f in item[2]:
            f_path = os.path.join(item[0], f)
            if "agr" not in f_path:
                continue
            print(f_path)
            data = np.genfromtxt(f_path,
                                 delimiter=" ")
            L = data[:, 0]
            eps_SL = data[:, 1]
            # dd = {dict("para")}
            if "parallel.agr" in f_path:
                alpha_SL = L * (data[:, 1] - 1)  / (4 * pi)
                data_all[names[i]]["para"] = [L, eps_SL, alpha_SL]
                
            elif "perpendicular.agr" in f_path:
                alpha_SL = L * (data[:, 1] - 1) / (data[:, 1]) / (4 * pi)
                data_all[names[i]]["perp"] = [L, eps_SL, alpha_SL]

    for name in ["mos2", "mose2", "mote2",
                 "ws2", "wse2", "wte2"]:
        # Parallel
        L, eps_SL, alpha_SL = data_all[name]["para"]
        l, *_ = ax_b_t.plot(L, eps_SL, "o-",
                            markersize=4,
                            color=colors[name]
                )
        ax_b_b.plot(L, alpha_SL, "o-",
                    markersize=4,
                    color=l.get_c())
        # Perp
        L, eps_SL, alpha_SL = data_all[name]["perp"]
        l, *_ = ax_c_t.plot(L, eps_SL, "o-",
                            label=convert_name(name),
                            markersize=4,
                            color=colors[name])
        ax_c_b.plot(L, alpha_SL, "o-",
                    markersize=4,
                    color=l.get_c())
        

    pad = 9
    ax_b_b.set_xlabel("$L$ (\\AA{})")
    ax_b_t.set_ylabel("$\\varepsilon^{\\parallel}_{\mathrm{SL}}$", labelpad=pad)
    ax_b_b.set_ylabel("$\\alpha^{\\parallel}_{\\mathrm{2D}}/(4\\pi \\varepsilon_0)$ (\\AA{})")
    ax_b_b.set_ylim(5, 12)
    ax_b_t.set_xticklabels([])

    ax_c_b.set_xlabel("$L$ (\\AA{})")
    ax_c_t.set_ylabel("$\\varepsilon^{\\perp}_{\mathrm{SL}}$", labelpad=pad)
    ax_c_b.set_ylabel("$\\alpha^{\\perp}_{\\mathrm{2D}}/(4\\pi \\varepsilon_0)$ (\\AA{})")
    ax_c_t.set_ylim(1, 4)

    ax_c_b.set_ylim(0.3, 0.75)
    ax_c_t.set_xticklabels([])

    # ax_b_t.legend()
    ax_c_t.legend()

    

def plot_main():
    h1 = 1.1
    h2 = 0.7
    fig, ax = gridplots(3, 2, r=0.90, ratio=0.95,
                        span=[(0, 0, 1, 2),
                              (1, 0, 1, 1),
                              (1, 1, 1, 1),
                              (2, 0, 1, 1),
                              (2, 1, 1, 1)],
                        gridspec_kw=dict(height_ratios=(h1, h2,
                                                        2 - h1 - h2),
                                         hspace=0.02))
    plot_a(fig, ax[0])
    plot_bc(fig, ax[1: 5])
    grid_labels(fig, ax[0: 3], offsets=[(0, 0),
                                        (0, -0.15),
                                        (0, -0.15)])
    savepgf(fig, img_path / "fig-problem.pgf", preview=True)
    
    
if __name__ == '__main__':
    plot_main()
