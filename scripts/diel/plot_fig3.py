import numpy as np
from . import data_path, img_path
from helper import gridplots, grid_labels, savepgf
from helper import get_color, add_cbar, add_img_ax
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os


from scipy.constants import epsilon_0, pi
from scipy.optimize import curve_fit, fsolve
from pathlib import Path


def fit_para(L, d, eps_2D):
    return (eps_2D - 1) * d / L + 1


def fit_vert(L, d, eps_2D):
    return 1 / (d / L * (1 / eps_2D - 1) + 1)


def convert_name(s):
    dict = {"mos2": "2H-MoS$_2$",
            "mose2": "2H-MoSe$_2$",
            "mote2": "2H-MoTe$_2$",
            "ws2": "2H-WS$_2$",
            "wse2": "2H-WSe$_2$",
            "wte2": "2H-WTe$_2$", }
    return dict[s]


def plot_bcd(fig, ax):
    root = data_path / "distance"
    g = os.walk(root)
    names = next(g)[1]
    tags = {
        "mos2": "MoS$_{2}$",
        "mose2": "MoSe$_{2}$",
        "mote2": "MoTe$_{2}$",
        "ws2": "WS$_{2}$",
        "wse2": "WSe$_{2}$",
        "wte2": "WTe$_{2}$",
    }

    limits = {
        "mos2": (4.98, 6.15),
        "mose2": (5.60, 6.46),
        "mote2": (6.12, 6.98),
        "ws2": (5.00, 6.15),
        "wse2": (5.42, 6.49),
        "wte2": (6.33, 7.06),
    }

    raw_data_para = dict()
    raw_data_perp = dict()
    fit_all_para = dict()               # eps_2D, delta
    fit_all_perp = dict()               # eps_2D, delta

    # data_all = {k: dict(para=None, perp=None)
    # for k, v in colors.items()}

    for i, item in enumerate(g):
        if "old" in item:
            continue
        for f in item[2]:
            f_path = Path(item[0]) / f
            if "agr" not in f_path.as_posix():
                continue
            print(f_path)
            data = np.genfromtxt(f_path,
                                 delimiter=" ")
            L = data[:, 0]
            eps_SL = data[:, 1]
            if "par" in f_path.as_posix():
                raw_data_para[names[i]] = (L, eps_SL)
                param, _ = curve_fit(fit_para, L[1:], eps_SL[1:],
                                     p0=(5, 10),
                                     bounds=((0.5, 1.0),
                                             (10, 50))
                                     )
                fit_all_para[names[i]] = param
            elif "perp" in f_path.as_posix():
                raw_data_perp[names[i]] = (L, eps_SL)
                param, _ = curve_fit(fit_vert, L[1:], eps_SL[1:],
                                     p0=(5, 10),
                                     bounds=((0.5, 1.0),
                                             (10, 50))
                                     )
                fit_all_perp[names[i]] = param

    print(fit_all_para, fit_all_perp)

    # ax 0: diff
    diff = []
    name_disp = []
    color_pos = np.array([59, 117, 175]) / 255 + 0.2
    color_neg = np.array([246, 198, 158]) / 255
    colors = []
    for n, name in tags.items():
        delta_para, _ = fit_all_para[n]
        delta_perp, _ = fit_all_perp[n]
        name_disp.append("2H-" + name)
        # diff.append(delta_para / delta_perp)
        diff.append(delta_para - delta_perp)
        
        ax[0].axhline(y=0, alpha=0.8, color="k")
        if delta_para > delta_perp:
            c = color_pos
        else:
            c = color_neg
        colors.append(c)
            
    ax[0].bar(range(len(name_disp)), diff, width=0.6, color=colors)
    ax[0].set_xticks(range(len(name_disp)))
    ax[0].set_xticklabels(name_disp, rotation=-30,
                          ha="left", va="top")
    ax[0].set_ylabel("Estimation Error of $\\delta^{*}_{\\mathrm{2D}}$ (Å)")
    ax[0].text(x=0.98, y=0.98,
               s=("Error=$\\delta_{\\mathrm{2D}}^{\\mathrm{fit},"
                  " \\parallel}"
                  "- \\delta_{\\mathrm{2D}}^{\\mathrm{fit},\\perp}$"),
               ha="right", va="top",
               transform=ax[0].transAxes)
        # ""between $\\delta_{\\mathrm{2D}}^{{\parallel}, \\mathrm{fit}}$  and $\\delta_{\\mathrm{2D}}^{\\perp, \\mathrm{fit}}$ ($\\mathrm{\\AA{}}$)")

    # ax 1:
    name_disp = []
    for n, name in tags.items():
        L, eps_para = raw_data_para[n]
        L = L[6]
        eps_para = eps_para[6]
        delta_para, _ = fit_all_para[n]
        delta_perp, _ = fit_all_perp[n]
        delta_avg = (delta_para + delta_perp) / 2
        name_disp.append("2H-" + name)
        # xx = np.linspace(-0.25, 0.25, 256)
        # dd_ = xx + delta_avg
        diff = 7.5
        xx = np.linspace(-diff / 100, diff / 100, 256)
        dd_ = (1 + xx) * delta_avg
        f = dd_ / L
        eps_2D_para = (eps_para + f - 1) / f

        l, = ax[1].plot(xx * 100, eps_2D_para, label="2H-" + name)
        ax[1].plot(xx[::20] * 100, eps_2D_para[::20],
                   "o", markersize=4, color=l.get_c())

    for n, name in tags.items():
        L, eps_perp = raw_data_perp[n]
        L = L[6]
        eps_perp = eps_perp[6]
        delta_perp, _ = fit_all_perp[n]
        delta_para, _ = fit_all_para[n]
        # delta_avg = (delta_para + delta_perp) / 2
        delta_avg = delta_perp
        name_disp.append("2H-" + name)
        name_disp.append("2H-" + name)
        diff = 7.5
        xx = np.linspace(-diff / 100, diff / 100, 256)
        dd_ = (1 + xx) * delta_avg
        # xx = np.linspace(-0.25, 0.25, 256)
        # dd_ = xx + delta_avg
        f = dd_ / L
        eps_2D_perp = f / (1 / eps_perp + f - 1)
        cond = np.where((eps_2D_perp > 0) & (eps_2D_perp < 1000))
        l, = ax[2].plot(xx[cond] * 100, eps_2D_perp[cond], label="2H-" + name)
        ax[2].plot(xx[::20] * 100, eps_2D_perp[::20],
                   "o", markersize=4, color=l.get_c())

    # ax[1].set_ylim(14, 23)
    ax[1].set_xlabel("Uncertainty of $\\delta^{*}_{\\mathrm{2D}}$ (%)")
    ax[1].set_ylabel("Estimated $\\varepsilon_{\\mathrm{2D}}^{\\parallel}$")

    ax[2].set_xlim(-7.5, 7.5)
    ax[2].set_ylim(15, 500)
    ax[2].set_xlabel("Uncertainty of $\\delta^{*}_{\\mathrm{2D}}$ (%)")
    ax[2].set_ylabel("Estimated $\\varepsilon_{\\mathrm{2D}}^{\\perp}$")

    ax[2].legend()

def plot_a(fig, ax):
    ax.set_axis_off()
    H, W = (908, 580)
    ax_img = inset_axes(ax, height="100%", width="100%",
                        loc="lower center",
                        bbox_to_anchor=(0, -0.3, 1, 1.3),
                        bbox_transform=ax.transAxes)
    add_img_ax(ax_img, img_path / "sub_img" / "edm.png")
    ax_img.text(x=0.5, y=0.96,
                s="Effective Dielectric Model (EDM)",
                ha="center", va="center",
                transform=ax_img.transAxes)
    ax_img.text(x=138 / W, y=464 / H,
                s="$L$",
                ha="left", va="center",
                transform=ax_img.transAxes)
    ax_img.text(x=0.5, y=760 / H,
                s="$\\varepsilon_{\\mathrm{SL}}$",
                ha="center", va="center",
                transform=ax_img.transAxes)
    ax_img.text(x=0.5, y=0.55,
                s="$\\varepsilon_{0}$ (Vacuum)",
                ha="center", va="center",
                transform=ax_img.transAxes)
    ax_img.text(x=0.5, y=315 / H,
                s="$\\varepsilon_{\\mathrm{2D}}$",
                ha="center", va="center",
                transform=ax_img.transAxes)
    ax_img.text(x=530 / W, y=225 / H,
                s="$\\delta^{*}_{\\mathrm{2D}}$ (uncertain)",
                ha="center", va="center",
                rotation=90,
                transform=ax_img.transAxes)
    
    

def plot_main():
    fig, ax = gridplots(2, 2, r=1, ratio=1.1)
    plot_a(fig, ax[0])
    plot_bcd(fig, ax[1:])
    grid_labels(fig, ax)

    savepgf(fig, img_path / "fig-emt-uncertainty.pgf", preview=True)


if __name__ == '__main__':
    plot_main()
