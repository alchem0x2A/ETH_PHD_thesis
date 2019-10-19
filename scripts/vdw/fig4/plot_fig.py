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
                       s=26, marker="s", alpha=0.3, vmax=0.85, vmin=-0.1,
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
    # subfig a
    # i = 0                       # Vacuum case
    # ax[0].plot(ds / 1e-10, gs[0] * 1000, color="grey")
    # j = 10
    # ax[0].text(x=ds[j] / 1e-10 + 0.5, y=gs[0][j] * 1000 * 0.95, s=r"$\Phi^{0}$", ha="left", va="top", color="grey")
    # ax[1].plot(freq_matsu, -1e6 * G[0], color="grey")
    # ax[1].text(x=freq_matsu[j] + 3, y=-1e6 * G[0][j] * 0.95, s=r"←$G^{0}$", ha="left", va="top", color="grey")
    # for i in range(1, len(names)):
    #     eg = Eg[i]
    #     n = names[i]
    #     if n not in candidates:
    #         a_ = 0.1
    #         lw = 0.8
    #     else:
    #         a_ = 1
    #         lw = 1
        
    #     c = get_color(eg, min=0, max=np.max(Eg[1:]),)
    #     ax[0].plot(ds / 1e-10, gs[i] * 1000, color=c, alpha=a_, lw=lw)
    #     ax[1].plot(freq_matsu, -1e6 * G[i], color=c, alpha=a_, lw=lw)
    #     ax[2].plot(freq_matsu, transparency[i], color=c, alpha=a_, lw=lw)
    #     if n in candidates:
    #         j = 1
    #         ax[1].text(x=freq_matsu[j] + 0.05, y= -1e6 * G[i][j] + 0.04,
    #                    s=disp_names[n], ha="left", va="bottom", color=c)

    #         ax[2].text(x=freq_matsu[j] + 0.05, y= transparency[i][j] + 0.04,
    #                    s=disp_names[n], ha="left", va="bottom", color=c)

    # # ax[0].set_ylim(0.9, 20)
    # # ax[0].set_yscale("log")
    # # ax[1].set_ylim(0.9, 4)
    # ax[1].set_xscale("log")
    # ax[1].set_ylim(0, 1.5)
    # ax[2].set_xscale("log")
    # ax[2].set_ylim(0, 0.88)
    # ax[2].axhline(y=1, ls="--", color="grey", alpha=0.6)
    # ax[0].set_xlabel(r"$d$ (\AA{})")
    # for i in (1, 2):
    #     ax[i].set_xlabel(r"$\hbar \xi$ (eV)")
    # ax[0].set_ylabel(r"$\Phi^{\mathrm{AmB}}$ (μJ$\cdot{}$cm$^{-2}$)")
    # ax[1].set_ylabel(r"$G(i \xi)$ (μJ$\cdot{}$cm$^{-2}$)")
    # ax[2].set_ylabel(r"$\tau (i \xi)$")

    # Fix lim
    # ax[0].set_xticklabels([])
    # ax[0].tick_params(bottom=False, labelbottom=False, axis="x", direction="in")

    # Figure right
    ax[-1].set_axis_off()
    cax_inside = inset_axes(ax[-1], height="50%", width="45%",
                            loc="center left")
    cb1 = add_cbar(fig, None, -0.1, 0.85,  cax=cax_inside,  shrink=0.6)
    cb1.ax.set_title(r"$\eta^{\mathrm{vdW}}$ ", pad=5, ha="center")

    # cb2 = add_cbar(fig, ax[5], -0.1, 0.7,  shrink=0.6)
    # cb2.ax.set_title("$E_{\mathrm{g}}$ (eV)", pad=1)

    # label

    # Hamaker constants

    # def fit_fun(x, a):
        # return -a * x ** -2

    # h0, *_ = curve_fit(fit_fun, ds, gs[0])

    # res = []

    # for i in range(1, len(names)):
        # h, *_  = curve_fit(fit_fun, ds[ds < 5e-9], gs[i][ds < 5e-9])
        # hamaker = h / h0
        # res.append([Eg[i], hamaker[0]])
    # res = np.array(res)
    # print("res", res)

    # Plot in the insets
    # ax_in1 = inset_axes(ax[0], height="80%", width="80%",
                        # bbox_to_anchor=(0.25, 0.2, 0.75, 0.75),
                        # bbox_transform=ax[0].transAxes,
                        # loc="lower right")
    # ax_in1.tick_params(labelsize="small")
    # ax_in1.set_xlabel(r"$E_{\mathrm{g}}$ (eV)", size="small", labelpad=-0.5)
    # ax_in1.set_ylabel(r"$A_{\mathrm{eff}} / A_{\mathrm{eff}}^{0}$",
                      # size="small", labelpad=-0.5)
    # ax_in1.scatter(res[:, 0], res[:, 1], s=3.5 ** 2, c=res[:, 0],
                   # c=[get_color(r, min=0, max=np.max(Eg[1:])) \
                      # for r in res[:, 0]],
                   # cmap="rainbow", alpha=0.2)

    # ax_in1.scatter(Eg, trans_para, s=10, color="#3287a8",
                   # marker="^", alpha=0.5)

    # ax_in2 = inset_axes(ax[1], height="58%", width="35%",
                        # loc="upper right")

    # ax_in2.scatter(Eg, trans_perp, s=10, color="#a87e31",
                   # marker="o",
                   # alpha=0.5)
    # ax_in2.tick_params(labelsize="small")
    # ax_in2.set_xlabel(r"$E_{\mathrm{g}}$ (eV)", size="small", labelpad=-1)
    # ax_in2.set_ylabel(r"$\hbar \xi_{\mathrm{tr}}^{\perp}$ (eV)",
                      # size="small", labelpad=-1)
    
    grid_labels(fig, [ax[0], ax[3]], reserved_space=(0, -0.05),)
    labels = ["i", "ii", "iii"]
    for i in range(6):
        ii = i % 3
        ax_ = ax[i]
        ax_.set_xlim(0, 12)
        ax_.set_xlim(0, 12)
        ax_.set_aspect("equal")
        ax_.text(x=0.05, y=0.98,
                 s=labels[ii], weight="bold",
                 ha="left", va="top",
                 transform=ax_.transAxes)
    savepgf(fig, img_path / "bulk_compare_slom.pgf")
    

if __name__ == '__main__':
    plot_main()
