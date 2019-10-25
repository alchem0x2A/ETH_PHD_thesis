import numpy as np
from . import data_path, img_path
from .transition import get_transition_freq, freq_matsu
from .gm_ratio import get_gm2
from helper import gridplots, grid_labels, savepgf
from helper import get_color, add_cbar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

d = 2                           # Default is 2 nm

candidates = ("MoS2-MoS2",
               "BN-BN",
               "C2-C")

disp_names = {"MoS2-MoS2": "2H-MoS$_{2}$",
               "BN-BN": "hBN",
               "C2-C": "Gr"}



def plot_main():
    fig, ax = gridplots(2, 2, span=[(0, 0, 1, 1), (1, 0, 1, 1),
                                    (0, 1, 2, 1)],
                        ratio=2)
    
    # plot eps
    names, Eg, eps_para, eps_perp, gm2 = get_gm2(d)
    for i in range(len(names)):
        eg = Eg[i]
        n = names[i]
        if n not in candidates:
            a_ = 0.1
            lw = 1
        else:
            a_ = 1
            lw = 1.25
        
        c = get_color(eg, min=0, max=np.max(Eg),)
        ax[0].plot(freq_matsu, eps_para[i], color=c, alpha=a_, lw=lw)
        ax[1].plot(freq_matsu, eps_perp[i], color=c, alpha=a_, lw=lw)
        ax[2].plot(freq_matsu, 1 / gm2[i], color=c, alpha=a_, lw=lw)
        if n in candidates:
            j = 2
            ax[2].text(x=freq_matsu[j], y=(1 / gm2[i])[j]+0.05,
                       s=disp_names[n], ha="left", va="bottom", color=c)

    ax[0].set_ylim(0.9, 20)
    ax[0].set_yscale("log")
    ax[1].set_ylim(0.9, 4)
    # ax[1].set_yscale("log")
    for ax_ in ax:
        ax_.set_xscale("log")
        ax_.axhline(y=1, ls="--", color="grey", alpha=0.6)
    for i in (1, 2):
        ax[i].set_xlabel(r"$\hbar \xi$ (eV)")
    ax[0].set_ylabel(r"$\varepsilon_{\mathrm{m}}^{\parallel} (i \xi)$")
    ax[1].set_ylabel(r"$\varepsilon_{\mathrm{m}}^{\perp} (i \xi)$")
    ax[2].set_ylabel(r"$1/g_{\mathrm{m}} (i \xi)$")

    # Fix lim
    ax[0].set_xticklabels([])
    ax[0].tick_params(bottom=False, labelbottom=False, axis="x", direction="in")

    # Figure right
    cax_inside = inset_axes(ax[2], height="70%", width="50%",
                            bbox_to_anchor=(0.78, 0.05, 0.1, 0.80),
                            bbox_transform=ax[2].transAxes,
                            loc="lower left")
    cb = add_cbar(fig, ax[2], 0, np.max(Eg), cax=cax_inside, shrink=0.5)
    cb.ax.set_title("$E_{\mathrm{g}}$ (eV)", pad=1)

    # label
    grid_labels(fig, [ax[0], ax[2]], reserved_space=(0, -0.05),)


    # Plot in the insets
    names, Eg, trans_para, trans_perp = get_transition_freq(d)
    ax_in1 = inset_axes(ax[0], height="58%", width="35%",
                        loc="upper right")
    ax_in1.tick_params(labelsize="small")
    ax_in1.set_xlabel(r"$E_{\mathrm{g}}^{\mathrm{2D}}$ (eV)", size="small", labelpad=-1)
    ax_in1.set_ylabel(r"$\hbar \xi_{\mathrm{tr}}^{\parallel}$ (eV)",
                      size="small", labelpad=-1)

    ax_in1.scatter(Eg, trans_para, s=10, color="#3287a8",
                   marker="^", alpha=0.4, edgecolor=None)

    ax_in2 = inset_axes(ax[1], height="58%", width="35%",
                        loc="upper right")

    ax_in2.scatter(Eg, trans_perp, s=10, color="#a87e31",
                   marker="o",
                   alpha=0.5)
    ax_in2.tick_params(labelsize="small")
    ax_in2.set_xlabel(r"$E_{\mathrm{g}}^{\mathrm{2D}}$ (eV)", size="small", labelpad=-1)
    ax_in2.set_ylabel(r"$\hbar \xi_{\mathrm{tr}}^{\perp}$ (eV)",
                      size="small", labelpad=-1)
    
    savepgf(fig, img_path / "eps_and_gm2.pgf")
    

if __name__ == '__main__':
    plot_main()
