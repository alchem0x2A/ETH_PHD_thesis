import numpy as np
from . import data_path, img_path
from .constants import Const
from . import equations as eqs
from scipy.optimize import fsolve
from helper import gridplots, grid_labels, savepgf
from helper import get_color, add_cbar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Plot the figures for fig 2


def subfig_a(ax, dim=256):
    # Without bias and gate
    psi_B = -Const.E_g / 2 + np.linspace(0, 1, dim) * Const.E_g
    n, p = eqs.func_np(0, psi_B)
    ND = n / 1e6
    psi = np.array([fsolve(lambda Psi: eqs.solve_psi_s(Psi, pb_, 0, 0), 0)[0]
                    for pb_ in psi_B])
    qg = eqs.func_q_g(0, eqs.func_E_psi(psi, psi_B))
    Delta_FE_graphene = -eqs.func_delta_phi_g(qg)
    print(psi.shape, psi_B.shape, qg.shape, Delta_FE_graphene.shape)  # should be all (dim,)
    ax.plot(ND, Delta_FE_graphene)
    ax.set_xscale("log")
    ax.set_xlabel("$n_0$ (cm$^{-3}$)")
    ax.set_ylabel("$\\Delta E_{\\mathrm{F, G}}$ (eV)")
    ax.set_xlim(1e0, 1e20)
    ax.set_xticks(np.power(10, np.arange(0, 22, 5)))
    ax.set_xticklabels(["$10^{{{0}}}$".format(i) for i in (0, 5, 10, 15, 20)])
    ax.set_ylim(-0.4, 0.4)
    # Line markers
    ax.axhline(y=0, color="k", ls="--", alpha=0.6)
    ax.axvline(x=Const.n_i / 1e6, color="k", ls="--", alpha=0.6)
    ax.text(x=Const.n_i / 1e6, y=0.3,
            s="←$n_0=n_{\\mathrm{i}}$", ha="left")


def subfig_b(ax, dim=256):
    ND = np.array([1e2, Const.n_i, 1e18])
    NQ_gate = np.linspace(-2, 2, dim) * 1e13
    Q_gate = NQ_gate * Const.q * 1e4  # SI unit
    ax.plot(NQ_gate / 1e13, -eqs.func_delta_phi_g(-Q_gate), color="k")
    for i, nd in enumerate(ND):
        Delta_EF = []
        psi_b = eqs.func_Psi_B(nd * 1e6)
        psi_s = np.array([fsolve(lambda Psi: eqs.solve_psi_s(Psi, psi_b,
                                                             qgate, 0), 0)[0]
                          for qgate in Q_gate])
        Delta_EF = -eqs.func_delta_phi_g(eqs.func_q_g(Q_gate,
                                                      eqs.func_E_psi(psi_s, psi_b)))
        ax.plot(NQ_gate / 1e13, Delta_EF)

    ax.axhline(y=0, color="k", ls="--", alpha=0.6)
    ax.axvline(x=0, color="k", ls="--", alpha=0.6)
    ax.set_xlabel("$\\sigma_{\\mathrm{M}}$ ($10^{13}$ $e \cdot$cm$^{-2}$)")
    ax.set_ylabel("$\\Delta E_{\\mathrm{F, G}}$ (eV)")
    ax.legend(["MOG",
               "p-type MOGS",
               "Intrinsic MOGS",
               "n-type MOGS"], loc=0)


def subfig_c(ax, dim=256):
    NQ_gate = np.linspace(-2, 2, dim) * 1e13
    Q_gate = NQ_gate * 1e4 * Const.q
    ND = 1e16 * 1e6
    psi_b = eqs.func_Psi_B(ND)
    # Potential with and without graphene
    Psi_with = np.array([fsolve(lambda Psi: eqs.solve_psi_s(Psi, psi_b,
                                                            qgate, 0), 0)[0]
                         for qgate in Q_gate])
    Psi_no = np.array([fsolve(lambda Psi: eqs.solve_psi_s_no_graphene(Psi,
                                                                      psi_b,
                                                                      qgate),
                              psi_b)[0]
                       for i, qgate in enumerate(Q_gate)])

    # Regions
    ax.set_ylim(-1.3, 0.4)
    print(ax.get_ylim())
    X = [-2, 0, 0, -2]
    Y = [ax.get_ylim()[0], ax.get_ylim()[0],
         -2 * psi_b, -2 * psi_b]
    print(X, Y)
    ax.fill(X, Y, color="red", alpha=0.3, lw=0)

    X = [0, 2, 2, 0]
    Y = [0, 0,
         ax.get_ylim()[1], ax.get_ylim()[1]]
    print(X, Y)
    ax.fill(X, Y, color="blue", alpha=0.3, lw=0)
    
    ax.axhline(y=-psi_b * 2, color="k", ls="--", alpha=0.6)
    ax.axhline(y=-psi_b - Const.E_g / 2, color="k", ls="--", alpha=0.6)
    ax.axhline(y=-psi_b + Const.E_g / 2, color="k", ls="--", alpha=0.6)
    ax.axhline(y=0, color="k", ls="--", alpha=0.6)

    l_with, = ax.plot(NQ_gate / 1e13, Psi_with)
    l_no, = ax.plot(NQ_gate / 1e13, Psi_no)

    ax.text(x=0.05, y=-0.4, s="←MOGS",
            ha="left", color=l_with.get_c())
    ax.text(x=-0.05, y=-0.2, s="MOS→",
            ha="right", color=l_no.get_c())
    ax.text(x=0.05, y=0.38,
            s="Accumulation",
            ha="left", va="top")
    ax.text(x=-1.95, y=-1.25,
            s="Strong Inversion",
            ha="left", va="bottom")
    ax.text(x=1.85, y=-1.25,
            s="$N_{\\mathrm{D}} = 10^{16}$ cm$^{-3}$",
            ha="right", va="bottom")

    ax.set_xlabel("$\\sigma_{\\mathrm{M}}$ (10$^{13}$ $e \cdot$cm$^{-3}$)")
    ax.set_ylabel("$\\psi_0$ (V)")

    # Labeling for E levels
    ax.text(x=2.1, y=-psi_b + Const.E_g / 2,
            s="$E_{\\mathrm{C}}$",
            ha="left")

    ax.text(x=2.1, y=0,
            s="$E_{\\mathrm{F}}$",
            ha="left")

    ax.text(x=2.1, y=-psi_b - Const.E_g / 2,
            s="$E_{\\mathrm{V}}$",
            ha="left")

    ax.text(x=1.8, y=-2 * psi_b + 0.025,
            s="$2(E_{\\mathrm{F, \\infty}} - E_{\\mathrm{i, \\infty}})$",
            ha="right", va="bottom")

    ax.annotate("", xy=(1.9, -2 * psi_b), xytext=(1.9, 0),
                arrowprops=dict(arrowstyle="<->",
                                lw=0.75))

    # text(-0.75, 0.14, 'MOS', 'Color', 'red');
    # text(0.35, -0.25, 'MOGS', 'Color', 'blue');
    # text(2.1, 0.2, 'E_C');
    # text(2.1, -psi_b, 'E_i');
    # text(2.1, -psi_b-Const.E_g/2, 'E_V');
    # text(-1.8, 2.5, 'N_D=10^{16} cm^{-3}');

    # xlabel('Q_{gate} (10^{13} e*cm^{-2})');
    # ylabel('\psi_s (V)');
    # set(gca, 'XTick', [-2, -1, 0, 1, 2]);
    # set(gca, 'YTick', [-1.2, -0.8, -0.4, 0, 0.4]);
    # set(gca, 'YLim', [-1.22,0.42])
    # % title('N_D = 10^{16} cm^{-3}');
    # % l=legend('With graphene', 'Without graphene','location','northwest');
    # % set(l,'Box','off');
    # modify_plot(1,1,2);
    # save_plot(20,20,'../Figure/Fig3-a.png')


def plot_main():
    fig, ax = gridplots(2, 2, r=0.95,
                        ratio=1)

    subfig_a(ax[0])
    subfig_b(ax[1])
    subfig_c(ax[2])
    grid_labels(fig, ax)
    # plot eps
    # names, Eg, eps_para, eps_perp, gm2 = get_gm2(d)
    # for i in range(len(names)):
    #     eg = Eg[i]
    #     n = names[i]
    #     if n not in candidates:
    #         a_ = 0.1
    #         lw = 1
    #     else:
    #         a_ = 1
    #         lw = 1.25
        
    #     c = get_color(eg, min=0, max=np.max(Eg),)
    #     ax[0].plot(freq_matsu, eps_para[i], color=c, alpha=a_, lw=lw)
    #     ax[1].plot(freq_matsu, eps_perp[i], color=c, alpha=a_, lw=lw)
    #     ax[2].plot(freq_matsu, 1 / gm2[i], color=c, alpha=a_, lw=lw)
    #     if n in candidates:
    #         j = 2
    #         ax[2].text(x=freq_matsu[j], y=(1 / gm2[i])[j]+0.05,
    #                    s=disp_names[n], ha="left", va="bottom", color=c)

    # ax[0].set_ylim(0.9, 20)
    # ax[0].set_yscale("log")
    # ax[1].set_ylim(0.9, 4)
    # # ax[1].set_yscale("log")
    # for ax_ in ax:
    #     ax_.set_xscale("log")
    #     ax_.axhline(y=1, ls="--", color="grey", alpha=0.6)
    # for i in (1, 2):
    #     ax[i].set_xlabel(r"$\hbar \xi$ (eV)")
    # ax[0].set_ylabel(r"$\varepsilon_{\mathrm{m}}^{\parallel} (i \xi)$")
    # ax[1].set_ylabel(r"$\varepsilon_{\mathrm{m}}^{\perp} (i \xi)$")
    # ax[2].set_ylabel(r"$1/g_{\mathrm{m}} (i \xi)$")

    # # Fix lim
    # ax[0].set_xticklabels([])
    # ax[0].tick_params(bottom=False, labelbottom=False, axis="x", direction="in")

    # # Figure right
    # cax_inside = inset_axes(ax[2], height="70%", width="50%",
    #                         bbox_to_anchor=(0.78, 0.05, 0.1, 0.80),
    #                         bbox_transform=ax[2].transAxes,
    #                         loc="lower left")
    # cb = add_cbar(fig, ax[2], 0, np.max(Eg), cax=cax_inside, shrink=0.5)
    # cb.ax.set_title("$E_{\mathrm{g}}$ (eV)", pad=1)

    # # label


    # # Plot in the insets
    # names, Eg, trans_para, trans_perp = get_transition_freq(d)
    # ax_in1 = inset_axes(ax[0], height="58%", width="35%",
    #                     loc="upper right")
    # ax_in1.tick_params(labelsize="small")
    # ax_in1.set_xlabel(r"$E_{\mathrm{g}}^{\mathrm{2D}}$ (eV)", size="small", labelpad=-1)
    # ax_in1.set_ylabel(r"$\hbar \xi_{\mathrm{tr}}^{\parallel}$ (eV)",
    #                   size="small", labelpad=-1)

    # ax_in1.scatter(Eg, trans_para, s=10, color="#3287a8",
    #                marker="^", alpha=0.4, edgecolor=None)

    # ax_in2 = inset_axes(ax[1], height="58%", width="35%",
    #                     loc="upper right")

    # ax_in2.scatter(Eg, trans_perp, s=10, color="#a87e31",
    #                marker="o",
    #                alpha=0.5)
    # ax_in2.tick_params(labelsize="small")
    # ax_in2.set_xlabel(r"$E_{\mathrm{g}}^{\mathrm{2D}}$ (eV)", size="small", labelpad=-1)
    # ax_in2.set_ylabel(r"$\hbar \xi_{\mathrm{tr}}^{\perp}$ (eV)",
    #                   size="small", labelpad=-1)
    
    savepgf(fig, img_path / "EFermi-prof.pgf", preview=True)
    

if __name__ == '__main__':
    plot_main()
