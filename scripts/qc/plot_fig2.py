import numpy as np
from . import data_path, img_path
from .constants import Const
from . import equations as eqs
from scipy.optimize import fsolve
from scipy.interpolate import BivariateSpline
from helper import gridplots, grid_labels, savepgf
from helper import get_color, add_cbar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Plot the figures for fig 2


def subfig_a(fig, ax, dim=256):
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


def subfig_b(fig, ax, dim=256):
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


def subfig_c(fig, ax, dim=256):
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
    ax.set_ylim(-1.22, 0.4)
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
    ax.text(x=-1.95, y=-1.20,
            s="Strong Inversion",
            ha="left", va="bottom")
    ax.text(x=1.85, y=-1.20,
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


def subfig_d(fig, ax, dim=256):
    NQ_gate = np.linspace(-2, 2, dim) * 1e13
    Q_gate = NQ_gate * 1e4 * Const.q
    # psi_b = eqs.func_Psi_B Psi_B(ND);

    # %Figure of Psi_s with graphene or not

    # figure(1);
    psi_b = np.linspace(-Const.E_g / 2, Const.E_g / 2, dim)
    n, p = eqs.func_np(0, psi_b)
    NDs = n
    data_zz = data_path / "fig22_z_{0:d}.npy".format(dim)
    if data_zz.exists():
        zz = np.load(data_zz)
        print("Loaded data from {0}".format(data_zz.as_posix()))
    else:
        zz = np.array([[fsolve(lambda Psi: eqs.solve_psi_s(Psi, pb, qg, 0), 0)[0]
                        for pb in psi_b]
                       for qg in Q_gate])
        np.save(data_zz, zz)



    phi_bs0 = np.array([fsolve(lambda Psi: eqs.solve_Psi_B(Psi, qg, 0), 0)[0]
                       for qg in Q_gate])
    ax.plot(NQ_gate / 1e13,
            np.log10(eqs.func_np(0, phi_bs0)[0] / 1e6),
            color="k",
            alpha=0.6,
            ls="--")

    # Manual tweaking is needed for correct Fsolve!
    phi_bs1 = np.array([fsolve(lambda Psi: eqs.solve_Psi_B(Psi, qg, -1),
                               -0.5 * NQ_gate[i] / 1e13 \
                               + np.sign(NQ_gate[i]) * -0.5)[0]
                        for i, qg in enumerate(Q_gate)])
    ax.plot(NQ_gate / 1e13,
            np.log10(eqs.func_np(0, phi_bs1)[0] / 1e6),
            color="k",
            alpha=0.6,
            ls="--")

    phi_bs2 = np.array([fsolve(lambda Psi: eqs.solve_Psi_B(Psi, qg, -2), 0)[0]
                       for qg in Q_gate])
    ax.plot(NQ_gate / 1e13,
            np.log10(eqs.func_np(0, phi_bs2)[0] / 1e6),
            color="k",
            alpha=0.6,
            ls="--")

    ax.axhline(y=np.log10(Const.n_i / 1e6),
               alpha=0.6,
               color="k", ls="--")
    
    xx, yy = np.meshgrid(NQ_gate / 1e13, np.log10(NDs / 1e6))
    xx_fine, yy_fine = np.meshgrid(np.linspace(xx.min(), xx.max(), 256),
                                   np.linspace(yy.min(), yy.max(), 256))
    cm = ax.imshow(zz.T[::-1, :], extent=[xx.min(), xx.max(),
                              yy.min(), yy.max()],
                   aspect="auto",
                   cmap="rainbow",
                   interpolation="bicubic")
    # cm = ax.pcolor(xx, yy, zz,
                   # rasterized=True, cmap="rainbow")
    cax_outside = inset_axes(ax, height="70%", width="50%",
                            bbox_to_anchor=(1.01, 0.05, 0.05, 0.80),
                            bbox_transform=ax.transAxes,
                            loc="lower left")
    cb = fig.colorbar(cm, cax_outside)
    # cb = add_cbar(fig, ax, 0, , cax=cax_inside, shrink=0.5)
    cb.ax.set_title("$\\psi_0$ (V)", pad=1, ha="left")
    # f = pcolor(NQ_gate/10^13,log10(NDs/10^6),z);
    ax.set_xlabel("$\\sigma_{\\mathrm{M}}\ (10^{13}\ e\\cdot{}\\mathrm{cm}^{-2})$")
    # set(gca, 'XTick', [-2,  -1,  0,  1,  2]);
    ax.set_ylabel("$n_0$ (cm$^{-3}$)")
    ax.set_yticks(np.arange(2, 19, 4))
    ax.set_yticklabels(["$10^{{{0}}}$".format(i) for i in np.arange(2, 19, 4)])

    ax.text(x=0.02, y=0.52, s="n-type", va="bottom", transform=ax.transAxes)
    ax.text(x=0.02, y=0.48, s="p-type", va="top", transform=ax.transAxes)

    ss = ["Ac", "Dp", "WI", "SI", "Ac", "Dp", "WI", "SI"]
    for i, s in enumerate(ss):
        angle = np.pi / 8 + np.pi / 4 * i
        r = 0.4
        pos_x = 0.5 + np.cos(angle) * r
        pos_y = 0.5 + np.sin(angle) * r
        ax.text(x=pos_x, y=pos_y, s=s,
                transform=ax.transAxes,
                size="large")
    



def plot_main():
    fig, ax = gridplots(2, 2, r=0.95,
                        ratio=1)

    subfig_a(fig, ax[0])
    subfig_b(fig, ax[1])
    subfig_c(fig, ax[2])
    subfig_d(fig, ax[3])
    grid_labels(fig, ax)
    
    savepgf(fig, img_path / "EFermi-prof.pgf", preview=True)
    

if __name__ == '__main__':
    plot_main()
