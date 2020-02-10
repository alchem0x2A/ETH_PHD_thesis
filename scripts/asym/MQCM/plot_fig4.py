import numpy as np
from . import data_path, img_path
from scipy.optimize import fsolve
from scipy.interpolate import BivariateSpline
from scipy.signal import medfilt
from helper import gridplots, grid_labels, savepgf
from helper import get_color, add_cbar, add_img_ax
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from .electronic import solve_stack, stack_charge, gen_input
from .electronic import k_SI_ecm, k_ecm_SI, eps0
from .electronic import get_tot_charge


def plot_a(fig, ax):
    k_Debye_SI = 3.34e-30

    n_gr = n_mos2 = 1
    eps = 2.5
    d0 = np.array([3.3, 4.9 * 1.02, 5.8]) * 10 ** -10

    mat_in, dis_in, eps_in = gen_input(n_gr, n_mos2, eps, d0)
    E_ext = np.linspace(-8e9, 8e9, 30)
    sigma_gr = []
    phi_gr = []
    phi_mos2 = []

    for E in E_ext:
        phi_0 = solve_stack(mat_in, dis_in,
                            eps_in, E)
        sigmas, phis, Es = stack_charge(phi_0,
                                        mat_in,
                                        dis_in,
                                        eps_in,
                                        E,
                                        use_solve=False)
        sigma_gr.append(sigmas[0])
        phi_gr.append(phis[0])
        phi_mos2.append(phis[1])

    sigma_gr = np.array(sigma_gr)
    phi_gr = np.array(phi_gr)
    phi_mos2 = np.array(phi_mos2)

    # Work function
    l1 = ax.plot(E_ext / 10 ** 9, phi_gr, "-o", label="$\\phi_{\\mathrm{Gr}}$")
    l2 = ax.plot(E_ext / 10 ** 9, phi_mos2, "-v", label="$\\phi_{\\mathrm{MoS_{2}}}$")
    ax.set_xlabel("{\\itshape Ɛ}$_{\\mathrm{ext}}$ (V$\\cdot{}$nm$^{-1}$)")
    ax.set_ylabel("Work Function (V)")
    # limits
    ax.set_ylim(4, 5.75)
    ax.set_xlim(-8, 8)
    ax.set_xticks(np.arange(-8, 8.1, 2))
    c1 = -2.5
    c2 = 5.3
    ax.fill_betweenx(y=ax.get_ylim(),
                     x1=-8, x2=c1,
                     color="green", alpha=0.15,
                     linewidth=0,
                     zorder=0)
    ax.fill_betweenx(y=ax.get_ylim(),
                     x1=c1, x2=c2,
                     color="k", alpha=0.15,
                     linewidth=0,
                     zorder=0)
    ax.fill_betweenx(y=ax.get_ylim(),
                     x1=c2, x2=8,
                     color="red", alpha=0.15,
                     linewidth=0,
                     zorder=0)
    ax.text(x=0.02, y=0.7, s="n-doped MoS$_{2}$",
            size="small",
            ha="left", transform=ax.transAxes)
    ax.text(x=0.98, y=0.7, s="p-doped MoS$_{2}$",
            size="small",
            ha="right", transform=ax.transAxes)

    ax.text(x=0.02, y=0.1, s="←CB",
            ha="left", transform=ax.transAxes)
    ax.text(x=0.98, y=0.9, s="VB→",
            ha="right", transform=ax.transAxes)
    ax.legend(loc=0)

    
def P_debye(sigma, d_t):
    k_Debye_SI = 3.34e-30
    a_cell = 13.06e-10
    A = (3 ** 0.5 / 2) * a_cell ** 2
    P_area = sigma * d_t
    return P_area * A / k_Debye_SI


def plot_bc(fig, ax):
    ax_b, ax_c = ax
    eps = 2.5
    d0 = [3.3e-10, 4.9e-10, 6.2e-10]
    colors = ["#094eac", "#ff6633", "#91b96d"]
    styles = ["-o", "-^", "-s"]
    area  = (17.06 * 10 ** -8) ** 2 * (3 ** 0.5 / 2)

    for n_gr in range(1, 4):
        for n_mos2 in range(1, 4):
            mat_in, dis_in, eps_in = gen_input(n_gr, n_mos2, eps, d0)
            # E_ext = np.linspace(-1.2e9, 1.2e9, 50)
            E_ext = np.linspace(-1.75e9, 1.75e9, 50)
            P = []
            Chg = []

            for E in E_ext:
                phi_0 = solve_stack(mat_in, dis_in,
                                    eps_in, E)
                sigmas, phis, Es = stack_charge(phi_0,
                                                mat_in,
                                                dis_in,
                                                eps_in,
                                                E,
                                                use_solve=False)
                chg = sum(sigmas[0: n_gr])
                t_sigma = get_tot_charge(sigmas)
                P.append(P_debye(t_sigma, sum(dis_in)))
                Chg.append(k_SI_ecm * chg)
            Chg = np.array(Chg)
            P = np.array(P)
            
            # P = k_SI_ecm * P

            l1, = ax_b.plot(E_ext / 1e9, P,
                            styles[n_mos2 - 1],
                            color=colors[n_gr - 1],
                            label="{{{0:d}}}L G/{{{1:d}}}L MoS$_{{2}}$".format(n_gr, n_mos2))

            l2, = ax_c.plot(E_ext / 1e9, Chg / 1e12,
                            styles[n_mos2 - 1],
                            color=colors[n_gr - 1],
                            label="{{{0:d}}}L G/{{{1:d}}}L MoS$_{{2}}$".format(n_gr, n_mos2))

    ax_b.set_xlabel("{\\itshape Ɛ}$_{\\mathrm{ext}}$ (V/nm)")
    ax_b.set_ylabel("$\\mu(m, n)$ (Debye)")
    ax_b.set_ylim(-2, 6.5)
    ax_b.legend(loc=0, handlelength=1,
                labelspacing=0.2,
                fontsize="small",)
    # ax_b.invert_yaxis()

    ax_c.set_xlabel("{\\itshape Ɛ}$_{\\mathrm{ext}}$ (V$\\cdot{}$nm$^{-1}$)")
    ax_c.set_ylabel("$\\sigma_{\\mathrm{G}}$ ($10^{13} e\\cdot$cm$^{-2}$)")
    ax_c.legend(loc=0, handlelength=1,
                labelspacing=0.2,
                fontsize="small",)
    ax_c.text(x=0.02, y=0.1, s="n-doped MoS$_{2}$",
              size="small",
              ha="left", transform=ax_c.transAxes)
    ax_c.text(x=0.98, y=0.1, s="p-doped MoS$_{2}$",
              size="small",
              ha="right", transform=ax_c.transAxes)
    ax_c.fill_betweenx(y=ax_c.get_ylim(),
                       x1=ax_c.get_xlim()[0], x2=0,
                       color="green", alpha=0.15,
                       linewidth=0,
                       zorder=0)
    ax_c.fill_betweenx(y=ax_c.get_ylim(),
                       x1=0, x2=ax_c.get_xlim()[1],
                       color="red", alpha=0.15,
                       linewidth=0,
                       zorder=0)
    # ax_c.invert_yaxis()

def plot_d(fig, ax):
    ax.set_axis_off()
    ax_large = inset_axes(ax, width="100%", height="100%",
                          bbox_to_anchor=(-0.1, -0.1, 1.2, 1.2),
                          bbox_transform=ax.transAxes)
    add_img_ax(ax_large, fname=img_path / "sub_img" / "band_diagram.png")
    ax_large.text(x=0.02, y=1.00,
                  ha="left",
                  s="{\\itshape Ɛ}$_{\\mathrm{ext}}=-2$ V$\\cdot{}$nm$^{-1}$",
                  transform=ax_large.transAxes)
    
    ax_large.text(x=0.02, y=0.41,
                  ha="left",
                  s="{\\itshape Ɛ}$_{\\mathrm{ext}}=2$ V$\\cdot{}$nm$^{-1}$",
                  transform=ax_large.transAxes)

    ax_large.text(x=0.98, y=0.93,
                  ha="right", va="bottom",
                  s="$E_{\\mathrm{vac}}$",
                  transform=ax_large.transAxes)
    
    ax_large.text(x=0.98, y=0.42,
                  ha="right", va="bottom",
                  s="$E_{\\mathrm{vac}}$",
                  transform=ax_large.transAxes)

    ax_large.text(x=0.98, y=0.52,
                  ha="right", va="top",
                  s="$x$",
                  transform=ax_large.transAxes)

    ax_large.text(x=0.98, y=-0.02,
                  ha="right", va="top",
                  s="$x$",
                  transform=ax_large.transAxes)

    ax_large.text(x=-0.03, y=0.70,
                  ha="center", va="bottom",
                  s="$E - E_{\\mathrm{F}}$",
                  rotation=90,
                  transform=ax_large.transAxes)

    ax_large.text(x=-0.03, y=0.18,
                  ha="center", va="bottom",
                  s="$E - E_{\\mathrm{F}}$",
                  rotation=90,
                  transform=ax_large.transAxes)
    
# def plot_d(fig, ax):
#     """Plot the position of bands"""
#     ax.set_axis_off()
#     ax_t = inset_axes(ax, width="100%", height="45%",
#                       bbox_to_anchor=(-0.1, -0.1, 1.2, 1.2),
#                       bbox_transform=ax.transAxes,
#                       loc="upper center")
#     ax_b = inset_axes(ax, width="100%", height="45%",
#                       bbox_to_anchor=(-0.1, -0.1, 1.2, 1.2),
#                       bbox_transform=ax.transAxes,
#                       loc="lower center")
#     eps = 2.5
#     d0 = [3.3e-10, 4.9e-10, 5.8e-10]
#     delta = 0.1e-10

#     def plot_band(ax, dist, ng, nm, sigmas, phis):
#         tot_d = 0
#         efermi = 0
#         phi0 = [4.6] * ng + [4.49] * nm
#         for i, d in enumerate([0] + dist):
#             tot_d += d
#             ax.hlines(y=efermi + phis[i] - phi0[i],
#                       xmin=tot_d - delta,
#                       xmax=tot_d + delta)
#             ax.hlines(y=efermi + phis[i],
#                       xmin=tot_d - delta,
#                       xmax=tot_d + delta)
#         ax.set_ylim(-3, 5.5)
#         ax.set_xlim(-0.4e-9, 2e-9)
#         ax.set_xticks([])
#         ax.axhline(y=0)



#     for ax_, E_ext in zip([ax_t, ax_b], [-2e9, 2e9]):
#         mat_in, dis_in, eps_in = gen_input(n_gr=1, n_mos2=3,
#                                            eps=eps,
#                                            d=d0)

#         phi_0 = solve_stack(mat_in, dis_in, eps_in, E_ext)
#         sigmas, phis, Es = stack_charge(phi_0, mat_in, dis_in,
#                                         eps_in, E_ext,
#                                         use_solve=False)

#         plot_band(ax_, dis_in, 1, 3, sigmas, phis)

    



def plot_main():
    h = 1.1
    fig, ax = gridplots(2, 2, r=1.0, ratio=1.25)
    plot_a(fig, ax[0])
    plot_bc(fig, ax[1: 3])
    plot_d(fig, ax[-1])
    grid_labels(fig, ax)
    savepgf(fig, img_path / "mqcm_results.pgf", preview=True)
    
    
if __name__ == '__main__':
    plot_main()
