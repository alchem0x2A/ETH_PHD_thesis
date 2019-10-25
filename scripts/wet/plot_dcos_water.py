import matplotlib
from matplotlib import patches
# from pubfigure.FigureCollection import FigureCollection
import numpy
import scipy
import scipy.constants as const
# import pycse.orgmode as org

eps_w = 80*const.epsilon_0
d_H = 0.3*10**-9                # Helmholtz plane
n_L = numpy.linspace(-4, 4, 100)
sigma_L = n_L*const.e*10**13*10**4
T = 298
C_H = eps_w/d_H
gamma_w = 72.8e-3               # surface tension in SI
f_MD = scipy.poly1d([0.2157, -0.691, -3.71973, 0]) / 1000 #In mJ/m^2!!


def cal_2D(c0, sigma_, what="Delta_cos", z=1, add_MD=False):
    # c0 should use mol/m^3
    sigma = -sigma_
    psi_L = -2*const.k*T/z/const.e*scipy.arcsinh(
            sigma/scipy.sqrt(8*c0*const.N_A*eps_w*const.k*T))
    psi_2D = psi_L - sigma/C_H
    A = scipy.sqrt(2*z**2*const.e**2*eps_w*c0*const.N_A/const.k/T)
    B = z*const.e*psi_L/(2*const.k*T)
    C_L = A*scipy.cosh(B)
    l_D = scipy.sqrt(eps_w*const.k*T/(2*z**2*const.e**2*c0*const.N_A))
    Delta_Phi_el = -sigma**2/(2*C_H) - sigma**2/(C_L+eps_w/l_D)
    if add_MD is True:
        n = sigma_/(const.e*10**13*10**4)
        Delta_Phi_MD = f_MD(n)
        Delta_Phi_el += Delta_Phi_MD
    Delta_cos = -Delta_Phi_el/gamma_w

    # Classical value
    # C = scipy.sqrt(32*const.k**3*T**3*eps_w*c0*const.N_A/z**2/const.e**2)
    # Delta_Phi_el = -sigma**2/(2*C_H) - C*(scipy.cosh(B)-1)
    # Delta_cos = -Delta_Phi_el/gamma_w

    # Classical value
    # sigma = scipy.sqrt(8*c0*const.N_A*eps_w*const.k*T)*scipy.sinh(z*const.e*psi_L/2/const.k/T)
    # C = scipy.sqrt(32*const.k**3*T**3*eps_w*c0*const.N_A/z**2/const.e**2)
    # Delta_Phi_el = -sigma**2/(2*C_H) - C*(scipy.cosh(B)-1)
    # Delta_cos = -Delta_Phi_el/gamma_w
    if what is "Delta_Phi_el":
        return Delta_Phi_el
    elif what is "Delta_cos":
        return Delta_cos

def plot_ph_dep(ax, MD=False):
    # Plot the Delta theta as function of sigma
    for ph in numpy.arange(0, -8, -1):
        pH = 10**numpy.int(ph)  # why raise error??
        pH_SI = pH*1000
        res = cal_2D(pH_SI, sigma_L, what="Delta_cos", add_MD=MD)
        # res = scipy.arccos(res)/scipy.pi*180
        ax.plot(n_L, res)
    ax.set_xlabel(r"$\sigma_{\mathrm{2D}}$ (10$^{13}$ $e\cdot$cm$^{-2}$)")
    if MD == False:
        ax.set_ylabel(r"$(\Delta\cos\theta)^{\mathrm{EDL}}$")
    else:
        ax.set_ylabel(r"$(\Delta\cos\theta)^{\mathrm{Orien+EDL}}$")
        # Annotation now
    # ax2.set_ylim(ax.get_ylim())
    # ax2_yticks = -numpy.arange(0, int(max(f_MD(n_L))))
    # # ax2_real_ytick = -ax2_yticks/1000/gamma_w
    # ax2.set_yticks(ax2_yticks)
    # ax2.set_yticklabels(list(map(str, ax2_yticks)))
    # ax2.set_ylabel(r"$\Delta\Phi_{\mathrm{2D-w}}^{el}$ (mJ$\cdot$m$^{-2}$)")
    if MD == False:
        ax.text(0, 0.04,
                s=r"$c_{0}=10^{0}\sim{}10^{-7}$ mol$\cdot$L$^{-1}$",
                ha="center",
                va="center")
    # Extreme care with the arrow. Use annotate!
        ax.annotate("",
                    xy=(1.5, 0.03),
                    xytext=(3, 0.005),
                    arrowprops=dict(
                        width=0.25,
                        headwidth=4,
                        headlength=4,
                        facecolor="k",
                        edgecolor=None,))
    else:
        ax.text(-0.8, 0.10,
                s=r"$c_{0}=10^{0}\sim{}10^{-7}$ mol$\cdot$L$^{-1}$",
                ha="center",
                va="center")
    # Extreme care with the arrow. Use annotate!
        ax.annotate("",
                    xy=(0.5, 0.07),
                    xytext=(1.5, 0.02),
                    arrowprops=dict(
                        width=0.25,
                        headwidth=4,
                        headlength=4,
                        facecolor="k",
                        edgecolor=None,))

def plot_theta_2D(ax):
    theta_0 = numpy.linspace(40, 100, 100)
    ss, tt = numpy.meshgrid(sigma_L, theta_0)
    nn, tt_ = numpy.meshgrid(n_L, theta_0)
    c0 = 10**3*10**-7           # The concentration
    dd = scipy.arccos(scipy.cos(tt/180*scipy.pi)+cal_2D(c0, ss))/scipy.pi*180-tt
    pmesh = ax.pcolormesh(nn, tt, dd,
                        linewidth=0, rasterized=True,
                        cmap="viridis_r",
                        vmax=0)
    ax.set_xlabel(r"$\sigma_{\mathrm{2D}}$ (10$^{13} e\cdot$cm$^{-2}$)")
    ax.set_ylabel(r"$\theta*$ ($^{\circ}$)")
    # cbar.ax.tick_params(labelsize="small")
    # cbar.set_label(label=r"$\Delta\theta$ ($^{\circ}$)",
    #                size="small")

def plot_main():
    from . import data_path, img_path
    from helper import gridplots, grid_labels, savepgf
    fig, ax = gridplots(1, 2, r=0.9, ratio=2.5)
    plot_ph_dep(ax[0], MD=False)
    ax[0].text(x=0.5, y=0.95, s="EDL only", ha="center", va="top", transform=ax[0].transAxes)
    ax[1].text(x=0.5, y=0.95, s="EDL+Orientation", ha="center", va="top", transform=ax[1].transAxes)
    plot_ph_dep(ax[1], MD=True)
    grid_labels(fig, ax)
    savepgf(fig, img_path / "dcos_analytical.pgf")

if __name__ == "__main__":
    plot_main()
    # fc = FigureCollection(pagesize=(3, 5.5),
    #                       figure_style="science",
    #                       col=1, row=9)
    # # fc.fc_param["figure.lpad"] = 0.02
    # # fc.fc_param["figure.rpad"] = 0.0
    # fc.fc_param["figure.tpad"] = 0.05
    # fc.fc_param["figure.bpad"] = 0.05
    # # fc.fc_param["annotation.location"] = (0,0)
    # fig1, num1 = fc.add_figure(loc=(0, 0, 1, 3), label=True)
    # fig1.add_file_figure("../img/scheme-EDL.pdf")
    # fig2, num1 = fc.add_figure(loc=(0, 3, 1, 3), label=True)
    # fig2.set_plot_func(plot_ph_dep, MD=False)
    # fig3, num2 = fc.add_figure(loc=(0, 6, 1, 3), label=True)
    # fig3.set_plot_func(plot_ph_dep, MD=True)
    # org.figure(fc.save_all("../img/2d-ph-dependency+MD.pdf", outline=False),
    #            label="fig:res-EDL",
    #            caption=("(a) Scheme of the interface between the 2D material "
    #                     "and the aqueous phase. "
    #                     r"(b) $(\Delta\cos\theta)^{\mathrm{EDL}}$ "
    #                     "as a function of "
    #                     r"$\sigma_{\mathrm{2D}}$ with varied solute concentrations. "
    #                     r"The concentration $c_{0}$ varies from "
    #                     r"$10^{0}$ to $10^{-7}$ mol$\cdot\mathrm{L}^{-1}$ "
    #                     r"(c) Overall change of contact angle $(\Delta \cos \theta)^{\mathrm{Orien+EDL}}$ "
    #                     "combining the orientation and EDL effects, "
    #                     "with varied solute concentrations as in (b)."),
    #            attributes=[("latex", ":width 0.5\linewidth")])
