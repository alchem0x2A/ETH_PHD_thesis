import matplotlib, numpy, scipy
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
from . import data_path, img_path
from helper import grid_labels, gridplots, savepgf
import scipy.constants as const
# import pycse.orgmode as org
from scipy.integrate import cumtrapz, trapz
from .dcos_sigma import cal_2D
# from pubfigure.FigureCollection import FigureCollection

Materials = {}
# The parameters are using values of 10^13 e/cm^2 for sigma
# and uF/cm^2 for C
Styles = {"MoS2": "--",
          "MoSe2": "-.",
          "MoTe2": ":",
          "WS2": "--",
          "WSe2": "-.",
          "WTe2": ":",
          "Gr": "-",
          "Si": "-",
          "Ge": "-",
          "P": "--"
}
Materials['MoS2'] = dict(n=[48.32, 0, 3.6270e-13],
                         p=[186.6, 0, 9.6567e-13],
                         name=r"MoS$_{2}$",)
Materials['MoSe2'] = dict(n=[55.94, 0, 4.2354e-13],
                          p=[74.76, 0, 4.7792e-14],
                          name=r"MoSe$_{2}$",)
Materials['MoTe2'] = dict(n=[61.67, 0, 4.7299e-13],
                          p=[82.52, 0, 1.0820e-13],
                          name=r"MoTe$_{2}$",)
Materials['WS2'] = dict(n=[33.92, 0, 3.6270e-13],
                        p=[169.5, 0, 9.1869e-13],
                        name=r"WS$_{2}$",)
Materials['WSe2'] = dict(n=[36.99, 0, 3.955e-13],
                         p=[52.01, 0, 3.0965e-13],
                         name=r"WSe$_{2}$",)
Materials['WTe2'] = dict(n=[37.87, 0, 3.8405e-13],
                         p=[52.01, 0, 4.0845e-13],
                         name=r"WTe$_{2}$",)
Materials['P'] = dict(n=[54.47, 0, 8.9640e-14],
                      p=[67.86, 0, 6.7077e-15],
                      name="Phosphorene",)
Materials['Gr'] = dict(n=[0, 2.745969059762e-06, 0],
                       p=[0, 2.747402905456e-06, 0],
                       name="Graphene",)
Materials['Si'] = dict(n=[0, 4.872842161338e-06, 0],
                       p=[0, 4.663485703981e-06, 0],
                       name="Silicene")
Materials['Ge'] = dict(n=[0, 5.447917304238e-06, 0],
                       p=[0, 4.868667384166e-06, 0],
                       name="Germanene")

# Only single unit!
def f_C_2D(sigma_, mater):
    # Receive the sigma in SI
    param_n = Materials[mater]["n"]
    param_p = Materials[mater]["p"]
    n_13 = sigma_/const.e/10**4
    # Return the C_2D in SI
    if n_13>0:
        return (param_p[0]
                + param_p[1]*scipy.absolute(n_13)**0.5
                + param_p[2]*scipy.absolute(n_13))/100
    else:
        return (param_n[0]
                + param_n[1]*scipy.absolute(n_13)**0.5
                + param_n[2]*scipy.absolute(n_13))/100

def f_dgamma(sigma_lim, mater):
    # sigma_lim is using the absolute value
    param_n = Materials[mater]["n"]
    param_p = Materials[mater]["p"]
    sigma_p = numpy.linspace(sigma_lim*10**-6, sigma_lim, 200)
    sigma_n = numpy.linspace(-sigma_lim*10**-6, -sigma_lim, 200)
    C_2D_p = numpy.array([f_C_2D(sigma_, mater) for sigma_ in sigma_p])
    C_2D_n = numpy.array([f_C_2D(sigma_, mater) for sigma_ in sigma_n])
    dgamma_p = cumtrapz(-sigma_p/C_2D_p, sigma_p, initial=0)
    dgamma_n = cumtrapz(-sigma_n/C_2D_n, sigma_n, initial=0)
    sigmas = numpy.hstack([sigma_n[::-1], sigma_p])
    dgammas = numpy.hstack([dgamma_n[::-1], dgamma_p])
    return sigmas, dgammas

def plot_dgamma_sigma(ax):
    n_lim = 4
    sigma_lim = n_lim*10**13*10**4*const.e

    for m in ["Gr", "Si", "Ge", "P", "MoS2", "MoSe2", "MoTe2", "WS2", "WSe2", "WTe2",]:
        sigmas, dgammas = f_dgamma(sigma_lim, m)
        ax.plot(sigmas/const.e/10**17, dgammas*1000, Styles[m], label=Materials[m]["name"])
    ax.set_xlabel(r"$\sigma_{\mathrm{2D}}$ ($10^{13}\ e\cdot$cm$^{-2}$)")
    ax.set_ylabel(r"$\Delta\gamma_{\mathrm{2D}}$ (mJ$\cdot$m$^{-2}$)")
    ax.legend(loc=0,
              # prop=dict(size="small")
    )
    # ax.set_xlim(-2, 2)
    # ax.set_ylim(0, 0.15)
    # fig.tight_layout(pad=0.1)

def plot_main():
    fig, ax = gridplots(r=0.65,  ratio=1.25)
    plot_dgamma_sigma(ax)
    savepgf(fig, img_path / "dgamma-sigma.pgf")

if __name__ == "__main__":
    # fc = FigureCollection(pagesize=(3, 2.5),
    #                       figure_style="science",
    #                       col=1, row=1)
    # fig2, _ = fc.add_figure(label=False, outline=True)
    # fig2.set_plot_func(plot_dgamma_sigma)
    # org.figure(fc.save_all("../img/dgamma-sigma.pdf", outline=False),
    #            attributes=[("latex", ":width 0.95\linewidth")],
    #            label="fig:dgamma-sigma",
    #            caption=(r"$\Delta \gamma_{\mathrm{2D}}$ "
    #                     "as a function of "
    #                     r"$\sigma_{\mathrm{2D}}$ "
    #                     "for selected 2D materials: graphene, silicene, germanene, phosphorene, "
    #                     r"MoS$_{2}$, MoSe$_{2}$, MoTe$_{2}$, WS$_{2}$, WSe$_{2}$ and WTe$_{2}$"))
    plot_main()

