import matplotlib, numpy, scipy
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
import scipy.constants as const
# import pycse.orgmode as org
from scipy.integrate import cumtrapz, trapz
from .plot_dgamma_sigma import cal_2D
# from pubfigure.FigureCollection import FigureCollection

Materials = {}
# The parameters are using values of 10^13 e/cm^2 for sigma
# and uF/cm^2 for C

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

f_MD = scipy.poly1d([0.1647, -0.5857, -3.4094, 0])/1000 #In mJ/m^2!!

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

eps_hfo2 = 24
t0 = 2e-9
C_hfo2 = const.epsilon_0*eps_hfo2/t0
c0 = 10**-7*1000
# print(C_hfo2*100)

def cal_V_2D(sigma, mater):
    # Return array-like V_2D
    # C_2D = numpy.array([f_C_2D(s, mater) for s in sigma])
    V_2D_ = []
    for s in sigma:
        if s is 0:
            V_2D_.append(0)
        else:
            ss = numpy.linspace(1e-5*s, s, 100)
            C_2D_ = numpy.array([f_C_2D(s_, mater) for s_ in ss])
            V_2D_.append(trapz(1/C_2D_, ss))
    V_2D_ = numpy.array(V_2D_)
    # V_2D_ = cumtrapz(1/(C_2D), sigma, initial=0)
    # pos_0 = numpy.argmin(numpy.absolute(sigma))  # The minimal sigma close to 0
    # V_2D_ = V_2D_ - V_2D_[pos_0]
    V_ox = sigma/C_hfo2
    return V_2D_ + V_ox

def plot_dcos_all(ax, MD=False):
    # ax = fig.add_subplot(111)
    n_e = numpy.linspace(-15, 15, 201)
    sigma_e = n_e*10**13*10**4*const.e

    for m in ["Gr", "Si", "Ge", "P", "MoS2", "MoSe2", "MoTe2", "WS2", "WSe2", "WTe2",]:
        V = cal_V_2D(sigma_e, m)
        n_e = sigma_e/(10**13*10**4*const.e)
        dcos_MD = f_MD(n_e)
        dcos = cal_2D(c0, sigma_e)
        if MD == True:
            dcos += dcos_MD
        ax.plot(V, dcos, Styles[m], label=Materials[m]["name"])
    ax.set_xlabel(r"$V_{\mathrm{G}}$ (V)")
    ax.set_ylabel(r"$(\Delta\cos\theta)^{\mathrm{EDL}}$")
    ax.legend(loc=0)
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 0.25)
    # fig.tight_layout()


def plot_main():
    from . import data_path, img_path
    from helper import gridplots
    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["svg.fonttype"] = "none"
    fig, ax = gridplots(r=0.45, ratio=1.25)
    plot_dcos_all(ax, MD=False)
    fig.savefig(img_path / "gating.svg")
    
if __name__ == "__main__":
    # fc = FigureCollection(pagesize=(3, 4.8),
    #                       figure_style="science",
    #                       col=1, row=11)
    # fc.fc_param["fig.tpad"] = 0.1
    # fc.fc_param["annotation.size"] = 12
    # fig1, _ = fc.add_figure(loc=(0, 0, 1, 5),
    #                         fig_file="../img/scheme-2D-elw.pdf")
    # fig2, _ = fc.add_figure(loc=(0, 4, 1, 6))
    # fig2.set_plot_func(plot_dcos_all, MD=False)
    # org.figure(fc.save_all("../img/dcos-all-2D.pdf", outline=False),
    #            attributes=[("latex", ":width 0.65\linewidth")],
    #            label="fig:dcos-all-2D",
    #            caption=("(a) Schematic illustration of the "
    #                     "2D-material-based electrowetting device, "
    #                     "where the 2D material is electostatically doped. "
    #                     r"(b) $(\Delta\cos\theta)^{\mathrm{EDL}}$ "
    #                     r"as a function of $V_{\mathrm{G}}$ "
    #                     "for selected 2D materials."))
    plot_main()
