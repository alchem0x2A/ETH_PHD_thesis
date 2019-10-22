import numpy
from scipy.optimize import curve_fit
import os, os.path
import matplotlib.pyplot as plt
from scipy.constants import pi
plt.style.use("science")

def fit_para(L, d, eps_2D):
    return (eps_2D - 1) * d / L  + 1

def fit_vert(L, d, eps_2D):
    return 1 / (d / L * (1 / eps_2D - 1) + 1)

root = "../../data/distance/"
g = os.walk(root)
names = next(g)[1]
for i, item in enumerate(g):
    if "old" in item:
        continue
    for f in item[2]:
        f_path = os.path.join(item[0], f)
        if "agr" not in f_path:
            continue
        print(f_path)
        data = numpy.genfromtxt(f_path,
                                delimiter=" "
                                # skip_header=297,
                                # skip_footer=2
         )
        # data = data[1: , :]
        L = data[:, 0]
        eps_SL = data[:, 1]
        if "parallel.agr" in f_path:
            alpha_SL = L * (data[:, 1] - 1)  / (4 * pi)
        elif "perpendicular.agr" in f_path:
            alpha_SL = L * (data[:, 1] - 1) / (data[:, 1]) / (4 * pi)
        fig = plt.figure(figsize=(2.5, 2.5))
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()
        ax.plot(L, eps_SL, "o-", color="#00AAD4")
        ax2.plot(L, alpha_SL, "o-", color="#FF6600")

        ax.set_xlabel("$L$ ($\\AA$)")
        if "par" in f_path:
            param, _ = curve_fit(fit_para, L, eps_SL,
                                 p0=(5, 10),
                                 bounds=((0.5, 1.0),
                                         (10, 50))
            )
            LL = numpy.linspace(10, 70)
            # ax.plot(LL, fit_para(LL, *param), "--")
            # ax.set_title("{2}, ${{\\epsilon_{{2D}}^{{\\parallel}} }}={1:.3f}$ ${{d}}={0:.3f}$".format(*param, names[i]))
            # ax.set_ylabel("$\\epsilon_{\\parallel}$")
            ax.set_title("{} parallel".format(names[i]))
            ax.set_ylabel("$\\epsilon^{\\parallel}_{\mathrm{SL}}$")
            ax2.set_ylabel("$\\alpha_{\\parallel}/(4\\pi\\varepsilon_0) (\\AA)$")
            # ax.set_ylim(0, 12)
            ax2.set_ylim(min(alpha_SL) * 0.75, max(alpha_SL)* 1.25)
            fig.savefig(os.path.join(item[0], "{}-fit-para.svg".format(names[i])))
        elif "perp" in f_path:
            param, _ = curve_fit(fit_vert, L, eps_SL,
                                 p0=(5, 10),
                                 bounds=((0.5, 1.0),
                                         (10, 50))
            )
            LL = numpy.linspace(10, 70)
            # ax.plot(LL, fit_vert(LL, *param), "--")
            ax.set_ylabel("$\\epsilon^{\\perp}_{\mathrm{SL}}$")
            ax2.set_ylabel("$\\alpha_{\\perp}/(4\\pi \\varepsilon_0) (\\AA)$")
            # ax.set_title("{2}, ${{\\epsilon_{{2D}}^{{\\perp}} }}={1:.3f}$ ${{d}}={0:.3f}$".format(*param, names[i]))
            # ax.set_title("{} perp".format(names[i]))
            # ax.set_ylim(0, 1)
            ax2.set_ylim(min(alpha_SL) * 0.75, max(alpha_SL)* 1.25)
            fig.savefig(os.path.join(item[0], "{}-fit-perp.svg".format(names[i])))
