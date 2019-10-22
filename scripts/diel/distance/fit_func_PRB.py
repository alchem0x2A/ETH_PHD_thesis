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
tags = {
    "mos2": "MoS$_{2}$",
    "mose2": "MoSe$_{2}$",
    "mote2": "MoTe$_{2}$",
    "ws2": "WS$_{2}$",
    "wse2": "WSe$_{2}$",
    "wte2": "WTe$_{2}$",
}
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
        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_subplot(111)
        # ax2 = ax.twinx()
        ax.plot(L, eps_SL, "o")
        # ax2.plot(L, alpha_SL, "o-", color="#FF6600")

        ax.set_xlabel("$L$ ($\\AA$)")
        if "par" in f_path:
            param, _ = curve_fit(fit_para, L[1:], eps_SL[1:],
                                 p0=(5, 10),
                                 bounds=((0.5, 1.0),
                                         (10, 50))
            )
            LL = numpy.linspace(10, 70)
            ax.plot(LL, fit_para(LL, *param), "--")
            ax.set_title("{2}, ${{\\epsilon_{{2D}}^{{\\parallel}} }}={1:.3f}$ ${{\\delta}}={0:.3f}\\ \\AA$".format(*param, tags[names[i]]))
            ax.set_ylabel("$\\epsilon_{\\parallel}$")
            # ax.set_title("{} parallel".format(names[i]))
            ax.set_ylabel("$\\epsilon^{\\parallel}_{\mathrm{SL}}$")
            # ax.set_ylim(0, 12)
            # ax2.set_ylim(min(alpha_SL) * 0.75, max(alpha_SL)* 1.25)
            fig.savefig(os.path.join(item[0], "{}-fit-paral-SL.svg".format(names[i])))
        elif "perp" in f_path:
            param, _ = curve_fit(fit_vert, L[1:], eps_SL[1:],
                                 p0=(5, 10),
                                 bounds=((0.5, 1.0),
                                         (10, 50))
            )
            LL = numpy.linspace(10, 70)
            ax.plot(LL, fit_vert(LL, *param), "--")
            ax.set_ylabel("$\\epsilon^{\\perp}_{\mathrm{SL}}$")
            ax.set_title("{2}, ${{\\epsilon_{{2D}}^{{\\perp}} }}={1:.3f}$ ${{\\delta}}={0:.3f}\\ \\AA$".format(*param, tags[names[i]]))
            # ax.set_title("{} perp".format(names[i]))
            # ax.set_ylim(0, 1)
            fig.savefig(os.path.join(item[0], "{}-fit-perp-SL.svg".format(names[i])))
