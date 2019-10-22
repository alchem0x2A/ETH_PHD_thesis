import numpy
from scipy.optimize import curve_fit
import os, os.path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.constants import pi
from model_eps import plot_eps_para, plot_eps_perp
plt.style.use("science")


def fit_para(L, d, eps_2D):
    return (eps_2D - 1) * d / L  + 1

def fit_vert(L, d, eps_2D):
    return 1 / (d / L * (1 / eps_2D - 1) + 1)

root = "../../data/distance/"
g = os.walk(root)
# print(g)
names = next(g)[1]
# print(names)
# names = sorted(names)
fig1 = plt.figure(figsize=(4.5, 2))
ax1_r = fig1.add_subplot(121)
# ax1 = fig1.add_subplot(gs[:5 ])
ax2_r = fig1.add_subplot(122)
# ax2 = fig2.add_subplot(gs[:5 ])

colors = {"mos2": "#1f77b4",
          "mose2": "#ff7f0e",
          "mote2": "#2ca02c",
          "ws2": "#d62728",
          "wse2": "#9467bd",
          "wte2": "#8c564b",
}

symbs = {"mos2": "MoS$_{2}$",
          "mose2": "MoSe$_{2}$",
          "mote2": "MoTe$_{2}$",
          "ws2": "WS$_{2}$",
          "wse2": "WSe$_{2}$",
          "wte2": "WTe$_{2}$",
}

# plot_eps_para(ax1, alpha_0=8.5, dev=4)
# plot_eps_perp(ax2, alpha_0=0.47, dev=0.15)

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
            error_alpha = numpy.abs(alpha_SL - alpha_SL[-1]) / alpha_SL[-1] 
            # l, *_ = ax1.plot(L, eps_SL, "o",
            #                  label=names[i],
            #                  markersize=4,
            #                  color=colors[names[i]]
            # )
            ax1_r.plot(L, error_alpha, "o",
                       markersize=4,)
        elif "perpendicular.agr" in f_path:
            alpha_SL = L * (data[:, 1] - 1) / (data[:, 1]) / (4 * pi)
            error_alpha = numpy.abs(alpha_SL - alpha_SL[-1]) / alpha_SL[-1]
            # l, *_ = ax2.plot(L, eps_SL, "o-",
            #                  label=names[i],
            #                  markersize=4,
            #                  color=colors[names[i]]
            # )
            ax2_r.plot(L, error_alpha, "o",
                       markersize=4,
                       label=symbs[names[i]])

ax1_r.set_xlabel("$L$ ($\\AA$)")
ax1_r.set_title("parallel")
# ax1.set_ylabel("$\\epsilon^{\\parallel}_{\mathrm{SL}}$")
ax1_r.set_ylabel("$\\Delta \\alpha^{\\parallel}_{\\mathrm{2D}}/(4\\pi\\varepsilon_0) (\\AA)$")
ax1_r.set_yscale("log")
# ax1.set_xticklabels([])
# ax1_r.set_ylim(min(alpha_SL) * 0.75, max(alpha_SL)* 1.25)

    # ax.plot(LL, fit_vert(LL, *param), "--")
ax2_r.set_xlabel("$L$ ($\\AA$)")
# ax2.set_ylabel("$\\epsilon^{\\perp}_{\mathrm{SL}}$")
ax2_r.set_ylabel("$\\Delta \\alpha^{\\perp}_{\\mathrm{2D}}/(4\\pi \\varepsilon_0) (\\AA)$")
ax2_r.set_title("perp")
ax2_r.set_yscale("log")
# ax2.set_ylim(1, 4)
    # ax.set_title("{2}, ${{\\epsilon_{{2D}}^{{\\perp}} }}={1:.3f}$ ${{d}}={0:.3f}$".format(*param, names[i]))
# ax2_r.set_ylim(0.3, 0.8)
# ax2.set_xticklabels([])
# ax2.set_ylim(min(alpha_SL) * 0.75, max(alpha_SL)* 1.25)

ax2_r.legend()
# ax2.legend()

fig1.tight_layout()
fig1.savefig(os.path.join("../../tmp_img/", "all-para-error.svg"))
