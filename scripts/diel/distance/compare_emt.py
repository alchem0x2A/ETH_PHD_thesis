import numpy
from scipy.optimize import curve_fit, fsolve
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

limits = {
    "mos2": (4.98, 6.15),
    "mose2": (5.60, 6.46),
    "mote2": (6.12, 6.98),
    "ws2": (5.00, 6.15),
    "wse2": (5.42, 6.49),
    "wte2": (6.33, 7.06),
}

# limits = {
    # "mos2": (5.22, 6.15),
    # "mose2": (5.73, 6.46),
    # "mote2": (6.37, 6.98),
    # "ws2": (5.20, 6.15),
    # "wse2": (5.75, 6.49),
    # "wte2": (6.38, 7.06),
# }

raw_data_para = dict()
raw_data_perp = dict()
fit_all_para = dict()               # eps_2D, delta
fit_all_perp = dict()               # eps_2D, delta

def combine_data():
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
            L = data[:, 0]
            eps_SL = data[:, 1]
            if "par" in f_path:
                raw_data_para[names[i]] = (L, eps_SL)
                param, _ = curve_fit(fit_para, L[1:], eps_SL[1:],
                                     p0=(5, 10),
                                     bounds=((0.5, 1.0),
                                             (10, 50))
                )
                fit_all_para[names[i]] = param
            elif "perp" in f_path:
                raw_data_perp[names[i]] = (L, eps_SL)
                param, _ = curve_fit(fit_vert, L[1:], eps_SL[1:],
                                     p0=(5, 10),
                                     bounds=((0.5, 1.0),
                                             (10, 50))
                )
                fit_all_perp[names[i]] = param

combine_data()

print(fit_all_para, fit_all_perp)

fig, ax = plt.subplots(1, 3, figsize=(8.5, 2.4))

def plot_diff():
# ax 0: diff
    diff = []
    name_disp = []
    for n, name in tags.items():
        delta_para, _ = fit_all_para[n]
        delta_perp, _ = fit_all_perp[n]
        name_disp.append("2H-" + name)
        # diff.append(delta_para / delta_perp)
        diff.append((delta_para - delta_perp) / delta_perp * 100)
    ax[0].axhline(y=0, alpha=0.8)
    ax[0].bar(range(len(name_disp)), diff, width=0.6)
    ax[0].set_xticks(range(len(name_disp)))
    ax[0].set_xticklabels(name_disp, rotation=-30)
    ax[0].set_ylabel("Estimation Error (%) of $\\delta_{\\mathrm{2D}}$")
# ""between $\\delta_{\\mathrm{2D}}^{{\parallel}, \\mathrm{fit}}$  and $\\delta_{\\mathrm{2D}}^{\\perp, \\mathrm{fit}}$ ($\\mathrm{\\AA{}}$)")


def plot_type1():
    # ax 1:
    name_disp = []
    for n, name in tags.items():
        L, eps_para = raw_data_para[n]
        L = L[6]
        eps_para = eps_para[6]
        delta_para, _ = fit_all_para[n]
        # name_disp.append("2H-" + name)
        dmin, dmax = limits[n]
        dd_ = numpy.linspace(dmin, dmax, 256)
        xx =  (dd_ - dmin) / (dmax - dmin)                     # normalized xx
        f = dd_ / L
        eps_2D_para = (eps_para + f - 1) / f
        x_mark = (delta_para - dmin) / (dmax - dmin)
        print(name, delta_para, dmin, dmax)
        f_p = delta_para / L
        eps_2D = (eps_para + f_p - 1) / f_p

        ax[1].plot(xx, eps_2D_para, label=name)
        ax[1].plot(x_mark, eps_2D, "*")


    for n, name in tags.items():
        L, eps_perp = raw_data_perp[n]
        L = L[6]
        eps_perp = eps_perp[6]
        delta_perp, _ = fit_all_perp[n]
        name_disp.append("2H-" + name)
        dmin, dmax = limits[n]
        dd_ = numpy.linspace(dmin, dmax, 256)
        xx =  (dd_ - dmin) / (dmax - dmin)                     # normalized xx
        f = dd_ / L
        eps_2D_perp = f / (1 / eps_perp + f - 1)
        x_mark = (delta_perp - dmin) / (dmax - dmin)
        f_p = delta_perp / L
        eps_2D = 1 / ((1 / eps_perp + f_p - 1) / f_p)
        ax[2].plot(xx, eps_2D_perp, label=name)
        ax[2].plot(x_mark, eps_2D, "*")

def plot_type2():
    # ax 1:
    name_disp = []
    for n, name in tags.items():
        L, eps_para = raw_data_para[n]
        L = L[6]
        eps_para = eps_para[6]
        delta_para, _ = fit_all_para[n]
        delta_perp, _ = fit_all_perp[n]
        delta_avg = (delta_para + delta_perp) / 2
        name_disp.append("2H-" + name)
        # xx = numpy.linspace(-0.25, 0.25, 256)
        # dd_ = xx + delta_avg
        diff = 7.5
        xx = numpy.linspace(-diff / 100, diff / 100, 256)
        dd_ = (1 + xx) * delta_avg
        f = dd_ / L
        eps_2D_para = (eps_para + f - 1) / f

        l, = ax[1].plot(xx * 100, eps_2D_para, label="2H-" + name)
        ax[1].plot(xx[::20] * 100, eps_2D_para[::20], "o", markersize=4, color=l.get_c())


    for n, name in tags.items():
        L, eps_perp = raw_data_perp[n]
        L = L[6]
        eps_perp = eps_perp[6]
        delta_perp, _ = fit_all_perp[n]
        delta_para, _ = fit_all_para[n]
        # delta_avg = (delta_para + delta_perp) / 2
        delta_avg = delta_perp
        name_disp.append("2H-" + name)
        name_disp.append("2H-" + name)
        diff = 7.5
        xx = numpy.linspace(-diff / 100, diff / 100, 256)
        dd_ = (1 + xx) * delta_avg
        # xx = numpy.linspace(-0.25, 0.25, 256)
        # dd_ = xx + delta_avg
        f = dd_ / L
        eps_2D_perp = f / (1 / eps_perp + f - 1)
        cond = numpy.where((eps_2D_perp > 0) & (eps_2D_perp < 1000))
        l, = ax[2].plot(xx[cond] * 100, eps_2D_perp[cond], label="2H-" + name)
        ax[2].plot(xx[::20] * 100, eps_2D_perp[::20], "o", markersize=4, color=l.get_c())

# ax[1].set_ylim(14, 23)
ax[1].set_xlabel("Uncertainty of $\\delta^{*}_{\\mathrm{2D}}$ (%)")
ax[1].set_ylabel("Estimated $\\varepsilon_{\\mathrm{2D}}^{\\parallel}$")

ax[2].set_xlim(-7.5, 7.5)
ax[2].set_ylim(15, 500)
ax[2].set_xlabel("Uncertainty of $\\delta^{*}_{\\mathrm{2D}}$ (%)")
ax[2].set_ylabel("Estimated $\\varepsilon_{\\mathrm{2D}}^{\\perp}$")

plot_diff()
plot_type2()
# ax[0].set_ylabel("")
# ax[1].set_ylim(10, 25)
ax[1].legend()
# ax[2].set_yscale("log")
    
# ax[0].set_xticklabels(name_disp)

fig.tight_layout()
fig.savefig("../../tmp_img/emt_res.svg")

