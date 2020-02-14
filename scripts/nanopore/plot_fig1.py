import numpy as np
from . import data_path, img_path
from helper import gridplots, grid_labels, savepgf
from helper import get_color, add_cbar, add_img_ax
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_a(fig, ax):
    """
    Plot experimental slope
    """
    def get_time(string):
        string = string.decode("ascii")
        h, m, s = map(float, string.split(":"))
        return 3600 * h + 60 * m + 1 * s

    data = np.genfromtxt(data_path / "exp" / "diffusion-slope.csv",
                         converters={1: get_time},
                         delimiter=",")

    cond1 = np.where(data[:, 1] < 3600 * 1.5)
    cond2 = np.where(data[:, 1] > 3600 * 1.5)
    l1, = ax.plot(data[cond1][:, 1] / 3600, data[:, 2][cond1] / 1e-3)
    x1, y1 = data[cond1][-1, 1:3]
    x2, y2 = data[cond2][0, 1: 3]
    print(data[cond1][-1, 1:3])
    print(x1, x2)
    ax.plot((x1 / 3600, x2 / 3600), (y1 / 1e-3, y2 / 1e-3), lw=2)
    l2, = ax.plot(data[cond2][:, 1] / 3600, data[cond2][:, 2] / 1e-3,
                  color=l1.get_c())
    ax.plot()
    ax.set_xlabel("Time $t$ (h)")
    ax.set_ylabel("Conductivity LCR (μS$\\cdot{}$cm$^{-1}$)")
    # Labeling
    ax.fill_betweenx(y=ax.get_ylim(), x1=0, x2=x1 / 3600, lw=0,
                     color="green", alpha=0.15, zorder=0)
    ax.fill_betweenx(y=ax.get_ylim(), x1=x1 / 3600, x2=x2 / 3600, lw=0,
                     color="red", alpha=0.15, zorder=0)
    ax.fill_betweenx(y=ax.get_ylim(), x1=x2 / 3600, x2=data[-1, 1] / 3600,
                     lw=0,
                     color="green", alpha=0.15, zorder=0)
    for i in range(3):
        ax.text(x=1 / 6 + i / 3, y=0.05, s="$s_{{{0:d}}}$".format(i + 1),
                ha="center", va="center", transform=ax.transAxes)

    ax.text(x=1 / 6 + 0 / 3, y=0.7, s="No gating",
            ha="center", va="center", transform=ax.transAxes)
    ax.text(x=1 / 6 + 1 / 3, y=0.7, s="$V_{\\mathrm{G}}>0$",
            ha="center", va="center", transform=ax.transAxes)
    ax.text(x=1 / 6 + 2 / 3, y=0.7, s="No gating",
            ha="center", va="center", transform=ax.transAxes)

def convert_name(s):
    s = s.replace("2", "$_{2}$").replace("4", "$_{4}$")
    if "Fe" in s:
        s = "K$_{3}$[Fe(CN)$_{6}$]"
    return s

def plot_b(fig, ax):
    
    salts = ["KCl", "NaCl", "LiCl", "CaCl2", "MgSO4", "K2SO4", "KFeCN"]
    data = np.genfromtxt(data_path / "exp" / "diffuse-pcte-salt.csv",
                         delimiter=",",
                         skip_header=2)
    # Measured
    data = data / 1e-6
    print(data.shape)
    x = np.arange(len(salts))
    w = 1.0 / 6
    # bare pcte measure
    ax.bar(x - w * 3 / 2, data[:, 6], width=w,
           yerr=data[:, 7], color="#3aafe0",
           label="$\\mathbf{J}_{\\mathrm{PCTE}}$ Exp.")

    # Simulate
    ax.bar(x - w / 2, data[:, 3], width=w, color="#e0b13a",
           yerr=data[:, 5: 3: -1].T,
           label="$\\mathbf{J}_{\\mathrm{PCTE}}$ Model.")

    # Bare
    ax.bar(x + 1 / 2 * w, data[:, 1], width=w,
           yerr=data[:, 2], color="#d1473d",
           label="$\\mathbf{J}_{\\mathrm{PG0}}$ Exp.")

    ax.set_xticks(range(0, 7))
    ax.set_xlim(-0.5, 6.5)
    ax.set_xticklabels(map(convert_name, salts), ha="left", rotation=-45)
    ax.tick_params("x", pad=2)
    ax.legend(loc=0)
    ax.set_ylabel("$\\mathbf{J}$ (μmol$\\cdot{}$m$^{-2}\\cdot{}$s$^{-1}$)")


def plot_c(fig, ax):
    ax.set_axis_off()
    ax_img = inset_axes(ax, height="100%", width="100%",
                        bbox_to_anchor=(-0.1, -0.15, 1.1, 1.2),
                        bbox_transform=ax.transAxes)
    add_img_ax(ax_img, img_path / "raster" / "circuit.png")


def plot_d(fig, ax):
    """
    Plot estimated delta from experimental data
    """

    salts = ["KCl", "NaCl", "LiCl", "CaCl2",
             "MgSO4", "K3FeCN6", "K2SO4"]
    data = np.genfromtxt(data_path / "exp" / "diffuse-pcte-salt-delta.csv",
                         skip_header=2,
                         delimiter=",")
    exclude = ["LiCl", "MgSO4"]

    cond = [i for i, s in enumerate(salts) if s not in exclude]
    cond_salts = [s for i, s in enumerate(salts) if s not in exclude]
    c = data[cond, -3]
    low = data[cond, -2]
    up = data[cond, -1]
    delta = 2.20

    ax.errorbar(range(len(cond)), y=c, yerr=[c - low, up - c],
                fmt="s", label="Experiment")
    ax.axhline(y=delta, ls="--", label="Model", color="k", alpha=0.5)
    ax.set_xticks(range(len(cond)))
    ax.legend(loc=0)
    ax.set_xticklabels(map(convert_name, cond_salts), rotation=-45, ha="left")
    ax.tick_params("x", pad=2)
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylabel("$f_{\\mathrm{PCTE/PG0}}$")
    ax.set_ylim(0, 10)



def plot_main():
    fig, ax = gridplots(2, 2, r=1, ratio=1.25)
    plot_a(fig, ax[0])
    plot_b(fig, ax[1])
    plot_c(fig, ax[2])
    plot_d(fig, ax[3])
    grid_labels(fig, ax)
    savepgf(fig, img_path / "fig-rejection.pgf", preview=True)


if __name__ == '__main__':
    plot_main()
