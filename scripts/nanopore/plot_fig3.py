import numpy as np
from . import data_path, img_path
from helper import gridplots, grid_labels, savepgf
from helper import get_color, add_cbar, add_img_ax
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import linregress

def plot_a(fig, ax):
    """
    Plot experimental rectification of individual salts
    """
    def read(name):
        file_name = data_path / "exp" / "{0}.csv".format(name)
        err = None
        with open(file_name) as f:
            err = float(f.readline().strip())
        data = np.genfromtxt(file_name,
                             skip_header=1,
                             delimiter=",")
        data[:, 1] -= data[data[:, 0] == 0, 1]
        return data, err
    def convert(eta, delta=2.2):
        assert delta > 0
        xi_inv = (1 - eta) / (delta * eta + 1)
        return 1 - xi_inv

    name = "KCl"
    ax.axvline(x=0, ls="--", color="grey")
    ax.axhline(y=0, ls="--", color="grey")
    data, err = read(name)
    v = data[:, 0]
    eta = data[:, 1]
    ax.errorbar(data[:, 0], convert(eta),
                yerr=err,
                fmt="o",
                markersize=6)
    ax.set_ylim(-0.15, 0.7)
    ax.set_xlim(-1.25, 1.25)
    ax.set_xlabel("$V_{\\mathrm{G}}$ (V)")
    ax.set_ylabel("$\\xi$")
    ax.text(x=0.1, y=0.7,
            s="10$^{-4}$ mol$\\cdot{}$L$^{-1}$ KCl",
            size="small",
            transform=ax.transAxes)


def plot_b(fig, ax):
    """Plot conc KCL"""

    def read():
        file_name = data_path / "exp" / "rect_KCl_conc.csv"
        err = None
        with open(file_name) as f:
            err = float(f.readline().strip())
        data = np.genfromtxt(file_name,
                             skip_header=1,
                             delimiter=",")
        return data, err

    def convert(delta, eta):
        assert delta > 0
        xi_inv = (1 - eta) / (delta * eta + 1)
        return 1 - xi_inv
    
    delta0 = 2.2
    data, err = read()
    conc = data[:, 0]
    eta = data[:, 1]
    xi = convert(delta=delta0, eta=eta)
    xi_err_lo = xi - convert(delta=delta0, eta=eta - err / 2)
    xi_err_hi = convert(delta=delta0, eta=eta + err / 2) - xi
    log_x = np.log10(conc)
    log_y = np.log10(xi)
    ax.errorbar(conc, xi, yerr=(xi_err_lo, xi_err_hi), fmt="o",
                markersize=6)
    p = linregress(log_x, log_y)
    s = p.slope
    b = p.intercept
    xx = np.logspace(-4.45, -1.5, 100)
    yy = 10 ** (np.log10(xx) * s + b)
    ax.text(x=0.99, y=0.99,
            s=("$\\log_{10} \\xi_{\\mathrm{max}}"
               "= -2.38 - 0.52$"
               "log$_{10}$(c$_0 \\cdot{}$mol$^{-1} \\cdot{}$L)"
               "\n$R^{2}=0.98$"
               ),
            size="small",
            ha="right", va="top",
            transform=ax.transAxes)
    ax.plot(xx, yy, "--")
    # ax.xscale("log")
    ax.set_xlabel("$c_0$ (mol$\\cdot{}$L$^{-1}$)")
    ax.set_ylabel("$\\xi_{\\mathrm{max}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(3e-2, 1)

def plot_main():
    fig, ax = gridplots(1, 2, r=0.85, ratio=2.5)
    plot_a(fig, ax[0])
    plot_b(fig, ax[1])
    grid_labels(fig, ax)
    savepgf(fig, img_path / "fig-conductivity.pgf", preview=True)


if __name__ == '__main__':
    plot_main()
