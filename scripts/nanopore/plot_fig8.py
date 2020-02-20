import numpy as np
from . import data_path, img_path
from helper import gridplots, grid_labels, savepgf
from helper import get_color, add_cbar, add_img_ax
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata, interp2d
import pickle
from .utils import *

def plot_a(fig, ax):
    """
    Exp pore distr with lambda_D
    """
    lambda_D = Debye_length(c0=1.1e-4)
    print(lambda_D)
    rr = np.arange(10, 110)
    xx = lambda_D / ((rr + 10) * 1e-9) * 1.75

    func_rec_interp = pickle.load(open(data_path / "FEM/concentration/1D" \
                                       / "rect_2d_intep.pickle", "rb"))
    # Experimental
    file_name = data_path / "exp/pore-dist.csv"
    data = np.genfromtxt(file_name, delimiter=",")

    xi = func_rec_interp(1.25, xx)

    repeat = [d for d, n in data for _ in range(int(n * (d / 2) ** 2))]
    _, _, h = ax.hist(repeat, np.arange(5, 66, 5),
                density=True)
    ax2 = ax.twiny()
    ax3 = ax.twinx()
    arr_ticks = np.array([0.5, 1, 2, 5])
    ax2.set_xticks(lambda_D / arr_ticks * 2 / 1e-9)
    ax2.set_xticklabels(list(map(str, arr_ticks)))
    l, = ax3.plot(rr, xi.flat[::-1], color="r")
    ax.set_xlabel("Pore diameter (nm)")
    ax.set_ylabel("$w(r_{\mathrm{G}})$", color=h[0].get_fc())
    ax2.set_xlabel("$\lambda_{\\mathrm{D}} / r_{\\mathrm{G}}$")
    ax3.set_ylabel("$\\xi(r_{\\mathrm{G}})$", color=l.get_c())
    ax3.set_ylim(0, 0.98)
    ax.tick_params(axis='y', colors=h[0].get_fc())
    ax.text(x=0.02, y=0.6, ha="left",
            s="←", color=h[0].get_fc(), transform=ax.transAxes)
    ax.text(x=0.98, y=0.6, ha="right",
            s="→", color=l.get_c(), transform=ax.transAxes)
    ax3.tick_params(axis='y', colors=l.get_c())

    
def plot_b(fig, ax):
    """
    Plot rectification based on pore distribution
    """

    func_rec_interp = pickle.load(open(data_path / "FEM/concentration/1D" /
                                       "rect_2d_intep.pickle", "rb"))

    data_pore = np.genfromtxt(data_path / "exp/pore-dist.csv", delimiter=",")
    r_exp = data_pore[:, 0]
    w_exp = data_pore[:, 1] * r_exp ** 2
    w_exp = w_exp / np.sum(w_exp)            # Frequencies

    r_g = 20
    conc = 1.1e-4
    lambda_d = Debye_length(conc) / 1e-9

    v = np.linspace(0, 1.25, 128)
    # l = np.ones_like(v) *
    l = lambda_d / r_g
    xi_sim = func_rec_interp(v, l)

    conc = 2e-4
    lambda_d = Debye_length(conc) / 1e-9
    l_exp = lambda_d / r_exp
    xi_exp = func_rec_interp(v, l_exp)
    xi_exp = np.dot(w_exp, xi_exp)
    
    ax.plot(v, xi_sim.flat, label="Single pore")
    ax.plot(v, xi_exp.flat, label="With pore distribution")
    ax.set_xlabel("$V_{\\mathrm{G}}$ (V)")
    ax.set_ylabel("$\\xi$")
    ax.legend(loc="lower right", fontsize="small")


def plot_main():
    fig, ax = gridplots(1, 2, r=0.95, ratio=2.3)
    plot_a(fig, ax[0])
    plot_b(fig, ax[1])
    grid_labels(fig, ax, offsets=((0, 0),
                                  (0.06, 0)))
    savepgf(fig, img_path / "fig-pore-dist.pgf", preview=True)


if __name__ == '__main__':
    plot_main()
