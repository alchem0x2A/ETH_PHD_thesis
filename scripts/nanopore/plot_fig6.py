import numpy as np
from . import data_path, img_path
from helper import gridplots, grid_labels, savepgf
from helper import get_color, add_cbar, add_img_ax
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from .utils import *


def plot_ab(fig, ax):
    ax_a, ax_b = ax

    quantities = ("V", "c_p", "c_n", "zflux_cp", "zflux_cn")

    Vg_all = [0.001, *np.arange(0.025, 0.251, 0.025)]
    file_template = "{0}.npy"
    sigma_file_template = "sigma_{0}.txt"

    concentrations = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1)

    ratio_conc = 10**3

    out_path = data_path / "FEM/concentration/1D"

    # get the index of column in the new matrix
    def get_col_index(Vg, quantity):
        idx_V = Vg_all.index(Vg)
        idx_quant = quantities.index(quantity)
        len_quant = len(quantities)
        return 1 + len_quant * idx_V + idx_quant
    # Use C/m^2

    plot_quantities = ("zflux_cp", "zflux_cn", "c_p", "c_n")
    yl = dict(zflux_cp="$\\mathbf{J}_{z+}$ (mol$\\cdot{}$m$^{-2} \\cdot{}$s$^{-1}$)",
              zflux_cn="$\\mathbf{J}_{z-}$ (mol$\\cdot{}$m$^{-2} \\cdot{}$s$^{-1}$)",
              c_p="$c_{+}$",
              c_n="$c_{-}$")

    r0 = 10
    conc = 0.001
    file_name = out_path / file_template.format(conc)
    data = np.load(file_name)
    data[:, 0] /= 1e-9
    r = data[:, 0]              # r distance
    # ax_a

    def plot_individual(ax, quant):
        # quant = "zflux_cp"
        lines = []
        for V in Vg_all:
            y = data[:, get_col_index(V, quant)]
            lines.append(ax.plot(np.hstack([-r[::-1], r]) / r0,
                                 np.hstack([y[::-1], y]))[0])
        sigma_file = out_path / sigma_file_template.format(conc)
        sigma_data = np.genfromtxt(sigma_file, comments="%")
        Vg_true = sigma_data[:, 0] + delta_phi_gr(-sigma_data[:, 1])  # True Vg
        for vg, line in zip(Vg_true, lines):
            print(vg)
            line.set_color(get_color(vg, min=0, max=Vg_true.max()))
        ax.set_xlabel("$r/r_{\\mathrm{G}}$")
        ax.set_ylabel(yl[quant])
    plot_individual(ax_a, "zflux_cp")
    plot_individual(ax_b, "zflux_cn")
    ax_a.text(x=0.02, y=0.02, ha="left", va="bottom",
              s="K$^{+}$", transform=ax_a.transAxes)
    ax_b.text(x=0.02, y=0.02, ha="left", va="bottom",
              s="Cl$^{+}$", transform=ax_b.transAxes)
    cax_b = inset_axes(ax_b, width="100%", height="100%",
                       bbox_to_anchor=(0.5, 0.08, 0.05, 0.5),
                       bbox_transform=ax_b.transAxes)
    add_cbar(fig, cax=cax_b, min=0, max=1.25)
    cax_b.set_title("$V_{\\mathrm{G}}$ (V)")


def plot_c(fig, ax):
    quantities = ("V", "c_p", "c_n", "zflux_cp", "zflux_cn")
    units = ("V", "mol/m$^{3}$", "mol/m$^{3}$", "mol/(m$^{2}$*s)", "mol/(m$^{2}$*s)")
    Vg_all = [0.001, *np.arange(0.025, 0.251, 0.025)]
    Vg_true = []
    file_template = "0.{0}.npy"
    sigma_file_template = "sigma_0.{0}.txt"
    concentrations = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1)
    radii = (5, 10, 15, 20)
    ratio_conc = 10 ** 3
    out_path = data_path / "FEM/radius/10/1D"

    # get the index of column in the new matrix
    def get_col_index(Vg, quantity):
        idx_V = Vg_all.index(Vg)
        idx_quant = quantities.index(quantity)
        len_quant = len(quantities)
        return 1 + len_quant * idx_V + idx_quant

    # conc_plot = [0.001]
    avg_tflux = []
    avg_nflux = []
    avg_pflux = []
    quant_flux = ("zflux_cp", "zflux_cn")
    conc = 0.001
    conc_base = str(conc).split(".")[-1]
    print(conc_base)
    tflux = []; nflux = []; pflux = []
    file_name = out_path / file_template.format(conc_base)
    data = np.load(file_name)
    data[np.isnan(data)] = 0
    data[:, 0] /= 1e-9
    r = data[:, 0]
    for V in Vg_all:
        avg = 0
        idx = get_col_index(V, "zflux_cp")
        avg += np.abs(np.mean(data[:, idx]))
        pflux.append(np.abs(np.mean(data[:, idx])))
        idx = get_col_index(V, "zflux_cn")
        avg += np.abs(np.mean(data[:, idx]))
        nflux.append(np.abs(np.mean(data[:, idx])))
        tflux.append(avg)
    avg_tflux.append(tflux)
    avg_pflux.append(pflux)
    avg_nflux.append(nflux)
    sigma_file = out_path / sigma_file_template.format(conc_base)
    sigma_data = np.genfromtxt(sigma_file, comments="%")
    V_ = sigma_data[:, 0]; sig_ = -sigma_data[:, 1]
    Vg_true.append(V_ + delta_phi_gr(sig_))

    avg_tflux = np.array(avg_tflux)
    avg_nflux = np.array(avg_nflux)
    avg_pflux = np.array(avg_pflux)

    i = 0
    ax.plot(Vg_true[i], np.abs(avg_tflux[i, :]), "-o",
             markersize=5,
             label="Total")
    ax.plot(Vg_true[i], np.abs(avg_nflux[i, :]), "-o",
             markersize=5,
             label="K$^{+}$", alpha=0.5)
    ax.plot(Vg_true[i], np.abs(avg_pflux[i, :]), "-o",
             markersize=5,
             label="Cl$^{-}$", alpha=0.5)
    ax.set_xlabel("$V_{\\mathrm{G}}$ (V)")
    ax.set_ylabel("|$\\mathbf{J}_{z}$| (mol$\\cdot{}$m$^{-2} \\cdot{}$s$^{-1}$)")
    ax.legend(loc=0)


def plot_main():
    fig, ax = gridplots(1, 3, r=1, ratio=3)
    plot_ab(fig, ax[:2])
    plot_c(fig, ax[-1])
    grid_labels(fig, ax, offsets=((0, 0),
                                  (0.01, 0),
                                  (0.02, 0)))
    savepgf(fig, img_path / "fig-average-rejection.pgf", preview=True)


if __name__ == '__main__':
    plot_main()
