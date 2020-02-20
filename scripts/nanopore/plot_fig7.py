import numpy as np
from . import data_path, img_path
from helper import gridplots, grid_labels, savepgf
from helper import get_color, add_cbar, add_img_ax
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata, interp2d
import pickle
from .utils import *


def plot_b(fig, ax):
    """
    Plot contour plot of surface rectification
    """
    quantities = ("V", "c_p", "c_n", "zflux_cp", "zflux_cn")
    Vg_all = [0.001, *np.arange(0.025, 0.251, 0.025)]
    Vg_true = []

    file_template = "{0}.npy"
    sigma_file_template = "sigma_{0}.txt"

    concentrations = (0.0001, 0.0002, 0.0005,
                      0.001, 0.002, 0.005,
                      0.01, 0.02, 0.05, 0.1)

    out_path = data_path / "FEM/concentration/1D/"

    # get the index of column in the new matrix
    def get_col_index(Vg, quantity):
        idx_V = Vg_all.index(Vg)
        idx_quant = quantities.index(quantity)
        len_quant = len(quantities)
        return 1 + len_quant * idx_V + idx_quant

    avg_tflux = []

    quant_flux = ("zflux_cp", "zflux_cn")

    res = []
    for conc in concentrations:
        tflux = []
        file_name = out_path / file_template.format(conc)
        data = np.load(file_name)
        data[:, 0] /= 1e-9
        r = data[:, 0]
        for V in Vg_all:
            avg = 0
            idx = get_col_index(V, "zflux_cp")
            avg += np.abs(np.mean(data[:, idx]))
            idx = get_col_index(V, "zflux_cn")
            avg += np.abs(np.mean(data[:, idx]))
            tflux.append(avg)
        rec_ = 1 - tflux / tflux[0]
        sigma_file = out_path / sigma_file_template.format(conc)
        sigma_data = np.genfromtxt(sigma_file, comments="%")
        V_ = sigma_data[:, 0]; sig_ = -sigma_data[:, 1]
        Vg_ = V_ + delta_phi_gr(sig_)
        [res.append((Vg_[i], conc, rec_[i])) for i in range(len(V_))]

    res = np.array(res)

    v = res[:, 0]
    lambda_d = Debye_length(res[:, 1]) / 1e-9
    rec = res[:, 2]
    cond = np.where(rec > 0)[0]
    print(cond)

    res[:, 1] = lambda_d

    v_uni = np.linspace(0.0, 1.25, 128)
    l_uni = np.linspace(min(lambda_d), 35, 128)
    vv, ll = np.meshgrid(v_uni, l_uni)
    z_uni = griddata(res[:, :2], rec,
                     (vv, ll), method="cubic",
                     fill_value=0)

    r_p = 10
    func_rect_2d = interp2d(v_uni * 1.085,
                            l_uni * 1.085 / r_p,
                            z_uni,
                            kind="cubic")
    with open(out_path / "rect_2d_intep.pickle", "wb") as f:
        pickle.dump(func_rect_2d, f)


    # Fix the issue with void after bicubic
    cs = ax.pcolor(vv * 1.085, ll * 1.085 / r_p, z_uni,
                   rasterized=True,
                   cmap="rainbow")
    # ax.colorbar()

    ax.set_xlim(0.1, 1.25)
    ax.set_ylim(0.1, 2.8)

    ax.set_xlabel("$V_{\\mathrm{G}}$ (V)")
    ax.set_ylabel("$\\lambda_{\\mathrm{D}} / r_{\\mathrm{G}}$")
    cax = inset_axes(ax, width="100%", height="100%",
                     bbox_to_anchor=(1.05, 0.02, 0.05, 0.5),
                     bbox_transform=ax.transAxes)
    cb = fig.colorbar(cs, cax=cax)
    cb.ax.set_title("$\\xi$")

def plot_a(fig, ax):
    """
    Plot Vg surface contour
    """

    # # Use C/m^2
    # def delta_phi_gr(sigma):
    #     fac = hbar * vf / e * np.sqrt(pi * np.abs(sigma) / e)
    #     return fac * np.sign(sigma)

    quantities = ("V", "c_p", "c_n", "zflux_cp", "zflux_cn")

    # get the index of column in the new matrix
    def get_col_index(Vg, quantity):
        idx_V = Vg_all.index(Vg)
        idx_quant = quantities.index(quantity)
        len_quant = len(quantities)
        return 1 + len_quant * idx_V + idx_quant

    Vg_true = []
    Vg_all = [0.001, *np.arange(0.025, 0.251, 0.025)]

    file_template = "{0}.npy"
    sigma_file_template = "sigma_{0}.txt"

    concentrations = (0.0001, 0.0002, 0.0005,
                      0.001, 0.002, 0.005,
                      0.01, 0.02, 0.05, 0.1)

    ratio_conc = 10**3

    out_path = data_path /  "FEM/concentration/1D/"

    res = []
    for i, conc in enumerate(concentrations):
        sigma_file = out_path / sigma_file_template.format(conc)
        sigma_data = np.genfromtxt(sigma_file, comments="%")
        V_ = sigma_data[:, 0]; sig_ = -sigma_data[:, 1]
        Vg_ = V_ + delta_phi_gr(sig_)
        [res.append((Vg_[i], conc, V_[i])) for i in range(len(V_))]

    res = np.array(res)

    v = res[:, 0]
    lambda_d = Debye_length(res[:, 1]) / 1e-9
    psi_g = res[:, 2]

    res[:, 1] = lambda_d

    v_uni = np.linspace(0.0, 1.5, 128)
    l_uni = np.linspace(min(lambda_d), max(lambda_d), 128)
    vv, ll = np.meshgrid(v_uni, l_uni)
    z_uni = griddata(res[:, :2], psi_g,
                     (vv, ll), method="cubic")

    r_p = 10

    ax.set_xlim(0, 1.25)
    ax.set_xlabel("$V_{\\mathrm{G}}$ (V)")
    ax.set_ylabel("$\\lambda_{\\mathrm{D}} / r_{\\mathrm{G}}$")
    ax.scatter(v[psi_g>0.001], lambda_d[psi_g>0.001] / r_p,
               c=psi_g[psi_g>0.001],
               s=25,
               cmap="rainbow_r", alpha=0.25)
    cs = ax.contour(vv, ll / r_p, z_uni,
                    cmap="rainbow_r",
                    levels=Vg_all, vmin=0.0, vmax=0.25)
    cax = inset_axes(ax, width="100%", height="100%",
                     bbox_to_anchor=(1.08, 0.02, 0.05, 0.5),
                     bbox_transform=ax.transAxes)
    # cb = fig.colorbar(cs, cax=cax)
    cb = add_cbar(fig, min=0, max=0.25, cax=cax)
    cb.ax.set_title("$\\psi_{\\mathrm{G}}$ (V)", pad=4)
    
def plot_main():
    w = 1.45
    fig, ax = gridplots(1, 3, r=0.95, ratio=2.2,
                        gridspec_kw=dict(width_ratios=(w, 3 - 2 * w, w )))
    ax[1].set_axis_off()
    plot_a(fig, ax[0])
    plot_b(fig, ax[2])
    grid_labels(fig, ax[[0, 2]])
    savepgf(fig, img_path / "fig-rejection-contour.pgf", preview=True)


if __name__ == '__main__':
    plot_main()
