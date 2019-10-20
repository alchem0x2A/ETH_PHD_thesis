import numpy
from ..utils.kkr import kkr, matsubara_freq
from ..utils.lifshitz import alpha_to_eps, g_amb_alpha, g_amb
from ..utils.lifshitz import g_amb_alpha_part, g_amb_part, transparency_single
from ..utils.eps_tools import get_alpha, get_eps, data_2D, data_bulk, get_index
from ..utils.img_tools import get_color, add_cbar
from scipy.interpolate import interp1d
from . import data_path, img_path
from helper import gridplots, grid_labels

def omega_Eg_model(Eg, omega_p=None):
    if omega_p is None:
        omega_p = 10
    omega_g = 1.086 * Eg + 2.377
    return omega_p, omega_g

def eps_osc(Eg, freq, omega_p=None, trans_func=omega_Eg_model, Gamma=0.05):
    # freq = matsubara_freq(numpy.arange(0, 1000),
                             # mode="energy")
    omega_p, omega_g = trans_func(Eg, omega_p)
    eps_ = 1 + omega_p ** 2 / (omega_g ** 2 + freq ** 2 - freq * Gamma)
    eps_all = numpy.vstack([eps_,] * 3)
    return eps_all

def get_energy_freq_dep(mater_2D, mater_bulk_a,
                        mater_bulk_b=None, d=1e-9):
    eps_a, freq_matsu, *_ = get_eps(mater_bulk_a)
    if mater_bulk_b is  None:
        eps_b = eps_a
    else:
        eps_b, *_ = get_eps(mater_bulk_b)
    alpha, freq_alpha, *_ = get_alpha(mater_2D)
    print(eps_a.shape, eps_b.shape)
    g_part = g_amb_alpha_part(eps_a, alpha, eps_b,
                              freq_matsu, freq_alpha, d)
    return g_part, freq_matsu


def get_eps_data(mater_m=("C2", "C"), mater_a="Au-exp", d=0.8e-9):

    ind_a = get_index(mater_a, kind="bulk")
    ind_m = get_index(mater_m, kind="2D")

    eps_a, freq_, *_ = get_eps(ind_a)
    eps_b = eps_osc(0.1, freq_, omega_p=3.0)
    eps_v = numpy.ones_like(eps_a)
    alpha_m, freq_alpha, *_ = get_alpha(ind_m)

    # Total enegy
    g_ = g_amb_alpha(eps_a, alpha_m, eps_b, freq_, freq_alpha, d)
    g_v_ = g_amb(eps_a, eps_v, eps_b, freq_,  d)
    
    # Partial
    g_part = g_amb_alpha_part(eps_a, alpha_m, eps_b, freq_, freq_alpha, d)
    g_part_v = g_amb_part(eps_a, eps_v, eps_b, freq_,  d)


    # Save eps plot
    eps_m_x = kkr(freq_alpha, alpha_to_eps(alpha_m[0], d, direction="x"), freq_)
    eps_m_z = kkr(freq_alpha, alpha_to_eps(alpha_m[-1], d, direction="z"), freq_)    
    eps_m = numpy.sqrt(eps_m_x * eps_m_z)


    return freq_, g_part, g_part_v, eps_a[0], eps_b[0], eps_m

def plot_main():
    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["svg.fonttype"] = "none"
    fig, ax = gridplots(2, 1, r=0.5, ratio=0.8)

    # Au-gr case
    mater_a = "Au-exp"
    freq_, g_part, g_part_v, eps_a, eps_b, eps_m = get_eps_data(mater_a=mater_a)
    ax[0].plot(freq_, eps_a)
    ax[0].plot(freq_, eps_m)
    ax[0].plot(freq_, eps_b)
    l, = ax[1].plot(freq_, -g_part * 1e6, "o", alpha=1, markersize=3)
    ax[1].plot(freq_, -g_part_v * 1e6, "s", alpha=0.5, color=l.get_c(), markersize=3)
    

    mater_a = "SiO2-exp"
    freq_, g_part, g_part_v, eps_a, eps_b, eps_m = get_eps_data(mater_a=mater_a)
    ax[0].plot(freq_, eps_a)
    # ax[0].plot(freq_, eps_m)
    # ax[0].plot(freq_, eps_b)
    l, = ax[1].plot(freq_, -g_part * 1e6, "o", alpha=1, markersize=3)
    ax[1].plot(freq_, -g_part_v * 1e6, "s", alpha=0.5, color=l.get_c(), markersize=3)

    ax[0].set_xscale("log")
    ax[0].set_xticklabels([])
    ax[0].tick_params(bottom=False, labelbottom=False)
    ax[1].set_xscale("log")

    ax[0].set_ylim(0.5, 10)
    ax[1].axhline(y=0, ls='--', color="grey")
    ax[1].set_ylim(-10, 10)
    ax[1].set_xlabel("Xlabel")
    ax[0].set_ylabel("ylabel")
    ax[1].set_ylabel("ylabel")

    fig.savefig(img_path / "test_repul.svg")
    
    

if __name__ == "__main__":
    plot_main()
