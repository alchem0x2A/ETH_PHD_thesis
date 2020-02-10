import numpy as np
from . import data_path, img_path
from scipy.optimize import fsolve
from scipy.interpolate import BivariateSpline
from scipy.signal import medfilt
import scipy.constants as const
from helper import gridplots, grid_labels, savepgf
from helper import get_color, add_cbar, add_img_ax
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from .electronic import solve_stack, stack_charge, gen_input
from .electronic import k_SI_ecm, k_ecm_SI, eps0
from .electronic import get_tot_charge


def plot_a(fig, ax):
    """MQCM part"""
    k_Debye_SI = 3.34e-30
    a_cell = 13.06e-10

    def P_debye(sigma, d_t):
        A = (3 ** 0.5 / 2) * a_cell ** 2
        P_area = sigma * d_t
        return P_area * A / k_Debye_SI

    def chi_SI(sigma, d_t, E):
        P = sigma
        return P / eps0 / E

    eps = 2.5
    d0 = [3.3e-10, 4.9e-10, 6.2e-10]

    colors = ["#094eac", "#ff6633", "#91b96d"]
    styles = ["-o", "-^", "-s"]
    for n_gr in range(1, 4):
        for n_mos2 in range(1, 4):
            mat_in, dis_in, eps_in = gen_input(n_gr, n_mos2, eps, d0)
            E_ext = np.linspace(-1.5e9, 1.5e9, 50)
            d = d0[0] * (n_gr - 1) + d0[1] * 1 + d0[2] * (n_mos2 - 1)
            d = d
            tot_pot = []
            tot_chg = []

            for E in E_ext:
                phi_0 = solve_stack(mat_in, dis_in,
                                    eps_in, E)
                sigmas, phis, Es = stack_charge(phi_0,
                                                mat_in,
                                                dis_in,
                                                eps_in,
                                                E,
                                                use_solve=False)
                # print(sigmas)
                # chg = -sum(sigmas[0: n_gr])
                # tot_chg.append(get_tot_charge(sigmas))
                for i in range(len(phis)):
                    if np.sign(phis[0]) != np.sign(phis[i]):
                        break
                # print(i)
                tot_chg.append(sum(sigmas[0: i]))
                tot_pot.append(phis[0] - phis[-1])
            tot_chg = np.array(tot_chg)
            tot_pot = np.array(tot_pot)
            # print(tot_chg, len(tot_chg))
            # print(tot_pot, len(tot_pot))
            CQ = np.abs(np.gradient(tot_chg, tot_pot))

            l1 = ax.plot(E_ext / 10 ** 9, CQ * 10 ** 3,
                         styles[n_mos2 - 1],
                         color=colors[n_gr - 1],
                         label="{}L G / {}L MoS$_2$".format(n_gr, n_mos2))
        ax.set_xlabel("{\\itshape Ɛ}$_{\mathrm{ext}}$ (V$\\cdot{}$nm$^{-1}$)")
        ax.set_ylabel("$C_{\\mathrm{Q}}^{\\mathrm{MQCM}}$ (μF$\\cdot{}$cm$^{-2}$)")
        # ax.set_xlim(-1, 1)
        ax.legend(loc=0, handlelength=1,
                labelspacing=0.2,
                fontsize="small",)
        # ax.set_ylim(0, 50)

def plot_b(fig, ax):
    """DFT part"""
    dir_template = "{0}ms-{1}g/{2}Efield/EIG/{3}mos2-{4}graphene.EIG.{5:.3f}.dat"
    def get_single_DOS(n_mos, n_gr, charge, polarity=1):
        if polarity > 0:
            pol = "pos"
        else:
            pol = "neg"
        f_name = data_path / dir_template.format(n_mos, n_gr, pol,
                                                 n_mos, n_gr, charge)
        data = np.genfromtxt(f_name, skip_header=13, delimiter="")
        return data[:, 0], data[:, 2]

    def DOS_to_CQ(dos):
        a = 13.06e-10
        area = a ** 2 * (3 ** 0.5 / 2)
        return dos * const.e / area * 100
    
    possible_chg = [0.000, 0.001, 0.005, 0.010, 0.020, 0.030, 0.040, 0.050,
                    0.060, 0.070, 0.090, 0.100, 0.150, ]

    data = {}

    str_pol = {1: "pos",
               -1: "neg"}

    styles = ["-o", "-^", "-s"]
    colors = ["#094eac", "#ff6633", "#91b96d"]

    for ng in range(1, 4):
        for nm in range(1, 4):
            res = []
            for pol in [1, -1]:
                for chg in possible_chg:
                    file_name = data_path / dir_template.format(nm, ng, str_pol[pol],
                                                                nm, ng, chg)
                    print(file_name)
                    real_chg = chg * np.sign(pol)
                    if file_name.exists():
                        E, occu = get_single_DOS(nm, ng, chg, pol)
                        occu_Fermi = (occu[E < 0][-1] + occu[E > 0][0]) / 2
                        res.append([real_chg, occu_Fermi])
            res = np.array(res)
            if len(res) > 0:
                seq = np.argsort(res[:, 0])
                Es = res[:, 0][seq]
                CQs = res[:, 1][seq]
                ax.plot(- Es * 12, DOS_to_CQ(CQs),
                        styles[nm - 1],
                        color=colors[ng - 1],
                        label="{}MoS2-{}G".format(nm, ng))
    ax.set_xlim(-1, 1)
    ax.set_xlabel("{\\itshape Ɛ}$_{\\mathrm{ext}}$ (V$\\cdot{}$nm$^{-1}$)")
    ax.set_ylabel("$C_{\\mathrm{Q}}^{\\mathrm{DFT}}$ (μF$\\cdot{}$cm$^{-2}$)")



def plot_main():
    h = 1.1
    fig, ax = gridplots(1, 2, r=1.0, ratio=2.5)
    plot_a(fig, ax[0])
    plot_b(fig, ax[1])
    grid_labels(fig, ax)
    savepgf(fig, img_path / "qc-compare.pgf", preview=True)
    
    
if __name__ == '__main__':
    plot_main()
