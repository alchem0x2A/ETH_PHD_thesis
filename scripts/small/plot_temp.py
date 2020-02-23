import numpy as np
import matplotlib as mpl
from dill import loads
from . import data_path, img_path
import os.path
from os.path import join, exists, abspath, dirname
from matplotlib.colors import Normalize
from helper import gridplots
import scipy.signal
from scipy.signal import medfilt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import cos, arccos, pi, degrees, radians, exp

mpl.use("Agg")
mpl.rcParams["text.usetex"] = False
mpl.rcParams["svg.fonttype"] = "none"

def plot_T_id():
    import matplotlib.pyplot as plt
    import scipy
    from scipy.signal import medfilt
    import scipy.constants as const
    import numpy

    data_file = data_path / "temp" / "IdVG/temp-id-vg-20-100.csv"

    data = numpy.genfromtxt(data_file, delimiter=",",
                            skip_header=13)

    T = range(20, 101, 20)

    col = plt.cm.rainbow(numpy.linspace(0, 1, 5))

    Jds = []
    n_block = 201

    S = (95e-4) ** 2 * const.pi
    for i in range(len(T)) :
        Id = medfilt(numpy.abs(data[2 * i * n_block: (2 * i + 1) * n_block, 1]))
        J = Id / 1e-3 / S
        Jds.append(J)
    Vg = data[0 : n_block, 0]


    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    for i, J in enumerate(Jds):
        ax.plot(Vg, J, "-o", markersize=3,
                color=col[i],
                label="{}".format(T[i]))
    ax.set_xlabel("$V_{\\mathrm{g}}$ (V)")
    ax.set_ylabel("$J$ (mA$\cdot$cm$^{-2}$)")
    # ax.set_yscale("log")
    # ax.plot(T, Id_ratio, "-o")
    fig.savefig(img_path / "T_IdVg.svg")


def plot_T():
    import matplotlib.pyplot as plt
    import scipy
    from scipy.signal import medfilt
    import numpy
    import scipy.constants as const
    from scipy.optimize import curve_fit

    # data_file = "T-Id.csv"
    # scale = 1.57
    # data = numpy.genfromtxt(data_file, delimiter=",")

    # T = data[:, 0] + 273.15
    # Id_ratio = (data[:, 1] / data[0, 1] - 1) * scale + 1


    data_file = data_path / "temp" / "Vg-temp/summary.csv"
    data = numpy.genfromtxt(data_file,
                            skip_header=1,
                            delimiter=",")
    T = numpy.arange(20, 101, 10)
    Vg = numpy.arange(-100, 101, 25)
    sr = np.array([78, 50, 32.5, 24.3, 18.2, 15.3, 11.2, 9.7, 6.8])
    ratios = []

    col = plt.cm.rainbow(numpy.linspace(0, 1, 9))  # colomap


    for i, V in enumerate(Vg):
        I_data = data[:, 2 * i + 1 : 2* i +3]
        # print(I_data.shape)
        I_mean = numpy.mean(I_data, axis=1)
        ratio = I_mean / I_mean[0]
        ratio = 1 + (ratio - 1) / (numpy.max(ratio) - 1) * (sr[i] - 1)
        upper = I_data[:, 1] / I_mean[0]
        lower = I_data[:, 0] / I_mean[0]
        ratios.append((ratio, upper, lower))

    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)


    def I(T, phi):
        T = 273.15 + T
        return T ** 2* numpy.exp(-const.e * phi / (const.k * T))

    def ratio_func(T, phi):
        return I(T, phi) / I(20, phi)



    Vb = []
    for i, V in enumerate(Vg):
    # for i, V in zip([0], Vg[0:1]):
        ratio, upper, lower = ratios[i]
        phi_guess, _ = curve_fit(ratio_func, T, ratio, p0=0.2,
                              bounds=(0.05, 0.8), )
        Vb.append(phi_guess)
        c = col[i]
        l, = ax.plot(T, ratio,"o", markersize=4,
                     alpha=0.8,
                     color=c,
                     label="{} (V)".format(V))
        ax.plot(T, ratio_func(T, phi_guess),
                "--",
                color=c)
        # ax.fill_between(T, lower, upper,
                        # edgecolor=None,
                        # interpolate=True,
                        # color=l.get_c(),
                        # alpha=0.3)
    ax.set_ylabel("$I / I(T=293\ K)$")
    ax.set_xlabel("$T$ ($^\\circ C$)")
    ax.legend()
    ax.set_yscale("log")
    # ax.set_xscale("log")
    fig.savefig(img_path / "T_Vg.svg")

    from scipy.integrate import cumtrapz

    def e_cm2_to_SI(n):
        return n*const.e*10**4

    def SI_to_e_cm2(sigma):
        return sigma/const.e/10**4

    def EF_gr_from_sigma(sigma):
        v_f = 1.1e6
        A = scipy.sign(sigma)*const.hbar*v_f/const.e
        B = scipy.sqrt(scipy.pi*scipy.absolute(sigma)/const.e)
        return A*B

    def sigma_from_EF(EF):
        v_f = 1.1.e6
        return scipy.sign(EF)*EF**2*const.e**3/const.pi/const.hbar**2/v_f**2

    def sigma_from_sio2(V_M, sigma0=0, t=280e-9):
        eps_sio2 = 3.9
        Cox = const.epsilon_0*eps_sio2 / t
        # VM to be voltage applied to 2D surface
        return Cox*V_M + sigma0
    EF_gr = -4.6
    EF_F16 = -4.97
    V_cnp = 18
    V = Vg - V_cnp
    VV = numpy.linspace(-100, 100, 100) 
    V_gr = -EF_gr_from_sigma(sigma_from_sio2(V)) + (EF_gr - EF_F16)
    V_gr_smooth = -EF_gr_from_sigma(sigma_from_sio2(VV - V_cnp)) + (EF_gr - EF_F16)




    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.plot(Vg, Vb, "-s", markersize=6, label="EXP")
    l, = ax.plot(Vg, V_gr, "s", markersize=6)
    ax.plot(VV, V_gr_smooth, "--", color=l.get_c())
    ax.axvline(x=V_cnp, linestyle="--", alpha=0.8)
    ax.set_ylabel("Schottky barrier (eV)")
    ax.set_xlabel("$V_{\mathrm{G}}$ (V)")
    ax.legend()
    fig.savefig(img_path / "Vb_Vg.svg")

    fig = plt.figure(figsize=(3.5, 3.5))
    cm = plt.imshow([[], []], vmin=-100, vmax=100, cmap="rainbow")
    fig.colorbar(cm, ticks=numpy.arange(-100, 101, 50))
    fig.savefig(img_path / "I_Vg_cmap.svg")





def plot_main():
    plot_T()
    plot_T_id()


if __name__ == '__main__':
    print(("The script outputs Temperature plots into {} as svg!"
           " Do manual editing after").format(img_path.as_posix()))
    
    plot_main()
