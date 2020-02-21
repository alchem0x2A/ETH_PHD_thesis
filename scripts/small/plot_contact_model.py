import numpy as np
import matplotlib as mpl
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

#Plot contact angle and model


def Cassie(theta, r, f):
    c = r * cos(radians(theta)) * f + f - 1
    return degrees(arccos(c))


def Wenzel(theta, r):
    c = r * cos(radians(theta))
    return degrees(arccos(c))

def get_data(filename):
    data = np.genfromtxt(data_path / "CA" / filename, delimiter=",")
    return data


def plot_main():
    fig1, ax1 = gridplots(1, 1, r=0.4, ratio=1)
    fig2, ax2 = gridplots(1, 1, r=0.4, ratio=1)

    p_gr = get_data("coeff-f-gr.csv")
    p_sio2 = get_data("coeff-f-sio2.csv")
    gr_macro = get_data("gr-f16-macro.csv")[:, 1:]
    sio2_macro = get_data("sio2-f16-macro.csv")[:, 1:]
    gr_esem = get_data("gr-esem.csv")[:, 1:]
    sio2_esem = get_data("sio2-esem.csv")[:, 1:]


    k_gr, b_gr = p_gr
    func_gr = lambda x: k_gr*x + b_gr
    k_sio2, b_sio2 = p_sio2
    func_sio2 = lambda x: k_sio2*x + b_sio2

    f_gr = func_gr(np.array(gr_macro[0]))
    f_sio2 = func_gr(np.array(sio2_macro[0]))

    t_to_f = lambda x: func_gr(np.array(x))

    up_axis_gr = np.linspace(0, 750, 6)
    up_axis_sio2 = np.linspace(0, 150, 6)

    up_tick_gr = func_gr(up_axis_gr)
    # (up_axis_gr - b_gr)/k_gr
    up_tick_sio2 = func_sio2(up_axis_sio2)


    # (up_axis_sio2 - b_sio2)/k_sio2
    theta_R = 78
    theta_R_gr = 50
    theta_A = 107
    theta_A_gr = 103
    theta_Y = 94

    ff = np.linspace(0, 8400, 1000)
    rough_1 = np.linspace(1, 3.5, len(ff))
    rough_2 = np.linspace(1, 3.5, len(ff))
    f_fac_params = [0.15, 1, 1/1500]
    f_fac = f_fac_params[0] + (f_fac_params[1] - f_fac_params[0]) * exp(-ff * f_fac_params[2])
    # f_fac = f_fac[::-1]
    # f_fac = np.linspace(1, 0.2, len(ff))
    print(ff[:11], f_fac[:11])

    #print(gr_macro[1])

    colors = ["#245AA4", "#F75B73", "#FF9400", "#2F6B16"]

    #plot gr-F16CuPc
    ax12 = ax1.twiny()
    ax1.plot(f_gr, gr_macro[2], "^", label="Advancing-Gr-F$_{16}$CuPc",
             color=colors[0],
    )
    # ax1.plot(t_to_f(gr_macro[0]), gr_macro[1], "s-", label="Static-Gr-F$_{16}$CuPc", color=colors[0])
    ax1.plot(f_gr, gr_macro[3], "v", label="Receding-Gr-F$_{16}$CuPc",
             color=colors[1],
    )

    ax1.plot(t_to_f(gr_esem[0]), gr_esem[1], "H", label="ESEM-Gr-F$_{16}$CuPc", color=colors[3])


    ax1.plot(ff, Cassie(theta_A_gr, rough_2, f_fac), "-.")
    ax1.plot(ff, Cassie(theta_R_gr, rough_2, f_fac), "-.")
    ax1.plot(ff, Cassie(theta_Y, rough_2, f_fac), "-.")

    # ax1.set_xlim([-10, 760])
    ax1.set_ylim([35, 165])

    # ax1.set_xlabel("QCM Value [nm]")
    ax1.set_xlabel("$\Delta f$ of QCM (Hz)")
    ax1.set_ylabel(r"Contact Angle ($^\circ$)")
    ax1.legend(loc=0)


    ax12.set_xticks(list(up_tick_gr))
    ax12.set_xticklabels(["%d"%i for i in up_axis_gr])
    ax12.set_xlim(ax1.get_xlim())
    ax12.set_xlabel("Thickness [nm]")

    fig1.savefig(img_path / "gr-f16-macro-esem.svg")



    #plot sio2-f16
    ax22 = ax2.twiny()

    ax2.plot(f_sio2, sio2_macro[2], "^", label="Advancing-SiO$_{2}$-F$_{16}$CuPc",
            color=colors[0],
    )
    # ax2.plot(t_to_f(sio2_macro[0]), sio2_macro[1], "s-", label="Static-SiO$_{2}$-F$_{16}$CuPc", color=colors[1])
    ax2.plot(f_sio2, sio2_macro[3], "v", label="Receding-SiO$_{2}$-F$_{16}$CuPc",
             color=colors[1],
    )

    ax2.plot(t_to_f(sio2_esem[0]), sio2_esem[1], "H", label="ESEM-SiO$_{2}$-F$_{16}$CuPc", color=colors[3])

    ax2.plot(ff, Wenzel(theta_A, rough_1), "-.")
    ax2.plot(ff, Wenzel(theta_R, rough_1), "-.")
    ax2.plot(ff, Wenzel(theta_Y, rough_1), "-.")

    # ax2.set_xlabel("QCM Value [nm]")
    # ax2.set_ylabel(r"Contact Angle [$^\circ$]")
    ax2.set_xlabel("$\Delta f$ of QCM (Hz)")
    ax2.set_ylabel(r"Contact Angle ($^\circ$)")
    ax2.legend(loc=0)

    # ax2.set_xlim([-10, 760])
    ax22.set_ylim([15, 165])


    ax22.set_xticks(list(up_tick_sio2))

    ax22.set_xticklabels(["%d"%i for i in up_axis_sio2])
    ax22.set_xlim(ax2.get_xlim())

    ax22.set_xlabel("Thickness [nm]")

    fig2.savefig(img_path / "sio2-f16-macro-esem.svg")





if __name__ == '__main__':
    print(("The script outputs contact angle models into {} as svg!"
           " Do manual editing after").format(img_path.as_posix()))
    
    plot_main()
