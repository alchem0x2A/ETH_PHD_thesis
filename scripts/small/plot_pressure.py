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

def plot_model():
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy
    import scipy
    from numpy import sin, cos, tan, pi
    from scipy.optimize import fsolve

    gamma0 = 485e-3
    g = 9.81
    rho = 13.6e3

    # V0 = 1e-10
    # theta0 = 165 / 180 * pi

    def partial_delta(delta, theta1, theta2):
        # print(scipy.degrees([theta1, theta2]))
        # assert pi / 2 < theta1 <= pi
        # assert pi / 2 < theta2 <= pi
        delta1 = delta * cos(theta1) / (cos(theta1) + cos(theta2))
        delta2 = delta * cos(theta2) / (cos(theta1) + cos(theta2))
        r = -delta / (cos(theta2) + cos(theta1))
        return delta1, delta2, r

    def f1(theta):
        res = theta - pi / 2 + sin(2 * theta - pi) / 2 + 2 * sin(theta) * cos(theta)
        return res

    def f2(theta):
        # assert pi/2 < theta < pi
        res = tan(theta) * (theta - pi / 2 + sin(2 * theta - pi) / 2 + 2 * cos(theta)) \
              - 1 / 3 * (cos(theta)) ** 2 + (1 - sin(theta)) ** 2
        return res

    def V_sym(R, delta, theta):     # Definition of delta only as half!
        _, _, r = partial_delta(2 * delta, theta, theta)
        a = R + delta * (1 - sin(theta)) / cos(theta)
        a_ = R - r + r * sin(theta)
        # print(a, a_)

        # assert abs((a - a_) / a) < 1e-5
        V = 2 * pi * a **2 * delta + 2* pi * (delta / cos(theta)) ** 2 * (a * f1(theta) + delta * f2(theta))
        return V

    def solve_V_two_angles(V, delta, theta1, theta2):
        delta1, delta2, r = partial_delta(delta, theta1, theta2)

        def _target(R):
            V1 = V_sym(R, delta1, theta1)
            V2 = V_sym(R, delta2, theta2)
            return (V1 + V2) / 2 - V
        R_solution,  = fsolve(_target, delta)
        a1 = R_solution + delta * (1 - sin(theta1)) / (cos(theta1) + cos(theta2))
        a2 = R_solution + delta * (1 - sin(theta2)) / (cos(theta1) + cos(theta2))
        # r = -delta / (cos(theta1) + cos(theta2))
        return R_solution, r, a1, a2


    def max_delta(V, theta):
        return (3 * V / 4 / pi) ** (1 / 3) * sin(theta - pi / 2)



    def curv_pressure(r, R, gamma):
        return gamma * (1 / r + 1/ R)

    def curv_pressure_gravity(r, R, h, gamma):
        return gamma * (1 / r + 1/ R) + rho * h * g




    file_name = data_path / "pressure" / "steps3_new_res.csv"
    data = numpy.genfromtxt(file_name, delimiter=",",
                            skip_header=1)
    # print(data)
    R1 = data[1:, 2] * 1e-6 * 2 /3
    R2= data[1:, 3]* 1e-6 * 2/ 3
    p=curv_pressure(R1, R2, gamma0) * 0.6

    # print(p - p[0])

    H0 = data[0, -2] * 1e-6
    theta1_ = (data[0, -4] + 5 )/ 180 * pi
    theta2_ = data[0, -3] / 180 * pi
    r_t = data[0, 1] * 1e-6
    R0 = H0 / 2

    V0 = 4 / 3 * pi * R0 ** 3
    theta_1s = scipy.radians(data[1: , -4])
    theta_2s = scipy.radians(data[1:, -3])
    H_s = data[1:, -2] * 1e-6

    ps = []
    Rs = []
    a1_s = []
    a2_s = []
    rs = []
    ds = []


    for i in range(len(H_s)):
        R, r, a1, a2 = solve_V_two_angles(V0,
                                          H_s[i],
                                          theta_1s[i],
                                          theta_2s[i])
        print(H_s[i], R, r, a1, a2)
        ds.append(list(partial_delta(H_s[i],
                                     theta_1s[i],
                                     theta_2s[i])))
        Rs.append(R)
        rs.append(r)
        # print(R, r)
        a1_s.append(a1)
        a2_s.append(a2)
        ps.append(curv_pressure(r, R, gamma0))


    H_theory = numpy.linspace(578, 450, 100) * 1e-6
    theta_1s_theory = numpy.radians(numpy.linspace(147, 145, 100))
    theta_2s_theory = numpy.radians(numpy.linspace(159, 155, 100))

    ps_theory = []
    Rs_theory = []
    a1_s_theory = []
    a2_s_theory = []
    rs_theory = []
    ds_theory = []

    for i in range(len(H_theory)):
        R, r, a1, a2 = solve_V_two_angles(V0,
                                          H_theory[i],
                                          theta_1s_theory[i],
                                          theta_2s_theory[i])
        # print(H_theory[i], R, r, a1, a2)
        ds_theory.append(list(partial_delta(H_theory[i],
                                            theta_1s_theory[i],
                                            theta_2s_theory[i])))
        Rs_theory.append(R)
        rs_theory.append(r)
        # print(R, r)
        a1_s_theory.append(a1)
        a2_s_theory.append(a2)
        ps_theory.append(curv_pressure(r, R, gamma0))


    Rs = numpy.array(Rs)
    rs = numpy.array(rs)
    a1_s = numpy.array(a1_s)
    a2_s = numpy.array(a2_s)
    ps = numpy.array(ps)
    ps = ps - ps[0]
    ds = numpy.array(ds)

    Rs_theory = numpy.array(Rs_theory)
    rs_theory = numpy.array(rs_theory)
    a1_s_theory = numpy.array(a1_s_theory)
    a2_s_theory = numpy.array(a2_s_theory)
    ps_theory = numpy.array(ps_theory)
    ps_theory = ps_theory - ps_theory[0]
    ds_theory = numpy.array(ds_theory)

    print(V0 / 1e-9)

    # fig = plt.figure(figsize=(3, 6))
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False,
                                   figsize=(4, 2.5))

    strain = (H_s[0] - H_s) / H_s[0]
    strain_theory = (H_theory[0] - H_theory) / H_theory[0]
    I = data[1:, -1]
    I_ratio_exp = I / I[0]
    # rb = data[1:, 0]
    I_ratio_theory =  ( a2_s_theory / a2_s_theory[0]) ** 3
    p_theory = ps_theory
    p_exp = p - p[0]



    ax2.plot(strain, I_ratio_exp, "-o",
             label="ratio_exp",
             markersize=4)
    ax2.plot(strain_theory, I_ratio_theory, "-",
             label="ratio_theory",
             markersize=4)
    ax2.set_ylim(1, 5.9)
    ax2.set_xlim(0, max(strain))
    ax2.legend()

    ax1.plot(strain, p_exp, "-o",
             label="p_exp",
             markersize=4)
    ax1.plot(strain_theory, p_theory,
             "-",
             label="p_theory",
             markersize=4)
    ax1.set_ylim(0, 200)
    ax1.set_xlim(0, max(strain))
    ax1.legend()

    fig.savefig(img_path / "pressure_calc.svg")


def plot_exp():
    import scipy
    import matplotlib.pyplot as plt
    from scipy.signal import medfilt
    from dill import load
    import numpy

    file_name = data_path / "pressure" / "pressure_6.csv"
    fil, mm = load(open(data_path / "pressure" / "fil.dil", "rb"))


    p = numpy.array([0, 35.5, 65.2, 87.8, 113.6, 127.6])
    r = numpy.array([76, 113, 139, 149, 166, 200]) * 2/3 * 1e-6



    col = plt.cm.jet(numpy.linspace(0, 1, 150))

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r"$V_{\mathrm{g}}$ (V)")
    ax.set_ylabel(r"$I_{\mathrm{d}}$ (A)")
    # ax.set_yscale("log")
    raw_data = numpy.genfromtxt(file_name, delimiter=",", skip_header=13)

    space = 201
    ratio = []

    for i in range(6):
        di = i * 2
        data = raw_data[i * 2 * space :(i * 2 + 1) * space, :]
        Vg = data[:, 0]
        Id = numpy.abs(data[:, 1]) / 1e-6
        # J = Id / (scipy.pi * r[i] ** 2) / 10
        # J = stretch_y(J, minmax[i])
        Id = fil(Id, mm[i])
        ax.plot(Vg, Id, '-o',
                label="Cycle-{}".format(i + 1),markersize=3,color=col[int(p[i])])
        ratio.append([i, max(Id) / min(Id)])

    ax.legend(frameon=True)
    fig.savefig(img_path / "pressure-Id-Vg.svg")


    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Ratio")
    cm = ax.imshow([[0, 0], [0, 0]],
                   vmin=0, vmax=150,
                   cmap="jet", )
    ratio = numpy.array(ratio)
    fig.colorbar(cm, ticks=numpy.linspace(0, 150, 7))
    ax.plot(ratio[:, 0], ratio[:, 1], "s-")
    fig.savefig(img_path / "pressure-ratio.svg")

def plot_iv_cycles():
    import numpy
    import matplotlib.pyplot as plt

    IV_name = data_path / "pressure" / "pressure_IV4.csv"
    IV_data = numpy.genfromtxt(IV_name,
                               delimiter=",",
                               skip_header=1)

    delay = 7.45
    # delay = 0

    t = IV_data[:, 0] - delay
    I = IV_data[:, 1]
    I0 = I[0]

    params = numpy.array([
        [322.667 / 2, 377.8, 950.3],
        [664.0 / 2, 280.14, 1029.3]]) * 2e-6

    gamma = 0.485

    p = [(1/item[1] + 1/item[2]) * gamma for item in params]

    print(p[1] - p[0])


    fig = plt.figure(figsize=(5, 2))
    ax = fig.add_subplot(111)
    ax.set_ylim(0.1, 15)
    ax.set_xlim(0, 130)
    ax.set_xlabel("$t$ (s)")
    ax.set_ylabel("$I / I_0$")

    ax.plot(t, I / I0, markersize=3)

    fig.savefig(img_path / "plain_pressure_iv.svg")








def plot_main():
    plot_model()
    plot_exp()
    plot_iv_cycles()


if __name__ == '__main__':
    print(("The script outputs pressure plots into {} as svg!"
           " Do manual editing after").format(img_path.as_posix()))
    
    plot_main()
