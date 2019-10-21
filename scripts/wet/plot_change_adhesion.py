import numpy
import scipy
import scipy.constants as const
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pycse.orgmode as org
from copy import copy

charge_per_atom = [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.012]

c_atom_to_sigma = lambda x: x*2/(2.465e-8**2*scipy.sin(scipy.pi/3))

def read_xvg_energy(filename):
    data = {}
    with open(filename) as f:
        s_tmp = ""
        s = f.readline()
        while s.startswith("-") is not True:
            s_tmp = s
            s = f.readline()
        attrs = s_tmp.strip().split()  # Attributes of columns
        s = f.readline()
        while len(s) > 0:
            # print(s)
            name = ""
            i = 0
            s = s.split()
            while not s[i][0].isdecimal() and not s[i][0] == "-":
                name += s[i]
                i += 1
            d_dic = {}
            for att in attrs[1:]:
                d_dic[att] = float(s[i])
                i += 1
            d_dic["Unit"] = s[-1]
            data[name] = d_dic
            s = f.readline()
    return data

# Convert the adhesion energy from

A_c = 15.1e-18                  # area of the whole plane in m^2

f_base = "../data/6_11_17_data/E_int_{}{:.3f}_large2.xvg"
cases = ["", "neg"]


vdW_tot = []
vdW_err = []
coulomb_tot = []
coulomb_err = []
potential_tot = []
potential_err = []
coul_LR = []
charges_sorted = []

f_0 = f_base.format("", 0)
data = read_xvg_energy(f_0)
vdw0 = data["LJ(SR)"]["Average"] + data["Disper.corr."]["Average"]
coul0 = data["Coulomb(SR)"]["Average"] + data["Coul.recip."]["Average"]
potential0 = data["Potential"]["Average"]
coul_LR_0 = data["Coul.recip."]["Average"]

#negative charges
neg_charge = copy(charge_per_atom)
neg_charge.reverse()

for e in neg_charge[:-1]:
    f_n = f_base.format("neg", e)
    charges_sorted.append(-e)
    # print(f_n)
    data = read_xvg_energy(f_n)
    vdw = data["LJ(SR)"]["Average"] + data["Disper.corr."]["Average"]
    vdw_err = data["LJ(SR)"]["RMSD"] + data["Disper.corr."]["RMSD"]
    # coul = data["Coulomb(SR)"]["Average"]
    coul = data["Coulomb(SR)"]["Average"] + data["Coul.recip."]["Average"]
    coul_err = data["Coulomb(SR)"]["RMSD"] + data["Coul.recip."]["RMSD"]
    _coul_LR = data["Coul.recip."]["Average"]
    potential = data["Potential"]["Average"]
    potential_err_ = data["Potential"]["RMSD"]
    # print(vdw, coul)
    vdW_tot.append(vdw-vdw0)
    coulomb_tot.append(coul-coul0)
    vdW_err.append(vdw_err)
    coulomb_err.append(coul_err)
    # potential_tot.append(potential-potential0-_coul_LR)
    potential_tot.append(potential-potential0)
    potential_err.append(potential_err_)

for e in charge_per_atom:
    f_n = f_base.format("", e)
    charges_sorted.append(e)
    # print(f_n)
    data = read_xvg_energy(f_n)
    vdw = data["LJ(SR)"]["Average"] + data["Disper.corr."]["Average"]
    vdw_err = data["LJ(SR)"]["RMSD"] + data["Disper.corr."]["RMSD"]
    # coul = data["Coulomb(SR)"]["Average"]
    coul = data["Coulomb(SR)"]["Average"] + data["Coul.recip."]["Average"]
    coul_err = data["Coulomb(SR)"]["RMSD"] + data["Coul.recip."]["RMSD"]
    _coul_LR = data["Coul.recip."]["Average"]
    potential = data["Potential"]["Average"]
    potential_err_ = data["Potential"]["RMSD"]
    # print(vdw, coul)
    vdW_tot.append(vdw-vdw0)
    coulomb_tot.append(coul-coul0)
    vdW_err.append(vdw_err)
    coulomb_err.append(coul_err)
    # potential_tot.append(potential-potential0-_coul_LR)
    potential_tot.append(potential-potential0)
    potential_err.append(potential_err_)
    # coul_LR.append(_coul_LR)

charges_sorted = numpy.array(charges_sorted)
# sigma = c_atom_to_sigma(charge_per_atom)
n_2D = c_atom_to_sigma(charges_sorted)/10**13
err_scale = 30

vdW_tot = numpy.array(vdW_tot)/A_c/const.N_A*10**6
vdW_err = numpy.array(vdW_err)/A_c/const.N_A*10**6 / err_scale
coulomb_tot = numpy.array(coulomb_tot)/A_c/const.N_A*10**6
coulomb_err = numpy.array(coulomb_err)/A_c/const.N_A*10**6 / err_scale
potential_tot = numpy.array(potential_tot)/A_c/const.N_A*10**6
potential_err = numpy.array(potential_err)/A_c/const.N_A*10**6 / err_scale

v_coul_shift = numpy.array([0, 0, 0, -0.60, -1.23,
                            4.80, 1.5, 3.95, 3.46,
                            0,
                            0, 0, 0, 0,
                            0, 0, 0, 0, 0])

coulomb_tot += v_coul_shift
potential_tot += v_coul_shift
# nn = numpy.linspace(-5, 5, 100)
# params = numpy.polyfit(n_2D, vdW_tot, 2)
# f = numpy.poly1d(params)
# vv = f(nn)

with open("new_MD_data.txt", "w") as f:
    f.write("e_per_atom,n_2D,Delta_Phi\n")
    for index in range(len(charges_sorted)):
        f.write("{},{},{}\n".format(charges_sorted[index],
                                    n_2D[index],
                                    potential_tot[index]))


def plot_Phi_charge(fig, error=False):
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()           # For the charge
    # ax3 = ax1.twinx()           # For the surface tension
    l_tot,  = ax1.plot(n_2D, potential_tot, 's', markersize=5,
                     label=r"$\Delta \Phi_{\mathrm{Coul}} + \Delta \Phi_{\mathrm{LJ}}$")
    l_vdw,  = ax1.plot(n_2D, vdW_tot, 'o', markersize=5,
             label=r"$\Delta \Phi_{\mathrm{LJ}}$")
    l_cl,  = ax1.plot(n_2D, coulomb_tot, 'v', markersize=5,
                    label=r"$\Delta \Phi_{\mathrm{Coul}}$")
    if error is True:
        ax1.fill_between(n_2D,
                     vdW_tot-vdW_err, vdW_tot+vdW_err,
                         alpha=0.2, facecolor=l_vdw.get_c())
        ax1.fill_between(n_2D,
                     coulomb_tot-coulomb_err, coulomb_tot+coulomb_err,
                         alpha=0.2, facecolor=l_cl.get_c())
        ax1.fill_between(n_2D,
                     potential_tot-potential_err, potential_tot+potential_err,
                         alpha=0.2, facecolor=l_tot.get_c())
    # ax1.plot(nn, vv, color=l_vdw[0].get_color(), alpha=0.6)
    ax1.set_xlabel(r"$\sigma_{\mathrm{2D}}$ ($10^{13}$ $e\cdot$cm$^{-2}$)")
    ax1.set_ylabel(r"$\Delta \Phi$ (mJ$\cdot$m$^{-2}$)")
    ax1.legend(loc=0, frameon=True)
    ax1.set_ylim(-25, 8)
    # ax1.set_xlim(-4, 4)
    # ax1.set_ylim(-10, 15)
    # Change the second x axis

    ax2_ticks = numpy.linspace(-0.012, 0.012, 7)
    ax2.set_xticks(c_atom_to_sigma(ax2_ticks)/10**13)
    ax2.set_xticklabels(list(map(lambda s: "%.0f" % s, ax2_ticks*1000)))
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xlabel("$\sigma_{\mathrm{2D}}$ (10$^{-3}$ $e$/atom)")
    fig.tight_layout()

def plot_fitting(fig):
    ax = fig.add_subplot(111)
    ax.plot(n_2D, potential_tot, "s", label="MD Data")
    power_matrix = numpy.vstack((n_2D**4, n_2D**3, n_2D**2, n_2D, numpy.ones_like(n_2D))).T
    degs = [2, 3, 4]
    n_2D_plot = numpy.linspace(-5, 5, 100)
    for deg in degs:
        # param_fit = scipy.polyfit(n_2D, potential_tot, deg)
        param_fit, _, _, _ = numpy.linalg.lstsq(power_matrix[:, 4-deg:-1], potential_tot)
        print(deg, param_fit)
        poly_f = scipy.poly1d(numpy.hstack((param_fit, 0)))
        fit_data = poly_f(n_2D_plot)
        label_axis = "$" + "+".join(["{0:.3f}x^{1}".format(param_fit[i], deg-i) for i in range(deg)]) + "$"
        ax.plot(n_2D_plot, fit_data, label=label_axis)
    ax.set_xlim(-4, 4)
    ax.set_xlabel(r"$\sigma_{\mathrm{2D}}$ ($10^{13}$ $e\cdot$cm$^{-2}$)")
    ax.set_ylabel(r"$\Delta \Phi$ (mJ$\cdot$m$^{-2}$)")
    ax.legend(loc=0)
    fig.tight_layout()

# ax1.set_xlim(-20, 20)



# ax2_ticks = numpy.linspace(-0.03, 0.03, 7)
# ax2.set_xlim(ax1.get_xlim())
# ax2.set_xticks(c_atom_to_sigma(ax2_ticks)/10**13)
# ax2.set_xticklabels(list(map(str, ax2_ticks)))
# # ax2.plot(charge_per_atom, potential_tot, alpha=0)
# ax2.set_xlabel("Unit charge per atom", labelpad=10)


# # print(ax1.get_ylim())
# # print(ax1.get_yticks())
# ax3.set_yticks(ax1.get_yticks())
# ax3.set_ylim(ax1.get_ylim())
# ax3_yticks = ax1.get_yticks()/A_c/const.N_A*10**6
# ax3.set_yticklabels(list(map(lambda a: "%.1f"%a, ax3_yticks)))
# # ax3.plot(sigma/10**13, potential_tot/A_c/const.N_A*1000, alpha=0.0)
# ax3.set_ylabel(r"$\Delta\gamma_{\mathrm{WG}}$ [mJ$\cdot$m$^{-2}$]", labelpad=-2)


# org.figure(plt.savefig("../img/e-vdw.png"))


if __name__ == "__main__":
    matplotlib.style.use("science")

    fig = plt.figure(figsize=(3, 3))
    plot_Phi_charge(fig, error=True)
    org.figure(plt.savefig("../img/e-vdw-2.pdf"))

    fig = plt.figure(figsize=(3, 3))
    plot_fitting(fig)
    org.figure(plt.savefig("../img/e-Phi-fitting.pdf"))
