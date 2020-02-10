import numpy
import scipy
from numpy import exp
import scipy.constants as const
from scipy.optimize import curve_fit, fsolve
from scipy.integrate import trapz
from warnings import warn

e = const.e
hbar = const.hbar
pi = const.pi
eps0 = const.epsilon_0

k_ecm_SI = e * (10 ** -2) ** -2                     # 1 e/cm^2 to SI
k_SI_ecm = k_ecm_SI ** -1                     # SI to 1 e/cm^2
k_muF_SI = 10 ** -2                           # muF to SI
k_SI_mu = 10 ** 2                             # SI to muF

# def exp(n):
    # if n > 30:
        # return numpy.inf
    # else:
        # return scipy.exp(n)

# exp = numpy.vectorize(exp)
exp = scipy.exp

def fermi_dirac(E, mu, T=298):
    return 1 / (exp((E - mu) * e / (const.k * T)) + 1)
# charge density of graphene
# as function of phi

rat = 0.5

def charge_graphene(phi, ratio=rat):
    phi0 = 4.6  # eV
    vf = 1.1 * 10 ** 6
    fac = (hbar * vf / e) ** 2 * pi / e
    sigma = ratio * (phi - phi0) ** 2 / fac * scipy.sign(phi - phi0)
    return sigma


def gen_DOS(E_C, E_V, CQ_n, CQ_p, n=1000, lim=1.5):
    E_arr = numpy.linspace(E_V - lim, E_C + lim, n)

    def __DOS(E):
        if E <= E_V:
            return CQ_p
        elif E >= E_C:
            return CQ_n
        else:
            return 0
    _DOS = numpy.vectorize(__DOS)
    DOS_arr = _DOS(E_arr)
    return E_arr, DOS_arr

NC_MoS2 = 35 * k_muF_SI
NV_MoS2 = 50 * k_muF_SI
EC = -4.0
EV = -5.8


def get_tot_charge(chg_arr):
    chg_p = chg_n = 0
    for chg in chg_arr:
        if chg > 0:
            chg_p = chg_p + chg
        else:
            chg_n = chg_n + chg
    eps = 1e-4
    if abs(chg_p / chg_n + 1) > eps:
        warn("Charge not consistent!")
    # The first layer is always Gr
    return scipy.sign(chg_arr[0]) * chg_p
# DOS_arr = [NV_MoS2] * 200 + [0] * 200 + [NC_MoS2] * 200
# DOS_arr = numpy.array(DOS_arr)
# E_arr = numpy.linspace(EV - 0.5, EC + 0.5, 600)

E_arr, DOS_arr = gen_DOS(EC, EV, NC_MoS2, NV_MoS2)
# print(len(DOS_arr), len(E_arr))
# charge density of MoS2


def _charge_MoS2(phi):
    NC = 150 * k_muF_SI                   # constant assumption
    NV = 180 * k_muF_SI                    # constant assumption
    phi_C = -EC
    phi_V = -EV
    phi0 = 4.49
    EF0 = -phi0
    EF1 = -phi
    dq_arr = DOS_arr * (fermi_dirac(E_arr, EF1)
                        - fermi_dirac(E_arr, EF0))
    sigma = -trapz(dq_arr, E_arr)
    return sigma

charge_MoS2 = numpy.vectorize(_charge_MoS2)


def phi_graphene(sigma, ratio=rat):
    phi0 = 4.6
    vf = 1.1 * 10 ** 6
    fac = hbar * vf / e
    return phi0 + scipy.sign(sigma / ratio) * fac * \
        scipy.sqrt(scipy.pi * scipy.abs(sigma / ratio) / e)

# Not really used method
def _phi_MoS2(sigma):
    f_target = lambda E: sigma - charge_MoS2(E)
    p0 = 4.49
    phi = fsolve(f_target, p0, xtol=1e-5)
    return phi

phi_MoS2 = numpy.vectorize(_phi_MoS2)

def select_charge_func(name):
    if name[0].lower() == "g":
        return charge_graphene
    elif name[0].lower() == "m":
        return charge_MoS2
    else:
        raise ValueError("Wrong type of material!")

def stack_charge(phi0,
                 materials,
                 distances,
                 epsilons,
                 E_out,
                 use_solve=True,
                 ratio=1):
    epsilons = eps0 * numpy.array(epsilons)
    tot_maters = len(materials)
    if (tot_maters != len(distances) + 1) \
       or (tot_maters != len(epsilons) - 1):
        raise IndexError("The length of the matrix is wrong!")
    sigmas = numpy.empty(tot_maters)
    phis = numpy.empty(tot_maters)
    Es = numpy.empty(len(epsilons))
    Es[0] = Es[-1] = E_out
    # Calculate the sigma of each layer using SC approach
    sigmas[0] = select_charge_func(materials[0])(phi0)
    phis[0] = phi0
    Es[1] = (Es[0] * epsilons[0] + sigmas[0]) / epsilons[1]
    # All layers
    # print(sigmas[0], phis[0])
    for i in range(1, tot_maters):
        # Delta = Es[i] * epsilons[i]
        Delta = Es[i] * distances[i - 1]
        # print("Delta: ", Delta)
        phis[i] = phis[i - 1] + Delta
        sigmas[i] = select_charge_func(materials[i])(phis[i])
        # print(phis[i], sigmas[i])
        Es[i + 1] = (Es[i] * epsilons[i] + sigmas[i]) / epsilons[i + 1]
    # Whether to use as target for fsolve?
    if use_solve:
        if E_out == 0:
            return E_out
        else:
            return Es[-1] / E_out - 1
    else:
        return sigmas, phis, Es


def solve_stack(materials,
                distances,
                epsilons,
                E_out):
    phi0_guess = 4.6
    phi0 = fsolve(stack_charge, phi0_guess,
                  args=(materials, distances, epsilons, E_out, True))
    return phi0

def gen_input(n_gr, n_mos2, eps, d):
    materials = ["G"] * n_gr + ["M"] * n_mos2
    # change the distance to the interlayer distance
    distances = [d[0]] * (n_gr - 1) + [d[1]] * 1 + [d[2]] * (n_mos2 - 1)
    epsilons = [1] + [eps] * (n_gr + n_mos2 - 1) + [1]
    return materials, distances, epsilons


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # phi = numpy.linspace(3.85, 6.0, 100)
    # sigma = charge_MoS2(phi) + k_ecm_SI * 10 ** 8
    # phi2 = phi_MoS2(sigma)
    # sigma = fermi_dirac(-4.49, -4.49) * DOS_arr
    # plt.plot(phi, abs(k_SI_ecm * sigma))
    # plt.plot(phi2, abs(k_SI_ecm * sigma))
    # plt.yscale("log")
    # plt.plot(E_arr, DOS_arr)
    # plt.savefig("test_charge_MoS2.png")
    for j in [1, 2, 3]:
        materials = ["G"] * 3  + ["M"] * j
        distances = [2.78e-10] * (j + 2) 
        epsilons = [1] + [2.4] * (j + 2) + [1]

        E_out = numpy.linspace(-2e9, 2e9, 30)
        tot_sigma = []
        for E in E_out:
            phi_0 = solve_stack(materials,
                                distances,
                                epsilons,
                                E)
            sigmas, phis, Es = stack_charge(phi_0,
                                            materials,
                                            distances,
                                            epsilons,
                                            E,
                                            use_solve=False)
            tot_sigma.append(abs(sum(sigmas[0 : 3])))
        plt.plot(E_out, numpy.array(tot_sigma) * k_SI_ecm)
    plt.yscale("log")
    plt.savefig("test_G_MoS.png")
    
