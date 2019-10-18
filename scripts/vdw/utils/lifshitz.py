import numpy
from numpy import sqrt, sin, cos, log, trapz, exp
import matplotlib.pyplot as plt
from scipy.constants import hbar, e, electron_volt, pi, k, m_e, epsilon_0
from .kkr import matsubara_freq, kkr

T = 300
freq_matsu = matsubara_freq(numpy.arange(0, 1000),
                             mode="energy")

def alpha_to_eps(alpha, d,
                 direction="x",
                 imag=True):
    assert(d > 0)
    d = d / 1e-10                # in angstrom
    if ("x" in direction) or ("y" in direction):
        eps = 1 + 4 * pi * alpha / d
    elif "z" in direction:
        eps = 1 / (1 - 4 * pi * alpha / d)
    else:
        raise ValueError("Must define the dielectric direction!")



    if imag:
        return eps.imag
    else:
        return eps              # Full complex number

def doping_correction(d, freq, doping=0):
    # Doping in C/m^2
    if doping != 0:
        eta = 1e-4
        ratio = 10 ** 13 * e / 0.01 ** 2  # 10^13 e/cm^2 to C/m^2
        omega_p2 = numpy.abs(doping * ratio) * e / (epsilon_0 * m_e)  # omega_p^2
        eps_intra = omega_p2 * hbar ** 2 / electron_volt ** 2 / (freq) ** 2 / d
        eps_intra[numpy.isnan(eps_intra)] = 0
        eps_intra[numpy.isinf(eps_intra)] = 1e4
        # print(eps_intra)
        # print(eps_intra)
        return eps_intra
    else:
        return numpy.zeros_like(freq)


# g_amb using alpha, hardcoded
def g_amb_alpha(eps_a, alpha_m, eps_b,
                freq, freq_alpha, d, renormalized=True, doping=0):
    # freq: matsubara_freq for bulk materials
    # freq_alpha: grid frequencies for 2D material
    eps_x = alpha_to_eps(alpha_m[0], d, direction="x", imag=True)
    eps_y = alpha_to_eps(alpha_m[1], d, direction="y", imag=True)
    eps_z = alpha_to_eps(alpha_m[2], d, direction="z", imag=True)
    correction = doping_correction(d, freq,  doping=doping)
    eps_x_iv = kkr(freq_alpha, eps_x, freq) + correction
    eps_y_iv = kkr(freq_alpha, eps_y, freq) + correction
    eps_z_iv = kkr(freq_alpha, eps_z, freq)
    eps_m = numpy.array([eps_x_iv, eps_y_iv, eps_z_iv])
    return g_amb(eps_a, eps_m, eps_b, freq, d, renormalized=renormalized)

# g_amb using alpha, hardcoded
def g_amb_alpha_part(eps_a, alpha_m, eps_b,
                     freq, freq_alpha, d, renormalized=True, doping=0):
    # freq: matsubara_freq for bulk materials
    # freq_alpha: grid frequencies for 2D material
    eps_x = alpha_to_eps(alpha_m[0], d, direction="x", imag=True)
    eps_y = alpha_to_eps(alpha_m[1], d, direction="y", imag=True)
    eps_z = alpha_to_eps(alpha_m[2], d, direction="z", imag=True)
    correction = doping_correction(d, freq, doping=doping)
    eps_x_iv = kkr(freq_alpha, eps_x, freq) + correction
    eps_y_iv = kkr(freq_alpha, eps_y, freq) + correction
    eps_z_iv = kkr(freq_alpha, eps_z, freq)
    eps_m = numpy.array([eps_x_iv, eps_y_iv, eps_z_iv])
    return g_amb_part(eps_a, eps_m, eps_b, freq, d, renormalized=renormalized)
    
# Non-retarded calculations
def g_amb(eps_a, eps_m, eps_b, freq, d, renormalized=True):
    # Make sure they are on the same grid
    assert(eps_a.shape == (3, len(freq)))
    assert(eps_b.shape == (3, len(freq)))
    assert(eps_m.shape == (3, len(freq)))
    eps_a_geo = numpy.sqrt(eps_a[0] * eps_a[2])
    eps_b_geo = numpy.sqrt(eps_b[0] * eps_b[2])
    eps_m_geo = numpy.sqrt(eps_m[0] * eps_m[2])
    if renormalized:
        # print("gm^2 considered")
        gm2 = eps_m[0] / eps_m[2]   # g_m^2=eps_x/epx_z
    else:
        # print("gm^2 NOT considered")
        gm2 = numpy.ones_like(eps_m[0])
        # gm2 = numpy.ones_like(eps_m_geo)
    delta_am = (eps_a_geo - eps_m_geo) / (eps_a_geo + eps_m_geo)
    delta_bm = (eps_b_geo - eps_m_geo) / (eps_b_geo + eps_m_geo)
    # Firstly, no gm
    def _integral_part(a, b, cutoff=15):
        x = numpy.linspace(0, cutoff, 1000)
        D = 1 - a * b * exp(-x)
        part = x * log(D)
        integ_sum = trapz(y=part, x=x)
        return integ_sum
    int_part2 = numpy.array([_integral_part(delta_am[i],
                                            delta_bm[i])/ gm2[i] \
                             for i in range(len(freq))])
    multiplier = numpy.ones_like(int_part2)
    multiplier[0] = 0.5
    prefactor = k * T / (8 * pi * d ** 2)
    return numpy.sum(int_part2 * multiplier) * prefactor

'''
g_amb frequency parts
'''
def g_amb_part(eps_a, eps_m, eps_b, freq, d, renormalized=True):
    # Make sure they are on the same grid
    assert(eps_a.shape == (3, len(freq)))
    assert(eps_b.shape == (3, len(freq)))
    assert(eps_m.shape == (3, len(freq)))
    eps_a_geo = numpy.sqrt(eps_a[0] * eps_a[2])
    eps_b_geo = numpy.sqrt(eps_b[0] * eps_b[2])
    eps_m_geo = numpy.sqrt(eps_m[0] * eps_m[2])
    if renormalized:
        gm2 = eps_m[0] / eps_m[2]   # g_m^2=eps_x/epx_z
    else:
        gm2 = numpy.ones_like(eps_m[0])
        # gm2 = numpy.ones_like(eps_m_geo)
    delta_am = (eps_a_geo - eps_m_geo) / (eps_a_geo + eps_m_geo)
    delta_bm = (eps_b_geo - eps_m_geo) / (eps_b_geo + eps_m_geo)
    # Firstly, no gm
    def _integral_part(a, b, cutoff=15):
        x = numpy.linspace(0, cutoff, 1000)
        D = 1 - a * b * exp(-x)
        part = x * log(D)
        integ_sum = trapz(y=part, x=x)
        return integ_sum
    int_part2 = numpy.array([_integral_part(delta_am[i],
                                            delta_bm[i])/ gm2[i] \
                             for i in range(len(freq))])
    multiplier = numpy.ones_like(int_part2)
    multiplier[0] = 0.5
    prefactor = k * T / (8 * pi * d ** 2)
    return int_part2 * multiplier * prefactor

def g_single_coating(eps_a, eps_m, eps_c, eps_b, freq, d, delta):
    # Calculation on single coating surface
    assert(eps_a.shape == (3, len(freq)))
    assert(eps_b.shape == (3, len(freq)))
    assert(eps_m.shape == (3, len(freq)))
    eps_a_geo = numpy.sqrt(eps_a[0] * eps_a[2])
    eps_b_geo = numpy.sqrt(eps_b[0] * eps_b[2])
    eps_c_geo = numpy.sqrt(eps_c[0] * eps_c[2])
    eps_m_geo = numpy.sqrt(eps_m[0] * eps_m[2])
    gm2 = eps_m[0] / eps_m[2]   # g_m^2=eps_x/epx_z
    delta_am = (eps_a_geo - eps_m_geo) / (eps_a_geo + eps_m_geo)
    delta_cm = (eps_c_geo - eps_m_geo) / (eps_c_geo + eps_m_geo)
    delta_bc = (eps_b_geo - eps_c_geo) / (eps_b_geo + eps_c_geo)
    def _integral_part(a, c, b, cutoff=15):
        x = numpy.linspace(0, cutoff, 1000)
        k = delta / d           # renorm factor for coating layer
        b_eff = (b * exp(-x * k) + c) / (1 + b * c * exp(-x * k))
        D = 1 - a * b_eff * exp(-x)
        part = x * log(D)
        integ_sum = trapz(y=part, x=x)
        return integ_sum, part
    # Summ by frequencies

    fk = []
    int_part2 = []
    for i in range(len(freq)):
        int_sum, part = _integral_part(delta_am[i], delta_cm[i], delta_bc[i]) / gm2[i]
        int_part2.append(int_sum)
        fk.append(part)
    # Int part
    int_part2 = numpy.array(int_part2)
    # dq
    fk = numpy.array(fk)        # [freq, q]
    # int_part2 = numpy.array([_integral_part(delta_am[i],
                                            # delta_bm[i])/ gm2[i] \
                             # for i in range(len(freq))]    
    multiplier = numpy.ones_like(int_part2)
    multiplier[0] = 0.5
    prefactor = k * T / (8 * pi * d ** 2)
    G = int_part2 * multiplier * prefactor
    return G, int_part2, fk
    


'''
Transparency single
'''
def transparency_single(eps_a, alpha_m, eps_b,
                        freq, freq_alpha, d, doping=0):
    eps_v = numpy.ones_like(eps_a)
    E0 = g_amb(eps_a, eps_v, eps_b, freq, d)  # Vacuum energy
    E1 = g_amb_alpha(eps_a, alpha_m, eps_b,
                     freq, freq_alpha, d,
                     renormalized=True, doping=doping)  # 2D screened
    return E1 / E0

def transparency_single_eps(eps_a, eps_m, eps_b,
                            freq,  d):
    eps_v = numpy.ones_like(eps_a)
    E0 = g_amb(eps_a, eps_v, eps_b, freq, d)  # Vacuum energy
    E1 = g_amb(eps_a, eps_m, eps_b, freq, d)  # Vacuum energy
    return E1 / E0

if __name__ == "__main__":
    raise NotImplementedError("Should not run this module as main!")
    # data = numpy.genfromtxt("../../spectrum_files/converted/water-lds-eps2.txt",
    #                         comments="#")
    # freq = data[:, 0]
    # eps2 = data[:, 1]
    # freq_matsu = matsubara_freq(numpy.arange(0, 1001)) * hbar / electron_volt
    # eps_iv = kkr(freq, eps2, freq_matsu)
    # eps_a = numpy.array([eps_iv] * 3)
    # eps_b = numpy.array([eps_iv] * 3)
    # eps_m = numpy.array([numpy.ones_like(eps_iv)] * 3)
    # d = 5e-10
    # E = cal_energy(eps_a, eps_m, eps_b, freq_matsu, d)
    # A = -E * (12 * pi * d ** 2) / 1e-21
    # print(A)



