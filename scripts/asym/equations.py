from .constants import Const
from numpy import sqrt, sign, exp, abs, pi, log
import numpy as np
from scipy.optimize import fsolve


def func_np(Psi, Psi_B):
    # Returns the concentration of n and p at different Psi value (C/m^3)
    # when Psi = 0 they are the concentration of D and A
    n_0 = Const.n_i * exp(Const.q * Psi_B / Const.k / Const.T)
    p_0 = Const.n_i * exp(-Const.q * Psi_B / Const.k / Const.T)
    n = n_0 * exp(Const.q * Psi / Const.k / Const.T)
    p = p_0 * exp(-Const.q * Psi / Const.k / Const.T)
    return n, p


def func_Psi_B(ND):
    #  Calculate psi_b from the given ND value
    p = log(ND / Const.n_i) * Const.k * Const.T / Const.q
    return p


def func_delta_phi_g(Q_g):
    """Calculate the difference in the work function of graphene"""

    p_g = sign(Q_g) * Const.h_bar * Const.v_F / Const.q \
        * sqrt(pi * abs(Q_g) / Const.q)
    return p_g


def func_E_psi(Psi, Psi_B):
    """
    Calculate the E field given interface Psi and Schottky barrier Psi_B
    """
    a = Const.q * Const.n_i / (Const.epsilon_0 * Const.epsilon_r)
    gamma = Const.q / (Const.k * Const.T)
    A = a * exp(gamma * Psi_B)
    B = a * exp(-gamma * Psi_B)
    E = sign(Psi) * sqrt(2 * (A / gamma * exp(gamma * Psi) +
                              B / gamma * exp(-gamma * Psi) +
                              (B - A) * Psi - (A / gamma + B / gamma)))
    return E


def func_q_g(Q_gate, E):
    """
       Returns the charge in graphene in C/cm^2, 
       given Q_gate and electric field E
    """
    qg = -Q_gate - func_q_semi(E)
    return qg


def func_q_semi(E):
    """ Charge density at the semiconductor surface"""
    qs = -Const.epsilon_0 * Const.epsilon_r * E
    return qs


def solve_psi_s(Psi, Psi_B, Q_gate, V_D):
    """Equation for potential balance
       SOLVE_PSI Summary of this function goes here
       Solved the Psi_s value at surface, with applied voltage V_D (elevating E_F)
       Psi: potential at interface
       Psi_B: Schottky barrier
       Q_gate: gate charge
       V_D: bias
    """
    q_g = func_q_g(Q_gate, func_E_psi(Psi, Psi_B))
    F = Psi + func_delta_phi_g(q_g) + Const.phi_g0 - \
        (Const.phi_i-Psi_B) - V_D  # Solve F=0
    return F


def solve_psi_s_no_graphene(Psi, Psi_B, Q_gate):
    """   Solved the Psi_s value at surface, in the case that no graphene is
    present
    """
    F = (-Q_gate-func_q_semi(func_E_psi(Psi, Psi_B))) / (Q_gate + 1e-20)
    return F


def solve_Psi_B(psi_b, Q_gate, i):
    """   returns the Q_gate value when Psi =0; -Psi_B and -2Psi_B (no bias)
    """
    psi = i * psi_b
    q_g = func_q_g(Q_gate, func_E_psi(psi, psi_b))
    F = psi + Const.phi_g0 + func_delta_phi_g(q_g) \
        - (Const.phi_i - psi_b)
    return F


def solve_NL(N, psi_S, psi_b):
    """Give the sigma_S values when sigma_M and 
    guess value sigma_G1 is provided"""
    E_S = func_E_psi(psi_S, psi_b)
    sigma_S = -Const.epsilon_0 * Const.epsilon_r * E_S
    sigma_GS = np.zeros(N)
    phi_GS = np.zeros(N)
    E_GS = np.zeros(N)
    # Solve from the last layer
    phi_GS[-1] = Const.phi_i - psi_b - psi_S
    sigma_GS[-1] = func_sigma_g_phi(phi_GS[-1])
    E_GS[-1] = -(sigma_GS[-1] + sigma_S) / Const.epsilon_0 / Const.epsilon_g

    for i in range(N - 2, -1, -1):
        dV = E_GS[i + 1] * Const.d0
        phi_GS[i] = phi_GS[i + 1] - dV
        sigma_GS[i] = func_sigma_g_phi(phi_GS[i])
        E_GS[i] = (E_GS[i + 1] * Const.epsilon_0 * Const.epsilon_g
                   - sigma_GS[i + 1]) / Const.epsilon_0 / Const.epsilon_g

    return sigma_GS, phi_GS, sigma_S


def func_fsolve_NL(N, sigma_M, psi_S, psi_b):
    """Gives the solution for NL system"""
    sigma_GS, phi_GS, sigma_S = solve_NL(N, psi_S, psi_b)
    out = sigma_M + sigma_S + sum(sigma_GS)
    return out


def fsolve_NL(N, sigma_M, guess_psi_S, psi_b):
    """Gives the solution for NL system"""
    p_S = fsolve(lambda symb_psi_S:
                 func_fsolve_NL(N, sigma_M, symb_psi_S, psi_b),
                 guess_psi_S)[0]
    psi_S = p_S
    [sigma_GS, phi_GS, sigma_S] = solve_NL(N, p_S, psi_b)
    return sigma_GS, phi_GS, sigma_S, psi_S



def func_phi_g_sigma(sigma_g):
    """PHI_G Summary of this function goes here
    calculate the difference in the work function of graphene
    """
    phi_g = Const.phi_g0 + sign(sigma_g) * Const.h_bar * Const.v_F \
        / Const.q * sqrt(pi * abs(sigma_g) / Const.q)
    return phi_g


def func_sigma_g_phi(phi_g):
    sigma_g = sign(phi_g - Const.phi_g0) * Const.q ** 3 \
        * (phi_g - Const.phi_g0) ** 2 \
        / (pi * Const.v_F ** 2 * Const.h_bar ** 2)
    return sigma_g
