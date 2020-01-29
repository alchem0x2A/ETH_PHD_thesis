from .constants import Const
from numpy import sqrt, sign, exp, abs, pi


def func_np(Psi, Psi_B):
    # Returns the concentration of n and p at different Psi value (C/m^3)
    # when Psi = 0 they are the concentration of D and A
    n_0 = Const.n_i * exp(Const.q * Psi_B / Const.k / Const.T)
    p_0 = Const.n_i * exp(-Const.q * Psi_B / Const.k / Const.T)
    n = n_0 * exp(Const.q * Psi / Const.k / Const.T)
    p = p_0 * exp(-Const.q * Psi / Const.k / Const.T)
    return n, p


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
