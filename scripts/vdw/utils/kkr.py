import numpy
from scipy.constants import hbar, electron_volt, pi
from scipy.constants import e, m_e, epsilon_0, k

def kkr(x, y, xx, source="eps_2_rf", target="eps_if", cshift_=1e-6): 
    '''                                                                                         
    Kramers-Kronig Relations for                                                                
    1) \\epsilon_2(\\xi) --> \\epsilon(i \\nu)                                                          
    2) \\epsilon_2(\\xi) --> \\epsilon_1(\\xi)                                                         
    ... etc                                                                                     
    The default option is 2)                                                                    
                                                                                                
    The Kramers-Kronig Relations                                                                
    A small complex shift is used for smoothing                                                 
    see https://github.com/utf/kramers-kronig                                                   
                                                                                                
    Args:                                                                                       
    	x (numpy.ndarray) : frequency / energy field                                              
        y (numpy.ndarray) : dielectric function source                                          
        xx (numpy.ndarray) : frequency / energy field for the target dielectric function        
    	source : "eps_2_rf" -- Imaginary dielectric function, real frequency                      
    	target : "eps_1_rf" -- Real dielectric function, real frequency                           
    		 "eps_if" -- Dielectric function, imaginary frequency                             
    Returns:                                                                                    
    	target dielectric with frequency field xx                                                 
    '''                                                                                         
    
    xx = numpy.array(xx)
    cshift_ = complex(0, cshift_)
    def _KKR_eps2_to_eps1(omega_i_, eps2_, omega_r_):  # omega_r to be single value
        factor = omega_i_ * eps2_ / (omega_i_ ** 2 - omega_r_ ** 2 + cshift_)
        int_res = numpy.real(numpy.trapz(factor, omega_i_))
        return 1.0 + 2.0 / pi * int_res

    def _KKR_eps2_to_eps_if(omega_i_, eps2_, omega_iv_):  # omega_iv_ to be single value
        factor = omega_i_ * eps2_ / (omega_i_ ** 2 + omega_iv_ ** 2 + cshift_)
        int_res = numpy.real(numpy.trapz(factor, omega_i_))
        return 1.0 + 2.0 / pi * int_res

    def _KKR_eps1_to_eps2(omega_1_, eps1_, omega_2_):  # omega_r to be single value
        factor = (eps1_  - 1) / (omega_1_ ** 2 - omega_2_ ** 2 + cshift_)
        int_res = numpy.real(numpy.trapz(factor, omega_1_))
        return -2.0 / pi * int_res
    
    if (source == "eps_2_rf") and (target == "eps_if"):
        return numpy.array([_KKR_eps2_to_eps_if(x, y, xx_) for xx_ in xx])  # Vectorize later
    if (source == "eps_2_rf") and (target == "eps_1_rf"):
        return numpy.array([_KKR_eps2_to_eps1(x, y, xx_) for xx_ in xx])
    if (source == "eps_1_rf") and (target == "eps_2_rf"):
        return numpy.array([_KKR_eps1_to_eps2(x, y, xx_) for xx_ in xx])  # Vectorize later

def matsubara_freq(n, T=300, mode="frequency"):
    """
    Calculate the Matsubara frequencies in rad/s
    Args:
        n: ndarray or int
    returns:
        freq: Matsubara frequencies in rad/s if mode=="frequency"
        otherwise: Matsubara energy in eV
    """
    if mode == "frequency":
        freq = 2 * pi * k * T / hbar * n
    elif mode == "energy":
        freq = 2 * pi * k * T / electron_volt * n 
    return freq
