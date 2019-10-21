esem=[["NAME", "CA", "CA-err", "WF", "WF-err"], ["PSS", 73.97, 3.92, 4.98, 0.092], ["PAA", 75.0, 2.96, 4.96, 0.096], ["SiO2", 80.88, 2.95, 4.6, 0.026], ["PAH", 75.01, 4.02, 4.16, 0.05], ["PLL", 74.03, 1.98, 4.12, 0.09]]
elw=[["V", "CA"], [-100, 78], [0, 88], [100, 60]]
import scipy
import scipy.constants as const
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
import numpy
from dcos_sigma import cal_2D
import pycse.orgmode as org

v_f = 1.1e6
eps_sio2 = 3.9



def e_cm2_to_SI(n):
    return n*const.e*10**4

def SI_to_e_cm2(sigma):
    return sigma/const.e/10**4

def EF_gr_from_sigma(sigma):
    A = scipy.sign(sigma)*const.hbar*v_f/const.e
    B = scipy.sqrt(scipy.pi*scipy.absolute(sigma)/const.e)
    return A*B

def sigma_from_EF(EF):
    return scipy.sign(EF)*EF**2*const.e**3/const.pi/const.hbar**2/v_f**2

def sigma_from_sio2(V_M, sigma0=0, t=280e-9):
    Cox = const.epsilon_0*eps_sio2 / t
    # VM to be voltage applied to 2D surface
    return Cox*V_M + sigma0


EF_plt = numpy.linspace(-0.8, 0.8, 200)
sigma_plt = sigma_from_EF(EF_plt)
n_plt = SI_to_e_cm2(sigma_plt)/10**13

c0 = 10**-7*1000

dcos_plt = cal_2D(c0, sigma_plt, add_MD=True)

# Data for the ESEM
data_esem = []
sigma_esem = []
sigma_err_esem = []
dcos_esem = []
dcos_err_esem = []

esem_label = []

theta0_esem = 81.0
dcos0_esem = scipy.cos(theta0_esem/180*scipy.pi)
for line in esem[1:]:
    esem_label.append(line[0])
    data_esem.append(line[1:])
    dEF = line[3] - 4.6
    dEF_r = dEF+line[4]
    dEF_l = dEF-line[4]
    sigma = sigma_from_EF(dEF)
    sigma_l = sigma_from_EF(dEF_l)
    sigma_r = sigma_from_EF(dEF_r)
    ca = line[1]
    ca_err = line[2]
    dcos = scipy.cos(ca/180*scipy.pi) - dcos0_esem
    dcos_l = scipy.cos((ca+ca_err)/180*scipy.pi) - dcos0_esem
    dcos_r = scipy.cos((ca-ca_err)/180*scipy.pi) - dcos0_esem
    sigma_esem.append(sigma)
    sigma_err_esem.append([abs(sigma-sigma_l), abs(sigma-sigma_r)])
    dcos_esem.append(dcos)
    dcos_err_esem.append([abs(dcos-dcos_l), abs(dcos-dcos_r)])

sigma_esem = numpy.array(sigma_esem)
sigma_err_esem = numpy.array(sigma_err_esem)
n_esem = SI_to_e_cm2(sigma_esem)/10**13
n_err_esem = numpy.transpose(SI_to_e_cm2(sigma_err_esem))/10**13
dcos_esem = numpy.array(dcos_esem)
dcos_err_esem = numpy.transpose(numpy.array(dcos_err_esem))

nn_esem = numpy.linspace(-2, 2, 200)
dcos_theory = cal_2D(10**-7, sigma_esem, add_MD=True)
param_esem = numpy.polyfit(n_esem, dcos_esem - dcos_theory, 2)
func_esem = numpy.poly1d(param_esem)
dd_esem = func_esem(nn_esem)


l_D = scipy.sqrt(const.epsilon_0*80*const.k*298/(2*c0*const.N_A*const.e**2))
C_D = const.epsilon_0*80/l_D
func_esem_max = lambda s: 1/2*s**2/C_D
dcos_esem_max = func_esem_max(sigma_plt)/0.072

param_esem_max = numpy.polyfit(n_plt, dcos_esem_max, 2)
f_esem = param_esem[0]/param_esem_max[0]
sigma_i_esem = -(nn_esem[numpy.argmin(dd_esem)])
# print(f_esem, sigma_i_esem)

# Data for electrowetting
theta0_elw = 88
dcos0_elw = scipy.cos(theta0_elw/180*scipy.pi)
data_elw = numpy.array(elw[1:])
sigma_elw = sigma_from_sio2(data_elw[:,0])
n_elw = SI_to_e_cm2(sigma_elw)/10**13
dcos_elw = scipy.cos(data_elw[:,1]/180*scipy.pi) - dcos0_elw

dcos_theory = cal_2D(10**-7, sigma_elw, add_MD=True)
param_elw = numpy.polyfit(n_elw, dcos_elw - dcos_theory, 2)
func_elw = numpy.poly1d(param_elw)

C_ox = const.epsilon_0*eps_sio2/280e-9
func_elw_max = lambda s: 1/2*s**2/C_ox
dcos_elw_max = func_elw_max(sigma_plt)/0.072
param_elw_max = numpy.polyfit(n_plt, dcos_elw_max, 2)

nn_elw = numpy.linspace(-1.5, 1, 200)
dd_elw = func_elw(nn_elw)

f_elw = param_elw[0]/param_elw_max[0]
sigma_i_elw = -(nn_elw[numpy.argmin(dd_elw)])
# print(f_elw, sigma_i_elw)


def plot_fitting_f(fig):
    ax = fig.add_subplot(111)
    ax.plot(n_plt, dcos_plt, color="#666666", label="Theoretical",
            alpha=0.8)
    l_esem = ax.errorbar(x=n_esem, y=dcos_esem,
                xerr=n_err_esem, yerr=dcos_err_esem,
                         fmt="s", label="ESEM Data",)
    l_elw = ax.plot(n_elw, dcos_elw, "o", label="Electrowetting Data")
    ax.text(x=-0.85, y=0.25, ha="left", size="medium",
            s= "".join((r"$f$=",
	                "{:.3f}\n".format(f_elw),)
	                ),
    )
    ax.plot(nn_elw, dd_elw + dcos_plt, "--", alpha=0.5, color=l_elw[0].get_color())
    ax.plot(nn_esem, dd_esem + dcos_plt, "--", alpha=0.5, color=l_esem[0].get_color())

    ax.text(x=0.85, y=0.2, ha="left", size="medium",
            s= "".join((r"$f$=",
	                "{:.3f}\n".format(f_esem),)
	                ),
    )
    ax.set_xlabel(r"$\sigma_{\mathrm{2D}}$ ($10^{13}$ $e\cdot$cm$^{-2}$)")
    ax.set_ylabel(r"$\Delta\cos\theta$")
    ax.legend(loc=0)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.05, 0.5)
    fig.tight_layout()

matplotlib.style.use("science")
fig = plt.figure(figsize=(4.0, 3.0))

if __name__ == "__main__":
    plot_fitting_f(fig)
    org.figure(plt.savefig("../img/plot-fitting.pdf"),
	       attributes=[("latex", ":width 0.95\linewidth")],
	       label="fig:f-nc-exp",
	       caption=("Theoretical and fitted experimental data of "
                        r"$\Delta\cos\theta$ "
                        "as a function of "
                        r"$\sigma_{\mathrm{2D}}$. "
                        "The electrowetting data are extracted from Ref. "
                        "[[cite:hong_mechanism_2016]]; "
                        "the ESEM data are extracted from Ref. "
                        "[[cite:ashraf_doping-induced_2016]]. "))
