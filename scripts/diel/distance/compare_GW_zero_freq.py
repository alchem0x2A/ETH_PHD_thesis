import numpy
import matplotlib.pyplot as plt
import os

gw_in = numpy.array([[20, 1.4114],
                     [30, 1.2706],
                     [40, 1.2018],
                     [50, 1.1599],
                     [60, 1.1326]])

gw_out = numpy.array([[20, 1.2103],
                      [30, 1.1385],
                      [40, 1.1031],
                      [50, 1.0815],
                      [60, 1.0671]])

pbe_in = numpy.array([[20, 1.6458],
                      [30, 1.4297],
                      [40, 1.3225],
                      [50, 1.257],
                      [60, 1.2139]])

pbe_out = numpy.array([[20, 1.2852],
                       [30, 1.1892],
                       [40, 1.1413],
                       [50, 1.112],
                       [60, 1.0925]])

plt.style.use("science")

plt.figure(figsize=(6, 3.5))

plt.subplot(221)
plt.xlabel("$L$ ($\\rm{\\AA}$)")
plt.ylabel("$\\varepsilon_{\\rm{SL}}^{\parallel}$")
L_pbe = pbe_in[:, 0]
eps_pbe = pbe_in[:, 1]
L_gw = gw_in[:, 0]
eps_gw = gw_in[:, 1]
plt.plot(L_pbe, eps_pbe, "-o", label="PBE")
plt.plot(L_gw, eps_gw, "-o", label="G$_{0}$W$_{0}$")
plt.legend()

plt.subplot(222)
plt.xlabel("$L$ ($\\rm{\AA}$)")
plt.ylabel("$\\alpha_{\\rm{2D}}^{\parallel}/(4 \pi \\varepsilon_0)$ ($\\rm{\\AA}$)")
alpha_pbe = (eps_pbe - 1) * L_pbe / (4 * numpy.pi)
alpha_gw = (eps_gw - 1) * L_gw / (4 * numpy.pi)
plt.plot(L_pbe, alpha_pbe, "-o", label="PBE")
plt.plot(L_gw, alpha_gw, "-o", label="GW")
plt.ylim(0.5, 1.2)


plt.subplot(223)
plt.xlabel("$L$ ($\\rm{\AA}$)")
plt.ylabel("$\\varepsilon_{\\rm{SL}}^{\perp}$")
L_pbe = pbe_out[:, 0]
eps_pbe = pbe_out[:, 1]
L_gw = gw_out[:, 0]
eps_gw = gw_out[:, 1]

plt.plot(L_pbe, eps_pbe, "-o", label="PBE")
plt.plot(L_gw, eps_gw, "-o", label="PBE")
plt.ylim(1.05, 1.35)

plt.subplot(224)
plt.xlabel("$L$ ($\\rm{\\AA}$)")
plt.ylabel("$\\alpha_{\\rm{2D}}^{\perp}/(4\\pi \\varepsilon_{0})$ ($\\rm{\\AA}$)")
# alpha_pbe = (1 - 1 / eps_pbe) * L_pbe / (4 * numpy.pi) 
# alpha_gw = (1 - 1 / eps_gw) * L_gw / (4 * numpy.pi) 
alpha_pbe = (eps_pbe - 1) * L_pbe / (4 * numpy.pi) / 2
alpha_gw = ( eps_gw - 1) * L_gw / (4 * numpy.pi) / 2
plt.plot(L_pbe, alpha_pbe, "-o", label="PBE")
plt.plot(L_gw, alpha_gw, "-o", label="PBE")
plt.ylim(0.1, 0.25)

plt.tight_layout()
plt.savefig(os.path.join("../../tmp_img/",
                         "compare_alpha_zero.svg"))
