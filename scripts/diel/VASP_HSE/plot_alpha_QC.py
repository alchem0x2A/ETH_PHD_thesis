import matplotlib.pyplot as plt
from scipy.constants import pi
from scipy.stats import linregress
import json
import numpy

materials = json.load(open("../../data/HSE-data/all_data.json", "r"))
data = []
for m in materials:
    E_2D = m["gap_2D"]
    alphax_2D = m["alphax_2D"]
    alphaz_2D = m["alphaz_2D"]
    n_2D = m["n_2D"]
    qc_n = m["qc_n"]
    qc_p = m["qc_p"]
    if None not in (E_2D, alphax_2D, alphaz_2D,
                    n_2D, qc_n, qc_p):
        # alpha_3D = (eps_x - 1) * L / (4 * pi)
        data.append((E_2D, alphax_2D, alphaz_2D, qc_n, qc_p))

data = numpy.array(data)
plt.style.use("science")

plt.figure(figsize=(3.5, 3.5))
plt.plot(data[:, -2], data[:, 1], "o")
plt.xlabel("$C_{\\mathrm{Q}}^{\mathrm{C}}$ ($\\mathrm{\\mu F} \\cdot \\mathrm{cm}^{-2}$)")
plt.ylabel("$\\alpha^{\\parallel}_{\\rm{2D}}/(4 \\pi \\varepsilon_0)$ ($\\AA$)")
plt.xscale("log")
plt.savefig("../../tmp_img/alpha_xx_QC_n.svg")

plt.figure(figsize=(3.5, 3.5))
plt.plot(data[:, -1], data[:, 1],"o")
plt.xlabel("$C_{\\mathrm{Q}}^{\\mathrm{V}}$ ($\\mathrm{\\mu F} \\cdot  \\mathrm{cm}^{-2}$)")
plt.ylabel("$\\alpha^{\\parallel}_{\\rm{2D}}/(4 \\pi \\varepsilon_0)$ ($\\AA$)")
plt.xscale("log")
plt.savefig("../../tmp_img/alpha_xx_QC_p.svg")

plt.figure(figsize=(3.5, 3.5))
plt.plot(data[:, -2], data[:, 2],"o")
plt.xlabel("$C_{\\mathrm{Q}}^{\mathrm{C}}$ ($\\mathrm{\\mu F} \\cdot \\mathrm{cm}^{-2}$)")
plt.ylabel("$\\alpha^{\\perp}_{\\rm{2D}}/(4 \\pi \\varepsilon_0)$ ($\\AA$)")
plt.xscale("log")
plt.savefig("../../tmp_img/alpha_zz_QC_n.svg")

plt.figure(figsize=(3.5, 3.5))
plt.plot(data[:, -1], data[:, 2],"o")
plt.xlabel("$C_{\\mathrm{Q}}^{\\mathrm{V}}$ ($\\mathrm{\\mu F} \\cdot \\mathrm{cm}^{-2}$)")
plt.ylabel("$\\alpha^{\\perp}_{\\rm{2D}}/(4 \\pi \\varepsilon_0)$ ($\\AA$)")
plt.xscale("log")
plt.savefig("../../tmp_img/alpha_zz_QC_p.svg")



