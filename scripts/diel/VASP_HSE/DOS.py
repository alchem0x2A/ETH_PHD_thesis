import matplotlib.pyplot as plt
from scipy.constants import pi
import json
import numpy

materials = json.load(open("../../data/HSE-data/all_data.json", "r"))
data = []
for m in materials:
    E_2D = m["gap_2D"]
    alpha_2D = m["alphax_2D"]
    E_3D = m["gap_3D"]
    L = m["L_3D"]
    qc_n = m["qc_n"]
    qc_p = m["qc_p"]
    if None not in (E_2D, alpha_2D, E_3D, L, qc_n, qc_p):
        data.append((E_2D, alpha_2D, qc_n, qc_p))

data = numpy.array(data)
# plt.plot(data[:, 0], data[:, 2], "o")  # 2D
# plt.plot(data[:, 2], data[:, 1], "o")  # 2D
plt.plot(data[:, 3], data[:, 1], "o")  # 2D
plt.yscale("log")
# plt.xlim(0, 2)

plt.savefig("../../tmp_img/test_alpha_QC.pdf")
    
