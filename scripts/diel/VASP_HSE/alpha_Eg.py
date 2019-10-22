import matplotlib.pyplot as plt
from scipy.constants import pi
import json
import numpy

materials = json.load(open("../../data/HSE-data/all_data.json", "r"))
data = []
for m in materials:
    E_2D = m["gap_2D"]
    alpha_2D = m["alphax_2D"]
    # E_3D = m["gap_3D"]
    # L = m["L_3D"]
    # eps_x = m["epsx_3D"]
    n_2D = m["n_2D"]
    if None not in (E_2D, alpha_2D, n_2D):
        # alpha_3D = (eps_x - 1) * L / (4 * pi)
        data.append((E_2D, alpha_2D, n_2D))

data = numpy.array(data)
# plt.plot(data[:, 0], data[:, 2], "o")  # 2D
# plt.plot(data[:, 1] ** -0.5, data[:, 4], "o")  # 2D
plt.plot(1/data[:, 1], data[:, 0], "o")
# plt.xlim(0, 2)

plt.savefig("../../tmp_img/test_alpha_EG.pdf")
    
