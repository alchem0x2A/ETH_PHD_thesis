# import os, os.path
import numpy
# from os.path import join, exists, dirname, abspath
from pathlib import Path
from .. import data_path

from scipy.constants import e, epsilon_0

# curdir = os.path.abspath(os.path.dirname(__file__))
# file_bulk = os.path.join(curdir, "../../data/bulk/bulk_eps_all.npz")
# file_2D = os.path.join(curdir, "../../data/2D/2D_alpha.npz")
# data_bulk = numpy.load(file_bulk, allow_pickle=True)["data"]
# data_2D = numpy.load(file_2D, allow_pickle=True)["data"]  # Requirements on newer version of numpy


file_bulk = data_path /  "bulk/bulk_eps_all.npz"
file_2D = data_path / "2D/2D_alpha.npz"
data_bulk = numpy.load(file_bulk, allow_pickle=True)["data"]
data_2D = numpy.load(file_2D, allow_pickle=True)["data"]  # Requirements on newer version of numpy

def get_alpha(index, correction=True):
    entry_2D = data_2D[index]
    alpha_m = numpy.array((entry_2D["alpha_x"],
                           entry_2D["alpha_y"],
                           entry_2D["alpha_z"]))
    freq_alpha = entry_2D["freq"]
    Eg = entry_2D["Eg_gllb"]
    # Treat intra part of graphene
    if correction and (index == 3):              # graphene
        alpha_intra = 1j * 1 / 4 * e / (freq_alpha + 1e-3) / (4 * numpy.pi * epsilon_0) / 1e-10  # in Angstrom
        # print(alpha_intra)
        # print(alpha_m[0, :])
        alpha_m[0, :] = alpha_m[0] + alpha_intra
        alpha_m[1, :] = alpha_m[1] + alpha_intra
    return alpha_m, freq_alpha, Eg[0]

def get_eps(index):             # get eps from file
    entry = data_bulk[index]
    # print(entry)
    eps = numpy.array((entry["eps_x_iv"],
                       entry["eps_y_iv"],
                       entry["eps_z_iv"]))
    freq = entry["freq_imag"]
    gap_min = entry["gap"]
    gap_dir = entry["gap_dir"]
    return eps, freq, gap_min, gap_dir

def get_index(mater, kind="2D"):
    kind = kind.lower()
    assert kind in ["2d", "bulk"]
    if kind == "2d":
        data = data_2D
        formula, prototype = mater
        index = [i for i in range(len(data)) \
                if (data[i]["formula"] == formula) \
                    and (data[i]["prototype"] == prototype)]
    else:
        data = data_bulk
        index = [i for i in range(len(data)) \
                 if (data[i]["name"] == mater)]
    if len(index) > 0:
        return index[0]
    else:
        return None
