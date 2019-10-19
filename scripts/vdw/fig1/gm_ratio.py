import numpy
# Add the utils
from ..utils.kkr import kkr, matsubara_freq
from ..utils.lifshitz import alpha_to_eps, g_amb_alpha, g_amb, alpha_to_eps
from ..utils.img_tools import get_color, add_cbar
from ..utils.eps_tools import get_alpha, get_index, data_2D
from . import data_path, img_path


freq_matsu = matsubara_freq(numpy.arange(0, 1000),
                             mode="energy")

def gm2_(alpha, freq, d):
    # d is in nm, convert to Angstrom
    pi = numpy.pi
    alpha_x = alpha[0]
    alpha_z = alpha[2]
    # print(alpha_x)
    # eps_x = 1 + 4 * pi * alpha_x / d
    # eps_z = 1 / (1 - 4 * pi * alpha_z / d)
    # Take real distance
    eps_x = alpha_to_eps(alpha_x, d, direction="x")
    eps_z = alpha_to_eps(alpha_z, d, direction="z")
    # print(eps_x)

    # Convert to kkr!
    eps_x_iv = kkr(freq, eps_x, freq_matsu)
    eps_z_iv = kkr(freq, eps_z, freq_matsu)
    return eps_x_iv, eps_z_iv, eps_x_iv / eps_z_iv

def get_gm2(d_, force=False):            # in nm
    res_file = data_path / "2D" / "gm2_{:.1f}.npz".format(d_)
    if force is not True:
        if res_file.exists():
            data = numpy.load(res_file, allow_pickle=True)
            return data["names"], data["Eg"], data["eps_para"], \
                data["eps_perp"], data["gm2"]

    names = []
    Eg = []
    gm2 = []
    eps_para = []
    eps_perp = []
    list_matter = range(len(data_2D))
    for i in list_matter:
        print(i)
        alpha, freq, eg = get_alpha(i)
        formula = data_2D[i]["formula"]
        prototype = data_2D[i]["prototype"]
        # if alpha[2][0].real > 1:
            # continue            # probably bad data
        ex_, ez_, g_  = gm2_(alpha, freq, d_ * 1e-9)
        if numpy.max(1 / g_) > 1.1:
            continue
        #
        names.append("{}-{}".format(formula, prototype))
        Eg.append(eg)
        eps_para.append(ex_)
        eps_perp.append(ez_)
        gm2.append(g_)
    Eg = numpy.array(Eg)
    eps_para = numpy.array(eps_para)
    eps_perp = numpy.array(eps_perp)
    gm2 = numpy.array(gm2)
    numpy.savez(res_file.as_posix(), names=names, Eg=Eg,
                eps_para=eps_para, eps_perp=eps_perp, gm2=gm2)
    return names, Eg, eps_para, eps_perp, gm2

def main(force=False):
    d = 2
    get_gm2(d, force=force)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run main function")
    parser.add_argument("-f", "--force", dest="force", action="store_true")
    args = parser.parse_args()
    main(args.force)
