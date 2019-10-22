import numpy
import os
import matplotlib.pyplot as plt


def calc_alpha_freq(L=20,
                    direction="in_plane",
                    method="BSE",):
    assert direction in ("in_plane", "out_of_plane")
    # assert method in ("PBE", "GW")
    base_dir = os.path.join(os.path.dirname(__file__),
                            "../../data/distance/GW/BSE/{0:d}/{1}/".format(L,
                                                                        direction))
    img_file = os.path.join(base_dir, "eps_imaginary_{}.dat".format(direction))
    real_file = os.path.join(base_dir, "eps_real_{}.dat".format(direction))
    img_data = numpy.genfromtxt(img_file)
    real_data = numpy.genfromtxt(real_file)
    freq = img_data[:, 0]
    eps = real_data[:, 1] + 1j * img_data[:, 1]
    if direction == "in_plane":
        alpha = (eps - 1) * L / (numpy.pi * 4)
    else:
        alpha = (eps - 1) * L / (numpy.pi * 4)
    return freq, alpha, eps

def main(method="BSE", direction="in_plane"):
    fig = plt.figure(figsize=numpy.array((6, 4)) * 1.2)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel("Energy (eV)")
    if method == "GW":
        if direction == "in_plane":
            lim = [0, 12]
        else:
            lim = [0, 17]
    else:
        if direction == "in_plane":
            lim = [0, 10]
        else:
            lim = [0, 15]
    # for ax in [ax3, ax4]:
        # ax.set_xlim(6, )
    # Set ylabel
    if direction == "in_plane":
        tag = "\\parallel"
    else:
        tag = "\\perp"
        
    ax1.set_ylabel("Re $\\varepsilon^{{{0}}}_{{\\rm{{SL}}}}(\\omega)$".format(tag))
    ax2.set_ylabel("Re $\\alpha^{{{0}}}_{{\\rm{{2D}}}}(\\omega)/(4\\pi \\varepsilon_0)$ ($\\rm{{\\AA}}$)".format(tag))
    ax3.set_ylabel("Im $\\varepsilon^{{{0}}}_{{\\rm{{SL}}}}(\\omega)$".format(tag))
    ax4.set_ylabel("Im $\\alpha^{{{0}}}_{{\\rm{{2D}}}}(\\omega)/(4\\pi \\varepsilon_0)$ ($\\rm{{\\AA}}$)".format(tag))
    
    for L in (20, 30, 40,):
        freq, alpha, eps = calc_alpha_freq(L, direction=direction, method=method)
        cond = numpy.where((freq > lim[0]) & (freq < lim[1]))
        freq = freq[cond]
        alpha = alpha[cond]
        eps  = eps[cond]
        # freq_out, alpha_out, eps_out = calc_alpha_freq(L, direction="out_of_plane", method=method)
        ax1.plot(freq, eps.real, label="{} $\\rm{{\\AA}}$".format(L), linewidth=1.25)
        ax2.plot(freq, alpha.real, linewidth=1.25)
        ax3.plot(freq, eps.imag, linewidth=1.25)
        ax4.plot(freq, alpha.imag, linewidth=1.25)
    l = ax1.legend()
    l.set_title("{} $L$".format("BSE"))
    fig.tight_layout()
    fig.savefig("../../tmp_img/comparison-BSE-{}.svg".format(direction))
    return


if __name__== "__main__":
    plt.style.use("science")
    main("BSE", "in_plane")
    main("BSE", "out_of_plane")
    # main("PBE", "out_of_plane")
    # main("GW", "in_plane")
    # main("GW", "out_of_plane")
