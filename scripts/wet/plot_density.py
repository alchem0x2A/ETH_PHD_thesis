import numpy, matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
import scipy.constants as const
import scipy
# import pycse.orgmode as org
from scipy.interpolate import interp1d
from . import data_path, img_path
from helper import gridplots, grid_labels, savepgf

charge_per_atom = [-12, -6, 0, 6, 12]
name = ["neg0.012", "neg0.006", "0.000", "0.006", "0.012"]
label_name = ["-0.012", "-0.006", "0", "0.006", "0.012"]

c_atom_to_sigma = lambda x: x*2/(2.465e-8**2*scipy.sin(scipy.pi/3))
z_gr = 1.980

f_charge_base = "6_11_17_data/charge_int_{}_large2.xvg"
f_charge_water = "6_11_17_data/charge_int_water-surf.xvg"

f_dens_base = "6_11_17_data/density_int_{}_large2.xvg"
f_dens_water = "6_11_17_data/density_int_water-surf.xvg"

charge_per_atom.sort()

c_water = numpy.genfromtxt(data_path / f_charge_water, delimiter=(12, 17), skip_header=19)
d_water = numpy.genfromtxt(data_path / f_dens_water, delimiter=(12, 17), skip_header=19)

# ax1.plot(c_water[:, 0] - z_gr, c_water[:, 1], label="Water Only")

def plot_den(ax, what="mass"):
    if what is "mass":
        for index, c in enumerate(charge_per_atom):
            d_sys = numpy.genfromtxt(data_path / f_dens_base.format(name[index]),
                                     delimiter=(12, 17), skip_header=19)
            # special treatment for negative value using spline
            d_sys_x = d_sys[:, 0]; d_sys_y = d_sys[:, 1]
            cond = numpy.where(d_sys_y > 100)
            d_sys_x = d_sys_x[cond]; d_sys_y = d_sys_y[cond]
            zz = numpy.linspace(d_sys_x.min(), d_sys_x.max(), 50000)
            f_yy = interp1d(d_sys_x, d_sys_y, kind="cubic")
            yy = f_yy(zz)
            zzz = numpy.linspace(0, d_sys_x.min(), 50000)
            f_yyy = lambda x: d_sys_y[0]*scipy.exp(150*(x-d_sys_x.min()))
            yyy = f_yyy(zzz)
            zz = numpy.hstack((zzz, zz))
            yy = numpy.hstack((yyy, yy))
            # ax.plot(d_sys[:, 0] - z_gr,
                    # d_sys[:, 1], label=r"%d$\times10^{-3}$ $e$/atom" % (c))
            ax.plot(zz - z_gr,
                    yy, label=r"%s $e$/atom" % (label_name[index]))
        ax.set_ylabel(r"$\rho_{\mathrm{L}}$ (kg$\cdot$m$^{-3}$)")
        ax.set_xlabel(r"$z$ (nm)")
        ax.set_xlim(0, 1)
        # ax.legend(loc=0, title=r"$\sigma_{\mathrm{2D}}$")
    elif what is "charge":
        for index, c in enumerate(charge_per_atom):
            c_sys = numpy.genfromtxt(data_path / f_charge_base.format(name[index]),
                                     delimiter=(12, 17), skip_header=19)
            zz = numpy.linspace(c_sys[:, 0].min(), c_sys[:, 0].max(), 50000)
            f_y = interp1d(c_sys[:, 0], c_sys[:, 1], kind="cubic")
            yy = f_y(zz)
            # ax.plot(c_sys[:, 0] - z_gr, c_sys[:, 1],
                    # label=r"%d$\times10^{-3}$ $e$/atom" % (c) )
            ax.plot(zz - z_gr, yy,
                    label=r"%s $e$/atom" % (label_name[index]) )
        ax.set_ylabel(r"$\delta_{\mathrm{L}}$ ($e\cdot$nm$^{-3}$)")
        ax.set_xlabel(r"$z$ (nm)")
        ax.set_xlim(0, 1)
        ax.legend(loc=0, title=r"$\sigma_{\mathrm{2D}}$")


def plot_main():
    fig, ax = gridplots(1, 2, r=0.85, ratio=2.2)
    plot_den(ax[0], what="mass")
    plot_den(ax[1], what="charge")
    grid_labels(fig, ax)
    savepgf(fig, img_path / "MD_density.pgf")

if __name__ == "__main__":
    plot_main()
    # matplotlib.style.use("science")

    # fig = plt.figure(figsize=(2.5, 2.5))
    # plot_den(fig, what="mass")
    # org.figure(plt.savefig("../img/density_m_small.pdf"))
    # plt.cla()

    # fig = plt.figure(figsize=(2.5, 2.5))
    # plot_den(fig, what="charge")
    # org.figure(plt.savefig("../img/density_c_small.pdf"))
