import numpy
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz, trapz
import os, os.path
from os.path import join, exists, abspath, dirname
from . import data_path, img_path
from helper import gridplots, grid_labels, savepgf



def integ(x_max=10, Delta=0.1):
    xx = numpy.linspace(0, x_max, 1024)
    y = xx * numpy.log(1 - Delta ** 2 * numpy.exp(-xx))
    cum = cumtrapz(y=y, x=xx, initial=0)
    return xx, cum

def single_integ(Delta, x_max=15):
    xx = numpy.linspace(0, x_max, 1024)
    y = xx * numpy.log(1 - Delta * numpy.exp(-xx))
    cum = trapz(y=y, x=xx)
    return cum


def plot_main():
    fig, ax = gridplots(1, 2, ratio=2.5)
    # First plot the initial part
    ax_ = ax[0]
    dd = numpy.linspace(-1, 1, 128)  # Delta values
    yy = numpy.array([single_integ(d_) for d_ in dd])
    ax_.plot(dd, yy)
    ax_.plot(dd, -dd, "--", color="grey", alpha=0.8)
    ax_.set_xlabel(r"$\Gamma$")
    ax_.set_ylabel(r"$I(\Gamma, \infty)$")
    # Second plot the cumtrapz part?
    ax_.annotate(r"$I(-\Gamma, \infty) \approx -\Gamma$", xy=(-0.4, 0.4), xytext=(0, 0.4),
                 arrowprops=dict(arrowstyle="->"),
                 color="grey", alpha=0.8)
    ax_.text(x=0.15, y=0.15,
            s=r"$I(\Gamma, x) = {\displaystyle \int_0^{x}} x' \ln(1 - \Gamma e^{-x'}) \mathrm{d} x'$",
            ha="left",
             # size="small",
            transform=ax_.transAxes)
    ax_ = ax[1]
    for dd in numpy.arange(0.1, 1.0, 0.2):
        xx, cum = integ(Delta=dd)
        cum = cum / cum[-1]
        ax_.plot(xx, cum / cum[-1], label="{0:.1f}".format(dd))
        if dd > 0.8:           # plot the limits
            cond_ = numpy.where(cum > 0.95)[0][0]
            ax_.plot([xx[cond_], xx[cond_]], [0, cum[cond_]], "--", color="grey", alpha=0.6)
            ax_.plot([0, xx[cond_]], [cum[cond_], cum[cond_]], "--", color="grey", alpha=0.6)
            ax_.text(x=0.5, y=cum[cond_] * 0.98, s=r"$I(x) = 0.95I(\infty)$",
                    va="top", color="grey", alpha=0.8)
    l = ax_.legend(loc=0)
    l.set_title(r"Model value of $\Gamma$")
    ax_.set_xlabel("$x$")
    ax_.set_ylabel(r"$I(x, \Gamma) / I(x, \infty)$")
    ax_.annotate(r'Increasing $\Gamma$', xytext=(2.8, 0.5),
                xy=(1, 0.8),
                 arrowprops=dict(arrowstyle="->"),)
    grid_labels(fig, ax)

    # Add annotation
    savepgf(fig, img_path / "interg_xx.pgf")
    # fig.savefig(join(img_path, "integ_xx.svg"))

if __name__ == "__main__":
    plot_main()
