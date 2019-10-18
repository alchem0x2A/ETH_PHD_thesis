import numpy
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
import os, os.path
from os.path import join, exists, abspath, dirname
from . import data_path, img_path
from helper import gridplots, grid_labels, savepgf



def integ(x_max=10, Delta=0.1):
    xx = numpy.linspace(0, x_max, 1024)
    y = xx * numpy.log(1 - Delta ** 2 * numpy.exp(-xx))
    cum = cumtrapz(y=y, x=xx, initial=0)
    return xx, cum


def plot_main():
    fig = plt.figure(figsize=(3.0, 2.5))
    ax = fig.add_subplot(111)
    fig, ax = gridplots(1, 1, r=0.8)
    for dd in numpy.arange(0.1, 1.0, 0.2):
        xx, cum = integ(Delta=dd)
        cum = cum / cum[-1]
        ax.plot(xx, cum / cum[-1], label="{0:.1f}".format(dd))
        if dd > 0.8:           # plot the limits
            cond_ = numpy.where(cum > 0.95)[0][0]
            ax.plot([xx[cond_], xx[cond_]], [0, cum[cond_]], "--", color="grey", alpha=0.6)
            ax.plot([0, xx[cond_]], [cum[cond_], cum[cond_]], "--", color="grey", alpha=0.6)
            ax.text(x=1, y=cum[cond_] * 0.98, s=r"$I(x) = 0.95I(\infty)$",
                    va="top", color="grey", alpha=0.8)
    l = ax.legend(loc=0)
    l.set_title(r"Model value of $\Delta$")
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"Normalized Integral $I(x) / I(\infty)$")

    # Add annotation
    ax.text(x=0.25, y=0.15,
            s=r"Cumulative Integral: $I(x) = {\displaystyle \int_0^{x}} x' \ln[1 - \Delta^{2}e^{-x'}] \mathrm{d} x'$",
            ha="left",
            transform=ax.transAxes)
    ax.annotate(r'Increasing $\Delta$', xytext=(2.8, 0.5),
                xy=(1, 0.8),
            arrowprops=dict(arrowstyle="->"),
            )
    savepgf(fig, img_path / "interg_xx.pgf")
    # fig.savefig(join(img_path, "integ_xx.svg"))

if __name__ == "__main__":
    plot_main()
