import matplotlib
import numpy

def get_color(c, min=0, max=12, levels=1000):
    # Helper function to plot colors
    # default colormap is rainbow 
    dc = (max - min) / levels
    i = int((c - min) / dc)
    color = matplotlib.cm.rainbow(numpy.linspace(0, 1, levels + 1))
    return color[i]

def add_cbar(fig, ax, n_min=0, n_max=12):
    # Helper function to add colorbar to fig and ax
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # plot the colorbar outside
    sc = ax.scatter([xlim[0] - 10, xlim[0] - 10],
                    [ylim[0] - 10, ylim[0] - 10],
                    c=[n_min, n_max],  # automated vmin vmax
                    cmap="rainbow")
    cb = fig.colorbar(sc)
    # cb.set_ticks([1, 5, 10, 15])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim) 
    return cb
