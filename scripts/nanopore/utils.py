from matplotlib.patches import Rectangle, Wedge
import numpy
from scipy.constants import e, epsilon_0, k, N_A, hbar, pi

"""
Utilities for adding components on the plot
"""

pixels = (512, 512)

values = ("V", "c_p", "c_n", "cpz", "cnz")

V_g = [0.001, 0.025, 0.05, 0.1, 0.15, 0.2] + \
    list(numpy.arange(0.25, 1.35, 0.1))

file_template = "{0}.npy"

concentrations = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1)

ratio_conc = 10**3              # 1 mol/L  to 1 mol/m^3


def delta_phi_gr(sigma):
    vf = 1.1e6
    fac = hbar * vf / e * numpy.sqrt(pi * numpy.abs(sigma) / e)
    return fac * numpy.sign(sigma)


def Debye_length(c0, eps=78,
                 zp=1, zn=-1):
    # Assume the c0 is in mol/L
    c0 *= ratio_conc
    I = (zp ** 2 * c0 + zn ** 2 * c0) / 2.0
    return (eps * epsilon_0 * k * 300 / (2 * N_A * e ** 2 * I)) ** 0.5


# reflex the data along the z-axis
def reflex_data(X, Y, D):
    XX = numpy.hstack((numpy.fliplr(-X), X))
    YY = numpy.hstack((numpy.fliplr(Y), Y))
    DD = numpy.hstack((numpy.fliplr(D), D))
    return XX, YY, DD


def get_data(filename):
    data = numpy.load(filename)  # load the npy data
    data[:, 0:2] /= 1e-9         # convert the data into nm scale
    return data


# Add path of graphene to the ax
# Both left and right edges
def add_graphene(ax, R_p=20, t_gr=0.67, t_helm=0.3):
    W = 40
    h = t_gr + 2 * t_helm
    h2 = t_gr
    facecolor_out = "#fafafa"
    facecolor_in = "#7c7c7c"
    rect1 = Rectangle((R_p, -h/2), W - R_p, h,
                      facecolor=facecolor_out)
    rect2 = Rectangle((-R_p, -h/2), -(W - R_p), h,
                      facecolor=facecolor_out)
    wedge1 = Wedge((R_p, 0), h / 2, 90, 270, facecolor=facecolor_out)
    wedge2 = Wedge((-R_p, 0), h / 2, -90, 90, facecolor=facecolor_out)

    rect3 = Rectangle((R_p, -t_gr/2), W - R_p, t_gr,
                      facecolor=facecolor_in)
    rect4 = Rectangle((-R_p, -t_gr/2), -(W - R_p), t_gr,
                      facecolor=facecolor_in)
    wedge3 = Wedge((R_p, 0), t_gr / 2,
                   90, 270, facecolor=facecolor_in)
    wedge4 = Wedge((-R_p, 0), t_gr / 2,
                   -90, 90, facecolor=facecolor_in)
    for a in [rect1, rect2, wedge1, wedge2,
              rect3, rect4, wedge3, wedge4]:
        ax.add_artist(a)


def plot_data(ax, X, Y, D, vmin=None, vmax=None):
    ax.set_aspect('equal')
    XX, YY, DD = reflex_data(X, Y, D)
    return ax.pcolormesh(XX, YY, DD,
                         vmin=vmin, vmax=vmax,
                         rasterized=True,
                         cmap="jet")


# if __name__ == "__main__":
#     data = get_data(os.path.join("../result/concentration", file_template.format(0.01)))
#     X = data[:, 0].reshape(*pixels); Y = data[:, 1].reshape(*pixels)
#     D = -data[:, 2 + 5 * 0+ 4].reshape(*pixels)

#     fig = plt.figure(figsize=(3, 3))
#     plt.style.use("science")
#     ax = fig.add_subplot(111)

#     mesh = plot_data(ax, X, Y, D)
#     print(mesh.get_clim())
#     add_graphene(ax)

#     # plt.xlim(-30, 30)
#     # plt.ylim(-20, 20)
#     ax.set_aspect('equal')

#     plt.colorbar(mesh, fraction=0.026, pad=0.04)
#     plt.tight_layout()
#     plt.savefig("../Test2.pdf")
