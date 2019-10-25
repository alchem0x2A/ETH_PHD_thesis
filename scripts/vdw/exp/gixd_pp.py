# coding: utf-8
import numpy
import PIL.Image as Pim
# import matplotlib.pyplot as plt
# import os
# import os.path
# from os.path import join, exists, abspath, dirname
from scipy.signal import medfilt
from json import load
from . import data_path, img_path

# plt.style.use("science")


data_gixd = data_path / "GIXD" / "calibrated/"
# img_path = join(curdir, "../../img", "fig6")



def get_filename(name, date, condition):
    return data_gixd / date / condition / "{0}.tif".format(name)


def get_powder_data(name="xrd_powder.txt",
                    width=0.05, q_range=(3.5, 25)):
    """Get powder data from hkl data file
    """
    from scipy.stats import norm
    name = data_gixd / name
    raw_data = numpy.genfromtxt(name, delimiter=[4, 5, 5, 11,
                                                 13, 13, 13,
                                                 11, 11, 5])
    d = raw_data[1:, 3] / 10     # In naometer
    I = raw_data[1:, -2]
    q = numpy.pi * 2 / d        # In nm^-1
    I[numpy.isnan(I)] = 0       # Remove NaN values

    qq = numpy.linspace(*q_range, 2048)
    ii = numpy.array([norm(q_, width).pdf(qq) * i_
                      for q_, i_ in zip(q, I)])
    # ii[numpy.isnan(ii)] = 0
    ii = numpy.sum(ii, axis=0)  # Sum all gaussian peaks
    return qq, ii


def read_param(file_name, name, date, condition):
    with open(file_name, "r") as f:
        json_data = load(f)
    try:
        name = name.split("_")[0]
        return json_data[date][condition][name]
    except TypeError:
        return None


def get_gixd_data(name, date, condition,
                  q_range=None, *argc, **argv):
    """Get raw gixd data from origin point
    """
    assert condition in ["graphene", "no-graphene"]
    # assert date in ["0625", "0726", "1008", ""]
    f_name = get_filename(name, date, condition)
    img = Pim.open(f_name, mode="r")
    img_data = numpy.array(img)
    nx, ny = img_data.shape
    # Use scaling from datafile
    json_file = data_gixd / "scaling.json"
    param = read_param(json_file, name, date, condition)
    assert param is not None
    center = param["center"]
    scaling = param["scaling"]
    # len_px, len_q = ref         # length in px and length in Q (nm^-1)
    # scaling = len_q / len_px
    # print(name, scaling)
    X = numpy.arange(nx)
    Y = numpy.arange(ny)
    ox, oy = center
    # Substract center
    X = (ox - X) * scaling
    Y = (oy - Y) * scaling
    if q_range is not None:
        qxl, qxr, qyb, qyt = q_range
        x_, = numpy.where((qxl > X) & (X > qxr))  # Why?
        y_, = numpy.where((qyb < Y) & (Y < qyt))
        print(x_.min(), x_.max())
        print(y_.min(), y_.max())
        img_plot = img_data[y_.min(): y_.max() + 1,
                            x_.min(): x_.max() + 1]  # Very complicated!
        X = X[x_]
        Y = Y[y_]
    else:
        img_plot = img_data
        q_range = [X[0], X[-1], Y[-1], Y[0]]
    return X, Y, img_plot


# def show_image(name, date, condition,
#                v_minmax=None, q_range=None):
#     X, Y, img_plot = get_gixd_data(name, date, condition,
#                                    q_range)
#     if v_minmax is not None:
#         vmin, vmax = v_minmax  # min max for plot
#     else:
#         vmin = numpy.min(img_plot)
#         vmax = numpy.max(img_plot)
#     # Select plot region
#     # Plot job
#     fig = plt.figure(figsize=(3.0, 3.0))
#     ax = fig.add_subplot(111)
#     if name == "Au":
#         img_plot = (img_plot - img_plot.min()) ** 2.5
#         vmax = (vmax - vmin) ** 2.5
#         vmin = 0
#     ax.imshow(img_plot,
#               # normalized=True,
#               vmin=vmin, vmax=vmax,
#               extent=q_range,
#               rasterized=True)
#     ax.set_xlabel("$q_{xy}$ (nm$^{-1}$)")
#     ax.set_ylabel("$q_{z}$ (nm$^{-1}$)")
#     # ax.set_aspect(1)
#     fig.tight_layout()
#     fig.savefig(join(img_path,
#                      "{0}-2d-gixd.svg".format(name)))
#     del fig, ax


def angle_dist(X, Y, data, q_range=None):
    """\chi-averaged data """
    # Convert to radial
    XX, YY = numpy.meshgrid(X, Y)
    print(X)
    RR = numpy.sqrt(XX ** 2 + YY ** 2)
    TT = numpy.arctan(YY / XX)
    spacing = 1024
    II = data * RR
    # II = data
    R_1D = numpy.linspace(RR.min(), RR.max(), spacing + 1)  # Leave 1 spacing
    spectrum = []
    # BT-solve
    for i in range(spacing):
        r_min = R_1D[i]
        r_max = R_1D[i + 1]
        # tt = TT[(RR >= r_min) & (RR < r_max)
        # & ((XX > 2.5) | (YY > 3.5))].flatten()
        # ii = II[(RR >= r_min) & (RR < r_max)
        # & ((XX > 2.5) | (YY > 3.5))].flatten()
        ii = II[(RR >= r_min) & (RR < r_max)].flatten()
        # print(ii)
        if len(ii) > 0:
            spectrum.append(numpy.max(ii) - numpy.mean(ii))
            # spectrum.append(numpy.sum(ii))
        else:
            spectrum.append(0)

    spectrum = numpy.array(spectrum)
    if q_range is not None:
        qmin, qmax = q_range
        cond = numpy.where((R_1D >= qmin) & (R_1D <= qmax))
        R_1D = R_1D[cond]
        spectrum = spectrum[cond]
    spectrum = medfilt(spectrum, kernel_size=3)
    spectrum_norm = (spectrum - spectrum.min()) \
        / (spectrum.max() - spectrum.min())
    return R_1D, spectrum, spectrum_norm


# def plot_angle(names, maters, q_range):
#     fig = plt.figure(figsize=(6.0, 6.0))
#     ax = fig.add_subplot(111)
#     for i, name in enumerate(names):
#         X, Y, data = get_gixd_data(**maters[name],
#                                    q_range=q_range)  # Already cut
#         R_1D, spectrum, spectrum_norm = angle_dist(X, Y, data)
#         ax.plot(R_1D, spectrum_norm * 0.8 + i*0.5, label=name)

#     Q, I = get_powder_data(
#         join(curdir, "../../data/GIXD/calibrated/xrd_powder.txt"))
#     I = I / numpy.max(I)
#     ax.plot(Q, I, "k")
#     ax.legend(loc=0)

#     ax.set_xlim(3.5, 22)
#     ax.set_xlabel("$|q|$ (nm$^{-1}$)")
#     ax.set_ylabel("Intens. (a.u.)")
#     ax.set_yticks([])
#     fig.savefig(join(img_path,
#                      "diff_1d_data_nobg.png"))


if __name__ == "__main__":
    raise NotImplementedError("Do not run as module!")
    # origin = (680, 731)
    # scaling = 17.816 / 430.086
    # X, Y, img_data = get_gixd_data(join(curdir,
    #                                     "../../data/GIXD/calibrated/Au.tif"),
    #                                origin=origin,
    #                                scaling=scaling)
    # print(X, Y, img_data)
    # plt.imshow(img_data,
    #            extent=[X[0], X[-1],
    #                    Y[-1], Y[0]])
    # plt.show()

    # q_range = None
    # Zoom-in data
    # q_range = (10, -0.5, 12, 25)
    # q_range = (20, -2, -0.5, 25)
    # maters = dict()
    # condition = "graphene"
    # date_old = "0625"
    # date = "0726"
    # maters["Plasma_nobg"] = dict(name="Plasma_nobg",
    #                              date=date,
    #                              condition=condition,
    #                              # center=(680, 731),
    #                              # ref=(444.319, 18.523),
    #                              # v_minmax=(400, 9000),
    #                              )

    # maters["OTS_nobg"] = dict(name="OTS_nobg",
    #                           date=date,
    #                           condition=condition,
    #                           # center=(680, 730),
    #                           # ref=(469.165, 19.431),
    #                           # v_minmax=(500, 22000),
    #                           )

    # maters["Au_nobg"] = dict(name="Au_nobg",
    #                          date=date,
    #                          condition=condition,
    #                          # center=(680, 725),
    #                          # ref=(430.086, 17.816),
    #                          # v_minmax=(90, 320),
    #                          )

    # maters["Plasma"] = dict(name="Plasma_nobg",
    #                         date=date_old,
    #                         condition=condition,
    #                         # center=(680, 731),
    #                         # ref=(444.319, 18.523),
    #                         # v_minmax=(400, 9000),
    #                         )

    # maters["OTS"] = dict(name="OTS_nobg",
    #                      date=date_old,
    #                      condition=condition,
    #                      # center=(680, 730),
    #                      # ref=(469.165, 19.431),
    #                      # v_minmax=(500, 22000),
    #                      )

    # maters["Au"] = dict(name="Au_nobg",
    #                     date=date_old,
    #                     condition=condition,
    #                     # center=(680, 725),
    #                     # ref=(430.086, 17.816),
    #                     # v_minmax=(90, 320),
    #                     )

    # # Comment below for full-scale

    # show_image(**maters["Plasma_nobg"],
    #            q_range=q_range)

    # show_image(**maters["OTS_nobg"],
    #            q_range=q_range)

    # show_image(**maters["Au_nobg"],
    #            q_range=q_range)

    # q_sum_range = (25, 0, 0, 25)
    # plot_angle(["Plasma_nobg", "OTS_nobg", "Au_nobg",
    #             "Plasma", "OTS", "Au"],
    #            maters,
    #            q_range=q_sum_range)
