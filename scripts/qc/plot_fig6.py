import numpy as np
from . import data_path, img_path
from .constants import Const
from . import equations as eqs
from scipy.integrate import cumtrapz
from scipy.io import loadmat
from scipy.optimize import fsolve
from scipy.interpolate import BivariateSpline
from helper import gridplots, grid_labels, savepgf
from helper import get_color, add_cbar, add_img_ax
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Plot the figures for fig 6
# import matplotlib
# matplotlib.rcdefaults()


def cal_qc(a, EF, DOS, off_l, off_r):
    e = 1.60218e-19
    ef = np.min(np.abs(DOS[:, 0] - EF))
    index = np.argmin(np.abs(DOS[:, 0] - EF))
    # Handle misaligned data cells
    for i in range(1, DOS.shape[0] - 2):
        if DOS[i, 0] == DOS[i + 1, 0]:
            DOS[i, :] = DOS[i - 1, :] * 0.51 + DOS[i + 2, :] * 0.49
            DOS[i + 1, :] = DOS[i - 1, :] * 0.49 + DOS[i + 2, :] * 0.51

    ef = np.min(np.abs(DOS[:, 0] - EF))
    index_EF = np.argmin(np.abs(DOS[:, 0] - EF))
    # [ef ,index_EF] = min(abs(DOS(:,1) - EF));
    # np.where gives a tuple
    index_E_VB = np.where(DOS[index_EF - 10000:index_EF, 1] > 0)[0][-1]
    index_E_CB = np.where(DOS[index_EF: index_EF + 10000, 1] > 0)[0][0]
    index_E_VB = index_EF - 10000 + index_E_VB
    index_E_CB = index_EF + index_E_CB
    tDOS = cumtrapz(y=DOS[:, 1], x=DOS[:, 0])
    # print(DOS[index_E_VB - off_l - 1000: index_E_VB - off_l])
    # print(tDOS[index_E_VB - off_l - 1000: index_E_VB - off_l])
    # For P4, transformed to hexagonal a
    A = (a * 100) ** 2 * np.sin(np.pi / 3)
    QC = DOS[:, 1] * e / A * 1e6
    QC = QC.flatten()
    # print("QC shape", QC.shape)

    # %p-region
    # print(index_E_CB, index_E_VB, off_l, off_r)
    tN_p = tDOS[index_E_VB - off_l - 5000: index_E_VB - off_l + 1] - \
        tDOS[index_E_VB - off_l]  # enforce negative
    tN_p = -tN_p / A
    QC_p = QC[index_E_VB - off_l - 5000: index_E_VB - off_l + 1]
    # Q_QC_p = QC[index_E_VB - off_l - 1000: index_E_VB - off_l]

    # %n-region
    tN_n = tDOS[index_E_CB + off_r: index_E_CB +
                off_r + 2000] - tDOS[index_E_CB + off_r]
    tN_n = tN_n / A
    QC_n = QC[index_E_CB + off_r: index_E_CB + off_r + 2000]
    # Q_QC_n = QC[index_E_CB + off_r: index_E_CB + off_r + 1000]
    # print(tN_n, QC_n, -tN_p, QC_p)
    return tN_n.flatten(), QC_n.flatten(), tN_p.flatten(), QC_p.flatten()


def plot_ab(fig, axes):
    ax_a, ax_b, _ = axes
    e = 1.60218e-19
    seps_n = [3, 3, 2, 3, 4, 4, 5, 5, 5, 10]
    seps_p = [1, 3, 2, 1, 3, 3, 1, 1, 1, 10]
    shape = ['v-', '^-', 'x-', 'd-', '+-', 'h-', '-', '-', '-', 's-']
    colors = np.array([[207, 63, 30], [0, 0, 0], [146, 39, 133], [20, 112, 48], [239, 130, 179],
                       [15, 94, 172], [230, 0, 28], [91, 180, 55], [44, 62, 147], [238, 140, 60]]) / 255

    materials = ['MoS2', 'MoSe2', 'MoTe2', 'WS2', 'WSe2', 'WTe2',
                 'Graphene', 'Silicene', 'Germanene', 'Phosphorene']
    offset = [[18, 2],
              [1, 1],
              [1, 1],
              [60, 1],
              [1, 1],
              [1, 1],
              [0, 0],
              [0, 0],
              [0, 0],
              [550, 640], ]

    for i, mater in enumerate(materials):
        fn = data_path / "CQ" / "DOS_{0}.mat".format(mater)
        data = loadmat(fn)
        a = data["a"]
        EF = data["EF"]
        DOS = data["DOS"]
        l, r = offset[i]
        # Conversion between Matlab and python
        l = l + 1
        r = r + 1
        print(l, r, mater, shape[i])
        Q_n, QC_n, Q_p, QC_p = cal_qc(a, EF, DOS, l, r)
        if mater == "WTe2":
            QC_n = QC_n * 1.04  # slight offset
        print(Q_n.shape, QC_n.shape, Q_p.shape, QC_p.shape)
        cond_n = np.where(Q_n < 3e13)
        cond_p = np.where(Q_p < 3e13)
        Q_n = Q_n[cond_n]; QC_n = QC_n[cond_n]
        Q_p = Q_p[cond_p]; QC_p = QC_p[cond_p]
        # Reset the edge width
        if mater in ("MoTe2", "WSe2"):
            mew = 1
        else:
            mew = 0

        if "2" in mater:
            lb = mater.replace("2", "$_{2}$")
        else:
            lb = mater
        ln, = ax_a.plot(Q_n[::seps_n[i]] / 1e13, QC_n[::seps_n[i]], shape[i],
                        color=colors[i], markeredgewidth=mew)
        lp, = ax_b.plot(Q_p[::seps_p[i]] / 1e13, QC_p[::seps_p[i]], shape[i],
                        color=colors[i], markeredgewidth=mew)
        ll, = _.plot([], [], shape[i], color=colors[i], markeredgewidth=mew, label=lb)

        # ln, = ax_a.plot(Q_n / 1e13, QC_n, shape[i],
                        # color=colors[i])
        # lp, = ax_b.plot(Q_p / 1e13, QC_p, shape[i],
                        # color=colors[i])

    ax_a.set_xlim(0, 2)
    ax_a.set_ylim(0, 80)
    ax_b.set_xlim(0, 2)
    ax_b.set_ylim(0, 230)
    # Lables
    ax_a.set_xlabel("$\\sigma_{\\mathrm{2D}}^{\\mathrm{n}}$ ($10^{13}$ $e \cdot{}$cm$^{-2}$)")
    ax_a.set_ylabel("$C_{\\mathrm{Q}}$ (μF$\cdot{}$cm$^{-2}$)")
    
    ax_b.set_xlabel("$\\sigma_{\\mathrm{2D}}^{\\mathrm{p}}$ ($10^{13}$ $e \cdot{}$cm$^{-2}$)")
    ax_b.set_ylabel("$C_{\\mathrm{Q}}$ (μF$\cdot{}$cm$^{-2}$)")

    # ax_b.legend(bbox_to_anchor=(1.02, 0.0, 0.3, 1.0),
                # bbox_transform=ax_b.transAxes,
                # loc="center left")

    _.set_axis_off()
    _.legend(loc="center left", labelspacing=0.7)
    # ax_legend.legend(loc=0)
    


def plot_c(fig, ax):
    add_img_ax(ax, img_path / "sub_img" / "2D_struct.png")
    ax.text(x=0.125, y=-0.1, s="Graphene",
            ha="center",
            transform=ax.transAxes)
    ax.text(x=0.375, y=-0.1, s="Silicene/Germanene",
            ha="center",
            transform=ax.transAxes)
    ax.text(x=0.625, y=-0.1, s="2H-TMDCs",
            ha="center",
            transform=ax.transAxes)
    ax.text(x=0.875, y=-0.1, s="Phosphorene",
            ha="center",
            transform=ax.transAxes)


def plot_main():
    w = 1.4
    h = 1.2
    fig, ax = gridplots(2, 3, r=1, ratio=h / w * 1.8,
                        span=[(0, 0, 1, 1),
                              (0, 1, 1, 1),
                              (0, 2, 1, 1),
                              (1, 0, 1, 3)],
                        gridspec_kw=dict(width_ratios=(w, w, 3 - 2 * w),
                                         height_ratios=(h, 2 - h)))
    # plot_a(fig, ax[0])
    # plot_b(fig, ax[1])
    plot_ab(fig, [ax[0], ax[1], ax[2]])
    plot_c(fig, ax[3])
    grid_labels(fig, [ax[0], ax[1], ax[3]], offsets=[(0, 0), (-0.06, 0), (0, -0.05)])
    savepgf(fig, img_path / "QC_compare.pgf", preview=True)
    # fig.savefig(img_path / "QC_compare.png")


if __name__ == '__main__':
    plot_main()
