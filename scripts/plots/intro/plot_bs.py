# plot band structure vs structure
from gpaw import GPAW
from helper import gridplots, grid_labels, add_img_ax, savepgf
from . import data_path, img_path
import matplotlib as mpl

mpl.rcParams.update(**{"figure.constrained_layout.hspace": 0,
                     "figure.constrained_layout.wspace": 0})

names = dict(gr="Graphene",
             p="Phosphorene",
             mos2="2H-MoS$_{2}$",
             bn="hBN",)
gap = dict(gr=0,
           p=1.5,
           mos2=2.0,
           bn=5.9,)
color = dict(gr="#545454",
             p="#ff9945",
             mos2="#f2de6d",
             bn="#b26df2",)

def get_bs(name, gap=0):
    gpw = data_path / "bs" / "{0}-bs.gpw".format(name)
    g = GPAW(gpw.as_posix())
    nb = int(g.get_number_of_electrons() // 2)
    # print(nb)
    bs = g.band_structure()
    # eigen, kpts, k_ax =
    eigen = bs.energies[0]      # spin paired
    all_kpts, kpts, labels = bs.get_labels()
    l = []
    for s in labels:
        if s != "G":
            l.append(s)
        else:
            l.append("Γ")
    labels = l
    efermi = bs.reference
    # for i in range(eigen.shape[1] - 1):
        # if (eigen[:, i].max() <= efermi) \
           # and (eigen[:, i + 1].min() > efermi):
            # break
    # vb = eigen[eigen <= efermi]
    # cb = eigen[eigen > efermi]
    # i = 3
    vb = eigen[:, :nb]
    cb = eigen[:, nb:]
    vb = vb - vb.max()
    cb = cb - cb.min() + gap
    return vb, cb, all_kpts, kpts, labels
    # return eigen, all_kpts, kpts, labels

def plot_main():
    fig, ax = gridplots(2, 4, r=1, ratio=1.7,
                        gridspec_kw=dict(height_ratios=[0.4, 0.6]))
    for i, name in enumerate(names.keys()):
        vb, cb, all_kpts, kpts, labels = get_bs(name, gap=gap[name])
        # eigen, all_kpts, kpts, labels = get_bs(name, gap=gap[name])
        ax[i].set_axis_off()
        add_img_ax(ax[i], img_path / "3D" / "{0}.png".format(name))
        ax[i].text(x=-0.07, y=1, s="abcd"[i], weight="bold", size="x-large",
                   transform=ax[i].transAxes)
        ax[i].text(x=0.5, y=1.01, s=names[name],
                   ha="center", va="baseline",
                   transform=ax[i].transAxes)
        ax_ = ax[i + 4]
        ax_.plot(all_kpts, vb, color=color[name], linewidth=1.0)
        ax_.plot(all_kpts, cb, color=color[name], linewidth=1.0)
        for x_ in kpts:
            ax_.axvline(x=x_, color="k", alpha=0.2, linewidth=1.0)
        # ax_.plot(all_kpts, eigen)
        ax_.set_ylim(-5, 7)
        if i > 0:
            ax_.set_yticks([])
        else:
            ax_.set_ylabel("Energy (eV)", size="medium")
        ax_.text(x=0.50, y=5.5 / 12, s="$E_{\\mathrm{g}}$ = " \
                 + "{0:.1f} eV".format(gap[name]),
                 # size="small",
                 transform=ax_.transAxes)
        ax_.set_xticks(kpts)
        ax_.axhline(y=0, ls="--", color="#ACACAC", linewidth=1.0)
        ax_.set_xticklabels(list(labels))
        
        # ax_.set_xlabel(names[name])

    # grid_labels(fig, ax[:4], reserved_space=[0, 0],
                # offsets=[(0.07, 0),
                         # (0.05, 0),
                         # (0.03, 0),
                         # (0.01, 0)])
            
    savepgf(fig, img_path / "bs.pgf", preview=True)
            


if __name__ == '__main__':
    plot_main()

