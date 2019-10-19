from . import data_path, img_path
from helper import grid_labels, gridplots, add_img_ax, savepgf
from .get_repulsion import get_energy, get_energy_two_body
from ..utils.eps_tools import get_index


maters_m = [("C2", "C"),
            ("BN", "BN")]

disp_names = { "BN": "hBN",
               "C2": "Gr"}

def plot_main():
    # plot figure!
    fig, ax = gridplots(2, 2, ratio=1.1, r=0.85)
    # First part the water
    ind_a = get_index("H2O-exp", "bulk")
    for m in maters_m:
        ind_m = get_index(m, "2D")
        name = disp_names[m[0]]
        dd, Phi_amb = get_energy(ind_m, ind_a)
        dd, Phi_mb = get_energy_two_body(ind_m, ind_a)
        Phi_tot = Phi_amb + Phi_mb
        l, = ax[1].plot(dd / 1e-10, Phi_amb * 1000, "-^", label="{}".format(name))
        ax[1].plot(dd / 1e-10, Phi_tot * 1000, label="{} Total".format(name), color=l.get_c())
        ax[1].plot(dd / 1e-10, Phi_mb * 1000, "-v", label="{} Two-body only".format(name), color=l.get_c(), alpha=0.8)

    ind_a = get_index("Au-exp", "bulk")
    for m in maters_m:
        ind_m = get_index(m, "2D")
        name = disp_names[m[0]]
        dd, Phi_amb = get_energy(ind_m, ind_a)
        dd, Phi_mb = get_energy_two_body(ind_m, ind_a)
        Phi_tot = Phi_amb + Phi_mb
        l, = ax[3].plot(dd / 1e-10, Phi_amb * 1000, "-^", label="{}".format(name))
        ax[3].plot(dd / 1e-10, Phi_tot * 1000, label="{} Total".format(name), color=l.get_c())
        ax[3].plot(dd / 1e-10, Phi_mb * 1000, "-v", label="{} Two-body only".format(name),
                   color=l.get_c(), alpha=0.8)
    for ax_ in ax:
        ax_.axhline(y=0, ls="--", color="grey")
    for ax_ in ax[:2]:
        ax_.set_xticklabels([])
    for ax_ in ax[2:]:
        ax_.set_xlabel("$d$ (Å)")

    for ax_ in [ax[0], ax[2]]:
        ax_.set_ylabel(r"$\Phi^{\mathrm{vdW}}$ (mJ·cm$^{-2}$)")
    # zoom in!
    ax[1].set_ylim(-1, 0.1)
    ax[3].set_ylim(-1, 0.6)
    grid_labels(fig, ax, reserved_space=(0, 0))
    savepgf(fig, img_path / "suspend_energy.pgf")

if __name__ == "__main__":
    plot_main()
