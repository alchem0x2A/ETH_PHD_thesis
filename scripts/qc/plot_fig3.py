import numpy as np
from . import data_path, img_path
from .constants import Const
from . import equations as eqs
from scipy.optimize import fsolve
from scipy.interpolate import BivariateSpline
from helper import gridplots, grid_labels, savepgf
from helper import get_color, add_cbar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Plot the figures for fig 3


def plot_main():
    fig, ax = gridplots(1, 1, r=0.55,
                        ratio=1)
    dim = 256
    NQ_gate = np.linspace(-2, 2, dim) * 1e13
    Q_gate = NQ_gate * 1e4 * Const.q
    psi_b = np.linspace(-Const.E_g / 2, Const.E_g / 2, dim)
    NDs = eqs.func_np(0, psi_b)[0]

    data_eta = data_path / "fig23_eta_{0:d}.npy".format(dim)
    if data_eta.exists():
        eta = np.load(data_eta)
        print("Loaded data from {0}".format(data_eta.as_posix()))
    else:
        eta = np.empty([dim, dim])
        for i in range(dim):
            for j in range(dim):
                psi0 = fsolve(lambda Psi: eqs.solve_psi_s(Psi, psi_b[i], Q_gate[j], 0), 0)[0]
                B = Const.h_bar * Const.v_F / Const.q * (np.pi / Const.q) ** 0.5
                E_s = eqs.func_E_psi(psi0, psi_b[i])
                q_gr = eqs.func_q_g(Q_gate[j], E_s)
                nps = eqs.func_np(psi0, psi_b[i])
                nps0 = eqs.func_np(0, psi_b[i])
                rho = (nps[1] - nps0[1]) - (nps[0] - nps0[0])
                eta[i, j] = 1 / (1 - 2 * (np.abs(q_gr)) ** 0.5 \
                                 / (B) * E_s / (Const.q * rho))
        np.save(data_eta, eta)
    
    xx, yy = np.meshgrid(NQ_gate / 1e13, np.log10(NDs / 1e6))
    cm = ax.imshow(np.log10(eta)[::-1, :],
                   extent=[xx.min(), xx.max(),
                           yy.min(), yy.max()],
                   aspect="auto",
                   cmap="rainbow",
                   interpolation="bicubic")
    ax.set_xlabel("$\\sigma_{\\mathrm{M}}\ (10^{13}\ e\\cdot{}\\mathrm{cm}^{-2})$")
    ax.set_ylabel("$n_0$ (cm$^{-3}$)")
    cax_outside = inset_axes(ax, height="70%", width="50%",
                             bbox_to_anchor=(1.01, 0.05, 0.12, 0.80),
                             bbox_transform=ax.transAxes,
                             loc="lower left")
    cb = fig.colorbar(cm, cax_outside,
                      ticks=[0, -1, -2, -3])
    cb.ax.set_title("$\\eta_{\\mathrm{FE}}$", pad=7)
    cb.ax.set_yticklabels(["1", "$10^{-1}$",
                           "$10^{-2}$", "$10^{-3}$"])

    ax.axhline(y=np.log10(Const.n_i / 1e6),
               color="k",
               ls="--")
    ax.text(x=0.02, y=0.52,
            s="n-type",
            va="bottom",
            transform=ax.transAxes)
    ax.text(x=0.02, y=0.48,
            s="p-type",
            va="top",
            transform=ax.transAxes)
    
    

# xlabel('Q_{gate} (10^{13} e*cm^{-2})');
# % set(gca, 'XTick', [-2,  -1,  0,  1,  2]);
# ylabel('N_D (cm^3)');
# set(gca, 'YTick', [2, 6, 10, 14, 18]);
# set(gca, 'YTickLabel', {'10^{2}', '10^{6}', '10^{10}', '10^{14}', '10^{18}'});
# colormap(jet);
# set(f,'LineStyle','None');

# c=colorbar('eastoutside');
# c.Label.String = '\Delta Q_{semi}/\Delta Q_{gr}';
# caxis([-3.5,0]);
# set(c, 'FontSize', 16);
# c.Ticks = [-3,-2,-1,0];
# c.TickLabels = {'10^{-3}', '10^{-2}', '10^{-1}', '1'};
# set(c, 'TickDir', 'out');
# set(c, 'LineWidth', 2);


    
    savepgf(fig, img_path / "eta-fe.pgf", preview=True)
    

if __name__ == '__main__':
    plot_main()
