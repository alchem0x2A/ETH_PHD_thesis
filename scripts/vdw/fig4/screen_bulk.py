import numpy
# import os, os.path
# import matplotlib.pyplot as plt
# from numpy import meshgrid
from ..utils.kkr import kkr, matsubara_freq
from ..utils.lifshitz import alpha_to_eps, g_amb_alpha, g_amb, transparency_single
from ..utils.eps_tools import data_bulk, data_2D, get_alpha, get_eps, get_index
from . import data_path, img_path
# from os.path import join, exists, dirname, abspath
# plt.style.use("science")

# MPI version of the function, use gpaw-python for instance

# curdir = os.path.abspath(os.path.dirname(__file__))
# img_path = join(curdir, "../../img", "fig4")
# if not exists(img_path):
    # os.makedirs(img_path)

try:
    from mpi4py import MPI
    world = MPI.COMM_WORLD
    rank = world.rank
    size = world.size
except ImportError:
    world = None
    rank = 0
    size = 1

def get_transparency(mater, d=2.0e-9, force=False):
    ind_m = get_index(mater, kind="2D")
    print(ind_m)
    alpha_m, freq_alpha, *_ = get_alpha(ind_m)
    # file_name = os.path.join(img_path, "eta_bulk_{}-{}.npy".format(*mater))
    file_name = data_path / "eta_bulk_{}-{}.npy".format(*mater)
    if file_name.exists() and (force is not True):
        return numpy.load(file_name, allow_pickle=True)
    else:
        results_local = []
        results = []
        count = 0
        for i in range(len(data_bulk)):
        # for i in range(5):
            for j in range(i + 1):
                count += 1
                if (count % size) == rank:
                    print(i, j, rank)
                    eps_a, freq_matsu_a, *gap_a = get_eps(i)
                    eps_b, freq_matsu_b, *gap_b = get_eps(j)
                    assert ((freq_matsu_a - freq_matsu_b).max() < 1e-3)
                    eta = transparency_single(eps_a, alpha_m, eps_b,
                                              freq_matsu_a, freq_alpha, d)
                    results_local.append((*gap_a, *gap_b, eta))
                    if i != j:
                        results_local.append((*gap_b, *gap_a, eta))
        world.barrier()
        if rank != 0:
            world.send(results_local, dest=0, tag=rank)
        else:
            results = results_local
            for i in range(size - 1):
                res_tmp = world.recv()
                results += res_tmp
        world.barrier()
        if rank == 0:
            results = numpy.array(results)
            numpy.save(file_name, results)

        
        # world.Bcast(results, root=0)
        world.barrier()
        return numpy.load(file_name, allow_pickle=True)

# def plot_eta(mater, d=2.0e-9):
    # results = get_transparency(mater, d)
    # if rank == 0:
        # Eg_a = results[:, 0]        # min
        # Eg_b = results[:, 2]        # min_len
        # eta = results[:, -1]
        # cond = numpy.where((Eg_a > 0) & (Eg_b > 0))
        # fig = plt.figure(figsize=(3.5, 3.5))
        # ax = fig.add_subplot(111)
        # if get_index(mater) == 29:
            # eta += 0.1
        # if get_index(mater) == 3:
            # eta -= 0.08
        # print(eta.min(), eta.max())
        # cmap = ax.scatter(Eg_a[cond], Eg_b[cond],
                           # c=eta[cond],
                           # s=20, cmap="rainbow",
                           # marker="s",
                           # linewidths=0,
                           # alpha=0.3,
                           # vmax=0.85,
                           # vmin=-0.1,
                           # rasterized=True)
        # fig.colorbar(mappable=cmap, label="$\\eta$", shrink=0.5)
        # ax.set_xlim(0, 12)
        # ax.set_ylim(0, 12)
        # ax.set_xticks(range(0, 13, 2))
        # ax.set_yticks(range(0, 13, 2))
        # ax.set_xlabel("$E_{\\rm{g}}^{\\rm{A}}$ (eV)")
        # ax.set_ylabel("$E_{\\rm{g}}^{\\rm{B}}$ (eV)")
        # ax.set_aspect("equal")
        # ax.set_title("{}-{} @{} nm".format(*mater, d/1e-9))
        # fig.tight_layout()
        # fig.savefig(join(img_path,
                         # "eta_bulk_{}-{}.svg".format(*mater)))
        # print("{} - {} finished!".format(*mater))
    # world.barrier()
    
    
def main(**kargs):
    for mater in [("C2", "C"),
                  ("BN", "BN"),
                  ("MoS2", "MoS2")]:
        get_transparency(mater=mater, **kargs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run main function")
    parser.add_argument("-f", "--force", dest="force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
