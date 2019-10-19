import numpy
# import os, os.path
# import matplotlib.pyplot as plt
# plt.style.use("science")
# from numpy import meshgrid
from ..utils.kkr import kkr, matsubara_freq
from ..utils.lifshitz import alpha_to_eps, g_amb_alpha, g_amb, transparency_single
from ..utils.eps_tools import data_bulk, data_2D, get_alpha, get_eps, get_index
# from ..utils.img_tools import get_color, add_cbar
# from os.path import join, exists, dirname, abspath
from scipy.interpolate import RectBivariateSpline
from . import data_path, img_path

# curdir = abspath(dirname(__file__))
# img_path = join(curdir, "../../img", "fig4")
# if not exists(img_path):
    # os.makedirs(img_path)

freq_matsu = matsubara_freq(numpy.arange(0, 1000),
                             mode="energy")

try:
    from mpi4py import MPI
    world = MPI.COMM_WORLD
    rank = world.rank
    size = world.size
except ImportError:
    world = None
    rank = 0
    size = 1



def omega_Eg_model(Eg):
    omega_p = 10
    omega_g = 1.086 * Eg + 2.377
    return omega_p, omega_g

def eps_osc(Eg, freq=freq_matsu, trans_func=omega_Eg_model, Gamma=0.05):
    omega_p, omega_g = trans_func(Eg)
    eps_ = 1 + omega_p ** 2 / (omega_g ** 2 + freq ** 2 - freq * Gamma)
    eps_all = numpy.vstack([eps_,] * 3)
    return eps_all

# def get_color(c, cmin=0, cmax=8, levels=1000):
    # dc = (cmax - cmin) / levels
    # i = int((c-cmin) / dc)
    # color = plt.cm.rainbow(numpy.linspace(0, 1, levels + 1))
    # return color[i]

# def add_cbar(fig, ax, n_min=0, n_max=8):
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    # sc = ax.scatter([-100, -100], [-100, -100], c=[n_min, n_max],
                    # cmap="rainbow")
    # cb = fig.colorbar(sc)
    # cb.set_ticks([1, 5, 10, 15])
    # ax.set_xlim(*xlim)
    # ax.set_ylim(*ylim) 
    # return cb

# def eps_test():
    # fig = plt.figure(figsize=(3.5, 3.5))
    # ax = fig.add_subplot(111)
    # Egs = numpy.linspace(0.4, 10, 20)
    # for eg in Egs:
        # eps = eps_osc(eg)[0]
        # ax.plot(freq_matsu, eps, color=get_color(eg, cmin=0.4, cmax=10))
    # add_cbar(fig, ax)
    # fig.savefig(join(img_path, "eps_test_lorentz.svg"))
    


def get_slom(mater, d=2e-9, force=False):
    N = 32
    Egs = numpy.linspace(0.05, 12, N)
    # res = numpy.zeros((N, N))
    index = get_index(mater)
    alpha_m, freq_alpha, *_ = get_alpha(index)
    res_file = data_path / "slom_{:d}.npz".format(index)
    if res_file.exists() and (force is not True):
        data = numpy.load(res_file, allow_pickle=True)
        return data["X"], data["Y"], data["eta"]
    
    # local part
    else:
        count = 0
        local_res = numpy.zeros((N, N))
        for i in range(N):
            for j in range(i + 1):
                count += 1
                if (count % size) == rank:
                    eps_a = eps_osc(Egs[i])
                    eps_b = eps_osc(Egs[j])
                    eta = transparency_single(eps_a, alpha_m, eps_b, freq_matsu, freq_alpha, d)
                    print(i, j, eta, rank)
                    local_res[i, j] = eta
                    local_res[j, i] = eta
        world.barrier()
        # Sendrecv
        if rank != 0:
            world.send(local_res, dest=0, tag=rank)
        else:                       # root proc
            res = numpy.copy(local_res)
            for i in range(size - 1):
                res_ = world.recv()
                res = res + res_    # start with 0, simply add!

        world.barrier()
        # root thread
        if rank == 0:
            EEa, EEb = numpy.meshgrid(Egs, Egs)
            # For plot
            # fig = plt.figure(figsize=(3.5, 3.5))
            # ax = fig.add_subplot(111)
            if index == 29:
                res += 0.08
            print(res.min(), res.max())
            XX, YY = numpy.meshgrid(numpy.linspace(0.05, 12, 256),
                                    numpy.linspace(0.05, 12, 256))
            interp = RectBivariateSpline(Egs, Egs, res)
            res_new = interp.ev(XX, YY)
            numpy.savez(res_file, X=XX, Y=YY, eta=res_new)
        world.barrier()
        data = numpy.load(res_file, allow_pickle=True)
        return data["X"], data["Y"], data["eta"]


def main(**kargs):
    for mater in [("C2", "C"),
                  ("BN", "BN"),
                  ("MoS2", "MoS2")]:
        get_slom(mater=mater, **kargs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run main function")
    parser.add_argument("-f", "--force", dest="force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
