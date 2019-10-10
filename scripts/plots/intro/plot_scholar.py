import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from helper import gridplots, grid_labels, savepgf
from . import data_path, img_path

results = data_path / "scholar_result.npz"

display = dict(graphene="Graphene",
               hbn="hBN",
               mos2="MoS$_{2}$",
               p="Phosphorene / BP")
def plot_main():
    fig, ax = gridplots(1, 1,
                        r=0.95)

    try:
        data = np.load(results, allow_pickle=True)
    except FileNotFoundError:
        print("The npz file does not exist.\n Please run the extract script at analysis.intri.get_scholar_count First!")
        return False

    for k, n in display.items():
        print(data[k])
        year, count = data[k]
        if len(count) > len(year):
            count = count[:-1]  # Hardcore
        count[:-1] = count[:-1] / 3
        count[-1] = count[-1] / 2.7
        ax.plot(year, count / 1e4 , "-", label=n)
    ax.legend()
    ax.set_xlabel("Year")
    ax.set_ylabel("No. Publications / Year $\\times{}10^{4}$")
    ax.set_xlim(1990, 2019)
    l1 = ax.axvline(x=2004, ls="--", color="#757575")
    l2 = ax.axvline(x=2010, ls="--", color="#757575")
    ax.text(x=2003.8, y=2, s="Discovery of graphene →", ha="right",
            # size="small"
    )
    t= ax.text(x=2009.8, y=3, s="Nobel prize for graphene →", ha="right",
            # size="small"
    )

    # ax.set_yscale("log")
    savepgf(fig, img_path / "scholar.pgf", preview=True)


if __name__ == '__main__':
    plot_main()

