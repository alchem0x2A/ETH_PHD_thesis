# Helper functions for making plots
from pathlib import Path
import matplotlib as mpl
from matplotlib.pyplot import figure
from matplotlib.gridspec import GridSpec as GS
import numpy as np
import string
from .tex_helper import _get_textwidth, _get_preamble
from .path_helper import _module_path
from .img_helper import _get_pdf_size, _get_img_size, _get_img_type
from shutil import copy
import re

mpl_file = _module_path / "matplotlibrc"
# Update the rcParams
mpl.rcdefaults()
mpl.rc_file(mpl_file.as_posix())
mpl.use("pgf")

# Load the preamble
preamble = _get_preamble()
print(preamble)
if len(preamble) > 0:
    mpl.rcParams["pgf.preamble"] = preamble



_textwidth = _get_textwidth()
if _textwidth is None:
    _textwidth = 455.2            # Default values

def relsize(r=1.0, width=_textwidth, ratio=1.618, ppi=72):
    """Relative figure size to \\textwidth"""
    # print("Textwidth is: ", width)
    width_inch = r * width / ppi
    height_inch = r * width / ratio / ppi
    return width_inch, height_inch

def gridplots(nrows=1, ncols=1, span=[],
              r=1.0, ratio=1.618, ppi=72,
              width=None, **args):
    """Modified plt.subplots routine to allow combined grid cells
       For each grid, two digits are provided (row_span, col_span)
       Instead of 2D array, gridplots returns a plain list
    """
    if width is None:           # use automatic detection
        width = _textwidth
    figsize = relsize(r, width, ratio, ppi)
    print(figsize)
    fig = figure(figsize=figsize)
    print("Until here!")
    if (span is None) or (span == []) \
       or (ncols == 1 and nrows == 1):
        axs = fig.subplots(nrows, ncols, **args)
        try:
            return fig, axs.flatten()
        except AttributeError:
            return fig, axs     # 1x1 plot
    else:
        try:
            gs_kw = args["gridspec_kw"]
        except KeyError:
            gs_kw = dict()
        gs = GS(nrows, ncols, figure=fig, **gs_kw)        # Better way?
        # print(gs)
        axs = []
        r = 0
        c = 0            # counter for row and col
        for r, c, rs, cs in span:
            axs.append(fig.add_subplot(gs[r: r + rs, c: c + cs]))
        axs = np.array(axs)
        return fig, axs


def grid_labels(fig, axes, offsets=[],
                type="latin",
                decorate="{0}",
                reserved_space=(-0.03, 0.02),  # reserved space for label
                **args):
    """Put the grid labels in a controlled position. 
       Guess the position
    """
    dummy_ax = fig.add_subplot(111)
    dummy_ax.set_axis_off()

    def label_(i, key):
        k = key[0]
        allowed_keys = ["l", "L",
                        "r", "R",
                        "n"]
        if k not in allowed_keys:
            k = "l"
        if k == "l":
            return string.ascii_lowercase[i]
        elif k == "L":
            return string.ascii_uppercase[i]
        elif k == "r":
            raise NotImplementedError
        elif k == "R":
            raise NotImplementedError
        else:                   # n
            return i + 1
    
    gs = axes[0].get_gridspec()
    nr, nc = gs.get_geometry()
    # Determine the spacing each column and row
    hr = gs.get_height_ratios()
    if hr is None:
        h_spacings = np.ones(nr) / nr
    else:
        h_spacings = np.array(hr) / nr

    wr = gs.get_width_ratios()
    if wr is None:
        w_spacings = np.ones(nc) / nc
    else:
        w_spacings = np.array(wr) / nc

    # Add some sacrifice labels to ensure enough space
    try:
        dx, dy = reserved_space
        for i, ax in enumerate(axes):
            ax.text(x=dx, y=1 + dy,
                    s=decorate.format(label_(i, type)),
                    transform=ax.transAxes,
                    alpha=0
            )  # Fully transparent?
    except ValueError:
        pass

    labels = []
    for i, ax in enumerate(axes):
        nr, nc, ri, rf, ci, cf = ax.get_subplotspec().get_rows_columns()
        print(i, ri, rf, ci, cf)
        left = np.sum(w_spacings[: ci])
        top = 1 - np.sum(h_spacings[: ri])
        print(left, top)
        try:
            w_offset, h_offset = offsets[i]
            left += w_offset
            top += h_offset
        except (TypeError, IndexError):
            pass

        defaults = dict(size="x-large", weight="bold",
                        ha="left", va="top")
        defaults.update(**args)
        t = dummy_ax.text(x=left, y=top,
                      s=decorate.format(label_(i, type)),
                      transform=fig.transFigure,
                      **defaults)
        labels.append(t)
    return labels

def _replace_pgf_cmd(string):
    pattern = r"\\pgfimage"
    new_pat = r"\\includegraphics"
    cont = re.sub(pattern, new_pat, string)
    return cont

def _replace_pgf_img(string, index, new):
    pattern = r"\\pgftext\[.+?\]\{\\includegraphics\[.+?\]\{(.+?img" \
               + "{:d}".format(index) + r".png)\}\}"
    match = re.findall(pattern, string)
    if len(match) > 0:
        cont = re.sub(match[0], new, string)
    return cont

def add_img_ax(ax, fname, index=None, debug=False):
    """Higher level function to add a image with fname by default,
    register the information of fname with the ax

    """
    fname = Path(fname)
    f_type = _get_img_type(fname)
    if f_type == "pdf":
        w, h = _get_pdf_size(fname)
    else:
        w, h = _get_img_size(fname)
    if debug:
        print(w, h)
    # w, h are in inches
    # Create a "fake" image
    if not debug:
        ax.set_axis_off()
    ax.plot([0, w, 0, w], [0, h, h, 0], rasterized=True)
    ax.set_aspect("equal")
    # ax.axis("equal")
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.is_img = True
    ax.img_kw = dict(filename=fname,
                     index=index)
    return True

def savepgf(fig, out_file, preview=True):
    """Save fig into pgf file and replace the 
    \\pdfimage part
    if preview is True, a pdf version is output at the same directory
    """
    out_file = Path(out_file)

    suffix = out_file.suffix
    if suffix == ".pgf":        # already pgf
        pgf_f = out_file
        pdf_f = out_file.with_suffix(".pdf")
    elif suffix == "":          # name root
        pgf_f = out_file.with_suffix(".pgf")
        pdf_f = out_file.with_suffix(".pdf")
    else:
        raise NameError("Should either provide a filename or ending with .pgf!")

    if preview:
        fig.savefig(pdf_f)

    fig.savefig(pgf_f)
    with open(pgf_f, "r") as f:
        pgf_content = f.read()
    if len(pgf_content) == 0:
        raise FileExistsError("File problem?")
    pgf_content = _replace_pgf_cmd(pgf_content)

    # PGF images
    img_dir = pgf_f.parent
    imgs = []
    for ax in fig.get_axes():
        if hasattr(ax, "is_img"):
            if getattr(ax, "is_img"):
                index = ax.img_kw["index"]
                filename = ax.img_kw["filename"]
                imgs.append([ax, filename, index])

    # reorder index
    size = len(imgs)
    if size > 0:
        for i in range(size):
            ax, filename, index = imgs[i]
            if index is None:   # try to use automatic numbering
                index = i
            new_name = filename.name
            copy(filename, img_dir / new_name)
            pgf_content = _replace_pgf_img(pgf_content, index, new_name)
    
    with open(pgf_f, "w") as f:
        f.write(pgf_content)
        return

# Helper function to get color from certain colormap
def get_color(c, min=0, max=12, levels=100, name="rainbow"):
    # Helper function to plot colors
    # default colormap is rainbow 
    dc = (max - min) / levels
    i = int((c - min) / dc)
    cm = mpl.cm.get_cmap(name)
    color = cm(np.linspace(0, 1, levels + 1))
    return color[i]

def add_cbar(fig, ax=None, min=0, max=12, name="rainbow", **karg):
    # Ax ??
    # Helper function to add colorbar to fig and ax
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    # plot the colorbar outside
    # sc = ax.scatter([xlim[0] - 10, xlim[0] - 10],
                    # [ylim[0] - 10, ylim[0] - 10],
                    # c=[n_min, n_max],  # automated vmin vmax
                    # cmap=name)
    cmap = mpl.cm.get_cmap(name)
    sm = mpl.cm.ScalarMappable(cmap=cmap,
                               norm=mpl.colors.Normalize(vmin=min, vmax=max))
    cb = fig.colorbar(sm, **karg)
    # cb.set_ticks([1, 5, 10, 15]) 
    return cb
