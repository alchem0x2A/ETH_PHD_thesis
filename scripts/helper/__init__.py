from . import mpl_helper       # Load mpl setting as well
from .mpl_helper import gridplots, grid_labels
from .mpl_helper import savepgf, add_img_ax
from .path_helper import root_path, script_path
from .path_helper import tex_path, build_path
from .tex_helper import clean_width_cache

# Some other path stuff
data_root = root_path / "data/"
img_root = root_path / "img/"
