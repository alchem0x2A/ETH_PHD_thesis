# Handled several things about TeX output 

from pathlib import Path
from os import makedirs

def _ensure_dir(folder):
    # Make sure dir exists
    p = Path(folder)
    if not p.is_dir():
        makedirs(p)

# module of the helper module
_module_path = Path(__file__).parents[0]
script_path = _module_path.parent
root_path = script_path.parent

# Path for img
img_path = root_path / "img/"
tex_path = root_path
build_path = root_path / "build"

for p_ in [img_path, tex_path, build_path]:
    _ensure_dir(p_)
