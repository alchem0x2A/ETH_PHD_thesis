from pathlib import Path
from importlib import import_module
from helper import clean_width_cache, img_root
from subprocess import call
from multiprocessing import Pool, cpu_count, current_process
import warnings


# Customize this to exclude the submodules
exclude_mods = ["helper", "test"]
exclude_files = ["__init__", ]
func="plot_main"


# Minimal class for terminal ANSI color output
class TColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def convert_(infile):
    # Convert svg to pdf
    infile = Path(infile)
    assert infile.suffix == ".svg"
    # pdf
    base_name = infile.name
    dirname = infile.parent
    outfile = infile.with_suffix(".pdf")
    print(outfile)
    program = "inkscape"
    params = ["--without-gui", "--export-area-page", ]
    io = ["--file={}".format(infile),
          "--export-dpi=300",
          "--export-pdf={}".format(outfile),
          "--export-latex"]
    success = call([program, *params, *io])
    
    if success != 0:
        warnings.warn(TColors.FAIL + "File {} cannot be converted!".format(infile) + TColors.ENDC)
    else:
        print(TColors.OKGREEN +
              "File {} converted successfully on thread {}.".format(infile,
                                                                    current_process()) \
              +TColors.ENDC)

        
def main():
    jobs = []
    for f in img_root.rglob("*.svg"):
        # parents starting from root
        name = f.name
        if name not in exclude_files:
            jobs.append(f)
    
    ncores = cpu_count()
    with Pool(ncores) as p:
        p.map(convert_, jobs)
    # Clean up width

if __name__ == '__main__':
    main()
