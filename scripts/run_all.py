from pathlib import Path
from importlib import import_module
from helper import clean_width_cache
from subprocess import call
# from multiprocessing import Pool, cpu_count
try:
    has_mpi = True
    from mpi4py import rank, size, world
except ImportError:
    has_mpi = False
    rank = 0
    size = 1

root = Path(".")

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

def run_plot(mod_name, func=func):
    """Single function to run the plot_main inside module m"""
    try:
        m = import_module(mod_name)
    except ImportError:
        print(TColors.FAIL + "Module {0} import error!".format(mod_name) \
              + TColors.ENDC)
        return False
    if hasattr(m, func):
        try:
            getattr(m, func)()
            print(TColors.OKGREEN + "Module {0} plot finish!".format(mod_name) \
                  + TColors.ENDC)
            return True
        except Exception as e:
            print(TColors.FAIL + "Module {0} plot failed with {1}!".format(mod_name, e) \
                  + TColors.ENDC)
            return False
        
def main():
    jobs = []
    for f in root.rglob("*.py"):
        # parents starting from root
        parents = list(map(lambda x: x.name, f.parents))[-1::-1]
        script_name = f.with_suffix("").name
        if (len(parents) > 1) and \
           (all([p not in exclude_mods for p in parents[1:]])) and \
           (script_name not in exclude_files):
            # import the module and test if plot_main is inside
            mod_name = ".".join(parents[1:] + [script_name])
            jobs.append(mod_name)
            # print(mod_name)
            # try:
            #     m = import_module(mod_name)
            # except ImportError:
            #     print(TColors.FAIL + "Module {0} import error!".format(mod_name) \
            #           + TColors.ENDC)
            #     continue
            # if hasattr(m, func):
                # try:
                #     getattr(m, func)()
                #     print("Module {0} plot finish!".format(mod_name))
                # except Exception as e:
                #     print("Module {0} plot failed with {1}!".format(mod_name, e))
    if has_mpi:
        world.barrier()
    for i, mod_ in enumerate(jobs):
        if i % size == rank:
            run_plot(mod_)
            
    if rank == 0:    
        clean_width_cache()
        
    if has_mpi:
        world.barrier()

    # Clean up width

if __name__ == '__main__':
    main()
