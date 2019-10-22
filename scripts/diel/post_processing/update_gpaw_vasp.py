import ase, ase.db
import numpy
from scipy.constants import pi, e, epsilon_0

db = ase.db.connect("../../data/2D-bulk/bulk.db")

csv_file = "../../data/gpaw_data/gpaw_vasp_aa.csv"

with open(csv_file, "r", encoding="utf-8") as f:
    while True:
        line = f.readline()
        if len(line) == 0:
            break
        sys, gap, L, ex, ey, ez = line.strip().split(",")
        if any(len(s) == 0 for s in [ex, ey, ez]):  # bad results, discard
            continue
        try:
            gap = float(gap); L = float(L)
            ex = float(ex); ey = float(ey); ez = float(ez)
        except ValueError:
            continue
        formula, proto = sys.split("-")
        # print(sys.encode("utf8"), gap, L, ex, ey, ez)
        res = list(db.select(formula=formula, prototype=proto))
        
        if len(res) == 0:
            continue
        mol = res[0]
        db_id = mol.id
        db.update(db_id, bulk_L_vasp=L,
                  bulk_gap_vasp=gap,
                  bulk_eps_x_vasp=ex,
                  bulk_eps_y_vasp=ey,
                  bulk_eps_z_vasp=ez)
        print(sys, "Suscessful!")

