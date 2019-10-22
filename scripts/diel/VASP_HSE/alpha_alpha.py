import ase.db
import warnings
import numpy
import matplotlib.pyplot as plt
from ase.data import covalent_radii
from scipy.stats import linregress
from scipy.optimize import curve_fit
import os, os.path
from scipy.constants import pi, epsilon_0
from gpaw_data import get_data
import scipy
import csv

"""
Extract the alpha from the HSE xlsx files
"""

def get_2D3D():
    aniso_data = "../../data/other_dimension/2D3D.npz"
    if not os.path.exists(aniso_data):  # then need to create
        db_file = "../../data/gpaw_data/c2db.db"
        bulk_file = "../../data/2D-bulk/bulk.db"
        if not os.path.exists(db_file):
            raise FileExistsError(("Please download the c2db data into ../../data/gpaw_data/ folder,"
                           "from https://cmr.fysik.dtu.dk/_downloads/c2db.db"))

        def get_bulk(name, proto, id=None, method="gpaw"):
            # Get bulk properties
            if id is None:
                res = list(db.select(formula=name, prototype=proto))
                if len(res) == 0:
                    return None
                r = res[0]
            else:
                r = db.get(id)    
            try:
                if method.lower() == "gpaw":
                    L = r.bulk_L
                    eps_para = (r.bulk_eps_x + r.bulk_eps_y) / 2
                    eps_perp = r.bulk_eps_z
                    e = r.gap_hse
                # VASP version below:
                elif method.lower() == "vasp":
                    L = r.bulk_L_vasp
                    eps_para = (r.bulk_eps_x_vasp + r.bulk_eps_y_vasp) / 2
                    eps_perp = r.bulk_eps_z_vasp
                    if r.bulk_gap_vasp < 0:
                        r = r.gap_hse
                    else:
                        r = r.bulk_gap_vasep
                else:
                    return None
                if eps_para < 0 or eps_perp < 0:
                    return None
            except Exception:
                return None
            return L, eps_para, eps_perp, e

        db = ase.db.connect(db_file)
        bulk_db = ase.db.connect(bulk_file)

        materials = []
        eps_x = []
        eps_z = []
        alpha_x = []
        alpha_z = []
        Eg_HSE = []
        thick = []

        def get_thick(atom_row):
            pos = atom_row.positions[:, -1]
            diff = covalent_radii[atom_row.numbers]
            zmax = numpy.max(pos + diff) - numpy.min(pos - diff)
            return zmax

        reader = csv.reader(open("../../data/HSE-data/2D_HSE.csv", encoding="utf8"))
        next(reader)                    # skip line1
        for row in reader:
            if row[4] != "":
                name, proto = row[: 2]
                print(name, proto)
                L, E, ex, ey, ez, *_ = map(float, row[2:])
                if ez < ex:
                    eps_z.append(ez)
                    materials.append("-".join((name, proto)))
                    e_xy = numpy.sqrt(ex * ey)
                    ax = (e_xy - 1) / (4 * pi) * L
                    az = (1 - 1/ez) * L / (4 * pi)
                    ax = max(1 / 1.2, ax)
                    eps_x.append(e_xy); eps_z.append(ez)
                    alpha_x.append(ax); alpha_z.append(az)
                    Eg_HSE.append(E)
                    mol = list(db.select(formula=name, prototype=proto))[0]
                    thick.append(get_thick(mol))

        print(len(alpha_x))
        alpha_x = numpy.array(alpha_x)
        alpha_z = numpy.array(alpha_z)
        Eg_HSE = numpy.array(Eg_HSE)
        thick = numpy.array(thick)

        cond = numpy.where(Eg_HSE > 0.6)
        Eg_HSE = Eg_HSE[cond]
        alpha_x = alpha_x[cond]
        alpha_z = alpha_z[cond]
        thick = thick[cond]

        gp_data = get_data()

        import relation_2D3D as bulk

        # cnt_eg = numpy.sqrt(cnt_r * B / (cnt_x - A))
        Eg_2D = numpy.append(gp_data[2], Eg_HSE)
        Eg_3D = numpy.append(bulk.Eg_gpaw, bulk.Eg_HSE)
        eta_2D = numpy.append(gp_data[1] / gp_data[0],  alpha_z / alpha_x)
        print(len(eta_2D))
        eta_3D = numpy.append(numpy.min([bulk.eps_z_gpaw[:, 0] / bulk.eps_x_gpaw[:, 0],
                                            bulk.eps_x_gpaw[:, 0] / bulk.eps_z_gpaw[:, 0]],
                                           axis=0),
                              numpy.min([bulk.eps_z_3D[:, 0] / bulk.eps_x_3D[:, 0],
                                           bulk.eps_x_3D[:, 0] / bulk.eps_z_3D[:, 0]],
                                          axis=0), )
        numpy.savez(aniso_data,
                    **{"Eg_2D": Eg_2D, "Eg_3D": Eg_3D,
                     "eta_2D": eta_2D, "eta_3D": eta_3D})
    else:
        d = numpy.load(aniso_data)
        Eg_2D = d["Eg_2D"]; Eg_3D = d["Eg_3D"]
        eta_2D = d["eta_2D"]; eta_3D = d["eta_3D"]
    return Eg_2D, Eg_3D, eta_2D, eta_3D



plt.style.use("science")
fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(111)

Eg_2D, Eg_3D, eta_2D, eta_3D = get_2D3D()
ax.scatter(Eg_2D, eta_2D, marker="^", alpha=0.1, s=40,
           linewidth=0)
ax.scatter(Eg_3D, eta_3D, marker="s", alpha=0.1, s=40,
           linewidth=0)

# LinearSVM classification
from sklearn.svm import LinearSVC
svc = LinearSVC(C=1,
                max_iter=100000,
                class_weight={1:1, 2:3})
class_feature = numpy.vstack([list(zip(Eg_2D, eta_2D)), list(zip(Eg_3D, eta_3D))])
print(class_feature.shape)
class_tag = numpy.append(1 * numpy.ones_like(Eg_2D), 2 * numpy.ones_like(Eg_3D))
svc.fit(class_feature, class_tag)
xx, yy = numpy.meshgrid(numpy.linspace(0, 8, 30), numpy.linspace(0, 1, 30))
xy = numpy.vstack([xx.ravel(), yy.ravel()]).T
zz = svc.decision_function(xy).reshape(xx.shape)
ax.contour(xx, yy, zz, levels=[0])

def anisotropy(data):
    a_max = numpy.max(data, axis=1)
    a_min = numpy.min(data, axis=1)
    return a_min / a_max

def anis_from_file(file_name):
    data = numpy.genfromtxt(file_name, delimiter=",",
                            comments="#")  # csv ending
    Eg = data[:, 1]
    anis = anisotropy(data[:, 2:5])
    return Eg, anis


marks = {"CNT": "o",
         "polyacene": "p",
         "MPc": "<",
         "covalent": ">",
         "fullerene": "*",
         "polyacetylene": "D"}
for f in ["CNT", "covalent", "polyacene", "MPc", "fullerene", "polyacetylene"]:
    f_name = "../../data/other_dimension/{}.csv".format(f)
    Eg, anis = anis_from_file(f_name)
    ax.scatter(Eg, anis, label=f,
               marker=marks[f], s=40,
               alpha=0.6,
               linewidth=0)

xx = yy = numpy.linspace(0, 8, 100)
ax.plot(xx, numpy.ones_like(xx), "--")

# ax.set_title("$y={0:.4f}x+{1:.4f},\ R^2={2:.4f}$".format(res.slope, res.intercept, res.rvalue))
ax.set_xlabel("$E_{\mathrm{g}}$")
ax.set_ylabel("Dielectric Anisotropy")
# ax.set_ylabel("$\\alpha_{zz}/(4\\pi \\varepsilon_0)$ ($\\AA$)")
ax.set_xlim(0, 8)
ax.set_ylim(0, 1.05)
ax.legend()
# ax.set_xticks([1,3,5,7])
fig.tight_layout()
fig.savefig(os.path.join("../../tmp_img/", "alpha_alpha.svg"))
