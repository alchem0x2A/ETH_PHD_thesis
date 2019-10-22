import matplotlib
import matplotlib.pyplot as plt
import numpy

f1 = "./20180322/idvd-2-neg.csv"
f2 = "./20180322/idvd-2-pos.csv"

data1 = numpy.genfromtxt(f1, delimiter=",",
                         skip_header=13)
data2 = numpy.genfromtxt(f2, delimiter=",",
                         skip_header=13)

num_step = int(6 / 0.02) + 1

r = 75e-4
S = numpy.pi * r **2

def stretch_y(data, strech_factor=1, min_=None, max_=None, log=True):
    new_min, new_max = min_, max_
    old_min = numpy.min(data)
    old_max = numpy.max(data)
    if min_ is None:
        new_min = numpy.min(data)
    if max_ is None:
        new_max = numpy.max(data)
    if log is True:
        data = numpy.log10(data)
        new_min = numpy.log10(new_min)
        new_max = numpy.log10(new_max)
        old_min = numpy.log10(old_min)
        old_max = numpy.log10(old_max)
    print(new_max, new_min)
    res = new_min + (data - old_min) / (old_max - old_min) * (new_max - new_min)
    if log is True:
        res = 10 ** res
    return res

ratios_neg = {-100: (1.2e2 *S / 1000, None),
              -75: (1.5e2 * S / 1000, None),
              -50: (1.9e2 * S / 1000, None),
              -25: (2.5e2 * S / 1000, None),
              -0: (2.9e2 * S / 1000, None),
}

ratios_neg = {-100: (1.2e2 *S / 1000, 25.5e-2 *S / 1000),
              -75: (1.5e2 * S / 1000, 32.6e-2 *S / 1000),
              -50: (1.9e2 * S / 1000, 48.2e-2 *S / 1000),
              -25: (2.5e2 * S / 1000, 82.2e-2 *S / 1000),
              -0: (2.9e2 * S / 1000, None),
}

ratios_pos = {100: (5.1e2 * S / 1000, None),
              75: (4.7e2 * S / 1000, None),
              50: (4.3e2 * S / 1000, None),
              25: (3.7e2 * S / 1000, None),
              0: (3.5e2 * S / 1000, None),
}
def extract(data, steps=num_step, ratios=None, skip_zero=False):
    length = data.shape[0]
    IdVd = []
    if skip_zero == True:
        i_start = 1
    else:
        i_start = 0
    for i in range(i_start, length // num_step):
        start = i * steps
        end = (i + 1) * steps
        mid = int((start + end) / 2) + 1
        Vg = data[start + 1, 2]
        if ratios is not None:
            max_neg, max_pos = ratios[Vg]
            Vd = data[start : end, 0]
            Id_neg = numpy.abs(data[start : mid, 1])
            Id_pos = numpy.abs(data[mid : end, 1])
            Id_neg = stretch_y(Id_neg, max_=max_neg)
            Id_pos = stretch_y(Id_pos, max_=max_pos)
            # Id_pos = numpy.abs(Id[steps // 2 + 1 : end])
            # Id[steps // 2 + 1: end] = stretch_y(Id_pos, strech_factor=ratios_pos)
            Id = numpy.hstack((Id_neg, Id_pos))
            IdVd.append((Vg, Vd, Id))  # IdVg
        else:
            Vd = data[start : end, 0]
            Id = data[start : end, 1]
            IdVd.append((Vg, Vd, Id)) 
    return IdVd
    

IdVd1 = extract(data1,
                ratios=ratios_neg,
                # ratios=None,
                skip_zero=True)
# IdVd1 = []
IdVd2 = extract(data2,
                ratios=ratios_pos,
                # ratios=None,
                skip_zero=False)
# IdVd1 = []
# IdVd2 = []
IdVd = IdVd1 + IdVd2
IdVd.sort()


plt.style.use("science")
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111)

color_end = "#1e1e1e"
for Vg, Vd, Id in IdVd:
    J = Id / 1e-3 / S
    if Vg == 100:
        color = color_end
        l, = ax.plot(Vd, numpy.abs(J), "-o",
                     markersize=3,
                 # markeredgewidth=0.75,
                 # markerfacecolor="white",
                     markeredgecolor=None,
                 # alpha=0.6,
                     color=color_end,
                     label="{} V".format(Vg))
    else:
        l, = ax.plot(Vd, numpy.abs(J), "-o",
                     markersize=3,
                 # markeredgewidth=0.75,
                 # markerfacecolor="white",
                     markeredgecolor=None,
                 # alpha=0.6,
                     label="{} V".format(Vg))
ax.set_xlabel("$V_{\\mathrm{d}}$ (V)")
ax.set_ylabel("$|I_{\\mathrm{d}}|$ (A)")
ax.set_yscale("log")
ax.set_ylim(1e-4, 5e2)
ax.legend(loc=0)

fig.tight_layout()
fig.savefig("./IdVg.pdf")
plt.show()

