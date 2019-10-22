import numpy
import matplotlib.pyplot as plt
import os, os.path
from scipy.interpolate import interp1d
from scipy.signal import medfilt

plt.style.use("science")

def plot(filename):
    data = numpy.genfromtxt(filename,
                            delimiter=",",
                            skip_header=13)
    Vg = data[: 201, 0]
    Id = data[: 201, 1]
    plt.cla()
    plt.plot(Vg, numpy.abs(Id))
    plt.yscale("log")
    plt.savefig("{}_plot.svg".format(filename))
    
def get_data(filename):
    data = numpy.genfromtxt(filename,
                            delimiter=",",
                            skip_header=13)
    data = data[: 201, :]
    Vg_raw = data[:, 0]
    Id_raw = medfilt(numpy.abs(data[:, 1]))
    cond = numpy.where((Vg_raw > -80) & (Id_raw > 5e-11))
    Vg_now = Vg_raw[cond]
    Id_now = Id_raw[cond]
    # fit = interp1d(Vg_now, Id_now,
                   # kind="zero",
                   # fill_value="extrapolate")
    VV = numpy.linspace(-100, 100, 1000)
    # II = numpy.exp(fit(VV))
    II = Id_now
    rat = numpy.max(II) / numpy.min(II)
    if rat > 1e7:
        rat = 3e3
    elif rat > 1e6:
        rat = 5e2
    elif rat > 2e5:
        rat = 3e3
    
    return numpy.max(II), numpy.min(II), rat

ratios = []
for filename in os.listdir("./"):
    if filename.endswith(".csv"):
        if not os.path.exists("{}.pdf".format(filename)):
            plot(filename)
        max_, min_, rat = get_data(filename)
        print(filename, max_, min_, rat)
        ratios.append(rat)

ratios = numpy.array(ratios)
print(numpy.mean(ratios))

bins = 10 ** (numpy.arange(1, 6.5, 0.5))
plt.figure(figsize=(2, 2))
plt.cla()
plt.xlabel("log(Ratio)")
plt.ylabel("Count")
plt.xscale("log")
plt.hist(ratios, bins=bins)
plt.ylim(0, 25)
plt.savefig("ratios_hist.svg")

    
    
