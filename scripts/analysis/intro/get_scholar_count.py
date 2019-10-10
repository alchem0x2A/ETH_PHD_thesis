# Simple scripts to get the total counts of publications per keyword series
# Need to run several times due to the query limit by google
from requests import get
import re
import time
import random
import numpy as np
from . import data_path


_HOST = "https://scholar.google.ch/scholar?"
results = data_path / "scholar_result.npz"

def struct_string(kw):
    """Construct query string"""
    strings = []
    assert "q" in kw
    q = kw["q"].replace(" ", "+")
    q = q.replace("\"", "%22")
    strings.append("q={0}".format(q))
        
    strings.append("hl=en")
    if "start" in kw.keys():
        strings.append("as_ylo={:d}".format(kw["start"]))
        if "end" in kw.keys():
            strings.append("as_yhi={:d}".format(kw["end"]))
    return "&".join(strings)

def get_count(kw):
    """Get counts of publications per kw"""
    time.sleep(3 + random.uniform(0, 5))
    url = _HOST + struct_string(kw)
    r = get(url)
    # print(url)
    # print(r.text)
    pattern = "About\s+?(.+?)\s+?results"
    match = re.findall(pattern, r.text)
    try:
        s = int("".join(match[0].split("&#8217;")))
        return s
    except IndexError:
        return 0

def main(start=1980, end=2020, spacing=3):
    years = np.arange(start, end, spacing)
    queries = dict(graphene="\"graphene\"",
                   hbn="\"hexagonal boron nitride\" OR \"boron nitride\" -\"cubic\"",
                   mos2="\"molybdenum disulfide\" OR \"MoS2\"",
                   p="\"phosphorene\" OR \"black phosphorus\"")
    if results.is_file():
        data_ = np.load(results, allow_pickle=True)
        data = {k: data_[k] for k in data_.files}
    else:
        data = dict()
    for k, q in queries.items():
        if k in data.keys():
            _, counts = data[k]
        else:
            counts = np.ones_like(years[1:]) * -1
        for i, y in enumerate(years[1:]):
            if counts[i] <= 0:
                kw = dict(q=q + " AND \"2D\"", start=y - 3, end=y)
                print("Now getting count for {} in year {}".format(k, y))
                cnt = get_count(kw)
                print(cnt)
                counts[i] = cnt
        data[k] = [years[1:], counts]

    np.savez(results, **data)

if __name__ == "__main__":
    main()
