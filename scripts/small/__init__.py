from helper import data_root, img_root
import os
base = "small/"
data_path = data_root / base
img_path = img_root / base

for p in [data_path, img_path]:
    if not p.exists():
        os.makedirs(p)
