from helper import data_root
import os
data_path = data_root / "intro/"
if not data_path.exists():
    os.makedirs(data_path)
