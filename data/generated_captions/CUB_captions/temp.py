from pathlib import Path 
import numpy as np 

root_dir = Path("~/NSERC/data/scanpaths/cub_scanpaths/").expanduser()

total = length = 0
for npy_path in root_dir.rglob("*.npy"): 
    arr = np.load(npy_path, allow_pickle=True).item()
    length += len(arr['X'])
    total += 1

print(f"Average length of the scanpaths is {length / total}")