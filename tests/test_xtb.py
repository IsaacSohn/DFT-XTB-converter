import numpy as np
from ase.io import read

dft = read("cmon.xyz", index=":")

all_mags = []
for atoms in dft:
    mags = np.linalg.norm(atoms.arrays["force"], axis=1)
    all_mags.extend(mags)

print("Avg DFT force:", np.mean(all_mags))