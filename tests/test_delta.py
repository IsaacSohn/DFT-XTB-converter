from ase.io import read
import numpy as np
from ase.io import read

def test_read_forces():
    atoms = read("cmon_xtb.extxyz", index=0)

    print(atoms.arrays["force"])

def test_average_forces():

    frames = read("delta.extxyz", index=":")

    all_magnitudes = []

    for atoms in frames:
        forces = atoms.arrays["force"]  # this is delta_F
        mags = np.linalg.norm(forces, axis=1)  # |F| per atom
        all_magnitudes.extend(mags)

    all_magnitudes = np.array(all_magnitudes)

    print("Average |delta F|:", all_magnitudes.mean(), "eV/Å")
    print("Median  |delta F|:", np.median(all_magnitudes), "eV/Å")
    print("Max     |delta F|:", all_magnitudes.max(), "eV/Å")


if __name__ == "__main__":
    test_average_forces()