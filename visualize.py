from ase.io import read
import pandas as pd

atoms = read("cmon_xtb.extxyz")  # or index=":" for multiple frames

df = pd.DataFrame({
    "element": atoms.get_chemical_symbols(),
    "x": atoms.positions[:, 0],
    "y": atoms.positions[:, 1],
    "z": atoms.positions[:, 2],
    "Z": atoms.get_atomic_numbers()
})

print(df)