# DFT/xTB Dataset Pipeline (XYZ / extended XYZ)

This repo provides two Python programs:

1) ```convert_xyz_dataset.py```
   - Reads **XYZ / extended-XYZ / custom extxyz-like** files
   - Normalizes them into a consistent **extended XYZ (extxyz)** format
   - Ensures each atom has: **species, position, force, atomic number (Z)**
   - Stamps `method="dft"` or `method="xtb"` metadata

2) ```run_qm.py```
   - Reads structures (XYZ/extxyz, multi-frame supported)
   - Runs **xTB** (GFN2-xTB via ASE) or **DFT via Quantum ESPRESSO** (via ASE)
   - Outputs **extxyz** with **TotEnergy** and **forces** per atom in a consistent schema

## General Pipeline (Recommended)

### An example scenario
1) We have a database for a molecule called ```h2o.xyz```
   - this is just info about h2o but really small. like its just 3 lines because in this perfect world, you don't need 12937162837619827368 data points to make an accurate model of anything related to machine learning
   ```bash
   3
   water molecule
   O  0.000000  0.000000  0.000000
   H  0.757000  0.586000  0.000000
   H -0.757000  0.586000  0.000000
   ```
2) With this dataset, we would naturally want to use xtb or dft to generate a force field in order for us to feed it to mace.
3) We have 2 pipelines you can go through in order to achieve this


### Pipeline A: Build an xTB dataset fast, then optionally add DFT labels later
This approach is basically if you have structures in mind, and you just want xtb calculations of them for testing a model(or if you don't have a supercomputer COUGH COUGH)
1) **Normalize your raw structure file**
   - If you have mixed formats (plain xyz, partially labeled extxyz, etc.), normalize first:
   ```bash
   python convert_xyz.py -i h2o.xyz -o h2o_normalized.extxyz --method xtb
   ```
2) **Confirm your data looks something like this(Still no physics computed — just formatting.)**
```bash
3
TotEnergy=nan cutoff=-1.00000000 nneightol=1.20000000 method="xtb" pbc="F F F" Properties=species:S:1:pos:R:3:force:R:3:Z:I:1
O  0.00000000  0.00000000  0.00000000  0.00000000  0.00000000  0.00000000  8
H  0.75700000  0.58600000  0.00000000  0.00000000  0.00000000  0.00000000  1
H -0.75700000  0.58600000  0.00000000  0.00000000  0.00000000  0.00000000  1
```

3) **Run xTB to generate energy + forces**
   ```bash
   python run_qm.py -i h2o_normalized.extxyz -o h2o_xtb.extxyz --method xtb
   ```

   Output should look something like this
   ```bash
   3
   TotEnergy=-76.42189300 cutoff=-1.00000000 nneightol=1.20000000 method="xtb" pbc="F F F" Properties=species:S:1:pos:R:3:force:R:3:Z:I:1
   O  0.00000000  0.00000000  0.00000000  -0.01234500  0.00123400  0.00098700  8
   H  0.75700000  0.58600000  0.00000000   0.00678900 -0.00234500 -0.00043200  1
   H -0.75700000  0.58600000  0.00000000   0.00555600  0.00111100 -0.00055500  1
   ```

### Pipeline B: DFT labeling (Quantum ESPRESSO via ASE)
This approach is basically if you have structures in mind, and you just want dft calculations of them because you have a supercomputer or something
1) Use the same normalized structure BUT replace xtb with dft

   ```bash
   3
   TotEnergy=-76.42189300 cutoff=-1.00000000 nneightol=1.20000000 method="dft" pbc="F F F" Properties=species:S:1:pos:R:3:force:R:3:Z:I:1
   O  0.00000000  0.00000000  0.00000000  -0.01234500  0.00123400  0.00098700  8
   H  0.75700000  0.58600000  0.00000000   0.00678900 -0.00234500 -0.00043200  1
   H -0.75700000  0.58600000  0.00000000   0.00555600  0.00111100 -0.00055500  1
   ```


2) Run QE
   ```bash
   python run_qm.py -i h2o_normalized.extxyz -o h2o_dft.extxyz
      --method qe
      --qe-pseudo-dir ./pseudos
      --qe-pseudo-map '{"O":"O.UPF","H":"H.UPF"}'
      --qe-kpts "1,1,1"
      --qe-ecutwfc 50
      --qe-ecutrho 400
   ```
   Output would look something like this format
   ```bash
   3
   TotEnergy=-76.43651200 method="dft-qe" pbc="F F F" ...
   O ...
   H ...
   H ...
   ```