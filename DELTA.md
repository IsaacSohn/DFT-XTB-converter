# Delta Dataset Pipeline (delta.py)

This module adds a third stage to our dataset pipeline:
- xTB → fast approximate labels
- DFT → high-accuracy labels
- Δ-learning (delta) → correction between them

## What is Delta Learning?
Instead of training directly on DFT (expensive), you train a model to learn:
```
ΔE = E_dft - E_xtb
```
Then during inference:
```
E_corrected = E_xtb + ΔE_model
```
This gives DFT-level accuracy at near-xTB cost.

## What delta.py Does

### delta.py:
- Reads multi-frame extxyz
- Finds matching:
    - method="xtb"
    - method="dft" or method="dft-qe"
- Matches structures by:
    - species
    - positions (rounded)
    - PBC
- Computes:
```
delta = dft_energy - xtb_energy
```
- Writes a new dataset with:
    - ```method="delta"```
    - ```TotEnergy = delta```
    - same geometry
    - zero forces (by default)
    - atomic numbers (Z)

## Input Requirements

`delta.py` takes **two separate files** — one xTB file and one DFT file — with the same structures in the same order.

`h2o_xtb.extxyz`:
```
3
TotEnergy=-76.42189300 method="xtb" ...
O ...
H ...
H ...
```

`h2o_dft.extxyz`:
```
3
TotEnergy=-76.43651200 method="dft-qe" ...
O ...
H ...
H ...
```

### Important:
- Same atom order in both files
- Same geometry per frame (within rounding tolerance, default 6 decimals)
- Each structure must have a match in both files

## Usage

Basic usage:
```
python delta.py -x cmon_xtb.extxyz -d cmon.extxyz -o delta.extxyz
```