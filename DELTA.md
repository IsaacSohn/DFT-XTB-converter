# Delta Dataset Pipeline (delta.py)

This module adds a third stage to our dataset pipeline:
- xTB → fast approximate labels
- DFT → high-accuracy labels
- Δ-learning (delta) → correction between them

## What is Delta Learning?
Instead of training directly on DFT (expensive), you train a model to learn:
```
ΔE = E_xtb - E_dft
```
Then during inference:
```
E_corrected = E_xtb - ΔE_model
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
delta = xtb_energy - dft_energy
```
- Writes a new dataset with:
    - ```method="delta"```
    - ```TotEnergy = delta```
    - same geometry
    - zero forces (by default)
    - atomic numbers (Z)

## Input Requirements

Your input file must contain paired structures:
```
3
TotEnergy=-76.42189300 method="xtb" ...
O ...
H ...
H ...

3
TotEnergy=-76.43651200 method="dft-qe" ...
O ...
H ...
H ...
```
### Important:
- Same atom order
- Same geometry (within rounding tolerance)
- Both xtb and dft must exist for each structure

## Usage

Basic usage:
```
python delta.py -x cmon_xtb.extxyz -d cmon.extxyz -o delta.extxyz
```