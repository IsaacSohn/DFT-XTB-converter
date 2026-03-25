# DFT/xTB Dataset Pipeline (XYZ / extended XYZ)

This repo provides two Python programs:

## 1) convert_xyz_dataset.py

-   Reads XYZ / extended-XYZ / custom extxyz-like files
-   Normalizes them into a consistent extended XYZ (extxyz) format
-   Ensures each atom has: species, position, force, atomic number (Z)
-   Stamps metadata: method="dft" or method="xtb"

## 2) run_qm.py

-   Reads structures (XYZ/extxyz, multi-frame supported)
-   Runs:
    -   xTB (GFN2 for molecules, GFN1 for periodic systems)
    -   DFT via Quantum ESPRESSO (via ASE)
-   Outputs extxyz with TotEnergy and forces

------------------------------------------------------------------------

# General Pipeline (Recommended)

## Pipeline A: Fast xTB Dataset

### Step 1: Normalize

python convert_xyz.py -i h2o.xyz -o h2o_normalized.extxyz --method xtb

### Step 2: Run xTB

python run_qm.py -i h2o_normalized.extxyz -o h2o_xtb.extxyz --method xtb

Notes: - Nonperiodic → GFN2-xTB - Periodic → GFN1-xTB (automatic) - If
convergence issues: --xtb-iterations 1000 --xtb-etemp 1000

------------------------------------------------------------------------

## Pipeline B: DFT Labeling (QE)

python run_qm.py -i h2o_normalized.extxyz -o h2o_dft.extxyz\
--method qe\
--qe-pseudo-dir ./pseudos\
--qe-pseudo-map '{"O":"O.UPF","H":"H.UPF"}'\
--qe-kpts "1,1,1"\
--qe-ecutwfc 50\
--qe-ecutrho 400

------------------------------------------------------------------------

## Pipeline C: DFT → xTB (Cheap Relabeling / Distillation)

Use when: - You already have DFT structures - You want fast approximate
labels - You want large datasets cheaply

### Step 1: Use existing DFT extxyz

(No conversion needed)

### Step 2: Run xTB on same structures

python run_qm.py -i dft_dataset.extxyz -o xtb_dataset.extxyz\
--method xtb\
--xtb-iterations 1000\
--xtb-etemp 1000

### What this does

-   Keeps identical geometry
-   Recomputes energy + forces with xTB
-   Much faster than DFT
-   Useful for:
    -   pretraining
    -   dataset expansion
    -   teacher-student setups

### Important Notes

-   Periodic systems automatically use GFN1
-   Molecular systems use GFN2
-   xTB != DFT accuracy (approximation)

------------------------------------------------------------------------

# Summary

  Pipeline   Speed   Accuracy   Use Case
  ---------- ------- ---------- ------------------------------
  xTB        Fast    Medium     Prototyping / large datasets
  DFT        Slow    High       Ground truth
  DFT→xTB    Fast    Medium     Scaling datasets cheaply
