# DFT/xTB Dataset Pipeline (XYZ / extended XYZ)

This repo provides four Python programs for building quantum chemistry datasets:

1. `convert_xyz.py` — Normalizes raw XYZ/extxyz files into a consistent extended-XYZ format
2. `run_qm.py` — Runs xTB or Quantum ESPRESSO (DFT) on structures and writes energies + forces
3. `delta.py` — Computes delta (correction) energies between xTB and DFT for delta-learning
4. `comparisons.py` — Evaluates ML model predictions against ground-truth DFT labels

---

## Prerequisites

```bash
pip install ase numpy
pip install xtb          # required for --method xtb (run_qm.py)
```

For DFT (`--method qe`), you also need [Quantum ESPRESSO](https://www.quantum-espresso.org/) installed with `pw.x` on your PATH, plus pseudopotential UPF files.

Python 3.8+ is required.

---

## Example scenario

Say you have a plain XYZ file `h2o.xyz` with no energy or force information:

```
3
water molecule
O  0.000000  0.000000  0.000000
H  0.757000  0.586000  0.000000
H -0.757000  0.586000  0.000000
```

The pipelines below walk through how to turn this into a labeled dataset.

---

## Pipeline A: Build an xTB dataset (fast, no supercomputer needed)

### Step 1 — Normalize your raw structure file

```bash
python convert_xyz.py -i h2o.xyz -o h2o_normalized.extxyz --method xtb
```

Verify the output looks like this (no physics yet — just formatting):

```
3
TotEnergy=nan cutoff=-1.00000000 nneightol=1.20000000 method="xtb" pbc="F F F" Properties=species:S:1:pos:R:3:force:R:3:Z:I:1
O   0.00000000  0.00000000  0.00000000  0.00000000  0.00000000  0.00000000  8
H   0.75700000  0.58600000  0.00000000  0.00000000  0.00000000  0.00000000  1
H  -0.75700000  0.58600000  0.00000000  0.00000000  0.00000000  0.00000000  1
```

### Step 2 — Run xTB to compute energy + forces

```bash
python run_qm.py -i h2o_normalized.extxyz -o h2o_xtb.extxyz --method xtb
```

Output will look like:

```
3
TotEnergy=-76.42189300 cutoff=-1.00000000 nneightol=1.20000000 method="xtb" pbc="F F F" Properties=species:S:1:pos:R:3:force:R:3:Z:I:1
O   0.00000000  0.00000000  0.00000000  -0.01234500  0.00123400  0.00098700  8
H   0.75700000  0.58600000  0.00000000   0.00678900 -0.00234500 -0.00043200  1
H  -0.75700000  0.58600000  0.00000000   0.00555600  0.00111100 -0.00055500  1
```

**Periodic systems** (e.g., crystals): pass `--default-pbc "T T T"` in Step 1 and `--xtb-iterations 1000 --xtb-etemp 1000` in Step 2. GFN1-xTB is used automatically for periodic structures because GFN2 fails under PBC.

---

## Pipeline B: Build a DFT dataset (Quantum ESPRESSO, needs a supercomputer)

### Step 1 — Normalize with method="dft"

```bash
python convert_xyz.py -i h2o.xyz -o h2o_normalized.extxyz --method dft
```

### Step 2 — Run Quantum ESPRESSO

```bash
python run_qm.py \
  -i h2o_normalized.extxyz \
  -o h2o_dft.extxyz \
  --method qe \
  --qe-pseudo-dir ./pseudos \
  --qe-pseudo-map '{"O":"O.UPF","H":"H.UPF"}' \
  --qe-kpts "1,1,1" \
  --qe-ecutwfc 50 \
  --qe-ecutrho 400
```

Output format:

```
3
TotEnergy=-76.43651200 cutoff=-1.00000000 nneightol=1.20000000 method="dft-qe" pbc="F F F" ...
O  ...
H  ...
H  ...
```

---

## Pipeline C: DFT → xTB distillation (cheap relabeling)

Use when you already have DFT structures and want fast approximate labels for pretraining or dataset expansion.

```bash
python run_qm.py \
  -i dft_dataset.extxyz \
  -o xtb_dataset.extxyz \
  --method xtb \
  --xtb-iterations 1000 \
  --xtb-etemp 1000
```

This keeps the DFT geometry and recomputes energy + forces with xTB.

---

## Pipeline D: Delta learning dataset

Delta learning trains a model to predict the correction `ΔE = E_dft - E_xtb`. At inference time you run cheap xTB and add the model's correction to get near-DFT accuracy.

### Step 1 — You need both files from the same structures

Run Pipelines A and B (or C) on the same input to get:
- `h2o_xtb.extxyz` — xTB energies and forces
- `h2o_dft.extxyz` — DFT energies and forces

### Step 2 — Compute delta energies

```bash
python delta.py \
  -x h2o_xtb.extxyz \
  -d h2o_dft.extxyz \
  -o h2o_delta.extxyz
```

Output has `method="delta"` and `TotEnergy = E_dft - E_xtb` (default convention). Both source energies are preserved as `xtb_energy` and `dft_energy` in the header.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--energy-mode` | `dft-minus-xtb` | Sign convention for ΔE |
| `--force-mode` | `delta` | `delta`, `zeros`, `keep-dft`, or `keep-xtb` |
| `--decimals` | `6` | Rounding precision for geometry matching |
| `--strict-count` | off | Raise error if frame counts differ |

---

## Evaluating ML model predictions

`comparisons.py` compares prediction files against a ground-truth DFT reference. Edit the file paths at the top of the script:

```python
COMPARE_TO_FILE = Path("comparisons/dft_to_compare_to.xyz")  # ground truth
PREDICTION_FILES = [
    Path("comparisons/xtb_predictions.xyz"),
    Path("comparisons/dft_predictions.xyz"),
    Path("comparisons/delta_predictions.xyz"),
]
```

Then run:

```bash
python comparisons.py
```

It trims the ground-truth file to the last 10% of frames to match prediction file size, then reports per-file energy and force errors (MAE, RMSE, MaxAE).

---

## Extended XYZ format reference

Every output file uses this schema per frame:

```
N
TotEnergy=<eV> cutoff=<Ry> nneightol=<float> method="<name>" pbc="T/F T/F T/F" [Lattice="<9 floats>"] Properties=species:S:1:pos:R:3:force:R:3:Z:I:1
<sym>  <x>  <y>  <z>  <fx>  <fy>  <fz>  <Z>
...
```

- Positions and forces in Ångström / eV·Å⁻¹
- `TotEnergy=nan` means the field was missing in the source file (unfilled by computation)
- `Lattice` is only written for periodic structures (9 floats, row-major)

---

## convert_xyz.py flags

| Flag | Required | Description |
|------|----------|-------------|
| `-i` / `--input` | yes | Input XYZ or extxyz file |
| `-o` / `--output` | yes | Output extxyz file |
| `--method` | yes | `dft` or `xtb` |
| `--default-pbc` | no | PBC string, e.g. `"T T T"` (default `"F F F"`) |
| `--default-cutoff` | no | Cutoff value if missing (default `-1.0`) |

## run_qm.py flags

| Flag | Required | Description |
|------|----------|-------------|
| `-i` / `--input` | yes | Input extxyz (multi-frame OK) |
| `-o` / `--output` | yes | Output extxyz |
| `--method` | yes | `xtb` or `qe` |
| `--optimize` | no | Run BFGS geometry optimization |
| `--charge` | no | Total charge for xTB (default 0) |
| `--uhf` | no | Unpaired electrons for xTB (default 0) |
| `--xtb-gfn` | no | GFN level: 0, 1, or 2 (auto-selected if omitted) |
| `--xtb-iterations` | no | Max SCC iterations |
| `--xtb-etemp` | no | Electronic temperature in K |
| `--qe-pseudo-dir` | qe only | Directory containing UPF files |
| `--qe-pseudo-map` | qe only | JSON mapping element → UPF filename |
| `--qe-kpts` | no | k-point mesh like `"2,2,2"` (default `"1,1,1"`) |
| `--qe-ecutwfc` | no | Plane-wave cutoff in Ry (default 50) |
| `--qe-ecutrho` | no | Charge density cutoff in Ry (default 400) |
