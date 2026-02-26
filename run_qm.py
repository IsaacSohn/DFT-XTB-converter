#!/usr/bin/env python3
"""
run_qm.py

Run xTB or DFT (Quantum ESPRESSO) on structures from XYZ/extxyz, then output extxyz with:
TotEnergy, pbc, Lattice (if any), Properties=species,pos,force,Z

Examples:

# xTB singlepoint:
python run_qm.py -i mols.xyz -o mols_xtb.extxyz --method xtb --charge 0 --uhf 0

# xTB geometry-optimized? (optional): you can add --optimize
python run_qm.py -i mols.xyz -o mols_xtb_opt.extxyz --method xtb --optimize

# QE singlepoint (DFT) example:
python run_qm.py -i cell.extxyz -o cell_dft.extxyz --method qe \
  --qe-pseudo-dir ./pseudos \
  --qe-pseudo-map '{"C":"C.pbe-n-kjpaw_psl.1.0.0.UPF","H":"H.pbe-kjpaw_psl.1.0.0.UPF","O":"O.pbe-n-kjpaw_psl.1.0.0.UPF"}' \
  --qe-kpts "2,2,2" \
  --qe-ecutwfc 50 --qe-ecutrho 400

Notes:
- QE needs periodic cell info for solids; for isolated molecules, set large cell + pbc="F F F" or do gamma-only with big vacuum.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List, Tuple, Optional

from ase.io import iread
from ase import Atoms

# ---------- utilities ----------
Z_MAP: Dict[str, int] = {
    "H": 1, "He": 2,
    "C": 6, "N": 7, "O": 8, "F": 9,
    "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53,
    "Si": 14, "B": 5,
    "Na": 11, "Mg": 12, "Al": 13,
    "K": 19, "Ca": 20,
    "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
}
def infer_Z(symbol: str) -> int:
    sym = symbol.strip()
    if len(sym) >= 2 and sym[1].islower():
        sym = sym[0].upper() + sym[1].lower()
    else:
        sym = sym[0].upper() + (sym[1:].lower() if len(sym) > 1 else "")
    if sym not in Z_MAP:
        # fallback: ASE knows numbers, so we only need this for formatting
        raise ValueError(f"Unknown element '{symbol}' for Z. Add it to Z_MAP in run_qm.py.")
    return Z_MAP[sym]

def cell_to_lattice_string(atoms: Atoms) -> Optional[str]:
    try:
        cell = atoms.get_cell()
        if cell is None:
            return None
        a = cell.array
        if a.shape != (3, 3):
            return None
        # consider "valid" if any cell vector is nonzero
        if float(abs(a).sum()) == 0.0:
            return None
        return " ".join(f"{a[i,j]:.8f}" for i in range(3) for j in range(3))
    except Exception:
        return None

def pbc_string(atoms: Atoms) -> str:
    try:
        return " ".join("T" if b else "F" for b in atoms.get_pbc())
    except Exception:
        return "F F F"

def format_extxyz(atoms: Atoms, energy: float, forces: List[Tuple[float,float,float]], method: str) -> str:
    symbols = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    n = len(symbols)
    lattice = cell_to_lattice_string(atoms)
    pbc = pbc_string(atoms)

    header_parts = []
    header_parts.append(f"TotEnergy={energy:.8f}")
    header_parts.append("cutoff=-1.00000000")
    header_parts.append("nneightol=1.20000000")
    header_parts.append(f'method="{method}"')
    header_parts.append(f'pbc="{pbc}"')
    if lattice:
        header_parts.append(f'Lattice="{lattice}"')
    header_parts.append("Properties=species:S:1:pos:R:3:force:R:3:Z:I:1")

    out = [str(n), " ".join(header_parts)]
    numbers = atoms.get_atomic_numbers()
    for sym, (x,y,z), (fx,fy,fz), Z in zip(symbols, pos, forces, numbers):
        out.append(f"{sym:<2s}  {x: .8f}  {y: .8f}  {z: .8f}  {fx: .8f}  {fy: .8f}  {fz: .8f}  {int(Z):d}")
    return "\n".join(out) + "\n"

# ---------- calculators ----------
def run_xtb(atoms: Atoms, charge: int, uhf: int, optimize: bool) -> Tuple[float, List[Tuple[float,float,float]]]:
    """
    xTB runner that works even when ase.calculators.xtb is missing.

    Strategy:
      1) Try ASE's native XTB calculator if available.
      2) Otherwise fall back to an external subprocess wrapper (Windows-safe encoding).
    """
    from ase.optimize import BFGS
    import os
    import subprocess
    import tempfile
    import numpy as np
    from ase.calculators.calculator import Calculator, all_changes

    # --- Try native ASE calculator first (if your ASE ever gains it later) ---
    try:
        from ase.calculators.xtb import XTB  # type: ignore
        atoms.calc = XTB(method="GFN2-xTB", charge=charge, uhf=uhf)
        if optimize:
            BFGS(atoms, logfile=None).run(fmax=0.05)
        e = float(atoms.get_potential_energy())
        forces = [tuple(map(float, row)) for row in atoms.get_forces()]
        return e, forces
    except ModuleNotFoundError:
        pass

    # --- Fallback: external xtb.exe wrapper ---
    class XTBExternal(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, xtb_command="xtb", charge=0, uhf=0, gfn=2, **kwargs):
            super().__init__(**kwargs)
            self.xtb_command = xtb_command
            self.charge = int(charge)
            self.uhf = int(uhf)
            self.gfn = int(gfn)

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)

            with tempfile.TemporaryDirectory() as d:
                xyz_path = os.path.join(d, "input.xyz")
                atoms.write(xyz_path)

                cmd = [
                    self.xtb_command,
                    xyz_path,
                    f"--gfn{self.gfn}",
                    "--chrg", str(self.charge),
                    "--uhf", str(self.uhf),
                    "--grad",
                ]

                p = subprocess.run(
                    cmd,
                    cwd=d,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                out = p.stdout
                if p.returncode != 0:
                    raise RuntimeError(f"xTB failed (code {p.returncode}). Output:\n{out}")

                # Parse TOTAL ENERGY (Eh)
                energy_eh = None
                for line in out.splitlines():
                    if "TOTAL ENERGY" in line and "Eh" in line:
                        toks = line.replace("=", " ").split()
                        for i, t in enumerate(toks):
                            if t == "Eh" and i > 0:
                                try:
                                    energy_eh = float(toks[i - 1])
                                except ValueError:
                                    pass
                if energy_eh is None:
                    raise RuntimeError("Could not parse TOTAL ENERGY from xTB output.")

                # Parse gradient file (Eh/Bohr)
                grad_path = os.path.join(d, "gradient")
                if not os.path.exists(grad_path):
                    raise RuntimeError("xTB did not produce 'gradient' file (expected with --grad).")

                floats = []
                with open(grad_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        for tok in line.split():
                            try:
                                floats.append(float(tok))
                            except ValueError:
                                continue

                n = len(atoms)
                if len(floats) < 3 * n:
                    raise RuntimeError("Could not parse enough gradient floats from gradient file.")

                grad = np.array(floats[-3 * n :], dtype=float).reshape((n, 3))

                # Units: Eh -> eV, Eh/Bohr -> eV/Å
                EH_TO_EV = 27.211386245988
                BOHR_TO_ANG = 0.529177210903
                energy_ev = energy_eh * EH_TO_EV
                forces_ev_per_ang = -(grad * EH_TO_EV / BOHR_TO_ANG)

                self.results["energy"] = float(energy_ev)
                self.results["forces"] = forces_ev_per_ang

    atoms.calc = XTBExternal(charge=charge, uhf=uhf, gfn=2)

    if optimize:
        BFGS(atoms, logfile=None).run(fmax=0.05)

    e = float(atoms.get_potential_energy())
    forces = [tuple(map(float, row)) for row in atoms.get_forces()]
    return e, forces

def run_qe(atoms: Atoms,
           pseudo_dir: str,
           pseudo_map: Dict[str, str],
           kpts: Tuple[int,int,int],
           ecutwfc: float,
           ecutrho: float,
           optimize: bool) -> Tuple[float, List[Tuple[float,float,float]]]:
    """
    QE via ASE. Requires Quantum ESPRESSO installed (pw.x).
    """
    from ase.calculators.espresso import Espresso, EspressoProfile
    from ase.optimize import BFGS

    # Build input_data
    input_data = {
        "control": {
            "calculation": "scf",
            "tstress": True,
            "tprnfor": True,
            "verbosity": "high",
        },
        "system": {
            "ecutwfc": float(ecutwfc),
            "ecutrho": float(ecutrho),
            # smearing settings can be added for metals
        },
        "electrons": {
            "conv_thr": 1.0e-8,
        },
    }

    # If optimization, switch calculation and let ASE optimize positions
    if optimize:
        input_data["control"]["calculation"] = "scf"  # ASE optimizer calls forces each step

    pseudopotentials = {el: pseudo_map[el] for el in set(atoms.get_chemical_symbols())}

    profile = EspressoProfile(command="pw.x")
    calc = Espresso(
        profile=profile,
        pseudopotentials=pseudopotentials,
        pseudo_dir=pseudo_dir,
        input_data=input_data,
        kpts=kpts
    )
    atoms.calc = calc

    if optimize:
        dyn = BFGS(atoms, logfile=None)
        dyn.run(fmax=0.05)

    e = float(atoms.get_potential_energy())
    f = atoms.get_forces()
    forces = [tuple(map(float, row)) for row in f]
    return e, forces

# ---------- main ----------
def parse_kpts(s: str) -> Tuple[int,int,int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError('kpts must look like "2,2,2"')
    return (int(parts[0]), int(parts[1]), int(parts[2]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input xyz/extxyz (can be multi-frame)")
    ap.add_argument("-o", "--output", required=True, help="Output extxyz with energies+forces")
    ap.add_argument("--method", required=True, choices=["xtb", "qe"], help="Which method to run")
    ap.add_argument("--optimize", action="store_true", help="Run a simple geometry optimization (BFGS)")

    # xTB params
    ap.add_argument("--charge", type=int, default=0, help="Total charge (xTB)")
    ap.add_argument("--uhf", type=int, default=0, help="Number of unpaired electrons (xTB)")

    # QE params
    ap.add_argument("--qe-pseudo-dir", default=None, help="Directory containing UPF pseudopotentials")
    ap.add_argument("--qe-pseudo-map", default=None,
                    help='JSON dict mapping element->UPF filename, e.g. \'{"C":"C.UPF","H":"H.UPF"}\'')
    ap.add_argument("--qe-kpts", default="1,1,1", help='k-point mesh like "2,2,2"')
    ap.add_argument("--qe-ecutwfc", type=float, default=50.0, help="Plane-wave cutoff (Ry)")
    ap.add_argument("--qe-ecutrho", type=float, default=400.0, help="Charge density cutoff (Ry)")

    args = ap.parse_args()

    frames = list(iread(args.input, index=":"))
    if not frames:
        raise RuntimeError(f"No structures read from {args.input}")

    out_chunks: List[str] = []
    for idx, atoms in enumerate(frames):
        # Ensure we have positions; pbc/cell can be whatever input provides.
        if args.method == "xtb":
            energy, forces = run_xtb(atoms, charge=args.charge, uhf=args.uhf, optimize=args.optimize)
            out_chunks.append(format_extxyz(atoms, energy, forces, method="xtb"))
        else:
            if not args.qe_pseudo_dir or not args.qe_pseudo_map:
                raise ValueError("--qe-pseudo-dir and --qe-pseudo-map are required for --method qe")
            pseudo_map = json.loads(args.qe_pseudo_map)
            kpts = parse_kpts(args.qe_kpts)
            energy, forces = run_qe(
                atoms,
                pseudo_dir=args.qe_pseudo_dir,
                pseudo_map=pseudo_map,
                kpts=kpts,
                ecutwfc=args.qe_ecutwfc,
                ecutrho=args.qe_ecutrho,
                optimize=args.optimize,
            )
            out_chunks.append(format_extxyz(atoms, energy, forces, method="dft-qe"))

        print(f"[{idx+1}/{len(frames)}] done: E={energy:.8f} eV-ish (ASE units depend on backend)")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("".join(out_chunks))

    print(f"Wrote {len(frames)} frame(s) -> {args.output}")

if __name__ == "__main__":
    main()