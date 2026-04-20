#!/usr/bin/env python3
"""
run_qm.py

Run xTB or DFT (Quantum ESPRESSO) on structures from XYZ/extxyz, then output extxyz with:
TotEnergy, pbc, Lattice (if any), Properties=species,pos,force,Z

Examples:

# xTB singlepoint:
python run_qm.py -i mols.xyz -o mols_xtb.extxyz --method xtb --charge 0 --uhf 0

# xTB geometry optimization (optional):
python run_qm.py -i mols.xyz -o mols_xtb_opt.extxyz --method xtb --optimize

# xTB periodic singlepoint (defaults to GFN1 for periodic):
python run_qm.py -i cell.extxyz -o cell_xtb.extxyz --method xtb --xtb-iterations 1000 --xtb-etemp 1000

# QE singlepoint (DFT) example:
python run_qm.py -i cell.extxyz -o cell_dft.extxyz --method qe \
  --qe-pseudo-dir ./pseudos \
  --qe-pseudo-map '{"C":"C.pbe-n-kjpaw_psl.1.0.0.UPF","H":"H.pbe-kjpaw_psl.1.0.0.UPF","O":"O.pbe-n-kjpaw_psl.1.0.0.UPF"}' \
  --qe-kpts "2,2,2" \
  --qe-ecutwfc 50 --qe-ecutrho 400

Notes:
- QE needs periodic cell info for solids.
- The xTB fallback wrapper supports:
    * nonperiodic systems via temporary XYZ
    * periodic systems via temporary Turbomole-style coord file with $periodic/$lattice
- For periodic systems, this script defaults to GFN1-xTB because GFN2-xTB under PBC
  can fail with "Multipoles not available with PBC".
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple, Optional

from ase.io import iread
from ase import Atoms


def cell_to_lattice_string(atoms: Atoms) -> Optional[str]:
    try:
        cell = atoms.get_cell()
        if cell is None:
            return None
        a = cell.array
        if a.shape != (3, 3):
            return None
        if float(abs(a).sum()) == 0.0:
            return None
        return " ".join(f"{a[i, j]:.8f}" for i in range(3) for j in range(3))
    except Exception:
        return None


def xtb_periodic_dim(atoms: Atoms) -> int:
    try:
        pbc = list(bool(x) for x in atoms.get_pbc())
        return sum(pbc)
    except Exception:
        return 0


def write_xtb_coord(path: str, atoms: Atoms) -> None:
    """
    Write Turbomole-style coord file for xTB.
    Uses Angstrom for both coordinates and lattice.

    Nonperiodic:
      $coord angs
      ...
      $end

    Periodic:
      $coord angs
      ...
      $periodic N
      $lattice angs
      ...
      $end
    """
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    cell = atoms.get_cell().array
    npbc = xtb_periodic_dim(atoms)

    with open(path, "w", encoding="utf-8") as f:
        f.write("$coord angs\n")
        for sym, (x, y, z) in zip(symbols, positions):
            f.write(f"  {x: .12f}  {y: .12f}  {z: .12f}  {sym.lower()}\n")

        if npbc > 0:
            f.write(f"$periodic {npbc}\n")
            f.write("$lattice angs\n")

            if npbc == 3:
                for i in range(3):
                    f.write(f"  {cell[i,0]: .12f}  {cell[i,1]: .12f}  {cell[i,2]: .12f}\n")
            elif npbc == 2:
                for i in range(2):
                    f.write(f"  {cell[i,0]: .12f}  {cell[i,1]: .12f}  {cell[i,2]: .12f}\n")
            elif npbc == 1:
                f.write(f"  {cell[0,0]: .12f}  {cell[0,1]: .12f}  {cell[0,2]: .12f}\n")

        f.write("$end\n")


def pbc_string(atoms: Atoms) -> str:
    try:
        return " ".join("T" if b else "F" for b in atoms.get_pbc())
    except Exception:
        return "F F F"


def format_extxyz(
    atoms: Atoms,
    energy: float,
    forces: List[Tuple[float, float, float]],
    method: str,
) -> str:
    symbols = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    n = len(symbols)
    lattice = cell_to_lattice_string(atoms)
    pbc = pbc_string(atoms)

    header_parts = [
        f"TotEnergy={energy:.8f}",
        "cutoff=-1.00000000",
        "nneightol=1.20000000",
        f'method="{method}"',
        f'pbc="{pbc}"',
    ]
    if lattice:
        header_parts.append(f'Lattice="{lattice}"')
    header_parts.append("Properties=species:S:1:pos:R:3:force:R:3:Z:I:1")

    out = [str(n), " ".join(header_parts)]
    numbers = atoms.get_atomic_numbers()
    for sym, (x, y, z), (fx, fy, fz), Z in zip(symbols, pos, forces, numbers):
        out.append(
            f"{sym:<2s}  {x: .8f}  {y: .8f}  {z: .8f}  "
            f"{fx: .8f}  {fy: .8f}  {fz: .8f}  {int(Z):d}"
        )
    return "\n".join(out) + "\n"


# ---------- calculators ----------
def run_xtb(
    atoms: Atoms,
    charge: int,
    uhf: int,
    optimize: bool,
    gfn: Optional[int] = None,
    iterations: Optional[int] = None,
    etemp: Optional[float] = None,
) -> Tuple[float, List[Tuple[float, float, float]]]:
    """
    Robust xTB runner using xtb-python's ASE calculator.

    Strategy:
    - default to GFN2 for nonperiodic, GFN1 for periodic
    - try user-requested settings first
    - if SCC fails, retry with higher electronic temperature / more iterations
    - raise a clearer error if all retries fail
    """
    from xtb.ase.calculator import XTB
    from ase.optimize import BFGS
    from ase.calculators.calculator import CalculationFailed

    if gfn is None:
        gfn = 1 if atoms.get_pbc().any() else 2

    method_name = f"GFN{gfn}-xTB"

    # First attempt uses user inputs or sane defaults
    retry_plan = [
        {
            "max_iterations": iterations if iterations is not None else 250,
            "electronic_temperature": etemp if etemp is not None else 300.0,
        },
        {
            "max_iterations": max(iterations or 250, 500),
            "electronic_temperature": max(etemp or 300.0, 1000.0),
        },
        {
            "max_iterations": max(iterations or 250, 1000),
            "electronic_temperature": max(etemp or 300.0, 2000.0),
        },
    ]

    last_err = None

    for attempt_idx, opts in enumerate(retry_plan, start=1):
        trial_atoms = atoms.copy()

        try:
            trial_atoms.calc = XTB(
                method=method_name,
                charge=charge,
                uhf=uhf,
                max_iterations=opts["max_iterations"],
                electronic_temperature=opts["electronic_temperature"],
            )

            if optimize:
                BFGS(trial_atoms, logfile=None).run(fmax=0.05)

            energy = float(trial_atoms.get_potential_energy())
            forces = [tuple(map(float, row)) for row in trial_atoms.get_forces()]

            # Copy optimized positions/cell back if optimization was requested
            if optimize:
                atoms.positions[:] = trial_atoms.positions
                try:
                    atoms.cell[:] = trial_atoms.cell
                except Exception:
                    pass

            if attempt_idx > 1:
                print(
                    f"[xTB] converged on retry {attempt_idx} "
                    f"(etemp={opts['electronic_temperature']}, "
                    f"max_iterations={opts['max_iterations']})"
                )

            return energy, forces

        except CalculationFailed as e:
            last_err = e
            print(
                f"[xTB] attempt {attempt_idx} failed "
                f"(etemp={opts['electronic_temperature']}, "
                f"max_iterations={opts['max_iterations']}): {e}"
            )

    raise RuntimeError(
        "xTB failed after multiple SCC retry attempts. "
        "This is usually a convergence problem with the structure, charge/spin, "
        "or method choice. Try one or more of:\n"
        "  - --xtb-etemp 1000\n"
        "  - --xtb-iterations 1000\n"
        "  - --xtb-gfn 1\n"
        "  - pre-optimizing the geometry\n"
        f"Last error: {last_err}"
    )

def run_qe(
    atoms: Atoms,
    pseudo_dir: str,
    pseudo_map: Dict[str, str],
    kpts: Tuple[int, int, int],
    ecutwfc: float,
    ecutrho: float,
    optimize: bool,
) -> Tuple[float, List[Tuple[float, float, float]]]:
    """
    QE via ASE. Requires Quantum ESPRESSO installed (pw.x).
    """
    from ase.calculators.espresso import Espresso, EspressoProfile
    from ase.optimize import BFGS

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
        },
        "electrons": {
            "conv_thr": 1.0e-8,
        },
    }

    if optimize:
        input_data["control"]["calculation"] = "scf"

    pseudopotentials = {el: pseudo_map[el] for el in set(atoms.get_chemical_symbols())}

    profile = EspressoProfile(command="pw.x")
    calc = Espresso(
        profile=profile,
        pseudopotentials=pseudopotentials,
        pseudo_dir=pseudo_dir,
        input_data=input_data,
        kpts=kpts,
    )
    atoms.calc = calc

    if optimize:
        dyn = BFGS(atoms, logfile=None)
        dyn.run(fmax=0.05)

    energy = float(atoms.get_potential_energy())
    forces = [tuple(map(float, row)) for row in atoms.get_forces()]
    return energy, forces


# ---------- main ----------
def parse_kpts(s: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError('kpts must look like "2,2,2"')
    return int(parts[0]), int(parts[1]), int(parts[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input xyz/extxyz (can be multi-frame)")
    ap.add_argument("-o", "--output", required=True, help="Output extxyz with energies+forces")
    ap.add_argument("--method", required=True, choices=["xtb", "qe"], help="Which method to run")
    ap.add_argument("--optimize", action="store_true", help="Run a simple geometry optimization (BFGS)")

    # xTB params
    ap.add_argument("--charge", type=int, default=0, help="Total charge (xTB)")
    ap.add_argument("--uhf", type=int, default=0, help="Number of unpaired electrons (xTB)")
    ap.add_argument(
        "--xtb-gfn",
        type=int,
        choices=[0, 1, 2],
        default=None,
        help="xTB Hamiltonian level. Default: GFN2 for nonperiodic, GFN1 for periodic",
    )
    ap.add_argument(
        "--xtb-iterations",
        type=int,
        default=None,
        help="Maximum SCC iterations for xTB",
    )
    ap.add_argument(
        "--xtb-etemp",
        type=float,
        default=None,
        help="Electronic temperature (K) for xTB",
    )

    # QE params
    ap.add_argument("--qe-pseudo-dir", default=None, help="Directory containing UPF pseudopotentials")
    ap.add_argument(
        "--qe-pseudo-map",
        default=None,
        help='JSON dict mapping element->UPF filename, e.g. \'{"C":"C.UPF","H":"H.UPF"}\'',
    )
    ap.add_argument("--qe-kpts", default="1,1,1", help='k-point mesh like "2,2,2"')
    ap.add_argument("--qe-ecutwfc", type=float, default=50.0, help="Plane-wave cutoff (Ry)")
    ap.add_argument("--qe-ecutrho", type=float, default=400.0, help="Charge density cutoff (Ry)")

    args = ap.parse_args()

    frames = list(iread(args.input, index=":"))
    if not frames:
        raise RuntimeError(f"No structures read from {args.input}")

    out_chunks: List[str] = []

    for idx, atoms in enumerate(frames):
        if args.method == "xtb":
            energy, forces = run_xtb(
                atoms,
                charge=args.charge,
                uhf=args.uhf,
                optimize=args.optimize,
                gfn=args.xtb_gfn,
                iterations=args.xtb_iterations,
                etemp=args.xtb_etemp,
            )
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

        print(f"[{idx + 1}/{len(frames)}] done: E={energy:.8f} eV")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("".join(out_chunks))

    print(f"Wrote {len(frames)} frame(s) -> {args.output}")


if __name__ == "__main__":
    main()