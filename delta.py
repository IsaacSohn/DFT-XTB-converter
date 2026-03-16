#!/usr/bin/env python3
"""
delta.py

Build a delta-labeled extxyz dataset from frames that already contain
xTB and DFT labels.

Delta definition:
    delta_energy = xtb_energy - dft_energy

This script:
- Reads XYZ / extXYZ (multi-frame supported)
- Groups frames by species + geometry
- Matches method="xtb" with method="dft" or method="dft-qe"
- Writes a new extxyz file with method="delta"
- Stores delta in TotEnergy
- Preserves atomic structure
- Writes zero forces by default

Example:
    python delta.py -i labeled.extxyz -o delta.extxyz
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from ase import Atoms
from ase.io import read, write


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create delta dataset from xtb and dft extxyz frames"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input xyz/extxyz file (can contain multiple frames)"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output extxyz file with method='delta'"
    )
    parser.add_argument(
        "--xtb-method",
        default="xtb",
        help="Method label to treat as xTB (default: xtb)"
    )
    parser.add_argument(
        "--dft-methods",
        nargs="+",
        default=["dft", "dft-qe"],
        help='Method labels to treat as DFT (default: dft dft-qe)'
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-8,
        help="Relative tolerance for geometry hashing rounding (default: 1e-8 style via decimals option)"
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=8,
        help="Number of decimals used when matching geometries (default: 8)"
    )
    parser.add_argument(
        "--keep-existing-forces",
        action="store_true",
        help="If set, keep force array from one input frame when present instead of writing zeros"
    )
    return parser.parse_args()


def get_method(atoms: Atoms) -> str:
    method = atoms.info.get("method", "")
    if method is None:
        return ""
    return str(method).strip().lower()


def get_energy(atoms: Atoms) -> float:
    """
    Pull total energy from common places.
    Preference:
    1) atoms.info["TotEnergy"]
    2) atoms.info["energy"]
    """
    if "TotEnergy" in atoms.info:
        return float(atoms.info["TotEnergy"])
    if "energy" in atoms.info:
        return float(atoms.info["energy"])
    raise ValueError("Frame is missing TotEnergy/energy metadata")


def canonical_key(atoms: Atoms, decimals: int = 8) -> Tuple:
    """
    Build a matching key from:
    - chemical symbols
    - rounded cartesian positions
    - pbc

    This assumes xtb and dft frames for the same structure keep the same atom order.
    """
    symbols = tuple(atoms.get_chemical_symbols())
    positions = np.round(atoms.get_positions(), decimals=decimals)
    pos_key = tuple(map(tuple, positions.tolist()))
    pbc_key = tuple(bool(x) for x in atoms.get_pbc())
    return (symbols, pos_key, pbc_key)


def make_delta_frame(
    template: Atoms,
    delta_energy: float,
    keep_existing_forces: bool = False
) -> Atoms:
    """
    Create a new Atoms object for the delta dataset.

    Writes:
    - method="delta"
    - TotEnergy=<xtb-dft>
    - force array (zeros unless keep_existing_forces is enabled and available)
    - Z array
    """
    out = template.copy()

    # Remove calculator to avoid accidental carry-over
    out.calc = None

    natoms = len(out)

    # Force array
    if keep_existing_forces and "force" in out.arrays:
        force_arr = np.asarray(out.arrays["force"], dtype=float)
        if force_arr.shape != (natoms, 3):
            force_arr = np.zeros((natoms, 3), dtype=float)
    elif keep_existing_forces and "forces" in out.arrays:
        force_arr = np.asarray(out.arrays["forces"], dtype=float)
        if force_arr.shape != (natoms, 3):
            force_arr = np.zeros((natoms, 3), dtype=float)
    else:
        force_arr = np.zeros((natoms, 3), dtype=float)

    out.arrays["force"] = force_arr
    out.arrays["Z"] = np.asarray(out.get_atomic_numbers(), dtype=int)

    # Clean up possible conflicting arrays
    if "forces" in out.arrays and "force" in out.arrays:
        del out.arrays["forces"]

    # Metadata
    out.info["TotEnergy"] = float(delta_energy)
    out.info["method"] = "delta"

    # Optional consistency fields if you want the same style as your pipeline
    out.info.setdefault("cutoff", -1.0)
    out.info.setdefault("nneightol", 1.2)

    return out


def main() -> None:
    args = parse_args()

    frames = read(args.input, index=":")
    if not isinstance(frames, list):
        frames = [frames]

    if len(frames) == 0:
        raise ValueError("No frames found in input file")

    grouped: Dict[Tuple, List[Atoms]] = defaultdict(list)
    for atoms in frames:
        key = canonical_key(atoms, decimals=args.decimals)
        grouped[key].append(atoms)

    out_frames: List[Atoms] = []
    total_groups = 0
    matched_groups = 0
    skipped_groups = 0

    xtb_method = args.xtb_method.strip().lower()
    dft_methods = {m.strip().lower() for m in args.dft_methods}

    for _, group in grouped.items():
        total_groups += 1

        xtb_frame = None
        dft_frame = None

        for atoms in group:
            method = get_method(atoms)
            if method == xtb_method and xtb_frame is None:
                xtb_frame = atoms
            elif method in dft_methods and dft_frame is None:
                dft_frame = atoms

        if xtb_frame is None or dft_frame is None:
            skipped_groups += 1
            continue

        try:
            xtb_energy = get_energy(xtb_frame)
            dft_energy = get_energy(dft_frame)
        except ValueError as exc:
            print(f"[warn] skipping group due to missing energy: {exc}")
            skipped_groups += 1
            continue

        delta_energy = xtb_energy - dft_energy

        delta_atoms = make_delta_frame(
            template=xtb_frame,
            delta_energy=delta_energy,
            keep_existing_forces=args.keep_existing_forces
        )

        out_frames.append(delta_atoms)
        matched_groups += 1

    if len(out_frames) == 0:
        raise RuntimeError(
            "No matching xtb/dft pairs were found. "
            "Check method labels and whether geometries match exactly."
        )

    write(args.output, out_frames, format="extxyz")

    print(f"[done] wrote {len(out_frames)} delta frames to {args.output}")
    print(f"[info] total groups:   {total_groups}")
    print(f"[info] matched groups: {matched_groups}")
    print(f"[info] skipped groups: {skipped_groups}")


if __name__ == "__main__":
    main()