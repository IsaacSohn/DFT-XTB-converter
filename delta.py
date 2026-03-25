#!/usr/bin/env python3
"""
delta.py

Build a delta-labeled extxyz dataset from TWO separate files:
- one file containing xTB-labeled frames
- one file containing DFT-labeled frames

Delta definition:
    delta_energy = xtb_energy - dft_energy

This script:
- Reads XYZ / extXYZ (multi-frame supported) from TWO files
- Groups/matches frames by species + geometry
- Writes a new extxyz file with method="delta"
- Stores delta in TotEnergy
- Preserves atomic structure
- Writes zero forces by default

Example:
    python delta.py -x cmon_xtb.extxyz -d cmon.extxyz -o delta.extxyz
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np
from ase import Atoms
from ase.io import read, write


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create delta dataset from separate xTB and DFT extxyz files"
    )
    parser.add_argument(
        "-x", "--xtb-input",
        required=True,
        help="Input xyz/extxyz file containing xTB frames"
    )
    parser.add_argument(
        "-d", "--dft-input",
        required=True,
        help="Input xyz/extxyz file containing DFT frames"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output extxyz file with method='delta'"
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
        help="If set, keep force array from xTB frame when present instead of writing zeros"
    )
    return parser.parse_args()


def load_frames(path: str) -> List[Atoms]:
    frames = read(path, index=":")
    if not isinstance(frames, list):
        frames = [frames]

    if len(frames) == 0:
        raise ValueError(f"No frames found in input file: {path}")

    return frames


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

    This assumes xTB and DFT frames for the same structure keep the same atom order.
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

    # Optional consistency fields
    out.info.setdefault("cutoff", -1.0)
    out.info.setdefault("nneightol", 1.2)

    return out


def build_frame_map(frames: List[Atoms], decimals: int) -> Dict[Tuple, Atoms]:
    """
    Build a map from canonical geometry key -> frame.

    If duplicate geometries exist in the same file, the first one is kept.
    """
    frame_map: Dict[Tuple, Atoms] = {}

    for atoms in frames:
        key = canonical_key(atoms, decimals=decimals)
        if key not in frame_map:
            frame_map[key] = atoms

    return frame_map


def main() -> None:
    args = parse_args()

    xtb_frames = load_frames(args.xtb_input)
    dft_frames = load_frames(args.dft_input)

    xtb_map = build_frame_map(xtb_frames, decimals=args.decimals)
    dft_map = build_frame_map(dft_frames, decimals=args.decimals)

    out_frames: List[Atoms] = []

    total_xtb_groups = len(xtb_map)
    matched_groups = 0
    skipped_groups = 0

    for key, xtb_frame in xtb_map.items():
        dft_frame = dft_map.get(key)

        if dft_frame is None:
            skipped_groups += 1
            continue

        try:
            xtb_energy = get_energy(xtb_frame)
            dft_energy = get_energy(dft_frame)
        except ValueError as exc:
            print(f"[warn] skipping structure due to missing energy: {exc}")
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
            "No matching xTB/DFT pairs were found. "
            "Check that the two files contain identical geometries "
            "(same atom order and positions after rounding)."
        )

    write(args.output, out_frames, format="extxyz")

    print(f"[done] wrote {len(out_frames)} delta frames to {args.output}")
    print(f"[info] xTB structures:   {total_xtb_groups}")
    print(f"[info] matched groups:  {matched_groups}")
    print(f"[info] skipped groups:  {skipped_groups}")
    print(f"[info] DFT structures:  {len(dft_map)}")


if __name__ == "__main__":
    main()