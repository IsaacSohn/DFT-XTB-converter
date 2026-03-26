#!/usr/bin/env python3
"""
delta.py

Build a delta extxyz dataset from TWO files:
- one xTB file
- one DFT file

For each matched structure:
    delta_energy = E_dft - E_xtb
    delta_force  = F_dft - F_xtb

Designed to be safer for periodic systems by wrapping coordinates
before matching and including the cell in the matching key.

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
        default=6,
        help="Decimals used when matching wrapped geometries (default: 6)"
    )
    parser.add_argument(
        "--energy-mode",
        choices=["dft-minus-xtb", "xtb-minus-dft"],
        default="dft-minus-xtb",
        help="Delta energy convention (default: dft-minus-xtb)"
    )
    parser.add_argument(
        "--force-mode",
        choices=["delta", "zeros", "keep-dft", "keep-xtb"],
        default="delta",
        help=(
            "How to write forces:\n"
            "  delta    -> dft_forces - xtb_forces if both exist, else zeros\n"
            "  zeros    -> always zeros\n"
            "  keep-dft -> copy DFT forces\n"
            "  keep-xtb -> copy xTB forces\n"
            "(default: delta)"
        )
    )
    parser.add_argument(
        "--strict-count",
        action="store_true",
        help="Raise an error if number of frames in xTB and DFT files differ"
    )
    return parser.parse_args()


def load_frames(path: str) -> List[Atoms]:
    frames = read(path, index=":")
    if not isinstance(frames, list):
        frames = [frames]
    if len(frames) == 0:
        raise ValueError(f"No frames found in: {path}")
    return frames


def wrapped_copy(atoms: Atoms) -> Atoms:
    """
    Return a copy with periodic coordinates wrapped into the unit cell.
    Non-periodic structures are returned unchanged.
    """
    out = atoms.copy()
    if out.get_pbc().any():
        out.wrap()
    return out


def get_energy(atoms: Atoms) -> float:
    if "TotEnergy" in atoms.info:
        return float(atoms.info["TotEnergy"])
    if "energy" in atoms.info:
        return float(atoms.info["energy"])
    raise ValueError("Frame is missing TotEnergy/energy metadata")


def get_forces_array(atoms: Atoms) -> np.ndarray | None:
    for key in ("force", "forces"):
        if key in atoms.arrays:
            arr = np.asarray(atoms.arrays[key], dtype=float)
            if arr.shape == (len(atoms), 3):
                return arr
    return None


def canonical_key(atoms: Atoms, decimals: int = 6) -> Tuple:
    """
    Matching key for a structure.

    Uses:
    - chemical symbols
    - wrapped cartesian positions
    - cell
    - pbc

    Assumes atom ordering is the same between xTB and DFT frames.
    """
    a = wrapped_copy(atoms)

    symbols = tuple(a.get_chemical_symbols())
    positions = np.round(a.get_positions(), decimals=decimals)
    pos_key = tuple(map(tuple, positions.tolist()))

    cell = np.round(a.cell.array, decimals=decimals)
    cell_key = tuple(map(tuple, cell.tolist()))

    pbc_key = tuple(bool(x) for x in a.get_pbc())

    return (symbols, pos_key, cell_key, pbc_key)


def compute_delta_energy(xtb_energy: float, dft_energy: float, mode: str) -> float:
    if mode == "dft-minus-xtb":
        return dft_energy - xtb_energy
    if mode == "xtb-minus-dft":
        return xtb_energy - dft_energy
    raise ValueError(f"Unknown energy mode: {mode}")


def compute_output_forces(xtb_frame: Atoms, dft_frame: Atoms, mode: str) -> np.ndarray:
    natoms = len(xtb_frame)
    zeros = np.zeros((natoms, 3), dtype=float)

    xtb_forces = get_forces_array(xtb_frame)
    dft_forces = get_forces_array(dft_frame)

    if mode == "zeros":
        return zeros
    if mode == "keep-dft":
        return dft_forces if dft_forces is not None else zeros
    if mode == "keep-xtb":
        return xtb_forces if xtb_forces is not None else zeros
    if mode == "delta":
        if xtb_forces is not None and dft_forces is not None:
            return dft_forces - xtb_forces
        return zeros

    raise ValueError(f"Unknown force mode: {mode}")


def make_delta_frame(
    xtb_frame: Atoms,
    dft_frame: Atoms,
    delta_energy: float,
    force_mode: str,
    energy_mode: str,
) -> Atoms:
    out = wrapped_copy(xtb_frame)
    out.calc = None

    force_arr = compute_output_forces(xtb_frame, dft_frame, force_mode)

    out.arrays["force"] = np.asarray(force_arr, dtype=float)
    out.arrays["Z"] = np.asarray(out.get_atomic_numbers(), dtype=int)

    if "forces" in out.arrays:
        del out.arrays["forces"]

    xtb_energy = get_energy(xtb_frame)
    dft_energy = get_energy(dft_frame)

    out.info["TotEnergy"] = float(delta_energy)
    out.info["method"] = "delta"
    out.info["delta_definition"] = energy_mode
    out.info["xtb_energy"] = float(xtb_energy)
    out.info["dft_energy"] = float(dft_energy)
    out.info.setdefault("cutoff", -1.0)
    out.info.setdefault("nneightol", 1.2)

    return out


def build_frame_map(frames: List[Atoms], decimals: int) -> Dict[Tuple, Atoms]:
    frame_map: Dict[Tuple, Atoms] = {}

    for i, atoms in enumerate(frames):
        key = canonical_key(atoms, decimals=decimals)
        if key in frame_map:
            print(f"[warn] duplicate geometry detected at frame {i}; keeping first occurrence")
            continue
        frame_map[key] = atoms

    return frame_map


def main() -> None:
    args = parse_args()

    xtb_frames = load_frames(args.xtb_input)
    dft_frames = load_frames(args.dft_input)

    if args.strict_count and len(xtb_frames) != len(dft_frames):
        raise RuntimeError(
            f"Frame count mismatch: xTB has {len(xtb_frames)} frames, "
            f"DFT has {len(dft_frames)} frames"
        )

    xtb_map = build_frame_map(xtb_frames, decimals=args.decimals)
    dft_map = build_frame_map(dft_frames, decimals=args.decimals)

    out_frames: List[Atoms] = []
    matched = 0
    skipped = 0

    for key, xtb_frame in xtb_map.items():
        dft_frame = dft_map.get(key)
        if dft_frame is None:
            skipped += 1
            continue

        try:
            xtb_energy = get_energy(xtb_frame)
            dft_energy = get_energy(dft_frame)
        except ValueError as exc:
            print(f"[warn] skipping frame due to missing energy: {exc}")
            skipped += 1
            continue

        delta_energy = compute_delta_energy(xtb_energy, dft_energy, args.energy_mode)

        delta_atoms = make_delta_frame(
            xtb_frame=xtb_frame,
            dft_frame=dft_frame,
            delta_energy=delta_energy,
            force_mode=args.force_mode,
            energy_mode=args.energy_mode,
        )

        out_frames.append(delta_atoms)
        matched += 1

    if not out_frames:
        raise RuntimeError(
            "No matching xTB/DFT pairs found.\n"
            "For periodic systems, this usually means wrapped positions, cell, "
            "or atom ordering do not match closely enough."
        )

    write(args.output, out_frames, format="extxyz")

    print(f"[done] wrote {len(out_frames)} delta frames to {args.output}")
    print(f"[info] xTB frames:      {len(xtb_frames)}")
    print(f"[info] DFT frames:      {len(dft_frames)}")
    print(f"[info] unique xTB keys:  {len(xtb_map)}")
    print(f"[info] unique DFT keys:  {len(dft_map)}")
    print(f"[info] matched:         {matched}")
    print(f"[info] skipped:         {skipped}")
    print(f"[info] energy mode:     {args.energy_mode}")
    print(f"[info] force mode:      {args.force_mode}")
    print(f"[info] decimals:        {args.decimals}")


if __name__ == "__main__":
    main()