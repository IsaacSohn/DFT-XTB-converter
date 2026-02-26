#!/usr/bin/env python3
"""
convert_xyz_dataset.py

Reads XYZ / extended XYZ / "custom extxyz-like" files and outputs a normalized extended XYZ:
- Consistent header: TotEnergy=..., pbc, Lattice (optional), Properties=species:pos:force:Z
- Ensures forces exist (fills zeros if missing)
- Ensures Z exists (infers from element symbol if missing)
- Lets you stamp metadata for method: dft or xtb

Supports multi-frame XYZ.

Usage examples:
  python convert_xyz_dataset.py -i in.xyz -o out.extxyz --method xtb
  python convert_xyz_dataset.py -i in.xyz -o out.extxyz --method dft --default-pbc "T T T"
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

# ---- minimal periodic table mapping (enough for most organic/inorganic) ----
# Add more if needed.
Z_MAP: Dict[str, int] = {
    "H": 1,  "He": 2,
    "Li": 3, "Be": 4, "B": 5,  "C": 6,  "N": 7,  "O": 8,  "F": 9,  "Ne": 10,
    "Na": 11,"Mg": 12,"Al": 13,"Si": 14,"P": 15, "S": 16,"Cl": 17,"Ar": 18,
    "K": 19, "Ca": 20,
    "Sc": 21,"Ti": 22,"V": 23, "Cr": 24,"Mn": 25,"Fe": 26,"Co": 27,"Ni": 28,"Cu": 29,"Zn": 30,
    "Ga": 31,"Ge": 32,"As": 33,"Se": 34,"Br": 35,"Kr": 36,
    "Rb": 37,"Sr": 38,"Y": 39, "Zr": 40,"Nb": 41,"Mo": 42,"Tc": 43,"Ru": 44,"Rh": 45,"Pd": 46,"Ag": 47,"Cd": 48,
    "In": 49,"Sn": 50,"Sb": 51,"Te": 52,"I": 53, "Xe": 54,
    "Cs": 55,"Ba": 56,
    "La": 57,"Ce": 58,"Pr": 59,"Nd": 60,"Pm": 61,"Sm": 62,"Eu": 63,"Gd": 64,"Tb": 65,"Dy": 66,"Ho": 67,"Er": 68,"Tm": 69,"Yb": 70,"Lu": 71,
    "Hf": 72,"Ta": 73,"W": 74, "Re": 75,"Os": 76,"Ir": 77,"Pt": 78,"Au": 79,"Hg": 80,
    "Tl": 81,"Pb": 82,"Bi": 83,
}

def infer_Z(symbol: str) -> int:
    sym = symbol.strip()
    # Normalize capitalization: "cl" -> "Cl"
    if len(sym) >= 2 and sym[1].islower():
        sym = sym[0].upper() + sym[1].lower()
    else:
        sym = sym[0].upper() + (sym[1:].lower() if len(sym) > 1 else "")
    if sym not in Z_MAP:
        raise ValueError(f"Unknown element symbol '{symbol}'. Add it to Z_MAP.")
    return Z_MAP[sym]

@dataclass
class Frame:
    symbols: List[str]
    pos: List[Tuple[float, float, float]]
    force: List[Tuple[float, float, float]]
    Z: List[int]
    comment_kv: Dict[str, str]  # header key-values
    lattice: Optional[str] = None
    pbc: Optional[str] = None

KV_RE = re.compile(r'(\w+)=(".*?"|[^\s]+)')

def parse_comment_kv(line: str) -> Dict[str, str]:
    kv = {}
    for m in KV_RE.finditer(line):
        k, v = m.group(1), m.group(2)
        kv[k] = v.strip('"')
    return kv

def try_parse_custom_block(lines: List[str], start: int) -> Tuple[Optional[Frame], int]:
    """
    Attempts to parse a block in the user's sample format:
      N
      comment line (TotEnergy=..., pbc="T T T" Lattice="..." Properties=species... etc)
      N atom lines with: symbol x y z fx fy fz Z
    Returns (Frame or None, next_index)
    """
    if start >= len(lines):
        return None, start
    first = lines[start].strip()
    if not first:
        return None, start + 1
    if not first.isdigit():
        return None, start
    n = int(first)
    if start + 1 >= len(lines):
        return None, start
    comment = lines[start + 1].strip()
    kv = parse_comment_kv(comment)

    symbols: List[str] = []
    pos: List[Tuple[float, float, float]] = []
    force: List[Tuple[float, float, float]] = []
    Zs: List[int] = []

    i = start + 2
    if i + n > len(lines):
        return None, start

    ok = True
    for _ in range(n):
        parts = lines[i].strip().split()
        i += 1
        # Expect at least: sym x y z
        if len(parts) < 4:
            ok = False
            break
        sym = parts[0]
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except Exception:
            ok = False
            break

        # Optional force + Z in your format:
        fx = fy = fz = 0.0
        Zval: Optional[int] = None
        if len(parts) >= 7:
            try:
                fx, fy, fz = float(parts[4]), float(parts[5]), float(parts[6])
            except Exception:
                # if not parseable, keep zeros
                fx = fy = fz = 0.0
        if len(parts) >= 8:
            try:
                Zval = int(parts[7])
            except Exception:
                Zval = None

        if Zval is None:
            Zval = infer_Z(sym)

        symbols.append(sym)
        pos.append((x, y, z))
        force.append((fx, fy, fz))
        Zs.append(Zval)

    if not ok:
        return None, start

    lattice = kv.get("Lattice")
    pbc = kv.get("pbc")

    return Frame(
        symbols=symbols,
        pos=pos,
        force=force,
        Z=Zs,
        comment_kv=kv,
        lattice=lattice,
        pbc=pbc,
    ), i

def read_frames_any(path: str) -> List[Frame]:
    """
    Reads file by trying:
      1) Your custom extxyz-like block parser
      2) ASE (if installed) for xyz/extxyz/etc
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = f.read().splitlines()

    frames: List[Frame] = []
    idx = 0
    while idx < len(raw_lines):
        # Skip blank lines
        if not raw_lines[idx].strip():
            idx += 1
            continue
        fr, next_idx = try_parse_custom_block(raw_lines, idx)
        if fr is not None:
            frames.append(fr)
            idx = next_idx
        else:
            # If we hit something that's not a custom block, stop and fall back to ASE.
            frames = []
            break

    if frames:
        return frames

    # Fallback to ASE
    try:
        from ase.io import iread
    except Exception as e:
        raise RuntimeError(
            "Could not parse with custom parser AND ASE is not available. "
            "Install ASE: pip install ase"
        ) from e

    ase_frames = []
    for atoms in iread(path, index=":"):
        symbols = atoms.get_chemical_symbols()
        pos = [tuple(map(float, row)) for row in atoms.get_positions()]
        # forces may be missing
        if atoms.has("forces"):
            forces = [tuple(map(float, row)) for row in atoms.get_array("forces")]
        else:
            forces = [(0.0, 0.0, 0.0) for _ in symbols]

        # Z: infer if not available
        if atoms.has("numbers"):
            Zs = [int(z) for z in atoms.get_atomic_numbers()]
        else:
            Zs = [infer_Z(s) for s in symbols]

        kv = {}
        # ASE extxyz stores stuff in atoms.info
        if hasattr(atoms, "info") and isinstance(atoms.info, dict):
            for k, v in atoms.info.items():
                kv[str(k)] = str(v)

        lattice = None
        pbc = None
        try:
            cell = atoms.get_cell()
            if cell is not None and cell.rank == 3 and cell.volume > 0:
                # extxyz expects 9 numbers row-major
                a = cell.array
                lattice = " ".join(f"{a[i,j]:.8f}" for i in range(3) for j in range(3))
        except Exception:
            lattice = None
        try:
            pbc = " ".join("T" if b else "F" for b in atoms.get_pbc())
        except Exception:
            pbc = None

        ase_frames.append(Frame(symbols, pos, forces, Zs, kv, lattice=lattice, pbc=pbc))

    if not ase_frames:
        raise RuntimeError(f"No frames read from {path}.")
    return ase_frames

def format_extxyz_frame(fr: Frame, method: str, default_pbc: str, default_cutoff: float) -> str:
    n = len(fr.symbols)

    # Normalize energy key
    # Accept many possible energy keys, keep TotEnergy as the output key.
    energy = None
    for k in ("TotEnergy", "energy", "E", "Energy", "dft_energy", "xtb_energy"):
        if k in fr.comment_kv:
            energy = fr.comment_kv[k]
            break
    if energy is None:
        # If not present, keep NaN so downstream knows it's missing
        energy = "nan"

    # Lattice / pbc
    pbc = fr.pbc or default_pbc
    lattice = fr.lattice  # can be None

    # Keep some optional keys if present
    cutoff = fr.comment_kv.get("cutoff", f"{default_cutoff:.8f}")
    nneightol = fr.comment_kv.get("nneightol", "1.20000000")

    # Build comment line
    # extxyz: pbc="T T T" Lattice="..." Properties=species:S:1:pos:R:3:force:R:3:Z:I:1
    parts = []
    parts.append(f'TotEnergy={float(energy) if energy!="nan" else math.nan:.8f}')
    parts.append(f'cutoff={float(cutoff):.8f}' if cutoff != "nan" else "cutoff=nan")
    parts.append(f'nneightol={float(nneightol):.8f}' if nneightol != "nan" else "nneightol=nan")
    parts.append(f'method="{method}"')
    parts.append(f'pbc="{pbc}"')
    if lattice:
        parts.append(f'Lattice="{lattice}"')
    parts.append('Properties=species:S:1:pos:R:3:force:R:3:Z:I:1')

    header = "\n".join([str(n), " ".join(parts)])

    # Atom lines
    atom_lines = []
    for sym, (x, y, z), (fx, fy, fz), Z in zip(fr.symbols, fr.pos, fr.force, fr.Z):
        atom_lines.append(
            f"{sym:<2s}  {x: .8f}  {y: .8f}  {z: .8f}  {fx: .8f}  {fy: .8f}  {fz: .8f}  {int(Z):d}"
        )

    return header + "\n" + "\n".join(atom_lines) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input xyz/extxyz/custom file")
    ap.add_argument("-o", "--output", required=True, help="Output extxyz")
    ap.add_argument("--method", required=True, choices=["dft", "xtb"], help="Stamp method metadata")
    ap.add_argument("--default-pbc", default="F F F", help='Default pbc string, e.g. "T T T" or "F F F"')
    ap.add_argument("--default-cutoff", type=float, default=-1.0, help="Default cutoff if missing")
    args = ap.parse_args()

    frames = read_frames_any(args.input)

    with open(args.output, "w", encoding="utf-8") as f:
        for fr in frames:
            # Ensure forces length
            if len(fr.force) != len(fr.symbols):
                fr.force = [(0.0, 0.0, 0.0) for _ in fr.symbols]
            # Ensure Z length
            if len(fr.Z) != len(fr.symbols):
                fr.Z = [infer_Z(s) for s in fr.symbols]
            f.write(format_extxyz_frame(fr, args.method, args.default_pbc, args.default_cutoff))

    print(f"Wrote {len(frames)} frame(s) -> {args.output}")

if __name__ == "__main__":
    main()