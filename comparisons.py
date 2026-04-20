from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# =========================================================
# Config
# =========================================================
COMPARE_TO_FILE = Path("comparisons/dft_to_compare_to.xyz")
LAST_10_FILE = Path("comparisons/dft_to_compare_to_last10pct.xyz")

PREDICTION_FILES = [
    Path("comparisons/xtb_predictions.xyz"),
    Path("comparisons/dft_predictions.xyz"),
    Path("comparisons/delta_predictions.xyz"),
]

# If your prediction file uses a different header key, add it here
ENERGY_KEYS_PRED = ["predicted_energy", "energy", "TotEnergy"]
ENERGY_KEYS_TRUE = ["energy", "TotEnergy"]

# Force column priority:
# - for prediction files, prefer predicted_forces, otherwise forces, otherwise force
# - for compare_to / DFT, prefer forces, otherwise force
FORCE_KEYS_PRED = ["predicted_forces", "forces", "force"]
FORCE_KEYS_TRUE = ["forces", "force"]


# =========================================================
# Helpers
# =========================================================
float_pattern = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)=(".*?"|\S+)')


def parse_header(header_line: str) -> Dict[str, str]:
    """
    Parse extxyz-like header line into a dict of key -> raw string value.
    """
    out = {}
    for key, value in float_pattern.findall(header_line):
        out[key] = value.strip('"')
    return out


def parse_properties(properties_str: str) -> List[Tuple[str, str, int]]:
    """
    Parse extxyz Properties field like:
    species:S:1:pos:R:3:force:R:3:predicted_forces:R:3:forces:R:3:energies:R:1

    Returns:
        [(name, dtype_code, count), ...]
    """
    parts = properties_str.split(":")
    if len(parts) % 3 != 0:
        raise ValueError(f"Malformed Properties string: {properties_str}")

    props = []
    for i in range(0, len(parts), 3):
        name = parts[i]
        dtype_code = parts[i + 1]
        count = int(parts[i + 2])
        props.append((name, dtype_code, count))
    return props


def split_columns(
    atom_line: str,
    props: List[Tuple[str, str, int]]
) -> Dict[str, List[str]]:
    """
    Split one atomic line according to the parsed Properties layout.
    """
    tokens = atom_line.split()
    out = {}
    idx = 0

    for name, _dtype, count in props:
        out[name] = tokens[idx: idx + count]
        idx += count

    if idx != len(tokens):
        raise ValueError(
            f"Token count mismatch. Expected {idx}, got {len(tokens)}.\n"
            f"Line: {atom_line}"
        )

    return out


def read_frames(path: Path) -> List[List[str]]:
    """
    Read xyz/extxyz file into a list of frames, where each frame is a list of lines.
    """
    text = path.read_text(encoding="utf-8").splitlines()
    frames = []

    i = 0
    n_lines = len(text)

    while i < n_lines:
        if not text[i].strip():
            i += 1
            continue

        natoms = int(text[i].strip())
        frame = text[i:i + natoms + 2]

        if len(frame) < natoms + 2:
            raise ValueError(f"Incomplete frame near line {i+1} in {path}")

        frames.append(frame)
        i += natoms + 2

    return frames


def write_frames(path: Path, frames: List[List[str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for frame in frames:
            f.write("\n".join(frame))
            f.write("\n")


def keep_last_fraction(frames: List[List[str]], frac: float) -> List[List[str]]:
    """
    Keep the last ceil(frac * N) frames.
    """
    n_total = len(frames)
    n_keep = max(1, math.ceil(n_total * frac))
    return frames[-n_keep:]


def extract_energy(header: Dict[str, str], keys: List[str]) -> float:
    for key in keys:
        if key in header:
            return float(header[key])
    raise KeyError(f"Could not find any energy key from {keys} in header: {header}")


def extract_forces_from_frame(frame: List[str], preferred_keys: List[str]) -> np.ndarray:
    """
    Returns forces as shape (natoms, 3).
    """
    natoms = int(frame[0].strip())
    header = parse_header(frame[1])

    if "Properties" not in header:
        raise KeyError("Header missing Properties field")

    props = parse_properties(header["Properties"])

    found_key = None
    prop_names = [name for name, _, _ in props]
    for key in preferred_keys:
        if key in prop_names:
            found_key = key
            break

    if found_key is None:
        raise KeyError(
            f"Could not find any force key from {preferred_keys}. "
            f"Available properties: {prop_names}"
        )

    forces = []
    for line in frame[2:2 + natoms]:
        cols = split_columns(line, props)
        vals = [float(x) for x in cols[found_key]]
        if len(vals) != 3:
            raise ValueError(f"{found_key} is not length 3 on line: {line}")
        forces.append(vals)

    return np.asarray(forces, dtype=float)


def get_species_and_positions(frame: List[str]) -> Tuple[List[str], np.ndarray]:
    """
    Used to sanity-check alignment between files.
    """
    natoms = int(frame[0].strip())
    header = parse_header(frame[1])

    if "Properties" not in header:
        raise KeyError("Header missing Properties field")

    props = parse_properties(header["Properties"])

    prop_names = [name for name, _, _ in props]
    if "species" not in prop_names or "pos" not in prop_names:
        raise KeyError(f"Expected species and pos in properties, got {prop_names}")

    species = []
    positions = []

    for line in frame[2:2 + natoms]:
        cols = split_columns(line, props)
        species.append(cols["species"][0])
        positions.append([float(x) for x in cols["pos"]])

    return species, np.asarray(positions, dtype=float)


def compare_single_file(pred_path: Path, true_frames: List[List[str]]) -> None:
    pred_frames = read_frames(pred_path)

    if len(pred_frames) < len(true_frames):
        raise ValueError(
            f"{pred_path} has fewer frames ({len(pred_frames)}) than compare target ({len(true_frames)})"
        )

    # Take the last N frames so it lines up with the trimmed compare_to file
    pred_frames = pred_frames[-len(true_frames):]

    energy_abs_errors = []
    force_component_abs_errors = []
    force_vector_mae_per_atom = []
    force_vector_rmse_per_atom = []

    for idx, (pred_frame, true_frame) in enumerate(zip(pred_frames, true_frames), start=1):
        pred_header = parse_header(pred_frame[1])
        true_header = parse_header(true_frame[1])

        pred_species, pred_pos = get_species_and_positions(pred_frame)
        true_species, true_pos = get_species_and_positions(true_frame)

        if pred_species != true_species:
            raise ValueError(f"Species mismatch in frame {idx} for {pred_path.name}")

        if pred_pos.shape != true_pos.shape:
            raise ValueError(f"Position shape mismatch in frame {idx} for {pred_path.name}")

        # sanity check that geometries align
        max_pos_diff = np.max(np.abs(pred_pos - true_pos))
        if max_pos_diff > 1e-5:
            raise ValueError(
                f"Geometry mismatch in frame {idx} for {pred_path.name}. "
                f"Max |Δpos| = {max_pos_diff}"
            )

        pred_energy = extract_energy(pred_header, ENERGY_KEYS_PRED)
        true_energy = extract_energy(true_header, ENERGY_KEYS_TRUE)
        energy_abs_errors.append(abs(pred_energy - true_energy))

        pred_forces = extract_forces_from_frame(pred_frame, FORCE_KEYS_PRED)
        true_forces = extract_forces_from_frame(true_frame, FORCE_KEYS_TRUE)

        diff = pred_forces - true_forces
        force_component_abs_errors.extend(np.abs(diff).ravel())

        per_atom_norm = np.linalg.norm(diff, axis=1)
        force_vector_mae_per_atom.extend(per_atom_norm)
        force_vector_rmse_per_atom.extend(per_atom_norm ** 2)

    energy_abs_errors = np.asarray(energy_abs_errors, dtype=float)
    force_component_abs_errors = np.asarray(force_component_abs_errors, dtype=float)
    force_vector_mae_per_atom = np.asarray(force_vector_mae_per_atom, dtype=float)
    force_vector_rmse_per_atom = np.asarray(force_vector_rmse_per_atom, dtype=float)

    print("=" * 70)
    print(f"FILE: {pred_path.name}")
    print(f"Frames compared: {len(true_frames)}")
    print()

    print("Energy errors")
    print(f"  MAE   : {energy_abs_errors.mean():.8f}")
    print(f"  RMSE  : {np.sqrt(np.mean(energy_abs_errors ** 2)):.8f}")
    print(f"  MaxAE : {energy_abs_errors.max():.8f}")
    print()

    print("Force errors (component-wise, over x/y/z values)")
    print(f"  MAE   : {force_component_abs_errors.mean():.8f}")
    print(f"  RMSE  : {np.sqrt(np.mean(force_component_abs_errors ** 2)):.8f}")
    print(f"  MaxAE : {force_component_abs_errors.max():.8f}")
    print()

    print("Force errors (per-atom vector norm |F_pred - F_true|)")
    print(f"  Mean  : {force_vector_mae_per_atom.mean():.8f}")
    print(f"  RMSE  : {np.sqrt(np.mean(force_vector_rmse_per_atom)):.8f}")
    print(f"  Max   : {force_vector_mae_per_atom.max():.8f}")
    print()


def main() -> None:
    if not COMPARE_TO_FILE.exists():
        raise FileNotFoundError(f"Missing compare file: {COMPARE_TO_FILE}")

    compare_frames = read_frames(COMPARE_TO_FILE)
    trimmed_frames = keep_last_fraction(compare_frames, 0.10)
    write_frames(LAST_10_FILE, trimmed_frames)

    print(f"Original compare_to frames: {len(compare_frames)}")
    print(f"Last 10% frames kept      : {len(trimmed_frames)}")
    print(f"Wrote trimmed file        : {LAST_10_FILE}")
    print()

    for pred_path in PREDICTION_FILES:
        if not pred_path.exists():
            print(f"[skip] Missing file: {pred_path}")
            continue
        compare_single_file(pred_path, trimmed_frames)


if __name__ == "__main__":
    main()