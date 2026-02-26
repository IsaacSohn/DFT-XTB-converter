from __future__ import annotations

import os
import subprocess
import tempfile
import numpy as np

from ase.build import molecule
from ase.calculators.calculator import Calculator, all_changes


class XTBExternal(Calculator):
    """ASE calculator wrapper that calls the external `xtb` executable.

    Outputs:
      - energy in eV
      - forces in eV/Å
    """

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
    errors="replace",   # or "ignore"
)
            out = p.stdout
            if p.returncode != 0:
                raise RuntimeError(f"xTB failed (code {p.returncode}). Output:\n{out}")

            # Parse TOTAL ENERGY from stdout (in Eh)
            energy_eh = None
            for line in out.splitlines():
                if "TOTAL ENERGY" in line and "Eh" in line:
                    # Grab the last float before 'Eh'
                    toks = line.replace("=", " ").split()
                    for i, t in enumerate(toks):
                        if t == "Eh" and i > 0:
                            try:
                                energy_eh = float(toks[i - 1])
                            except ValueError:
                                pass
            if energy_eh is None:
                raise RuntimeError("Could not parse TOTAL ENERGY from xTB output.")

            # Parse gradient file (Eh/Bohr): last N lines are typically the gradient vectors
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

            # Convert units:
            # Eh -> eV
            # Eh/Bohr -> eV/Å
            EH_TO_EV = 27.211386245988
            BOHR_TO_ANG = 0.529177210903

            energy_ev = energy_eh * EH_TO_EV
            forces_ev_per_ang = -(grad * EH_TO_EV / BOHR_TO_ANG)

            self.results["energy"] = float(energy_ev)
            self.results["forces"] = forces_ev_per_ang


atoms = molecule("H2O")
atoms.calc = XTBExternal(charge=0, uhf=0, gfn=2)

print("Energy (eV):", atoms.get_potential_energy())
print("Forces (eV/Å):")
print(atoms.get_forces())