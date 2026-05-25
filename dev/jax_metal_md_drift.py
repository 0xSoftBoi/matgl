"""NVE MD energy-drift check: TensorNet on JAX-Metal vs PyTorch-CPU.

Loads a pretrained TensorNet potential, runs 1 ps of NVE MD (1000 steps x 1 fs)
on a small LiFePO4 supercell with each backend, and reports the per-step
total-energy drift (slope and std). A well-conserving MD should show drift of
~1 meV/atom/ps or better.

Both backends use float32 internally on Apple Silicon (PyTorch on MPS, JAX on
applejax/MLX), so the relevant question is whether float32 + Metal introduces
significantly more drift than the established PyTorch path.

    JAX_PLATFORMS=mps .venv-metal/bin/python dev/jax_metal_md_drift.py
"""

from __future__ import annotations

import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.units import fs

import matgl
from matgl.ext.ase import PESCalculator
from matgl.ext.jax import JAXPESCalculator

MODEL = "TensorNet-PES-MatPES-r2SCAN-2025.2"
STEPS = 1000
DT_FS = 1.0
TEMP_K = 300.0
SEED = 42


def _get_lifepo4():
    """LiFePO4 (28 atoms) from pymatgen testing utilities."""
    try:
        from pymatgen.util.testing import MatSciTest

        return MatSciTest().get_structure("LiFePO4")
    except ImportError:
        from pymatgen.util.testing import PymatgenTest

        return PymatgenTest().get_structure("LiFePO4")


def _run_nve(calculator, atoms, label: str) -> tuple[np.ndarray, float]:
    """Run NVE MD, sampling total energy every step. Return (energies, wall_s)."""
    atoms = atoms.copy()
    atoms.calc = calculator
    rng = np.random.default_rng(SEED)
    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMP_K, rng=rng)
    dyn = VelocityVerlet(atoms, timestep=DT_FS * fs)
    energies = np.empty(STEPS + 1, dtype=np.float64)
    energies[0] = atoms.get_total_energy()
    t0 = time.perf_counter()
    for i in range(STEPS):
        dyn.run(1)
        energies[i + 1] = atoms.get_total_energy()
    return energies, time.perf_counter() - t0


def _drift_metrics(energies: np.ndarray, n_atoms: int) -> dict:
    """Linear fit of total energy vs step -> slope (drift) and residual std."""
    t = np.arange(len(energies), dtype=np.float64)
    slope, intercept = np.polyfit(t, energies, 1)
    residuals = energies - (slope * t + intercept)
    e0 = energies[0]
    return {
        "E0_eV": float(e0),
        "Eend_eV": float(energies[-1]),
        "dE_total_meV_per_atom": float((energies[-1] - e0) / n_atoms * 1e3),
        "drift_meV_per_atom_per_ps": float(slope / n_atoms * 1e3 / (DT_FS * 1e-3)),
        "residual_std_meV_per_atom": float(residuals.std() / n_atoms * 1e3),
    }


def main() -> None:
    """Run the NVE drift comparison and print the summary."""
    structure = _get_lifepo4()
    n_atoms = len(structure)
    print(f"NVE MD: {structure.composition.reduced_formula} ({n_atoms} atoms)")
    print(f"steps={STEPS}, dt={DT_FS} fs, initial T={TEMP_K} K, seed={SEED}")
    print(f"model: {MODEL}\n")

    pot = matgl.load_model(MODEL)
    pot.eval()

    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = AseAtomsAdaptor.get_atoms(structure)

    results = {}
    for label, calc in [
        ("PyTorch CPU", PESCalculator(pot, stress_unit="eV/A3")),
        ("JAX Metal", JAXPESCalculator(pot, stress_unit="eV/A3")),
    ]:
        energies, wall = _run_nve(calc, atoms, label)
        metrics = _drift_metrics(energies, n_atoms)
        results[label] = {**metrics, "wall_s": wall, "ms_per_step": wall / STEPS * 1e3}

    cols = [
        ("wall_s", "wall (s)", "{:>10.2f}"),
        ("ms_per_step", "ms/step", "{:>10.2f}"),
        ("E0_eV", "E0 (eV)", "{:>12.4f}"),
        ("Eend_eV", "Eend (eV)", "{:>12.4f}"),
        ("dE_total_meV_per_atom", "dE total meV/atom", "{:>18.4f}"),
        ("drift_meV_per_atom_per_ps", "drift meV/atom/ps", "{:>18.4f}"),
        ("residual_std_meV_per_atom", "residual std meV/atom", "{:>22.4f}"),
    ]
    width_label = max(len(label) for label in results) + 2
    header = " " * width_label + "".join(f"{h:>{len(fmt.format(0.0)) + 1}s}" for _, h, fmt in cols)
    print(header)
    print("-" * len(header))
    for label, m in results.items():
        line = f"{label:<{width_label}s}"
        for key, _h, fmt in cols:
            line += " " + fmt.format(m[key])
        print(line)
    print(
        "\ndE total meV/atom = (E_end - E_0) / n_atoms"
        "\ndrift meV/atom/ps = linear-fit slope of total energy"
        "\nresidual std meV/atom = std of total energy after removing the linear trend"
    )


if __name__ == "__main__":
    main()
