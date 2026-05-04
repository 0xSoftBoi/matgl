# MatGL → LAMMPS pair_style

`pair_matgl` is a LAMMPS pair style that loads a TorchScript-compiled
**MatGL TensorNet** PES (PyG backend, no-Warp, extensive head) and uses
LibTorch to evaluate energies, forces, and the virial tensor on every
timestep.

This directory ships:

- `src/ML-MATGL/pair_matgl.{cpp,h}` — the CPU/serial pair style.
- `cmake/ML-MATGL.cmake` — drop-in CMake snippet.
- `tests/in.matgl_si` — sample input deck for a single-point parity check.

The Python side (one repo up) ships `mgl create-lammps-model`, which
produces the `.pt` artifact this pair style consumes.

> **Status — Phase 2 of the MatGL LAMMPS-Kokkos plugin.** Single-rank /
> multi-rank CPU only. The Kokkos / GPU variant lands in Phase 3.

## Building

### 1. Export a LAMMPS-loadable model

```bash
# From your matgl checkout:
uv run mgl create-lammps-model \
    -m materialyze/TensorNet-MatPES-r2SCAN \
    -o tensornet_matpes_r2scan.pt \
    --dtype float32
```

The CLI prints `r_max`, `n_species`, the dtype, and the species list — all
of which you'll need for `pair_coeff`.

### 2. Build LAMMPS with the package

Drop the package into a stock LAMMPS source tree and configure:

```bash
# 1) Copy or symlink the source files.
ln -s /path/to/matgl/lammps/src/ML-MATGL <lammps>/src/ML-MATGL

# 2) Tell LAMMPS' CMake about the package.
echo 'include(/path/to/matgl/lammps/cmake/ML-MATGL.cmake)' \
    >> <lammps>/cmake/CMakeLists.txt

# 3) Configure + build. Match libtorch's CXX11 ABI to LAMMPS'.
cmake -B build -S <lammps>/cmake \
    -D PKG_ML-MATGL=ON \
    -D CMAKE_PREFIX_PATH=/path/to/libtorch \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_MPI=ON
cmake --build build -j 8
```

Tested with:

- LibTorch 2.2.x – 2.5.x (CXX11 ABI, CPU build).
- LAMMPS develop branch (Aug 2024 or newer for the `add_request` /
  `REQ_GHOST` neighbor-list API).
- C++17, MPI optional.

## LAMMPS input syntax

```lammps
units           metal
atom_style      atomic
atom_modify     map yes        # required: pair_matgl needs the atom map
newton          on             # required: ghost contributions

pair_style      matgl
pair_coeff      * * tensornet_matpes_r2scan.pt Si C O
```

`pair_coeff` arguments after the `.pt` path are **species symbols** in
LAMMPS atom-type order: type 1 = first symbol, type 2 = second, …

The cutoff (`r_max`) is read from the model — you don't pass it.

### Optional pair_style flags

```lammps
pair_style matgl no_domain_decomposition
```

Reserved for future single-rank optimisations (mirrors the MACE flag).
Currently a no-op.

## Limitations

- **No per-atom energies / virials.** `eflag_atom`, `vflag_atom`, and
  `compute … pe/atom` will error. The model returns a single
  `total_energy_local` scalar plus a 3×3 virial tensor; per-atom
  decompositions would require a different export.
- **`atom_style atomic` only** for now. Charged systems aren't supported
  (the model has no charge head).
- **TorchScript artifacts are dtype-specific.** Re-run
  `mgl create-lammps-model --dtype float64` to get a double-precision
  model; mixing dtypes between LAMMPS and the model will error at load
  time.
- **Multi-rank**: works for CPU MPI, but each rank loads the model
  independently (memory adds up). The `data_mean` buffer baked into the
  TorchScript is added once per rank — keep `data_mean = 0` (the default
  for trained MatGL PES models). Non-zero `data_mean` will over-count
  proportionally to the number of ranks.
- **No restart support.** The model lives on disk; `restart` files don't
  capture the path. Re-issue `pair_style` / `pair_coeff` after a restart.
- **TensorNet only.** M3GNet, CHGNet, MEGNet, SO3Net, QET are DGL-only
  in the matgl repo and would need PyG ports first.

## Verifying a build

```bash
cd lammps/tests
<lammps>/build/lmp -in in.matgl_si
```

The test deck prints energy, forces, and stress on a small Si supercell.
Compare against the Python reference:

```bash
uv run python tests/python_reference.py    # in this directory
```

Energies should match within `1e-5 eV`, forces within `1e-4 eV/Å`, and
stresses (when nonzero) within `1e-3 GPa`.

## Implementation notes

- The pair style requests a **full neighbor list with ghost atoms**
  (`REQ_FULL | REQ_GHOST`). The model expects edge indices that span both
  owned and ghost atoms.
- Bond vectors are computed from LAMMPS' already-imaged ghost positions,
  so `unit_shifts` is always zero — the strain-based stress trick still
  works because the strain is applied to all atomic positions (owned and
  ghost) on the model side.
- Forces are accumulated for **all** atoms (owned + ghost). LAMMPS' usual
  `comm->reverse_comm` step then sums ghost contributions back to the
  rank that owns each atom. This requires `newton on`.
- Virials are written into the global `virial[6]` array directly. We set
  `no_virial_fdotr_compute = 1` in the constructor so LAMMPS doesn't
  recompute the virial from forces.

## Reference

Plan and design notes:
[`develop-a-kokkos-plugin-eventual-hare.md`](https://github.com/materialyzeai/matgl/tree/lammps).

The Python wrapper is documented inline at
`src/matgl/ext/_lammps.py` in the matgl repo.
