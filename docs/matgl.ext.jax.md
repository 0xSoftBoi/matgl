---
layout: default
title: matgl.ext.jax.md
nav_exclude: true
---

# matgl.ext.jax package

JAX-accelerated inference for matgl TensorNet / QET (PyG backend).

This optional subpackage reimplements the inference path (energy + forces +
stress) of the PyG-backend `TensorNet` and `QET` models in JAX. A converted
model is JIT-compiled by XLA into a single fused program, giving a portable
(CPU / CUDA / Apple-Silicon) speedup over eager PyTorch.

It requires the optional `jax` dependency:

```default
pip install matgl[jax]
```

Public entry points:

* [`convert_potential()`]() – torch `Potential` -> JAX pytree.
* [`make_potential_fn()`]() – jitted `(E, forces, stress)` fn.
* [`JAXPESCalculator`]() – ASE calculator, a twin of
  `matgl.ext.ase.PESCalculator`.

## *class* matgl.ext.jax.JAXPESCalculator(potential, , stress_unit: str = ‘GPa’, stress_weight: float = 1.0, use_voigt: bool = False, dtype: str = ‘float32’, pad_edges: bool = True, \*\*kwargs)

Bases: `Calculator`

ASE calculator that runs a converted matgl `Potential` under JAX/XLA.

Initialize from a (converted-on-the-fly) matgl `Potential`.

* **Parameters:**
  * **potential** – a `matgl.apps.pes.Potential` wrapping TensorNet / QET.
  * **stress_unit** – `"GPa"` or `"eV/A3"` (use the latter for ASE MD/relax).
  * **stress_weight** – extra multiplier applied to the stress.
  * **use_voigt** – emit stress as a Voigt 6-vector instead of a 3x3 matrix.
  * **dtype** – `"float32"` (default) or `"float64"`.
  * **pad_edges** – pad the edge list to a bucket capacity so the jitted
    program is shape-stable across MD steps.
  * **\*\*kwargs** – forwarded to `ase.calculators.calculator.Calculator`.

### calculate(atoms=None, properties=None, system_changes=[‘positions’, ‘numbers’, ‘cell’, ‘pbc’, ‘initial_charges’, ‘initial_magmoms’])

Build the graph, run the jitted JAX potential, store ASE results.

### implemented_properties  *: list[str]*  *= (‘energy’, ‘free_energy’, ‘forces’, ‘stress’)*

Properties calculator can handle (energy, forces, …)

## matgl.ext.jax.convert_potential(potential) → tuple[dict, dict, dict]

Convert a torch `Potential` (wrapping TensorNet/QET) to JAX.

Returns `(params, cfg, extras)` where `extras` carries the
denormalisation scalars and the optional per-element reference offsets.

## matgl.ext.jax.make_potential_fn(params, cfg, extras, num_graphs: int = 1, energy_model=None)

Build a jitted `(E, forces, stress)` function for a converted model.

The returned callable takes `(pos, strain, frac_coords, lat3, pbc_offset, z, edge_index, batch, edge_mask)` and returns energy (eV), forces (eV/A) and the
3x3 stress (GPa). `cfg` / `num_graphs` / denorm constants are closed over.

Forces and stress correspond exactly to `Potential.forward`’s two autograd
leaves: `pos` carries the force gradient, while `strain` deforms *both*
the PBC offshift and the atomic positions (via `frac_coords`), so the stress
captures the full position-deformation term.

## matgl.ext.jax._basis module

JAX ports of the radial basis expansions used by TensorNet / QET.

Covers the three bases `matgl.layers.BondExpansion` can select:

* smooth spherical Bessel  (`rbf_type="SphericalBessel", use_smooth=True`)  –
  the basis used by the MatPES foundation potentials;
* plain spherical Bessel   (`use_smooth=False`)  – `j_l` for `l = 0..4`;
* Gaussian expansion.

The non-learned Bessel-root / normalisation constants are NOT part of a model’s
`state_dict`; they are rederived here exactly as `SphericalBesselFunction`
does, reusing the in-package `SPHERICAL_BESSEL_ROOTS` table.

### matgl.ext.jax._basis.bond_expansion(cfg: dict, r_safe)

Expand a 1-D distance array `r_safe` into radial-basis features.

`r_safe` must be strictly positive (callers substitute a dummy distance for
padded edges) so the `sin(x)/x` terms never hit `0/0`.

### matgl.ext.jax._basis.build_basis_config(model) → dict

Extract the radial-basis config from a torch `TensorNet` / `QET`.

## matgl.ext.jax._calculator module

ASE calculator backed by the JAX TensorNet/QET inference path.

`JAXPESCalculator` is a near drop-in twin of `matgl.ext.ase.PESCalculator`:
it reuses matgl’s existing (numpy/CPU) neighbour-list build, pads the edge list
to a bucket capacity for shape-stable XLA compilation, and evaluates a single
jitted `(E, forces, stress)` function. It plugs straight into matgl’s
`MolecularDynamics` / `Relaxer`.

### *class* matgl.ext.jax._calculator.JAXPESCalculator(potential, , stress_unit: str = ‘GPa’, stress_weight: float = 1.0, use_voigt: bool = False, dtype: str = ‘float32’, pad_edges: bool = True, \*\*kwargs)

Bases: `Calculator`

ASE calculator that runs a converted matgl `Potential` under JAX/XLA.

Initialize from a (converted-on-the-fly) matgl `Potential`.

* **Parameters:**
  * **potential** – a `matgl.apps.pes.Potential` wrapping TensorNet / QET.
  * **stress_unit** – `"GPa"` or `"eV/A3"` (use the latter for ASE MD/relax).
  * **stress_weight** – extra multiplier applied to the stress.
  * **use_voigt** – emit stress as a Voigt 6-vector instead of a 3x3 matrix.
  * **dtype** – `"float32"` (default) or `"float64"`.
  * **pad_edges** – pad the edge list to a bucket capacity so the jitted
    program is shape-stable across MD steps.
  * **\*\*kwargs** – forwarded to `ase.calculators.calculator.Calculator`.

#### calculate(atoms=None, properties=None, system_changes=[‘positions’, ‘numbers’, ‘cell’, ‘pbc’, ‘initial_charges’, ‘initial_magmoms’])

Build the graph, run the jitted JAX potential, store ASE results.

#### implemented_properties  *: list[str]*  *= (‘energy’, ‘free_energy’, ‘forces’, ‘stress’)*

Properties calculator can handle (energy, forces, …)

## matgl.ext.jax._convert module

Convert a PyTorch `TensorNet` / `QET` (or `Potential`) to a JAX pytree.

The JAX side keeps weights in a nested `dict` keyed to mirror the PyTorch
`state_dict`. The only non-trivial transforms are:

* `nn.Linear`  – torch stores `weight` as `(out, in)` and computes
  `x @ W.T`; we store the transpose `(in, out)` and compute `x @ W`.
* `nn.LayerNorm` / `nn.Embedding` – copied verbatim.
* Bessel-root buffers are not in the `state_dict`; they are rebuilt by
  [`_basis`]().

### matgl.ext.jax._convert.build_config(model) → dict

Static architecture config (Python scalars + non-learned basis arrays).

### matgl.ext.jax._convert.convert_potential(potential) → tuple[dict, dict, dict]

Convert a torch `Potential` (wrapping TensorNet/QET) to JAX.

Returns `(params, cfg, extras)` where `extras` carries the
denormalisation scalars and the optional per-element reference offsets.

### matgl.ext.jax._convert.convert_qet(model) → dict

Convert a torch `QET` to a JAX parameter pytree.

Reuses [`convert_tensornet()`]() for the shared feature stack (and the wider
`final_layer` gated readout) and adds the charge-equilibration head.

### matgl.ext.jax._convert.convert_tensornet(model) → dict

Convert a torch `TensorNet` to a JAX parameter pytree.

## matgl.ext.jax._math module

JAX ports of the tensor-math and cutoff primitives used by TensorNet / QET.

Every function here is a 1:1 numerical port of its PyTorch counterpart in
`matgl.utils.maths` / `matgl.utils.cutoff`. They are pure functions so they
compose cleanly under `jax.jit` and `jax.grad`.

### matgl.ext.jax._math.cosine_cutoff(r, cutoff: float)

Cosine cutoff envelope (matgl.utils.cutoff.cosine_cutoff).

### matgl.ext.jax._math.decompose_tensor(tensor)

(…, 3, 3) -> (scalar-part, skew-part, symmetric-traceless-part).

### matgl.ext.jax._math.layer_norm(params: dict, x, eps: float = 1e-05)

LayerNorm over the last axis. Matches `torch.nn.LayerNorm` (biased var).

### matgl.ext.jax._math.linear(params: dict, x)

Dense layer. `params` has key `w` (in, out) and optionally `b`.

### matgl.ext.jax._math.new_radial_tensor(scalars, skew, traceless, f_i, f_a, f_s)

Multiply per-edge invariant features into the irreducible tensor components.

### matgl.ext.jax._math.polynomial_cutoff(r, cutoff: float, exponent: int = 3)

Polynomial cutoff envelope (matgl.utils.cutoff.polynomial_cutoff).

### matgl.ext.jax._math.scatter_add(x, idx, dim_size: int)

Sum rows of `x` into `dim_size` segments keyed by `idx` (axis 0).

### matgl.ext.jax._math.tensor_linear(w, t)

Apply a bias-free Linear over the `units` axis of a (N, units, 3, 3) tensor.

Mirrors `linears_tensor[i](t.permute(0,2,3,1)).permute(0,3,1,2)`.

### matgl.ext.jax._math.tensor_norm(tensor)

Frobenius norm-squared over the trailing (3, 3) axes.

### matgl.ext.jax._math.vector_to_skewtensor(vector)

(…, 3) vector -> (…, 3, 3) skew-symmetric tensor.

### matgl.ext.jax._math.vector_to_symtensor(vector)

(…, 3) vector -> (…, 3, 3) symmetric traceless tensor.

## matgl.ext.jax._pad module

Edge padding / bucketing for static-shape XLA compilation.

A neighbour list has a variable edge count `E` that changes every MD step,
while the atom count `N` is fixed for a trajectory. Padding `E` up to a
bucket capacity keeps the jitted program shape-stable, so it is compiled once
and reused (at most ~log2 recompilations as `E` crosses bucket boundaries).

### matgl.ext.jax._pad.next_bucket(n_edges: int) → int

Smallest bucket capacity >= `n_edges`.

### matgl.ext.jax._pad.pad_graph(edge_index, pbc_offset, e_cap: int)

Pad `edge_index` / `pbc_offset` to capacity `e_cap`.

Padded edges are sentinel self-loops on atom 0 with PBC image `[1, 0, 0]`;
the returned boolean `edge_mask` is `1` for real edges and `0` for
padding. Callers must multiply per-edge contributions by `edge_mask` before
any scatter so padded edges contribute exactly zero.

The non-zero PBC image is deliberate: it gives padded edges a non-zero bond
vector (`= lattice[0]`), so `grad(||bond_vec||)` stays finite. A zero
bond vector would make the norm gradient `0/0` and the masked-out `0 * NaN`
would poison atom 0’s force.

## matgl.ext.jax._potential module

JAX port of `matgl.apps.pes.Potential` (inference path: E, forces, stress).

Reproduces the strain-based stress derivation of `matgl.apps._pes.Potential`:
a symbolic strain `eps` is introduced, the lattice becomes `lat @ (I + eps)`,
and `stress = (dE/d eps) / V`. Energy and gradients are produced by a single
`jax.value_and_grad` under one `jax.jit`.

### matgl.ext.jax._potential.make_potential_fn(params, cfg, extras, num_graphs: int = 1, energy_model=None)

Build a jitted `(E, forces, stress)` function for a converted model.

The returned callable takes `(pos, strain, frac_coords, lat3, pbc_offset, z, edge_index, batch, edge_mask)` and returns energy (eV), forces (eV/A) and the
3x3 stress (GPa). `cfg` / `num_graphs` / denorm constants are closed over.

Forces and stress correspond exactly to `Potential.forward`’s two autograd
leaves: `pos` carries the force gradient, while `strain` deforms *both*
the PBC offshift and the atomic positions (via `frac_coords`), so the stress
captures the full position-deformation term.

## matgl.ext.jax._qet module

Functional JAX port of the PyG QET model.

QET = TensorNet feature extractor + a charge-equilibration head:

* per-atom electronegativity `chi` / hardness / Gaussian width `sigma`;
* a closed-form charge-equilibration solve (`LinearQeq` – O(N), no linear
  system, just segment sums);
* a Gaussian-smeared Coulomb electrostatic potential;
* a LayerNorm + gated readout over `[node_feat, charge, elec_pot, magmom?]`.

The TensorNet `forward_features` is reused verbatim.

### matgl.ext.jax._qet.electrostatic_potential(charge, sigma, pos, edge_index, pbc_offshift, cutoff: float, edge_mask)

Gaussian-smeared Coulomb potential (port of `ElectrostaticPotential`).

### matgl.ext.jax._qet.linear_qeq(chi, hardness, batch, num_graphs: int, total_charge: float)

Closed-form charge equilibration (port of `LinearQeq.forward`).

`q_i = -chi_i/eta_i + (1/eta_i) (Q + sum_j chi_j/eta_j) / (sum_j 1/eta_j)`.

### matgl.ext.jax._qet.qet_energy(params, cfg, z, pos, edge_index, pbc_offshift, batch, num_graphs, edge_mask)

Per-graph QET energy (raw model output, before Potential denorm).

## matgl.ext.jax._tensornet module

Functional JAX port of the PyG TensorNet forward pass (inference path).

This reproduces `matgl.models.TensorNet.forward` (the `use_warp=False` branch)
plus the embedding / interaction layers and the extensive / intensive readouts.
Everything is a pure function of `(params, geometry)` so it composes under
`jax.jit` and `jax.grad`; `cfg` carries the architecture (Python scalars +
non-learned constant arrays) and is closed over at fn-build time.

### matgl.ext.jax._tensornet.forward_features(params, cfg, z, pos, edge_index, pbc_offshift, edge_mask)

Run TensorNet feature extraction up to the per-atom `readout` features.

### matgl.ext.jax._tensornet.gated_mlp(params: dict, x, activate_last: bool = False)

Gated MLP — `value(x) * sigmoid_gate(x)`. Internal activation is SiLU.

### matgl.ext.jax._tensornet.interaction(params, cfg, edge_index, edge_weight, edge_attr, x, edge_mask)

Port of `matgl.layers._graph_convolution.TensorNetInteraction.forward`.

### matgl.ext.jax._tensornet.mlp(layers: list, x, activation, activate_last: bool)

Plain MLP — Linear layers with `activation` between them.

### matgl.ext.jax._tensornet.pair_vector_and_distance(pos, edge_index, pbc_offshift)

Port of `compute_pair_vector_and_distance`.

### matgl.ext.jax._tensornet.tensor_embedding(params, cfg, z, edge_index, edge_attr, edge_weight, edge_vec, edge_mask)

Port of `matgl.layers._embedding.TensorEmbedding.forward`.

### matgl.ext.jax._tensornet.tensornet_energy(params, cfg, z, pos, edge_index, pbc_offshift, batch, num_graphs, edge_mask)

Per-graph TensorNet energy (raw model output, before Potential denorm).