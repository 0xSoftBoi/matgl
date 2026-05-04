"""LAMMPS-compatible TorchScript wrapper for MatGL Potentials.

Mirrors the MACE pattern (``mace.cli.create_lammps_model``): produces a
``torch.jit.ScriptModule`` that takes plain tensors and returns a dict of
energy / per-atom energy / forces / virials, ready to be loaded by a LAMMPS
``pair_style`` via ``torch::jit::load``.

Currently supports TensorNet on the PyG backend only; M3GNet/CHGNet are
DGL-only and need PyG ports first.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from pymatgen.core.periodic_table import Element
from torch import Tensor, nn
from torch.autograd import grad

from matgl.graph._compute_pyg import compute_pair_vector_and_distance
from matgl.utils.maths import decompose_tensor, tensor_norm

if TYPE_CHECKING:
    from matgl.apps._pes_pyg import Potential

logger = logging.getLogger(__name__)


_MAX_Z = 119


def _build_z_to_index(element_types: tuple[str, ...]) -> Tensor:
    """Build a length-_MAX_Z lookup buffer mapping atomic number -> internal index."""
    z_to_index = torch.full((_MAX_Z,), -1, dtype=torch.long)
    for idx, sym in enumerate(element_types):
        z = int(Element(sym).Z)
        z_to_index[z] = idx
    return z_to_index


class LAMMPSMatGLModel(nn.Module):
    """TorchScript-friendly wrapper around a MatGL ``Potential`` for LAMMPS.

    Takes plain tensors (Cartesian positions, edge_index, integer image shifts,
    cell, atomic numbers, ghost mask) and returns a dict of total local energy,
    per-atom node energies, forces, and virials. Replicates the autograd
    machinery from :class:`matgl.apps.pes.Potential` but driven by Cartesian
    inputs so LAMMPS doesn't have to materialize fractional coordinates.

    Limitations:
        * Inner model must be ``TensorNet`` (PyG backend, ``use_warp=False``,
          ``is_intensive=False``).
        * Per-atom virials and the Hessian path are not exported.
        * ``data_mean`` is added once to ``total_energy_local``; multi-rank
          LAMMPS therefore requires ``data_mean == 0`` for correctness. The
          standard MatGL PES checkpoints satisfy this — element offsets are
          carried by ``element_refs`` instead.
    """

    # NOTE: deliberately *no* class-level Tensor annotations on the registered
    # buffers — they trip TorchScript's annotation resolver (it sees them as
    # unresolved string annotations under ``from __future__ import annotations``
    # and fails with "Unknown type annotation"). Buffer types are still
    # inferred correctly from ``register_buffer`` calls. mypy access to
    # ``self.data_mean`` etc. is narrowed locally with ``cast`` where needed.

    def __init__(
        self,
        potential: Potential,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        # Imports kept local so the public ``matgl.ext`` namespace doesn't
        # hard-require these submodules at import time.
        from matgl.models._tensornet_pyg import TensorNet

        if not isinstance(potential.model, TensorNet):
            raise NotImplementedError(
                "LAMMPSMatGLModel currently supports TensorNet (PyG) only; "
                f"got {type(potential.model).__name__}. M3GNet/CHGNet are "
                "DGL-only and need PyG ports first."
            )

        if getattr(potential.model, "_use_warp", False):
            raise ValueError(
                "Inner TensorNet was constructed with use_warp=True. Re-load "
                "or rebuild it with use_warp=False before exporting for "
                "LAMMPS — Warp custom-ops are not TorchScript-compatible."
            )

        if potential.model.is_intensive:
            raise ValueError(
                "Inner TensorNet has is_intensive=True (intensive head). "
                "LAMMPS expects an extensive PES; only is_intensive=False "
                "is supported."
            )

        if potential.calc_repuls:
            raise NotImplementedError("ZBL repulsion (calc_repuls=True) export is not yet supported.")

        if potential.calc_magmom or potential.calc_charge:
            raise NotImplementedError("Magmom / charge heads are not exported for LAMMPS.")

        # Reference inner submodules directly — the wrapper calls them with
        # plain tensors so we never construct a PyG Data inside scripted code.
        model = potential.model
        self.bond_expansion = model.bond_expansion
        self.tensor_embedding = model.tensor_embedding
        self.layers = model.layers
        self.out_norm = model.out_norm
        self.linear = model.linear
        self.final_layer = model.final_layer

        # Bake in the cutoff and group as Python attrs (immutable in script).
        self.cutoff: float = float(model.cutoff)
        self.r_max: float = float(model.cutoff)
        self.n_species: int = len(model.element_types)

        # Buffers carried over from Potential. ``Potential`` stores these
        # via ``register_buffer``, so mypy sees them as ``Tensor | Module``;
        # the ``cast`` keeps the typing tight without any runtime cost.
        from typing import cast

        data_mean = cast("Tensor", potential.data_mean).detach().to(dtype)
        data_std = cast("Tensor", potential.data_std).detach().to(dtype)
        self.register_buffer("data_mean", data_mean)
        self.register_buffer("data_std", data_std)

        # Per-element reference energies (1D, indexed by internal element idx).
        if potential.element_refs is not None:
            ref = potential.element_refs.property_offset.detach().to(dtype)
            if ref.dim() != 1:
                raise NotImplementedError("State-conditional element_refs (>1D) not supported.")
            if ref.numel() != self.n_species:
                # AtomRefPyG can be sized to max_z=89; we only need the first
                # n_species entries when indexed by internal element idx, so
                # pad/truncate.
                if ref.numel() < self.n_species:
                    ref = torch.cat([ref, torch.zeros(self.n_species - ref.numel(), dtype=dtype)])
                else:
                    ref = ref[: self.n_species]
        else:
            ref = torch.zeros(self.n_species, dtype=dtype)
        self.register_buffer("element_refs", ref)

        # Z -> internal element index lookup. C++ passes Z; we translate.
        self.register_buffer("z_to_index", _build_z_to_index(model.element_types))

        # Atomic numbers in element_types order — useful for inspection.
        atomic_numbers = torch.tensor([int(Element(s).Z) for s in model.element_types], dtype=torch.long)
        self.register_buffer("atomic_numbers", atomic_numbers)

        if abs(float(data_mean.item() if data_mean.ndim == 0 else 0.0)) > 1e-8:
            logger.warning(
                "Exported model has non-zero data_mean=%s. Multi-rank LAMMPS "
                "runs will double-count this offset; use single-rank Kokkos.",
                float(data_mean.item()) if data_mean.ndim == 0 else data_mean,
            )

        # Cast the wrapper to the target dtype. The inner model is already the
        # checkpoint dtype; the conversion above ensures buffers match.
        self.to(dtype)

    def forward(
        self,
        positions: Tensor,
        edge_index: Tensor,
        unit_shifts: Tensor,
        cell: Tensor,
        atomic_numbers: Tensor,
        local_or_ghost: Tensor,
        compute_virials: bool,
    ) -> dict[str, Tensor]:
        """Energy / forces / virials for a single LAMMPS configuration.

        Args:
            positions: Cartesian coordinates, shape (N, 3). N includes
                ghost atoms when running in domain-decomposed mode.
            edge_index: COO edges, shape (2, E), int64. Indices into
                ``positions``.
            unit_shifts: Integer image vectors per edge, shape (E, 3),
                int64. Equivalent to ``g.pbc_offset`` in MatGL: the integer
                triple ``(nx, ny, nz)`` such that the destination atom's
                effective position is
                ``positions[dst] + unit_shifts @ cell``.
            cell: Lattice basis as row vectors, shape (3, 3).
            atomic_numbers: Per-atom Z, shape (N,), int64.
            local_or_ghost: True for owned atoms, False for ghosts; shape
                (N,), bool. Only owned atoms contribute to the energy sum.
            compute_virials: Whether to compute the virial tensor. When
                False the returned ``virials`` is a (3, 3) zero tensor.

        Returns:
            dict with keys ``total_energy_local`` (scalar), ``node_energy``
            (N,), ``forces`` (N, 3), and ``virials`` (3, 3). The virial is
            the raw stress * volume tensor (no division by volume), matching
            the MACE-LAMMPS convention so the C++ side can do its own Voigt
            mapping.
        """
        # Map Z -> internal element index. -1 means "unknown species"; the
        # subsequent embedding will then read garbage. We don't bounds-check
        # in script mode because that requires a runtime branch — the C++
        # side validates Z against the model's species list at pair_coeff time.
        atom_types = self.z_to_index[atomic_numbers]

        # Strain tensor for stress/virial via the same trick as
        # matgl.apps.pes.Potential.
        strain = torch.zeros((3, 3), dtype=positions.dtype, device=positions.device)
        if compute_virials:
            strain.requires_grad_(True)

        eye = torch.eye(3, dtype=positions.dtype, device=positions.device)
        deformation = eye + strain
        pos_s = positions @ deformation
        cell_s = cell @ deformation

        # We always need positions to require grad — forces are -dE/dpos.
        pos_s.requires_grad_(True)

        # PBC offset shifts in Cartesian space.
        pbc_offshift = unit_shifts.to(positions.dtype) @ cell_s

        # Bond geometry.
        bond_vec, bond_dist = compute_pair_vector_and_distance(pos_s, edge_index, pbc_offshift)

        # RBF expansion.
        edge_attr = self.bond_expansion(bond_dist)

        # Tensor embedding — TensorEmbedding (PyG) signature:
        #   forward(z, edge_index, edge_attr, edge_weight, edge_vec, state_attr=None)
        x_tensor, _ = self.tensor_embedding(atom_types, edge_index, edge_attr, bond_dist, bond_vec, None)

        # Interaction stack.
        for layer in self.layers:
            x_tensor = layer(edge_index, bond_dist, edge_attr, x_tensor)

        # Decompose the per-node tensor representation, take scalar norms.
        scalars, skew, traceless = decompose_tensor(x_tensor)
        x = torch.cat(
            (tensor_norm(scalars), tensor_norm(skew), tensor_norm(traceless)),
            dim=-1,
        )
        x = self.out_norm(x)
        x = self.linear(x)

        # Per-atom raw energies.
        atomic_energies_raw = self.final_layer(x).view(-1)

        # Apply std scaling, add per-element reference, mask ghosts.
        node_energy = self.data_std * atomic_energies_raw + self.element_refs[atom_types]

        masked = node_energy.masked_fill(~local_or_ghost, 0.0)
        total_energy_local = masked.sum() + self.data_mean

        # Forces = -dE/dpositions. Strain grad gives the virial tensor.
        # TorchScript demands the explicit Optional[Tensor] typing on grad_outputs.
        grad_outputs: list[torch.Tensor | None] = [torch.ones_like(total_energy_local)]

        if compute_virials:
            grads = grad(
                outputs=[total_energy_local],
                inputs=[pos_s, strain],
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=False,
            )
            pos_grad = grads[0]
            strain_grad = grads[1]
            forces = -pos_grad if pos_grad is not None else torch.zeros_like(positions)
            virials = (
                strain_grad
                if strain_grad is not None
                else torch.zeros((3, 3), dtype=positions.dtype, device=positions.device)
            )
        else:
            grads = grad(
                outputs=[total_energy_local],
                inputs=[pos_s],
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=False,
            )
            pos_grad = grads[0]
            forces = -pos_grad if pos_grad is not None else torch.zeros_like(positions)
            virials = torch.zeros((3, 3), dtype=positions.dtype, device=positions.device)

        return {
            "total_energy_local": total_energy_local,
            "node_energy": node_energy,
            "forces": forces,
            "virials": virials,
        }


def export_lammps_model(
    potential: Potential,
    output_path: str,
    dtype: torch.dtype = torch.float32,
    script: bool = True,
) -> LAMMPSMatGLModel:
    """Wrap a :class:`Potential` and save a LAMMPS-loadable artifact.

    Args:
        potential: A trained MatGL ``Potential`` (TensorNet PyG, extensive,
            no-Warp).
        output_path: Where to write the ``.pt`` file. The C++ pair_style
            loads this with ``torch::jit::load``.
        dtype: Wrapper buffer dtype. Forces ``float32`` or ``float64``.
        script: When True (default), runs ``torch.jit.script`` and saves the
            script module. When False, saves the eager wrapper as a regular
            PyTorch checkpoint (useful for debugging — not loadable from C++).

    Returns:
        The wrapper instance (eager).
    """
    wrapper = LAMMPSMatGLModel(potential=potential, dtype=dtype)
    wrapper.eval()

    if script:
        scripted = torch.jit.script(wrapper)
        scripted.save(output_path)
    else:
        torch.save(wrapper, output_path)

    return wrapper
