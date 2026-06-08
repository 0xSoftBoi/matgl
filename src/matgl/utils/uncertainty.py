"""Monte Carlo Dropout wrapper for uncertainty-aware inference with MatGL models.

Enables predictive uncertainty estimation from any pretrained ``MatGLModel`` via
the MC Dropout approximation (Gal & Ghahramani, 2016): the model is set to
``eval()`` to freeze BatchNorm statistics while the readout dropout layers are
kept in ``train()`` mode, yielding a stochastic distribution over predictions.

Typical use case: iterative active learning where each candidate is ranked by an
acquisition function (e.g. UCB = mean - lambda * std) to prioritise DFT evaluation of
structures with the highest expected stability.

Reference:
    Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation.
    ICML. https://arxiv.org/abs/1506.02142
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core import Molecule, Structure

    from matgl.graph._converters import GraphConverter
    from matgl.models._core import MatGLModel

# Readout attribute names searched by default:
#   CHGNet      → final_dropout (nn.Identity placeholder when p=0), final_layer
#   M3GNet      → final_layer (MLP when is_intensive=True; WeightedReadOut otherwise — unsupported)
#   TensorNet   → final_layer
_DEFAULT_READOUT_ATTRS: tuple[str, ...] = ("final_dropout", "final_layer")


def _inject_dropout(module: nn.Module, p: float) -> nn.Dropout | None:
    """Append an ``nn.Dropout`` to the first ``layers`` ModuleList/Sequential in *module*.

    Works for ``MLP`` (``MLP.layers`` is a ``ModuleList`` whose ``forward`` iterates
    linearly) so the injected dropout runs after the final ``Linear``.

    Returns the new ``Dropout`` on success, ``None`` if no suitable ``layers``
    attribute was found.
    """
    layers = getattr(module, "layers", None)
    if isinstance(layers, (nn.ModuleList, nn.Sequential)):
        drop = nn.Dropout(p)
        layers.append(drop)
        return drop
    return None


class MCDropoutWrapper:
    """Uncertainty-aware wrapper for MatGL models using Monte Carlo Dropout.

    Injects a dropout probability into the model's readout layers at construction
    time. During ``predict_uncertainty``, the backbone stays deterministic (eval)
    while the readout dropout layers remain stochastic (train), giving per-structure
    predictive mean and standard deviation via N forward passes.

    Example:
        >>> from matgl.models._chgnet import CHGNet
        >>> from matgl.utils.uncertainty import MCDropoutWrapper
        >>> model = CHGNet.load()
        >>> wrapper = MCDropoutWrapper(model, dropout_p=0.1)
        >>> structures = [...]  # list of pymatgen Structure objects
        >>> mean, std = wrapper.predict_uncertainty(structures, n_passes=20)
    """

    def __init__(
        self,
        model: MatGLModel,
        dropout_p: float = 0.1,
        readout_attrs: Sequence[str] = _DEFAULT_READOUT_ATTRS,
    ) -> None:
        """Initialise MCDropoutWrapper and inject dropout into the model's readout layers.

        Args:
            model: Any pretrained ``MatGLModel`` (CHGNet, M3GNet, TensorNet, …).
            dropout_p: Dropout probability to inject into readout layers. Default = 0.1.
            readout_attrs: Attribute names on ``model`` to search for dropout layers.
                Defaults to ``("final_dropout", "final_layer")``. Override when using
                a custom architecture with differently-named readout modules.

        Raises:
            ValueError: If no dropout layers are found within the specified readout
                attributes and none can be injected (i.e. there are no ``nn.Identity``
                placeholders to replace).
        """
        if not 0.0 < dropout_p < 1.0:
            raise ValueError(f"dropout_p must be in (0, 1), got {dropout_p}")

        self.model = model
        self.dropout_p = dropout_p
        self._stochastic_modules: list[nn.Dropout] = []

        for attr in readout_attrs:
            if self._stochastic_modules:
                # Already wired up dropout from an earlier attr; stop so we don't
                # accidentally mutate a downstream module (e.g. CHGNet's final_layer
                # uses indexed layers[-1] access, so appending there would corrupt it).
                break
            readout_module = getattr(model, attr, None)
            if readout_module is None:
                continue
            if isinstance(readout_module, nn.Identity):
                new_drop = nn.Dropout(p=dropout_p)
                setattr(model, attr, new_drop)
                self._stochastic_modules.append(new_drop)
            elif isinstance(readout_module, nn.Dropout):
                readout_module.p = dropout_p
                self._stochastic_modules.append(readout_module)
            else:
                found = [m for m in readout_module.modules() if isinstance(m, nn.Dropout)]
                if found:
                    for m in found:
                        m.p = dropout_p
                    self._stochastic_modules.extend(found)
                else:
                    injected = _inject_dropout(readout_module, dropout_p)
                    if injected is not None:
                        self._stochastic_modules.append(injected)

        if not self._stochastic_modules:
            raise ValueError(
                f"No dropout layers found or injected in {list(readout_attrs)} on "
                f"{type(model).__name__}. Initialise the model with dropout > 0 "
                "(e.g. CHGNet(final_dropout=0.1)) or supply the correct readout_attrs."
            )

    @contextmanager
    def _stochastic_mode(self):
        """Set the full model to eval then re-enable only the readout dropout layers."""
        self.model.eval()
        for m in self._stochastic_modules:
            m.train()
        try:
            yield
        finally:
            self.model.eval()

    def _to_graph(
        self,
        structure: Structure | Molecule,
        graph_converter: GraphConverter | None,
    ) -> tuple:
        """Convert a structure to a (graph, state_feats) pair.

        Replicates the graph-setup boilerplate shared by all MatGL
        ``predict_structure`` implementations so we can convert each structure
        once and reuse the graph across all N forward passes.
        """
        if graph_converter is None:
            from matgl.ext.pymatgen import Structure2Graph

            graph_converter = Structure2Graph(
                element_types=self.model.element_types,  # type: ignore[attr-defined]
                cutoff=self.model.cutoff,  # type: ignore[attr-defined]
            )

        g, lat, state_feats_default = graph_converter.get_graph(structure)
        g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
        g.pos = g.frac_coords @ lat[0]
        state_feats = torch.tensor(state_feats_default)
        return g, state_feats

    def predict_uncertainty(
        self,
        structures: Structure | Molecule | Sequence[Structure | Molecule],
        n_passes: int = 20,
        graph_converter: GraphConverter | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run N stochastic forward passes and return predictive mean and std.

        All structures are batched into a single PyG ``Batch`` per pass, so cost
        scales as O(n_passes) forward calls rather than O(N · n_passes). Graph
        conversion still happens exactly once per structure.

        Args:
            structures: A single or list of pymatgen ``Structure`` / ``Molecule``.
            n_passes: Number of stochastic forward passes. Must be ≥ 2. Increase
                for tighter uncertainty estimates at the cost of compute. Default = 20.
            graph_converter: Optional graph converter. If ``None``, uses the
                model's default ``Structure2Graph`` (requires ``model.element_types``
                and ``model.cutoff`` attributes, present on all standard MatGL models).

        Returns:
            mean: Float tensor of shape ``(N,)`` (or ``(N, ntargets)`` for
                multi-target models) with the per-structure predictive mean.
            std:  Float tensor of the same shape with the per-structure predictive
                standard deviation.

        Raises:
            ValueError: If ``n_passes < 2`` (std of a single sample is undefined).
        """
        from pymatgen.core import Molecule as _Molecule
        from pymatgen.core import Structure as _Structure
        from torch_geometric.data import Batch as _Batch

        if n_passes < 2:
            raise ValueError(f"n_passes must be ≥ 2, got {n_passes}")

        single = isinstance(structures, (_Structure, _Molecule))
        if single:
            structures = [structures]

        graphs = [self._to_graph(s, graph_converter) for s in structures]
        batch = _Batch.from_data_list([g for g, _ in graphs])
        sfs = [sf for _, sf in graphs]
        batch_sf = torch.stack(sfs) if sfs[0].numel() > 0 else None

        with self._stochastic_mode(), torch.no_grad():
            samples = [self.model(g=batch, state_attr=batch_sf) for _ in range(n_passes)]

        stacked = torch.stack(samples)  # (n_passes, N) or (n_passes, N, ntargets)
        mean, std = stacked.mean(dim=0), stacked.std(dim=0)
        if single:
            return mean.squeeze(0), std.squeeze(0)
        return mean, std
