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

# Readout attribute names common across MatGL model architectures:
#   CHGNet  → final_layer, final_dropout
#   M3GNet  → final_layer
#   TensorNet → final_layer
_DEFAULT_READOUT_ATTRS: tuple[str, ...] = ("final_dropout", "final_layer")


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

    def __init__(
        self,
        model: MatGLModel,
        dropout_p: float = 0.1,
        readout_attrs: Sequence[str] = _DEFAULT_READOUT_ATTRS,
    ) -> None:
        """Initialise MCDropoutWrapper and inject dropout into the model's readout layers."""
        if not 0.0 < dropout_p < 1.0:
            raise ValueError(f"dropout_p must be in (0, 1), got {dropout_p}")

        self.model = model
        self.dropout_p = dropout_p

        # Collect the dropout modules that will be set to train() during inference.
        # For each named readout attribute:
        #   - nn.Dropout  → update p in place
        #   - nn.Identity → replace with nn.Dropout (CHGNet default: final_dropout=0)
        #   - other nn.Module → recurse into children looking for nn.Dropout
        self._stochastic_modules: list[nn.Dropout] = []

        for attr in readout_attrs:
            readout_module = getattr(model, attr, None)
            if readout_module is None:
                continue
            if isinstance(readout_module, nn.Identity):
                new_drop = nn.Dropout(p=dropout_p)
                setattr(model, attr, new_drop)
                self._stochastic_modules.append(new_drop)
            else:
                for sub in readout_module.modules():
                    if isinstance(sub, nn.Dropout):
                        sub.p = dropout_p
                        self._stochastic_modules.append(sub)

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
        import matgl

        if graph_converter is None:
            from matgl.ext.pymatgen import Structure2Graph

            graph_converter = Structure2Graph(
                element_types=self.model.element_types,  # type: ignore[attr-defined]
                cutoff=self.model.cutoff,  # type: ignore[attr-defined]
            )

        g, lat, state_feats_default = graph_converter.get_graph(structure)
        g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
        g.pos = g.frac_coords @ lat[0]
        state_feats = torch.tensor(state_feats_default, dtype=matgl.float_th)
        return g, state_feats

    def predict_uncertainty(
        self,
        structures: Structure | Molecule | Sequence[Structure | Molecule],
        n_passes: int = 20,
        graph_converter: GraphConverter | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run N stochastic forward passes and return predictive mean and std.

        Each structure is converted to a graph exactly once; the N forward passes
        reuse the cached graph, so the cost scales as O(N · n_passes) not
        O(N · n_passes · conversion_time).

        Args:
            structures: A single or list of pymatgen ``Structure`` / ``Molecule``.
            n_passes: Number of stochastic forward passes. Increase for tighter
                uncertainty estimates at the cost of compute. Default = 20.
            graph_converter: Optional graph converter. If ``None``, uses the
                model's default ``Structure2Graph`` (requires ``model.element_types``
                and ``model.cutoff`` attributes, present on all standard MatGL models).

        Returns:
            mean: Float tensor of shape ``(N,)`` (or ``(N, ntargets)`` for
                multi-target models) with the per-structure predictive mean.
            std:  Float tensor of the same shape with the per-structure predictive
                standard deviation.
        """
        from pymatgen.core import Molecule as _Molecule
        from pymatgen.core import Structure as _Structure

        if isinstance(structures, (_Structure, _Molecule)):
            structures = [structures]

        graphs = [self._to_graph(s, graph_converter) for s in structures]

        pass_results: list[torch.Tensor] = []
        with self._stochastic_mode(), torch.no_grad():
            for _ in range(n_passes):
                preds = [self.model(g=g, state_attr=sf).squeeze() for g, sf in graphs]
                pass_results.append(torch.stack(preds))

        stacked = torch.stack(pass_results)  # (n_passes, N) or (n_passes, N, ntargets)
        return stacked.mean(dim=0), stacked.std(dim=0)
