from __future__ import annotations

import abc
import warnings
from abc import ABCMeta

import torch
from torch import nn

from matgl.utils.io import IOMixIn


def _warn_feature_dict_kwarg(name: str) -> None:
    """Emit a DeprecationWarning for legacy feature-dict flags.

    ``return_all_layer_output`` / ``return_features`` will be removed in matgl v5;
    access ``model.feature_dict`` after calling ``forward`` / ``predict_structure``
    instead.
    """
    warnings.warn(
        f"`{name}` is deprecated and will be removed in matgl v5. "
        "Intermediate layer outputs are now always stored on `model.feature_dict` "
        "after each forward pass — use that instead.",
        DeprecationWarning,
        stacklevel=3,
    )


class MatGLModel(nn.Module, IOMixIn, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()
        # Populated by ``forward`` with the intermediate layer features
        # (e.g. ``embedding``, ``gc_<i>``, ``readout``, ``final``).
        # Always refreshed on every ``forward`` call. Use this in place of the
        # deprecated ``return_all_layer_output`` / ``return_features`` flags.
        self.feature_dict: dict = {}

    @abc.abstractmethod
    def predict_structure(self, structure, *args, **kwargs) -> torch.Tensor:
        """Convenience method to directly predict property from structure.

        Args:
            structure: An input crystal/molecule.
            *args: Any additional positional arguments.
            **kwargs: Any additional keyword arguments.

        Returns:
            output (torch.tensor): output property
        """
