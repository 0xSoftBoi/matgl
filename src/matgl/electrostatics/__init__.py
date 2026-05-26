"""Deprecated alias for :mod:`matgl.layers`.

All electrostatics layers (:class:`ElectrostaticPotential` and
:class:`LinearQeq`) have been moved to :mod:`matgl.layers`. Importing from
:mod:`matgl.electrostatics` is deprecated and will be removed in a future
release.
"""

from __future__ import annotations

import warnings

from matgl.layers import ElectrostaticPotential, LinearQeq

warnings.warn(
    "matgl.electrostatics is deprecated and will be removed in a future release. "
    "Import ElectrostaticPotential and LinearQeq from matgl.layers instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ElectrostaticPotential", "LinearQeq"]
