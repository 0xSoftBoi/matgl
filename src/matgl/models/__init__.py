"""Graph neural network model implementations.

``MatGLModel`` (base) and ``TransformedTargetModel`` (wrapper) are available
under both backends.
"""

from __future__ import annotations

from matgl.config import BACKEND

from ._core import MatGLModel

if BACKEND == "DGL":
    from ._chgnet import CHGNet
    from ._m3gnet_dgl import M3GNet
    from ._megnet_dgl import MEGNet
    from ._qet_dgl import QET
    from ._so3net import SO3Net
    from ._tensornet_dgl import TensorNet
else:
    from ._chgnet_pyg import CHGNet  # type: ignore[assignment]
    from ._grace import GRACE
    from ._m3gnet_pyg import M3GNet  # type: ignore[assignment]
    from ._megnet_pyg import MEGNet  # type: ignore[assignment]
    from ._qet_pyg import QET  # type: ignore[assignment]
    from ._so3net_pyg import SO3Net  # type: ignore[assignment]
    from ._tensornet_pyg import TensorNet  # type: ignore[assignment]

from ._wrappers import TransformedTargetModel

__all__ = [
    "QET",
    "CHGNet",
    "M3GNet",
    "MEGNet",
    "MatGLModel",
    "SO3Net",
    "TensorNet",
    "TransformedTargetModel",
]

if BACKEND != "DGL":
    __all__ += ["GRACE"]
