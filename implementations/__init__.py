from enum import Enum

from .base import MatrixBackend
from .numpy_backend import NumpyBackend
from .pure_python_backend import PurePythonBackend
from .torch_backend import TorchBackend
from .triton_backend import TritonBackend


class BACKENDS(Enum):
    # NUMPY = NumpyBackend
    # PURE_PYTHON = PurePythonBackend
    TORCH = TorchBackend
    # TRITON = TritonBackend


__all__ = ["MatrixBackend", "BACKENDS"]
