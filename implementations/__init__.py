from .base import MatrixBackend
from .numpy_backend import NumpyBackend
from .pure_python import PurePythonBackend
from .torch_backend import TorchBackend
# from .triton_backend import TritonBackend

BACKENDS = {
    "numpy": NumpyBackend,
    "purepython": PurePythonBackend,
    "torch": TorchBackend,
    # "triton": TritonBackend,
}

__all__ = ["MatrixBackend", "BACKENDS"]
