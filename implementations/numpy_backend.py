import numpy as np
from numpy.typing import NDArray
from .base import MatrixBackend


class NumpyBackend(MatrixBackend[NDArray]):
    @staticmethod
    def generate_matrix(rows: int, cols: int) -> NDArray:
        return np.random.randn(rows, cols)

    @staticmethod
    def multiply_matrices(a: NDArray, b: NDArray) -> NDArray:
        return np.matmul(a, b)
