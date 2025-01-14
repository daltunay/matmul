import random
import typing as tp

from .base import MatrixBackend

Matrix = list[list[float]]


class PurePythonBackend(MatrixBackend[Matrix]):
    @staticmethod
    def generate_matrix(rows: int, cols: int, dtype: tp.Any = None, *_, **__) -> Matrix:
        return [[random.normalvariate(0, 1) for _ in range(cols)] for _ in range(rows)]

    @staticmethod
    def multiply_matrices(a: Matrix, b: Matrix) -> Matrix:
        """Compute C = A @ B."""
        M = len(a)
        K = len(a[0])
        N = len(b[0])

        result = [[0.0] * N for _ in range(M)]

        for i in range(M):
            for j in range(N):
                for k in range(K):
                    result[i][j] += a[i][k] * b[k][j]

        return result

    @staticmethod
    def convert_dtype(
        dtype_str: tp.Literal["fp8", "fp16", "fp32", "fp64"]
    ) -> tp.Any | tp.NoReturn:
        dtype = {
            "fp8": None,
            "fp16": None,
            "fp32": None,
            "fp64": float,
        }[dtype_str]

        if not dtype:
            raise ValueError(f"Unsupported dtype: {dtype_str}")

        return dtype
