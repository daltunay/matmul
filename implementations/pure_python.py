import random

from .base import MatrixBackend

Matrix = list[list[float]]


class PurePythonBackend(MatrixBackend[Matrix]):
    @staticmethod
    def generate_matrix(rows: int, cols: int, *_, **__) -> Matrix:
        return [[random.normalvariate() for _ in range(cols)] for _ in range(rows)]

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
