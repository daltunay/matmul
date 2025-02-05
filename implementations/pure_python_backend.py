import typing as tp

import numpy as np
from numpy.typing import NDArray

from .base import DType, MatrixBackend


class PurePythonBackend(MatrixBackend[NDArray]):
    DTYPE_MAP = {
        "fp8": None,
        "fp16": None,
        "fp32": None,
        "fp64": float,
    }

    @staticmethod
    def generate_matrix(
        rows: int,
        cols: int,
        dtype: np.dtype,
        device: str,
        *_: tp.Any,
        **__: tp.Any,
    ) -> NDArray:
        return np.random.randn(rows, cols).astype(dtype)

    @staticmethod
    def multiply_matrices(a: NDArray, b: NDArray) -> NDArray:
        result = np.zeros((a.shape[0], b.shape[1]), dtype=a.dtype)
        for i in range(a.shape[0]):
            for j in range(b.shape[1]):
                for k in range(a.shape[1]):
                    result[i, j] += a[i, k] * b[k, j]
        return result

    @classmethod
    def convert_dtype(cls, dtype_str: DType) -> float | tp.NoReturn:
        dtype = cls.DTYPE_MAP[dtype_str]

        if not dtype:
            raise ValueError(f"Unsupported dtype: {dtype_str}")

        return dtype
